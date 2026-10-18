// SPDX-License-Identifier: MIT OR Apache-2.0

use super::test_fixtures::*;
use super::*;
use crate::error::HybridError;
use crate::types::ExtractTokenOptions;
use std::fs::remove_file;
use std::mem::size_of;
use std::path::PathBuf;

fn stub() -> MoeRouter {
    MoeRouter::load_with_mode("", 8, 1, RoutingMode::StubUniform).expect("stub load should succeed")
}

#[test]
fn test_stub_mode_loads() {
    let model = stub();
    assert!(!model.is_loaded());
    assert_eq!(model.quantization(), "stub");
}

#[test]
fn test_stub_forward_uniform_weights() {
    let mut model = stub();
    let out = model.forward(&vec![0.1; EMBEDDING_DIM]).unwrap();
    for weight in &out.expert_weights {
        assert!((*weight - 0.125).abs() < 1e-5);
    }
}

#[test]
fn test_dense_sim_uses_real_gate_weights() {
    let mut gate = vec![0.0f32; EMBEDDING_DIM * 64];
    for (expert, value) in gate.iter_mut().take(64).enumerate() {
        *value = if expert == 0 { 8.0 } else { -8.0 };
    }
    let gate_bytes: Vec<u8> = gate.iter().flat_map(|value| value.to_le_bytes()).collect();
    let path = write_temp_file(&build_real_size_checkpoint(gate_bytes), "dense-real");

    let mut model =
        MoeRouter::load_with_mode(path.to_str().unwrap(), 8, 2, RoutingMode::DenseSim).unwrap();
    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    embedding[0] = 1.0;
    let out = model.forward(&embedding).unwrap();
    assert_eq!(out.selected_experts[0], 0);
    assert_eq!(model.routing_tensor_name(), "blk.0.ffn_gate_inp.weight");

    let _ = remove_file(path);
}

#[test]
fn load_routes_through_checkpoint_gate_weights_not_stub() {
    // A gate that favours expert 3 for a unit vector on dimension 0. `load`
    // must select the checkpoint-backed default mode, so the result differs
    // from the uniform stub output (`selected=[0]`, zero hidden state).
    // Expert rows are contiguous (`[expert][hidden]`); column 0 of each row
    // scores embedding dimension 0.
    let mut gate = vec![0.0f32; EMBEDDING_DIM * 64];
    for expert in 0..64 {
        gate[expert * EMBEDDING_DIM] = if expert == 3 { 8.0 } else { -8.0 };
    }
    let gate_bytes: Vec<u8> = gate.iter().flat_map(|value| value.to_le_bytes()).collect();
    let path = write_temp_file(&build_real_size_checkpoint(gate_bytes), "load-default-mode");

    let mut model = MoeRouter::load(path.to_str().unwrap(), 8, 2).unwrap();
    assert!(model.is_loaded());
    assert_eq!(model.routing_mode(), RoutingMode::default());
    assert_eq!(model.routing_mode(), RoutingMode::DenseSim);

    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    embedding[0] = 1.0;
    let out = model.forward(&embedding).unwrap();
    // Same `num_experts` / `top_k` as the loaded router so the comparison is
    // about routing content, not output length.
    let stub_out = MoeRouter::load_with_mode("", 8, 2, RoutingMode::StubUniform)
        .expect("stub load should succeed")
        .forward(&embedding)
        .unwrap();
    assert_eq!(stub_out.selected_experts, vec![0, 1]);

    assert_eq!(out.selected_experts[0], 3);
    assert_ne!(out.selected_experts, stub_out.selected_experts);
    assert_ne!(out.expert_weights, stub_out.expert_weights);
    assert!(
        out.expert_weights[3] > 0.99,
        "gate-driven weights should concentrate on expert 3, got {:?}",
        out.expert_weights
    );
    assert!(
        out.hidden.iter().any(|value| *value != 0.0),
        "loaded routing should produce a non-zero routed hidden state"
    );

    let _ = remove_file(path);
}

#[test]
fn load_with_empty_path_still_defaults_to_dense_sim_stub_router() {
    let model = MoeRouter::load("", 8, 2).unwrap();
    assert!(!model.is_loaded());
    assert_eq!(model.routing_mode(), RoutingMode::DenseSim);
}

#[test]
fn dense_sim_ranks_raw_gate_scores_before_softmax() {
    let mut model =
        MoeRouter::load_with_mode("", 8, 2, RoutingMode::DenseSim).expect("stub dense load");
    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    let chunk = (EMBEDDING_DIM / 8).max(1);
    embedding[3 * chunk] = f32::INFINITY;
    let out = model.forward(&embedding).unwrap();
    assert_eq!(out.selected_experts, vec![3, 0]);
}

#[test]
fn dense_sim_rejects_nan_gate_score() {
    let mut model =
        MoeRouter::load_with_mode("", 8, 2, RoutingMode::DenseSim).expect("stub dense load");
    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    let chunk = (EMBEDDING_DIM / 8).max(1);
    embedding[3 * chunk] = f32::NAN;
    assert!(matches!(
        model.forward(&embedding).unwrap_err(),
        HybridError::NanRoutingScore { expert_id: 3 }
    ));
}

#[test]
fn test_real_checkpoint_probe_via_env() {
    let Some(path) = std::env::var("GGUF_CHECKPOINT_PATH").ok() else {
        return;
    };

    let metadata = MoeRouter::probe_model(&path).unwrap();
    assert!(!metadata.architecture.is_empty());
    assert!(metadata.hidden_size > 0);
    assert!(metadata.num_experts > 0);
    assert!(!metadata.routing_tensor_name.is_empty());
}

#[test]
fn test_q5k_dequant_follows_engram_group_order() {
    let mut block = [0u8; 176];
    block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    block[4] = 1;
    for byte in block[48..80].iter_mut() {
        *byte = 0x05;
    }
    let out = engram_parser::dequantize_q5_k(&block, &[256]).expect("Q5_K row should dequantize");
    assert_eq!(out.len(), 256);
    assert!(out[..32].iter().all(|value| (value - 5.0).abs() <= 1e-4));
    assert!(out[32..64].iter().all(|value| value.abs() <= 1e-4));
}

fn adapter_probe(bytes: &[u8], label: &str) -> (PathBuf, super::checkpoint::MappedCheckpoint) {
    let path = write_temp_file(bytes, label);
    let mapped = super::checkpoint::open_mapped_checkpoint(path.to_str().unwrap())
        .expect("synthetic GGUF should map");
    (path, mapped)
}

fn adapter_fixture(
    gate_dims: Vec<usize>,
    gate_payload: Vec<u8>,
    token_ggml: u32,
    token_dims: Vec<usize>,
    token_payload: Vec<u8>,
) -> Vec<u8> {
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                gate_dims,
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![1, 1],
                GGML_TYPE_F32,
                vec![0u8; 4],
            ),
            ("token_embd.weight", token_dims, token_ggml, token_payload),
        ],
        32,
        EMBEDDING_DIM,
    )
}

#[test]
fn test_adapter_resolve_routing_insufficient_experts() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * size_of::<f32>()];
    let checkpoint = adapter_fixture(
        vec![EMBEDDING_DIM, 1],
        gate_payload,
        GGML_TYPE_F16,
        vec![EMBEDDING_DIM, 32],
        vec![0u8; EMBEDDING_DIM * 32 * 2],
    );
    let (path, mapped) = adapter_probe(&checkpoint, "routing-insufficient-experts");
    let result = super::adapter::resolve_adapter(&mapped, path.to_str().unwrap());
    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("only exposes 1 experts")
    ));
    let _ = remove_file(path);
}

#[test]
fn test_adapter_resolve_routing_invalid_orientation() {
    // Enough experts on min(d0, d1), but neither dim equals hidden_size.
    let gate_payload = vec![0u8; 64 * 128 * size_of::<f32>()];
    let checkpoint = adapter_fixture(
        vec![64, 128],
        gate_payload,
        GGML_TYPE_F16,
        vec![EMBEDDING_DIM, 32],
        vec![0u8; EMBEDDING_DIM * 32 * 2],
    );
    let (path, mapped) = adapter_probe(&checkpoint, "routing-invalid-orientation");
    let result = super::adapter::resolve_adapter(&mapped, path.to_str().unwrap());
    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("unsupported orientation")
    ));
    let _ = remove_file(path);
}

#[test]
fn extract_token_embedding_returns_native_hidden_size() {
    const HIDDEN: usize = 32;
    const VOCAB: usize = 4;
    let mut payload = Vec::with_capacity(HIDDEN * VOCAB * size_of::<f32>());
    for token in 0..VOCAB {
        for dim in 0..HIDDEN {
            payload.extend_from_slice(&(dim as f32).to_le_bytes());
        }
        let _ = token;
    }
    let gate_payload = vec![0u8; HIDDEN * 64 * size_of::<f32>()];
    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![HIDDEN, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![HIDDEN, HIDDEN],
                GGML_TYPE_F16,
                vec![0u8; HIDDEN * HIDDEN * 2],
            ),
            (
                "token_embd.weight",
                vec![HIDDEN, VOCAB],
                GGML_TYPE_F32,
                payload,
            ),
        ],
        32,
        HIDDEN,
    );
    let path = write_temp_file(&checkpoint, "native-hidden-32");
    let mut router =
        MoeRouter::load_with_mode(path.to_str().unwrap(), 8, 2, RoutingMode::StubUniform).unwrap();
    assert_eq!(router.hidden_size(), HIDDEN);

    let embedding = router.extract_token_embedding(0).unwrap();
    assert_eq!(embedding.len(), HIDDEN);
    for (dim, &value) in embedding.iter().enumerate() {
        assert!((value - dim as f32).abs() <= 1e-6, "dim {dim}: got {value}");
    }

    let l2_sum: f32 = embedding.iter().map(|v| v * v).sum();
    assert!(l2_sum > 1.0, "native extract must not L2-normalize");

    let projected = router
        .extract_token_embedding_with_options(0, ExtractTokenOptions::for_projector_forward())
        .unwrap();
    assert_eq!(projected.len(), EMBEDDING_DIM);
    let norm: f32 = projected.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() <= 1e-5);

    let _ = remove_file(path);
}

#[test]
fn extract_f16_neg_zero_keeps_sign_bit_at_native_length() {
    const HIDDEN: usize = 32;
    const F16_NEG_ZERO: u16 = 0x8000;
    const F16_ONE: u16 = 0x3C00;
    const F16_SUBNORMAL: u16 = 0x0001;
    let mut payload = vec![0u8; HIDDEN * 2];
    payload[0..2].copy_from_slice(&F16_NEG_ZERO.to_le_bytes());
    payload[2..4].copy_from_slice(&F16_ONE.to_le_bytes());
    payload[4..6].copy_from_slice(&F16_SUBNORMAL.to_le_bytes());

    let gate_payload = vec![0u8; HIDDEN * 64 * size_of::<f32>()];
    let checkpoint = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![HIDDEN, 64],
                GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![HIDDEN, HIDDEN],
                GGML_TYPE_F16,
                vec![0u8; HIDDEN * HIDDEN * 2],
            ),
            ("token_embd.weight", vec![HIDDEN, 1], GGML_TYPE_F16, payload),
        ],
        32,
        HIDDEN,
    );
    let path = write_temp_file(&checkpoint, "f16-neg-zero");
    let mut router =
        MoeRouter::load_with_mode(path.to_str().unwrap(), 8, 1, RoutingMode::StubUniform).unwrap();
    let embedding = router.extract_token_embedding(0).unwrap();
    assert_eq!(embedding.len(), HIDDEN);
    assert_eq!(embedding[0].to_bits(), (-0.0f32).to_bits());
    assert!((embedding[1] - 1.0).abs() <= f32::EPSILON);
    assert_eq!(embedding[2], engram_parser::f16_bits_to_f32(F16_SUBNORMAL));

    let _ = remove_file(path);
}

#[test]
fn test_adapter_resolve_rejects_iq3_s_token_embedding() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let checkpoint = adapter_fixture(
        vec![EMBEDDING_DIM, 64],
        gate_payload,
        GGML_TYPE_IQ3_S,
        vec![256, 1],
        vec![0u8; 110],
    );
    let (path, mapped) = adapter_probe(&checkpoint, "iq3s-token-embd");
    let result = super::adapter::resolve_adapter(&mapped, path.to_str().unwrap());
    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("token embedding tensor")
    ));
    let _ = remove_file(path);
}

#[test]
fn gguf_and_safetensors_bind_the_same_normalized_weights() {
    const HIDDEN: usize = 32;
    const EXPERTS: usize = 64;
    const VOCAB: usize = 3;

    let mut token = Vec::new();
    for token_id in 0..VOCAB {
        for dim in 0..HIDDEN {
            token.extend_from_slice(&(token_id as f32 * 100.0 + dim as f32).to_le_bytes());
        }
    }
    let mut gate = vec![0f32; EXPERTS * HIDDEN];
    gate[3 * HIDDEN] = 4.0;
    let gate_bytes: Vec<u8> = gate.iter().copied().flat_map(f32::to_le_bytes).collect();

    let gguf = build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![HIDDEN, EXPERTS],
                GGML_TYPE_F32,
                gate_bytes.clone(),
            ),
            (
                "token_embd.weight",
                vec![HIDDEN, VOCAB],
                GGML_TYPE_F32,
                token.clone(),
            ),
        ],
        32,
        HIDDEN,
    );
    let safetensors = safetensors_bytes(
        &[
            ("general.architecture", "reference_moe"),
            ("general.file_type", "1"),
            ("reference_moe.embedding_length", "32"),
            ("reference_moe.block_count", "16"),
            ("reference_moe.expert_count", "64"),
            ("reference_moe.expert_used_count", "8"),
        ],
        &[
            ("blk.0.ffn_gate_inp.weight", &[EXPERTS, HIDDEN], &gate_bytes),
            ("token_embd.weight", &[VOCAB, HIDDEN], &token),
        ],
    );

    let gguf_path = write_temp_file(&gguf, "equiv-gguf");
    let st_path = write_temp_file_with_extension(&safetensors, "equiv-st", "safetensors");
    let mut gguf_router =
        MoeRouter::load_with_mode(gguf_path.to_str().unwrap(), 0, 1, RoutingMode::DenseSim)
            .unwrap();
    let mut st_router =
        MoeRouter::load_with_mode(st_path.to_str().unwrap(), 0, 1, RoutingMode::DenseSim).unwrap();

    assert_eq!(gguf_router.architecture(), st_router.architecture());
    assert_eq!(gguf_router.hidden_size(), st_router.hidden_size());
    assert_eq!(gguf_router.checkpoint_num_experts(), EXPERTS);
    assert_eq!(
        gguf_router.checkpoint_num_experts(),
        st_router.checkpoint_num_experts()
    );
    assert_eq!(gguf_router.quantization(), st_router.quantization());
    assert_eq!(
        gguf_router.routing_tensor_name(),
        st_router.routing_tensor_name()
    );
    assert_eq!(
        gguf_router.extract_token_embedding(2).unwrap(),
        st_router.extract_token_embedding(2).unwrap()
    );

    let mut embedding = vec![0.0f32; HIDDEN];
    embedding[0] = 1.0;
    let gguf_scores = gguf_router.gate_scores(&embedding).unwrap();
    let st_scores = st_router.gate_scores(&embedding).unwrap();
    assert_eq!(gguf_scores, st_scores);
    assert!(gguf_scores[3] > gguf_scores[0]);

    let gguf_ckpt = super::checkpoint::open_mapped_checkpoint(gguf_path.to_str().unwrap()).unwrap();
    let st_ckpt = super::checkpoint::open_mapped_checkpoint(st_path.to_str().unwrap()).unwrap();
    let gguf_gate = gguf_ckpt
        .tensor_view("blk.0.ffn_gate_inp.weight", gguf_path.to_str().unwrap())
        .unwrap();
    let st_gate = st_ckpt
        .tensor_view("blk.0.ffn_gate_inp.weight", st_path.to_str().unwrap())
        .unwrap();
    assert_eq!(gguf_gate.dims, vec![EXPERTS, HIDDEN]);
    assert_eq!(gguf_gate.dims, st_gate.dims);
    assert_eq!(
        gguf_ckpt.tensor_names().collect::<Vec<_>>(),
        st_ckpt.tensor_names().collect::<Vec<_>>()
    );

    let _ = remove_file(gguf_path);
    let _ = remove_file(st_path);
}

#[test]
fn unsupported_token_embd_falls_through_to_the_next_candidate() {
    let hidden = 32usize;
    let experts = 4usize;
    let vocab = 2usize;
    let bytes = build_test_gguf_with_expert_count(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![hidden, experts],
                GGML_TYPE_F32,
                vec![0u8; hidden * experts * size_of::<f32>()],
            ),
            (
                "token_embd.weight",
                vec![1, 1],
                GGML_TYPE_F32,
                vec![0u8; size_of::<f32>()],
            ),
            (
                "tok_embeddings.weight",
                vec![hidden, vocab],
                GGML_TYPE_F32,
                vec![0u8; hidden * vocab * size_of::<f32>()],
            ),
        ],
        32,
        hidden,
        experts as u32,
    );
    let (path, mapped) = adapter_probe(&bytes, "token-fallback");
    let adapter = super::adapter::resolve_adapter(&mapped, path.to_str().unwrap()).unwrap();
    assert_eq!(adapter.token_embedding_tensor, "tok_embeddings.weight");
    let _ = remove_file(path);
}

#[test]
fn hugging_face_embed_tokens_name_resolves() {
    let hidden = 4usize;
    let experts = 4usize;
    let vocab = 2usize;
    let bytes = safetensors_bytes(
        &[
            ("general.architecture", "reference_moe"),
            ("general.file_type", "1"),
            ("reference_moe.embedding_length", "4"),
            ("reference_moe.block_count", "1"),
            ("reference_moe.expert_count", "4"),
            ("reference_moe.expert_used_count", "1"),
        ],
        &[
            (
                "model.layers.0.ffn_gate_inp.weight",
                &[experts, hidden],
                &vec![0u8; experts * hidden * size_of::<f32>()],
            ),
            (
                "model.embed_tokens.weight",
                &[vocab, hidden],
                &vec![0u8; vocab * hidden * size_of::<f32>()],
            ),
        ],
    );
    let path = write_temp_file_with_extension(&bytes, "hf-embed", "safetensors");
    let mapped = super::checkpoint::open_mapped_checkpoint(path.to_str().unwrap()).unwrap();
    let adapter = super::adapter::resolve_adapter(&mapped, path.to_str().unwrap()).unwrap();
    assert_eq!(adapter.token_embedding_tensor, "model.embed_tokens.weight");
    let _ = remove_file(path);
}

#[test]
fn stored_expert_count_of_zero_is_rejected() {
    let hidden = 32usize;
    let bytes = build_test_gguf_with_expert_count(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![hidden, 4],
                GGML_TYPE_F32,
                vec![0u8; hidden * 4 * size_of::<f32>()],
            ),
            (
                "token_embd.weight",
                vec![hidden, 2],
                GGML_TYPE_F32,
                vec![0u8; hidden * 2 * size_of::<f32>()],
            ),
        ],
        32,
        hidden,
        0,
    );
    let (path, mapped) = adapter_probe(&bytes, "zero-experts");
    let err = super::adapter::resolve_adapter(&mapped, path.to_str().unwrap()).unwrap_err();
    assert!(err.to_string().contains("expert_count"));
    let load_err =
        match MoeRouter::load_with_mode(path.to_str().unwrap(), 0, 1, RoutingMode::DenseSim) {
            Err(err) => err,
            Ok(_) => panic!("zero expert_count should not load"),
        };
    assert!(load_err.to_string().contains("expert_count"));
    let _ = remove_file(path);
}

#[test]
fn square_gate_scores_follow_contiguous_expert_rows() {
    const HIDDEN: usize = 4;
    let mut gate = [0f32; HIDDEN * HIDDEN];
    gate[2 * HIDDEN] = 5.0;
    gate[2] = 1.0;
    let gate_bytes: Vec<u8> = gate.iter().copied().flat_map(f32::to_le_bytes).collect();
    let bytes = build_test_gguf_with_expert_count(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![HIDDEN, HIDDEN],
                GGML_TYPE_F32,
                gate_bytes,
            ),
            (
                "token_embd.weight",
                vec![HIDDEN, 2],
                GGML_TYPE_F32,
                vec![0u8; HIDDEN * 2 * size_of::<f32>()],
            ),
        ],
        32,
        HIDDEN,
        HIDDEN as u32,
    );
    let path = write_temp_file(&bytes, "square-gate");
    let router =
        MoeRouter::load_with_mode(path.to_str().unwrap(), 0, 1, RoutingMode::DenseSim).unwrap();
    let mut embedding = vec![0.0f32; HIDDEN];
    embedding[0] = 1.0;
    let scores = router.gate_scores(&embedding).unwrap();
    assert!(
        (scores[2] - 5.0).abs() <= 1e-5,
        "expert-row layout should read 5.0, got {scores:?}"
    );
    assert!(
        scores
            .iter()
            .enumerate()
            .all(|(expert, score)| { expert == 2 || score.abs() <= 1e-5 })
    );
    let _ = remove_file(path);
}

#[test]
fn bound_safetensors_router_ignores_later_file_rewrites() {
    let hidden = 4usize;
    let experts = 4usize;
    let vocab = 2usize;
    let mut token = Vec::new();
    for token_id in 0..vocab {
        for dim in 0..hidden {
            token.extend_from_slice(&(token_id as f32 * 10.0 + dim as f32).to_le_bytes());
        }
    }
    let mut gate = vec![0f32; experts * hidden];
    gate[hidden] = 3.0;
    let gate_bytes: Vec<u8> = gate.iter().copied().flat_map(f32::to_le_bytes).collect();
    let bytes = safetensors_bytes(
        &[
            ("general.architecture", "reference_moe"),
            ("general.file_type", "1"),
            ("reference_moe.embedding_length", "4"),
            ("reference_moe.block_count", "1"),
            ("reference_moe.expert_count", "4"),
            ("reference_moe.expert_used_count", "1"),
        ],
        &[
            ("blk.0.ffn_gate_inp.weight", &[experts, hidden], &gate_bytes),
            ("token_embd.weight", &[vocab, hidden], &token),
        ],
    );
    let path = write_temp_file_with_extension(&bytes, "bound-st", "safetensors");
    let mut router =
        MoeRouter::load_with_mode(path.to_str().unwrap(), 0, 1, RoutingMode::DenseSim).unwrap();
    let row = router.extract_token_embedding(1).unwrap();
    let mut embedding = vec![0.0f32; hidden];
    embedding[0] = 1.0;
    let scores = router.gate_scores(&embedding).unwrap();
    std::fs::write(&path, b"not a checkpoint anymore").unwrap();
    assert_eq!(router.extract_token_embedding(1).unwrap(), row);
    assert_eq!(router.gate_scores(&embedding).unwrap(), scores);
    assert!((scores[1] - 3.0).abs() <= 1e-5);
    let _ = remove_file(path);
}

#[test]
fn malformed_safetensors_fails_closed() {
    let path = write_temp_file_with_extension(b"not a checkpoint", "bad-st", "safetensors");
    let err = super::checkpoint::open_mapped_checkpoint(path.to_str().unwrap()).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("unsupported")
            || message.contains("invalid")
            || message.contains("model load"),
        "expected an explicit parser error, got {message}"
    );
    let _ = remove_file(path);
}

fn safetensors_bytes(metadata: &[(&str, &str)], tensors: &[(&str, &[usize], &[u8])]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    let mut meta = serde_json::Map::new();
    for (key, value) in metadata {
        meta.insert(
            (*key).to_owned(),
            serde_json::Value::String((*value).to_owned()),
        );
    }
    header.insert("__metadata__".into(), serde_json::Value::Object(meta));
    for (name, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        let end = data.len();
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&json);
    out.extend_from_slice(&data);
    out
}
