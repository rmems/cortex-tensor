// SPDX-License-Identifier: MIT OR Apache-2.0

use super::test_fixtures::*;
use super::*;
use crate::error::HybridError;
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
    assert_eq!(model.family(), ModelFamily::Olmoe);
    assert_eq!(model.routing_tensor_name(), "blk.0.ffn_gate_inp.weight");

    let _ = remove_file(path);
}

#[test]
fn test_spiking_sim_state_can_reset() {
    let mut model = MoeRouter::load_with_mode("", 8, 2, RoutingMode::SpikingSim).unwrap();
    let _ = model.forward(&vec![1.0; EMBEDDING_DIM]).unwrap();
    assert!(model.has_state_activity());
    model.reset_state();
    assert!(!model.has_state_activity());
}

#[test]
fn test_real_checkpoint_probe_via_env() {
    let Some(path) = std::env::var("GGUF_CHECKPOINT_PATH").ok() else {
        return;
    };

    let metadata = MoeRouter::probe_model(&path, None).unwrap();
    assert!(!metadata.architecture.is_empty());
    assert!(metadata.hidden_size > 0);
    assert!(metadata.num_experts > 0);
    assert!(!metadata.routing_tensor_name.is_empty());
}

/// Smoke test to exercise the full Q5_K dequant loop (4 chunks of 32 bytes
/// for a 256-wide row). This runs the u1/u2 shifts and dequant math in debug
/// mode (where overflow would previously panic on <<= for u8).
#[test]
fn test_q5k_dequant_smoke_runs_full_block_loop() {
    // Zeroed 176-byte block is enough to execute all 4 ql chunks + shifts
    // without format errors (the math will produce garbage but the control
    // flow and bit ops execute).
    let block = vec![0u8; 176];
    let out =
        super::dequant::dequantize_row_q5_k(&block, 256).expect("Q5_K row size should be accepted");
    assert_eq!(out.len(), 256);
}

fn adapter_probe(bytes: &[u8], label: &str) -> (PathBuf, super::checkpoint::MappedGgufCheckpoint) {
    let path = write_temp_file(bytes, label);
    let (_metadata, mapped) = super::checkpoint::probe_and_map_checkpoint(path.to_str().unwrap())
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
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                GGML_TYPE_IQ3_S,
                vec![0u8; 16],
            ),
            ("token_embd.weight", token_dims, token_ggml, token_payload),
        ],
        32,
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
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());
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
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());
    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("unsupported orientation")
    ));
    let _ = remove_file(path);
}

#[test]
fn test_adapter_resolve_rejects_iq3_s_token_embedding() {
    let gate_payload = vec![0u8; EMBEDDING_DIM * 64 * size_of::<f32>()];
    let checkpoint = adapter_fixture(
        vec![EMBEDDING_DIM, 64],
        gate_payload,
        GGML_TYPE_IQ3_S,
        vec![EMBEDDING_DIM, 32],
        vec![0u8; 16],
    );
    let (path, mapped) = adapter_probe(&checkpoint, "iq3s-token-embd");
    let result =
        super::adapter::resolve_adapter(mapped.metadata(), &mapped, None, path.to_str().unwrap());
    assert!(matches!(
        result,
        Err(HybridError::UnsupportedFormat(msg)) if msg.contains("token embedding tensor")
    ));
    let _ = remove_file(path);
}
