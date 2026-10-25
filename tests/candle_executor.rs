// SPDX-License-Identifier: MIT OR Apache-2.0

//! Candle `AnnExecutor` coverage. Compiled only with `--features candle`.

#![cfg(feature = "candle")]

use std::collections::BTreeSet;

use cortex_tensor::adapter::{
    AttentionWeights, CandleExecutor, EmbeddingWeights, GatedMlpWeights, HiddenState,
    LayerNormWeights, LinearWeights, MlpWeights, RmsNormWeights, export_hidden_state,
    export_reference, import_hidden_state, import_reference,
};
use cortex_tensor::stage::{
    AnnExecutor, AnnStage, AnnTopology, DType, ExternalBindings, MlpKind, NormKind, StageId,
    StageInput, StageKind, StageSource, StageTensor, run_topology,
};
use cortex_tensor::tensor::ops::{try_embedding, try_layer_norm, try_matmul, try_rms_norm};
use cortex_tensor::{CortexError, Tensor};

fn stage_id(name: &str) -> StageId {
    StageId::new(name).expect("valid stage id")
}

fn descriptor(name: &str, kind: StageKind) -> AnnStage {
    AnnStage {
        id: stage_id(name),
        kind,
        inputs: Vec::new(),
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    let mut max_diff = 0.0f32;
    for (actual, expected) in actual.iter().zip(expected) {
        max_diff = max_diff.max((actual - expected).abs());
    }
    assert!(
        max_diff <= tol,
        "max abs diff {max_diff} exceeds {tol}\nactual {actual:?}\nexpected {expected:?}"
    );
}

#[test]
fn capabilities_advertise_candle_cpu_surface() {
    let caps = CandleExecutor::new().capabilities();
    assert_eq!(caps.backend_name, "candle");
    assert!(!caps.stateful);
    assert_eq!(caps.supported_dtypes, BTreeSet::from([DType::F32]));
    assert!(caps.supports(&StageKind::Embedding));
    assert!(caps.supports(&StageKind::Attention));
    assert!(caps.supports(&StageKind::Normalization {
        kind: NormKind::Layer
    }));
    assert!(caps.supports(&StageKind::Normalization {
        kind: NormKind::Rms
    }));
    assert!(caps.supports(&StageKind::Mlp {
        kind: MlpKind::Dense
    }));
    assert!(caps.supports(&StageKind::Mlp {
        kind: MlpKind::Gated
    }));
    assert!(caps.supports(&StageKind::Add));
    assert!(caps.supports(&StageKind::Readout));
    assert!(!caps.supports(&StageKind::MoeRouter {
        num_experts: 2,
        top_k: 1
    }));
    assert!(!caps.supports(&StageKind::MoeExpert { expert_index: 0 }));
    assert!(!caps.supports(&StageKind::Custom {
        name: "wild".to_string()
    }));
}

#[test]
fn hidden_state_roundtrip_preserves_bits() {
    let state = HiddenState {
        data: vec![1.0, -0.0, 0.5, -2.0, 3.0, 4.0],
        shape: vec![2, 3],
    };
    let tensor = import_hidden_state(&state).unwrap();
    let exported = export_hidden_state(&tensor).unwrap();
    assert_eq!(exported.data, state.data);
    assert_eq!(exported.shape, state.shape);

    let reference = Tensor::try_from_vec(state.data.clone(), &state.shape).unwrap();
    let imported = import_reference(&reference).unwrap();
    let back = export_reference(&imported).unwrap();
    assert_eq!(back.data(), reference.data());
    assert_eq!(back.shape(), reference.shape());
}

#[test]
fn import_rejects_shape_mismatch() {
    let err = import_hidden_state(&HiddenState {
        data: vec![1.0, 2.0, 3.0],
        shape: vec![2, 2],
    })
    .unwrap_err();
    assert!(matches!(err, CortexError::ShapeMismatch { .. }));
}

#[test]
fn unsupported_stage_does_not_drop_bound_params() {
    let mut exec = CandleExecutor::new();
    let weight = [1.0f32, 0.0, 0.0, 1.0];
    exec.bind_readout(
        stage_id("readout"),
        LinearWeights {
            weight: &weight,
            in_dim: 2,
            out_dim: 2,
            bias: None,
        },
    )
    .unwrap();

    let hidden = import_reference(&Tensor::try_from_vec(vec![1.0, 2.0], &[1, 2]).unwrap()).unwrap();
    let err = exec
        .execute(
            &descriptor(
                "router",
                StageKind::MoeRouter {
                    num_experts: 2,
                    top_k: 1,
                },
            ),
            &[StageInput::Tensor(&hidden)],
        )
        .unwrap_err();
    assert!(matches!(
        err,
        CortexError::UnsupportedStage {
            backend: "candle",
            ..
        }
    ));

    let out = exec
        .execute(
            &descriptor("readout", StageKind::Readout),
            &[StageInput::Tensor(&hidden)],
        )
        .unwrap();
    let exported = export_reference(&out).unwrap();
    assert_eq!(exported.data(), &[1.0, 2.0]);
}

#[test]
fn bind_failure_keeps_existing_params() {
    let mut exec = CandleExecutor::new();
    let weight = [1.0f32, 0.0, 0.0, 1.0];
    exec.bind_readout(
        stage_id("readout"),
        LinearWeights {
            weight: &weight,
            in_dim: 2,
            out_dim: 2,
            bias: None,
        },
    )
    .unwrap();
    let err = exec
        .bind_layer_norm(
            stage_id("norm"),
            LayerNormWeights {
                weight: &[1.0, 1.0],
                bias: &[0.0, 0.0],
                eps: 0.0,
            },
        )
        .unwrap_err();
    assert!(matches!(err, CortexError::InvalidEpsilon { .. }));

    let hidden = import_reference(&Tensor::try_from_vec(vec![3.0, 4.0], &[1, 2]).unwrap()).unwrap();
    let out = exec
        .execute(
            &descriptor("readout", StageKind::Readout),
            &[StageInput::Tensor(&hidden)],
        )
        .unwrap();
    assert_eq!(export_reference(&out).unwrap().data(), &[3.0, 4.0]);
    let missing = exec
        .execute(
            &descriptor(
                "norm",
                StageKind::Normalization {
                    kind: NormKind::Layer,
                },
            ),
            &[StageInput::Tensor(&hidden)],
        )
        .unwrap_err();
    assert!(matches!(missing, CortexError::MissingStageParams { .. }));
}

#[test]
fn execute_reports_arity_and_input_kind() {
    let mut exec = CandleExecutor::new();
    let weight = [1.0f32, 0.0, 0.0, 1.0];
    exec.bind_readout(
        stage_id("readout"),
        LinearWeights {
            weight: &weight,
            in_dim: 2,
            out_dim: 2,
            bias: None,
        },
    )
    .unwrap();
    let hidden = import_reference(&Tensor::try_from_vec(vec![1.0, 2.0], &[1, 2]).unwrap()).unwrap();
    let arity = exec
        .execute(
            &descriptor("readout", StageKind::Readout),
            &[StageInput::Tensor(&hidden), StageInput::Tensor(&hidden)],
        )
        .unwrap_err();
    assert!(matches!(
        arity,
        CortexError::StageInputArity {
            expected: 1,
            got: 2,
            ..
        }
    ));
    let ids = [0u32];
    let kind = exec
        .execute(
            &descriptor("readout", StageKind::Readout),
            &[StageInput::Tokens(&ids)],
        )
        .unwrap_err();
    assert!(matches!(kind, CortexError::StageInputKind { index: 0, .. }));
}

#[test]
fn embedding_matches_reference_bitwise() {
    let token_weight = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let position_weight = [0.01f32, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09];
    let ids = [2u32, 0, 1];
    let tok = Tensor::try_from_vec(token_weight.to_vec(), &[3, 3]).unwrap();
    let pos = Tensor::try_from_vec(position_weight.to_vec(), &[3, 3]).unwrap();
    let expected = try_embedding(&tok, &ids)
        .unwrap()
        .try_add(&try_embedding(&pos, &[0, 1, 2]).unwrap())
        .unwrap();

    let mut exec = CandleExecutor::new();
    exec.bind_embedding(
        stage_id("embedding"),
        EmbeddingWeights {
            token_weight: &token_weight,
            vocab: 3,
            dim: 3,
            position_weight: &position_weight,
            max_seq_len: 3,
        },
    )
    .unwrap();
    let out = exec
        .execute(
            &descriptor("embedding", StageKind::Embedding),
            &[StageInput::Tokens(&ids)],
        )
        .unwrap();
    let exported = export_reference(&out).unwrap();
    assert_eq!(exported.data(), expected.data());
    assert_eq!(exported.shape(), &[3, 3]);
}

#[test]
fn embedding_empty_sequence_and_token_errors() {
    let token_weight = [1.0f32, 2.0, 3.0, 4.0];
    let position_weight = [0.0f32, 0.0, 0.0, 0.0];
    let mut exec = CandleExecutor::new();
    exec.bind_embedding(
        stage_id("embedding"),
        EmbeddingWeights {
            token_weight: &token_weight,
            vocab: 2,
            dim: 2,
            position_weight: &position_weight,
            max_seq_len: 2,
        },
    )
    .unwrap();
    let empty = exec
        .execute(
            &descriptor("embedding", StageKind::Embedding),
            &[StageInput::Tokens(&[])],
        )
        .unwrap();
    let exported = export_reference(&empty).unwrap();
    assert_eq!(exported.shape(), &[0, 2]);
    assert!(exported.data().is_empty());

    let oov = exec
        .execute(
            &descriptor("embedding", StageKind::Embedding),
            &[StageInput::Tokens(&[2])],
        )
        .unwrap_err();
    match oov {
        CortexError::StageFailed { source, .. } => {
            assert!(matches!(
                *source,
                CortexError::TokenIndex {
                    index: 2,
                    vocab_size: 2
                }
            ));
        }
        other => panic!("expected StageFailed, got {other:?}"),
    }

    let too_long = exec
        .execute(
            &descriptor("embedding", StageKind::Embedding),
            &[StageInput::Tokens(&[0, 1, 0])],
        )
        .unwrap_err();
    match too_long {
        CortexError::StageFailed { source, .. } => {
            assert!(matches!(
                *source,
                CortexError::InputLengthMismatch {
                    expected: 2,
                    got: 3
                }
            ));
        }
        other => panic!("expected StageFailed, got {other:?}"),
    }
}

#[test]
fn readout_matches_reference_matmul() {
    // Cortex layout [in, out] = [[1, 2], [3, 4]]. x @ W = [4, 6].
    let weight = [1.0f32, 2.0, 3.0, 4.0];
    let x = Tensor::try_from_vec(vec![1.0, 1.0], &[1, 2]).unwrap();
    let w = Tensor::try_from_vec(weight.to_vec(), &[2, 2]).unwrap();
    let expected = try_matmul(&x, &w).unwrap();

    let mut exec = CandleExecutor::new();
    exec.bind_readout(
        stage_id("readout"),
        LinearWeights {
            weight: &weight,
            in_dim: 2,
            out_dim: 2,
            bias: Some(&[0.5, -0.5]),
        },
    )
    .unwrap();
    let hidden = import_reference(&x).unwrap();
    let out = exec
        .execute(
            &descriptor("readout", StageKind::Readout),
            &[StageInput::Tensor(&hidden)],
        )
        .unwrap();
    let exported = export_reference(&out).unwrap();
    let bias = Tensor::try_from_vec(vec![0.5, -0.5], &[1, 2]).unwrap();
    let with_bias = expected.try_add(&bias).unwrap();
    assert_eq!(exported.data(), with_bias.data());
}

#[test]
fn readout_accepts_rank3_batch() {
    let weight = [1.0f32, 0.0, 0.0, 1.0];
    let x = Tensor::try_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 1, 2]).unwrap();
    let mut exec = CandleExecutor::new();
    exec.bind_readout(
        stage_id("readout"),
        LinearWeights {
            weight: &weight,
            in_dim: 2,
            out_dim: 2,
            bias: None,
        },
    )
    .unwrap();
    let hidden = import_reference(&x).unwrap();
    let out = exec
        .execute(
            &descriptor("readout", StageKind::Readout),
            &[StageInput::Tensor(&hidden)],
        )
        .unwrap();
    assert_eq!(export_reference(&out).unwrap().data(), x.data());
    assert_eq!(out.shape(), &[2, 1, 2]);
}

#[test]
fn layer_and_rms_norm_match_reference() {
    let x = Tensor::try_from_vec(vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.5, 0.0], &[2, 4]).unwrap();
    let weight = [0.5f32, 1.0, 1.5, 2.0];
    let bias = [0.1f32, 0.0, -0.1, 0.2];
    let eps = 1e-5f32;
    let expected_ln = try_layer_norm(
        &x,
        &Tensor::try_from_vec(weight.to_vec(), &[4]).unwrap(),
        &Tensor::try_from_vec(bias.to_vec(), &[4]).unwrap(),
        eps,
    )
    .unwrap();
    let expected_rms = try_rms_norm(
        &x,
        &Tensor::try_from_vec(weight.to_vec(), &[4]).unwrap(),
        eps,
    )
    .unwrap();

    let mut exec = CandleExecutor::new();
    exec.bind_layer_norm(
        stage_id("ln"),
        LayerNormWeights {
            weight: &weight,
            bias: &bias,
            eps,
        },
    )
    .unwrap();
    exec.bind_rms_norm(
        stage_id("rms"),
        RmsNormWeights {
            weight: &weight,
            eps,
        },
    )
    .unwrap();
    let hidden = import_reference(&x).unwrap();
    let ln = exec
        .execute(
            &descriptor(
                "ln",
                StageKind::Normalization {
                    kind: NormKind::Layer,
                },
            ),
            &[StageInput::Tensor(&hidden)],
        )
        .unwrap();
    let rms = exec
        .execute(
            &descriptor(
                "rms",
                StageKind::Normalization {
                    kind: NormKind::Rms,
                },
            ),
            &[StageInput::Tensor(&hidden)],
        )
        .unwrap();
    assert_close(
        export_reference(&ln).unwrap().data(),
        expected_ln.data(),
        1e-5,
    );
    assert_close(
        export_reference(&rms).unwrap().data(),
        expected_rms.data(),
        1e-5,
    );
}

#[test]
fn dense_mlp_matches_reference_gelu() {
    let x = Tensor::try_from_vec(vec![0.5, -1.0], &[1, 2]).unwrap();
    let up = [0.2f32, -0.1, 0.0, 0.3];
    let down = [1.0f32, -0.5, 0.25, 0.0];
    let hidden = try_matmul(&x, &Tensor::try_from_vec(up.to_vec(), &[2, 2]).unwrap()).unwrap();
    let expected = try_matmul(
        &hidden.gelu(),
        &Tensor::try_from_vec(down.to_vec(), &[2, 2]).unwrap(),
    )
    .unwrap();

    let mut exec = CandleExecutor::new();
    exec.bind_dense_mlp(
        stage_id("mlp"),
        MlpWeights {
            up: LinearWeights {
                weight: &up,
                in_dim: 2,
                out_dim: 2,
                bias: None,
            },
            down: LinearWeights {
                weight: &down,
                in_dim: 2,
                out_dim: 2,
                bias: None,
            },
        },
    )
    .unwrap();
    let input = import_reference(&x).unwrap();
    let out = exec
        .execute(
            &descriptor(
                "mlp",
                StageKind::Mlp {
                    kind: MlpKind::Dense,
                },
            ),
            &[StageInput::Tensor(&input)],
        )
        .unwrap();
    assert_close(
        export_reference(&out).unwrap().data(),
        expected.data(),
        1e-5,
    );
}

#[test]
fn gated_mlp_matches_swiglu() {
    let x = Tensor::try_from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
    let gate_w = [1.0f32, 0.0, 0.0, 1.0];
    let up_w = [0.5f32, 0.0, 0.0, 0.25];
    let down_w = [1.0f32, 0.0, 0.0, 1.0];
    let gate = try_matmul(&x, &Tensor::try_from_vec(gate_w.to_vec(), &[2, 2]).unwrap()).unwrap();
    let up = try_matmul(&x, &Tensor::try_from_vec(up_w.to_vec(), &[2, 2]).unwrap()).unwrap();
    let hidden = gate.silu().try_mul(&up).unwrap();
    let expected = try_matmul(
        &hidden,
        &Tensor::try_from_vec(down_w.to_vec(), &[2, 2]).unwrap(),
    )
    .unwrap();

    let mut exec = CandleExecutor::new();
    exec.bind_gated_mlp(
        stage_id("gated"),
        GatedMlpWeights {
            gate: LinearWeights {
                weight: &gate_w,
                in_dim: 2,
                out_dim: 2,
                bias: None,
            },
            up: LinearWeights {
                weight: &up_w,
                in_dim: 2,
                out_dim: 2,
                bias: None,
            },
            down: LinearWeights {
                weight: &down_w,
                in_dim: 2,
                out_dim: 2,
                bias: None,
            },
        },
    )
    .unwrap();
    let input = import_reference(&x).unwrap();
    let out = exec
        .execute(
            &descriptor(
                "gated",
                StageKind::Mlp {
                    kind: MlpKind::Gated,
                },
            ),
            &[StageInput::Tensor(&input)],
        )
        .unwrap();
    assert_close(
        export_reference(&out).unwrap().data(),
        expected.data(),
        1e-5,
    );
}

#[test]
fn attention_matches_reference_within_tolerance() {
    let dim = 8;
    let heads = 2;
    let seq = 3;
    let x_data: Vec<f32> = (0..seq * dim).map(|i| (i as f32 - 8.0) * 0.05).collect();
    let proj = |seed: f32| -> Vec<f32> {
        (0..dim * dim)
            .map(|i| ((i as f32) * 0.01 + seed).sin() * 0.1)
            .collect()
    };
    let wq = proj(0.1);
    let wk = proj(0.2);
    let wv = proj(0.3);
    let wo = proj(0.4);
    let zeros = Tensor::try_zeros(&[1, dim]).unwrap();
    let attn = cortex_tensor::transformer::MultiHeadAttention {
        num_heads: heads,
        head_dim: dim / heads,
        dim,
        wq: Tensor::try_from_vec(wq.clone(), &[dim, dim]).unwrap(),
        wb_q: zeros.clone(),
        wk: Tensor::try_from_vec(wk.clone(), &[dim, dim]).unwrap(),
        wb_k: zeros.clone(),
        wv: Tensor::try_from_vec(wv.clone(), &[dim, dim]).unwrap(),
        wb_v: zeros.clone(),
        wo: Tensor::try_from_vec(wo.clone(), &[dim, dim]).unwrap(),
        wb_o: zeros,
    };
    let x = Tensor::try_from_vec(x_data, &[seq, dim]).unwrap();
    let expected = attn.try_forward(&x).unwrap();

    fn square(weight: &[f32], width: usize) -> LinearWeights<'_> {
        LinearWeights {
            weight,
            in_dim: width,
            out_dim: width,
            bias: None,
        }
    }
    let mut exec = CandleExecutor::new();
    exec.bind_attention(
        stage_id("attn"),
        AttentionWeights {
            q: square(&wq, dim),
            k: square(&wk, dim),
            v: square(&wv, dim),
            o: square(&wo, dim),
            num_heads: heads,
        },
    )
    .unwrap();
    let hidden = import_reference(&x).unwrap();
    let out = exec
        .execute(
            &descriptor("attn", StageKind::Attention),
            &[StageInput::Tensor(&hidden)],
        )
        .unwrap();
    let exported = export_reference(&out).unwrap();
    assert_eq!(exported.shape(), expected.shape());
    assert_close(exported.data(), expected.data(), 1e-4);

    let batched = import_reference(&x.try_reshape(&[1, seq, dim]).unwrap()).unwrap();
    let batched_out = exec
        .execute(
            &descriptor("attn", StageKind::Attention),
            &[StageInput::Tensor(&batched)],
        )
        .unwrap();
    assert_eq!(batched_out.shape(), &[1, seq, dim]);
    assert_close(
        export_reference(&batched_out).unwrap().data(),
        expected.data(),
        1e-4,
    );
}

#[test]
fn topology_keeps_hidden_state_inside_candle() {
    let x = Tensor::try_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let weight = [1.0f32, 1.0];
    let bias = [0.0f32, 0.0];
    let eps = 1e-5f32;
    let norm = try_layer_norm(
        &x,
        &Tensor::try_from_vec(weight.to_vec(), &[2]).unwrap(),
        &Tensor::try_from_vec(bias.to_vec(), &[2]).unwrap(),
        eps,
    )
    .unwrap();
    let expected = x.try_add(&norm).unwrap();

    let mut exec = CandleExecutor::new();
    exec.bind_layer_norm(
        stage_id("norm"),
        LayerNormWeights {
            weight: &weight,
            bias: &bias,
            eps,
        },
    )
    .unwrap();
    let norm_stage = AnnStage {
        id: stage_id("norm"),
        kind: StageKind::Normalization {
            kind: NormKind::Layer,
        },
        inputs: vec![StageSource::External("x".to_string())],
    };
    let add_stage = AnnStage {
        id: stage_id("add"),
        kind: StageKind::Add,
        inputs: vec![
            StageSource::External("x".to_string()),
            StageSource::Stage(stage_id("norm")),
        ],
    };
    let topology = AnnTopology::new(vec![norm_stage, add_stage]).unwrap();
    let hidden = import_reference(&x).unwrap();
    let bindings = ExternalBindings::new()
        .with_tensor("x", &hidden)
        .expect("bind x");
    let outputs = run_topology(&mut exec, &topology, &bindings).unwrap();
    let add = outputs.get(&stage_id("add")).unwrap();
    assert_eq!(add.meta().dtype, DType::F32);
    assert_close(export_reference(add).unwrap().data(), expected.data(), 1e-5);
    // The norm output stayed a Candle tensor; exporting it is a separate boundary.
    let norm_out = outputs.get(&stage_id("norm")).unwrap();
    assert_eq!(norm_out.shape(), &[2, 2]);
    assert!(matches!(norm_out.meta().dtype, DType::F32));
}

#[test]
fn bind_rejects_incompatible_weights() {
    let mut exec = CandleExecutor::new();
    let eye = [1.0f32, 0.0, 0.0, 1.0];
    let square = LinearWeights {
        weight: &eye,
        in_dim: 2,
        out_dim: 2,
        bias: None,
    };
    let wide = LinearWeights {
        weight: &[1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0],
        in_dim: 2,
        out_dim: 3,
        bias: None,
    };
    let dense = exec
        .bind_dense_mlp(
            stage_id("dense"),
            MlpWeights {
                up: wide,
                down: square,
            },
        )
        .unwrap_err();
    assert!(matches!(dense, CortexError::InvalidConfig(_)));

    let gated = exec
        .bind_gated_mlp(
            stage_id("gated"),
            GatedMlpWeights {
                gate: square,
                up: wide,
                down: square,
            },
        )
        .unwrap_err();
    assert!(matches!(gated, CortexError::InvalidConfig(_)));

    let nonsquare = exec
        .bind_attention(
            stage_id("attn"),
            AttentionWeights {
                q: wide,
                k: square,
                v: square,
                o: square,
                num_heads: 1,
            },
        )
        .unwrap_err();
    assert!(matches!(nonsquare, CortexError::InvalidConfig(_)));

    let heads = exec
        .bind_attention(
            stage_id("attn"),
            AttentionWeights {
                q: square,
                k: square,
                v: square,
                o: square,
                num_heads: 3,
            },
        )
        .unwrap_err();
    assert!(matches!(heads, CortexError::InvalidConfig(_)));

    let layer = exec
        .bind_layer_norm(
            stage_id("ln"),
            LayerNormWeights {
                weight: &[1.0, 1.0],
                bias: &[0.0],
                eps: 1e-5,
            },
        )
        .unwrap_err();
    assert!(matches!(layer, CortexError::ShapeMismatch { .. }));

    let rms = exec
        .bind_rms_norm(
            stage_id("rms"),
            RmsNormWeights {
                weight: &[1.0],
                eps: f32::NAN,
            },
        )
        .unwrap_err();
    assert!(matches!(rms, CortexError::InvalidEpsilon { .. }));

    let embed = exec
        .bind_embedding(
            stage_id("embed"),
            EmbeddingWeights {
                token_weight: &[1.0, 0.0],
                vocab: 2,
                dim: 2,
                position_weight: &[1.0, 0.0, 0.0, 1.0],
                max_seq_len: 2,
            },
        )
        .unwrap_err();
    assert!(matches!(embed, CortexError::ShapeMismatch { .. }));
}

#[test]
fn params_kind_mismatch_is_reported_as_mismatch_not_missing() {
    let mut exec = CandleExecutor::new();
    let weight = [1.0f32, 0.0, 0.0, 1.0];
    exec.bind_readout(
        stage_id("shared"),
        LinearWeights {
            weight: &weight,
            in_dim: 2,
            out_dim: 2,
            bias: None,
        },
    )
    .unwrap();
    let hidden = import_reference(&Tensor::try_from_vec(vec![1.0, 2.0], &[1, 2]).unwrap()).unwrap();
    let err = exec
        .execute(
            &descriptor(
                "shared",
                StageKind::Normalization {
                    kind: NormKind::Layer,
                },
            ),
            &[StageInput::Tensor(&hidden)],
        )
        .unwrap_err();
    // Mirrors `ReferenceExecutor` (#97): bound-but-wrong-kind is
    // `StageParamsMismatch`, so a wiring bug is not mistaken for an unbound
    // stage. An id with no binding at all stays `MissingStageParams`.
    assert!(
        matches!(
            &err,
            CortexError::StageParamsMismatch { stage_id, expected, got }
                if stage_id == "shared" && expected == "layer_norm" && got == "readout"
        ),
        "{err}"
    );
    let err = exec
        .execute(
            &descriptor("unbound", StageKind::Readout),
            &[StageInput::Tensor(&hidden)],
        )
        .unwrap_err();
    assert!(
        matches!(err, CortexError::MissingStageParams { .. }),
        "{err}"
    );
}

#[test]
fn npz_roundtrip_through_zip_facade() {
    let tensor =
        candle_core::Tensor::from_slice(&[1f32, 2.0, 3.0, 4.0], (2, 2), &candle_core::Device::Cpu)
            .expect("tensor");
    let path = std::env::temp_dir().join(format!("cortex-tensor-npz-{}.npz", std::process::id()));
    candle_core::Tensor::write_npz(&[("w", &tensor)], &path).expect("write npz");
    let loaded = candle_core::Tensor::read_npz(&path).expect("read npz");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].0, "w");
    let values = loaded[0]
        .1
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec");
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    let by_name = candle_core::Tensor::read_npz_by_name(&path, &["w"]).expect("by name");
    assert_eq!(by_name.len(), 1);
    let _ = std::fs::remove_file(&path);
}
