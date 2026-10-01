// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for the backend-neutral stage-execution contracts
//! (`cortex_tensor::stage`) and the dense CPU `ReferenceExecutor`.
//!
//! Every stage-vs-kernel comparison uses **bitwise** `f32` equality
//! (`assert_eq!` on `data()` slices): the reference executor delegates to the
//! same kernels the monolithic `TransformerLM` forward pass uses, so a composed
//! topology must reproduce the model's output exactly, not merely approximately.
//!
//! The test groups together prove the seven acceptance criteria from issue #57
//! (RM-1822):
//!
//! 1. Block-level equivalence against `TransformerBlock`/`MultiHeadAttention`/
//!    `FeedForward` kernels.
//! 2. Full composition via `run_topology` with token ingress, for zero-layer
//!    and multi-layer models.
//! 3. Hidden-state injection/extraction across sub-spans, with a test-only
//!    recording executor proving the trait is externally implementable.
//! 4. MoE rejection + failure-atomicity.
//! 5. Structure-only stable stage IDs/kinds/wiring.

use std::collections::BTreeMap;

use cortex_tensor::stage::{
    AnnCapabilities, AnnExecutor, AnnStage, AnnTopology, DType, ExternalBindings, NormKind,
    ReferenceExecutor, StageId, StageInput, StageKind, StageSource, StageTensor, run_topology,
};
use cortex_tensor::tensor::ops::try_layer_norm;
use cortex_tensor::transformer::{MultiHeadAttention, TransformerConfig, TransformerLM};
use cortex_tensor::{CortexError, Result, Tensor};

/// Epsilon the reference transformer uses for every LayerNorm. The crate const
/// is `pub(crate)`, so the integration test (an external crate) re-states it;
/// the full-composition tests independently prove the executor and the model
/// agree bit for bit, which would fail immediately if this drifted.
const LAYER_NORM_EPS: f32 = 1e-5;

/// A small, valid multi-layer model so the tests stay fast while still
/// exercising cross-block wiring.
fn micro_model() -> TransformerLM {
    TransformerLM::try_new(micro_config()).expect("valid micro model")
}

fn micro_config() -> TransformerConfig {
    TransformerConfig {
        vocab_size: 6,
        dim: 8,
        num_heads: 2,
        num_layers: 2,
        ff_dim: 16,
        max_seq_len: 4,
    }
}

/// A short, in-vocabulary, within-`max_seq_len` token slice.
fn tokens() -> Vec<u32> {
    vec![1u32, 3, 0]
}

fn id(s: &str) -> StageId {
    StageId::new(s).expect("valid stage id")
}

// ── (1) Block-level equivalence ──────────────────────────────────────────────

/// Each stage of `blocks.0` run individually must equal the corresponding step
/// of `TransformerBlock::try_forward` and the standalone attention / FFN
/// kernels, bit for bit, and report the expected shape and `DType::F32`.
#[test]
fn block_stages_match_kernels_bitwise() {
    let model = micro_model();
    let (mut exec, topo) = ReferenceExecutor::from_transformer(&model).expect("executor");
    let toks = tokens();
    let seq = toks.len();
    let dim = model.config.dim;

    // Produce every stage output through the driver with token ingress.
    let bindings = ExternalBindings::new().with_tokens("tokens", &toks);
    let outputs = run_topology(&mut exec, &topo, &bindings).expect("run topology");

    // The block operates on the embedding output.
    let embedding = &outputs[&id("embedding")];
    let block = &model.blocks[0];

    // norm1 = layer_norm(embedding, ln1)
    let expected_norm1 =
        try_layer_norm(embedding, &block.ln1_w, &block.ln1_b, LAYER_NORM_EPS).unwrap();
    let norm1 = &outputs[&id("blocks.0.norm1")];
    assert_eq!(norm1.data(), expected_norm1.data(), "norm1 bitwise");

    // attention = attn(norm1); also equals the standalone MultiHeadAttention.
    let expected_attn: Tensor = block.attn.try_forward(&expected_norm1).unwrap();
    let standalone_attn: &MultiHeadAttention = &block.attn;
    let standalone_attn_out = standalone_attn.try_forward(norm1).unwrap();
    let attention = &outputs[&id("blocks.0.attention")];
    assert_eq!(attention.data(), expected_attn.data(), "attention bitwise");
    assert_eq!(
        attention.data(),
        standalone_attn_out.data(),
        "attention equals standalone MultiHeadAttention::try_forward"
    );

    // residual1 = embedding + attention
    let expected_residual1 = embedding.try_add(&expected_attn).unwrap();
    let residual1 = &outputs[&id("blocks.0.residual1")];
    assert_eq!(
        residual1.data(),
        expected_residual1.data(),
        "residual1 bitwise"
    );

    // norm2 = layer_norm(residual1, ln2)
    let expected_norm2 = try_layer_norm(
        &expected_residual1,
        &block.ln2_w,
        &block.ln2_b,
        LAYER_NORM_EPS,
    )
    .unwrap();
    let norm2 = &outputs[&id("blocks.0.norm2")];
    assert_eq!(norm2.data(), expected_norm2.data(), "norm2 bitwise");

    // mlp = ffn(norm2); also equals the standalone FeedForward.
    let expected_mlp = block.ffn.try_forward(&expected_norm2).unwrap();
    let standalone_mlp = block.ffn.try_forward(norm2).unwrap();
    let mlp = &outputs[&id("blocks.0.mlp")];
    assert_eq!(mlp.data(), expected_mlp.data(), "mlp bitwise");
    assert_eq!(
        mlp.data(),
        standalone_mlp.data(),
        "mlp equals standalone FeedForward::try_forward"
    );

    // residual2 = residual1 + mlp, which is exactly TransformerBlock::try_forward.
    let expected_residual2 = expected_residual1.try_add(&expected_mlp).unwrap();
    let block_forward = block.try_forward(embedding).unwrap();
    let residual2 = &outputs[&id("blocks.0.residual2")];
    assert_eq!(
        residual2.data(),
        expected_residual2.data(),
        "residual2 bitwise"
    );
    assert_eq!(
        residual2.data(),
        block_forward.data(),
        "residual2 equals TransformerBlock::try_forward"
    );

    // Every block stage reports [seq, dim] and F32.
    for stage_id in [
        "blocks.0.norm1",
        "blocks.0.attention",
        "blocks.0.residual1",
        "blocks.0.norm2",
        "blocks.0.mlp",
        "blocks.0.residual2",
    ] {
        let meta = outputs[&id(stage_id)].meta();
        assert_eq!(meta.shape, vec![seq, dim], "{stage_id} shape");
        assert_eq!(meta.dtype, DType::F32, "{stage_id} dtype");
    }
}

// ── (2) Full composition via the driver ──────────────────────────────────────

/// Run the whole topology with token ingress and assert `readout` equals
/// `TransformerLM::try_forward` and `final_norm` equals
/// `TransformerLM::try_hidden_states`, bit for bit. Covers a multi-layer model.
#[test]
fn full_composition_matches_model_multi_layer() {
    let model = micro_model();
    assert_composition_matches_model(&model, &tokens());
}

/// Same, for a zero-layer model: the topology is just `embedding` → `final_norm`
/// → `readout`, and must still reproduce the model exactly.
#[test]
fn full_composition_matches_model_zero_layer() {
    let cfg = TransformerConfig {
        num_layers: 0,
        ..micro_config()
    };
    let model = TransformerLM::try_new(cfg).expect("valid zero-layer model");
    assert_composition_matches_model(&model, &tokens());

    // A zero-layer topology has no `blocks.*` stages.
    let (_exec, topo) = ReferenceExecutor::from_transformer(&model).expect("executor");
    let ids: Vec<&str> = topo.stages().iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, vec!["embedding", "final_norm", "readout"]);
}

fn assert_composition_matches_model(model: &TransformerLM, toks: &[u32]) {
    let (mut exec, topo) = ReferenceExecutor::from_transformer(model).expect("executor");
    let bindings = ExternalBindings::new().with_tokens("tokens", toks);
    let outputs = run_topology(&mut exec, &topo, &bindings).expect("run topology");

    let expected_logits = model.try_forward(toks).unwrap();
    let readout = &outputs[&id("readout")];
    assert_eq!(
        readout.data(),
        expected_logits.data(),
        "readout equals TransformerLM::try_forward"
    );
    assert_eq!(
        readout.meta().shape,
        vec![toks.len(), model.config.vocab_size]
    );

    let expected_hidden = model.try_hidden_states(toks).unwrap();
    let final_norm = &outputs[&id("final_norm")];
    assert_eq!(
        final_norm.data(),
        expected_hidden.data(),
        "final_norm equals TransformerLM::try_hidden_states"
    );
    assert_eq!(final_norm.meta().shape, vec![toks.len(), model.config.dim]);
}

// ── (3) Hidden-state injection / extraction + recording executor ─────────────

/// A test-only executor that wraps any `AnnExecutor`, delegating `execute` to
/// the inner backend while recording the id of every stage it runs. It both
/// proves the trait is externally implementable and lets a test assert which
/// stages a sub-span run actually executed.
struct RecordingExecutor<'a, E: AnnExecutor> {
    inner: &'a mut E,
    executed: Vec<StageId>,
}

impl<'a, E: AnnExecutor> RecordingExecutor<'a, E> {
    fn new(inner: &'a mut E) -> Self {
        Self {
            inner,
            executed: Vec::new(),
        }
    }
}

impl<E: AnnExecutor> AnnExecutor for RecordingExecutor<'_, E> {
    type Tensor = E::Tensor;

    fn capabilities(&self) -> AnnCapabilities {
        self.inner.capabilities()
    }

    fn execute(
        &mut self,
        stage: &AnnStage,
        inputs: &[StageInput<Self::Tensor>],
    ) -> Result<Self::Tensor> {
        // Record before delegating so a failing stage is still observed; the
        // inner executor remains the single source of truth for the output.
        self.executed.push(stage.id.clone());
        self.inner.execute(stage, inputs)
    }
}

/// Build a topology for the sub-span `start..=end`, re-wiring the span's
/// earlier-stage dependencies to a single external ingress named `ingress`.
///
/// `SubSpan` reports those dependencies via `external_stage_deps`; a span with
/// exactly one such dependency can be driven by binding one tensor.
fn span_topology(topo: &AnnTopology, start: &StageId, end: &StageId, ingress: &str) -> AnnTopology {
    let span = topo.sub_span(start, end).expect("valid span");
    assert_eq!(
        span.external_stage_deps.len(),
        1,
        "this helper expects exactly one external stage dependency"
    );
    let dep = span.external_stage_deps[0].clone();

    let rewired: Vec<AnnStage> = span
        .stages
        .iter()
        .map(|stage| {
            let inputs = stage
                .inputs
                .iter()
                .map(|source| match source {
                    StageSource::Stage(s) if *s == dep => {
                        StageSource::External(ingress.to_string())
                    }
                    other => other.clone(),
                })
                .collect();
            AnnStage {
                id: stage.id.clone(),
                kind: stage.kind.clone(),
                inputs,
            }
        })
        .collect();
    AnnTopology::new(rewired).expect("valid span topology")
}

/// Extract `blocks.0.residual2` from the first span, inject it as the ingress of
/// the second span (`blocks.1.norm1..=readout`), and assert the second span's
/// readout equals the full-composition readout bit for bit — while the
/// recording executor confirms the second run never executes `embedding`.
#[test]
fn sub_span_injection_skips_embedding_and_matches_full_run() {
    let model = micro_model();
    let toks = tokens();
    let (mut exec, topo) = ReferenceExecutor::from_transformer(&model).expect("executor");

    // Baseline: full composition readout.
    let full_outputs = {
        let bindings = ExternalBindings::new().with_tokens("tokens", &toks);
        run_topology(&mut exec, &topo, &bindings).expect("full run")
    };
    let full_readout = full_outputs[&id("readout")].clone();

    // Span A: embedding..=blocks.0.residual2, token ingress. Record stages.
    let span_a = topo
        .sub_span(&id("embedding"), &id("blocks.0.residual2"))
        .expect("span a")
        .stages
        .to_vec();
    let span_a_topo = AnnTopology::new(span_a).expect("span a topology");
    let residual2 = {
        let mut rec = RecordingExecutor::new(&mut exec);
        let bindings = ExternalBindings::new().with_tokens("tokens", &toks);
        let out = run_topology(&mut rec, &span_a_topo, &bindings).expect("span a run");
        // Span A does execute embedding.
        assert!(rec.executed.contains(&id("embedding")));
        out[&id("blocks.0.residual2")].clone()
    };

    // Span B: blocks.1.norm1..=readout, injecting the extracted residual2 as the
    // external ingress for the span's single cross-span dependency.
    let span_b_topo = span_topology(&topo, &id("blocks.1.norm1"), &id("readout"), "hidden_in");
    let mut rec = RecordingExecutor::new(&mut exec);
    let bindings = ExternalBindings::new().with_tensor("hidden_in", &residual2);
    let span_b_outputs = run_topology(&mut rec, &span_b_topo, &bindings).expect("span b run");

    // The injected run must NOT re-run embedding (or any block-0 stage).
    assert!(
        !rec.executed.contains(&id("embedding")),
        "span B must not execute embedding; executed={:?}",
        rec.executed
    );
    assert!(
        !rec.executed
            .iter()
            .any(|s| s.as_str().starts_with("blocks.0.")),
        "span B must not re-run block 0 stages; executed={:?}",
        rec.executed
    );

    // And the readout matches the full run bit for bit.
    let span_b_readout = &span_b_outputs[&id("readout")];
    assert_eq!(
        span_b_readout.data(),
        full_readout.data(),
        "sub-span injection readout equals full-composition readout"
    );
}

// ── (4) MoE rejection + failure-atomicity ────────────────────────────────────

/// The reference backend advertises no MoE support and rejects MoE stages with
/// a structured `UnsupportedStage` carrying its backend name, the stage id, and
/// the kind — and a rejected stage leaves the executor able to run a supported
/// stage with an unchanged result (failure-atomicity).
#[test]
fn moe_stages_are_rejected_and_rejection_is_atomic() {
    let model = micro_model();
    let caps = ReferenceExecutor::new().capabilities();

    // Capabilities report MoE kinds as unsupported.
    assert!(!caps.supports(&StageKind::MoeRouter {
        num_experts: 4,
        top_k: 2
    }));
    assert!(!caps.supports(&StageKind::MoeExpert { expert_index: 0 }));

    // Direct execute on a MoE stage returns UnsupportedStage with full context.
    let mut exec = ReferenceExecutor::new();
    let x = Tensor::try_from_vec(vec![0.0; model.config.dim], &[1, model.config.dim]).unwrap();
    let router = AnnStage {
        id: id("router"),
        kind: StageKind::MoeRouter {
            num_experts: 4,
            top_k: 2,
        },
        inputs: vec![],
    };
    let err = exec
        .execute(&router, &[StageInput::Tensor(&x)])
        .unwrap_err();
    match err {
        CortexError::UnsupportedStage {
            backend,
            stage_id,
            kind,
        } => {
            assert_eq!(backend, "reference");
            assert_eq!(stage_id, "router");
            assert!(kind.contains("MoeRouter"), "kind carried: {kind}");
        }
        other => panic!("expected UnsupportedStage, got {other:?}"),
    }

    // A topology with a MoE stage AFTER a normalization stage: the driver runs
    // the norm, then returns UnsupportedStage at the MoE stage.
    let norm_id = id("norm");
    let moe_id = id("expert");
    let mut composed = ReferenceExecutor::new().bind(
        norm_id.clone(),
        cortex_tensor::stage::ReferenceStageParams::Normalization {
            weight: &model.blocks[0].ln1_w,
            bias: Some(&model.blocks[0].ln1_b),
            eps: LAYER_NORM_EPS,
            kind: NormKind::Layer,
        },
    );
    let stages = vec![
        AnnStage {
            id: norm_id.clone(),
            kind: StageKind::Normalization {
                kind: NormKind::Layer,
            },
            inputs: vec![StageSource::External("hidden".to_string())],
        },
        AnnStage {
            id: moe_id.clone(),
            kind: StageKind::MoeExpert { expert_index: 0 },
            inputs: vec![StageSource::Stage(norm_id.clone())],
        },
    ];
    let moe_topo = AnnTopology::new(stages).expect("valid topology");
    let hidden = Tensor::try_from_vec(vec![0.5; model.config.dim], &[1, model.config.dim]).unwrap();
    let bindings = ExternalBindings::new().with_tensor("hidden", &hidden);
    let err = run_topology(&mut composed, &moe_topo, &bindings).unwrap_err();
    assert!(
        matches!(err, CortexError::UnsupportedStage { ref stage_id, .. } if stage_id == "expert"),
        "driver should fail at the MoE stage, got {err:?}"
    );

    // Failure-atomicity: after the rejection, the SAME executor still runs the
    // supported normalization stage, producing the unchanged expected output.
    let norm_descriptor = AnnStage {
        id: norm_id.clone(),
        kind: StageKind::Normalization {
            kind: NormKind::Layer,
        },
        inputs: vec![StageSource::External("hidden".to_string())],
    };
    let after = composed
        .execute(&norm_descriptor, &[StageInput::Tensor(&hidden)])
        .expect("supported stage still runs after a rejection");
    let expected = try_layer_norm(
        &hidden,
        &model.blocks[0].ln1_w,
        &model.blocks[0].ln1_b,
        LAYER_NORM_EPS,
    )
    .unwrap();
    assert_eq!(
        after.data(),
        expected.data(),
        "supported stage output is unchanged by the earlier rejection"
    );
}

// ── (5) Structure-only stable IDs ────────────────────────────────────────────

/// Two models from the SAME config but different random weights must produce
/// identical ordered stage ids, kinds, and wiring: topology derives only from
/// structure, never weights.
#[test]
fn topology_is_stable_across_weights() {
    let a = micro_model();
    let b = micro_model(); // fresh random weights, same config

    let (_ea, topo_a) = ReferenceExecutor::from_transformer(&a).expect("executor a");
    let (_eb, topo_b) = ReferenceExecutor::from_transformer(&b).expect("executor b");

    let wiring = |topo: &AnnTopology| -> Vec<(String, StageKind, Vec<StageSource>)> {
        topo.stages()
            .iter()
            .map(|s| (s.id.as_str().to_string(), s.kind.clone(), s.inputs.clone()))
            .collect()
    };
    assert_eq!(
        wiring(&topo_a),
        wiring(&topo_b),
        "ids, kinds, and wiring are identical across differing weights"
    );
}

/// A different layer count follows the `blocks.{i}.*` schema with the expected
/// number of stages: 2 fixed ends (embedding, final_norm, readout) is 3, plus
/// 6 per block.
#[test]
fn topology_length_follows_layer_count() {
    for num_layers in [0usize, 1, 3] {
        let cfg = TransformerConfig {
            num_layers,
            ..micro_config()
        };
        let model = TransformerLM::try_new(cfg).expect("valid model");
        let (_exec, topo) = ReferenceExecutor::from_transformer(&model).expect("executor");

        let expected_len = 3 + 6 * num_layers;
        assert_eq!(
            topo.stages().len(),
            expected_len,
            "num_layers={num_layers} should yield {expected_len} stages"
        );

        // The per-block ids follow blocks.{i}.{norm1,attention,residual1,norm2,mlp,residual2}.
        let ids: Vec<&str> = topo.stages().iter().map(|s| s.id.as_str()).collect();
        for i in 0..num_layers {
            for suffix in [
                "norm1",
                "attention",
                "residual1",
                "norm2",
                "mlp",
                "residual2",
            ] {
                let expected = format!("blocks.{i}.{suffix}");
                assert!(
                    ids.iter().any(|s| *s == expected),
                    "missing stage id {expected} for num_layers={num_layers}"
                );
            }
        }
    }
}

/// A free function kept to document that `BTreeMap` output keys sort by the
/// `StageId` ordering; unused helper warnings are avoided by referencing it in
/// a trivial assertion.
#[test]
fn output_map_is_keyed_by_stage_id() {
    let model = micro_model();
    let (mut exec, topo) = ReferenceExecutor::from_transformer(&model).expect("executor");
    let toks = tokens();
    let bindings = ExternalBindings::new().with_tokens("tokens", &toks);
    let outputs: BTreeMap<StageId, Tensor> =
        run_topology(&mut exec, &topo, &bindings).expect("run topology");
    assert!(outputs.contains_key(&id("embedding")));
    assert!(outputs.contains_key(&id("readout")));
}
