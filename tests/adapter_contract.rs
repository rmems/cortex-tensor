// SPDX-License-Identifier: MIT OR Apache-2.0

//! External-adapter contract tests (#58).
//!
//! These pin the core-owned adapter surface: capability negotiation facts,
//! hidden-state import/export, and structured rejection. Native Candle and Burn
//! types must not appear in the signatures under test.

use std::collections::BTreeSet;

use cortex_tensor::stage::{
    AnnCapabilities, AnnExecutor, AnnStage, AnnTopology, BackendLimitations, DType, DeviceClass,
    ExternalAdapter, ExternalBindings, HiddenStateBuffer, ReferenceExecutor, StageId, StageInput,
    StageKind, StageKindTag, StageSource, StageTensor, run_topology,
};
use cortex_tensor::transformer::{TransformerConfig, TransformerLM};
use cortex_tensor::{CortexError, Result, Tensor};

fn tiny_model(layers: usize) -> TransformerLM {
    TransformerLM::try_new(TransformerConfig {
        vocab_size: 8,
        dim: 4,
        num_heads: 2,
        num_layers: layers,
        ff_dim: 8,
        max_seq_len: 6,
    })
    .expect("tiny model")
}

fn add_stage() -> AnnStage {
    AnnStage {
        id: StageId::new("probe_add").expect("id"),
        kind: StageKind::Add,
        inputs: vec![
            StageSource::External("left".into()),
            StageSource::External("right".into()),
        ],
    }
}

fn hidden(rows: usize, cols: usize, fill: f32) -> Tensor {
    Tensor::try_from_vec(vec![fill; rows * cols], &[rows, cols]).expect("tensor")
}

#[test]
fn reference_reports_the_same_external_adapter_facts_as_ann_executor() {
    let model = tiny_model(1);
    let (executor, _) = ReferenceExecutor::from_transformer(&model).expect("bind");
    let ann = AnnExecutor::capabilities(&executor);
    let external = ExternalAdapter::capabilities(&executor);

    assert_eq!(external.ann.backend_name, ann.backend_name);
    assert_eq!(external.ann.supported_tags, ann.supported_tags);
    assert_eq!(external.ann.supported_dtypes, ann.supported_dtypes);
    assert_eq!(external.ann.stateful, ann.stateful);
    assert_eq!(external.devices, BTreeSet::from([DeviceClass::Cpu]));
    assert!(!external.batch);
    assert!(!external.sequence_cache);
    assert!(external.hidden_state_io);
    assert!(
        external
            .limitations
            .unsupported_tags
            .contains(&StageKindTag::MoeRouter)
    );
    assert!(
        external
            .limitations
            .unsupported_tags
            .contains(&StageKindTag::GatedMlp)
    );
    assert!(
        external
            .limitations
            .notes
            .iter()
            .any(|note| note.contains("f32"))
    );
}

#[test]
fn negotiation_document_names_only_cortex_facts() {
    let model = tiny_model(0);
    let (executor, _) = ReferenceExecutor::from_transformer(&model).expect("bind");
    let doc = ExternalAdapter::capabilities(&executor).negotiation_document();

    assert_eq!(doc.backend_name, "reference");
    assert!(doc.stage_kinds.iter().any(|kind| kind == "embedding"));
    assert!(doc.stage_kinds.iter().any(|kind| kind == "dense_mlp"));
    assert!(!doc.stage_kinds.iter().any(|kind| kind.contains("moe")));
    assert_eq!(doc.dtypes, vec!["f32".to_string()]);
    assert_eq!(doc.devices, vec!["cpu".to_string()]);
    assert!(doc.hidden_state_io);
    assert!(!doc.batch);
    assert!(!doc.sequence_cache);
    assert!(!doc.stateful);
    assert!(doc.unsupported.iter().any(|item| item == "moe_router"));
    assert!(
        doc.limitations
            .iter()
            .any(|note| note.contains("hybrid-fusion"))
    );
    let rendered = format!("{doc:?}");
    assert!(!rendered.contains("candle_core"));
    assert!(!rendered.contains("burn::"));
}

#[test]
fn hidden_state_round_trip_preserves_reference_bits() {
    let values = vec![1.0f32, -0.0, 3.5, 4.0];
    let tensor = Tensor::try_from_vec(values.clone(), &[2, 2]).expect("tensor");
    let exported = ReferenceExecutor::export_hidden(&tensor).expect("export");
    assert_eq!(exported.shape, vec![2, 2]);
    assert_eq!(exported.dtype, DType::F32);
    assert_eq!(exported.data, values);

    let imported = ReferenceExecutor::import_hidden(&exported).expect("import");
    assert_eq!(imported.data(), tensor.data());
    assert_eq!(imported.shape(), tensor.shape());
}

#[test]
fn hidden_state_import_rejects_non_f32_and_shape_mismatch() {
    let bad_dtype = HiddenStateBuffer {
        data: vec![1.0, 2.0],
        shape: vec![2],
        dtype: DType::F16,
    };
    let err = ReferenceExecutor::import_hidden(&bad_dtype).unwrap_err();
    assert!(
        matches!(err, CortexError::StageDTypeMismatch { .. }),
        "{err:?}"
    );

    let bad_len = HiddenStateBuffer {
        data: vec![1.0],
        shape: vec![2, 2],
        dtype: DType::F32,
    };
    let err = ReferenceExecutor::import_hidden(&bad_len).unwrap_err();
    assert!(matches!(err, CortexError::StageFailed { .. }), "{err:?}");
}

#[test]
fn reference_rejects_unsupported_stage_before_touching_inputs() {
    let model = tiny_model(1);
    let (mut executor, _) = ReferenceExecutor::from_transformer(&model).expect("bind");
    let stage = AnnStage {
        id: StageId::new("router").expect("id"),
        kind: StageKind::MoeRouter {
            num_experts: 2,
            top_k: 1,
        },
        inputs: vec![],
    };
    let err = ExternalAdapter::execute(&mut executor, &stage, &[]).unwrap_err();
    match err {
        CortexError::UnsupportedStage {
            backend,
            stage_id,
            kind,
        } => {
            assert_eq!(backend, "reference");
            assert_eq!(stage_id, "router");
            assert!(kind.contains("MoeRouter"), "{kind}");
        }
        other => panic!("expected UnsupportedStage, got {other:?}"),
    }
}

#[test]
fn adapter_add_matches_reference_kernel_and_reports_dtype() {
    let model = tiny_model(1);
    let (mut executor, _) = ReferenceExecutor::from_transformer(&model).expect("bind");
    let left = hidden(2, 4, 1.25);
    let right = hidden(2, 4, -0.5);
    let stage = add_stage();
    let out = ExternalAdapter::execute(
        &mut executor,
        &stage,
        &[StageInput::Tensor(&left), StageInput::Tensor(&right)],
    )
    .expect("add");
    let direct = Tensor::try_add(&left, &right).expect("kernel");
    assert_eq!(out.data(), direct.data());
    assert_eq!(out.meta().dtype, DType::F32);
    assert_eq!(out.shape(), &[2, 4]);
}

#[test]
fn run_topology_uses_imported_hidden_state_without_reembedding() {
    let model = tiny_model(1);
    let (executor, topology) = ReferenceExecutor::from_transformer(&model).expect("bind");
    let tokens = [1u32, 3, 0];
    let mut full = executor.clone();
    let outputs = run_topology(
        &mut full,
        &topology,
        &ExternalBindings::new().with_tokens("tokens", &tokens),
    )
    .expect("full");
    let residual_id = StageId::new("blocks.0.residual2").expect("id");
    let exported = ReferenceExecutor::export_hidden(outputs.get(&residual_id).expect("residual"))
        .expect("export");

    let start = StageId::new("final_norm").expect("id");
    let end = StageId::new("readout").expect("id");
    let span = topology.sub_span(&start, &end).expect("span");
    assert_eq!(
        span.external_stage_deps,
        vec![residual_id.clone()],
        "tail must depend on the extracted residual, not on embedding"
    );
    let mut tail_stages = span.stages.to_vec();
    tail_stages[0].inputs = vec![StageSource::External("hidden".into())];
    let tail = AnnTopology::new(tail_stages).expect("tail");
    let imported = ReferenceExecutor::import_hidden(&exported).expect("import");
    let mut tail_exec = executor;
    let tail_out = run_topology(
        &mut tail_exec,
        &tail,
        &ExternalBindings::new().with_tensor("hidden", &imported),
    )
    .expect("tail");
    let readout = StageId::new("readout").expect("id");
    assert_eq!(
        tail_out.get(&readout).expect("readout").data(),
        outputs.get(&readout).expect("full readout").data()
    );
}

#[test]
fn capability_gap_is_a_structured_error_not_a_ranking() {
    let offered = AnnCapabilities {
        backend_name: "probe",
        supported_tags: BTreeSet::from([StageKindTag::Add]),
        supported_dtypes: BTreeSet::from([DType::F32]),
        stateful: false,
    };
    let limitations = BackendLimitations {
        unsupported_tags: BTreeSet::from([StageKindTag::MoeRouter]),
        unsupported_dtypes: BTreeSet::new(),
        unsupported_devices: BTreeSet::from([DeviceClass::Cuda]),
        notes: vec!["dense reference only".into()],
    };
    let err = limitations
        .require_supported(
            "probe",
            &StageId::new("attn").expect("id"),
            &StageKind::MoeRouter {
                num_experts: 2,
                top_k: 1,
            },
            DType::F32,
            DeviceClass::Cpu,
        )
        .unwrap_err();
    match err {
        CortexError::UnsupportedOperation {
            backend,
            stage_id,
            category,
            detail,
        } => {
            assert_eq!(backend, "probe");
            assert_eq!(stage_id.as_deref(), Some("attn"));
            assert_eq!(category, "stage_kind");
            assert!(detail.contains("moe_router"), "{detail}");
        }
        other => panic!("expected UnsupportedOperation, got {other:?}"),
    }
    let dtype_gap = BackendLimitations {
        unsupported_tags: BTreeSet::new(),
        unsupported_dtypes: BTreeSet::from([DType::BF16]),
        unsupported_devices: BTreeSet::new(),
        notes: Vec::new(),
    };
    let dtype_err = dtype_gap
        .require_supported(
            "probe",
            &StageId::new("add").expect("id"),
            &StageKind::Add,
            DType::BF16,
            DeviceClass::Cpu,
        )
        .unwrap_err();
    assert!(matches!(
        dtype_err,
        CortexError::UnsupportedOperation {
            category,
            ..
        } if category == "dtype"
    ));
    let _ = offered;
}

fn _adapter_object_safety() {
    // Native engine types cannot satisfy this bound: it names only cortex types.
    fn assert_adapter<T: cortex_tensor::stage::ExternalAdapterMarker<Tensor = Tensor>>(_: &T) {}
    let model = tiny_model(0);
    let (executor, _) = ReferenceExecutor::from_transformer(&model).expect("bind");
    assert_adapter(&executor);
}

#[test]
fn candle_feature_adapter_is_wired_when_enabled() {
    #[cfg(feature = "candle")]
    {
        use cortex_tensor::stage::CandleAdapter;
        let adapter = CandleAdapter::cpu();
        let caps = ExternalAdapter::capabilities(&adapter);
        assert_eq!(caps.ann.backend_name, "candle");
        assert!(caps.hidden_state_io);
        assert!(caps.devices.contains(&DeviceClass::Cpu));
        let values = vec![1.5f32, -2.0, 0.0, 4.0];
        let buffer = HiddenStateBuffer {
            data: values.clone(),
            shape: vec![2, 2],
            dtype: DType::F32,
        };
        let tensor = adapter.import_hidden(&buffer).expect("import");
        let exported = adapter.export_hidden(&tensor).expect("export");
        assert_eq!(exported.data, values);
        assert_eq!(exported.dtype, DType::F32);
        let bad = HiddenStateBuffer {
            data: values,
            shape: vec![2, 2],
            dtype: DType::BF16,
        };
        let err = adapter.import_hidden(&bad).unwrap_err();
        assert!(matches!(
            err,
            CortexError::UnsupportedOperation { category, .. } if category == "dtype"
        ));
    }
    #[cfg(not(feature = "candle"))]
    {
        // Default builds must not require the Candle feature to exercise the contract.
        let _ = std::any::type_name::<HiddenStateBuffer>();
    }
}

#[test]
fn burn_feature_stays_a_structured_stub_on_this_toolchain() {
    #[cfg(feature = "burn")]
    {
        use cortex_tensor::stage::BurnAdapter;
        let adapter = BurnAdapter::unavailable();
        let caps = ExternalAdapter::capabilities(&adapter);
        assert_eq!(caps.ann.backend_name, "burn");
        assert!(caps.ann.supported_tags.is_empty());
        assert!(
            caps.limitations
                .notes
                .iter()
                .any(|note| note.contains("1.98.1"))
        );
        let stage = add_stage();
        let err = ExternalAdapter::execute(&mut { adapter }, &stage, &[]).unwrap_err();
        assert!(matches!(
            err,
            CortexError::UnsupportedOperation { backend, category, .. }
                if backend == "burn" && category == "toolchain"
        ));
    }
}

#[allow(dead_code)]
fn _result_alias(value: Result<()>) -> Result<()> {
    value
}
