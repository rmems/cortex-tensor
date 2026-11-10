// SPDX-License-Identifier: MIT OR Apache-2.0

//! Interchange metadata and transfer contract tests (RM-1827).
//!
//! These pin the public `interop` surface: tensors describe their shape,
//! dtype, device, and layout without naming a backend type, and `transfer`
//! refuses a hand-off before copying when either side cannot take it.

use std::collections::BTreeSet;

use cortex_tensor::interop::{
    HiddenStateShape, InterchangeMeta, InterchangeTensor, TensorDevice, TensorLayout, transfer,
};
use cortex_tensor::stage::{
    AdapterCapabilities, AnnCapabilities, AnnExecutor, AnnStage, BackendLimitations, DType,
    DeviceClass, ExternalAdapter, ExternalAdapterMarker, HiddenStateBuffer, ReferenceExecutor,
    StageInput, TensorMeta,
};
use cortex_tensor::{CortexError, Result, Tensor};

/// Adapter over reference tensors whose capabilities and import are scripted.
struct Probe {
    dtypes: BTreeSet<DType>,
    hidden_state_io: bool,
    /// Reshape every imported tensor to this shape, simulating a buggy import.
    import_shape: Option<Vec<usize>>,
}

impl Probe {
    fn f32() -> Self {
        Self {
            dtypes: BTreeSet::from([DType::F32]),
            hidden_state_io: true,
            import_shape: None,
        }
    }
}

impl AnnExecutor for Probe {
    type Tensor = Tensor;

    fn capabilities(&self) -> AnnCapabilities {
        AnnCapabilities {
            backend_name: "probe",
            supported_tags: BTreeSet::new(),
            supported_dtypes: self.dtypes.clone(),
            stateful: false,
        }
    }

    fn execute(&mut self, stage: &AnnStage, _inputs: &[StageInput<Tensor>]) -> Result<Tensor> {
        Err(CortexError::UnsupportedStage {
            backend: "probe",
            stage_id: stage.id.to_string(),
            kind: format!("{:?}", stage.kind),
        })
    }
}

impl ExternalAdapterMarker for Probe {
    fn adapter_capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            ann: self.capabilities(),
            devices: BTreeSet::from([DeviceClass::Cpu]),
            batch: false,
            sequence_cache: false,
            hidden_state_io: self.hidden_state_io,
            limitations: BackendLimitations {
                unsupported_tags: BTreeSet::new(),
                unsupported_dtypes: BTreeSet::new(),
                unsupported_devices: BTreeSet::new(),
                notes: Vec::new(),
            },
        }
    }

    fn export_hidden(&self, tensor: &Tensor) -> Result<HiddenStateBuffer> {
        ReferenceExecutor::export_hidden(tensor)
    }

    fn import_hidden(&self, buffer: &HiddenStateBuffer) -> Result<Tensor> {
        let tensor = ReferenceExecutor::import_hidden(buffer)?;
        match &self.import_shape {
            Some(shape) => Tensor::try_from_vec(tensor.data().to_vec(), shape),
            None => Ok(tensor),
        }
    }
}

fn hidden(values: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::try_from_vec(values, shape).expect("tensor")
}

#[test]
fn reference_tensor_reports_host_row_major_f32() {
    let tensor = hidden(vec![0.0; 12], &[3, 4]);
    let meta = tensor.interchange_meta();
    assert_eq!(meta.shape, vec![3, 4]);
    assert_eq!(meta.dtype, DType::F32);
    assert_eq!(meta.device, TensorDevice::HOST);
    assert_eq!(meta.layout, TensorLayout::RowMajor);
    assert_eq!(
        meta.hidden_state().expect("rank 2"),
        HiddenStateShape {
            batch: None,
            seq_len: 3,
            hidden_dim: 4
        }
    );
}

#[test]
fn reference_hidden_state_is_accepted_by_its_own_capabilities() {
    let executor = ReferenceExecutor::new();
    let caps = ExternalAdapter::capabilities(&executor);
    let tensor = hidden(vec![0.0; 8], &[2, 4]);
    tensor
        .interchange_meta()
        .require_hidden_state_supported_by(&caps)
        .expect("reference accepts its own hidden state");

    let batched = hidden(vec![0.0; 8], &[1, 2, 4]).interchange_meta();
    let err = batched
        .require_hidden_state_supported_by(&caps)
        .unwrap_err();
    assert!(
        matches!(
            err,
            CortexError::UnsupportedOperation {
                backend: "reference",
                category: "batch",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn transfer_between_adapters_preserves_bits_and_shape() {
    let values = vec![1.0f32, -0.0, f32::MIN_POSITIVE, 3.5, -7.25, 0.125];
    let tensor = hidden(values.clone(), &[2, 3]);
    let reference = ReferenceExecutor::new();
    let probe = Probe::f32();

    let moved = transfer(&reference, &tensor, &probe).expect("reference -> probe");
    let back = transfer(&probe, &moved, &reference).expect("probe -> reference");
    assert_eq!(back.shape(), &[2, 3]);
    let bits = |data: &[f32]| data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(back.data()), bits(&values));
}

#[test]
fn transfer_refuses_before_copying_when_a_side_lacks_hidden_state_io() {
    let tensor = hidden(vec![0.0; 4], &[2, 2]);
    let closed = Probe {
        hidden_state_io: false,
        ..Probe::f32()
    };
    let reference = ReferenceExecutor::new();

    for err in [
        transfer(&closed, &tensor, &reference).unwrap_err(),
        transfer(&reference, &tensor, &closed).unwrap_err(),
    ] {
        assert!(
            matches!(
                err,
                CortexError::UnsupportedOperation {
                    backend: "probe",
                    stage_id: None,
                    category: "hidden_state_io",
                    ..
                }
            ),
            "{err:?}"
        );
    }
}

#[test]
fn transfer_refuses_a_dtype_the_destination_does_not_support() {
    let tensor = hidden(vec![0.0; 4], &[2, 2]);
    let f16_only = Probe {
        dtypes: BTreeSet::from([DType::F16]),
        ..Probe::f32()
    };
    let err = transfer(&ReferenceExecutor::new(), &tensor, &f16_only).unwrap_err();
    assert!(
        matches!(
            &err,
            CortexError::UnsupportedOperation { category: "dtype", detail, .. } if detail.contains("f32")
        ),
        "{err:?}"
    );
}

#[test]
fn transfer_detects_an_import_that_changes_the_shape() {
    let tensor = hidden(vec![0.0; 6], &[2, 3]);
    let reshaping = Probe {
        import_shape: Some(vec![3, 2]),
        ..Probe::f32()
    };
    let err = transfer(&ReferenceExecutor::new(), &tensor, &reshaping).unwrap_err();
    assert!(
        matches!(
            &err,
            CortexError::ShapeMismatch { expected, got } if expected == &[2, 3] && got == &[3, 2]
        ),
        "{err:?}"
    );
}

#[test]
fn metadata_is_built_from_stage_meta_without_backend_types() {
    let meta = InterchangeMeta::new(
        TensorMeta {
            shape: vec![1, 4, 16],
            dtype: DType::BF16,
        },
        TensorDevice::new(DeviceClass::Cuda, 1),
        TensorLayout::RowMajor,
    );
    assert_eq!(meta.numel().expect("numel"), 64);
    assert_eq!(meta.hidden_state().expect("rank 3").batch, Some(1));
    assert_eq!(meta.device.to_string(), "cuda:1");
    assert_eq!(
        meta.tensor_meta(),
        TensorMeta {
            shape: vec![1, 4, 16],
            dtype: DType::BF16
        }
    );
    let rendered = format!("{meta:?}");
    assert!(!rendered.contains("candle"));
    assert!(!rendered.contains("burn"));
}

#[test]
fn burn_stub_refuses_transfer_without_copying() {
    #[cfg(feature = "burn")]
    {
        use cortex_tensor::stage::BurnAdapter;
        let tensor = hidden(vec![0.0; 4], &[2, 2]);
        let err = transfer(
            &ReferenceExecutor::new(),
            &tensor,
            &BurnAdapter::unavailable(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                CortexError::UnsupportedOperation {
                    backend: "burn",
                    category: "hidden_state_io",
                    ..
                }
            ),
            "{err:?}"
        );

        let host = tensor.interchange_meta();
        let caps = ExternalAdapter::capabilities(&BurnAdapter::unavailable());
        assert!(matches!(
            host.require_supported_by(&caps),
            Err(CortexError::UnsupportedOperation {
                category: "dtype",
                ..
            })
        ));
    }
}

#[cfg(feature = "candle")]
mod candle {
    use super::*;
    use cortex_tensor::adapter::{HiddenState, export_hidden_state, import_hidden_state};

    #[test]
    fn candle_tensor_reports_host_row_major_f32() {
        let state = HiddenState {
            data: vec![0.5; 6],
            shape: vec![2, 3],
        };
        let candle = import_hidden_state(&state).expect("import");
        let meta = candle.interchange_meta();
        assert_eq!(meta, hidden(vec![0.5; 6], &[2, 3]).interchange_meta());
    }

    #[test]
    fn reference_buffer_reaches_candle_and_back_bit_for_bit() {
        let values = vec![1.0f32, -0.0, 2.5, -3.75];
        let tensor = hidden(values.clone(), &[2, 2]);
        let buffer = ReferenceExecutor::export_hidden(&tensor).expect("export");

        let state = HiddenState::try_from(buffer.clone()).expect("f32 buffer");
        let candle = import_hidden_state(&state).expect("candle import");
        let returned: HiddenStateBuffer =
            export_hidden_state(&candle).expect("candle export").into();
        assert_eq!(returned, buffer);

        let restored = ReferenceExecutor::import_hidden(&returned).expect("import");
        assert_eq!(restored.data(), values.as_slice());
    }

    #[test]
    fn non_f32_or_short_buffer_does_not_become_candle_hidden_state() {
        let err = HiddenState::try_from(HiddenStateBuffer {
            data: vec![0.0; 4],
            shape: vec![2, 2],
            dtype: DType::F16,
        })
        .unwrap_err();
        assert!(
            matches!(err, CortexError::StageDTypeMismatch { .. }),
            "{err:?}"
        );

        let err = HiddenState::try_from(HiddenStateBuffer {
            data: vec![0.0; 3],
            shape: vec![2, 2],
            dtype: DType::F32,
        })
        .unwrap_err();
        assert!(matches!(err, CortexError::StageFailed { .. }), "{err:?}");
    }
}
