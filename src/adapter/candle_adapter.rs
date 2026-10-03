// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional Candle adapter.
//!
//! `candle_core::Tensor` and `candle_core::Device` stay in this module. The
//! public stage contract sees only [`crate::adapter::HiddenStateBuffer`] and
//! [`crate::adapter::AdapterCapabilities`]. This adapter copies CPU `f32`
//! hidden state. It does not bind stage kernels, so it reports no executable
//! stage kinds.

use super::{
    AdapterCapabilities, BackendLimitations, DeviceClass, ExternalAdapterMarker, HiddenStateBuffer,
    checked_numel,
};
use crate::error::{CortexError, Result};
use crate::stage::{
    AnnCapabilities, AnnExecutor, AnnStage, DType, StageInput, StageKindTag, StageTensor,
    TensorMeta,
};
use candle_core::{DType as CandleDType, Device, Tensor as CandleTensor};
use std::collections::BTreeSet;

/// Candle CPU tensor viewed through the stage boundary.
#[derive(Debug, Clone)]
pub struct CandleTensorView {
    inner: CandleTensor,
}

impl StageTensor for CandleTensorView {
    fn meta(&self) -> TensorMeta {
        TensorMeta {
            shape: self.inner.dims().to_vec(),
            dtype: match self.inner.dtype() {
                CandleDType::F32 => DType::F32,
                CandleDType::F16 => DType::F16,
                CandleDType::BF16 => DType::BF16,
                _ => DType::F32,
            },
        }
    }
}

/// Candle participation in the external-adapter contract.
///
/// Constructed for CPU `f32` only. CUDA and Metal are reported as unsupported
/// rather than compiled in: this crate does not ship those kernels.
#[derive(Debug, Clone)]
pub struct CandleAdapter {
    device: Device,
}

impl CandleAdapter {
    /// CPU adapter. Hidden-state import and export use Candle's CPU `f32` storage.
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    /// Copy `tensor` out to the interchange buffer.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::UnsupportedOperation`] when `tensor` is not `f32`
    /// on CPU.
    pub fn export_hidden(&self, tensor: &CandleTensorView) -> Result<HiddenStateBuffer> {
        self.require_native(tensor)?;
        let flat = tensor
            .inner
            .flatten_all()
            .and_then(|flat| flat.to_vec1::<f32>())
            .map_err(|err| CortexError::StageFailed {
                stage_id: "hidden_state".to_string(),
                source: Box::new(CortexError::Msg(err.to_string())),
            })?;
        Ok(HiddenStateBuffer {
            data: flat,
            shape: tensor.inner.dims().to_vec(),
            dtype: DType::F32,
        })
    }

    /// Copy `buffer` into a Candle CPU `f32` tensor.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::UnsupportedOperation`] for a non-`f32` buffer and
    /// [`CortexError::StageFailed`] when the element count does not match the shape.
    pub fn import_hidden(&self, buffer: &HiddenStateBuffer) -> Result<CandleTensorView> {
        if buffer.dtype != DType::F32 {
            return Err(CortexError::UnsupportedOperation {
                backend: "candle",
                stage_id: None,
                category: "dtype",
                detail: format!(
                    "{} is not imported by the CPU f32 Candle adapter",
                    buffer.dtype
                ),
            });
        }
        let numel = checked_numel(&buffer.shape)?;
        if buffer.data.len() != numel {
            return Err(CortexError::StageFailed {
                stage_id: "hidden_state".to_string(),
                source: Box::new(CortexError::ShapeMismatch {
                    expected: buffer.shape.clone(),
                    got: vec![buffer.data.len()],
                }),
            });
        }
        let inner =
            CandleTensor::from_vec(buffer.data.clone(), buffer.shape.as_slice(), &self.device)
                .map_err(|err| CortexError::StageFailed {
                    stage_id: "hidden_state".to_string(),
                    source: Box::new(CortexError::Msg(err.to_string())),
                })?;
        Ok(CandleTensorView { inner })
    }

    fn require_native(&self, tensor: &CandleTensorView) -> Result<()> {
        if tensor.inner.dtype() != CandleDType::F32 || !tensor.inner.device().is_cpu() {
            return Err(CortexError::UnsupportedOperation {
                backend: "candle",
                stage_id: None,
                category: "dtype",
                detail: "Candle adapter exports CPU f32 tensors only".to_string(),
            });
        }
        Ok(())
    }

    fn capabilities_value() -> AnnCapabilities {
        AnnCapabilities {
            backend_name: "candle",
            // `execute` refuses every stage. Do not advertise kinds that are not bound.
            supported_tags: BTreeSet::new(),
            supported_dtypes: BTreeSet::from([DType::F32]),
            stateful: false,
        }
    }
}

impl ExternalAdapterMarker for CandleAdapter {
    fn adapter_capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            ann: Self::capabilities_value(),
            devices: BTreeSet::from([DeviceClass::Cpu]),
            batch: false,
            sequence_cache: false,
            hidden_state_io: true,
            limitations: BackendLimitations {
                unsupported_tags: BTreeSet::from([
                    StageKindTag::Embedding,
                    StageKindTag::Attention,
                    StageKindTag::LayerNorm,
                    StageKindTag::RmsNorm,
                    StageKindTag::DenseMlp,
                    StageKindTag::GatedMlp,
                    StageKindTag::MoeRouter,
                    StageKindTag::MoeExpert,
                    StageKindTag::Add,
                    StageKindTag::Readout,
                    StageKindTag::Custom,
                ]),
                unsupported_dtypes: BTreeSet::from([DType::F16, DType::BF16]),
                unsupported_devices: BTreeSet::from([DeviceClass::Cuda, DeviceClass::Metal]),
                notes: vec![
                    "candle adapter is CPU f32 hidden-state I/O; stage kernels are not bound"
                        .to_string(),
                ],
            },
        }
    }

    fn export_hidden(&self, tensor: &Self::Tensor) -> Result<HiddenStateBuffer> {
        CandleAdapter::export_hidden(self, tensor)
    }

    fn import_hidden(&self, buffer: &HiddenStateBuffer) -> Result<Self::Tensor> {
        CandleAdapter::import_hidden(self, buffer)
    }
}

impl AnnExecutor for CandleAdapter {
    type Tensor = CandleTensorView;

    fn capabilities(&self) -> AnnCapabilities {
        Self::capabilities_value()
    }

    fn execute(
        &mut self,
        stage: &AnnStage,
        _inputs: &[StageInput<Self::Tensor>],
    ) -> Result<Self::Tensor> {
        // Stage kernels stay in the engine. This adapter refuses execution
        // that would require reimplementing them, and it does so before
        // looking at inputs.
        Err(CortexError::UnsupportedStage {
            backend: "candle",
            stage_id: stage.id.to_string(),
            kind: format!("{:?}", stage.kind),
        })
    }
}
