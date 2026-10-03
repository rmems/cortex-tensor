// SPDX-License-Identifier: MIT OR Apache-2.0

//! External Rust ML adapter contract.
//!
//! Core never names Candle, Burn, or any other engine's tensor, device, or
//! module types. An adapter implements [`ExternalAdapter`] with its own
//! associated tensor, reports capabilities in cortex terms, and converts
//! hidden state through [`HiddenStateBuffer`]. Optional engine crates live
//! behind features and are not required to use [`crate::stage::ReferenceExecutor`].
//!
//! Capability reports are facts an orchestrator can translate. This module
//! does not select a backend, rank engines, or place ANN stages against SNN
//! stages — that negotiation belongs to `hybrid-fusion` (see
//! [hybrid-fusion#41](https://github.com/rmems/hybrid-fusion/issues/41)).
//! [`NegotiationDocument`] is the cortex-owned document that translation
//! consumes. It is not `hybrid-fusion`'s `BackendCapabilities` type, which
//! has not landed.

use crate::error::{CortexError, Result};
use crate::stage::{
    AnnCapabilities, AnnExecutor, AnnStage, DType, StageId, StageInput, StageKind, StageKindTag,
    StageTensor,
};
use std::collections::BTreeSet;
use std::fmt;

/// Where an external adapter may place tensors.
///
/// A capability fact, not a device handle. Native engine device types stay
/// inside optional adapters.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceClass {
    /// Host CPU.
    Cpu,
    /// NVIDIA CUDA device. Reported only; this crate does not ship CUDA kernels.
    Cuda,
    /// Apple Metal device. Reported only; this crate does not ship Metal kernels.
    Metal,
}

impl DeviceClass {
    /// Stable negotiation token.
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceClass::Cpu => "cpu",
            DeviceClass::Cuda => "cuda",
            DeviceClass::Metal => "metal",
        }
    }
}

impl fmt::Display for DeviceClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What an adapter explicitly cannot do.
///
/// Absence from [`AnnCapabilities`] already means "not advertised." This
/// struct records the refusals an orchestrator should surface, including
/// toolchain gaps that are not a stage kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendLimitations {
    /// Stage kinds the adapter will reject.
    pub unsupported_tags: BTreeSet<StageKindTag>,
    /// Element types the adapter will reject.
    pub unsupported_dtypes: BTreeSet<DType>,
    /// Device classes the adapter will reject.
    pub unsupported_devices: BTreeSet<DeviceClass>,
    /// Human-readable limits. Not a performance ranking.
    pub notes: Vec<String>,
}

impl BackendLimitations {
    /// Refuse the first capability gap, in category order: stage kind, dtype, device.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::UnsupportedOperation`] for the first gap. A
    /// supported request returns `Ok(())`.
    pub fn require_supported(
        &self,
        backend: &'static str,
        stage_id: &StageId,
        kind: &StageKind,
        dtype: DType,
        device: DeviceClass,
    ) -> Result<()> {
        let tag = kind.tag();
        if self.unsupported_tags.contains(&tag) {
            return Err(CortexError::UnsupportedOperation {
                backend,
                stage_id: Some(stage_id.to_string()),
                category: "stage_kind",
                detail: format!("{tag} is not supported"),
            });
        }
        if self.unsupported_dtypes.contains(&dtype) {
            return Err(CortexError::UnsupportedOperation {
                backend,
                stage_id: Some(stage_id.to_string()),
                category: "dtype",
                detail: format!("{dtype} is not supported"),
            });
        }
        if self.unsupported_devices.contains(&device) {
            return Err(CortexError::UnsupportedOperation {
                backend,
                stage_id: Some(stage_id.to_string()),
                category: "device",
                detail: format!("{device} is not supported"),
            });
        }
        Ok(())
    }
}

/// Owned hidden-state bytes crossing an adapter boundary.
///
/// This is the conversion boundary, not a universal tensor format and not a
/// checkpoint. Adapters copy into and out of their native storage here so
/// core traits never name that storage.
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenStateBuffer {
    /// Row-major elements in the reference interchange dtype of `dtype`.
    /// `F32` payloads are IEEE-754 `f32` bits stored as `f32` values.
    pub data: Vec<f32>,
    /// Row-major extents, outermost dimension first.
    pub shape: Vec<usize>,
    /// Element type the producing adapter claims.
    pub dtype: DType,
}

/// Capability report every external adapter and the reference backend share.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterCapabilities {
    /// Stage kinds, dtypes, and statefulness, in the existing ANN vocabulary.
    pub ann: AnnCapabilities,
    /// Device classes the adapter can place tensors on.
    pub devices: BTreeSet<DeviceClass>,
    /// Whether a leading batch axis is accepted. The reference backend is unbatched.
    pub batch: bool,
    /// Whether the adapter keeps a KV / sequence cache across calls.
    pub sequence_cache: bool,
    /// Whether hidden state can be imported and exported through [`HiddenStateBuffer`].
    pub hidden_state_io: bool,
    /// Explicit refusals, including kinds absent from [`AnnCapabilities`].
    pub limitations: BackendLimitations,
}

/// Cortex-owned facts an orchestrator translates into its own negotiation type.
///
/// Field names here are stable for that translation. They are not
/// `hybrid-fusion`'s future `BackendCapabilities` struct: that type has not
/// landed (hybrid-fusion#41), and this crate does not depend on `hybrid-fusion`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegotiationDocument {
    /// [`AnnCapabilities::backend_name`].
    pub backend_name: &'static str,
    /// Supported [`StageKindTag`] tokens, sorted.
    pub stage_kinds: Vec<String>,
    /// Supported [`DType`] tokens, sorted.
    pub dtypes: Vec<String>,
    /// Supported [`DeviceClass`] tokens, sorted.
    pub devices: Vec<String>,
    /// Copied from [`AdapterCapabilities::batch`].
    pub batch: bool,
    /// Copied from [`AdapterCapabilities::sequence_cache`].
    pub sequence_cache: bool,
    /// Copied from [`AnnCapabilities::stateful`].
    pub stateful: bool,
    /// Copied from [`AdapterCapabilities::hidden_state_io`].
    pub hidden_state_io: bool,
    /// Unsupported stage-kind tokens, sorted.
    ///
    /// Prefixed `stage_kind:` so a caller can tell a refused stage from a
    /// refused dtype or device without parsing notes.
    pub unsupported: Vec<String>,
    /// Unsupported dtype tokens, sorted. Prefixed `dtype:`.
    pub unsupported_dtypes: Vec<String>,
    /// Unsupported device tokens, sorted. Prefixed `device:`.
    pub unsupported_devices: Vec<String>,
    /// Limitation notes, including the orchestrator boundary.
    pub limitations: Vec<String>,
}

impl AdapterCapabilities {
    /// Project this report into the document hybrid-fusion negotiation consumes.
    ///
    /// The projection is side-effect free. It does not choose a backend or
    /// fill a fallback.
    pub fn negotiation_document(&self) -> NegotiationDocument {
        let mut stage_kinds: Vec<String> = self
            .ann
            .supported_tags
            .iter()
            .map(|tag| tag.as_str().to_string())
            .collect();
        stage_kinds.sort();
        let mut dtypes: Vec<String> = self
            .ann
            .supported_dtypes
            .iter()
            .map(|dtype| dtype.as_str().to_string())
            .collect();
        dtypes.sort();
        let mut devices: Vec<String> = self
            .devices
            .iter()
            .map(|device| device.as_str().to_string())
            .collect();
        devices.sort();
        let mut unsupported: Vec<String> = self
            .limitations
            .unsupported_tags
            .iter()
            .map(|tag| format!("stage_kind:{}", tag.as_str()))
            .collect();
        unsupported.sort();
        let mut unsupported_dtypes: Vec<String> = self
            .limitations
            .unsupported_dtypes
            .iter()
            .map(|dtype| format!("dtype:{}", dtype.as_str()))
            .collect();
        unsupported_dtypes.sort();
        let mut unsupported_devices: Vec<String> = self
            .limitations
            .unsupported_devices
            .iter()
            .map(|device| format!("device:{}", device.as_str()))
            .collect();
        unsupported_devices.sort();
        let mut limitations = self.limitations.notes.clone();
        limitations.push(
            "translation target is hybrid-fusion negotiation (hybrid-fusion#41); this document does not place or rank backends".to_string(),
        );
        NegotiationDocument {
            backend_name: self.ann.backend_name,
            stage_kinds,
            dtypes,
            devices,
            batch: self.batch,
            sequence_cache: self.sequence_cache,
            stateful: self.ann.stateful,
            hidden_state_io: self.hidden_state_io,
            unsupported,
            unsupported_dtypes,
            unsupported_devices,
            limitations,
        }
    }
}

/// Marker for a backend that can be used through the external-adapter contract.
///
/// Generic code bounded by this trait can query capabilities, execute stages,
/// and move hidden state through [`HiddenStateBuffer`] without naming Candle
/// or Burn. Native engine tensors stay the backend's associated [`StageTensor`].
/// Inherent methods on each backend remain so a recording wrapper can call
/// conversion without implementing this trait itself.
pub trait ExternalAdapterMarker: AnnExecutor {
    /// Report the adapter capability surface. Must agree with
    /// [`AnnExecutor::capabilities`] on name, tags, dtypes, and statefulness.
    fn adapter_capabilities(&self) -> AdapterCapabilities;

    /// Copy `tensor` into the interchange buffer.
    ///
    /// # Errors
    ///
    /// Returns a [`CortexError`] when the adapter cannot export `tensor`.
    fn export_hidden(&self, tensor: &Self::Tensor) -> Result<HiddenStateBuffer>;

    /// Copy `buffer` into the backend's tensor type.
    ///
    /// # Errors
    ///
    /// Returns a [`CortexError`] when the adapter cannot import `buffer`.
    fn import_hidden(&self, buffer: &HiddenStateBuffer) -> Result<Self::Tensor>;
}

/// Checked element count for a hidden-state shape.
pub(crate) fn checked_numel(shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| CortexError::StageFailed {
                stage_id: "hidden_state".to_string(),
                source: Box::new(CortexError::ShapeMismatch {
                    expected: shape.to_vec(),
                    got: vec![dim],
                }),
            })
    })
}

/// Build the reference backend's adapter capability report from its ANN report.
pub(crate) fn reference_adapter_capabilities(ann: AnnCapabilities) -> AdapterCapabilities {
    let unsupported_tags = all_stage_tags()
        .into_iter()
        .filter(|tag| !ann.supported_tags.contains(tag))
        .collect();
    AdapterCapabilities {
        ann,
        devices: BTreeSet::from([DeviceClass::Cpu]),
        batch: false,
        sequence_cache: false,
        hidden_state_io: true,
        limitations: BackendLimitations {
            unsupported_tags,
            unsupported_dtypes: BTreeSet::from([DType::F16, DType::BF16]),
            unsupported_devices: BTreeSet::from([DeviceClass::Cuda, DeviceClass::Metal]),
            notes: vec![
                "reference backend is f32, CPU-only, and dense; no batch axis and no sequence cache".to_string(),
            ],
        },
    }
}

fn all_stage_tags() -> [StageKindTag; 11] {
    [
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
    ]
}

#[cfg(any(feature = "candle", feature = "burn"))]
fn policy_stub_capabilities(
    backend_name: &'static str,
    framework_name: &'static str,
) -> AdapterCapabilities {
    AdapterCapabilities {
        ann: AnnCapabilities {
            backend_name,
            supported_tags: BTreeSet::new(),
            supported_dtypes: BTreeSet::new(),
            stateful: false,
        },
        devices: BTreeSet::new(),
        batch: false,
        sequence_cache: false,
        hidden_state_io: false,
        limitations: BackendLimitations {
            unsupported_tags: BTreeSet::from(all_stage_tags()),
            unsupported_dtypes: BTreeSet::from([DType::F32, DType::F16, DType::BF16]),
            unsupported_devices: BTreeSet::from([
                DeviceClass::Cpu,
                DeviceClass::Cuda,
                DeviceClass::Metal,
            ]),
            notes: vec![format!(
                "{backend_name} adapter is a structured stub; cortex-tensor v0.3 does not depend on the {framework_name} crate"
            )],
        },
    }
}

#[cfg(any(feature = "candle", feature = "burn"))]
fn policy_stub_refusal(backend_name: &'static str, framework_name: &'static str) -> CortexError {
    CortexError::UnsupportedOperation {
        backend: backend_name,
        stage_id: None,
        category: "policy",
        detail: format!(
            "{framework_name} is not a dependency of cortex-tensor v0.3; the feature is a structured stub and does not link {framework_name}"
        ),
    }
}

#[cfg(any(feature = "candle", feature = "burn"))]
macro_rules! define_policy_stub_adapter {
    ($adapter:ident, $tensor:ident, $backend:literal, $framework:literal) => {
        #[doc = concat!("Tensor stand-in used while ", $framework, " stays outside core.")]
        #[derive(Debug, Clone, PartialEq)]
        pub struct $tensor {
            shape: Vec<usize>,
            dtype: $crate::stage::DType,
        }

        impl $crate::stage::StageTensor for $tensor {
            fn meta(&self) -> $crate::stage::TensorMeta {
                $crate::stage::TensorMeta {
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                }
            }
        }

        #[doc = concat!($framework, " participation stub.")]
        #[derive(Debug, Clone, Copy)]
        pub struct $adapter;

        impl $adapter {
            /// Adapter that reports the v0.3 dependency-policy gap.
            pub fn unavailable() -> Self {
                Self
            }
        }

        impl $crate::adapter::ExternalAdapterMarker for $adapter {
            fn adapter_capabilities(&self) -> $crate::adapter::AdapterCapabilities {
                $crate::adapter::policy_stub_capabilities($backend, $framework)
            }

            fn export_hidden(
                &self,
                _tensor: &Self::Tensor,
            ) -> $crate::Result<$crate::adapter::HiddenStateBuffer> {
                Err($crate::adapter::policy_stub_refusal($backend, $framework))
            }

            fn import_hidden(
                &self,
                _buffer: &$crate::adapter::HiddenStateBuffer,
            ) -> $crate::Result<Self::Tensor> {
                Err($crate::adapter::policy_stub_refusal($backend, $framework))
            }
        }

        impl $crate::stage::AnnExecutor for $adapter {
            type Tensor = $tensor;

            fn capabilities(&self) -> $crate::stage::AnnCapabilities {
                $crate::adapter::ExternalAdapterMarker::adapter_capabilities(self).ann
            }

            fn execute(
                &mut self,
                stage: &$crate::stage::AnnStage,
                _inputs: &[$crate::stage::StageInput<Self::Tensor>],
            ) -> $crate::Result<Self::Tensor> {
                Err($crate::CortexError::UnsupportedStage {
                    backend: $backend,
                    stage_id: stage.id.to_string(),
                    kind: format!("{:?}", stage.kind),
                })
            }
        }
    };
}

#[cfg(any(feature = "candle", feature = "burn"))]
pub(crate) use define_policy_stub_adapter;

/// Reject a hidden-state dtype the reference path cannot store.
pub(crate) fn require_f32_buffer(buffer: &HiddenStateBuffer) -> Result<()> {
    if buffer.dtype != DType::F32 {
        return Err(CortexError::StageDTypeMismatch {
            stage_id: "hidden_state".to_string(),
            expected: DType::F32.to_string(),
            got: buffer.dtype.to_string(),
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
    Ok(())
}

/// Namespace for generic calls against [`ExternalAdapterMarker`].
///
/// `ReferenceExecutor`, [`CandleAdapter`], and [`BurnAdapter`] also expose the
/// same methods inherently. This helper lets tests and generic callers write
/// `ExternalAdapter::capabilities(backend)` without naming an engine.
pub struct ExternalAdapter;

impl ExternalAdapter {
    /// [`AdapterCapabilities`] for any [`ExternalAdapterMarker`].
    pub fn capabilities<A: ExternalAdapterMarker + ?Sized>(adapter: &A) -> AdapterCapabilities {
        adapter.adapter_capabilities()
    }

    /// [`AnnExecutor::execute`] under the adapter contract name.
    pub fn execute<A: ExternalAdapterMarker + ?Sized>(
        adapter: &mut A,
        stage: &AnnStage,
        inputs: &[StageInput<A::Tensor>],
    ) -> Result<A::Tensor> {
        adapter.execute(stage, inputs)
    }

    /// [`ExternalAdapterMarker::export_hidden`] for a generic adapter.
    pub fn export_hidden<A: ExternalAdapterMarker + ?Sized>(
        adapter: &A,
        tensor: &A::Tensor,
    ) -> Result<HiddenStateBuffer> {
        adapter.export_hidden(tensor)
    }

    /// [`ExternalAdapterMarker::import_hidden`] for a generic adapter.
    pub fn import_hidden<A: ExternalAdapterMarker + ?Sized>(
        adapter: &A,
        buffer: &HiddenStateBuffer,
    ) -> Result<A::Tensor> {
        adapter.import_hidden(buffer)
    }
}

/// Confirm an adapter tensor's reported dtype is one it claims to support.
pub fn require_supported_dtype<T: StageTensor>(
    caps: &AnnCapabilities,
    stage_id: &StageId,
    tensor: &T,
) -> Result<()> {
    let dtype = tensor.meta().dtype;
    if caps.supported_dtypes.contains(&dtype) {
        Ok(())
    } else {
        Err(CortexError::StageDTypeMismatch {
            stage_id: stage_id.to_string(),
            expected: caps
                .supported_dtypes
                .iter()
                .map(|dtype| dtype.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            got: dtype.to_string(),
        })
    }
}

#[cfg(feature = "candle")]
mod candle_adapter;
#[cfg(feature = "candle")]
pub use candle_adapter::CandleAdapter;

#[cfg(feature = "burn")]
mod burn_adapter;
#[cfg(feature = "burn")]
pub use burn_adapter::BurnAdapter;
