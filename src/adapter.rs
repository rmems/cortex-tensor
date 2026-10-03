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
    pub unsupported: Vec<String>,
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
            .map(|tag| tag.as_str().to_string())
            .collect();
        unsupported.sort();
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
            limitations,
        }
    }
}

/// Marker for a backend that can be used through the external-adapter contract.
///
/// The methods callers use are inherent on each backend (`capabilities`,
/// `execute`, `import_hidden`, `export_hidden`) so a recording test double can
/// wrap [`crate::stage::ReferenceExecutor`] without reimplementing conversion.
/// This trait exists so generic code can name "an external adapter" without
/// naming Candle or Burn. Native engine tensors are not valid `Tensor`
/// parameters of this trait: only the backend's associated stage tensor is.
pub trait ExternalAdapterMarker: AnnExecutor {
    /// Report the adapter capability surface. Must agree with
    /// [`AnnExecutor::capabilities`] on name, tags, dtypes, and statefulness.
    fn adapter_capabilities(&self) -> AdapterCapabilities;
}

/// Checked element count for a hidden-state shape.
pub(crate) fn checked_numel(shape: &[usize]) -> Result<usize> {
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

/// Inherent-method namespace matching the call shape used by adapters.
///
/// `ReferenceExecutor`, [`CandleAdapter`], and [`BurnAdapter`] expose the same
/// methods directly. This helper lets tests and generic callers write
/// `ExternalAdapter::capabilities(backend)` without a separate trait method
/// that would force every recording wrapper to reimplement conversion.
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
