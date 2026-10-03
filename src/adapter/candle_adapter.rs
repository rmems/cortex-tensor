// SPDX-License-Identifier: MIT OR Apache-2.0

//! Candle adapter placeholder.
//!
//! v0.3 ships the adapter slot without a `candle-core` dependency so the core
//! crate remains framework-free. No `candle_core::` type appears in this
//! module. A sibling integration crate can implement the cortex-owned adapter
//! contract without changing core traits.

use super::{
    AdapterCapabilities, BackendLimitations, DeviceClass, ExternalAdapterMarker, HiddenStateBuffer,
    all_stage_tags,
};
use crate::error::{CortexError, Result};
use crate::stage::{
    AnnCapabilities, AnnExecutor, AnnStage, DType, StageInput, StageTensor, TensorMeta,
};
use std::collections::BTreeSet;

/// Tensor stand-in used while Candle lives outside the core dependency graph.
///
/// It never holds Candle storage. Import refuses before one can be built.
#[derive(Debug, Clone, PartialEq)]
pub struct CandleTensorView {
    shape: Vec<usize>,
    dtype: DType,
}

impl StageTensor for CandleTensorView {
    fn meta(&self) -> TensorMeta {
        TensorMeta {
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

/// Candle participation stub.
///
/// `unavailable` is the only constructor. Enabling the feature does not pull
/// the `candle-core` crate.
#[derive(Debug, Clone, Copy)]
pub struct CandleAdapter;

impl CandleAdapter {
    /// Adapter that reports the v0.3 dependency-policy gap.
    pub fn unavailable() -> Self {
        Self
    }

    fn refusal() -> CortexError {
        CortexError::UnsupportedOperation {
            backend: "candle",
            stage_id: None,
            category: "policy",
            detail: "candle is not a dependency of cortex-tensor v0.3; the feature is a structured stub and does not link candle".to_string(),
        }
    }
}

impl ExternalAdapterMarker for CandleAdapter {
    fn adapter_capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            ann: AnnCapabilities {
                backend_name: "candle",
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
                notes: vec![
                    "candle adapter is a structured stub; cortex-tensor v0.3 does not depend on the candle crate".to_string(),
                ],
            },
        }
    }

    fn export_hidden(&self, _tensor: &Self::Tensor) -> Result<HiddenStateBuffer> {
        Err(Self::refusal())
    }

    fn import_hidden(&self, _buffer: &HiddenStateBuffer) -> Result<Self::Tensor> {
        Err(Self::refusal())
    }
}

impl AnnExecutor for CandleAdapter {
    type Tensor = CandleTensorView;

    fn capabilities(&self) -> AnnCapabilities {
        self.adapter_capabilities().ann
    }

    fn execute(
        &mut self,
        stage: &AnnStage,
        _inputs: &[StageInput<Self::Tensor>],
    ) -> Result<Self::Tensor> {
        Err(CortexError::UnsupportedStage {
            backend: "candle",
            stage_id: stage.id.to_string(),
            kind: format!("{:?}", stage.kind),
        })
    }
}
