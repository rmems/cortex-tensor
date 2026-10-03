// SPDX-License-Identifier: MIT OR Apache-2.0

//! Burn adapter placeholder.
//!
//! v0.3 ships the adapter slot without a `burn` crate dependency. Published
//! `burn` 0.21.0 declares `rust-version = "1.92"`, which is a minimum and is
//! compatible with this crate's 1.98.1 pin. The feature stays a stub by
//! repository policy so core does not take a foundational ML-framework
//! dependency in this milestone. No `burn::` type appears in this module.
//!
//! A later milestone can replace [`BurnAdapter::unavailable`] with a CPU
//! backend inside this module. [`crate::stage::AnnExecutor`] does not gain
//! Burn types.

use super::{
    AdapterCapabilities, BackendLimitations, DeviceClass, ExternalAdapterMarker, HiddenStateBuffer,
};
use crate::error::{CortexError, Result};
use crate::stage::{
    AnnCapabilities, AnnExecutor, AnnStage, DType, StageInput, StageTensor, TensorMeta,
};
use std::collections::BTreeSet;

/// Tensor stand-in used while Burn cannot be linked.
///
/// It never holds Burn storage. Import refuses before one can be built.
#[derive(Debug, Clone, PartialEq)]
pub struct BurnTensorView {
    shape: Vec<usize>,
    dtype: DType,
}

impl StageTensor for BurnTensorView {
    fn meta(&self) -> TensorMeta {
        TensorMeta {
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

/// Burn participation stub.
///
/// `unavailable` is the only constructor. Enabling the feature does not pull
/// the `burn` crate.
#[derive(Debug, Clone, Copy)]
pub struct BurnAdapter;

impl BurnAdapter {
    /// Adapter that reports the v0.3 policy gap instead of linking Burn.
    pub fn unavailable() -> Self {
        Self
    }

    fn refusal(stage_id: Option<String>) -> CortexError {
        CortexError::UnsupportedOperation {
            backend: "burn",
            stage_id,
            category: "policy",
            detail: "burn is not a dependency of cortex-tensor v0.3; the feature is a structured stub and does not link burn".to_string(),
        }
    }
}

impl ExternalAdapterMarker for BurnAdapter {
    fn adapter_capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            ann: AnnCapabilities {
                backend_name: "burn",
                supported_tags: BTreeSet::new(),
                supported_dtypes: BTreeSet::new(),
                stateful: false,
            },
            devices: BTreeSet::new(),
            batch: false,
            sequence_cache: false,
            hidden_state_io: false,
            limitations: BackendLimitations {
                unsupported_tags: BTreeSet::new(),
                unsupported_dtypes: BTreeSet::from([DType::F32, DType::F16, DType::BF16]),
                unsupported_devices: BTreeSet::from([
                    DeviceClass::Cpu,
                    DeviceClass::Cuda,
                    DeviceClass::Metal,
                ]),
                notes: vec![
                    "burn adapter is a structured stub; cortex-tensor v0.3 does not depend on the burn crate".to_string(),
                ],
            },
        }
    }

    fn export_hidden(&self, _tensor: &Self::Tensor) -> Result<HiddenStateBuffer> {
        Err(Self::refusal(None))
    }

    fn import_hidden(&self, _buffer: &HiddenStateBuffer) -> Result<Self::Tensor> {
        Err(Self::refusal(None))
    }
}

impl AnnExecutor for BurnAdapter {
    type Tensor = BurnTensorView;

    fn capabilities(&self) -> AnnCapabilities {
        self.adapter_capabilities().ann
    }

    fn execute(
        &mut self,
        stage: &AnnStage,
        _inputs: &[StageInput<Self::Tensor>],
    ) -> Result<Self::Tensor> {
        Err(Self::refusal(Some(stage.id.to_string())))
    }
}
