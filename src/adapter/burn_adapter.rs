// SPDX-License-Identifier: MIT OR Apache-2.0

//! Burn adapter placeholder for the current toolchain.
//!
//! Published `burn` 0.21.0 declares `rust-version = "1.92"`. This crate's
//! pinned toolchain is 1.98.1, so depending on `burn` would make
//! `--all-features` fail CI. The adapter type is therefore compiled only when
//! the `burn` feature is enabled explicitly, and every operation returns a
//! structured toolchain refusal. No `burn::` type appears in this module.
//!
//! When the crate MSRV moves to a Burn-compatible toolchain, replace
//! [`BurnAdapter::unavailable`] with a CPU `ndarray` backend that implements
//! the same [`ExternalAdapterMarker`] methods. That change stays inside this
//! module: [`crate::stage::AnnExecutor`] does not gain Burn types.

use super::{AdapterCapabilities, BackendLimitations, DeviceClass, ExternalAdapterMarker};
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
/// `unavailable` is the only constructor on this toolchain. Enabling the
/// feature does not pull the `burn` crate.
#[derive(Debug, Clone, Copy)]
pub struct BurnAdapter;

impl BurnAdapter {
    /// Adapter that reports the toolchain gap instead of linking Burn.
    pub fn unavailable() -> Self {
        Self
    }

    fn refusal(stage_id: Option<String>) -> CortexError {
        CortexError::UnsupportedOperation {
            backend: "burn",
            stage_id,
            category: "toolchain",
            detail: "burn 0.21 requires rustc 1.92; this crate is pinned to 1.98.1, so the Burn adapter is not linked".to_string(),
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
                    "burn adapter is a structured stub on rustc 1.98.1; burn 0.21 requires rustc 1.92 and is not a core dependency".to_string(),
                ],
            },
        }
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
