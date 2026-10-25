// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal shared types used by the `moe` module.
//!
//! Ported from `corinth-canal` with SNN-specific items removed.

use serde::{Deserialize, Serialize};

/// Dimensionality of the dense embedding the projector hands to the router.
pub const EMBEDDING_DIM: usize = 2048;

/// Optional post-processing when extracting a token row from a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExtractTokenOptions {
    /// Resample the dequantized row to this length. `None` keeps checkpoint
    /// `hidden_size` (the default).
    pub target_dim: Option<usize>,
    /// When true, L2-normalize after optional resampling (default `false`).
    pub l2_normalize: bool,
}

impl ExtractTokenOptions {
    /// Resample to [`EMBEDDING_DIM`] and L2-normalize — the legacy projector
    /// path used before RM-1519. Prefer native-length extract for reference
    /// backends; call this only when feeding [`crate::moe::MoeRouter::forward`].
    pub fn for_projector_forward() -> Self {
        Self {
            target_dim: Some(EMBEDDING_DIM),
            l2_normalize: true,
        }
    }
}

/// Execution mode used by the router.
///
/// Spiking routing is not a mode of this enum. The ANN↔SNN step contract
/// (`SpikingNetwork`, `SpikeEncoder`, `SpikeDecoder`, `SpikingExpertRouter`)
/// lives in `hybrid-fusion`. This router stays on dense or stub ANN routing.
/// `DenseSim` is the default.
///
/// Serializes as the canonical slug wire form (`stub_uniform`, `dense_sim`);
/// the legacy PascalCase names remain accepted on deserialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RoutingMode {
    #[serde(alias = "StubUniform")]
    StubUniform,
    #[default]
    #[serde(alias = "DenseSim")]
    DenseSim,
}
