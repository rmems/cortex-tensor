// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal shared types used by the `moe` module.
//!
//! Ported from `corinth-canal` with SNN-specific items removed.

use serde::{Deserialize, Serialize};

/// Dimensionality of the dense embedding the projector hands to the router.
pub const EMBEDDING_DIM: usize = 2048;

/// Largest `num_experts` accepted by a synthetic stub router.
///
/// [`crate::moe::MoeRouter::load_with_mode`] sizes `expert_weights` and the
/// synthetic gate-score vector from the caller's `num_experts` when the model
/// path is empty, so an unbounded value is a capacity-overflow panic
/// (`Vec::with_capacity(usize::MAX)`) or an allocator abort. No real MoE
/// router places more experts than the embedding width it routes over, so
/// [`EMBEDDING_DIM`] is the ceiling for the stub.
///
/// This ceiling applies only to the stub path. A checkpoint-backed router
/// rejects a requested count above the checkpoint's `expert_count` with
/// [`CortexError::InvalidConfig`](crate::error::CortexError::InvalidConfig)
/// instead, because that value is bounded by a real tensor dimension.
pub const MAX_STUB_EXPERTS: usize = EMBEDDING_DIM;

/// Largest `target_dim` accepted by
/// [`crate::moe::MoeRouter::extract_token_embedding_with_options`].
///
/// Resampling allocates `target_dim` values, so the caller-supplied length is
/// bounded the same way [`MAX_STUB_EXPERTS`] is. 64 × [`EMBEDDING_DIM`] leaves
/// ample room for up-projection (the legacy projector path is 1 ×) while
/// keeping the largest accepted request at 512 KiB of `f32`.
pub const MAX_TARGET_DIM: usize = 64 * EMBEDDING_DIM;

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
/// Spiking routing is no longer a mode of this enum: it is provided by
/// `snn::SpikingMoeRouter`, which composes an `MoeRouter` with an external
/// [`crate::snn::SnnBackend`] (e.g. `neuromod`). `DenseSim` is the default
/// for backward-compatible default routing semantics.
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
