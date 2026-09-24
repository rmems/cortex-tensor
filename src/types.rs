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
    /// backends; call this only when feeding [`MoeRouter::forward`].
    pub fn for_projector_forward() -> Self {
        Self {
            target_dim: Some(EMBEDDING_DIM),
            l2_normalize: true,
        }
    }
}

/// Supported GGUF model families for the router bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelFamily {
    /// Default family for GGUF MoE checkpoints whose architecture string is not
    /// one of the named vendor families below.
    #[default]
    ReferenceMoe,
    Qwen3Moe,
    Gemma4,
    DeepSeek2,
    LlamaMoe,
}

impl ModelFamily {
    pub fn slug(self) -> &'static str {
        match self {
            Self::ReferenceMoe => "reference_moe",
            Self::Qwen3Moe => "qwen3_moe",
            Self::Gemma4 => "gemma4",
            Self::DeepSeek2 => "deepseek2",
            Self::LlamaMoe => "llama_moe",
        }
    }
}

/// Execution mode used by the router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RoutingMode {
    StubUniform,
    DenseSim,
    #[default]
    SpikingSim,
}
