// SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CortexError {
    // ── Tensor / math errors ──────────────────────────────────────────────
    #[error("tensor shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("dimension mismatch for matmul: [{m}×{k1}] × [{k2}×{n}]")]
    MatmulDim {
        m: usize,
        k1: usize,
        k2: usize,
        n: usize,
    },

    #[error("index {index} out of bounds for axis {axis} with size {size}")]
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        size: usize,
    },

    #[error("invalid epsilon {eps}: must be finite and greater than 0")]
    InvalidEpsilon { eps: f32 },

    #[error("{op}: cannot normalize a zero-width axis")]
    ZeroWidthAxis { op: &'static str },

    // ── Configuration errors ──────────────────────────────────────────────
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    // ── Model loading errors ──────────────────────────────────────────────
    #[error("model load failed for '{path}': {reason}")]
    ModelLoad { path: String, reason: String },

    #[error("unsupported model format: {0}")]
    UnsupportedFormat(String),

    #[error("missing tensor '{name}' in model '{path}'")]
    MissingTensor { name: String, path: String },

    // ── Forward-pass errors ───────────────────────────────────────────────
    #[error("input length mismatch: expected {expected}, got {got}")]
    InputLengthMismatch { expected: usize, got: usize },

    #[error("router forward pass failed: {0}")]
    OlmoeForward(String),

    // ── I/O / serde ───────────────────────────────────────────────────────
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("{0}")]
    Msg(String),
}

/// Alias retained so code ported from `corinth-canal` compiles unchanged.
pub type HybridError = CortexError;

pub type Result<T> = std::result::Result<T, CortexError>;
