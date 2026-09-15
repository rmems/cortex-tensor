// SPDX-License-Identifier: MIT OR Apache-2.0

//! # cortex-tensor
//!
//! Pure-Rust tensor + transformer + MoE building blocks. Zero GPU / CUDA /
//! Julia / framework dependencies.
//!
//! ## Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`tensor`] | `Tensor` type + core ops + finite-value policy |
//! | [`transformer`] | Transformer building blocks (attention, block, model) |
//! | [`moe`] | Mixture-of-Experts router + GGUF checkpoint bridge |
//! | [`types`] | Shared types used by `moe` |
//! | [`error`] | `CortexError` unified error type |

pub mod error;
pub mod moe;
pub mod tensor;
pub mod transformer;
pub mod types;

pub use error::{CortexError, HybridError, Result};
pub use tensor::Tensor;

/// Optional re-export of the Sentry SDK for error/performance monitoring.
///
/// Enable with `features = ["sentry"]` in Cargo.toml.
/// See README for init guard example and setup.
#[cfg(feature = "sentry")]
pub use sentry;
