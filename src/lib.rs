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
//! | [`tensor`] | `Tensor` type, fallible construction, core ops, finite-value policy |
//! | [`transformer`] | Transformer building blocks (attention, block, model) |
//! | [`moe`] | Mixture-of-Experts router + GGUF checkpoint bridge |
//! | [`types`] | Shared types used by `moe` |
//! | [`error`] | `CortexError` unified error type |
//!
//! Panic-style constructors and ops (`Tensor::from_vec`, `ops::matmul`, …)
//! remain as pre-1.0 compatibility wrappers. New code should use
//! [`Tensor::try_from_vec`] and the `try_*` functions in [`crate::tensor::ops`].

pub mod error;
pub mod moe;
pub mod tensor;
pub mod transformer;
pub mod types;

pub use error::{CortexError, HybridError, Result};
pub use tensor::Tensor;
