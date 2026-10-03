// SPDX-License-Identifier: MIT OR Apache-2.0

//! # cortex-tensor
//!
//! Backend-neutral ANN stage layer plus a deterministic `Vec<f32>` reference
//! backend for hybrid ANN/SNN research. External engines (Candle, then Burn)
//! plug in as optional adapters. Checkpoint parsing stays in `engram-parser`.
//! Placement and session orchestration stay in `hybrid-fusion`.
//!
//! ## Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`stage`] | Backend-neutral ANN stage contracts and the reference executor |
//! | [`adapter`] | External-engine capability and hidden-state contract |
//! | [`tensor`] | Reference-backend `Tensor`, fallible construction, core ops |
//! | [`transformer`] | Reference-backend transformer blocks |
//! | [`moe`] | Reference-backend MoE routing; in-tree GGUF is transitional (#47) |
//! | [`snn`] | SNN execution contract + optional crate-backed adapters |
//! | [`types`] | Shared types used by `moe` |
//! | [`error`] | `CortexError` unified error type |
//!
//! Panic-style constructors and ops (`Tensor::from_vec`, `ops::matmul`, …)
//! remain as pre-1.0 compatibility wrappers. New code should use
//! [`Tensor::try_from_vec`] and the `try_*` functions in [`crate::tensor::ops`].

pub mod adapter;
pub mod error;
pub mod moe;
pub mod reference_json;
pub mod snn;
pub mod stage;
pub mod tensor;
pub mod transformer;
pub mod types;

pub use error::{CortexError, HybridError, Result};
pub use stage::{AnnExecutor, AnnStage, AnnTopology, ReferenceExecutor, StageId, StageKind};
pub use tensor::Tensor;
