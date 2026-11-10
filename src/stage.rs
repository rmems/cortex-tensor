// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral stage-execution contracts for ANN inference.
//!
//! This module defines the vocabulary a backend-agnostic orchestrator uses to
//! describe and run an ANN forward pass as an ordered graph of *stages*:
//!
//! * **Describe:** [`StageKind`] names what a stage computes, [`AnnStage`]
//!   binds an identity ([`StageId`]) and input wiring ([`StageSource`]) to a
//!   kind, and [`AnnTopology`] is a validated ordered list of stages.
//! * **Capabilities:** [`AnnCapabilities`] reports which stage kinds
//!   ([`StageKindTag`]) and [`DType`]s a backend can execute, mirroring the
//!   `snn` module's `SnnCapabilities` contract.
//! * **Execute:** the [`AnnExecutor`] trait runs one stage at a time against
//!   backend-owned tensors, and [`run_topology`] composes an executor over a
//!   topology using an [`ExternalBindings`] source for ingress.
//!
//! The public execution contract is deliberately generic: [`AnnExecutor`]
//! never names [`crate::Tensor`] or `Vec<f32>`, only its associated
//! [`AnnExecutor::Tensor`] type (bounded by [`StageTensor`]). This keeps the
//! contract usable by future accelerated backends.
//!
//! # Intentional omissions
//!
//! A `device` field on [`TensorMeta`]/[`AnnCapabilities`] and a concrete
//! tensor conversion / wire format for crossing the stage boundary are
//! intentionally **not** modelled here; they are owned by RM-1827. Stage
//! descriptors ([`AnnStage`], [`StageSource`], [`AnnTopology`]) carry no serde
//! derives: they describe identity, kind, and wiring only, not a persisted
//! wire format.
//!
//! The optional `candle` feature adds `adapter::CandleExecutor`. That executor
//! runs stages with Candle's modules and copies row-major `f32` values through
//! `import_hidden_state` / `export_hidden_state`. That path is `HiddenState`,
//! not [`crate::adapter::HiddenStateBuffer`], and it is not the deferred
//! cross-backend wire format.

mod driver;
mod reference;
mod vocabulary;

#[cfg(feature = "burn")]
pub use crate::adapter::BurnAdapter;
pub use crate::adapter::{
    AdapterCapabilities, BackendLimitations, DeviceClass, ExternalAdapter, ExternalAdapterMarker,
    HiddenStateBuffer, NegotiationDocument,
};
pub use driver::{ExternalBindings, run_topology};
pub use reference::{ReferenceExecutor, ReferenceStageParams};
pub use vocabulary::{
    AnnCapabilities, AnnExecutor, AnnStage, AnnTopology, DType, MlpKind, NormKind, StageId,
    StageInput, StageKind, StageKindTag, StageSource, StageTensor, SubSpan, TensorMeta,
};
