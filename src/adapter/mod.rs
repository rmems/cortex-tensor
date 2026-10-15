// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional external ANN execution adapters.
//!
//! Core stage contracts live in [`crate::stage`] and do not name a framework.
//! This module is compiled only with the `candle` feature.

mod candle;

pub use candle::{
    AttentionWeights, CandleExecutor, CandleTensor, EmbeddingWeights, GatedMlpWeights, HiddenState,
    LayerNormWeights, LinearWeights, MlpWeights, RmsNormWeights, export_hidden_state,
    export_reference, import_hidden_state, import_reference,
};
