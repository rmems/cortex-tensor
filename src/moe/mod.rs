// SPDX-License-Identifier: MIT OR Apache-2.0

// NOTE: This module implements the main MoeRouter. High LOC is due to the
// full public API, loading logic, multiple routing modes, and extensive tests.
// Further modularization planned.

//! Public MoE router API backed by a GGUF checkpoint bridge.
//!
//! Private helpers live in:
//! - `moe/checkpoint.rs` for GGUF parsing + mapped tensor access
//! - `moe/adapter.rs` for checkpoint adapter resolution and tensor selection
//! - `moe/dequant.rs` for quantized tensor dequantization
//! - `moe/gguf.rs` for GGUF constants
//! - `moe/routing.rs` for routing math and embedding resampling
//!
//! ## Parser boundary (see #8 / #47)
//! `rmems/engram-parser` is the parser-layer provider for this
//! ecosystem: the canonical home for GGUF v3 layout parsing (header, KV
//! metadata, tensor directory) and MoE per-expert *raw* weight extraction
//! (see closed engram-parser#7 and corinth-canal#115). This crate stays the
//! consumer: f32 math, `Tensor` ops, routing, dequantization, and model
//! adapters on top. Parser/dtype freeze holds until a consume follow-up can
//! wrap `parse_checkpoint_layout` around engram-parser 0.2.0 — that work is
//! #47, blocked on engram-parser#45 (mmap + K-quant). Do not add Q6_K / IQ3_*
//! here. Dequantization itself stays owned by this crate; only widening the
//! supported dtype set is frozen, since new dtypes arrive with the parser's
//! type ids. No dependency on engram-parser is declared yet.
//!
//! ## Ecosystem note (see #9 / #32)
//! Safetensors header inspection, deterministic manifests, and MoE candidate
//! discovery belong in `rmems/engram-parser` behind the off-by-default
//! `safetensors` cargo feature (engram-parser#10, corinth-canal#116). This
//! crate still does not own that parse surface; the eventual dependency is
//! one crate (`engram-parser` with `features = ["safetensors"]`), not a
//! dedicated `safetensors-parser` crate. Planning/alignment only — no
//! Safetensors backend or dep is implemented here.

mod adapter;
mod checkpoint;
mod dequant;
mod gguf;
mod routing;

#[cfg(test)]
pub(crate) mod test_fixtures;

use self::adapter::{ModelAdapter, resolve_adapter};
use self::checkpoint::{MappedGgufCheckpoint, probe_and_map_checkpoint};
use self::routing::{
    apply_extract_token_options, checkpoint_gate_scores, normalize_l2, resample_embedding,
    synthetic_gate_scores,
};
use crate::error::{HybridError, Result};
pub use crate::types::RoutingMode;
pub use crate::types::{EMBEDDING_DIM, ExtractTokenOptions};

pub(crate) use self::gguf::{
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ3_S, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGUF_MAGIC,
    GGUF_VALUE_TYPE_ARRAY, GGUF_VALUE_TYPE_BOOL, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64,
    GGUF_VALUE_TYPE_INT8, GGUF_VALUE_TYPE_INT16, GGUF_VALUE_TYPE_INT32, GGUF_VALUE_TYPE_INT64,
    GGUF_VALUE_TYPE_STRING, GGUF_VALUE_TYPE_UINT8, GGUF_VALUE_TYPE_UINT16, GGUF_VALUE_TYPE_UINT32,
    GGUF_VALUE_TYPE_UINT64, GGUF_VERSION,
};

pub(crate) use self::routing::{reject_nan_routing_scores, route_top_k};

pub struct MoeRouter {
    model_path: String,
    num_experts: usize,
    top_k: usize,
    loaded: bool,
    metadata: RouterMetadata,
    adapter: Option<ModelAdapter>,
    routing_mode: RoutingMode,
    checkpoint: Option<MappedGgufCheckpoint>,
}

#[derive(Debug, Clone, Default)]
pub struct RouterMetadata {
    pub architecture: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub expert_used_count: usize,
    pub quantization: String,
    pub routing_tensor_name: String,
}

#[derive(Debug, Clone)]
pub struct MoeOutput {
    pub expert_weights: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub hidden: Vec<f32>,
}

impl MoeRouter {
    pub fn load(model_path: &str, num_experts: usize, top_k: usize) -> Result<Self> {
        Self::load_with_mode(model_path, num_experts, top_k, RoutingMode::StubUniform)
    }

    pub fn load_with_mode(
        model_path: &str,
        num_experts: usize,
        top_k: usize,
        routing_mode: RoutingMode,
    ) -> Result<Self> {
        if model_path.is_empty() {
            return Self::build_stub_router(num_experts, top_k, routing_mode);
        }

        let (metadata, checkpoint, adapter) = Self::probe_and_map(model_path)?;
        let effective_num_experts = if num_experts == 0 {
            metadata.num_experts
        } else {
            num_experts
        };
        if effective_num_experts > metadata.num_experts {
            return Err(HybridError::InvalidConfig(format!(
                "num_experts ({effective_num_experts}) exceeds checkpoint expert_count ({})",
                metadata.num_experts
            )));
        }

        let effective_top_k = if top_k == 0 {
            metadata.expert_used_count.max(1).min(effective_num_experts)
        } else {
            top_k.max(1).min(effective_num_experts)
        };
        Ok(Self {
            model_path: model_path.to_owned(),
            num_experts: effective_num_experts,
            top_k: effective_top_k,
            loaded: true,
            metadata,
            adapter: Some(adapter),
            routing_mode,
            checkpoint: Some(checkpoint),
        })
    }

    fn build_stub_router(
        num_experts: usize,
        top_k: usize,
        routing_mode: RoutingMode,
    ) -> Result<Self> {
        let inferred_experts = num_experts.max(1);
        let inferred_top_k = top_k.max(1).min(inferred_experts);
        Ok(Self {
            model_path: String::new(),
            num_experts: inferred_experts,
            top_k: inferred_top_k,
            loaded: false,
            metadata: RouterMetadata {
                architecture: "stub".into(),
                hidden_size: EMBEDDING_DIM,
                num_layers: 0,
                num_experts: inferred_experts,
                expert_used_count: inferred_top_k,
                quantization: "stub".into(),
                routing_tensor_name: "synthetic".into(),
            },
            adapter: None,
            routing_mode,
            checkpoint: None,
        })
    }

    pub fn probe_model(path: &str) -> Result<RouterMetadata> {
        let (metadata, _checkpoint, _adapter) = Self::probe_and_map(path)?;
        Ok(metadata)
    }

    pub fn forward(&mut self, embedding: &[f32]) -> Result<MoeOutput> {
        if embedding.len() != EMBEDDING_DIM {
            return Err(HybridError::InputLengthMismatch {
                expected: EMBEDDING_DIM,
                got: embedding.len(),
            });
        }

        match self.routing_mode {
            RoutingMode::StubUniform => Ok(self.stub_output()),
            RoutingMode::DenseSim => self.simulate_moe_routing(embedding),
        }
    }

    /// Dequantize the checkpoint token row at native `hidden_size` (no resample, no L2).
    pub fn extract_token_embedding(&mut self, token_id: usize) -> Result<Vec<f32>> {
        self.extract_token_embedding_with_options(token_id, ExtractTokenOptions::default())
    }

    /// Dequantize a token row, optionally resampling and/or L2-normalizing.
    pub fn extract_token_embedding_with_options(
        &mut self,
        token_id: usize,
        options: ExtractTokenOptions,
    ) -> Result<Vec<f32>> {
        let adapter = self
            .adapter
            .as_ref()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        let checkpoint = self
            .checkpoint
            .as_mut()
            .ok_or_else(|| HybridError::ModelLoad {
                path: self.model_path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        let embedding = checkpoint.extract_token_embedding(
            &adapter.token_embedding_tensor,
            &self.model_path,
            token_id,
        )?;
        Ok(apply_extract_token_options(&embedding, options))
    }

    fn probe_and_map(path: &str) -> Result<(RouterMetadata, MappedGgufCheckpoint, ModelAdapter)> {
        let (_raw_metadata, checkpoint) = probe_and_map_checkpoint(path)?;
        let adapter = resolve_adapter(checkpoint.metadata(), &checkpoint, path)?;
        let metadata = RouterMetadata {
            architecture: adapter.architecture.clone(),
            hidden_size: adapter.hidden_size,
            num_layers: adapter.num_layers,
            num_experts: adapter.num_experts,
            expert_used_count: adapter.expert_used_count,
            quantization: adapter.quantization.clone(),
            routing_tensor_name: adapter.routing_tensor.clone(),
        };
        Ok((metadata, checkpoint, adapter))
    }

    fn simulate_moe_routing(&self, embedding: &[f32]) -> Result<MoeOutput> {
        let gate_scores = self.gate_scores(embedding)?;
        let (expert_weights, selected_experts) = route_top_k(&gate_scores, self.top_k)?;
        let selected_mass: f32 = selected_experts
            .iter()
            .map(|&idx| expert_weights[idx])
            .sum();
        let hidden: Vec<f32> = embedding.iter().map(|&v| v * selected_mass).collect();

        Ok(MoeOutput {
            expert_weights,
            selected_experts,
            hidden,
        })
    }

    /// Gate scores for `embedding` under this router's checkpoint or synthetic
    /// policy. Exposed crate-wide so `snn::SpikingMoeRouter` can compose
    /// routing with an external [`crate::snn::SnnBackend`].
    pub(crate) fn gate_scores(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        if let (Some(checkpoint), Some(adapter)) = (&self.checkpoint, &self.adapter) {
            let mut routed_embedding = resample_embedding(embedding, adapter.hidden_size);
            normalize_l2(&mut routed_embedding);
            return checkpoint_gate_scores(
                checkpoint,
                &self.model_path,
                &adapter.routing_tensor,
                self.num_experts,
                &routed_embedding,
            );
        }

        Ok(synthetic_gate_scores(self.num_experts, embedding))
    }

    fn stub_output(&self) -> MoeOutput {
        let n = self.num_experts.max(1);
        MoeOutput {
            expert_weights: vec![1.0 / n as f32; n],
            selected_experts: (0..self.top_k.min(n)).collect(),
            hidden: vec![0.0; EMBEDDING_DIM],
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    pub fn architecture(&self) -> &str {
        &self.metadata.architecture
    }

    pub fn quantization(&self) -> &str {
        &self.metadata.quantization
    }

    pub fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }

    pub fn num_layers(&self) -> usize {
        self.metadata.num_layers
    }

    pub fn checkpoint_num_experts(&self) -> usize {
        self.metadata.num_experts
    }

    pub fn checkpoint_expert_used_count(&self) -> usize {
        self.metadata.expert_used_count
    }

    pub fn routing_tensor_name(&self) -> &str {
        &self.metadata.routing_tensor_name
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn routing_mode(&self) -> RoutingMode {
        self.routing_mode
    }
}

#[cfg(test)]
mod tests;
