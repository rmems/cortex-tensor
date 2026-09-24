// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-family adapter resolution for the GGUF router host.

use super::checkpoint::{GgufMetadata, GgufTensorInfo, MappedGgufCheckpoint};
use super::{GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0};
use crate::error::{HybridError, Result};
use crate::types::ModelFamily;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SynapseSource {
    Real,
    RoutingF32,
    SyntheticFallback,
}

#[derive(Debug, Clone)]
pub(super) struct ModelAdapter {
    pub(super) family: ModelFamily,
    pub(super) architecture: String,
    pub(super) hidden_size: usize,
    pub(super) num_layers: usize,
    pub(super) num_experts: usize,
    pub(super) expert_used_count: usize,
    pub(super) token_embedding_tensor: String,
    pub(super) routing_tensor: String,
    pub(super) preferred_gpu_synapse_tensor: Option<String>,
    pub(super) real_gpu_synapse_tensor: Option<String>,
    pub(super) synapse_source: SynapseSource,
    pub(super) quantization: String,
}

impl ModelAdapter {
    pub(super) fn synapse_source_label(&self) -> &'static str {
        match self.synapse_source {
            SynapseSource::Real => "real",
            SynapseSource::RoutingF32 => "routing-f32",
            SynapseSource::SyntheticFallback => "synthetic-fallback",
        }
    }
}

pub(super) fn resolve_adapter(
    metadata: &GgufMetadata,
    checkpoint: &MappedGgufCheckpoint,
    family_override: Option<ModelFamily>,
    path: &str,
) -> Result<ModelAdapter> {
    let architecture = metadata.architecture.clone();
    let family = infer_family(&architecture, family_override, path)?;
    let hidden_size = metadata
        .numeric(&format!("{architecture}.embedding_length"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.embedding_length' in '{path}'"
            ))
        })?;
    let num_layers = metadata
        .numeric(&format!("{architecture}.block_count"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.block_count' in '{path}'"
            ))
        })?;
    let num_experts = metadata
        .numeric(&format!("{architecture}.expert_count"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.expert_count' in '{path}'"
            ))
        })?;
    let expert_used_count = metadata
        .numeric(&format!("{architecture}.expert_used_count"))
        .unwrap_or(1);
    let token_embedding_tensor = resolve_token_embedding_tensor(checkpoint, hidden_size, path)?;
    let routing_tensor = resolve_routing_tensor(checkpoint, hidden_size, num_experts, path)?;

    let preferred_gpu_synapse_tensor = checkpoint
        .has_tensor("blk.0.attn_q.weight")
        .then(|| "blk.0.attn_q.weight".to_owned());
    let attn_info = preferred_gpu_synapse_tensor
        .as_ref()
        .and_then(|name| checkpoint.tensor_info(name, path).ok());
    let is_real_f16_attn = attn_info.as_ref().is_some_and(|info| {
        // Relaxed from strict square [hidden, hidden] to support GQA models (Qwen3-MoE etc)
        // where attn_q.weight may be e.g. [num_q_heads * head_dim, hidden_size] (or transposed).
        // As long as it's F16 and involves the model hidden_size, treat as "real" F16 synapse-capable.
        // The synapse_source label + README document the contract for consumers.
        info.ggml_type == GGML_TYPE_F16 && info.dims.len() == 2 && info.dims.contains(&hidden_size)
    });
    let real_gpu_synapse_tensor = if is_real_f16_attn {
        preferred_gpu_synapse_tensor.clone()
    } else if preferred_gpu_synapse_tensor.is_some() {
        // qwen3 IQ3_S (and other non-F16 attn) now route to checkpoint-backed routing tensor
        // as "routing-f32" synapse source instead of falling back to synthetic. This fulfills
        // dequantized synapse path requirements for SAAQ without full IQ3_S dequant in adapter.
        Some(routing_tensor.clone())
    } else {
        None
    };
    let synapse_source = if is_real_f16_attn {
        SynapseSource::Real
    } else if preferred_gpu_synapse_tensor.is_some() {
        SynapseSource::RoutingF32
    } else {
        SynapseSource::SyntheticFallback
    };

    Ok(ModelAdapter {
        family,
        architecture,
        hidden_size,
        num_layers,
        num_experts,
        expert_used_count,
        token_embedding_tensor,
        routing_tensor,
        preferred_gpu_synapse_tensor,
        synapse_source,
        real_gpu_synapse_tensor,
        quantization: metadata.quantization.clone(),
    })
}

fn is_token_embedding_supported(info: &GgufTensorInfo, hidden_size: usize) -> bool {
    info.dims.len() == 2
        && info.dims[0] == hidden_size
        && info.dims[1] > 0
        && match info.ggml_type {
            GGML_TYPE_F32 | GGML_TYPE_F16 => true,
            GGML_TYPE_Q8_0 => info.dims[0].is_multiple_of(32),
            GGML_TYPE_Q5_K => info.dims[0].is_multiple_of(256),
            _ => false,
        }
}

fn resolve_token_embedding_tensor(
    checkpoint: &MappedGgufCheckpoint,
    hidden_size: usize,
    path: &str,
) -> Result<String> {
    if checkpoint.has_tensor("token_embd.weight") {
        let info = checkpoint.tensor_info("token_embd.weight", path)?;
        if is_token_embedding_supported(info, hidden_size) {
            return Ok("token_embd.weight".to_owned());
        }
        return Err(HybridError::UnsupportedFormat(format!(
            "token embedding tensor 'token_embd.weight' in '{path}' has unsupported shape/type: ggml_type={} dims={:?} (expected [hidden_size={hidden_size}, vocab_size>0])",
            info.ggml_type, info.dims
        )));
    }
    if checkpoint.has_tensor("tok_embeddings.weight") {
        let info = checkpoint.tensor_info("tok_embeddings.weight", path)?;
        if is_token_embedding_supported(info, hidden_size) {
            return Ok("tok_embeddings.weight".to_owned());
        }
        return Err(HybridError::UnsupportedFormat(format!(
            "token embedding tensor 'tok_embeddings.weight' in '{path}' has unsupported shape/type: ggml_type={} dims={:?} (expected [hidden_size={hidden_size}, vocab_size>0])",
            info.ggml_type, info.dims
        )));
    }
    Err(HybridError::MissingTensor {
        name: "token_embd.weight".into(),
        path: path.to_owned(),
    })
}

/// Resolve the GGUF gate tensor name and reject shapes the runtime cannot index.
///
/// Gate scoring (`routing_weight_index`) requires one dimension to equal
/// `hidden_size` (metadata `embedding_length`, matching the resampled embedding
/// length in `MoeRouter::compute_gate_scores`) and the other to be ≥ `num_experts`.
/// Do **not** relax this to `min(d0, d1) >= num_experts`: that previously
/// accepted tensors that then failed on the first gate-score pass.
fn resolve_routing_tensor(
    checkpoint: &MappedGgufCheckpoint,
    hidden_size: usize,
    num_experts: usize,
    path: &str,
) -> Result<String> {
    let routing_tensor = checkpoint
        .find_first_tensor_with_suffix("ffn_gate_inp.weight")
        .or_else(|| checkpoint.find_first_tensor_with_suffix("ffn_gate.weight"))
        .ok_or_else(|| HybridError::MissingTensor {
            name: "ffn_gate_inp.weight".into(),
            path: path.to_owned(),
        })?
        .to_owned();
    let routing_info = checkpoint.tensor_info(&routing_tensor, path)?;
    if routing_info.ggml_type != GGML_TYPE_F32 || routing_info.dims.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' must be rank-2 F32 in '{path}', got dims={:?} ggml_type={}",
            routing_info.dims, routing_info.ggml_type
        )));
    }
    let (d0, d1) = (routing_info.dims[0], routing_info.dims[1]);
    // Same orientation contract as `routing_weight_index` (routing.rs).
    let has_hidden_size_dim = d0 == hidden_size || d1 == hidden_size;
    if !has_hidden_size_dim {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' in '{path}' has unsupported orientation dims={d0}x{d1}; expected one dimension to equal hidden_size={hidden_size} and the other to be at least {num_experts}"
        )));
    }
    let routing_experts = if d0 == hidden_size { d1 } else { d0 };
    if routing_experts < num_experts {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' in '{path}' only exposes {routing_experts} experts, expected at least {num_experts}"
        )));
    }
    Ok(routing_tensor)
}

fn infer_family(
    architecture: &str,
    family_override: Option<ModelFamily>,
    _path: &str,
) -> Result<ModelFamily> {
    let inferred = match architecture {
        "qwen3moe" => ModelFamily::Qwen3Moe,
        "gemma4" => ModelFamily::Gemma4,
        "deepseek2" => ModelFamily::DeepSeek2,
        "llama" => ModelFamily::LlamaMoe,
        _ => ModelFamily::ReferenceMoe,
    };

    #[allow(clippy::collapsible_if)]
    if let Some(expected) = family_override {
        if expected != inferred {
            return Err(HybridError::InvalidConfig(format!(
                "model_family override {:?} does not match GGUF architecture '{architecture}'",
                expected
            )));
        }
    }

    Ok(inferred)
}
