// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checkpoint adapter resolution for the GGUF router host.

use super::checkpoint::{GgufMetadata, GgufTensorInfo, MappedGgufCheckpoint};
use super::{GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0};
use crate::error::{HybridError, Result};

#[derive(Debug, Clone)]
pub(super) struct ModelAdapter {
    pub(super) architecture: String,
    pub(super) hidden_size: usize,
    pub(super) num_layers: usize,
    pub(super) num_experts: usize,
    pub(super) expert_used_count: usize,
    pub(super) token_embedding_tensor: String,
    pub(super) routing_tensor: String,
    pub(super) quantization: String,
}

pub(super) fn resolve_adapter(
    metadata: &GgufMetadata,
    checkpoint: &MappedGgufCheckpoint,
    path: &str,
) -> Result<ModelAdapter> {
    let architecture = metadata.architecture.clone();
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

    Ok(ModelAdapter {
        architecture,
        hidden_size,
        num_layers,
        num_experts,
        expert_used_count,
        token_embedding_tensor,
        routing_tensor,
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
/// length in `MoeRouter::gate_scores`) and the other to be ≥ `num_experts`.
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
