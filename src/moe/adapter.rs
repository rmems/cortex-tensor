// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checkpoint adapter resolution for the reference router.
//!
//! Tensor shapes are engram-parser's outermost-first order. A GGUF token
//! embedding stored as `ne = [hidden, vocab]` is `[vocab, hidden]` here, the
//! same order a Safetensors header already uses.

use super::checkpoint::{MappedCheckpoint, TensorKind, TensorView};
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

pub(super) fn resolve_adapter(checkpoint: &MappedCheckpoint, path: &str) -> Result<ModelAdapter> {
    let architecture = checkpoint.architecture().to_owned();
    let hidden_size = checkpoint
        .numeric(&format!("{architecture}.embedding_length"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.embedding_length' in '{path}'"
            ))
        })?;
    let num_layers = checkpoint
        .numeric(&format!("{architecture}.block_count"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.block_count' in '{path}'"
            ))
        })?;
    let num_experts = checkpoint
        .numeric(&format!("{architecture}.expert_count"))
        .ok_or_else(|| {
            HybridError::UnsupportedFormat(format!(
                "missing '{architecture}.expert_count' in '{path}'"
            ))
        })?;
    if num_experts == 0 {
        return Err(HybridError::UnsupportedFormat(format!(
            "{architecture}.expert_count in '{path}' must be greater than 0"
        )));
    }
    let expert_used_count = checkpoint
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
        quantization: checkpoint.quantization(),
    })
}

fn is_token_embedding_supported(info: &TensorView, hidden_size: usize) -> bool {
    info.dims.len() == 2
        && info.dims[1] == hidden_size
        && info.dims[0] > 0
        && match info.kind {
            TensorKind::F32 | TensorKind::F16 => true,
            TensorKind::Q8Block => hidden_size.is_multiple_of(32),
            TensorKind::Q5KBlock => hidden_size.is_multiple_of(256),
            TensorKind::Iq3S | TensorKind::Other(_) => false,
        }
}

const TOKEN_EMBEDDING_NAMES: &[&str] = &[
    "token_embd.weight",
    "tok_embeddings.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
];

fn token_embedding_candidates(checkpoint: &MappedCheckpoint) -> Vec<String> {
    let mut names = Vec::new();
    for name in TOKEN_EMBEDDING_NAMES {
        if checkpoint.has_tensor(name) {
            names.push((*name).to_owned());
        }
    }
    for name in checkpoint.tensor_names_with_suffix("embed_tokens.weight") {
        if !names.iter().any(|existing| existing == &name) {
            names.push(name);
        }
    }
    names
}

fn resolve_token_embedding_tensor(
    checkpoint: &MappedCheckpoint,
    hidden_size: usize,
    path: &str,
) -> Result<String> {
    let mut unsupported = None;
    for name in token_embedding_candidates(checkpoint) {
        let info = checkpoint.tensor_view(&name, path)?;
        if is_token_embedding_supported(&info, hidden_size) {
            return Ok(name);
        }
        if unsupported.is_none() {
            unsupported = Some(HybridError::UnsupportedFormat(format!(
                "token embedding tensor '{name}' in '{path}' has unsupported shape/type: dtype={} dims={:?} (expected [vocab>0, hidden_size={hidden_size}])",
                info.kind, info.dims
            )));
        }
    }
    if let Some(err) = unsupported {
        return Err(err);
    }
    Err(HybridError::MissingTensor {
        name: "token_embd.weight".into(),
        path: path.to_owned(),
    })
}

/// Resolve the gate tensor and reject shapes the runtime cannot index.
///
/// Gate scoring requires one outermost-first dimension to equal
/// `hidden_size` and the other to be at least the expert count. The
/// contiguous axis is the last dimension.
fn resolve_routing_tensor(
    checkpoint: &MappedCheckpoint,
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
        })?;
    let routing_info = checkpoint.tensor_view(&routing_tensor, path)?;
    if routing_info.kind != TensorKind::F32 || routing_info.dims.len() != 2 {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' must be rank-2 F32 in '{path}', got dims={:?} dtype={}",
            routing_info.dims, routing_info.kind
        )));
    }
    let (d0, d1) = (routing_info.dims[0], routing_info.dims[1]);
    let has_hidden_size_dim = d0 == hidden_size || d1 == hidden_size;
    if !has_hidden_size_dim {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' in '{path}' has unsupported orientation dims={d0}x{d1}; expected one dimension to equal hidden_size={hidden_size} and the other to be at least {num_experts}"
        )));
    }
    let routing_experts = if d1 == hidden_size { d0 } else { d1 };
    if routing_experts < num_experts {
        return Err(HybridError::UnsupportedFormat(format!(
            "routing tensor '{routing_tensor}' in '{path}' only exposes {routing_experts} experts, expected at least {num_experts}"
        )));
    }
    Ok(routing_tensor)
}
