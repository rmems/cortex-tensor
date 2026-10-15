// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reference-backend binding over engram-parser checkpoints.
//!
//! Format parsing, tensor inventory, raw payloads, and packed dequantization
//! belong to published `engram-parser` 0.3.0 (`features = ["safetensors"]`;
//! the `mmap` feature is not enabled). This module does not walk GGUF
//! headers or Safetensors JSON. `open_checkpoint` chooses the backend.
//! Malformed input returns that error; there is no in-tree parser to fall
//! back to.
//!
//! # Migration shim
//! [`MappedCheckpoint`] exists so [`super::MoeRouter`] can keep its load,
//! token-extract, and gate API while weight selection still lives here.
//! Remove the shim when callers take an [`engram_parser::Checkpoint`] and
//! perform this binding outside the crate. Do not add format parsers,
//! dtype coverage, or mmap policy here.
//!
//! `Checkpoint::tensor_bytes` does not promise `f32` alignment. Values are
//! copied with [`f32::from_le_bytes`], so this module has no `unsafe`.

use engram_parser::checkpoint::{Checkpoint, MetadataValue, TensorDType, TensorInfo};
use engram_parser::gguf::{
    DType, dequantize_q5_k, dequantize_q8_0, f16_bits_to_f32, packed_row_size,
};
use engram_parser::{ParserError, open_checkpoint};
use std::fmt;
use std::path::Path;

use crate::error::{HybridError, Result};

/// Validated engram-parser release this binding was written against.
#[cfg(test)]
pub(super) const ENGRAM_PARSER_VERSION: &str = "0.3.0";

/// Router-facing view of one opened checkpoint.
#[derive(Debug)]
pub(super) struct MappedCheckpoint {
    inner: engram_parser::checkpoint::AnyCheckpoint,
}

/// Outermost-first shape plus the dtypes the reference backend binds.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TensorView {
    pub(super) dims: Vec<usize>,
    pub(super) kind: TensorKind,
}

/// Dtypes the reference router knows how to bind. Anything else stays a label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TensorKind {
    F32,
    F16,
    Q8Block,
    Q5KBlock,
    Iq3S,
    Other(String),
}

impl fmt::Display for TensorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => f.write_str("F32"),
            Self::F16 => f.write_str("F16"),
            Self::Q8Block => f.write_str("Q8_0"),
            Self::Q5KBlock => f.write_str("Q5_K"),
            Self::Iq3S => f.write_str("IQ3_S"),
            Self::Other(label) => f.write_str(label),
        }
    }
}

impl MappedCheckpoint {
    pub(super) fn architecture(&self) -> &str {
        self.inner
            .metadata()
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
            .filter(|value| !value.is_empty())
            .unwrap_or("unknown")
    }

    pub(super) fn quantization(&self) -> String {
        quantization_label(self.inner.metadata())
    }

    pub(super) fn numeric(&self, key: &str) -> Option<usize> {
        metadata_usize(self.inner.metadata(), key)
    }

    pub(super) fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.inner
            .tensors()
            .iter()
            .map(|tensor| tensor.name.as_str())
    }

    pub(super) fn has_tensor(&self, name: &str) -> bool {
        self.inner.tensor(name).is_some()
    }

    pub(super) fn find_first_tensor_with_suffix(&self, suffix: &str) -> Option<String> {
        let mut matches: Vec<&str> = self
            .tensor_names()
            .filter(|name| name.ends_with(suffix))
            .collect();
        matches.sort_unstable_by_key(|name| tensor_block_sort_key(name));
        matches.first().map(|name| (*name).to_owned())
    }

    pub(super) fn tensor_view(&self, name: &str, path: &str) -> Result<TensorView> {
        let info = self.tensor_info(name, path)?;
        Ok(TensorView {
            dims: info.shape.dims().to_vec(),
            kind: kind_of(info),
        })
    }

    pub(super) fn f32_tensor(&self, name: &str, path: &str) -> Result<Vec<f32>> {
        let view = self.tensor_view(name, path)?;
        if view.kind != TensorKind::F32 {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be F32, got {}",
                view.kind
            )));
        }
        let bytes = self.payload(name)?;
        let elements = element_count(&view.dims, path, name)?;
        decode_f32(&bytes, elements, path, name)
    }

    pub(super) fn extract_token_embedding(
        &self,
        tensor_name: &str,
        path: &str,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        let view = self.tensor_view(tensor_name, path)?;
        let (vocab, hidden) = token_embedding_dims(&view, tensor_name, path)?;
        if token_id >= vocab {
            return Err(HybridError::InputLengthMismatch {
                expected: vocab,
                got: token_id,
            });
        }
        let bytes = self.payload(tensor_name)?;
        match view.kind {
            TensorKind::F32 => {
                let row = row_bytes(&bytes, hidden * 4, token_id, path, tensor_name)?;
                decode_f32(row, hidden, path, tensor_name)
            }
            TensorKind::F16 => {
                let row = row_bytes(&bytes, hidden * 2, token_id, path, tensor_name)?;
                let (chunks, remainder) = row.as_chunks::<2>();
                debug_assert!(remainder.is_empty());
                Ok(chunks
                    .iter()
                    .map(|chunk| f16_bits_to_f32(u16::from_le_bytes(*chunk)))
                    .collect())
            }
            TensorKind::Q8Block => dequantize_row(
                DType::Q8_0,
                &bytes,
                hidden,
                token_id,
                path,
                tensor_name,
                dequantize_q8_0,
            ),
            TensorKind::Q5KBlock => dequantize_row(
                DType::Q5_K,
                &bytes,
                hidden,
                token_id,
                path,
                tensor_name,
                dequantize_q5_k,
            ),
            TensorKind::Iq3S => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{tensor_name}' uses IQ3_S token embeddings; use llama.cpp prompt embeddings for this checkpoint"
            ))),
            TensorKind::Other(label) => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{tensor_name}' has unsupported dtype {label}"
            ))),
        }
    }

    fn tensor_info<'a>(&'a self, name: &str, path: &str) -> Result<&'a TensorInfo> {
        self.inner
            .tensor(name)
            .ok_or_else(|| HybridError::MissingTensor {
                name: name.to_owned(),
                path: path.to_owned(),
            })
    }

    fn payload(&self, name: &str) -> Result<std::borrow::Cow<'_, [u8]>> {
        self.inner.tensor_bytes(name).map_err(map_parser_error)
    }
}

pub(super) fn open_mapped_checkpoint(path: &str) -> Result<MappedCheckpoint> {
    let inner = open_checkpoint(Path::new(path)).map_err(map_parser_error)?;
    Ok(MappedCheckpoint { inner })
}

fn kind_of(info: &TensorInfo) -> TensorKind {
    match &info.dtype {
        TensorDType::F32 => TensorKind::F32,
        TensorDType::F16 => TensorKind::F16,
        TensorDType::GgufPacked { ggml_type } => match DType::from_ggml_type(*ggml_type) {
            DType::Q8_0 => TensorKind::Q8Block,
            DType::Q5_K => TensorKind::Q5KBlock,
            DType::IQ3_S => TensorKind::Iq3S,
            other => TensorKind::Other(other.label().to_owned()),
        },
        other => TensorKind::Other(other.label().to_owned()),
    }
}

fn token_embedding_dims(view: &TensorView, name: &str, path: &str) -> Result<(usize, usize)> {
    if view.dims.len() != 2 || view.dims[0] == 0 || view.dims[1] == 0 {
        return Err(HybridError::UnsupportedFormat(format!(
            "token embedding tensor '{name}' in '{path}' has unsupported shape/type: dtype={} dims={:?} (expected [vocab>0, hidden>0])",
            view.kind, view.dims
        )));
    }
    Ok((view.dims[0], view.dims[1]))
}

fn dequantize_row(
    dtype: DType,
    bytes: &[u8],
    hidden: usize,
    token_id: usize,
    path: &str,
    tensor_name: &str,
    dequant: fn(&[u8], &[usize]) -> engram_parser::Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    let row_size = packed_row_size(dtype.ggml_type(), hidden).map_err(map_parser_error)?;
    let row = row_bytes(bytes, row_size, token_id, path, tensor_name)?;
    // `dims` is innermost-first; a single row's width is the contiguous axis.
    dequant(row, &[hidden]).map_err(map_parser_error)
}

fn row_bytes<'a>(
    bytes: &'a [u8],
    row_size: usize,
    row_idx: usize,
    path: &str,
    tensor_name: &str,
) -> Result<&'a [u8]> {
    let start = row_idx
        .checked_mul(row_size)
        .ok_or_else(|| row_overflow(path, tensor_name))?;
    let end = start
        .checked_add(row_size)
        .ok_or_else(|| row_overflow(path, tensor_name))?;
    if end > bytes.len() {
        return Err(HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("tensor '{tensor_name}' row extends beyond payload"),
        });
    }
    Ok(&bytes[start..end])
}

fn decode_f32(bytes: &[u8], elements: usize, path: &str, name: &str) -> Result<Vec<f32>> {
    let expected = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| row_overflow(path, name))?;
    if bytes.len() != expected {
        return Err(HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!(
                "tensor '{name}' F32 payload is {} bytes, expected {expected}",
                bytes.len()
            ),
        });
    }
    let (chunks, remainder) = bytes.as_chunks::<4>();
    debug_assert!(remainder.is_empty());
    Ok(chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn element_count(dims: &[usize], path: &str, name: &str) -> Result<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("tensor '{name}' element count overflow"),
        })
}

fn row_overflow(path: &str, tensor_name: &str) -> HybridError {
    HybridError::ModelLoad {
        path: path.to_owned(),
        reason: format!("tensor '{tensor_name}' row offset overflow"),
    }
}

fn quantization_label(metadata: &engram_parser::checkpoint::CheckpointMetadata) -> String {
    if let Some(label) = metadata
        .get("general.quantization_type")
        .and_then(MetadataValue::as_str)
        .filter(|label| !label.is_empty())
    {
        return label.to_owned();
    }
    match metadata_usize(metadata, "general.file_type") {
        Some(0) => "F32".to_owned(),
        Some(1) => "F16".to_owned(),
        Some(other) => format!("GGUF({other})"),
        None => metadata
            .get("general.file_type")
            .and_then(MetadataValue::as_str)
            .filter(|label| !label.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| "unknown".to_owned()),
    }
}

fn metadata_usize(
    metadata: &engram_parser::checkpoint::CheckpointMetadata,
    key: &str,
) -> Option<usize> {
    match metadata.get(key)? {
        MetadataValue::UInt(value) => usize::try_from(*value).ok(),
        MetadataValue::Int(value) => usize::try_from(*value).ok(),
        MetadataValue::String(value) => value.parse().ok(),
        MetadataValue::F32(value) => whole_usize(f64::from(*value)),
        MetadataValue::F64(value) => whole_usize(*value),
        _ => None,
    }
}

fn whole_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    let rounded = value as u64;
    if rounded as f64 == value {
        usize::try_from(rounded).ok()
    } else {
        None
    }
}

fn tensor_block_sort_key(name: &str) -> (usize, &str) {
    let block = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(idx, _)| idx.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    (block, name)
}

fn map_parser_error(err: ParserError) -> HybridError {
    match err {
        ParserError::MissingTensor { name, path } => HybridError::MissingTensor { name, path },
        ParserError::Io { path, source } => HybridError::ModelLoad {
            path,
            reason: source.to_string(),
        },
        ParserError::InvalidLayout { path, reason } => HybridError::ModelLoad { path, reason },
        ParserError::UnsupportedFormat { .. }
        | ParserError::FeatureDisabled { .. }
        | ParserError::LimitExceeded { .. }
        | ParserError::HostSizeOverflow { .. }
        | ParserError::DuplicateTensorOwnership { .. }
        | ParserError::MissingShard { .. }
        | ParserError::ExpertOutOfRange { .. } => HybridError::UnsupportedFormat(err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::HybridError;
    use crate::moe::test_fixtures::*;
    use crate::types::EMBEDDING_DIM;

    #[test]
    fn validated_engram_release_is_recorded() {
        assert_eq!(ENGRAM_PARSER_VERSION, "0.3.0");
    }

    #[test]
    fn malformed_gguf_fails_without_an_in_tree_parser() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUS");
        push_u32(&mut bytes, GGUF_VERSION);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 0);
        let path = write_temp_file(&bytes, "bad-magic");
        let err = open_mapped_checkpoint(path.to_str().unwrap()).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("unsupported")
                || message.contains("invalid")
                || message.contains("model load"),
            "expected an explicit parser error, got {message}"
        );
        assert!(
            !message.contains("parse_checkpoint_layout"),
            "retired parser must not appear in the error: {message}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn tensor_inventory_uses_outermost_first_shapes() {
        let d1 = 3usize;
        let payload = vec![0u8; EMBEDDING_DIM * d1 * std::mem::size_of::<f32>()];
        let bytes = build_token_file(GGML_TYPE_F32, d1, payload);
        let path = write_temp_file(&bytes, "inventory-f32");
        let checkpoint = open_mapped_checkpoint(path.to_str().unwrap()).unwrap();

        let names: Vec<&str> = checkpoint.tensor_names().collect();
        assert_eq!(names, vec!["token_embd.weight"]);
        let view = checkpoint
            .tensor_view("token_embd.weight", path.to_str().unwrap())
            .unwrap();
        assert_eq!(view.dims, vec![d1, EMBEDDING_DIM]);
        assert_eq!(view.kind, TensorKind::F32);
        assert_eq!(
            checkpoint
                .find_first_tensor_with_suffix("embd.weight")
                .as_deref(),
            Some("token_embd.weight")
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn suffix_search_orders_blocks_numerically() {
        let hidden = 32usize;
        let gate = vec![0u8; hidden * 4 * std::mem::size_of::<f32>()];
        let bytes = build_test_gguf(
            vec![
                (
                    "blk.10.ffn_gate_inp.weight",
                    vec![hidden, 4],
                    GGML_TYPE_F32,
                    gate.clone(),
                ),
                (
                    "blk.2.ffn_gate_inp.weight",
                    vec![hidden, 4],
                    GGML_TYPE_F32,
                    gate,
                ),
                (
                    "token_embd.weight",
                    vec![hidden, 2],
                    GGML_TYPE_F32,
                    vec![0u8; hidden * 2 * std::mem::size_of::<f32>()],
                ),
            ],
            32,
            hidden,
        );
        let path = write_temp_file(&bytes, "block-order");
        let checkpoint = open_mapped_checkpoint(path.to_str().unwrap()).unwrap();
        assert_eq!(
            checkpoint
                .find_first_tensor_with_suffix("ffn_gate_inp.weight")
                .as_deref(),
            Some("blk.2.ffn_gate_inp.weight")
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn probe_extracts_f32_token_row_in_normalized_order() {
        let d1 = 3usize;
        let mut payload = Vec::new();
        for token in 0..d1 {
            for dim in 0..EMBEDDING_DIM {
                let value = token as f32 * 10.0 + dim as f32;
                payload.extend_from_slice(&value.to_le_bytes());
            }
        }
        let bytes = build_token_file(GGML_TYPE_F32, d1, payload);
        let path = write_temp_file(&bytes, "token_f32");
        let checkpoint = open_mapped_checkpoint(path.to_str().unwrap()).unwrap();
        let embedding = checkpoint
            .extract_token_embedding("token_embd.weight", path.to_str().unwrap(), 1)
            .unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        assert!((embedding[0] - 10.0).abs() <= 1e-6);
        assert!((embedding[1] - 11.0).abs() <= 1e-6);

        let err = checkpoint
            .extract_token_embedding("token_embd.weight", path.to_str().unwrap(), d1)
            .unwrap_err();
        assert!(matches!(
            err,
            HybridError::InputLengthMismatch { expected, got } if expected == d1 && got == d1
        ));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn f16_tensor_is_not_readable_as_f32() {
        let d1 = 3usize;
        let payload = vec![0u8; EMBEDDING_DIM * d1 * 2];
        let bytes = build_token_file(GGML_TYPE_F16, d1, payload);
        let path = write_temp_file(&bytes, "token_f16");
        let checkpoint = open_mapped_checkpoint(path.to_str().unwrap()).unwrap();
        let err = checkpoint
            .f32_tensor("token_embd.weight", path.to_str().unwrap())
            .unwrap_err();
        assert!(err.to_string().contains("must be F32"));
        let embedding = checkpoint
            .extract_token_embedding("token_embd.weight", path.to_str().unwrap(), 1)
            .unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        assert!(embedding.iter().all(|&value| value == 0.0));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn extract_rejects_iq3_s_after_a_successful_open() {
        let width = 256usize;
        let vocab = 2usize;
        let bytes = build_native_tensor(
            "token_embd.weight",
            GGML_TYPE_IQ3_S,
            width,
            vocab,
            vec![0u8; 110 * vocab],
        );
        let path = write_temp_file(&bytes, "token_iq3");
        let checkpoint = open_mapped_checkpoint(path.to_str().unwrap()).unwrap();
        let err = checkpoint
            .extract_token_embedding("token_embd.weight", path.to_str().unwrap(), 0)
            .unwrap_err();
        assert!(err.to_string().contains("IQ3_S token embeddings"));
        let err = checkpoint
            .extract_token_embedding("token_embd.weight", path.to_str().unwrap(), 5)
            .unwrap_err();
        assert!(matches!(
            err,
            HybridError::InputLengthMismatch { expected, got } if expected == vocab && got == 5
        ));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q8_0_and_q5_k_rows_delegate_to_engram() {
        let mut q8 = [0u8; 34];
        q8[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
        for (idx, quant) in q8[2..34].iter_mut().enumerate() {
            *quant = idx as u8;
        }
        let out = dequantize_q8_0(&q8, &[32]).unwrap();
        assert_eq!(out.len(), 32);
        assert!((out[3] - 6.0).abs() <= 1e-5);

        let mut q5 = [0u8; 176];
        q5[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        q5[4] = 1;
        for byte in q5[48..80].iter_mut() {
            *byte = 0x05;
        }
        let out = dequantize_q5_k(&q5, &[256]).unwrap();
        assert_eq!(out.len(), 256);
        assert!(out[..32].iter().all(|value| (value - 5.0).abs() <= 1e-4));
        assert!(out[32..64].iter().all(|value| value.abs() <= 1e-4));
    }

    fn build_token_file(ggml_type: u32, vocab: usize, payload: Vec<u8>) -> Vec<u8> {
        build_native_tensor(
            "token_embd.weight",
            ggml_type,
            EMBEDDING_DIM,
            vocab,
            payload,
        )
    }

    fn build_native_tensor(
        name: &str,
        ggml_type: u32,
        inner: usize,
        outer: usize,
        payload: Vec<u8>,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 1);
        push_u64(&mut out, 0);
        push_string(&mut out, name);
        push_u32(&mut out, 2);
        push_u64(&mut out, inner as u64);
        push_u64(&mut out, outer as u64);
        push_u32(&mut out, ggml_type);
        push_u64(&mut out, 0);
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend_from_slice(&payload);
        out
    }
}
