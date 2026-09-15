// SPDX-License-Identifier: MIT OR Apache-2.0

// NOTE: This file contains GGUF parsing logic. Some LOC metrics are high due to
// the complexity of supporting multiple quantization formats (F32, F16, Q8_0, Q5_K, etc.)
// and memory-mapped access. Refactoring is tracked separately.

//! GGUF checkpoint parsing and mapped tensor access for the router bridge.
//!
//! **Frozen for parser work (see #8 / #47).** The canonical home for GGUF v3
//! layout parsing and per-expert raw weight extraction is `rmems/engram-parser`
//! (closed engram-parser#7, source corinth-canal#115). Parser/dtype freeze
//! holds until #47 can wrap `parse_checkpoint_layout` around engram-parser
//! 0.2.0 (blocked on engram-parser#45). Do not add Q6_K / IQ3_* here. What
//! stays in this crate: mmap'd tensor access, dequantization to f32, routing,
//! and model adapters.

use super::dequant;
use super::{
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_IQ3_S, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGUF_MAGIC,
    GGUF_VALUE_TYPE_ARRAY, GGUF_VALUE_TYPE_BOOL, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64,
    GGUF_VALUE_TYPE_INT8, GGUF_VALUE_TYPE_INT16, GGUF_VALUE_TYPE_INT32, GGUF_VALUE_TYPE_INT64,
    GGUF_VALUE_TYPE_STRING, GGUF_VALUE_TYPE_UINT8, GGUF_VALUE_TYPE_UINT16, GGUF_VALUE_TYPE_UINT32,
    GGUF_VALUE_TYPE_UINT64, GGUF_VERSION,
};
use crate::error::{HybridError, Result};
use memmap2::{MmapMut, MmapOptions};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::slice;

#[derive(Debug)]
pub(super) struct MappedGgufCheckpoint {
    mmap: MmapMut,
    tensors: HashMap<String, GgufTensorInfo>,
    metadata: GgufMetadata,
}

#[derive(Debug, Clone)]
pub(super) struct GgufTensorInfo {
    pub(super) dims: Vec<usize>,
    pub(super) ggml_type: u32,
    pub(super) relative_offset: usize,
    pub(super) absolute_offset: usize,
    pub(super) n_elements: usize,
}

#[derive(Debug)]
pub(super) struct ParsedCheckpointLayout {
    pub(super) metadata: GgufMetadata,
    pub(super) tensors: HashMap<String, GgufTensorInfo>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct GgufMetadata {
    pub(super) architecture: String,
    pub(super) quantization: String,
    numerics: HashMap<String, u64>,
}

struct GgufCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl GgufMetadata {
    pub(super) fn numeric(&self, key: &str) -> Option<usize> {
        self.numerics.get(key).copied().map(|v| v as usize)
    }
}

impl MappedGgufCheckpoint {
    pub(super) fn extract_token_embedding(
        &mut self,
        tensor_name: &str,
        path: &str,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        let info = self.tensor_info(tensor_name, path)?.clone();
        let d0 = info.dims[0];
        let d1 = info.dims.get(1).copied().unwrap_or(0);

        if token_id >= d1 {
            return Err(HybridError::InputLengthMismatch {
                expected: d1,
                got: token_id,
            });
        }

        match info.ggml_type {
            GGML_TYPE_F32 => {
                let weights = self.f32_tensor(tensor_name, path)?;
                Ok(weights[token_id * d0..token_id * d0 + d0].to_vec())
            }
            GGML_TYPE_F16 => {
                let values = self.u16_tensor_values(&info, path, tensor_name)?;
                Ok(values[token_id * d0..token_id * d0 + d0]
                    .iter()
                    .map(|&b| dequant::f16_to_f32(b))
                    .collect())
            }
            GGML_TYPE_Q8_0 => dequant::dequantize_row_q8_0(
                self.row_bytes(&info, token_id, path, tensor_name)?,
                d0,
            ),
            GGML_TYPE_Q5_K => dequant::dequantize_row_q5_k(
                self.row_bytes(&info, token_id, path, tensor_name)?,
                d0,
            ),
            GGML_TYPE_IQ3_S => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{tensor_name}' uses IQ3_S token embeddings; use llama.cpp prompt embeddings for this checkpoint"
            ))),
            other => Err(HybridError::UnsupportedFormat(format!(
                "tensor '{tensor_name}' has unsupported ggml_type={other}"
            ))),
        }
    }
}

pub(super) fn probe_and_map_checkpoint(path: &str) -> Result<(GgufMetadata, MappedGgufCheckpoint)> {
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: e.to_string(),
        })?;
    // SAFETY: The file is a valid, readable file descriptor opened above.
    // `map_copy` creates a private copy-on-write mapping that does not
    // write back to the underlying file.  The writable mapping is required
    // by `cuMemHostRegister_v2`, which expects a non-const pointer even
    // though it does not modify the memory contents.
    let mmap =
        unsafe { MmapOptions::new().map_copy(&file) }.map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("copy-on-write mmap failed: {e}"),
        })?;

    let parsed = parse_checkpoint_layout(&mmap, path)?;

    Ok((
        parsed.metadata.clone(),
        MappedGgufCheckpoint {
            mmap,
            tensors: parsed.tensors,
            metadata: parsed.metadata,
        },
    ))
}

pub(super) fn parse_checkpoint_layout(bytes: &[u8], path: &str) -> Result<ParsedCheckpointLayout> {
    let mut cursor = GgufCursor::new(bytes);
    let (tensor_count, kv_count) = parse_header(&mut cursor, path)?;
    let mut architecture = String::new();
    let (alignment, file_type, numerics) =
        read_metadata_kv(&mut cursor, kv_count, path, &mut architecture)?;
    let mut tensors = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let (name, info) = read_tensor_info(&mut cursor, path)?;
        tensors.insert(name, info);
    }
    let tensor_data_offset = dequant::align_up(cursor.offset, alignment);
    for tensor in tensors.values_mut() {
        tensor.absolute_offset = tensor_data_offset + tensor.relative_offset;
    }
    Ok(ParsedCheckpointLayout {
        metadata: GgufMetadata {
            architecture: if architecture.is_empty() {
                "unknown".into()
            } else {
                architecture
            },
            quantization: dequant::quantization_label(file_type),
            numerics,
        },
        tensors,
    })
}

fn parse_header(cursor: &mut GgufCursor, path: &str) -> Result<(usize, usize)> {
    let magic = cursor.read_exact(4, path)?;
    if magic != GGUF_MAGIC {
        return Err(HybridError::UnsupportedFormat(format!(
            "unrecognised model magic bytes: {magic:?}"
        )));
    }
    let version = cursor.read_u32(path)?;
    if version != GGUF_VERSION {
        return Err(HybridError::UnsupportedFormat(format!(
            "unsupported GGUF version {version}; expected {GGUF_VERSION}"
        )));
    }
    let tensor_count = read_limited_count(cursor, path, "tensor_count")?;
    let kv_count = read_limited_count(cursor, path, "kv_count")?;
    Ok((tensor_count, kv_count))
}

fn read_limited_count(cursor: &mut GgufCursor, path: &str, label: &str) -> Result<usize> {
    let raw = cursor.read_u64(path)?;
    if raw > 100_000u64 {
        return Err(HybridError::UnsupportedFormat(format!(
            "{label} {raw} exceeds maximum allowed 100000"
        )));
    }
    Ok(raw as usize)
}

fn read_metadata_kv(
    cursor: &mut GgufCursor,
    kv_count: usize,
    path: &str,
    architecture: &mut String,
) -> Result<(usize, Option<u32>, HashMap<String, u64>)> {
    let mut alignment = 32usize;
    let mut file_type = None;
    let mut numerics = HashMap::new();
    for _ in 0..kv_count {
        read_one_kv(
            cursor,
            path,
            &mut alignment,
            &mut file_type,
            architecture,
            &mut numerics,
        )?;
    }
    Ok((alignment, file_type, numerics))
}

fn read_one_kv(
    cursor: &mut GgufCursor,
    path: &str,
    alignment: &mut usize,
    file_type: &mut Option<u32>,
    architecture: &mut String,
    numerics: &mut HashMap<String, u64>,
) -> Result<()> {
    let key = cursor.read_string(path)?;
    let value_type = cursor.read_u32(path)?;
    match key.as_str() {
        "general.alignment" => {
            *alignment = cursor.read_numeric_as_u32(value_type, path)? as usize;
            Ok(())
        }
        "general.file_type" => {
            let value = cursor.read_numeric_as_u32(value_type, path)?;
            *file_type = Some(value);
            numerics.insert("general.file_type".into(), value as u64);
            Ok(())
        }
        "general.architecture" => {
            *architecture = cursor.read_string(path)?;
            Ok(())
        }
        _ => read_unknown(cursor, path, key, value_type, numerics),
    }
}

fn read_unknown(
    cursor: &mut GgufCursor,
    path: &str,
    key: String,
    value_type: u32,
    numerics: &mut HashMap<String, u64>,
) -> Result<()> {
    if let Some(value) = cursor.read_numeric_value(value_type, path)? {
        numerics.insert(key, value);
    } else if value_type == GGUF_VALUE_TYPE_STRING {
        cursor.read_string(path)?;
    } else {
        cursor.skip_value(value_type, path)?;
    }
    Ok(())
}

fn read_tensor_info(cursor: &mut GgufCursor, path: &str) -> Result<(String, GgufTensorInfo)> {
    let name = cursor.read_string(path)?;
    let (dims, n_elements) = read_tensor_dimensions(cursor, path, &name)?;
    let ggml_type = cursor.read_u32(path)?;
    let relative_offset = cursor.read_u64(path)? as usize;
    Ok((
        name,
        GgufTensorInfo {
            dims,
            ggml_type,
            relative_offset,
            absolute_offset: 0,
            n_elements,
        },
    ))
}

fn read_tensor_dimensions(
    cursor: &mut GgufCursor,
    path: &str,
    name: &str,
) -> Result<(Vec<usize>, usize)> {
    let n_dims_raw = cursor.read_u32(path)? as usize;
    if n_dims_raw > 8 {
        return Err(HybridError::UnsupportedFormat(format!(
            "tensor '{name}' has {n_dims_raw} dims, which exceeds maximum allowed 8"
        )));
    }
    let mut dims = Vec::with_capacity(n_dims_raw);
    for _ in 0..n_dims_raw {
        dims.push(cursor.read_u64(path)? as usize);
    }
    let n_elements = dims
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("tensor '{name}' element count overflow"),
        })?;
    Ok((dims, n_elements))
}

impl MappedGgufCheckpoint {
    pub(super) fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    pub(super) fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub(super) fn find_first_tensor_with_suffix(&self, suffix: &str) -> Option<&str> {
        let mut matches: Vec<&str> = self
            .tensors
            .keys()
            .map(String::as_str)
            .filter(|name| name.ends_with(suffix))
            .collect();
        matches.sort_unstable_by_key(|name| dequant::tensor_block_sort_key(name));
        matches.into_iter().next()
    }

    pub(super) fn tensor_info<'a>(&'a self, name: &str, path: &str) -> Result<&'a GgufTensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| HybridError::MissingTensor {
                name: name.to_owned(),
                path: path.to_owned(),
            })
    }

    pub(super) fn f32_tensor<'a>(&'a self, name: &str, path: &str) -> Result<&'a [f32]> {
        let info = self.tensor_info(name, path)?;
        if info.ggml_type != GGML_TYPE_F32 {
            return Err(HybridError::UnsupportedFormat(format!(
                "tensor '{name}' must be F32, got ggml_type={}",
                info.ggml_type
            )));
        }

        let start = info.absolute_offset;
        let end = start + info.n_elements * std::mem::size_of::<f32>();
        if end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{name}' extends beyond mapped file"),
            });
        }

        // SAFETY: `start` is a valid byte offset into the mmap and `end` is
        // checked against `mmap.len()` above.  F32 alignment is guaranteed
        // because GGUF aligns all tensor data to at least 32 bytes (enforced
        // by the `alignment` field parsed from the file header).  The returned
        // slice borrows `self` for lifetime `'a`, keeping the mmap alive.
        let ptr = unsafe { self.mmap.as_ptr().add(start) as *const f32 };
        Ok(unsafe { slice::from_raw_parts(ptr, info.n_elements) })
    }

    pub(super) fn u16_tensor_values(
        &self,
        info: &GgufTensorInfo,
        path: &str,
        tensor_name: &str,
    ) -> Result<Vec<u16>> {
        let byte_start = info.absolute_offset;
        let byte_end = byte_start + info.n_elements * 2;
        if byte_end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' extends beyond mapped file"),
            });
        }
        Ok(self.mmap[byte_start..byte_end]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect())
    }

    fn row_bytes<'a>(
        &'a self,
        info: &GgufTensorInfo,
        row_idx: usize,
        path: &str,
        tensor_name: &str,
    ) -> Result<&'a [u8]> {
        let n_rows = info.dims.get(1).copied().unwrap_or(0);
        if row_idx >= n_rows {
            return Err(HybridError::InputLengthMismatch {
                expected: n_rows,
                got: row_idx,
            });
        }

        let row_size = dequant::tensor_row_size(info.ggml_type, info.dims[0])?;
        let overflow = || HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("tensor '{tensor_name}' row offset overflow"),
        };
        let start = info
            .absolute_offset
            .checked_add(row_idx.checked_mul(row_size).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
        let end = start.checked_add(row_size).ok_or_else(overflow)?;
        if end > self.mmap.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: format!("tensor '{tensor_name}' row extends beyond mapped file"),
            });
        }
        Ok(&self.mmap[start..end])
    }
}

impl<'a> GgufCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact(&mut self, len: usize, path: &str) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.to_owned(),
                reason: "cursor overflow".into(),
            })?;
        if end > self.bytes.len() {
            return Err(HybridError::ModelLoad {
                path: path.to_owned(),
                reason: "unexpected EOF while parsing GGUF".into(),
            });
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_u8(&mut self, path: &str) -> Result<u8> {
        Ok(self.read_exact(1, path)?[0])
    }

    fn read_u16(&mut self, path: &str) -> Result<u16> {
        let bytes = self.read_exact(2, path)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self, path: &str) -> Result<u32> {
        let bytes = self.read_exact(4, path)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("slice length is fixed"),
        ))
    }

    fn read_u64(&mut self, path: &str) -> Result<u64> {
        let bytes = self.read_exact(8, path)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("slice length is fixed"),
        ))
    }

    fn read_i8(&mut self, path: &str) -> Result<i8> {
        let bytes = self.read_exact(1, path)?;
        Ok(i8::from_le_bytes([bytes[0]]))
    }

    fn read_i16(&mut self, path: &str) -> Result<i16> {
        let bytes = self.read_exact(2, path)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i32(&mut self, path: &str) -> Result<i32> {
        let bytes = self.read_exact(4, path)?;
        Ok(i32::from_le_bytes(
            bytes.try_into().expect("slice length is fixed"),
        ))
    }

    fn read_i64(&mut self, path: &str) -> Result<i64> {
        let bytes = self.read_exact(8, path)?;
        Ok(i64::from_le_bytes(
            bytes.try_into().expect("slice length is fixed"),
        ))
    }

    fn read_string(&mut self, path: &str) -> Result<String> {
        let len = self.read_u64(path)? as usize;
        let bytes = self.read_exact(len, path)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| HybridError::ModelLoad {
            path: path.to_owned(),
            reason: format!("invalid UTF-8 in GGUF string: {e}"),
        })
    }

    fn read_numeric_as_u32(&mut self, value_type: u32, path: &str) -> Result<u32> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8
            | GGUF_VALUE_TYPE_UINT16
            | GGUF_VALUE_TYPE_UINT32
            | GGUF_VALUE_TYPE_UINT64 => {
                let value = self.read_unsigned_value(value_type, path)?;
                u32::try_from(value).map_err(|_| {
                    HybridError::UnsupportedFormat(format!(
                        "GGUF unsigned value {value} out of range for u32"
                    ))
                })
            }
            GGUF_VALUE_TYPE_INT8
            | GGUF_VALUE_TYPE_INT16
            | GGUF_VALUE_TYPE_INT32
            | GGUF_VALUE_TYPE_INT64 => {
                let value = self.read_signed_value(value_type, path)?;
                u32::try_from(value).map_err(|_| {
                    HybridError::UnsupportedFormat(format!(
                        "GGUF signed value {value} out of range for u32"
                    ))
                })
            }
            _ => Err(HybridError::UnsupportedFormat(format!(
                "GGUF numeric conversion from type {value_type} is not supported"
            ))),
        }
    }

    fn read_numeric_value(&mut self, value_type: u32, path: &str) -> Result<Option<u64>> {
        match value_type {
            GGUF_VALUE_TYPE_STRING | GGUF_VALUE_TYPE_ARRAY => Ok(None),
            GGUF_VALUE_TYPE_UINT8
            | GGUF_VALUE_TYPE_UINT16
            | GGUF_VALUE_TYPE_UINT32
            | GGUF_VALUE_TYPE_UINT64 => Ok(Some(self.read_unsigned_value(value_type, path)?)),
            GGUF_VALUE_TYPE_INT8
            | GGUF_VALUE_TYPE_INT16
            | GGUF_VALUE_TYPE_INT32
            | GGUF_VALUE_TYPE_INT64 => {
                let value = self.read_signed_value(value_type, path)?;
                let unsigned = u64::try_from(value).map_err(|_| {
                    HybridError::UnsupportedFormat(format!(
                        "GGUF signed value {value} cannot be represented as unsigned metadata"
                    ))
                })?;
                Ok(Some(unsigned))
            }
            GGUF_VALUE_TYPE_BOOL => Ok(Some(self.read_u8(path)? as u64)),
            GGUF_VALUE_TYPE_FLOAT32 => Ok(Some(self.read_u32(path)? as u64)),
            GGUF_VALUE_TYPE_FLOAT64 => Ok(Some(self.read_u64(path)?)),
            other => Err(HybridError::UnsupportedFormat(format!(
                "unsupported GGUF value type {other}"
            ))),
        }
    }

    fn read_unsigned_value(&mut self, value_type: u32, path: &str) -> Result<u64> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 => Ok(self.read_u8(path)? as u64),
            GGUF_VALUE_TYPE_UINT16 => Ok(self.read_u16(path)? as u64),
            GGUF_VALUE_TYPE_UINT32 => Ok(self.read_u32(path)? as u64),
            GGUF_VALUE_TYPE_UINT64 => Ok(self.read_u64(path)?),
            _ => unreachable!(),
        }
    }

    fn read_signed_value(&mut self, value_type: u32, path: &str) -> Result<i64> {
        match value_type {
            GGUF_VALUE_TYPE_INT8 => Ok(i64::from(self.read_i8(path)?)),
            GGUF_VALUE_TYPE_INT16 => Ok(i64::from(self.read_i16(path)?)),
            GGUF_VALUE_TYPE_INT32 => Ok(i64::from(self.read_i32(path)?)),
            GGUF_VALUE_TYPE_INT64 => Ok(self.read_i64(path)?),
            _ => unreachable!(),
        }
    }

    fn skip_value(&mut self, value_type: u32, path: &str) -> Result<()> {
        match value_type {
            GGUF_VALUE_TYPE_STRING => {
                self.read_string(path)?;
            }
            GGUF_VALUE_TYPE_ARRAY => self.skip_array(path)?,
            _ => self.skip_fixed_value(value_type, path)?,
        }
        Ok(())
    }

    fn skip_fixed_value(&mut self, value_type: u32, path: &str) -> Result<()> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 | GGUF_VALUE_TYPE_INT8 | GGUF_VALUE_TYPE_BOOL => {
                self.read_exact(1, path)?;
            }
            GGUF_VALUE_TYPE_UINT16 | GGUF_VALUE_TYPE_INT16 => {
                self.read_exact(2, path)?;
            }
            GGUF_VALUE_TYPE_UINT32 | GGUF_VALUE_TYPE_INT32 | GGUF_VALUE_TYPE_FLOAT32 => {
                self.read_exact(4, path)?;
            }
            GGUF_VALUE_TYPE_UINT64 | GGUF_VALUE_TYPE_INT64 | GGUF_VALUE_TYPE_FLOAT64 => {
                self.read_exact(8, path)?;
            }
            other => {
                return Err(HybridError::UnsupportedFormat(format!(
                    "unsupported GGUF value type {other}"
                )));
            }
        }
        Ok(())
    }

    fn skip_array(&mut self, path: &str) -> Result<()> {
        let nested_type = self.read_u32(path)?;
        let len = self.read_u64(path)? as usize;
        for _ in 0..len {
            self.skip_value(nested_type, path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::HybridError;
    use crate::moe::test_fixtures::*;
    use crate::moe::*;
    use crate::types::EMBEDDING_DIM;

    #[test]
    fn parse_checkpoint_layout_reads_integer_value_types() {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 0);
        push_u64(&mut out, 10);
        push_kv_raw(
            &mut out,
            "general.alignment",
            GGUF_VALUE_TYPE_UINT16,
            &32u16.to_le_bytes(),
        );
        push_kv_i8(&mut out, "general.file_type", 1);
        push_kv_string(&mut out, "general.architecture", "olmoe");
        push_kv_raw(
            &mut out,
            "custom.u8",
            GGUF_VALUE_TYPE_UINT8,
            &7u8.to_le_bytes(),
        );
        push_kv_raw(
            &mut out,
            "custom.u16",
            GGUF_VALUE_TYPE_UINT16,
            &1234u16.to_le_bytes(),
        );
        push_kv_raw(
            &mut out,
            "custom.u64",
            GGUF_VALUE_TYPE_UINT64,
            &42u64.to_le_bytes(),
        );
        push_kv_i8(&mut out, "custom.i8", 5);
        push_kv_i16(&mut out, "custom.i16", 1000);
        push_kv_i32(&mut out, "custom.i32", 50000);
        push_kv_i64(&mut out, "custom.i64", 1_000_000);

        let parsed = parse_checkpoint_layout(&out, "test").unwrap();
        assert_eq!(parsed.metadata.architecture, "olmoe");
        assert_eq!(parsed.metadata.numeric("custom.u8"), Some(7));
        assert_eq!(parsed.metadata.numeric("custom.u16"), Some(1234));
        assert_eq!(parsed.metadata.numeric("custom.u64"), Some(42));
        assert_eq!(parsed.metadata.numeric("custom.i8"), Some(5));
        assert_eq!(parsed.metadata.numeric("custom.i16"), Some(1000));
        assert_eq!(parsed.metadata.numeric("custom.i32"), Some(50000));
        assert_eq!(parsed.metadata.numeric("custom.i64"), Some(1_000_000));
    }

    #[test]
    fn parse_checkpoint_layout_reads_misc_value_types() {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 0);
        push_u64(&mut out, 6);
        push_kv_u32(&mut out, "general.alignment", 32);
        push_kv_u32(&mut out, "general.file_type", 0);
        push_kv_string(&mut out, "general.architecture", "olmoe");
        push_kv_bool(&mut out, "custom.bool", true);
        push_kv_f32(&mut out, "custom.float32", 1.5);
        push_kv_f64(&mut out, "custom.float64", 2.5);
        push_kv_string(&mut out, "custom.string", "hello");
        push_kv_array_u32(&mut out, "custom.array", &[1, 2]);

        let parsed = parse_checkpoint_layout(&out, "test").unwrap();
        assert_eq!(parsed.metadata.numeric("custom.bool"), Some(1));
        assert!(parsed.metadata.numeric("custom.string").is_none());
        assert!(parsed.metadata.numeric("custom.array").is_none());
    }

    #[test]
    fn parse_checkpoint_layout_rejects_bad_magic() {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUS");
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 0);
        push_u64(&mut out, 0);
        let err = parse_checkpoint_layout(&out, "test").unwrap_err();
        assert!(err.to_string().contains("unrecognised model magic bytes"));
    }

    #[test]
    fn parse_checkpoint_layout_rejects_bad_version() {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, 2);
        push_u64(&mut out, 0);
        push_u64(&mut out, 0);
        let err = parse_checkpoint_layout(&out, "test").unwrap_err();
        assert!(err.to_string().contains("unsupported GGUF version"));
    }

    #[test]
    fn parse_checkpoint_layout_rejects_excessive_tensor_count() {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 100_001);
        push_u64(&mut out, 0);
        let err = parse_checkpoint_layout(&out, "test").unwrap_err();
        assert!(
            err.to_string()
                .contains("tensor_count 100001 exceeds maximum allowed 100000")
        );
    }

    #[test]
    fn parse_checkpoint_layout_rejects_excessive_kv_count() {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 0);
        push_u64(&mut out, 100_001);
        let err = parse_checkpoint_layout(&out, "test").unwrap_err();
        assert!(
            err.to_string()
                .contains("kv_count 100001 exceeds maximum allowed 100000")
        );
    }

    #[test]
    fn parse_checkpoint_layout_rejects_alignment_overflow() {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 0);
        push_u64(&mut out, 1);
        push_kv_raw(
            &mut out,
            "general.alignment",
            GGUF_VALUE_TYPE_UINT64,
            &u64::MAX.to_le_bytes(),
        );
        let err = parse_checkpoint_layout(&out, "test").unwrap_err();
        assert!(err.to_string().contains("out of range for u32"));
    }

    #[test]
    fn parse_checkpoint_layout_rejects_negative_alignment() {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 0);
        push_u64(&mut out, 1);
        push_kv_i32(&mut out, "general.alignment", -1);
        let err = parse_checkpoint_layout(&out, "test").unwrap_err();
        assert!(err.to_string().contains("out of range for u32"));
    }

    #[test]
    fn parse_checkpoint_layout_rejects_negative_unknown_numeric() {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 0);
        push_u64(&mut out, 1);
        push_kv_i32(&mut out, "custom.negative", -1);
        let err = parse_checkpoint_layout(&out, "test").unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot be represented as unsigned metadata")
        );
    }

    #[test]
    fn parse_checkpoint_layout_rejects_unsupported_value_type() {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 0);
        push_u64(&mut out, 1);
        push_kv_raw(&mut out, "custom.bad", 99, &[]);
        let err = parse_checkpoint_layout(&out, "test").unwrap_err();
        assert!(err.to_string().contains("unsupported GGUF value type 99"));
    }

    #[test]
    fn parse_checkpoint_layout_rejects_too_many_tensor_dims() {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 1);
        push_u64(&mut out, 0);
        push_string(&mut out, "t");
        push_u32(&mut out, 9);
        for _ in 0..9 {
            push_u64(&mut out, 1);
        }
        push_u32(&mut out, GGML_TYPE_F32);
        push_u64(&mut out, 0);
        let err = parse_checkpoint_layout(&out, "test").unwrap_err();
        assert!(err.to_string().contains("exceeds maximum allowed 8"));
    }

    #[test]
    fn parse_checkpoint_layout_rejects_tensor_element_overflow() {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 1);
        push_u64(&mut out, 0);
        push_string(&mut out, "t");
        push_u32(&mut out, 2);
        push_u64(&mut out, usize::MAX as u64);
        push_u64(&mut out, 2);
        push_u32(&mut out, GGML_TYPE_F32);
        push_u64(&mut out, 0);
        let err = parse_checkpoint_layout(&out, "test").unwrap_err();
        assert!(err.to_string().contains("element count overflow"));
    }

    fn build_token_file(ggml_type: u32, d1: usize, payload: Vec<u8>) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, 1);
        push_u64(&mut out, 0);
        push_string(&mut out, "token_embd.weight");
        push_u32(&mut out, 2);
        push_u64(&mut out, EMBEDDING_DIM as u64);
        push_u64(&mut out, d1 as u64);
        push_u32(&mut out, ggml_type);
        push_u64(&mut out, 0);
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend_from_slice(&payload);
        out
    }

    #[test]
    fn probe_and_map_extracts_f32_token_embedding() {
        let d1 = 3usize;
        let payload = vec![0u8; EMBEDDING_DIM * d1 * std::mem::size_of::<f32>()];
        let bytes = build_token_file(GGML_TYPE_F32, d1, payload);
        let path = write_temp_file(&bytes, "token_f32");
        let (_, mut checkpoint) = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();

        assert!(checkpoint.has_tensor("token_embd.weight"));
        assert_eq!(
            checkpoint.find_first_tensor_with_suffix("embd.weight"),
            Some("token_embd.weight")
        );
        assert!(
            checkpoint
                .find_first_tensor_with_suffix("missing")
                .is_none()
        );

        let info = checkpoint
            .tensor_info("token_embd.weight", path.to_str().unwrap())
            .unwrap();
        assert_eq!(info.dims, vec![EMBEDDING_DIM, d1]);
        assert_eq!(info.ggml_type, GGML_TYPE_F32);

        let weights = checkpoint
            .f32_tensor("token_embd.weight", path.to_str().unwrap())
            .unwrap();
        assert_eq!(weights.len(), EMBEDDING_DIM * d1);

        let embedding = checkpoint
            .extract_token_embedding("token_embd.weight", path.to_str().unwrap(), 0)
            .unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        assert!(embedding.iter().all(|&v| v == 0.0));

        let err = checkpoint
            .extract_token_embedding("token_embd.weight", path.to_str().unwrap(), d1)
            .unwrap_err();
        assert!(
            matches!(err, HybridError::InputLengthMismatch { expected, got } if expected == d1 && got == d1)
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn probe_and_map_extracts_f16_token_embedding() {
        let d1 = 3usize;
        let payload = vec![0u8; EMBEDDING_DIM * d1 * 2];
        let bytes = build_token_file(GGML_TYPE_F16, d1, payload);
        let path = write_temp_file(&bytes, "token_f16");
        let (_, mut checkpoint) = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();

        let info = checkpoint
            .tensor_info("token_embd.weight", path.to_str().unwrap())
            .unwrap()
            .clone();
        assert_eq!(info.ggml_type, GGML_TYPE_F16);

        let err = checkpoint
            .f32_tensor("token_embd.weight", path.to_str().unwrap())
            .unwrap_err();
        assert!(err.to_string().contains("must be F32"));

        let values = checkpoint
            .u16_tensor_values(&info, path.to_str().unwrap(), "token_embd.weight")
            .unwrap();
        assert_eq!(values.len(), EMBEDDING_DIM * d1);

        let embedding = checkpoint
            .extract_token_embedding("token_embd.weight", path.to_str().unwrap(), 1)
            .unwrap();
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        assert!(embedding.iter().all(|&v| v == 0.0));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn extract_token_embedding_rejects_unsupported_ggml_type() {
        let bytes = build_token_file(GGML_TYPE_IQ3_S, 2, vec![]);
        let path = write_temp_file(&bytes, "token_iq3");
        let (_, mut checkpoint) = probe_and_map_checkpoint(path.to_str().unwrap()).unwrap();

        let err = checkpoint
            .extract_token_embedding("token_embd.weight", path.to_str().unwrap(), 0)
            .unwrap_err();
        assert!(err.to_string().contains("IQ3_S token embeddings"));

        let err = checkpoint
            .extract_token_embedding("token_embd.weight", path.to_str().unwrap(), 5)
            .unwrap_err();
        assert!(
            matches!(err, HybridError::InputLengthMismatch { expected, got } if expected == 2 && got == 5)
        );

        let _ = std::fs::remove_file(path);
    }
}
