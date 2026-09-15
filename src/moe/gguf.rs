// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF format constants.
//!
//! **Frozen for parser work (see #8 / #47).** New GGML type ids and dtype
//! handling belong in `rmems/engram-parser`, not here. Parser/dtype freeze
//! holds until #47 can wrap `parse_checkpoint_layout` around engram-parser
//! 0.2.0 (blocked on engram-parser#45). Do not add Q6_K / IQ3_* here. Note
//! for any future port: GGUF wire type 31 is the historical `Q4_0_4_4`
//! layout, not IQ3_M — see the corinth-canal reference (`src/moe/ggml.rs`)
//! before adding quant ids.

pub(crate) const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub(crate) const GGUF_VERSION: u32 = 3;
pub(crate) const GGML_TYPE_F32: u32 = 0;
pub(crate) const GGML_TYPE_F16: u32 = 1;
pub(crate) const GGML_TYPE_Q8_0: u32 = 8;
pub(crate) const GGML_TYPE_Q5_K: u32 = 13;
pub(crate) const GGML_TYPE_IQ3_S: u32 = 21;
pub(crate) const GGUF_VALUE_TYPE_UINT8: u32 = 0;
pub(crate) const GGUF_VALUE_TYPE_INT8: u32 = 1;
pub(crate) const GGUF_VALUE_TYPE_UINT16: u32 = 2;
pub(crate) const GGUF_VALUE_TYPE_INT16: u32 = 3;
pub(crate) const GGUF_VALUE_TYPE_UINT32: u32 = 4;
pub(crate) const GGUF_VALUE_TYPE_INT32: u32 = 5;
pub(crate) const GGUF_VALUE_TYPE_FLOAT32: u32 = 6;
pub(crate) const GGUF_VALUE_TYPE_BOOL: u32 = 7;
pub(crate) const GGUF_VALUE_TYPE_STRING: u32 = 8;
pub(crate) const GGUF_VALUE_TYPE_ARRAY: u32 = 9;
pub(crate) const GGUF_VALUE_TYPE_UINT64: u32 = 10;
pub(crate) const GGUF_VALUE_TYPE_INT64: u32 = 11;
pub(crate) const GGUF_VALUE_TYPE_FLOAT64: u32 = 12;
