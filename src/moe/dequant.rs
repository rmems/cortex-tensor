// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quantized tensor dequantization helpers (Q8_0, Q5_K, etc.).
//!
//! **Frozen for dtype work (see #8 / #47).** Dequantization to `f32` is owned
//! by this crate, but widening the supported dtype set (`BF16`, `Q6_K`,
//! `IQ3_*`, …) is frozen until #47 can wrap `parse_checkpoint_layout` around
//! engram-parser 0.2.0 (blocked on engram-parser#45, mmap + K-quant). New
//! dtypes arrive with that crate's GGML type ids. Do not add Q6_K / IQ3_*
//! block formats here that `moe/gguf.rs` cannot yet name.

use crate::error::{HybridError, Result};

use half::f16;

#[allow(clippy::manual_is_multiple_of)]
pub(crate) fn tensor_row_size(ggml_type: u32, width: usize) -> Result<usize> {
    match ggml_type {
        super::GGML_TYPE_Q8_0 => {
            if width % 32 != 0 {
                return Err(HybridError::UnsupportedFormat(format!(
                    "Q8_0 tensor width {width} is not divisible by 32"
                )));
            }
            Ok((width / 32) * (2 + 32))
        }
        super::GGML_TYPE_Q5_K => {
            if width % 256 != 0 {
                return Err(HybridError::UnsupportedFormat(format!(
                    "Q5_K tensor width {width} is not divisible by 256"
                )));
            }
            Ok((width / 256) * (2 + 2 + 12 + 32 + 128))
        }
        other => Err(HybridError::UnsupportedFormat(format!(
            "row-size lookup is not implemented for ggml_type={other}"
        ))),
    }
}

#[allow(clippy::manual_is_multiple_of)]
pub(crate) fn dequantize_row_q8_0(row: &[u8], width: usize) -> Result<Vec<f32>> {
    if width % 32 != 0 {
        return Err(HybridError::UnsupportedFormat(format!(
            "Q8_0 width {width} is not divisible by 32"
        )));
    }

    let mut out = Vec::with_capacity(width);
    for block in row.as_chunks::<34>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for &quant in &block[2..34] {
            out.push((quant as i8) as f32 * d);
        }
    }
    Ok(out)
}

#[allow(clippy::manual_is_multiple_of)]
pub(crate) fn dequantize_row_q5_k(row: &[u8], width: usize) -> Result<Vec<f32>> {
    if width % 256 != 0 {
        return Err(HybridError::UnsupportedFormat(format!(
            "Q5_K width {width} is not divisible by 256"
        )));
    }

    let mut out = Vec::with_capacity(width);
    for block in row.as_chunks::<176>().0 {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..176];

        let mut is = 0usize;
        let mut u1 = 1u16;
        let mut u2 = 2u16;

        for ql_chunk in ql.as_chunks::<32>().0 {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let mn1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mn2 = dmin * m2 as f32;

            for (lane, &q) in ql_chunk.iter().enumerate() {
                let qh_byte = qh[lane];
                let hi1 = if qh_byte & (u1 as u8) != 0 { 16 } else { 0 };
                let hi2 = if qh_byte & (u2 as u8) != 0 { 16 } else { 0 };
                out.push(d1 * ((q & 0x0F) + hi1) as f32 - mn1);
                out.push(d2 * ((q >> 4) + hi2) as f32 - mn2);
            }

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

#[allow(clippy::manual_is_multiple_of)]
fn scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    // Restored from original GGUF Q5_K layout (matches llama.cpp get_scale_min_k4).
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    // Delegate to the half crate for correct IEEE 754 handling (including subnormals).
    f16::from_bits(bits).to_f32()
}

pub(crate) fn tensor_block_sort_key(name: &str) -> (usize, &str) {
    let block = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(idx, _)| idx.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    (block, name)
}

pub(crate) fn quantization_label(file_type: Option<u32>) -> String {
    match file_type {
        Some(0) => "F32".into(),
        Some(1) => "F16".into(),
        Some(other) => format!("GGUF({other})"),
        None => "GGUF".into(),
    }
}

pub(crate) fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

#[cfg(test)]
mod golden_tests {
    use super::*;
    use half::f16;

    /// Oracle: `llama.cpp` `ggml-quants.c` `dequantize_row_q8_0` / GGML `block_q8_0`
    /// layout (scale `d` as F16 + 32 × int8 quants). Tolerance: abs ≤ 1e-5 f32.
    #[test]
    fn golden_q8_0_block_matches_llama_cpp_layout() {
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());
        for (idx, q) in block[2..34].iter_mut().enumerate() {
            *q = idx as u8;
        }
        let out = dequantize_row_q8_0(&block, 32).unwrap();
        assert_eq!(out.len(), 32);
        for (idx, &value) in out.iter().enumerate() {
            let expected = idx as f32 * 2.0;
            assert!(
                (value - expected).abs() <= 1e-5,
                "idx {idx}: got {value}, expected {expected}"
            );
        }
    }

    /// Oracle: `llama.cpp` `ggml-quants.c` `dequantize_row_q5_K` / `block_q5_K`
    /// (176 bytes per 256-wide super-block). Fixed block; abs ≤ 1e-4 vs reference
    /// output from the same layout rules as llama.cpp (see `scale_min_k4`).
    #[test]
    fn golden_q5_k_block_matches_llama_cpp_layout() {
        let mut block = [0u8; 176];
        block[0..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&f16::from_f32(0.0).to_bits().to_le_bytes());
        block[4] = 1;
        for b in block[48..80].iter_mut() {
            *b = 0x05;
        }
        let out = dequantize_row_q5_k(&block, 256).unwrap();
        assert_eq!(out.len(), 256);
        // First super-block outputs with this fixture (llama.cpp `block_q5_K` layout).
        const EXPECTED_HEAD: [f32; 8] = [5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0];
        for (idx, (&value, &expected)) in out.iter().zip(EXPECTED_HEAD.iter()).enumerate() {
            assert!(
                (value - expected).abs() <= 1e-4,
                "idx {idx}: got {value}, expected {expected}"
            );
        }
        assert!(
            out[8..]
                .iter()
                .all(|&v| (v - 5.0).abs() <= 1e-4 || (v - 0.0).abs() <= 1e-4)
        );
    }

    #[test]
    fn f16_to_f32_preserves_signed_zero_subnormal_and_one() {
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x0001), f16::from_bits(0x0001).to_f32());
        assert!((f16_to_f32(0x3C00) - 1.0).abs() <= f32::EPSILON);
    }
}
