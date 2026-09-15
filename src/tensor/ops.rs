// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Tensor, checked_numel, try_compute_strides};
use crate::error::{CortexError, Result, unwrap_compat};

/// Matrix multiply: [M×K] × [K×N] → [M×N]
///
/// Uses a cache-friendly tiled loop. When the `gpu` feature is enabled,
/// this will dispatch to a CUDA kernel instead.
///
/// # Panics
///
/// Panics on rank or inner-dimension mismatch, or if the output size
/// overflows `usize`. Prefer [`try_matmul`] in new code. This wrapper is
/// retained for pre-1.0 source compatibility.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    unwrap_compat(try_matmul(a, b), "matmul")
}

/// Fallible matrix multiply: [M×K] × [K×N] → [M×N].
///
/// Validates ranks, inner dimensions, and output size before allocating.
pub fn try_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    expect_rank(a, 2)?;
    expect_rank(b, 2)?;
    let (m, k1) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    if k1 != k2 {
        return Err(CortexError::MatmulDim { m, k1, k2, n });
    }
    let out_shape = [m, n];
    let mut out = alloc_zeros(&out_shape)?;
    if out.is_empty() {
        return Tensor::try_from_vec(out, &out_shape);
    }

    let ad = a.data();
    let bd = b.data();

    // Tiled matmul for better cache behaviour
    const TILE: usize = 32;
    for i0 in (0..m).step_by(TILE) {
        for j0 in (0..n).step_by(TILE) {
            for p0 in (0..k1).step_by(TILE) {
                let i_end = (i0 + TILE).min(m);
                let j_end = (j0 + TILE).min(n);
                let p_end = (p0 + TILE).min(k1);
                for i in i0..i_end {
                    for p in p0..p_end {
                        let a_ip = ad[i * k1 + p];
                        for j in j0..j_end {
                            out[i * n + j] += a_ip * bd[p * n + j];
                        }
                    }
                }
            }
        }
    }
    Tensor::try_from_vec(out, &out_shape)
}

/// Batched matmul: [B×M×K] × [B×K×N] → [B×M×N]
/// If `b` is 2-D, broadcasts across batches.
///
/// # Panics
///
/// Panics on rank, batch, or inner-dimension mismatch, or if the output size
/// overflows `usize`. Prefer [`try_batched_matmul`] in new code.
pub fn batched_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    unwrap_compat(try_batched_matmul(a, b), "batched_matmul")
}

/// Fallible batched matmul: [B×M×K] × [B×K×N] → [B×M×N].
///
/// If both operands are 2-D, this is equivalent to [`try_matmul`]. If `b` is
/// 2-D, it is broadcast across the batch of `a`. Validates every dimension
/// and the output size before allocating.
pub fn try_batched_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.ndim() == 2 && b.ndim() == 2 {
        return try_matmul(a, b);
    }
    if a.ndim() != 3 {
        return Err(CortexError::RankMismatch {
            expected: 3,
            got: a.ndim(),
        });
    }

    let batch = a.shape()[0];
    let m = a.shape()[1];
    let k = a.shape()[2];

    let (b_is_batched, n) = match b.ndim() {
        3 => {
            expect_dim("batched_matmul", 0, batch, b.shape()[0])?;
            expect_dim("batched_matmul", 1, k, b.shape()[1])?;
            (true, b.shape()[2])
        }
        2 => {
            expect_dim("batched_matmul", 0, k, b.shape()[0])?;
            (false, b.shape()[1])
        }
        got => {
            return Err(CortexError::RankMismatch { expected: 3, got });
        }
    };

    let out_shape = [batch, m, n];
    let mut out = alloc_zeros(&out_shape)?;
    if out.is_empty() {
        return Tensor::try_from_vec(out, &out_shape);
    }
    let ad = a.data();
    let bd = b.data();

    for bi in 0..batch {
        // Output numel being representable does not imply `bi * m * k` is:
        // a zero axis can make `batch * m * n` fit while a leading product
        // of the other extents overflows (debug panic / release wrap).
        let a_off = checked_mul3(bi, m, k, a.shape())?;
        let b_off = if b_is_batched {
            checked_mul3(bi, k, n, b.shape())?
        } else {
            0
        };
        let o_off = checked_mul3(bi, m, n, &out_shape)?;
        for i in 0..m {
            for p in 0..k {
                let a_ip = ad[a_off + i * k + p];
                for j in 0..n {
                    out[o_off + i * n + j] += a_ip * bd[b_off + p * n + j];
                }
            }
        }
    }
    Tensor::try_from_vec(out, &out_shape)
}

/// Layer normalization over the last axis.
/// Returns (normalized, mean, rstd) for backward pass if needed.
///
/// # Panics
///
/// Panics if `x` is not rank-2, the last axis is zero-width, `eps` is not
/// positive and finite, or `weight`/`bias` lengths do not match the last
/// axis. Prefer [`try_layer_norm`] in new code.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
    unwrap_compat(try_layer_norm(x, weight, bias, eps), "layer_norm")
}

/// Fallible layer normalization over the last axis of a rank-2 tensor.
pub fn try_layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
    expect_rank(x, 2)?;
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    if cols == 0 {
        return Err(CortexError::ZeroWidth {
            op: "layer_norm",
            axis: 1,
        });
    }
    check_eps(eps)?;
    expect_dim("layer_norm", 0, cols, weight.numel())?;
    expect_dim("layer_norm", 0, cols, bias.numel())?;

    let out_shape = [rows, cols];
    let mut out = alloc_zeros(&out_shape)?;
    let xd = x.data();
    let wd = weight.data();
    let bd = bias.data();

    for r in 0..rows {
        let off = r * cols;
        let row = &xd[off..off + cols];

        let mean: f32 = row.iter().sum::<f32>() / cols as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let rstd = 1.0 / (var + eps).sqrt();

        for c in 0..cols {
            out[off + c] = (row[c] - mean) * rstd * wd[c] + bd[c];
        }
    }
    Tensor::try_from_vec(out, &out_shape)
}

/// RMS normalization (used by LLaMA-family models).
///
/// # Panics
///
/// Panics if `x` is not rank-2, the last axis is zero-width, `eps` is not
/// positive and finite, or `weight` length does not match the last axis.
/// Prefer [`try_rms_norm`] in new code.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
    unwrap_compat(try_rms_norm(x, weight, eps), "rms_norm")
}

/// Fallible RMS normalization over the last axis of a rank-2 tensor.
pub fn try_rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    expect_rank(x, 2)?;
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    if cols == 0 {
        return Err(CortexError::ZeroWidth {
            op: "rms_norm",
            axis: 1,
        });
    }
    check_eps(eps)?;
    expect_dim("rms_norm", 0, cols, weight.numel())?;

    let out_shape = [rows, cols];
    let mut out = alloc_zeros(&out_shape)?;
    let xd = x.data();
    let wd = weight.data();

    for r in 0..rows {
        let off = r * cols;
        let row = &xd[off..off + cols];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let rstd = 1.0 / (ms + eps).sqrt();
        for c in 0..cols {
            out[off + c] = row[c] * rstd * wd[c];
        }
    }
    Tensor::try_from_vec(out, &out_shape)
}

/// Embedding lookup: [vocab_size, dim] indexed by token ids → [seq_len, dim]
///
/// # Panics
///
/// Panics if `table` is not rank-2, a token id is out of vocabulary, or the
/// output size overflows `usize`. Prefer [`try_embedding`] in new code.
pub fn embedding(table: &Tensor, ids: &[u32]) -> Tensor {
    unwrap_compat(try_embedding(table, ids), "embedding")
}

/// Fallible embedding lookup: [vocab_size, dim] indexed by token ids → [seq_len, dim].
///
/// Out-of-vocabulary ids return [`CortexError::TokenIndex`] without slicing.
pub fn try_embedding(table: &Tensor, ids: &[u32]) -> Result<Tensor> {
    expect_rank(table, 2)?;
    let vocab = table.shape()[0];
    let dim = table.shape()[1];
    for &id in ids {
        let index = id as usize;
        if index >= vocab {
            return Err(CortexError::TokenIndex {
                index,
                vocab_size: vocab,
            });
        }
    }

    let seq_len = ids.len();
    let out_shape = [seq_len, dim];
    let mut out = alloc_zeros(&out_shape)?;
    let td = table.data();
    for (i, &id) in ids.iter().enumerate() {
        let src_off = id as usize * dim;
        out[i * dim..(i + 1) * dim].copy_from_slice(&td[src_off..src_off + dim]);
    }
    Tensor::try_from_vec(out, &out_shape)
}

/// Causal attention mask: upper triangle = -inf, lower triangle + diagonal = 0.
pub fn causal_mask(seq_len: usize) -> Tensor {
    let out_len = unwrap_compat(checked_numel(&[seq_len, seq_len]), "causal_mask");
    let mut data = vec![0.0f32; out_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, &[seq_len, seq_len])
}

fn expect_rank(t: &Tensor, expected: usize) -> Result<()> {
    if t.ndim() != expected {
        Err(CortexError::RankMismatch {
            expected,
            got: t.ndim(),
        })
    } else {
        Ok(())
    }
}

fn expect_dim(op: &'static str, axis: usize, expected: usize, got: usize) -> Result<()> {
    if expected != got {
        Err(CortexError::DimMismatch {
            op,
            axis,
            expected,
            got,
        })
    } else {
        Ok(())
    }
}

fn check_eps(eps: f32) -> Result<()> {
    if eps.is_finite() && eps > 0.0 {
        Ok(())
    } else {
        Err(CortexError::InvalidEpsilon { eps })
    }
}

/// Allocate an output buffer only after numel and strides are representable.
fn alloc_zeros(shape: &[usize]) -> Result<Vec<f32>> {
    let numel = checked_numel(shape)?;
    let _ = try_compute_strides(shape)?;
    Ok(vec![0.0; numel])
}

fn checked_mul3(a: usize, b: usize, c: usize, err_shape: &[usize]) -> Result<usize> {
    a.checked_mul(b)
        .and_then(|v| v.checked_mul(c))
        .ok_or_else(|| CortexError::SizeOverflow {
            shape: err_shape.to_vec(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2x2() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), &[2, 2]);
        // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
        let tried = try_matmul(&a, &b).unwrap();
        assert_eq!(tried.data(), c.data());
        assert_eq!(tried.shape(), c.shape());
    }

    #[test]
    fn test_layer_norm() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let w = Tensor::ones(&[3]);
        let b = Tensor::zeros(&[3]);
        let y = layer_norm(&x, &w, &b, 1e-5);
        // Each row should have mean ≈ 0
        let row0_mean: f32 = y.data()[0..3].iter().sum::<f32>() / 3.0;
        assert!(row0_mean.abs() < 1e-4);
        let tried = try_layer_norm(&x, &w, &b, 1e-5).unwrap();
        assert_eq!(tried.data(), y.data());
    }

    #[test]
    fn test_rms_norm_matches_wrapper() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let w = Tensor::ones(&[3]);
        let y = rms_norm(&x, &w, 1e-5);
        let tried = try_rms_norm(&x, &w, 1e-5).unwrap();
        assert_eq!(y.data(), tried.data());
        assert_eq!(y.shape(), &[2, 3]);
    }

    #[test]
    fn test_causal_mask() {
        let m = causal_mask(3);
        assert_eq!(m.data()[0], 0.0); // [0,0]
        assert!(m.data()[1].is_infinite()); // [0,1] = -inf
        assert_eq!(m.data()[4], 0.0); // [1,1]
    }

    #[test]
    fn test_embedding() {
        let table = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]);
        let ids = vec![2u32, 0u32];
        let out = embedding(&table, &ids);
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data(), &[0.5, 0.6, 0.1, 0.2]);
        assert_eq!(try_embedding(&table, &ids).unwrap().data(), out.data());
    }

    #[test]
    fn try_embedding_accepts_empty_ids() {
        let table = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]);
        let out = try_embedding(&table, &[]).unwrap();
        assert_eq!(out.shape(), &[0, 2]);
        assert_eq!(out.numel(), 0);
    }

    #[test]
    fn try_ops_error_table() {
        let rank1 = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let a2 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b_inner = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let batch_a = Tensor::from_vec(vec![0.0; 2 * 2 * 3], &[2, 2, 3]);
        let batch_b_mismatch = Tensor::from_vec(vec![0.0; 3 * 3 * 4], &[3, 3, 4]);
        let table = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let w = Tensor::ones(&[3]);
        let b = Tensor::zeros(&[3]);
        let zero_width = Tensor::try_from_vec(vec![], &[2, 0]).unwrap();
        let w0 = Tensor::try_from_vec(vec![], &[0]).unwrap();

        struct Case {
            name: &'static str,
            err: CortexError,
            check: fn(&CortexError) -> bool,
        }
        let cases = [
            Case {
                name: "matmul rank",
                err: try_matmul(&rank1, &a2).unwrap_err(),
                check: |e| {
                    matches!(
                        e,
                        CortexError::RankMismatch {
                            expected: 2,
                            got: 1
                        }
                    )
                },
            },
            Case {
                name: "matmul inner dim",
                err: try_matmul(&a2, &b_inner).unwrap_err(),
                check: |e| {
                    matches!(
                        e,
                        CortexError::MatmulDim {
                            m: 2,
                            k1: 2,
                            k2: 3,
                            n: 2
                        }
                    )
                },
            },
            Case {
                name: "batched rank",
                err: try_batched_matmul(&rank1, &a2).unwrap_err(),
                check: |e| {
                    matches!(
                        e,
                        CortexError::RankMismatch {
                            expected: 3,
                            got: 1
                        }
                    )
                },
            },
            Case {
                name: "batched batch mismatch",
                err: try_batched_matmul(&batch_a, &batch_b_mismatch).unwrap_err(),
                check: |e| {
                    matches!(
                        e,
                        CortexError::DimMismatch {
                            op: "batched_matmul",
                            axis: 0,
                            expected: 2,
                            got: 3
                        }
                    )
                },
            },
            Case {
                name: "oov token",
                err: try_embedding(&table, &[2]).unwrap_err(),
                check: |e| {
                    matches!(
                        e,
                        CortexError::TokenIndex {
                            index: 2,
                            vocab_size: 2
                        }
                    )
                },
            },
            Case {
                name: "oov u32::MAX",
                err: try_embedding(&table, &[u32::MAX]).unwrap_err(),
                check: |e| {
                    matches!(
                        e,
                        CortexError::TokenIndex {
                            index,
                            vocab_size: 2
                        } if *index == u32::MAX as usize
                    )
                },
            },
            Case {
                name: "layer_norm rank",
                err: try_layer_norm(&rank1, &w, &b, 1e-5).unwrap_err(),
                check: |e| {
                    matches!(
                        e,
                        CortexError::RankMismatch {
                            expected: 2,
                            got: 1
                        }
                    )
                },
            },
            Case {
                name: "layer_norm weight dim",
                err: try_layer_norm(&x, &Tensor::ones(&[2]), &b, 1e-5).unwrap_err(),
                check: |e| {
                    matches!(
                        e,
                        CortexError::DimMismatch {
                            op: "layer_norm",
                            expected: 3,
                            got: 2,
                            ..
                        }
                    )
                },
            },
            Case {
                name: "layer_norm zero width",
                err: try_layer_norm(&zero_width, &w0, &w0, 1e-5).unwrap_err(),
                check: |e| {
                    matches!(
                        e,
                        CortexError::ZeroWidth {
                            op: "layer_norm",
                            axis: 1
                        }
                    )
                },
            },
            Case {
                name: "rms_norm zero width",
                err: try_rms_norm(&zero_width, &w0, 1e-5).unwrap_err(),
                check: |e| {
                    matches!(
                        e,
                        CortexError::ZeroWidth {
                            op: "rms_norm",
                            axis: 1
                        }
                    )
                },
            },
            Case {
                name: "non-positive eps",
                err: try_layer_norm(&x, &w, &b, 0.0).unwrap_err(),
                check: |e| matches!(e, CortexError::InvalidEpsilon { eps } if *eps == 0.0),
            },
            Case {
                name: "negative eps",
                err: try_rms_norm(&x, &w, -1e-5).unwrap_err(),
                check: |e| matches!(e, CortexError::InvalidEpsilon { eps } if *eps < 0.0),
            },
            Case {
                name: "nan eps",
                err: try_layer_norm(&x, &w, &b, f32::NAN).unwrap_err(),
                check: |e| matches!(e, CortexError::InvalidEpsilon { eps } if eps.is_nan()),
            },
            Case {
                name: "inf eps",
                err: try_rms_norm(&x, &w, f32::INFINITY).unwrap_err(),
                check: |e| matches!(e, CortexError::InvalidEpsilon { eps } if eps.is_infinite()),
            },
        ];
        for case in cases {
            assert!((case.check)(&case.err), "{}: {}", case.name, case.err);
        }
    }

    #[test]
    fn try_matmul_empty_output_skips_huge_extent_loop() {
        let a = Tensor::try_from_vec(vec![], &[usize::MAX, 0]).unwrap();
        let b = Tensor::try_from_vec(vec![], &[0, 0]).unwrap();
        let out = try_matmul(&a, &b).unwrap();
        assert_eq!(out.shape(), &[usize::MAX, 0]);
        assert_eq!(out.numel(), 0);
    }

    #[test]
    fn try_batched_matmul_empty_output_uses_checked_offsets() {
        let a = Tensor::try_from_vec(vec![], &[3, usize::MAX, 0]).unwrap();
        let b = Tensor::try_from_vec(vec![], &[3, 0, 0]).unwrap();
        let out = try_batched_matmul(&a, &b).unwrap();
        assert_eq!(out.shape(), &[3, usize::MAX, 0]);
        assert_eq!(out.numel(), 0);
    }

    #[test]
    fn try_batched_matmul_zero_inner_dim_is_all_zeros() {
        let a = Tensor::try_from_vec(vec![], &[2, 2, 0]).unwrap();
        let b = Tensor::try_from_vec(vec![], &[2, 0, 3]).unwrap();
        let out = try_batched_matmul(&a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 2, 3]);
        assert_eq!(out.data(), &[0.0; 12]);
    }

    #[test]
    fn try_matmul_output_overflow_without_allocation() {
        // A well-formed tensor can still request an unrepresentable product
        // when paired with another huge extent. Synthetic shapes use a zero
        // axis so the operands themselves are cheap to construct.
        let a = Tensor::try_from_vec(vec![], &[usize::MAX, 0]).unwrap();
        let b = Tensor::try_from_vec(vec![], &[0, 2]).unwrap();
        let err = try_matmul(&a, &b).unwrap_err();
        assert!(matches!(
            err,
            CortexError::SizeOverflow { shape } if shape == [usize::MAX, 2]
        ));
    }

    #[test]
    fn try_batched_matmul_broadcast_and_empty_batch() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let out = try_batched_matmul(&a, &b).unwrap();
        assert_eq!(out.shape(), &[2, 2, 2]);
        // Identity broadcast: each batch slice is unchanged.
        assert_eq!(out.data(), a.data());

        let empty_a = Tensor::try_from_vec(vec![], &[0, 2, 3]).unwrap();
        let empty_b = Tensor::try_from_vec(vec![], &[0, 3, 4]).unwrap();
        let empty_out = try_batched_matmul(&empty_a, &empty_b).unwrap();
        assert_eq!(empty_out.shape(), &[0, 2, 4]);
        assert_eq!(empty_out.numel(), 0);
    }
}
