// SPDX-License-Identifier: MIT OR Apache-2.0

use super::expect_shape;
use crate::error::{CortexError, Result, unwrap_compat};
use crate::tensor::Tensor;
use crate::tensor::finite::softmax_row;
use crate::tensor::ops::{try_batched_matmul, try_causal_mask, try_matmul};

/// Multi-head self-attention (replaces candle-nn attention layers).
///
/// Implements scaled dot-product attention with causal masking.
/// Weights are stored as dense matrices; no external framework needed.
#[derive(Clone)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub dim: usize,
    // Projection weights [dim, dim]
    pub wq: Tensor,
    pub wb_q: Tensor, // bias
    pub wk: Tensor,
    pub wb_k: Tensor,
    pub wv: Tensor,
    pub wb_v: Tensor,
    pub wo: Tensor,
    pub wb_o: Tensor,
}

reference_serde!(
    MultiHeadAttention,
    AttentionWire {
        num_heads: usize,
        head_dim: usize,
        dim: usize,
        wq: Tensor,
        wb_q: Tensor,
        wk: Tensor,
        wb_k: Tensor,
        wv: Tensor,
        wb_v: Tensor,
        wo: Tensor,
        wb_o: Tensor,
    }
);

impl MultiHeadAttention {
    pub(super) fn validate_wire(&self) -> Result<()> {
        if self.dim == 0
            || self.num_heads == 0
            || self.num_heads.checked_mul(self.head_dim) != Some(self.dim)
        {
            return Err(CortexError::InvalidConfig(
                "attention head dimensions disagree".into(),
            ));
        }
        for weight in [&self.wq, &self.wk, &self.wv, &self.wo] {
            expect_shape(weight, &[self.dim, self.dim])?;
        }
        for bias in [&self.wb_q, &self.wb_k, &self.wb_v, &self.wb_o] {
            expect_shape(bias, &[1, self.dim])?;
        }
        Ok(())
    }
    /// # Panics
    ///
    /// Panics if `num_heads` is zero or does not divide `dim`. Prefer
    /// [`MultiHeadAttention::try_new`] in new code.
    pub fn new(dim: usize, num_heads: usize) -> Self {
        unwrap_compat(Self::try_new(dim, num_heads), "MultiHeadAttention::new")
    }

    /// Fallible constructor: returns [`CortexError::InvalidConfig`] when
    /// `num_heads` is zero or does not evenly divide `dim`.
    pub fn try_new(dim: usize, num_heads: usize) -> Result<Self> {
        if dim == 0 || num_heads == 0 || !dim.is_multiple_of(num_heads) {
            return Err(CortexError::InvalidConfig(format!(
                "MultiHeadAttention: dim ({dim}) and num_heads ({num_heads}) must be nonzero, and num_heads must divide dim"
            )));
        }
        let head_dim = dim / num_heads;
        let scale = 1.0 / (dim as f32).sqrt();
        Ok(Self {
            num_heads,
            head_dim,
            dim,
            wq: Tensor::try_randn(&[dim, dim], 0.0, scale)?,
            wb_q: Tensor::try_zeros(&[1, dim])?,
            wk: Tensor::try_randn(&[dim, dim], 0.0, scale)?,
            wb_k: Tensor::try_zeros(&[1, dim])?,
            wv: Tensor::try_randn(&[dim, dim], 0.0, scale)?,
            wb_v: Tensor::try_zeros(&[1, dim])?,
            wo: Tensor::try_randn(&[dim, dim], 0.0, scale)?,
            wb_o: Tensor::try_zeros(&[1, dim])?,
        })
    }

    /// Forward pass: x is [seq_len, dim] → output [seq_len, dim]
    ///
    /// # Panics
    ///
    /// Panics on rank/shape mismatches. Prefer [`Self::try_forward`].
    pub fn forward(&self, x: &Tensor) -> Tensor {
        unwrap_compat(self.try_forward(x), "MultiHeadAttention::forward")
    }

    /// Fallible forward pass: validates `x` is `[seq_len, dim]` and that all
    /// intermediate shapes are consistent before computing.
    pub fn try_forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 2 {
            return Err(CortexError::RankMismatch {
                expected: 2,
                got: x.ndim(),
            });
        }
        let seq_len = x.shape()[0];
        let nh = self.num_heads;
        let hd = self.head_dim;
        if x.shape()[1] != self.dim {
            return Err(CortexError::DimMismatch {
                op: "MultiHeadAttention::try_forward",
                axis: 1,
                expected: self.dim,
                got: x.shape()[1],
            });
        }
        // Fields are `pub`; a literal-constructed module can violate the
        // `nh * hd == dim` and `[1, dim]` bias invariants that `try_new`
        // establishes. The head-reshape helpers below index on those
        // invariants, so check them before any slicing.
        if nh == 0 || nh.checked_mul(hd) != Some(self.dim) {
            return Err(CortexError::InvalidConfig(format!(
                "MultiHeadAttention: num_heads ({nh}) * head_dim ({hd}) != dim ({})",
                self.dim
            )));
        }
        for bias in [&self.wb_q, &self.wb_k, &self.wb_v, &self.wb_o] {
            if bias.shape() != [1, self.dim] {
                return Err(CortexError::ShapeMismatch {
                    expected: vec![1, self.dim],
                    got: bias.shape().to_vec(),
                });
            }
        }

        // Project Q, K, V: [seq_len, dim] × [dim, dim] → [seq_len, dim]
        let q = try_matmul(x, &self.wq)?.try_add(&broadcast_bias(&self.wb_q, seq_len)?)?;
        let k = try_matmul(x, &self.wk)?.try_add(&broadcast_bias(&self.wb_k, seq_len)?)?;
        let v = try_matmul(x, &self.wv)?.try_add(&broadcast_bias(&self.wb_v, seq_len)?)?;

        // Reshape to [num_heads, seq_len, head_dim] for batched attention
        let q = reshape_heads(&q, seq_len, nh, hd);
        let k = reshape_heads(&k, seq_len, nh, hd);
        let v = reshape_heads(&v, seq_len, nh, hd);

        // Scaled dot-product attention per head
        // scores = Q × K^T / sqrt(head_dim)  → [nh, seq_len, seq_len]
        let kt = transpose_last_two(&k); // [nh, hd, seq_len]
        let scale = 1.0 / (hd as f32).sqrt();
        let scores = try_batched_matmul(&q, &kt)?.scale(scale);

        // Apply causal mask
        let mask = try_causal_mask(seq_len)?;
        let scores = apply_mask_batched(&scores, &mask, nh);

        // Softmax per row
        let attn = batched_softmax(&scores, nh, seq_len);

        // attn × V → [nh, seq_len, hd]
        let ctx = try_batched_matmul(&attn, &v)?;

        // Reshape back to [seq_len, dim]
        let ctx = merge_heads(&ctx, seq_len, nh, hd);

        // Output projection
        try_matmul(&ctx, &self.wo)?.try_add(&broadcast_bias(&self.wb_o, seq_len)?)
    }

    /// Returns flattened parameter count for this layer.
    pub fn param_count(&self) -> usize {
        4 * self.dim * self.dim + 4 * self.dim
    }
}

// ── Helper functions ─────────────────────────────────────────────────

fn broadcast_bias(bias: &Tensor, seq_len: usize) -> Result<Tensor> {
    // bias is [1, dim], tile to [seq_len, dim]
    let dim = bias.shape()[1];
    let out_len = seq_len
        .checked_mul(dim)
        .ok_or_else(|| CortexError::SizeOverflow {
            shape: vec![seq_len, dim],
        })?;
    let bd = bias.data();
    if bd.len() != dim {
        return Err(CortexError::ShapeMismatch {
            expected: vec![1, dim],
            got: vec![bd.len()],
        });
    }
    let mut out = vec![0.0f32; out_len];
    for r in 0..seq_len {
        out[r * dim..(r + 1) * dim].copy_from_slice(bd);
    }
    Tensor::try_from_vec(out, &[seq_len, dim])
}

/// [seq_len, dim] → [num_heads, seq_len, head_dim]
fn reshape_heads(t: &Tensor, seq_len: usize, nh: usize, hd: usize) -> Tensor {
    let d = t.data();
    // t.numel() == seq_len * nh * hd for a valid [seq_len, dim] input, and
    // is already known representable because t exists — computing it as a
    // fresh product could overflow mid-expression for degenerate dims.
    let mut out = vec![0.0f32; t.numel()];
    for s in 0..seq_len {
        for h in 0..nh {
            for i in 0..hd {
                out[h * seq_len * hd + s * hd + i] = d[s * (nh * hd) + h * hd + i];
            }
        }
    }
    Tensor::from_vec(out, &[nh, seq_len, hd])
}

/// [num_heads, seq_len, head_dim] → [seq_len, dim]
fn merge_heads(t: &Tensor, seq_len: usize, nh: usize, hd: usize) -> Tensor {
    let d = t.data();
    let dim = nh * hd;
    let mut out = vec![0.0f32; t.numel()];
    for s in 0..seq_len {
        for h in 0..nh {
            for i in 0..hd {
                out[s * dim + h * hd + i] = d[h * seq_len * hd + s * hd + i];
            }
        }
    }
    Tensor::from_vec(out, &[seq_len, dim])
}

/// Transpose last two dims of a 3-D tensor: [B, M, N] → [B, N, M]
fn transpose_last_two(t: &Tensor) -> Tensor {
    let b = t.shape()[0];
    let m = t.shape()[1];
    let n = t.shape()[2];
    let d = t.data();
    let mut out = vec![0.0f32; t.numel()];
    for bi in 0..b {
        for i in 0..m {
            for j in 0..n {
                out[bi * n * m + j * m + i] = d[bi * m * n + i * n + j];
            }
        }
    }
    Tensor::from_vec(out, &[b, n, m])
}

/// Apply 2-D causal mask to each head in [nh, seq, seq] scores
fn apply_mask_batched(scores: &Tensor, mask: &Tensor, nh: usize) -> Tensor {
    let seq = mask.shape()[0];
    let sd = scores.data();
    let md = mask.data();
    let mut out = sd.to_vec();
    for h in 0..nh {
        let off = h * seq * seq;
        for i in 0..seq * seq {
            out[off + i] += md[i];
        }
    }
    Tensor::from_vec(out, scores.shape())
}

/// Row-wise softmax on each [seq, seq] slice of a [nh, seq, seq] tensor.
///
/// Uses [`crate::tensor::finite::softmax_row`] (RM-1354): NaN logits poison the
/// row; a single `+Inf` is one-hot; several `+Inf` values split evenly; an
/// all-`-Inf` row is uniform `1/seq` over the **entire** row vector.
///
/// Causal masking is applied before this step. Typical rows keep at least one
/// finite logit on indices `0..=query`, so masked future keys stay at zero
/// probability after softmax. If every logit in the row is `-Inf`, the shared
/// row policy still assigns `1/seq` to each index—including future masked
/// slots—by design. Do not rely on softmax alone to zero those positions in
/// that edge case.
fn batched_softmax(t: &Tensor, nh: usize, seq: usize) -> Tensor {
    let d = t.data();
    let mut out = vec![0.0f32; t.numel()];
    if seq == 0 {
        return Tensor::from_vec(out, t.shape());
    }
    for h in 0..nh {
        for r in 0..seq {
            let off = h * seq * seq + r * seq;
            softmax_row(&d[off..off + seq], &mut out[off..off + seq]);
        }
    }
    Tensor::from_vec(out, t.shape())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_shapes() {
        let attn = MultiHeadAttention::new(64, 4);
        let x = Tensor::randn(&[8, 64], 0.0, 0.1);
        let out = attn.forward(&x);
        assert_eq!(out.shape(), &[8, 64]);
    }
}
