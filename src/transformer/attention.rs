// SPDX-License-Identifier: MIT OR Apache-2.0

use super::expect_shape;
use crate::error::{CortexError, Result, unwrap_compat};
use crate::tensor::Tensor;
use crate::tensor::finite::softmax_row;
use crate::tensor::ops::{try_batched_matmul, try_matmul};

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

        // Causal softmax per row: only the visible prefix 0..=r is normalized;
        // future slots are exactly zero (no additive mask needed).
        let attn = causal_batched_softmax(&scores, nh, seq_len);

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

/// Causal row-wise softmax on each [seq, seq] slice of a [nh, seq, seq] tensor.
///
/// For query row `r`, softmax is taken over the visible prefix `0..=r` only
/// (via [`crate::tensor::finite::softmax_row`], RM-1354 policy: NaN poisons the
/// prefix; `+Inf` is one-hot / split evenly; an all-`-Inf` prefix is uniform
/// `1/(r+1)`). Future slots `r+1..seq` are written as exactly `0.0` and their
/// scores are never read, so no score value (including `-Inf` overflow of the
/// whole prefix) can move probability mass onto a future token (issue #94).
fn causal_batched_softmax(t: &Tensor, nh: usize, seq: usize) -> Tensor {
    let d = t.data();
    let mut out = vec![0.0f32; t.numel()];
    if seq == 0 {
        return Tensor::from_vec(out, t.shape());
    }
    for h in 0..nh {
        for r in 0..seq {
            let off = h * seq * seq + r * seq;
            softmax_row(&d[off..off + r + 1], &mut out[off..off + r + 1]);
        }
    }
    Tensor::from_vec(out, t.shape())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(dim: usize, sign: f32) -> Tensor {
        let mut d = vec![0.0f32; dim * dim];
        for i in 0..dim {
            d[i * dim + i] = sign;
        }
        Tensor::from_vec(d, &[dim, dim])
    }

    /// `dim=2, heads=1`, `wq=-I`, `wk=wv=wo=I`, zero biases (issue #94).
    fn overflow_attention() -> MultiHeadAttention {
        let mut attn = MultiHeadAttention::new(2, 1);
        attn.wq = identity(2, -1.0);
        attn.wk = identity(2, 1.0);
        attn.wv = identity(2, 1.0);
        attn.wo = identity(2, 1.0);
        attn
    }

    #[test]
    fn test_attention_shapes() {
        let attn = MultiHeadAttention::new(64, 4);
        let x = Tensor::randn(&[8, 64], 0.0, 0.1);
        let out = attn.forward(&x);
        assert_eq!(out.shape(), &[8, 64]);
    }

    #[test]
    fn all_neg_inf_visible_prefix_does_not_leak_future_tokens() {
        // Token 0's self-score is -1e40, which overflows to -Inf, so its whole
        // visible prefix is -Inf. Its output must not depend on token 1.
        let attn = overflow_attention();
        let a = attn.forward(&Tensor::from_vec(vec![1e20, 0.0, 0.0, 7.0], &[2, 2]));
        let b = attn.forward(&Tensor::from_vec(vec![1e20, 0.0, 0.0, -7.0], &[2, 2]));
        assert_eq!(&a.data()[..2], &b.data()[..2]);
        // Uniform over the single visible slot: row 0 attends to itself only.
        assert_eq!(&a.data()[..2], &[1e20, 0.0]);
    }

    #[test]
    fn batched_softmax_all_neg_inf_row_zeroes_future_slots() {
        let ninf = f32::NEG_INFINITY;
        let scores = Tensor::from_vec(
            vec![ninf, ninf, ninf, ninf, ninf, ninf, ninf, ninf, ninf],
            &[1, 3, 3],
        );
        let p = causal_batched_softmax(&scores, 1, 3);
        let d = p.data();
        assert!(d.iter().all(|w| !w.is_nan()));
        // Row 0: all weight on itself; future slots exactly zero.
        assert_eq!(&d[0..3], &[1.0, 0.0, 0.0]);
        // Row 1: uniform over the visible prefix only.
        assert_eq!(&d[3..6], &[0.5, 0.5, 0.0]);
        // Row 2: uniform over the full (visible) row.
        for w in &d[6..9] {
            assert!((w - 1.0 / 3.0).abs() < 1e-7);
        }
    }

    #[test]
    fn batched_softmax_partial_mask_matches_row_softmax() {
        let ninf = f32::NEG_INFINITY;
        // Future slots are -Inf as an additive causal mask would leave them;
        // row 1 also has a -Inf inside its visible prefix; row 2 is plain finite.
        let scores = Tensor::from_vec(
            vec![0.5, ninf, ninf, ninf, 1.0, ninf, 0.1, 0.2, 0.3],
            &[1, 3, 3],
        );
        let p = causal_batched_softmax(&scores, 1, 3);
        let d = p.data();
        assert_eq!(&d[0..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&d[3..6], &[0.0, 1.0, 0.0]);
        let mut expect = [0.0f32; 3];
        softmax_row(&[0.1, 0.2, 0.3], &mut expect);
        assert_eq!(&d[6..9], &expect);
        assert!((d[6..9].iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn causal_attention_output_independent_of_future_tokens() {
        let attn = MultiHeadAttention::new(8, 2);
        let base = Tensor::randn(&[4, 8], 0.0, 0.5);
        let mut changed = base.data().to_vec();
        for v in &mut changed[2 * 8..] {
            *v += 3.0; // perturb tokens 2 and 3
        }
        let changed = Tensor::from_vec(changed, &[4, 8]);
        let a = attn.forward(&base);
        let b = attn.forward(&changed);
        assert_eq!(&a.data()[..2 * 8], &b.data()[..2 * 8]);
    }
}
