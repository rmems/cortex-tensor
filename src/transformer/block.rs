// SPDX-License-Identifier: MIT OR Apache-2.0

use super::attention::MultiHeadAttention;
use crate::error::{CortexError, Result, unwrap_compat};
use crate::tensor::Tensor;
use crate::tensor::ops::{try_layer_norm, try_matmul};
use serde::{Deserialize, Serialize};

/// Feed-forward network (two linear layers with GELU activation).
#[derive(Clone, Serialize, Deserialize)]
pub struct FeedForward {
    pub w1: Tensor, // [dim, ff_dim]
    pub b1: Tensor, // [1, ff_dim]
    pub w2: Tensor, // [ff_dim, dim]
    pub b2: Tensor, // [1, dim]
}

impl FeedForward {
    /// # Panics
    ///
    /// Panics if a shape allocation overflows. Prefer
    /// [`FeedForward::try_new`].
    pub fn new(dim: usize, ff_dim: usize) -> Self {
        unwrap_compat(Self::try_new(dim, ff_dim), "FeedForward::new")
    }

    /// Fallible constructor: fails on shape/allocation overflow instead of
    /// panicking.
    pub fn try_new(dim: usize, ff_dim: usize) -> Result<Self> {
        let scale = 1.0 / (dim as f32).sqrt();
        Ok(Self {
            w1: Tensor::try_randn(&[dim, ff_dim], 0.0, scale)?,
            b1: Tensor::try_zeros(&[1, ff_dim])?,
            w2: Tensor::try_randn(&[ff_dim, dim], 0.0, scale)?,
            b2: Tensor::try_zeros(&[1, dim])?,
        })
    }

    /// # Panics
    ///
    /// Panics on rank/shape mismatches. Prefer [`Self::try_forward`].
    pub fn forward(&self, x: &Tensor) -> Tensor {
        unwrap_compat(self.try_forward(x), "FeedForward::forward")
    }

    /// Fallible forward pass: `x` must be `[seq_len, dim]` and consistent
    /// with the stored weights.
    pub fn try_forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 2 {
            return Err(CortexError::RankMismatch {
                expected: 2,
                got: x.ndim(),
            });
        }
        let seq_len = x.shape()[0];
        // x @ w1 + b1
        let h = try_matmul(x, &self.w1)?.try_add(&broadcast_row(&self.b1, seq_len)?)?;
        // GELU activation
        let h = h.gelu();
        // h @ w2 + b2
        try_matmul(&h, &self.w2)?.try_add(&broadcast_row(&self.b2, seq_len)?)
    }

    pub fn param_count(&self) -> usize {
        self.w1.numel() + self.b1.numel() + self.w2.numel() + self.b2.numel()
    }
}

/// Single transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
#[derive(Clone, Serialize, Deserialize)]
pub struct TransformerBlock {
    pub attn: MultiHeadAttention,
    pub ffn: FeedForward,
    pub ln1_w: Tensor,
    pub ln1_b: Tensor,
    pub ln2_w: Tensor,
    pub ln2_b: Tensor,
    pub dim: usize,
}

impl TransformerBlock {
    /// # Panics
    ///
    /// Panics if `num_heads` is invalid for `dim`. Prefer
    /// [`TransformerBlock::try_new`].
    pub fn new(dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        unwrap_compat(
            Self::try_new(dim, num_heads, ff_dim),
            "TransformerBlock::new",
        )
    }

    /// Fallible constructor: returns [`CortexError::InvalidConfig`] when
    /// `num_heads` is zero or does not evenly divide `dim`.
    pub fn try_new(dim: usize, num_heads: usize, ff_dim: usize) -> Result<Self> {
        Ok(Self {
            attn: MultiHeadAttention::try_new(dim, num_heads)?,
            ffn: FeedForward::try_new(dim, ff_dim)?,
            ln1_w: Tensor::try_ones(&[dim])?,
            ln1_b: Tensor::try_zeros(&[dim])?,
            ln2_w: Tensor::try_ones(&[dim])?,
            ln2_b: Tensor::try_zeros(&[dim])?,
            dim,
        })
    }

    /// Pre-norm transformer block (GPT-style):
    /// x = x + attn(layernorm(x))
    /// x = x + ffn(layernorm(x))
    ///
    /// # Panics
    ///
    /// Panics on rank/shape mismatches. Prefer [`Self::try_forward`].
    pub fn forward(&self, x: &Tensor) -> Tensor {
        unwrap_compat(self.try_forward(x), "TransformerBlock::forward")
    }

    /// Fallible forward pass.
    pub fn try_forward(&self, x: &Tensor) -> Result<Tensor> {
        let eps = 1e-5;

        // Attention sub-layer with residual
        let normed = try_layer_norm(x, &self.ln1_w, &self.ln1_b, eps)?;
        let attn_out = self.attn.try_forward(&normed)?;
        let x = x.try_add(&attn_out)?;

        // FFN sub-layer with residual
        let normed = try_layer_norm(&x, &self.ln2_w, &self.ln2_b, eps)?;
        let ffn_out = self.ffn.try_forward(&normed)?;
        x.try_add(&ffn_out)
    }

    pub fn param_count(&self) -> usize {
        self.attn.param_count() + self.ffn.param_count() + 4 * self.dim
    }
}

fn broadcast_row(bias: &Tensor, rows: usize) -> Result<Tensor> {
    let cols = bias.numel();
    let out_len = rows
        .checked_mul(cols)
        .ok_or_else(|| CortexError::SizeOverflow {
            shape: vec![rows, cols],
        })?;
    let bd = bias.data();
    let mut out = vec![0.0f32; out_len];
    for r in 0..rows {
        out[r * cols..(r + 1) * cols].copy_from_slice(bd);
    }
    Tensor::try_from_vec(out, &[rows, cols])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_shapes() {
        let block = TransformerBlock::new(64, 4, 256);
        let x = Tensor::randn(&[8, 64], 0.0, 0.02);
        let out = block.forward(&x);
        assert_eq!(out.shape(), &[8, 64]);
    }

    #[test]
    fn test_ffn_shapes() {
        let ffn = FeedForward::new(64, 256);
        let x = Tensor::randn(&[4, 64], 0.0, 0.1);
        let out = ffn.forward(&x);
        assert_eq!(out.shape(), &[4, 64]);
    }
}
