// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::tensor::Tensor;
use crate::tensor::finite::softmax_row;
use crate::tensor::ops::{batched_matmul, causal_mask, matmul};
use serde::{Deserialize, Serialize};

/// Multi-head self-attention (replaces candle-nn attention layers).
///
/// Implements scaled dot-product attention with causal masking.
/// Weights are stored as dense matrices; no external framework needed.
#[derive(Clone, Serialize, Deserialize)]
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

impl MultiHeadAttention {
    pub fn new(dim: usize, num_heads: usize) -> Self {
        assert_eq!(dim % num_heads, 0);
        let head_dim = dim / num_heads;
        let scale = 1.0 / (dim as f32).sqrt();
        Self {
            num_heads,
            head_dim,
            dim,
            wq: Tensor::randn(&[dim, dim], 0.0, scale),
            wb_q: Tensor::zeros(&[1, dim]),
            wk: Tensor::randn(&[dim, dim], 0.0, scale),
            wb_k: Tensor::zeros(&[1, dim]),
            wv: Tensor::randn(&[dim, dim], 0.0, scale),
            wb_v: Tensor::zeros(&[1, dim]),
            wo: Tensor::randn(&[dim, dim], 0.0, scale),
            wb_o: Tensor::zeros(&[1, dim]),
        }
    }

    /// Forward pass: x is [seq_len, dim] → output [seq_len, dim]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let seq_len = x.shape()[0];
        let _dim = self.dim;
        let nh = self.num_heads;
        let hd = self.head_dim;

        // Project Q, K, V: [seq_len, dim] × [dim, dim] → [seq_len, dim]
        let q = matmul(x, &self.wq).add(&broadcast_bias(&self.wb_q, seq_len));
        let k = matmul(x, &self.wk).add(&broadcast_bias(&self.wb_k, seq_len));
        let v = matmul(x, &self.wv).add(&broadcast_bias(&self.wb_v, seq_len));

        // Reshape to [num_heads, seq_len, head_dim] for batched attention
        let q = reshape_heads(&q, seq_len, nh, hd);
        let k = reshape_heads(&k, seq_len, nh, hd);
        let v = reshape_heads(&v, seq_len, nh, hd);

        // Scaled dot-product attention per head
        // scores = Q × K^T / sqrt(head_dim)  → [nh, seq_len, seq_len]
        let kt = transpose_last_two(&k); // [nh, hd, seq_len]
        let scale = 1.0 / (hd as f32).sqrt();
        let scores = batched_matmul(&q, &kt).scale(scale);

        // Apply causal mask
        let mask = causal_mask(seq_len);
        let scores = apply_mask_batched(&scores, &mask, nh);

        // Softmax per row
        let attn = batched_softmax(&scores, nh, seq_len);

        // attn × V → [nh, seq_len, hd]
        let ctx = batched_matmul(&attn, &v);

        // Reshape back to [seq_len, dim]
        let ctx = merge_heads(&ctx, seq_len, nh, hd);

        // Output projection
        matmul(&ctx, &self.wo).add(&broadcast_bias(&self.wb_o, seq_len))
    }

    /// Returns flattened parameter count for this layer.
    pub fn param_count(&self) -> usize {
        4 * self.dim * self.dim + 4 * self.dim
    }
}

// ── Helper functions ─────────────────────────────────────────────────

fn broadcast_bias(bias: &Tensor, seq_len: usize) -> Tensor {
    // bias is [1, dim], tile to [seq_len, dim]
    let dim = bias.shape()[1];
    let bd = bias.data();
    let mut out = vec![0.0f32; seq_len * dim];
    for r in 0..seq_len {
        out[r * dim..(r + 1) * dim].copy_from_slice(bd);
    }
    Tensor::from_vec(out, &[seq_len, dim])
}

/// [seq_len, dim] → [num_heads, seq_len, head_dim]
fn reshape_heads(t: &Tensor, seq_len: usize, nh: usize, hd: usize) -> Tensor {
    let d = t.data();
    let mut out = vec![0.0f32; nh * seq_len * hd];
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
    let mut out = vec![0.0f32; seq_len * dim];
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
    let mut out = vec![0.0f32; b * n * m];
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
/// Uses the crate finite-value policy so a fully masked (`-Inf`) row is
/// uniform rather than NaN, and NaN logits never rank through `f32::max`.
fn batched_softmax(t: &Tensor, nh: usize, seq: usize) -> Tensor {
    let d = t.data();
    let mut out = vec![0.0f32; nh * seq * seq];
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
