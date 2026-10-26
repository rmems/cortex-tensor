// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{add_row_bias, expect_shape};
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
        super::check_head_config(
            "MultiHeadAttention::validate_wire",
            self.dim,
            self.num_heads,
            0,
            1,
        )?;
        if self.num_heads.checked_mul(self.head_dim) != Some(self.dim) {
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

    /// Fallible constructor: returns [`CortexError::ZeroWidth`] when `dim`
    /// or `num_heads` is zero, and [`CortexError::InvalidConfig`] when
    /// `num_heads` does not evenly divide `dim`.
    pub fn try_new(dim: usize, num_heads: usize) -> Result<Self> {
        Self::try_new_with_rng(dim, num_heads, &mut rand::rng())
    }

    /// Like [`Self::try_new`] but seeded: the same `(dim, num_heads, seed)`
    /// always yields the same weights.
    pub fn try_new_seeded(dim: usize, num_heads: usize, seed: u64) -> Result<Self> {
        Self::try_new_with_rng(dim, num_heads, &mut crate::tensor::seeded_rng(seed))
    }

    /// Like [`Self::try_new`] but draws weights from the caller's RNG.
    pub fn try_new_with_rng<R: rand::Rng + ?Sized>(
        dim: usize,
        num_heads: usize,
        rng: &mut R,
    ) -> Result<Self> {
        super::check_head_config("MultiHeadAttention::try_new", dim, num_heads, 0, 1)?;
        let head_dim = dim / num_heads;
        let scale = 1.0 / (dim as f32).sqrt();
        Ok(Self {
            num_heads,
            head_dim,
            dim,
            wq: Tensor::try_randn_with_rng(&[dim, dim], 0.0, scale, rng)?,
            wb_q: Tensor::try_zeros(&[1, dim])?,
            wk: Tensor::try_randn_with_rng(&[dim, dim], 0.0, scale, rng)?,
            wb_k: Tensor::try_zeros(&[1, dim])?,
            wv: Tensor::try_randn_with_rng(&[dim, dim], 0.0, scale, rng)?,
            wb_v: Tensor::try_zeros(&[1, dim])?,
            wo: Tensor::try_randn_with_rng(&[dim, dim], 0.0, scale, rng)?,
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
        let project = |w: &Tensor, b: &Tensor| -> Result<Tensor> {
            let mut out = try_matmul(x, w)?;
            add_row_bias(&mut out, b)?;
            Ok(out)
        };
        let q = project(&self.wq, &self.wb_q)?;
        let k = project(&self.wk, &self.wb_k)?;
        let v = project(&self.wv, &self.wb_v)?;

        // Reshape to [num_heads, seq_len, head_dim] for batched attention
        let q = reshape_heads(&q, seq_len, nh, hd);
        let k = reshape_heads(&k, seq_len, nh, hd);
        let v = reshape_heads(&v, seq_len, nh, hd);

        // Scaled dot-product attention per head
        // scores = Q × K^T / sqrt(head_dim)  → [nh, seq_len, seq_len]
        let kt = transpose_last_two(&k); // [nh, hd, seq_len]
        let scale = 1.0 / (hd as f32).sqrt();
        let mut scores = try_batched_matmul(&q, &kt)?;

        // Scale + causal softmax per row, in place on the matmul output: only
        // the visible prefix 0..=r is scaled and normalized; future slots are
        // exactly zero (no mask tensor needed).
        causal_scale_softmax_in_place(&mut scores, scale, nh, seq_len);
        let attn = scores;

        // attn × V → [nh, seq_len, hd], accumulated over the visible prefix only
        // so a non-finite future V row cannot reach earlier rows via 0 × NaN.
        let ctx = causal_context(&attn, &v, nh, seq_len, hd);

        // Reshape back to [seq_len, dim]
        let ctx = merge_heads(&ctx, seq_len, nh, hd);

        // Output projection
        let mut out = try_matmul(&ctx, &self.wo)?;
        add_row_bias(&mut out, &self.wb_o)?;
        Ok(out)
    }

    /// Returns flattened parameter count for this layer.
    ///
    /// Summed from the `numel()` of the stored tensors (saturating), so it
    /// tracks the `pub` weight fields and cannot overflow.
    pub fn param_count(&self) -> usize {
        super::sum_numel([
            &self.wq, &self.wb_q, &self.wk, &self.wb_k, &self.wv, &self.wb_v, &self.wo, &self.wb_o,
        ])
    }
}

// ── Helper functions ─────────────────────────────────────────────────

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

/// Fused in-place scale + causal row-wise softmax on each [seq, seq] slice of a
/// [nh, seq, seq] tensor.
///
/// Each visible score is multiplied by `scale` (the same single `x * scale` the
/// separate `Tensor::scale` pass did) and then softmaxed, so results are
/// bit-identical to scale-then-softmax while using one buffer plus an O(seq)
/// scratch row instead of three `[nh, seq, seq]` copies.
///
/// For query row `r`, softmax is taken over the visible prefix `0..=r` only
/// (via [`crate::tensor::finite::softmax_row`], RM-1354 policy: NaN poisons the
/// prefix; `+Inf` is one-hot / split evenly; an all-`-Inf` prefix is uniform
/// `1/(r+1)`). Future slots `r+1..seq` are written as exactly `0.0` and their
/// scores are never read, so no score value (including `-Inf` overflow of the
/// whole prefix) can move probability mass onto a future token (issue #94).
fn causal_scale_softmax_in_place(t: &mut Tensor, scale: f32, nh: usize, seq: usize) {
    assert_eq!(
        t.shape(),
        &[nh, seq, seq],
        "scores must be [heads, seq, seq]"
    );
    if seq == 0 {
        return;
    }
    let mut scratch = vec![0.0f32; seq];
    for slice in t.data_mut().chunks_exact_mut(seq * seq) {
        for (r, row) in slice.chunks_exact_mut(seq).enumerate() {
            let (visible, future) = row.split_at_mut(r + 1);
            for (s, x) in scratch.iter_mut().zip(visible.iter()) {
                *s = *x * scale;
            }
            softmax_row(&scratch[..=r], visible);
            future.fill(0.0);
        }
    }
}

/// Causal `attn × V`: `[nh, seq, seq] × [nh, seq, hd] → [nh, seq, hd]`.
///
/// Row `r` accumulates only value rows `0..=r`. A dense matmul would also
/// multiply the exactly-zero future weights by future `V` rows, and
/// `0.0 * NaN` / `0.0 * ±Inf` is NaN, so a non-finite future value would poison
/// earlier tokens even though their attention weights are exactly zero.
/// For finite inputs the result is bit-identical to the dense product: the
/// skipped terms are `±0.0` additions and the accumulation order is unchanged.
fn causal_context(attn: &Tensor, v: &Tensor, nh: usize, seq: usize, hd: usize) -> Tensor {
    assert_eq!(
        attn.shape(),
        &[nh, seq, seq],
        "weights must be [heads, seq, seq]"
    );
    assert_eq!(
        v.shape(),
        &[nh, seq, hd],
        "values must be [heads, seq, head_dim]"
    );
    let a = attn.data();
    let vd = v.data();
    let mut out = vec![0.0f32; v.numel()];
    for h in 0..nh {
        for r in 0..seq {
            let a_off = h * seq * seq + r * seq;
            let o_off = h * seq * hd + r * hd;
            for p in 0..=r {
                let w = a[a_off + p];
                let v_off = h * seq * hd + p * hd;
                for j in 0..hd {
                    out[o_off + j] += w * vd[v_off + j];
                }
            }
        }
    }
    Tensor::from_vec(out, v.shape())
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

    /// Deterministic pseudo-random values in `[-1, 1)` (LCG), so a failure is
    /// reproducible; the properties below hold for every finite input.
    fn pseudo_random(shape: &[usize], seed: u64, scale: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let mut state = seed;
        let data = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
                (unit * 2.0 - 1.0) * scale
            })
            .collect();
        Tensor::from_vec(data, shape)
    }

    /// Unscaled causal softmax returning a fresh tensor (scale `1.0` is exact).
    fn causal_batched_softmax(t: &Tensor, nh: usize, seq: usize) -> Tensor {
        let mut out = t.clone();
        causal_scale_softmax_in_place(&mut out, 1.0, nh, seq);
        out
    }

    #[test]
    fn fused_scale_softmax_matches_scale_then_softmax() {
        let (nh, seq) = (3, 7);
        let scores = pseudo_random(&[nh, seq, seq], 9, 4.0);
        let scale = 1.0 / 5.0f32.sqrt();
        let mut fused = scores.clone();
        causal_scale_softmax_in_place(&mut fused, scale, nh, seq);
        let reference = causal_batched_softmax(&scores.scale(scale), nh, seq);
        assert_eq!(fused.data(), reference.data());
    }

    #[test]
    fn add_row_bias_matches_tiled_add() {
        let out = pseudo_random(&[5, 4], 11, 1.0);
        let bias = pseudo_random(&[1, 4], 12, 1.0);
        let mut tiled = Vec::new();
        for _ in 0..5 {
            tiled.extend_from_slice(bias.data());
        }
        let expect = out.try_add(&Tensor::from_vec(tiled, &[5, 4])).unwrap();
        let mut got = out.clone();
        add_row_bias(&mut got, &bias).unwrap();
        assert_eq!(got.data(), expect.data());
        let bad = pseudo_random(&[1, 3], 13, 1.0);
        assert!(add_row_bias(&mut got, &bad).is_err());
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
    fn seeded_attention_is_reproducible() {
        let a = MultiHeadAttention::try_new_seeded(16, 4, 5).unwrap();
        let b = MultiHeadAttention::try_new_seeded(16, 4, 5).unwrap();
        let c = MultiHeadAttention::try_new_seeded(16, 4, 6).unwrap();
        assert_eq!(a.wq.data(), b.wq.data());
        assert_eq!(a.wo.data(), b.wo.data());
        assert_ne!(a.wq.data(), c.wq.data());
    }

    #[test]
    fn test_attention_shapes() {
        let attn = MultiHeadAttention::new(64, 4);
        let x = pseudo_random(&[8, 64], 1, 0.1);
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
    fn causal_softmax_all_neg_inf_row_zeroes_future_slots() {
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
    fn causal_softmax_ignores_future_scores_and_matches_row_softmax() {
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
    fn nan_inside_prefix_poisons_only_that_row() {
        // RM-1354: NaN in a visible prefix poisons that row's prefix; future
        // slots stay exactly 0.0 and other rows are untouched.
        let scores = Tensor::from_vec(
            vec![0.5, 9.0, 9.0, f32::NAN, 1.0, 9.0, 0.1, 0.2, 0.3],
            &[1, 3, 3],
        );
        let p = causal_batched_softmax(&scores, 1, 3);
        let d = p.data();
        assert_eq!(&d[0..3], &[1.0, 0.0, 0.0]);
        assert!(d[3].is_nan() && d[4].is_nan());
        assert_eq!(d[5], 0.0);
        assert!(d[6..9].iter().all(|w| w.is_finite()));
    }

    #[test]
    fn plus_inf_inside_prefix_is_one_hot_and_future_inf_is_ignored() {
        // +Inf in the prefix wins outright; +Inf in a future slot (which an
        // additive mask would have turned into NaN via -Inf + Inf) is never read.
        let inf = f32::INFINITY;
        let scores = Tensor::from_vec(
            vec![0.0, inf, inf, 1.0, inf, inf, 0.1, 0.2, 0.3],
            &[1, 3, 3],
        );
        let p = causal_batched_softmax(&scores, 1, 3);
        let d = p.data();
        assert_eq!(&d[0..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&d[3..6], &[0.0, 1.0, 0.0]);
        assert!(d[6..9].iter().all(|w| w.is_finite()));
    }

    #[test]
    fn non_finite_future_value_does_not_poison_earlier_rows() {
        // Identity projections, so v = x. Token 1's value row is NaN / +Inf; its
        // attention weight in row 0 is exactly 0.0, and 0 * NaN must not reach row 0.
        let mut attn = MultiHeadAttention::new(2, 1);
        attn.wq = identity(2, 1.0);
        attn.wk = identity(2, 1.0);
        attn.wv = identity(2, 1.0);
        attn.wo = identity(2, 1.0);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let out = attn.forward(&Tensor::from_vec(vec![1.0, 2.0, bad, bad], &[2, 2]));
            let clean = attn.forward(&Tensor::from_vec(vec![1.0, 2.0, 0.0, 0.0], &[2, 2]));
            assert_eq!(&out.data()[..2], &clean.data()[..2], "bad = {bad}");
            assert!(out.data()[..2].iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn causal_context_matches_dense_matmul_for_finite_inputs() {
        let (nh, seq, hd) = (2, 5, 3);
        let scores = pseudo_random(&[nh, seq, seq], 2, 1.0);
        let attn = causal_batched_softmax(&scores, nh, seq);
        let v = pseudo_random(&[nh, seq, hd], 3, 1.0);
        let dense = try_batched_matmul(&attn, &v).unwrap();
        let causal = causal_context(&attn, &v, nh, seq, hd);
        assert_eq!(dense.data(), causal.data());
    }

    #[test]
    fn causal_attention_output_independent_of_future_tokens() {
        let attn = MultiHeadAttention::new(8, 2);
        let base = pseudo_random(&[4, 8], 4, 0.5);
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
