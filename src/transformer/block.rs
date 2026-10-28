// SPDX-License-Identifier: MIT OR Apache-2.0

use super::attention::MultiHeadAttention;
use super::{add_row_bias, expect_shape};
use crate::error::{CortexError, Result, unwrap_compat};
use crate::tensor::Tensor;
use crate::tensor::ops::{try_layer_norm, try_matmul};

fn check_nonzero_ffn(op: &'static str, dim: usize, ff_dim: usize) -> Result<()> {
    if dim == 0 {
        return Err(CortexError::ZeroWidth { op, axis: 0 });
    }
    if ff_dim == 0 {
        return Err(CortexError::ZeroWidth { op, axis: 1 });
    }
    Ok(())
}

/// Feed-forward network (two linear layers with GELU activation).
#[derive(Clone)]
pub struct FeedForward {
    pub w1: Tensor, // [dim, ff_dim]
    pub b1: Tensor, // [1, ff_dim]
    pub w2: Tensor, // [ff_dim, dim]
    pub b2: Tensor, // [1, dim]
}

reference_serde!(
    FeedForward,
    FeedForwardWire {
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
    }
);

impl FeedForward {
    pub(super) fn validate_wire(&self) -> Result<()> {
        let [dim, ff_dim] = self.w1.shape() else {
            return Err(CortexError::RankMismatch {
                expected: 2,
                got: self.w1.ndim(),
            });
        };
        // Same invariant as `try_new_with_rng`: a zero extent is not a shape
        // error (the tensors are mutually consistent) but it makes the FFN
        // collapse to `x + b2` without any signal, so reject it here too.
        check_nonzero_ffn("FeedForward::validate_wire", *dim, *ff_dim)?;
        expect_shape(&self.b1, &[1, *ff_dim])?;
        expect_shape(&self.w2, &[*ff_dim, *dim])?;
        expect_shape(&self.b2, &[1, *dim])
    }
    /// # Panics
    ///
    /// Panics if `dim` or `ff_dim` is zero or a shape allocation overflows.
    /// Prefer [`FeedForward::try_new`].
    pub fn new(dim: usize, ff_dim: usize) -> Self {
        unwrap_compat(Self::try_new(dim, ff_dim), "FeedForward::new")
    }

    /// Fallible constructor: returns [`CortexError::ZeroWidth`] when `dim`
    /// or `ff_dim` is zero, and fails on shape/allocation overflow instead of
    /// panicking.
    pub fn try_new(dim: usize, ff_dim: usize) -> Result<Self> {
        Self::try_new_with_rng(dim, ff_dim, &mut rand::rng())
    }

    /// Like [`Self::try_new`] but seeded for reproducible weights.
    pub fn try_new_seeded(dim: usize, ff_dim: usize, seed: u64) -> Result<Self> {
        Self::try_new_with_rng(dim, ff_dim, &mut crate::tensor::seeded_rng(seed))
    }

    /// Like [`Self::try_new`] but draws weights from the caller's RNG.
    ///
    /// `w1` is scaled by `1/sqrt(dim)` and `w2` by `1/sqrt(ff_dim)` (each
    /// matmul's fan-in), keeping output variance near that of the input.
    pub fn try_new_with_rng<R: rand::Rng + ?Sized>(
        dim: usize,
        ff_dim: usize,
        rng: &mut R,
    ) -> Result<Self> {
        check_nonzero_ffn("FeedForward::try_new", dim, ff_dim)?;
        let scale1 = 1.0 / (dim as f32).sqrt();
        let scale2 = 1.0 / (ff_dim as f32).sqrt();
        Ok(Self {
            w1: Tensor::try_randn_with_rng(&[dim, ff_dim], 0.0, scale1, rng)?,
            b1: Tensor::try_zeros(&[1, ff_dim])?,
            w2: Tensor::try_randn_with_rng(&[ff_dim, dim], 0.0, scale2, rng)?,
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
        // Same failure class (hidden width != dim) as attention and the block:
        // one variant, checked before the matmul can report `MatmulDim`.
        let want = self.w1.shape().first().copied().unwrap_or(0);
        if x.shape()[1] != want {
            return Err(CortexError::DimMismatch {
                op: "FeedForward::try_forward",
                axis: 1,
                expected: want,
                got: x.shape()[1],
            });
        }
        // x @ w1 + b1
        let mut h = try_matmul(x, &self.w1)?;
        add_row_bias(&mut h, &self.b1)?;
        // GELU activation
        let h = h.gelu();
        // h @ w2 + b2
        let mut out = try_matmul(&h, &self.w2)?;
        add_row_bias(&mut out, &self.b2)?;
        Ok(out)
    }

    pub fn param_count(&self) -> usize {
        super::sum_numel([&self.w1, &self.b1, &self.w2, &self.b2])
    }
}

/// Single transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
#[derive(Clone)]
pub struct TransformerBlock {
    pub attn: MultiHeadAttention,
    pub ffn: FeedForward,
    pub ln1_w: Tensor,
    pub ln1_b: Tensor,
    pub ln2_w: Tensor,
    pub ln2_b: Tensor,
    pub dim: usize,
}

reference_serde!(
    TransformerBlock,
    BlockWire {
        attn: MultiHeadAttention,
        ffn: FeedForward,
        ln1_w: Tensor,
        ln1_b: Tensor,
        ln2_w: Tensor,
        ln2_b: Tensor,
        dim: usize,
    }
);

impl TransformerBlock {
    pub(super) fn validate_wire(&self) -> Result<()> {
        if self.dim == 0 {
            return Err(CortexError::ZeroWidth {
                op: "TransformerBlock::validate_wire",
                axis: 0,
            });
        }
        if self.attn.dim != self.dim {
            return Err(CortexError::InvalidConfig(
                "block and attention dimensions disagree".into(),
            ));
        }
        self.attn.validate_wire()?;
        self.ffn.validate_wire()?;
        expect_shape(&self.ffn.w1, &[self.dim, self.ffn.w1.shape()[1]])?;
        for norm in [&self.ln1_w, &self.ln1_b, &self.ln2_w, &self.ln2_b] {
            expect_shape(norm, &[self.dim])?;
        }
        Ok(())
    }
    /// # Panics
    ///
    /// Panics if `num_heads` is invalid for `dim` or `ff_dim` is zero. Prefer
    /// [`TransformerBlock::try_new`].
    pub fn new(dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        unwrap_compat(
            Self::try_new(dim, num_heads, ff_dim),
            "TransformerBlock::new",
        )
    }

    /// Fallible constructor: returns [`CortexError::ZeroWidth`] when `dim`,
    /// `num_heads` or `ff_dim` is zero, and [`CortexError::InvalidConfig`]
    /// when `num_heads` does not evenly divide `dim`.
    pub fn try_new(dim: usize, num_heads: usize, ff_dim: usize) -> Result<Self> {
        Self::try_new_with_rng(dim, num_heads, ff_dim, &mut rand::rng())
    }

    /// Like [`Self::try_new`] but seeded for reproducible weights.
    pub fn try_new_seeded(dim: usize, num_heads: usize, ff_dim: usize, seed: u64) -> Result<Self> {
        Self::try_new_with_rng(dim, num_heads, ff_dim, &mut crate::tensor::seeded_rng(seed))
    }

    /// Like [`Self::try_new`] but draws weights from the caller's RNG.
    pub fn try_new_with_rng<R: rand::Rng + ?Sized>(
        dim: usize,
        num_heads: usize,
        ff_dim: usize,
        rng: &mut R,
    ) -> Result<Self> {
        super::check_head_config("TransformerBlock::try_new", dim, num_heads, 0, 1)?;
        if ff_dim == 0 {
            return Err(CortexError::ZeroWidth {
                op: "TransformerBlock::try_new",
                axis: 2,
            });
        }
        Ok(Self {
            attn: MultiHeadAttention::try_new_with_rng(dim, num_heads, rng)?,
            ffn: FeedForward::try_new_with_rng(dim, ff_dim, rng)?,
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
        let eps = super::LAYER_NORM_EPS;

        if x.ndim() == 2 && x.shape()[1] != self.dim {
            return Err(CortexError::DimMismatch {
                op: "TransformerBlock::try_forward",
                axis: 1,
                expected: self.dim,
                got: x.shape()[1],
            });
        }

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
        super::sum_numel([&self.ln1_w, &self.ln1_b, &self.ln2_w, &self.ln2_b])
            .saturating_add(self.attn.param_count())
            .saturating_add(self.ffn.param_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffn_rejects_zero_dims() {
        assert!(matches!(
            FeedForward::try_new_seeded(0, 8, 0),
            Err(CortexError::ZeroWidth { axis: 0, .. })
        ));
        assert!(matches!(
            FeedForward::try_new_seeded(8, 0, 0),
            Err(CortexError::ZeroWidth { axis: 1, .. })
        ));
    }

    #[test]
    fn seeded_stream_is_pinned() {
        let ffn = FeedForward::try_new_seeded(4, 8, 0).unwrap();
        let want = [-0.185_433_52_f32, -0.705_181_54, -0.790_492, 0.171_895_62];
        for (got, want) in ffn.w1.data()[..4].iter().zip(want) {
            assert!((got - want).abs() < 1e-5, "{got} != {want}");
        }
    }

    #[test]
    fn seeded_block_is_reproducible() {
        let a = TransformerBlock::try_new_seeded(16, 4, 64, 1).unwrap();
        let b = TransformerBlock::try_new_seeded(16, 4, 64, 1).unwrap();
        assert_eq!(a.ffn.w2.data(), b.ffn.w2.data());
        assert_eq!(a.attn.wq.data(), b.attn.wq.data());
    }

    #[test]
    fn ffn_output_variance_is_near_input_variance() {
        let ffn = FeedForward::try_new_seeded(64, 1024, 3).unwrap();
        let x = Tensor::ones(&[64, 64]);
        let y = ffn.forward(&x);
        let n = y.numel() as f32;
        let mean = y.data().iter().copied().sum::<f32>() / n;
        let var = y
            .data()
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / n;
        assert!(var < 2.0, "ffn output variance too large: {var}");
    }

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
