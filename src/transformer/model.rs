// SPDX-License-Identifier: MIT OR Apache-2.0

use super::block::TransformerBlock;
use super::expect_shape;
use crate::error::{CortexError, Result, unwrap_compat};
use crate::tensor::Tensor;
use crate::tensor::ops::{try_embedding, try_layer_norm, try_matmul};

/// Configuration for a decoder-only transformer LM.
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
}

reference_serde!(
    TransformerConfig,
    ConfigWire {
        vocab_size: usize,
        dim: usize,
        num_heads: usize,
        num_layers: usize,
        ff_dim: usize,
        max_seq_len: usize,
    }
);

impl TransformerConfig {
    pub(crate) fn validate_wire(&self) -> Result<()> {
        self.validate()
    }

    /// Single validation used by both the constructors and the wire
    /// (deserialize) path, so the same bad config yields the same error.
    ///
    /// Zero extents are [`CortexError::ZeroWidth`] whose `axis` is the
    /// field's declaration index (`vocab_size` 0, `dim` 1, `num_heads` 2,
    /// `num_layers` 3, `ff_dim` 4, `max_seq_len` 5). `num_layers == 0` is
    /// allowed. A `num_heads` that does not divide `dim` is
    /// [`CortexError::InvalidConfig`].
    pub fn validate(&self) -> Result<()> {
        const OP: &str = "TransformerConfig::validate";
        super::check_head_config(OP, self.dim, self.num_heads, 1, 2)?;
        for (axis, v) in [
            (0, self.vocab_size),
            (4, self.ff_dim),
            (5, self.max_seq_len),
        ] {
            if v == 0 {
                return Err(CortexError::ZeroWidth { op: OP, axis });
            }
        }
        Ok(())
    }

    /// A small config for testing (~2M params).
    pub fn tiny() -> Self {
        Self {
            vocab_size: 4096,
            dim: 128,
            num_heads: 4,
            num_layers: 4,
            ff_dim: 512,
            max_seq_len: 256,
        }
    }

    /// Approximate OLMo-1B scale config.
    pub fn olmo_1b() -> Self {
        Self {
            vocab_size: 50280,
            dim: 2048,
            num_heads: 16,
            num_layers: 16,
            ff_dim: 8192,
            max_seq_len: 2048,
        }
    }

    /// Parameter count implied by this config, saturating at `usize::MAX`
    /// when the arithmetic overflows.
    ///
    /// Counts every tensor [`TransformerLM::try_new`] allocates: token and
    /// positional embeddings, each block's attention (`4·dim² + 4·dim`,
    /// matching [`MultiHeadAttention::param_count`]), feed-forward
    /// (`2·dim·ff_dim + dim + ff_dim`), and two LayerNorms (`4·dim`), plus
    /// the final LayerNorm (`2·dim`) and the LM head (`dim·vocab_size`).
    /// For a constructed model this equals [`TransformerLM::param_count`].
    /// Prefer [`Self::try_estimated_params`] when an overflow must be
    /// reported instead of saturated.
    ///
    /// [`MultiHeadAttention::param_count`]: super::MultiHeadAttention::param_count
    pub fn estimated_params(&self) -> usize {
        self.try_estimated_params().unwrap_or(usize::MAX)
    }

    /// Checked variant of [`Self::estimated_params`]: returns
    /// [`CortexError::InvalidConfig`] when the parameter count does not fit in
    /// `usize`. The config fields are public, so no field combination is
    /// assumed to be representable.
    pub fn try_estimated_params(&self) -> Result<usize> {
        let overflow =
            || CortexError::InvalidConfig("transformer parameter count overflows usize".into());
        let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
        let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(overflow);

        let tok_embed = mul(self.vocab_size, self.dim)?;
        let pos_embed = mul(self.max_seq_len, self.dim)?;
        // With no layers there are no block tensors, so a per-block product
        // that would overflow on its own must not fail the estimate.
        let blocks = if self.num_layers == 0 {
            0
        } else {
            // MultiHeadAttention: wq/wk/wv/wo [dim, dim] + four [1, dim] biases.
            let attn = add(mul(mul(4, self.dim)?, self.dim)?, mul(4, self.dim)?)?;
            // FeedForward: w1 [dim, ff_dim], b1 [1, ff_dim], w2 [ff_dim, dim], b2 [1, dim].
            let ffn = add(
                add(mul(mul(2, self.dim)?, self.ff_dim)?, self.dim)?,
                self.ff_dim,
            )?;
            // Two LayerNorms, each with weight and bias of [dim].
            let ln = mul(4, self.dim)?;
            mul(self.num_layers, add(add(attn, ffn)?, ln)?)?
        };
        let final_ln = mul(2, self.dim)?;
        let lm_head = mul(self.dim, self.vocab_size)?;
        add(
            add(add(add(tok_embed, pos_embed)?, blocks)?, final_ln)?,
            lm_head,
        )
    }
}

/// Decoder-only transformer language model — candle-free.
///
/// Architecture: token embedding + positional embedding → N × TransformerBlock → LayerNorm → LM head
#[derive(Clone)]
pub struct TransformerLM {
    pub config: TransformerConfig,
    pub tok_embed: Tensor, // [vocab_size, dim]
    pub pos_embed: Tensor, // [max_seq_len, dim]
    pub blocks: Vec<TransformerBlock>,
    pub final_ln_w: Tensor, // [dim]
    pub final_ln_b: Tensor, // [dim]
    pub lm_head: Tensor,    // [dim, vocab_size]
}

reference_serde!(TransformerLM, ModelWire {
    config: TransformerConfig, tok_embed: Tensor, pos_embed: Tensor,
    blocks: Vec<TransformerBlock>, final_ln_w: Tensor, final_ln_b: Tensor,
    lm_head: Tensor,
});

impl TransformerLM {
    pub(crate) fn validate_wire(&self) -> Result<()> {
        let cfg = &self.config;
        cfg.validate()?;
        expect_shape(&self.tok_embed, &[cfg.vocab_size, cfg.dim])?;
        expect_shape(&self.pos_embed, &[cfg.max_seq_len, cfg.dim])?;
        expect_shape(&self.final_ln_w, &[cfg.dim])?;
        expect_shape(&self.final_ln_b, &[cfg.dim])?;
        expect_shape(&self.lm_head, &[cfg.dim, cfg.vocab_size])?;
        if self.blocks.len() != cfg.num_layers {
            return Err(CortexError::InvalidConfig(
                "block count disagrees with num_layers".into(),
            ));
        }
        for block in &self.blocks {
            block.validate_wire()?;
            if block.dim != cfg.dim
                || block.attn.num_heads != cfg.num_heads
                || block.ffn.w1.shape() != [cfg.dim, cfg.ff_dim]
            {
                return Err(CortexError::InvalidConfig(
                    "block weights disagree with model configuration".into(),
                ));
            }
        }
        Ok(())
    }
    /// # Panics
    ///
    /// Panics if `num_heads` is invalid for `dim` or a shape allocation
    /// overflows. Prefer [`TransformerLM::try_new`].
    pub fn new(cfg: TransformerConfig) -> Self {
        unwrap_compat(Self::try_new(cfg), "TransformerLM::new")
    }

    /// Fallible constructor: returns [`CortexError::ZeroWidth`] when
    /// any model extent is zero, [`CortexError::InvalidConfig`] when
    /// `cfg.num_heads` does not divide `cfg.dim`, and
    /// [`CortexError::SizeOverflow`] when a configured shape is not
    /// representable.
    pub fn try_new(cfg: TransformerConfig) -> Result<Self> {
        Self::try_new_with_rng(cfg, &mut rand::rng())
    }

    /// Like [`Self::try_new`] but seeded: the same config and seed always
    /// rebuild the same model.
    pub fn try_new_seeded(cfg: TransformerConfig, seed: u64) -> Result<Self> {
        Self::try_new_with_rng(cfg, &mut crate::tensor::seeded_rng(seed))
    }

    /// Like [`Self::try_new`] but draws all weights from the caller's RNG.
    pub fn try_new_with_rng<R: rand::Rng + ?Sized>(
        cfg: TransformerConfig,
        rng: &mut R,
    ) -> Result<Self> {
        // Validate up front: with num_layers == 0 the block iterator below
        // never runs, so per-block validation alone would accept bad configs.
        cfg.validate()?;
        let scale = 0.02;
        let blocks: Vec<TransformerBlock> = (0..cfg.num_layers)
            .map(|_| TransformerBlock::try_new_with_rng(cfg.dim, cfg.num_heads, cfg.ff_dim, rng))
            .collect::<Result<_>>()?;

        Ok(Self {
            tok_embed: Tensor::try_randn_with_rng(&[cfg.vocab_size, cfg.dim], 0.0, scale, rng)?,
            pos_embed: Tensor::try_randn_with_rng(&[cfg.max_seq_len, cfg.dim], 0.0, scale, rng)?,
            blocks,
            final_ln_w: Tensor::try_ones(&[cfg.dim])?,
            final_ln_b: Tensor::try_zeros(&[cfg.dim])?,
            lm_head: Tensor::try_randn_with_rng(&[cfg.dim, cfg.vocab_size], 0.0, scale, rng)?,
            config: cfg,
        })
    }

    /// Forward pass: token_ids → logits [seq_len, vocab_size]
    ///
    /// # Panics
    ///
    /// Panics when `token_ids.len() > config.max_seq_len`, a token id is out
    /// of vocabulary, or an intermediate shape is inconsistent. Prefer
    /// [`Self::try_forward`].
    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        unwrap_compat(self.try_forward(token_ids), "TransformerLM::forward")
    }

    /// Fallible forward pass.
    ///
    /// Returns [`CortexError::InputLengthMismatch`] when the sequence exceeds
    /// `config.max_seq_len` and [`CortexError::TokenIndex`] for
    /// out-of-vocabulary ids, without slicing or asserting.
    pub fn try_forward(&self, token_ids: &[u32]) -> Result<Tensor> {
        let x = self.try_hidden_states(token_ids)?;

        // LM head: [seq_len, dim] × [dim, vocab] → [seq_len, vocab]
        try_matmul(&x, &self.lm_head)
    }

    /// Get the hidden state after all transformer blocks (before LM head).
    /// Used by the SNN/LLM fusion layer.
    ///
    /// # Panics
    ///
    /// Panics when `token_ids.len() > config.max_seq_len` or a token id is
    /// out of vocabulary. Prefer [`Self::try_hidden_states`].
    pub fn hidden_states(&self, token_ids: &[u32]) -> Tensor {
        unwrap_compat(
            self.try_hidden_states(token_ids),
            "TransformerLM::hidden_states",
        )
    }

    /// Fallible hidden-state extraction: same contract as
    /// [`Self::try_forward`] but stops before the LM head.
    pub fn try_hidden_states(&self, token_ids: &[u32]) -> Result<Tensor> {
        let mut x = self.embed_tokens(token_ids)?;

        for block in &self.blocks {
            x = block.try_forward(&x)?;
        }

        try_layer_norm(
            &x,
            &self.final_ln_w,
            &self.final_ln_b,
            super::LAYER_NORM_EPS,
        )
    }

    /// Token + positional embedding, shared by [`Self::try_hidden_states`] and
    /// the stage executor's embedding stage so both compute byte-identical
    /// results.
    ///
    /// Returns [`CortexError::InputLengthMismatch`] when the sequence exceeds
    /// `config.max_seq_len` (or, defensively, when a position index overflows
    /// `u32`), and [`CortexError::TokenIndex`] for out-of-vocabulary ids.
    pub(crate) fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor> {
        let seq_len = token_ids.len();
        // `config` and `pos_embed` are both `pub`; bound by the table that is
        // actually indexed so a raised `max_seq_len` still reports a length
        // error rather than a misleading `TokenIndex` from the lookup.
        let max_pos = self
            .pos_embed
            .shape()
            .first()
            .copied()
            .unwrap_or(0)
            .min(self.config.max_seq_len);
        if seq_len > max_pos {
            return Err(CortexError::InputLengthMismatch {
                expected: max_pos,
                got: seq_len,
            });
        }

        // Token + positional embeddings. `try_from` guards the u32
        // truncation a literal `as` cast would silently allow.
        let tok = try_embedding(&self.tok_embed, token_ids)?;
        let pos_ids: Vec<u32> = (0..seq_len)
            .map(u32::try_from)
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| CortexError::InputLengthMismatch {
                expected: u32::MAX as usize,
                got: seq_len,
            })?;
        let pos = try_embedding(&self.pos_embed, &pos_ids)?;
        tok.try_add(&pos)
    }

    /// Total number of stored parameters, summed from the `numel()` of every
    /// tensor the model holds (blocks included), so it tracks edits to the
    /// public weight fields. Equals [`TransformerConfig::estimated_params`]
    /// for a model built from its config.
    pub fn param_count(&self) -> usize {
        let blocks = self
            .blocks
            .iter()
            .fold(0usize, |acc, b| acc.saturating_add(stored_block_params(b)));
        super::sum_numel([
            &self.tok_embed,
            &self.pos_embed,
            &self.final_ln_w,
            &self.final_ln_b,
            &self.lm_head,
        ])
        .saturating_add(blocks)
    }
}

/// `numel()` of every tensor stored in one block. Unlike
/// [`TransformerBlock::param_count`], which derives its count from `dim`,
/// this observes the tensors as they are.
fn stored_block_params(block: &TransformerBlock) -> usize {
    let attn = &block.attn;
    let ffn = &block.ffn;
    [
        &attn.wq,
        &attn.wk,
        &attn.wv,
        &attn.wo,
        &attn.wb_q,
        &attn.wb_k,
        &attn.wb_v,
        &attn.wb_o,
        &ffn.w1,
        &ffn.b1,
        &ffn.w2,
        &ffn.b2,
        &block.ln1_w,
        &block.ln1_b,
        &block.ln2_w,
        &block.ln2_b,
    ]
    .iter()
    .map(|t| t.numel())
    .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_model_is_reproducible() {
        let cfg = TransformerConfig::tiny();
        let a = TransformerLM::try_new_seeded(cfg.clone(), 42).unwrap();
        let b = TransformerLM::try_new_seeded(cfg, 42).unwrap();
        assert_eq!(a.forward(&[1, 2, 3]).data(), b.forward(&[1, 2, 3]).data());
    }

    #[test]
    fn test_tiny_forward() {
        let cfg = TransformerConfig::tiny();
        let model = TransformerLM::new(cfg.clone());
        let ids = vec![1u32, 42, 100, 7];
        let logits = model.forward(&ids);
        assert_eq!(logits.shape(), &[4, cfg.vocab_size]);
    }

    #[test]
    fn test_hidden_states() {
        let cfg = TransformerConfig::tiny();
        let model = TransformerLM::new(cfg.clone());
        let ids = vec![1u32, 2, 3];
        let h = model.hidden_states(&ids);
        assert_eq!(h.shape(), &[3, cfg.dim]);
    }

    /// Minimal config so seq-overflow / OOV tests stay fast.
    fn micro_cfg() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 4,
            dim: 8,
            num_heads: 2,
            num_layers: 1,
            ff_dim: 16,
            max_seq_len: 4,
        }
    }

    #[test]
    fn try_forward_rejects_seq_len_over_max() {
        let model = TransformerLM::try_new(micro_cfg()).unwrap();
        let ids = vec![0u32; 5];
        let err = model.try_forward(&ids).unwrap_err();
        assert!(matches!(
            err,
            CortexError::InputLengthMismatch {
                expected: 4,
                got: 5
            }
        ));
        let err = model.try_hidden_states(&ids).unwrap_err();
        assert!(matches!(err, CortexError::InputLengthMismatch { .. }));
    }

    #[test]
    fn try_forward_rejects_out_of_vocab_ids() {
        let model = TransformerLM::try_new(micro_cfg()).unwrap();
        let err = model.try_forward(&[1, 99999]).unwrap_err();
        assert!(matches!(
            err,
            CortexError::TokenIndex {
                index: 99999,
                vocab_size: 4
            }
        ));
    }

    #[test]
    fn try_new_rejects_invalid_num_heads() {
        let mut cfg = micro_cfg();
        cfg.num_heads = 3; // does not divide dim=8
        assert!(matches!(
            TransformerLM::try_new(cfg.clone()),
            Err(CortexError::InvalidConfig(_))
        ));
        cfg.num_heads = 0;
        assert!(matches!(
            TransformerLM::try_new(cfg),
            Err(CortexError::ZeroWidth { axis: 2, .. })
        ));
    }

    #[test]
    fn try_new_validates_heads_even_with_zero_layers() {
        let mut cfg = micro_cfg();
        cfg.num_layers = 0;
        cfg.num_heads = 0;
        assert!(matches!(
            TransformerLM::try_new(cfg.clone()),
            Err(CortexError::ZeroWidth { axis: 2, .. })
        ));
        cfg.num_heads = 3;
        assert!(matches!(
            TransformerLM::try_new(cfg),
            Err(CortexError::InvalidConfig(_))
        ));
    }

    #[test]
    fn estimated_params_matches_allocated_tensors() {
        // tiny() allocates 1,874,688 f32 values; the old estimate returned
        // 1,841,920 because it omitted pos_embed (256 * 128 = 32,768).
        let cfg = TransformerConfig::tiny();
        let model = TransformerLM::try_new(cfg.clone()).unwrap();
        // Per-layer formula check against the layers' own accounting.
        for block in &model.blocks {
            assert_eq!(stored_block_params(block), block.param_count());
        }
        let actual = model.param_count();
        assert_eq!(actual, 1_874_688);
        assert_eq!(cfg.estimated_params(), actual);
        assert_eq!(cfg.try_estimated_params().unwrap(), actual);
    }

    #[test]
    fn param_count_tracks_edited_block_tensors() {
        let mut model = TransformerLM::try_new(TransformerConfig::tiny()).unwrap();
        let before = model.param_count();
        // Public field: shrink ln1_w from [128] to [1].
        model.blocks[0].ln1_w = Tensor::try_ones(&[1]).unwrap();
        assert_eq!(model.param_count(), before - 127);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn estimated_params_zero_layers_skips_block_term() {
        // 4 * dim * dim overflows, but with no layers that tensor never
        // exists; the remaining terms fit comfortably.
        let dim = 1usize << 32;
        let cfg = TransformerConfig {
            vocab_size: 1,
            dim,
            num_heads: 1,
            num_layers: 0,
            ff_dim: 1,
            max_seq_len: 1,
        };
        // tok_embed + pos_embed + final_ln (2*dim) + lm_head = 5 * dim.
        assert_eq!(cfg.try_estimated_params().unwrap(), 5 * dim);
        assert_eq!(cfg.estimated_params(), 5 * dim);
    }

    #[test]
    fn estimated_params_uses_checked_arithmetic() {
        // vocab_size * dim overflows usize. The unchecked formula panicked
        // in debug builds and wrapped to 136 in release.
        let mut cfg = micro_cfg();
        cfg.vocab_size = usize::MAX / 2;
        assert!(matches!(
            cfg.try_estimated_params(),
            Err(CortexError::InvalidConfig(_))
        ));
        assert_eq!(cfg.estimated_params(), usize::MAX);

        // Products fit but the final sum does not.
        let cfg = TransformerConfig {
            vocab_size: usize::MAX / 2,
            dim: 1,
            num_heads: 1,
            num_layers: 0,
            ff_dim: 1,
            max_seq_len: usize::MAX / 2,
        };
        assert!(cfg.try_estimated_params().is_err());
        assert_eq!(cfg.estimated_params(), usize::MAX);
    }

    #[test]
    fn try_new_rejects_unrepresentable_shapes() {
        let mut cfg = micro_cfg();
        cfg.vocab_size = usize::MAX;
        assert!(matches!(
            TransformerLM::try_new(cfg),
            Err(CortexError::SizeOverflow { .. })
        ));
    }
}

#[cfg(test)]
mod error_consistency_tests {
    use super::*;
    use crate::transformer::{MultiHeadAttention, TransformerBlock};

    fn cfg() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 4,
            dim: 8,
            num_heads: 2,
            num_layers: 1,
            ff_dim: 16,
            max_seq_len: 4,
        }
    }

    #[test]
    fn constructor_and_wire_share_one_config_error() {
        let mut bad = cfg();
        bad.num_heads = 3;
        let ctor = TransformerLM::try_new(bad.clone()).err().unwrap();
        let wire = bad.validate().unwrap_err();
        assert_eq!(ctor.to_string(), wire.to_string());
        assert!(matches!(wire, CortexError::InvalidConfig(_)));
        // Attention and block constructors report the same text.
        let attn = MultiHeadAttention::try_new(8, 3).err().unwrap();
        let block = TransformerBlock::try_new(8, 3, 16).err().unwrap();
        assert_eq!(attn.to_string(), wire.to_string());
        assert_eq!(block.to_string(), wire.to_string());
    }

    #[test]
    fn zero_extents_are_zero_width() {
        for (axis, edit) in [
            (
                0usize,
                (|c: &mut TransformerConfig| c.vocab_size = 0) as fn(&mut TransformerConfig),
            ),
            (1, |c| c.dim = 0),
            (2, |c| c.num_heads = 0),
            (4, |c| c.ff_dim = 0),
            (5, |c| c.max_seq_len = 0),
        ] {
            let mut c = cfg();
            edit(&mut c);
            assert!(
                matches!(c.validate(), Err(CortexError::ZeroWidth { axis: a, .. }) if a == axis),
                "axis {axis}"
            );
        }
    }

    #[test]
    fn attention_wire_zero_heads_is_zero_width_like_the_constructor() {
        let mut attn = MultiHeadAttention::try_new(8, 2).unwrap();
        attn.num_heads = 0;
        assert!(matches!(
            attn.validate_wire(),
            Err(CortexError::ZeroWidth { axis: 1, .. })
        ));
    }

    #[test]
    fn position_bound_follows_the_table_actually_indexed() {
        let mut model = TransformerLM::try_new(cfg()).unwrap();
        model.config.max_seq_len = 100; // pub field raised past the 4-row table
        let err = model.try_forward(&[0, 1, 2, 3, 0]).unwrap_err();
        assert!(
            matches!(
                err,
                CortexError::InputLengthMismatch {
                    expected: 4,
                    got: 5
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn param_counts_cannot_overflow_on_pub_fields() {
        let mut attn = MultiHeadAttention::try_new(8, 2).unwrap();
        attn.dim = usize::MAX / 2;
        let _ = attn.param_count();
        let mut block = TransformerBlock::try_new(8, 2, 16).unwrap();
        block.dim = usize::MAX / 2;
        block.attn.dim = usize::MAX / 2;
        let _ = block.param_count();
    }
}
