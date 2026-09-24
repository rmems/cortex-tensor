// SPDX-License-Identifier: MIT OR Apache-2.0

use super::block::TransformerBlock;
use crate::error::{CortexError, Result, unwrap_compat};
use crate::tensor::Tensor;
use crate::tensor::ops::{try_embedding, try_layer_norm, try_matmul};
use serde::{Deserialize, Serialize};

/// Configuration for a decoder-only transformer LM.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
}

impl TransformerConfig {
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

    pub fn estimated_params(&self) -> usize {
        let embed = self.vocab_size * self.dim;
        let per_block = {
            let attn = 4 * self.dim * self.dim + 4 * self.dim;
            let ffn = 2 * self.dim * self.ff_dim + self.dim + self.ff_dim;
            let ln = 4 * self.dim;
            attn + ffn + ln
        };
        let final_ln = 2 * self.dim;
        let lm_head = self.dim * self.vocab_size;
        embed + self.num_layers * per_block + final_ln + lm_head
    }
}

/// Decoder-only transformer language model — candle-free.
///
/// Architecture: token embedding + positional embedding → N × TransformerBlock → LayerNorm → LM head
#[derive(Clone, Serialize, Deserialize)]
pub struct TransformerLM {
    pub config: TransformerConfig,
    pub tok_embed: Tensor, // [vocab_size, dim]
    pub pos_embed: Tensor, // [max_seq_len, dim]
    pub blocks: Vec<TransformerBlock>,
    pub final_ln_w: Tensor, // [dim]
    pub final_ln_b: Tensor, // [dim]
    pub lm_head: Tensor,    // [dim, vocab_size]
}

impl TransformerLM {
    /// # Panics
    ///
    /// Panics if `num_heads` is invalid for `dim` or a shape allocation
    /// overflows. Prefer [`TransformerLM::try_new`].
    pub fn new(cfg: TransformerConfig) -> Self {
        unwrap_compat(Self::try_new(cfg), "TransformerLM::new")
    }

    /// Fallible constructor: returns [`CortexError::InvalidConfig`] when
    /// `cfg.num_heads` is zero or does not evenly divide `cfg.dim`, and
    /// [`CortexError::SizeOverflow`] when a configured shape is not
    /// representable.
    pub fn try_new(cfg: TransformerConfig) -> Result<Self> {
        // Validate the head config up front: with num_layers == 0 the block
        // iterator below never runs, so per-block validation alone would
        // accept invalid configs.
        if cfg.num_heads == 0 || !cfg.dim.is_multiple_of(cfg.num_heads) {
            return Err(CortexError::InvalidConfig(format!(
                "TransformerLM: num_heads ({}) must be nonzero and divide dim ({})",
                cfg.num_heads, cfg.dim
            )));
        }
        let scale = 0.02;
        let blocks: Vec<TransformerBlock> = (0..cfg.num_layers)
            .map(|_| TransformerBlock::try_new(cfg.dim, cfg.num_heads, cfg.ff_dim))
            .collect::<Result<_>>()?;

        Ok(Self {
            tok_embed: Tensor::try_randn(&[cfg.vocab_size, cfg.dim], 0.0, scale)?,
            pos_embed: Tensor::try_randn(&[cfg.max_seq_len, cfg.dim], 0.0, scale)?,
            blocks,
            final_ln_w: Tensor::try_ones(&[cfg.dim])?,
            final_ln_b: Tensor::try_zeros(&[cfg.dim])?,
            lm_head: Tensor::try_randn(&[cfg.dim, cfg.vocab_size], 0.0, scale)?,
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
        let seq_len = token_ids.len();
        if seq_len > self.config.max_seq_len {
            return Err(CortexError::InputLengthMismatch {
                expected: self.config.max_seq_len,
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
        let mut x = tok.try_add(&pos)?;

        for block in &self.blocks {
            x = block.try_forward(&x)?;
        }

        try_layer_norm(&x, &self.final_ln_w, &self.final_ln_b, 1e-5)
    }

    pub fn param_count(&self) -> usize {
        self.config.estimated_params()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            Err(CortexError::InvalidConfig(_))
        ));
    }

    #[test]
    fn try_new_validates_heads_even_with_zero_layers() {
        let mut cfg = micro_cfg();
        cfg.num_layers = 0;
        cfg.num_heads = 0;
        assert!(matches!(
            TransformerLM::try_new(cfg.clone()),
            Err(CortexError::InvalidConfig(_))
        ));
        cfg.num_heads = 3;
        assert!(matches!(
            TransformerLM::try_new(cfg),
            Err(CortexError::InvalidConfig(_))
        ));
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
