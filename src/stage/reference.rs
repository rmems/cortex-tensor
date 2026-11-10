// SPDX-License-Identifier: MIT OR Apache-2.0

use super::vocabulary::{
    AnnCapabilities, AnnExecutor, AnnStage, AnnTopology, DType, MlpKind, NormKind, StageId,
    StageInput, StageKind, StageKindTag, StageSource, StageTensor,
};
use crate::adapter::{self, require_f32_buffer};
use crate::error::{CortexError, Result};
use crate::tensor::Tensor;
use crate::tensor::ops::{try_layer_norm, try_matmul, try_rms_norm};
use crate::transformer::block::FeedForward;
use crate::transformer::{MultiHeadAttention, TransformerLM};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Borrowed parameters a [`ReferenceExecutor`] needs to run one stage.
///
/// Each variant carries references into a model (or into standalone tensors
/// for test injection); the executor never owns weights. The variant selects
/// how [`ReferenceExecutor::execute`] delegates to the existing reference
/// kernels so a composed topology matches [`TransformerLM`] bit for bit.
///
/// `Debug` reports only each variant's shape-level metadata, not weights:
/// [`MultiHeadAttention`] and [`FeedForward`] intentionally do not implement
/// `Debug`, and dumping full weight tensors would be noise.
#[derive(Clone)]
pub enum ReferenceStageParams<'p> {
    /// Token + position embedding tables and the sequence-length cap.
    Embedding {
        /// Token embedding table `[vocab_size, dim]`.
        tok_embed: &'p Tensor,
        /// Position embedding table `[max_seq_len, dim]`.
        pos_embed: &'p Tensor,
        /// Maximum sequence length accepted (matches `TransformerConfig`).
        /// The effective cap is `min(max_seq_len, pos_embed rows)`.
        max_seq_len: usize,
    },
    /// Multi-head self-attention weights.
    Attention(&'p MultiHeadAttention),
    /// Normalization weights, optional bias, epsilon, and flavour.
    Normalization {
        /// Scale/gain vector `[dim]`.
        weight: &'p Tensor,
        /// Optional shift vector `[dim]`. Required for [`NormKind::Layer`]
        /// and must be `None` for [`NormKind::Rms`]; either violation is
        /// reported as [`CortexError::InvalidStageParams`] at execute time.
        bias: Option<&'p Tensor>,
        /// Numerical-stability epsilon.
        eps: f32,
        /// Whether to apply LayerNorm or RMSNorm.
        kind: NormKind,
    },
    /// Dense feed-forward network.
    Mlp(&'p FeedForward),
    /// Final projection weight `[dim, vocab_size]`.
    Readout {
        /// LM head / readout projection.
        lm_head: &'p Tensor,
    },
}

impl fmt::Debug for ReferenceStageParams<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReferenceStageParams::Embedding {
                tok_embed,
                pos_embed,
                max_seq_len,
            } => f
                .debug_struct("Embedding")
                .field("tok_embed", &tok_embed.shape())
                .field("pos_embed", &pos_embed.shape())
                .field("max_seq_len", max_seq_len)
                .finish(),
            ReferenceStageParams::Attention(_) => f.write_str("Attention(..)"),
            ReferenceStageParams::Normalization {
                weight,
                bias,
                eps,
                kind,
            } => f
                .debug_struct("Normalization")
                .field("weight", &weight.shape())
                .field("bias", &bias.map(Tensor::shape))
                .field("eps", eps)
                .field("kind", kind)
                .finish(),
            ReferenceStageParams::Mlp(_) => f.write_str("Mlp(..)"),
            ReferenceStageParams::Readout { lm_head } => f
                .debug_struct("Readout")
                .field("lm_head", &lm_head.shape())
                .finish(),
        }
    }
}

impl ReferenceStageParams<'_> {
    /// The [`StageKindTag`] this parameter set executes, used to confirm the
    /// bound params agree with the stage descriptor's kind.
    fn tag(&self) -> StageKindTag {
        match self {
            ReferenceStageParams::Embedding { .. } => StageKindTag::Embedding,
            ReferenceStageParams::Attention(_) => StageKindTag::Attention,
            ReferenceStageParams::Normalization {
                kind: NormKind::Layer,
                ..
            } => StageKindTag::LayerNorm,
            ReferenceStageParams::Normalization {
                kind: NormKind::Rms,
                ..
            } => StageKindTag::RmsNorm,
            ReferenceStageParams::Mlp(_) => StageKindTag::DenseMlp,
            ReferenceStageParams::Readout { .. } => StageKindTag::Readout,
        }
    }
}

/// Dense CPU executor that composes [`TransformerLM`] as a stage graph.
///
/// `ReferenceExecutor` is the reference `f32` backend for [`AnnExecutor`]. It
/// holds borrowed [`ReferenceStageParams`] keyed by [`StageId`] and, for every
/// supported stage, delegates to the *same* kernels the monolithic
/// [`TransformerLM::try_forward`] uses (the shared token + position
/// embedding built on [`crate::tensor::ops::try_embedding`], [`try_layer_norm`],
/// [`try_rms_norm`], [`MultiHeadAttention::try_forward`],
/// [`FeedForward::try_forward`], [`try_matmul`], and [`Tensor::try_add`]). It
/// reimplements no arithmetic, so running a topology built by
/// [`ReferenceExecutor::from_transformer`] reproduces the model's output bit
/// for bit.
///
/// It is stateless across calls (`stateful = false`) and `f32`-only. [`Add`]
/// stages carry no params; every other supported stage must have params bound
/// (by hand via [`ReferenceExecutor::bind`] or in bulk by `from_transformer`).
///
/// [`Add`]: StageKind::Add
#[derive(Debug, Clone, Default)]
pub struct ReferenceExecutor<'p> {
    params: BTreeMap<StageId, ReferenceStageParams<'p>>,
}

impl<'p> ReferenceExecutor<'p> {
    /// Backend name reported by [`AnnCapabilities::backend_name`].
    pub const BACKEND_NAME: &'static str = "reference";

    /// Numerical-stability epsilon the reference transformer uses for every
    /// normalization, so hand-bound [`ReferenceStageParams::Normalization`]
    /// can match [`ReferenceExecutor::from_transformer`] without re-stating
    /// the literal.
    pub const LAYER_NORM_EPS: f32 = crate::transformer::LAYER_NORM_EPS;

    /// An executor with no bound params.
    ///
    /// Bind each non-[`Add`](StageKind::Add) stage's params with
    /// [`ReferenceExecutor::bind`] before executing, or build a fully wired
    /// executor with [`ReferenceExecutor::from_transformer`].
    pub fn new() -> Self {
        Self {
            params: BTreeMap::new(),
        }
    }

    /// Bind `params` to `id`, replacing any existing binding. Returns `self`
    /// for chaining.
    pub fn bind(mut self, id: StageId, params: ReferenceStageParams<'p>) -> Self {
        self.params.insert(id, params);
        self
    }

    /// The tags this backend can execute, as advertised by [`capabilities`].
    ///
    /// [`capabilities`]: AnnExecutor::capabilities
    fn supported_tags() -> BTreeSet<StageKindTag> {
        BTreeSet::from([
            StageKindTag::Embedding,
            StageKindTag::Attention,
            StageKindTag::LayerNorm,
            StageKindTag::RmsNorm,
            StageKindTag::DenseMlp,
            StageKindTag::Add,
            StageKindTag::Readout,
        ])
    }

    /// Resolve `inputs[index]` as a tensor, or [`CortexError::StageInputKind`].
    fn tensor_arg<'a>(
        stage_id: &StageId,
        inputs: &'a [StageInput<'a, Tensor>],
        index: usize,
    ) -> Result<&'a Tensor> {
        match &inputs[index] {
            StageInput::Tensor(t) => Ok(t),
            StageInput::Tokens(_) => Err(CortexError::StageInputKind {
                stage_id: stage_id.to_string(),
                index,
            }),
        }
    }

    /// Resolve `inputs[index]` as token ids, or [`CortexError::StageInputKind`].
    fn tokens_arg<'a>(
        stage_id: &StageId,
        inputs: &'a [StageInput<'a, Tensor>],
        index: usize,
    ) -> Result<&'a [u32]> {
        match &inputs[index] {
            StageInput::Tokens(ids) => Ok(ids),
            StageInput::Tensor(_) => Err(CortexError::StageInputKind {
                stage_id: stage_id.to_string(),
                index,
            }),
        }
    }

    /// Require exactly `expected` inputs, else [`CortexError::StageInputArity`].
    fn expect_arity(
        stage_id: &StageId,
        inputs: &[StageInput<'_, Tensor>],
        expected: usize,
    ) -> Result<()> {
        if inputs.len() != expected {
            return Err(CortexError::StageInputArity {
                stage_id: stage_id.to_string(),
                expected,
                got: inputs.len(),
            });
        }
        Ok(())
    }

    /// Wrap a kernel error as [`CortexError::StageFailed`] tagged with `stage_id`.
    fn wrap_failure(stage_id: &StageId, err: CortexError) -> CortexError {
        CortexError::StageFailed {
            stage_id: stage_id.to_string(),
            source: Box::new(err),
        }
    }

    /// Embedding stage: token + position lookup through the same
    /// [`crate::transformer::model::embed_tokens`] that
    /// [`TransformerLM::try_hidden_states`] uses.
    fn run_embedding(
        stage_id: &StageId,
        tok_embed: &Tensor,
        pos_embed: &Tensor,
        max_seq_len: usize,
        token_ids: &[u32],
    ) -> Result<Tensor> {
        crate::transformer::model::embed_tokens(tok_embed, pos_embed, max_seq_len, token_ids)
            .map_err(|err| Self::wrap_failure(stage_id, err))
    }

    /// Construct a fully wired executor and matching topology from a model.
    ///
    /// Runs the model's structural validation once (so a publicly mutated or
    /// inconsistent model is rejected with a structured [`CortexError`]), then
    /// builds an [`AnnTopology`] whose stage ids are, in order: `embedding`;
    /// for each block `i`: `blocks.{i}.norm1`, `blocks.{i}.attention`,
    /// `blocks.{i}.residual1`, `blocks.{i}.norm2`, `blocks.{i}.mlp`,
    /// `blocks.{i}.residual2`; then `final_norm`; then `readout`. The wiring
    /// follows the pre-norm block semantics of
    /// [`crate::transformer::block::TransformerBlock::try_forward`] and
    /// [`TransformerLM::try_hidden_states`], and each supported stage is bound
    /// to borrowed params from `model`, so [`crate::stage::run_topology`] reproduces
    /// [`TransformerLM::try_forward`] bit for bit.
    ///
    /// The returned executor borrows from `model`, so `model` must outlive it.
    ///
    /// # Errors
    ///
    /// Returns the structured error from the model's internal wire validation when
    /// the model is malformed, or [`CortexError::InvalidStageId`] /
    /// [`CortexError::InvalidTopology`] if id construction or topology
    /// validation fails (not expected for a valid model).
    pub fn from_transformer(model: &'p TransformerLM) -> Result<(Self, AnnTopology)> {
        model.validate_wire()?;

        let eps = Self::LAYER_NORM_EPS;
        let mut stages: Vec<AnnStage> = Vec::new();
        let mut executor = ReferenceExecutor::new();

        // Embedding ingress.
        let embedding_id = StageId::new("embedding")?;
        stages.push(AnnStage {
            id: embedding_id.clone(),
            kind: StageKind::Embedding,
            inputs: vec![StageSource::External("tokens".to_string())],
        });
        executor = executor.bind(
            embedding_id.clone(),
            ReferenceStageParams::Embedding {
                tok_embed: &model.tok_embed,
                pos_embed: &model.pos_embed,
                max_seq_len: model.config.max_seq_len,
            },
        );

        // Each block feeds from the previous block's residual2 (or the
        // embedding output for the first block), mirroring the fold in
        // `try_hidden_states`.
        let mut block_input = embedding_id.clone();
        for (i, block) in model.blocks.iter().enumerate() {
            let norm1 = StageId::new(format!("blocks.{i}.norm1"))?;
            let attention = StageId::new(format!("blocks.{i}.attention"))?;
            let residual1 = StageId::new(format!("blocks.{i}.residual1"))?;
            let norm2 = StageId::new(format!("blocks.{i}.norm2"))?;
            let mlp = StageId::new(format!("blocks.{i}.mlp"))?;
            let residual2 = StageId::new(format!("blocks.{i}.residual2"))?;

            // norm1 = layer_norm(block_input)
            stages.push(AnnStage {
                id: norm1.clone(),
                kind: StageKind::Normalization {
                    kind: NormKind::Layer,
                },
                inputs: vec![StageSource::Stage(block_input.clone())],
            });
            executor = executor.bind(
                norm1.clone(),
                ReferenceStageParams::Normalization {
                    weight: &block.ln1_w,
                    bias: Some(&block.ln1_b),
                    eps,
                    kind: NormKind::Layer,
                },
            );

            // attention = attn(norm1)
            stages.push(AnnStage {
                id: attention.clone(),
                kind: StageKind::Attention,
                inputs: vec![StageSource::Stage(norm1.clone())],
            });
            executor = executor.bind(
                attention.clone(),
                ReferenceStageParams::Attention(&block.attn),
            );

            // residual1 = block_input + attention
            stages.push(AnnStage {
                id: residual1.clone(),
                kind: StageKind::Add,
                inputs: vec![
                    StageSource::Stage(block_input.clone()),
                    StageSource::Stage(attention.clone()),
                ],
            });

            // norm2 = layer_norm(residual1)
            stages.push(AnnStage {
                id: norm2.clone(),
                kind: StageKind::Normalization {
                    kind: NormKind::Layer,
                },
                inputs: vec![StageSource::Stage(residual1.clone())],
            });
            executor = executor.bind(
                norm2.clone(),
                ReferenceStageParams::Normalization {
                    weight: &block.ln2_w,
                    bias: Some(&block.ln2_b),
                    eps,
                    kind: NormKind::Layer,
                },
            );

            // mlp = ffn(norm2)
            stages.push(AnnStage {
                id: mlp.clone(),
                kind: StageKind::Mlp {
                    kind: MlpKind::Dense,
                },
                inputs: vec![StageSource::Stage(norm2.clone())],
            });
            executor = executor.bind(mlp.clone(), ReferenceStageParams::Mlp(&block.ffn));

            // residual2 = residual1 + mlp
            stages.push(AnnStage {
                id: residual2.clone(),
                kind: StageKind::Add,
                inputs: vec![
                    StageSource::Stage(residual1.clone()),
                    StageSource::Stage(mlp.clone()),
                ],
            });

            block_input = residual2;
        }

        // final_norm = layer_norm(last residual2, or embedding if zero blocks)
        let final_norm = StageId::new("final_norm")?;
        stages.push(AnnStage {
            id: final_norm.clone(),
            kind: StageKind::Normalization {
                kind: NormKind::Layer,
            },
            inputs: vec![StageSource::Stage(block_input.clone())],
        });
        executor = executor.bind(
            final_norm.clone(),
            ReferenceStageParams::Normalization {
                weight: &model.final_ln_w,
                bias: Some(&model.final_ln_b),
                eps,
                kind: NormKind::Layer,
            },
        );

        // readout = matmul(final_norm, lm_head)
        let readout = StageId::new("readout")?;
        stages.push(AnnStage {
            id: readout.clone(),
            kind: StageKind::Readout,
            inputs: vec![StageSource::Stage(final_norm.clone())],
        });
        executor = executor.bind(
            readout,
            ReferenceStageParams::Readout {
                lm_head: &model.lm_head,
            },
        );

        let topology = AnnTopology::new(stages)?;
        Ok((executor, topology))
    }
}

impl<'p> ReferenceExecutor<'p> {
    fn ann_capabilities() -> AnnCapabilities {
        AnnCapabilities {
            backend_name: Self::BACKEND_NAME,
            supported_tags: Self::supported_tags(),
            supported_dtypes: BTreeSet::from([DType::F32]),
            stateful: false,
        }
    }

    /// Copy a reference tensor into the adapter interchange buffer.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::StageDTypeMismatch`] when `tensor` does not report
    /// [`DType::F32`]. The reference tensor type always does.
    pub fn export_hidden(tensor: &Tensor) -> Result<crate::adapter::HiddenStateBuffer> {
        let meta = tensor.meta();
        if meta.dtype != DType::F32 {
            return Err(CortexError::StageDTypeMismatch {
                stage_id: "hidden_state".to_string(),
                expected: DType::F32.to_string(),
                got: meta.dtype.to_string(),
            });
        }
        Ok(crate::adapter::HiddenStateBuffer {
            data: tensor.data().to_vec(),
            shape: meta.shape,
            dtype: DType::F32,
        })
    }

    /// Copy an interchange buffer into a reference tensor.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::StageDTypeMismatch`] for a non-`f32` buffer and
    /// [`CortexError::StageFailed`] when the element count does not match the shape.
    pub fn import_hidden(buffer: &crate::adapter::HiddenStateBuffer) -> Result<Tensor> {
        require_f32_buffer(buffer)?;
        Tensor::try_from_vec(buffer.data.clone(), &buffer.shape).map_err(|err| {
            CortexError::StageFailed {
                stage_id: "hidden_state".to_string(),
                source: Box::new(err),
            }
        })
    }
}

impl<'p> crate::adapter::ExternalAdapterMarker for ReferenceExecutor<'p> {
    fn adapter_capabilities(&self) -> crate::adapter::AdapterCapabilities {
        adapter::reference_adapter_capabilities(Self::ann_capabilities())
    }

    fn export_hidden(&self, tensor: &Self::Tensor) -> Result<crate::adapter::HiddenStateBuffer> {
        Self::export_hidden(tensor)
    }

    fn import_hidden(&self, buffer: &crate::adapter::HiddenStateBuffer) -> Result<Self::Tensor> {
        Self::import_hidden(buffer)
    }
}

impl<'p> AnnExecutor for ReferenceExecutor<'p> {
    type Tensor = Tensor;

    fn capabilities(&self) -> AnnCapabilities {
        Self::ann_capabilities()
    }

    fn execute(
        &mut self,
        stage: &AnnStage,
        inputs: &[StageInput<Self::Tensor>],
    ) -> Result<Self::Tensor> {
        let stage_id = &stage.id;

        // Reject unsupported kinds first, independent of bound params, so an
        // MoE/gated/custom stage fails the same way whether or not params exist.
        if !Self::supported_tags().contains(&stage.kind.tag()) {
            return Err(CortexError::UnsupportedStage {
                backend: Self::BACKEND_NAME,
                stage_id: stage_id.to_string(),
                kind: format!("{:?}", stage.kind),
            });
        }

        // Add stages carry no params; everything else needs a binding whose
        // kind agrees with the descriptor. Missing and mismatched params are
        // reported separately so a wiring bug is not mistaken for an unbound
        // stage.
        let params = if matches!(stage.kind, StageKind::Add) {
            None
        } else {
            let params =
                self.params
                    .get(stage_id)
                    .ok_or_else(|| CortexError::MissingStageParams {
                        stage_id: stage_id.to_string(),
                    })?;
            if params.tag() != stage.kind.tag() {
                return Err(CortexError::StageParamsMismatch {
                    stage_id: stage_id.to_string(),
                    expected: stage.kind.tag().as_str().to_string(),
                    got: params.tag().as_str().to_string(),
                });
            }
            Some(params)
        };

        match (&stage.kind, params) {
            (
                StageKind::Embedding,
                Some(&ReferenceStageParams::Embedding {
                    tok_embed,
                    pos_embed,
                    max_seq_len,
                }),
            ) => {
                Self::expect_arity(stage_id, inputs, 1)?;
                let token_ids = Self::tokens_arg(stage_id, inputs, 0)?;
                Self::run_embedding(stage_id, tok_embed, pos_embed, max_seq_len, token_ids)
            }
            (StageKind::Attention, Some(&ReferenceStageParams::Attention(attn))) => {
                Self::expect_arity(stage_id, inputs, 1)?;
                let x = Self::tensor_arg(stage_id, inputs, 0)?;
                attn.try_forward(x)
                    .map_err(|err| Self::wrap_failure(stage_id, err))
            }
            (
                StageKind::Normalization { .. },
                Some(&ReferenceStageParams::Normalization {
                    weight,
                    bias,
                    eps,
                    kind,
                }),
            ) => {
                Self::expect_arity(stage_id, inputs, 1)?;
                let x = Self::tensor_arg(stage_id, inputs, 0)?;
                let out = match (kind, bias) {
                    (NormKind::Layer, Some(bias)) => try_layer_norm(x, weight, bias, eps),
                    (NormKind::Layer, None) => {
                        return Err(CortexError::InvalidStageParams {
                            stage_id: stage_id.to_string(),
                            reason: "layer_norm requires a bias, but none is bound".to_string(),
                        });
                    }
                    (NormKind::Rms, None) => try_rms_norm(x, weight, eps),
                    // RMSNorm has no bias term in the reference math; a bound
                    // bias would be silently ignored, so reject it instead.
                    (NormKind::Rms, Some(_)) => {
                        return Err(CortexError::InvalidStageParams {
                            stage_id: stage_id.to_string(),
                            reason: "rms_norm takes no bias, but one is bound".to_string(),
                        });
                    }
                };
                out.map_err(|err| Self::wrap_failure(stage_id, err))
            }
            (StageKind::Mlp { .. }, Some(&ReferenceStageParams::Mlp(ffn))) => {
                Self::expect_arity(stage_id, inputs, 1)?;
                let x = Self::tensor_arg(stage_id, inputs, 0)?;
                ffn.try_forward(x)
                    .map_err(|err| Self::wrap_failure(stage_id, err))
            }
            (StageKind::Readout, Some(&ReferenceStageParams::Readout { lm_head })) => {
                Self::expect_arity(stage_id, inputs, 1)?;
                let x = Self::tensor_arg(stage_id, inputs, 0)?;
                try_matmul(x, lm_head).map_err(|err| Self::wrap_failure(stage_id, err))
            }
            (StageKind::Add, _) => {
                if inputs.is_empty() {
                    return Err(CortexError::StageInputArity {
                        stage_id: stage_id.to_string(),
                        expected: 1,
                        got: 0,
                    });
                }
                // Left-fold with try_add so residual1 = r0 + attn and
                // residual2 = r1 + mlp preserve operand order.
                let first = Self::tensor_arg(stage_id, inputs, 0)?;
                let mut acc = first.clone();
                for index in 1..inputs.len() {
                    let rhs = Self::tensor_arg(stage_id, inputs, index)?;
                    acc = acc
                        .try_add(rhs)
                        .map_err(|err| Self::wrap_failure(stage_id, err))?;
                }
                Ok(acc)
            }
            // Unsupported kinds were rejected above, and the tag check makes
            // a kind/params disagreement unreachable here.
            _ => Err(CortexError::UnsupportedStage {
                backend: Self::BACKEND_NAME,
                stage_id: stage_id.to_string(),
                kind: format!("{:?}", stage.kind),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::{ExternalBindings, run_topology};
    use crate::transformer::TransformerConfig;

    fn stage(id: &str, kind: StageKind, inputs: Vec<StageSource>) -> AnnStage {
        AnnStage {
            id: StageId::new(id).expect("valid id"),
            kind,
            inputs,
        }
    }

    fn micro_model() -> TransformerLM {
        let cfg = TransformerConfig {
            vocab_size: 6,
            dim: 8,
            num_heads: 2,
            num_layers: 2,
            ff_dim: 16,
            max_seq_len: 4,
        };
        TransformerLM::try_new(cfg).expect("valid micro model")
    }

    #[test]
    fn capabilities_report_reference_backend_surface() {
        let exec = ReferenceExecutor::new();
        let caps = exec.capabilities();
        assert_eq!(caps.backend_name, "reference");
        assert!(!caps.stateful);
        assert_eq!(caps.supported_dtypes, BTreeSet::from([DType::F32]));
        assert_eq!(
            caps.supported_tags,
            BTreeSet::from([
                StageKindTag::Embedding,
                StageKindTag::Attention,
                StageKindTag::LayerNorm,
                StageKindTag::RmsNorm,
                StageKindTag::DenseMlp,
                StageKindTag::Add,
                StageKindTag::Readout,
            ])
        );
        // Unsupported kinds are not advertised.
        assert!(!caps.supports(&StageKind::Mlp {
            kind: MlpKind::Gated
        }));
        assert!(!caps.supports(&StageKind::MoeRouter {
            num_experts: 2,
            top_k: 1
        }));
    }

    #[test]
    fn execute_rejects_unsupported_kinds_before_param_lookup() {
        let mut exec = ReferenceExecutor::new();
        let x = Tensor::try_from_vec(vec![0.0; 8], &[1, 8]).unwrap();

        for kind in [
            StageKind::Mlp {
                kind: MlpKind::Gated,
            },
            StageKind::Custom {
                name: "wild".to_string(),
            },
            StageKind::MoeRouter {
                num_experts: 2,
                top_k: 1,
            },
            StageKind::MoeExpert { expert_index: 0 },
        ] {
            let descriptor = stage("unsupported", kind.clone(), vec![]);
            // No params are bound: an UnsupportedStage must still be returned,
            // proving the rejection precedes param lookup.
            let err = exec
                .execute(&descriptor, &[StageInput::Tensor(&x)])
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    CortexError::UnsupportedStage {
                        backend: "reference",
                        ..
                    }
                ),
                "expected UnsupportedStage for {kind:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn execute_reports_missing_params_for_supported_stage() {
        let mut exec = ReferenceExecutor::new();
        let x = Tensor::try_from_vec(vec![0.0; 8], &[1, 8]).unwrap();
        let descriptor = stage("readout", StageKind::Readout, vec![]);
        let err = exec
            .execute(&descriptor, &[StageInput::Tensor(&x)])
            .unwrap_err();
        assert!(matches!(
            err,
            CortexError::MissingStageParams { stage_id } if stage_id == "readout"
        ));
    }

    #[test]
    fn execute_reports_params_bound_for_a_different_kind() {
        // Readout params bound under an RmsNorm stage id: the params exist,
        // they are just for the wrong kind.
        let model = micro_model();
        let norm_id = StageId::new("n").unwrap();
        let mut exec = ReferenceExecutor::new().bind(
            norm_id,
            ReferenceStageParams::Readout {
                lm_head: &model.lm_head,
            },
        );
        let x = Tensor::try_from_vec(vec![0.0; 8], &[1, 8]).unwrap();
        let descriptor = stage(
            "n",
            StageKind::Normalization {
                kind: NormKind::Rms,
            },
            vec![],
        );
        let err = exec
            .execute(&descriptor, &[StageInput::Tensor(&x)])
            .unwrap_err();
        match err {
            CortexError::StageParamsMismatch {
                stage_id,
                expected,
                got,
            } => {
                assert_eq!(stage_id, "n");
                assert_eq!(expected, "rms_norm");
                assert_eq!(got, "readout");
            }
            other => panic!("expected StageParamsMismatch, got {other:?}"),
        }
    }

    #[test]
    fn layer_norm_without_bias_is_invalid_params_not_missing() {
        let model = micro_model();
        let block = &model.blocks[0];
        let norm_id = StageId::new("n").unwrap();
        let mut exec = ReferenceExecutor::new().bind(
            norm_id,
            ReferenceStageParams::Normalization {
                weight: &block.ln1_w,
                bias: None,
                eps: ReferenceExecutor::LAYER_NORM_EPS,
                kind: NormKind::Layer,
            },
        );
        let x = Tensor::randn(&[2, model.config.dim], 0.0, 0.1);
        let descriptor = stage(
            "n",
            StageKind::Normalization {
                kind: NormKind::Layer,
            },
            vec![],
        );
        let err = exec
            .execute(&descriptor, &[StageInput::Tensor(&x)])
            .unwrap_err();
        assert!(
            matches!(err, CortexError::InvalidStageParams { ref stage_id, .. } if stage_id == "n"),
            "expected InvalidStageParams, got {err:?}"
        );
    }

    #[test]
    fn rms_norm_rejects_a_bias_instead_of_ignoring_it() {
        let model = micro_model();
        let block = &model.blocks[0];
        let norm_id = StageId::new("n").unwrap();
        let mut exec = ReferenceExecutor::new().bind(
            norm_id,
            ReferenceStageParams::Normalization {
                weight: &block.ln1_w,
                bias: Some(&block.ln1_b),
                eps: ReferenceExecutor::LAYER_NORM_EPS,
                kind: NormKind::Rms,
            },
        );
        let x = Tensor::randn(&[2, model.config.dim], 0.0, 0.1);
        let descriptor = stage(
            "n",
            StageKind::Normalization {
                kind: NormKind::Rms,
            },
            vec![],
        );
        let err = exec
            .execute(&descriptor, &[StageInput::Tensor(&x)])
            .unwrap_err();
        assert!(
            matches!(err, CortexError::InvalidStageParams { ref stage_id, .. } if stage_id == "n"),
            "expected InvalidStageParams, got {err:?}"
        );
    }

    #[test]
    fn rms_norm_stage_matches_kernel_bitwise() {
        let model = micro_model();
        let block = &model.blocks[0];
        let norm_id = StageId::new("n").unwrap();
        let eps = ReferenceExecutor::LAYER_NORM_EPS;
        let mut exec = ReferenceExecutor::new().bind(
            norm_id,
            ReferenceStageParams::Normalization {
                weight: &block.ln1_w,
                bias: None,
                eps,
                kind: NormKind::Rms,
            },
        );
        let x = Tensor::randn(&[2, model.config.dim], 0.0, 0.1);
        let descriptor = stage(
            "n",
            StageKind::Normalization {
                kind: NormKind::Rms,
            },
            vec![],
        );
        let staged = exec
            .execute(&descriptor, &[StageInput::Tensor(&x)])
            .unwrap();
        let direct = try_rms_norm(&x, &block.ln1_w, eps).unwrap();
        assert_eq!(staged.data(), direct.data());
    }

    #[test]
    fn public_layer_norm_eps_matches_transformer_constant() {
        assert_eq!(
            ReferenceExecutor::LAYER_NORM_EPS,
            crate::transformer::LAYER_NORM_EPS
        );
    }

    #[test]
    fn execute_reports_wrong_arity() {
        let model = micro_model();
        let lm_head = &model.lm_head;
        let readout_id = StageId::new("readout").unwrap();
        let mut exec =
            ReferenceExecutor::new().bind(readout_id, ReferenceStageParams::Readout { lm_head });
        let descriptor = stage("readout", StageKind::Readout, vec![]);
        let a = Tensor::try_from_vec(vec![0.0; 8], &[1, 8]).unwrap();
        let b = Tensor::try_from_vec(vec![0.0; 8], &[1, 8]).unwrap();
        let err = exec
            .execute(
                &descriptor,
                &[StageInput::Tensor(&a), StageInput::Tensor(&b)],
            )
            .unwrap_err();
        assert!(matches!(
            err,
            CortexError::StageInputArity {
                expected: 1,
                got: 2,
                ..
            }
        ));
    }

    #[test]
    fn execute_reports_wrong_input_kind() {
        // A readout stage expects a Tensor; passing Tokens is a kind mismatch.
        let model = micro_model();
        let lm_head = &model.lm_head;
        let readout_id = StageId::new("readout").unwrap();
        let mut exec =
            ReferenceExecutor::new().bind(readout_id, ReferenceStageParams::Readout { lm_head });
        let descriptor = stage("readout", StageKind::Readout, vec![]);
        let ids = [0u32, 1u32];
        let err = exec
            .execute(&descriptor, &[StageInput::Tokens(&ids)])
            .unwrap_err();
        assert!(matches!(err, CortexError::StageInputKind { index: 0, .. }));
    }

    #[test]
    fn wrong_hidden_width_surfaces_as_stage_failed() {
        // Feed the attention stage a tensor whose feature width is not `dim`;
        // the kernel rejects it and the executor wraps it as StageFailed.
        let model = micro_model();
        let attn = &model.blocks[0].attn;
        let attn_id = StageId::new("attn").unwrap();
        let mut exec =
            ReferenceExecutor::new().bind(attn_id, ReferenceStageParams::Attention(attn));
        let descriptor = stage("attn", StageKind::Attention, vec![]);
        let wrong = Tensor::try_from_vec(vec![0.0; 3], &[1, 3]).unwrap(); // dim=8 expected
        let err = exec
            .execute(&descriptor, &[StageInput::Tensor(&wrong)])
            .unwrap_err();
        match err {
            CortexError::StageFailed { stage_id, source } => {
                assert_eq!(stage_id, "attn");
                assert!(matches!(*source, CortexError::DimMismatch { .. }));
            }
            other => panic!("expected StageFailed, got {other:?}"),
        }
    }

    #[test]
    fn from_transformer_rejects_malformed_model() {
        let mut model = micro_model();
        // Corrupt a tensor shape via the public field so validate_wire fails.
        model.lm_head = Tensor::try_from_vec(vec![0.0; 4], &[2, 2]).unwrap();
        let err = ReferenceExecutor::from_transformer(&model).unwrap_err();
        assert!(
            matches!(err, CortexError::ShapeMismatch { .. }),
            "expected a structured validation error, got {err:?}"
        );
    }

    #[test]
    fn from_transformer_builds_expected_topology_order() {
        let model = micro_model();
        let (_exec, topo) = ReferenceExecutor::from_transformer(&model).unwrap();
        let ids: Vec<&str> = topo.stages().iter().map(|s| s.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "embedding",
                "blocks.0.norm1",
                "blocks.0.attention",
                "blocks.0.residual1",
                "blocks.0.norm2",
                "blocks.0.mlp",
                "blocks.0.residual2",
                "blocks.1.norm1",
                "blocks.1.attention",
                "blocks.1.residual1",
                "blocks.1.norm2",
                "blocks.1.mlp",
                "blocks.1.residual2",
                "final_norm",
                "readout",
            ]
        );
    }

    #[test]
    fn single_normalization_stage_matches_kernel_bitwise() {
        let model = micro_model();
        let block = &model.blocks[0];
        let norm_id = StageId::new("blocks.0.norm1").unwrap();
        let mut exec = ReferenceExecutor::new().bind(
            norm_id,
            ReferenceStageParams::Normalization {
                weight: &block.ln1_w,
                bias: Some(&block.ln1_b),
                eps: ReferenceExecutor::LAYER_NORM_EPS,
                kind: NormKind::Layer,
            },
        );
        let x = Tensor::randn(&[3, model.config.dim], 0.0, 0.1);
        let descriptor = stage(
            "blocks.0.norm1",
            StageKind::Normalization {
                kind: NormKind::Layer,
            },
            vec![],
        );
        let staged = exec
            .execute(&descriptor, &[StageInput::Tensor(&x)])
            .unwrap();
        let direct = try_layer_norm(
            &x,
            &block.ln1_w,
            &block.ln1_b,
            crate::transformer::LAYER_NORM_EPS,
        )
        .unwrap();
        assert_eq!(staged.data(), direct.data());
        assert_eq!(staged.shape(), direct.shape());
    }

    #[test]
    fn single_readout_stage_matches_kernel_bitwise() {
        let model = micro_model();
        let lm_head = &model.lm_head;
        let readout_id = StageId::new("readout").unwrap();
        let mut exec =
            ReferenceExecutor::new().bind(readout_id, ReferenceStageParams::Readout { lm_head });
        let x = Tensor::randn(&[2, model.config.dim], 0.0, 0.1);
        let descriptor = stage("readout", StageKind::Readout, vec![]);
        let staged = exec
            .execute(&descriptor, &[StageInput::Tensor(&x)])
            .unwrap();
        let direct = try_matmul(&x, lm_head).unwrap();
        assert_eq!(staged.data(), direct.data());
    }

    #[test]
    fn composed_topology_matches_try_forward_bitwise() {
        let model = micro_model();
        let (mut exec, topo) = ReferenceExecutor::from_transformer(&model).unwrap();
        let ids = [1u32, 3, 0];
        let bindings = ExternalBindings::new().with_tokens("tokens", &ids).unwrap();
        let outputs = run_topology(&mut exec, &topo, &bindings).unwrap();
        let readout = outputs
            .get(&StageId::new("readout").unwrap())
            .expect("readout output");
        let expected = model.try_forward(&ids).unwrap();
        assert_eq!(readout.data(), expected.data());
        assert_eq!(readout.shape(), expected.shape());
    }
}
