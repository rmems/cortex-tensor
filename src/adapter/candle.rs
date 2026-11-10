// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU `f32` [`AnnExecutor`](crate::stage::AnnExecutor) backed by Candle.
//!
//! Hidden state enters and leaves through [`import_hidden_state`] and
//! [`export_hidden_state`]. Weights are converted once, when a stage is bound.
//! Stage bodies call Candle modules (`Embedding`, `Linear`, `LayerNorm`,
//! `RmsNorm`, `Activation`) and Candle's CPU flash-attention kernel. They do
//! not reimplement those kernels. Candle's fused `sdpa` op has no CPU
//! implementation and its CUDA/Metal features are not enabled here.
//!
//! MoE router, MoE expert, and custom stages are rejected with
//! [`CortexError::UnsupportedStage`](crate::CortexError::UnsupportedStage).

use crate::error::{CortexError, Result};
use crate::stage::{
    AnnCapabilities, AnnExecutor, AnnStage, DType, MlpKind, NormKind, StageId, StageInput,
    StageKind, StageKindTag, StageTensor, TensorMeta,
};
use crate::tensor::checked_numel;
use candle_core::{Device, Tensor};
use candle_nn::Module;
use candle_nn::attention::{AttnMask, flash_attn};
use candle_nn::{Activation, Embedding, LayerNorm, Linear, RmsNorm};
use std::collections::{BTreeMap, BTreeSet};

/// Row-major `f32` hidden state at the Candle boundary.
///
/// This is not a cross-backend wire format. It is the explicit copy between
/// caller-owned `f32` storage and a Candle CPU tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenState {
    /// Contiguous row-major elements, outermost dimension first.
    pub data: Vec<f32>,
    /// Extents matching [`Self::data`].
    pub shape: Vec<usize>,
}

/// Candle-owned tensor carried between stages.
///
/// The newtype keeps `candle_core::Tensor` inside this adapter. Callers move
/// values across the boundary with [`import_hidden_state`] and
/// [`export_hidden_state`].
#[derive(Clone)]
pub struct CandleTensor {
    inner: Tensor,
}

impl std::fmt::Debug for CandleTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleTensor")
            .field("shape", &self.inner.dims())
            .field("dtype", &self.inner.dtype())
            .finish()
    }
}

impl StageTensor for CandleTensor {
    fn meta(&self) -> TensorMeta {
        TensorMeta {
            shape: self.inner.dims().to_vec(),
            dtype: DType::F32,
        }
    }
}

impl CandleTensor {
    /// Row-major extents, outermost dimension first.
    pub fn shape(&self) -> &[usize] {
        self.inner.dims()
    }

    /// Wrap a Candle tensor after checking it is CPU `f32`.
    ///
    /// `stage_id` names the stage (or `"hidden_state"` at the import boundary)
    /// in the typed refusal.
    fn from_candle(inner: Tensor, stage_id: &str) -> Result<Self> {
        require_cpu_f32(&inner, stage_id)?;
        let inner = inner.contiguous().map_err(candle_msg)?;
        Ok(Self { inner })
    }
}

/// Stage id used for refusals at the hidden-state import/export boundary,
/// matching [`ReferenceExecutor::import_hidden`](crate::stage::ReferenceExecutor::import_hidden).
const HIDDEN_STATE: &str = "hidden_state";

/// Reject a non-`f32` tensor with [`CortexError::StageDTypeMismatch`] and a
/// non-CPU tensor with [`CortexError::UnsupportedOperation`] (`category: "device"`).
fn require_cpu_f32(inner: &Tensor, stage_id: &str) -> Result<()> {
    if inner.dtype() != candle_core::DType::F32 {
        return Err(CortexError::StageDTypeMismatch {
            stage_id: stage_id.to_string(),
            expected: DType::F32.to_string(),
            got: inner.dtype().as_str().to_string(),
        });
    }
    if !inner.device().is_cpu() {
        return Err(CortexError::UnsupportedOperation {
            backend: CandleExecutor::BACKEND_NAME,
            stage_id: (stage_id != HIDDEN_STATE).then(|| stage_id.to_string()),
            category: "device",
            detail: format!(
                "candle adapter only carries CPU tensors, got {:?}",
                inner.device().location()
            ),
        });
    }
    Ok(())
}

/// Copy `state` into a contiguous CPU `f32` Candle tensor.
///
/// # Errors
///
/// Returns [`CortexError::SizeOverflow`] when the element count or Candle's
/// shape arithmetic overflows,
/// [`CortexError::ShapeMismatch`] when `data.len()` disagrees with the element
/// count, or a `candle:`-prefixed [`CortexError::Msg`] when Candle rejects the
/// buffer.
///
/// A zero-sized shape whose non-zero prefix overflows `usize` (for example
/// `[usize::MAX, usize::MAX, 0]`) is accepted by
/// [`ReferenceExecutor::import_hidden`](crate::stage::ReferenceExecutor::import_hidden)
/// but returns `SizeOverflow` here: candle-core 0.11 multiplies every extent
/// without short-circuiting on zero and panics on the overflow.
pub fn import_hidden_state(state: &HiddenState) -> Result<CandleTensor> {
    let inner = f32_tensor(&state.data, &state.shape, &Device::Cpu)?;
    CandleTensor::from_candle(inner, HIDDEN_STATE)
}

/// Copy `tensor` back to row-major `f32` storage.
///
/// # Errors
///
/// Returns [`CortexError::StageDTypeMismatch`] when the tensor is not `f32`,
/// [`CortexError::UnsupportedOperation`] (`category: "device"`) when it is not
/// on the CPU, or a `candle:`-prefixed [`CortexError::Msg`] when Candle cannot
/// read it back.
pub fn export_hidden_state(tensor: &CandleTensor) -> Result<HiddenState> {
    require_cpu_f32(&tensor.inner, HIDDEN_STATE)?;
    let flat = tensor
        .inner
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(candle_msg)?;
    Ok(HiddenState {
        data: flat,
        shape: tensor.inner.dims().to_vec(),
    })
}

/// Import a reference-backend tensor at the Candle boundary.
///
/// # Errors
///
/// See [`import_hidden_state`].
pub fn import_reference(tensor: &crate::Tensor) -> Result<CandleTensor> {
    import_hidden_state(&HiddenState {
        data: tensor.data().to_vec(),
        shape: tensor.shape().to_vec(),
    })
}

/// Export a Candle tensor into a reference-backend tensor.
///
/// # Errors
///
/// See [`export_hidden_state`]. A shape Candle accepted but the reference
/// constructor rejects is returned unchanged.
pub fn export_reference(tensor: &CandleTensor) -> Result<crate::Tensor> {
    let state = export_hidden_state(tensor)?;
    crate::Tensor::try_from_vec(state.data, &state.shape)
}

/// Token and position tables for an [`StageKind::Embedding`](crate::stage::StageKind::Embedding) stage.
///
/// Both tables are row-major `[rows, dim]`, which is already Candle
/// `Embedding` layout, so binding them does not transpose.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingWeights<'a> {
    /// Token table, length `vocab * dim`.
    pub token_weight: &'a [f32],
    /// Rows in [`Self::token_weight`].
    pub vocab: usize,
    /// Columns in both tables.
    pub dim: usize,
    /// Position table, length `max_seq_len * dim`.
    pub position_weight: &'a [f32],
    /// Rows in [`Self::position_weight`].
    pub max_seq_len: usize,
}

/// One Candle `Linear`, bound from cortex row-major layout.
///
/// `weight` is `[in_dim, out_dim]` (the same layout as `Tensor` matmul). The
/// adapter transposes it into Candle's `[out_dim, in_dim]` weight once, at
/// bind time. `bias` is optional and has length `out_dim` when present.
#[derive(Debug, Clone, Copy)]
pub struct LinearWeights<'a> {
    /// Row-major `[in_dim, out_dim]` matrix.
    pub weight: &'a [f32],
    /// Input features.
    pub in_dim: usize,
    /// Output features.
    pub out_dim: usize,
    /// Optional bias of length `out_dim`.
    pub bias: Option<&'a [f32]>,
}

/// Q, K, V, and output projections for causal multi-head self-attention.
///
/// Each projection is a square `[dim, dim]` matrix in cortex layout. `dim`
/// must equal `num_heads * head_dim` with `head_dim = dim / num_heads`.
#[derive(Debug, Clone, Copy)]
pub struct AttentionWeights<'a> {
    /// Query projection.
    pub q: LinearWeights<'a>,
    /// Key projection.
    pub k: LinearWeights<'a>,
    /// Value projection.
    pub v: LinearWeights<'a>,
    /// Output projection.
    pub o: LinearWeights<'a>,
    /// Number of query heads. Must divide the projection width.
    pub num_heads: usize,
}

/// LayerNorm affine parameters. `weight` and `bias` each have length `hidden`.
#[derive(Debug, Clone, Copy)]
pub struct LayerNormWeights<'a> {
    /// Per-feature scale.
    pub weight: &'a [f32],
    /// Per-feature shift.
    pub bias: &'a [f32],
    /// Positive finite epsilon.
    pub eps: f32,
}

/// RMSNorm scale. `weight` has length `hidden`. There is no bias.
#[derive(Debug, Clone, Copy)]
pub struct RmsNormWeights<'a> {
    /// Per-feature scale.
    pub weight: &'a [f32],
    /// Positive finite epsilon.
    pub eps: f32,
}

/// Dense MLP: Candle `Linear`, then Candle's tanh-approximate GELU, then `Linear`.
#[derive(Debug, Clone, Copy)]
pub struct MlpWeights<'a> {
    /// Up projection.
    pub up: LinearWeights<'a>,
    /// Down projection. Its `in_dim` must equal `up.out_dim`.
    pub down: LinearWeights<'a>,
}

/// Gated MLP. Candle `swiglu` combines `silu(gate(x))` with `up(x)`.
#[derive(Debug, Clone, Copy)]
pub struct GatedMlpWeights<'a> {
    /// Gate projection. Same shape as [`Self::up`].
    pub gate: LinearWeights<'a>,
    /// Value projection.
    pub up: LinearWeights<'a>,
    /// Down projection. Its `in_dim` must equal `up.out_dim`.
    pub down: LinearWeights<'a>,
}

#[derive(Clone)]
struct EmbeddingParams {
    tokens: Embedding,
    positions: Embedding,
    vocab: usize,
    max_seq_len: usize,
    dim: usize,
}

#[derive(Clone)]
struct AttentionParams {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: usize,
    head_dim: usize,
    dim: usize,
}

#[derive(Clone)]
struct MlpParams {
    up: Linear,
    down: Linear,
}

#[derive(Clone)]
struct GatedParams {
    gate: Linear,
    up: Linear,
    down: Linear,
}

#[derive(Clone)]
enum StageParams {
    Embedding(EmbeddingParams),
    Attention(AttentionParams),
    LayerNorm(LayerNorm),
    RmsNorm(RmsNorm),
    DenseMlp(MlpParams),
    GatedMlp(GatedParams),
    Readout(Linear),
}

impl StageParams {
    fn tag(&self) -> StageKindTag {
        match self {
            StageParams::Embedding(_) => StageKindTag::Embedding,
            StageParams::Attention(_) => StageKindTag::Attention,
            StageParams::LayerNorm(_) => StageKindTag::LayerNorm,
            StageParams::RmsNorm(_) => StageKindTag::RmsNorm,
            StageParams::DenseMlp(_) => StageKindTag::DenseMlp,
            StageParams::GatedMlp(_) => StageKindTag::GatedMlp,
            StageParams::Readout(_) => StageKindTag::Readout,
        }
    }
}

/// Candle CPU executor.
///
/// Stateless across calls: parameters are fixed at bind time and `execute`
/// does not mutate them. A rejected stage leaves the binding map unchanged.
#[derive(Clone)]
pub struct CandleExecutor {
    device: Device,
    params: BTreeMap<StageId, StageParams>,
}

impl std::fmt::Debug for CandleExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stages: Vec<&str> = self.params.keys().map(StageId::as_str).collect();
        f.debug_struct("CandleExecutor")
            .field("backend", &Self::BACKEND_NAME)
            .field("stages", &stages)
            .finish()
    }
}

impl Default for CandleExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl CandleExecutor {
    /// Backend name reported by [`AnnCapabilities::backend_name`].
    pub const BACKEND_NAME: &'static str = "candle";

    /// CPU executor with no bound stages.
    pub fn new() -> Self {
        Self {
            device: Device::Cpu,
            params: BTreeMap::new(),
        }
    }

    /// Tags this backend can execute.
    fn supported_tags() -> BTreeSet<StageKindTag> {
        BTreeSet::from([
            StageKindTag::Embedding,
            StageKindTag::Attention,
            StageKindTag::LayerNorm,
            StageKindTag::RmsNorm,
            StageKindTag::DenseMlp,
            StageKindTag::GatedMlp,
            StageKindTag::Add,
            StageKindTag::Readout,
        ])
    }

    /// Bind token and position embeddings, replacing any previous params for `id`.
    ///
    /// # Errors
    ///
    /// Returns a shape or configuration error when the tables do not match
    /// `vocab`, `dim`, and `max_seq_len`. On error, existing bindings are kept.
    pub fn bind_embedding(&mut self, id: StageId, weights: EmbeddingWeights<'_>) -> Result<()> {
        let params = prepare_embedding(&self.device, &weights)?;
        self.params.insert(id, StageParams::Embedding(params));
        Ok(())
    }

    /// Bind causal multi-head self-attention projections.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidConfig`] when a projection is not square,
    /// the four widths disagree, or `num_heads` does not divide that width.
    pub fn bind_attention(&mut self, id: StageId, weights: AttentionWeights<'_>) -> Result<()> {
        let params = prepare_attention(&self.device, &weights)?;
        self.params.insert(id, StageParams::Attention(params));
        Ok(())
    }

    /// Bind LayerNorm scale and bias over the last axis.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidEpsilon`], [`CortexError::ZeroWidth`], or
    /// [`CortexError::ShapeMismatch`] when the affine vectors disagree.
    pub fn bind_layer_norm(&mut self, id: StageId, weights: LayerNormWeights<'_>) -> Result<()> {
        let hidden = norm_width(
            weights.weight,
            weights.bias,
            weights.eps,
            "candle layer_norm",
        )?;
        let scale = f32_tensor(weights.weight, &[hidden], &self.device)?;
        let bias = f32_tensor(weights.bias, &[hidden], &self.device)?;
        self.params.insert(
            id,
            StageParams::LayerNorm(LayerNorm::new(scale, bias, f64::from(weights.eps))),
        );
        Ok(())
    }

    /// Bind RMSNorm scale over the last axis.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidEpsilon`] or [`CortexError::ZeroWidth`].
    pub fn bind_rms_norm(&mut self, id: StageId, weights: RmsNormWeights<'_>) -> Result<()> {
        let hidden = scale_width(weights.weight, weights.eps, "candle rms_norm")?;
        let scale = f32_tensor(weights.weight, &[hidden], &self.device)?;
        self.params.insert(
            id,
            StageParams::RmsNorm(RmsNorm::new(scale, f64::from(weights.eps))),
        );
        Ok(())
    }

    /// Bind a dense GELU MLP.
    ///
    /// # Errors
    ///
    /// Returns a shape error from either projection, or
    /// [`CortexError::InvalidConfig`] when `down.in_dim != up.out_dim`.
    pub fn bind_dense_mlp(&mut self, id: StageId, weights: MlpWeights<'_>) -> Result<()> {
        if weights.up.out_dim != weights.down.in_dim {
            return Err(CortexError::InvalidConfig(format!(
                "candle dense MLP down.in_dim ({}) must equal up.out_dim ({})",
                weights.down.in_dim, weights.up.out_dim
            )));
        }
        let up = linear_from(&self.device, &weights.up)?;
        let down = linear_from(&self.device, &weights.down)?;
        self.params
            .insert(id, StageParams::DenseMlp(MlpParams { up, down }));
        Ok(())
    }

    /// Bind a SwiGLU MLP. `gate` and `up` must share input and output widths.
    ///
    /// # Errors
    ///
    /// Returns a shape error from any projection, or
    /// [`CortexError::InvalidConfig`] when the three widths do not line up.
    pub fn bind_gated_mlp(&mut self, id: StageId, weights: GatedMlpWeights<'_>) -> Result<()> {
        if weights.gate.in_dim != weights.up.in_dim || weights.gate.out_dim != weights.up.out_dim {
            return Err(CortexError::InvalidConfig(
                "candle gated MLP gate and up projections must share in_dim and out_dim"
                    .to_string(),
            ));
        }
        if weights.up.out_dim != weights.down.in_dim {
            return Err(CortexError::InvalidConfig(format!(
                "candle gated MLP down.in_dim ({}) must equal up.out_dim ({})",
                weights.down.in_dim, weights.up.out_dim
            )));
        }
        let gate = linear_from(&self.device, &weights.gate)?;
        let up = linear_from(&self.device, &weights.up)?;
        let down = linear_from(&self.device, &weights.down)?;
        self.params
            .insert(id, StageParams::GatedMlp(GatedParams { gate, up, down }));
        Ok(())
    }

    /// Bind a bias-optional readout projection.
    ///
    /// # Errors
    ///
    /// Returns the same shape errors as [`LinearWeights`].
    pub fn bind_readout(&mut self, id: StageId, weights: LinearWeights<'_>) -> Result<()> {
        let linear = linear_from(&self.device, &weights)?;
        self.params.insert(id, StageParams::Readout(linear));
        Ok(())
    }

    fn bound<'a>(&'a self, stage_id: &StageId, expected: StageKindTag) -> Result<&'a StageParams> {
        let params = self
            .params
            .get(stage_id)
            .ok_or_else(|| CortexError::MissingStageParams {
                stage_id: stage_id.to_string(),
            })?;
        // Same split as `ReferenceExecutor` (#97): a binding of the wrong
        // kind is a wiring bug, not an unbound stage, and is reported as such.
        if params.tag() != expected {
            return Err(CortexError::StageParamsMismatch {
                stage_id: stage_id.to_string(),
                expected,
                got: params.tag(),
            });
        }
        Ok(params)
    }
}

impl AnnExecutor for CandleExecutor {
    type Tensor = CandleTensor;

    fn capabilities(&self) -> AnnCapabilities {
        AnnCapabilities {
            backend_name: Self::BACKEND_NAME,
            supported_tags: Self::supported_tags(),
            supported_dtypes: BTreeSet::from([DType::F32]),
            stateful: false,
        }
    }

    fn execute(
        &mut self,
        stage: &AnnStage,
        inputs: &[StageInput<Self::Tensor>],
    ) -> Result<Self::Tensor> {
        let stage_id = &stage.id;
        if !Self::supported_tags().contains(&stage.kind.tag()) {
            return Err(CortexError::UnsupportedStage {
                backend: Self::BACKEND_NAME,
                stage_id: stage_id.to_string(),
                kind: stage.kind.clone(),
            });
        }

        match &stage.kind {
            StageKind::Embedding => {
                let StageParams::Embedding(params) =
                    self.bound(stage_id, StageKindTag::Embedding)?
                else {
                    return Err(missing(stage_id));
                };
                let params = params.clone();
                run_embedding(stage_id, &self.device, &params, inputs)
            }
            StageKind::Attention => {
                let StageParams::Attention(params) =
                    self.bound(stage_id, StageKindTag::Attention)?
                else {
                    return Err(missing(stage_id));
                };
                let params = params.clone();
                run_attention(stage_id, &self.device, &params, inputs)
            }
            StageKind::Normalization {
                kind: NormKind::Layer,
            } => {
                let StageParams::LayerNorm(norm) = self.bound(stage_id, StageKindTag::LayerNorm)?
                else {
                    return Err(missing(stage_id));
                };
                let norm = norm.clone();
                let width = norm.weight().elem_count();
                run_module(stage_id, inputs, width, width, |x| norm.forward(x))
            }
            StageKind::Normalization {
                kind: NormKind::Rms,
            } => {
                let StageParams::RmsNorm(norm) = self.bound(stage_id, StageKindTag::RmsNorm)?
                else {
                    return Err(missing(stage_id));
                };
                let norm = norm.clone();
                let width = norm.weight().elem_count();
                run_module(stage_id, inputs, width, width, |x| norm.forward(x))
            }
            StageKind::Mlp {
                kind: MlpKind::Dense,
            } => {
                let StageParams::DenseMlp(params) = self.bound(stage_id, StageKindTag::DenseMlp)?
                else {
                    return Err(missing(stage_id));
                };
                let params = params.clone();
                run_dense_mlp(stage_id, &params, inputs)
            }
            StageKind::Mlp {
                kind: MlpKind::Gated,
            } => {
                let StageParams::GatedMlp(params) = self.bound(stage_id, StageKindTag::GatedMlp)?
                else {
                    return Err(missing(stage_id));
                };
                let params = params.clone();
                run_gated_mlp(stage_id, &params, inputs)
            }
            StageKind::Readout => {
                let StageParams::Readout(linear) = self.bound(stage_id, StageKindTag::Readout)?
                else {
                    return Err(missing(stage_id));
                };
                let linear = linear.clone();
                let in_dim = linear_in_dim(&linear)?;
                let out_dim = linear_out_dim(&linear)?;
                run_module(stage_id, inputs, in_dim, out_dim, |x| linear.forward(x))
            }
            StageKind::Add => run_add(stage_id, inputs),
            StageKind::MoeRouter { .. }
            | StageKind::MoeExpert { .. }
            | StageKind::Custom { .. } => Err(CortexError::UnsupportedStage {
                backend: Self::BACKEND_NAME,
                stage_id: stage_id.to_string(),
                kind: stage.kind.clone(),
            }),
        }
    }
}

fn missing(stage_id: &StageId) -> CortexError {
    CortexError::MissingStageParams {
        stage_id: stage_id.to_string(),
    }
}

/// Residual Candle failure with no typed [`CortexError`] counterpart.
///
/// Shape, rank, dtype, and device failures are checked before Candle runs and
/// reported with typed variants; only Candle-internal errors reach this path.
/// The `candle: ` prefix identifies the source.
fn candle_msg(err: candle_core::Error) -> CortexError {
    CortexError::Msg(format!("candle: {err}"))
}

fn stage_failed(stage_id: &StageId, err: CortexError) -> CortexError {
    CortexError::StageFailed {
        stage_id: stage_id.to_string(),
        source: Box::new(err),
    }
}

fn fail_candle(stage_id: &StageId, err: candle_core::Error) -> CortexError {
    stage_failed(stage_id, candle_msg(err))
}

fn expect_arity(
    stage_id: &StageId,
    inputs: &[StageInput<'_, CandleTensor>],
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

fn tensor_arg<'a>(
    stage_id: &StageId,
    inputs: &'a [StageInput<'a, CandleTensor>],
    index: usize,
) -> Result<&'a CandleTensor> {
    match &inputs[index] {
        StageInput::Tensor(tensor) => Ok(tensor),
        StageInput::Tokens(_) => Err(CortexError::StageInputKind {
            stage_id: stage_id.to_string(),
            index,
        }),
    }
}

fn tokens_arg<'a>(
    stage_id: &StageId,
    inputs: &'a [StageInput<'a, CandleTensor>],
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

/// Whether candle-core 0.11 can represent `shape` without overflowing.
///
/// `Shape::elem_count` multiplies every extent left to right and
/// `Shape::stride_contiguous` multiplies every extent right to left, neither
/// short-circuiting on a zero axis, and both panic on overflow in debug
/// builds. So every prefix and every suffix product must fit in `usize`, even
/// when [`checked_numel`] is `0`. Returns [`CortexError::SizeOverflow`] otherwise.
fn candle_shape_fits(shape: &[usize]) -> Result<()> {
    let overflow = || CortexError::SizeOverflow {
        shape: shape.to_vec(),
    };
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(overflow)?;
    shape
        .iter()
        .rev()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(overflow)?;
    Ok(())
}

fn expect_matrix(data: &[f32], rows: usize, cols: usize) -> Result<()> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| CortexError::SizeOverflow {
            shape: vec![rows, cols],
        })?;
    if data.len() != expected {
        return Err(CortexError::ShapeMismatch {
            expected: vec![rows, cols],
            got: vec![data.len()],
        });
    }
    Ok(())
}

fn f32_tensor(data: &[f32], shape: &[usize], device: &Device) -> Result<Tensor> {
    let numel = checked_numel(shape)?;
    candle_shape_fits(shape)?;
    if data.len() != numel {
        return Err(CortexError::ShapeMismatch {
            expected: shape.to_vec(),
            got: vec![data.len()],
        });
    }
    Tensor::from_slice(data, shape, device).map_err(candle_msg)
}

/// Transpose row-major `[in_dim, out_dim]` into Candle `[out_dim, in_dim]`.
fn transpose_in_out(weight: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; weight.len()];
    for in_index in 0..in_dim {
        for out_index in 0..out_dim {
            transposed[out_index * in_dim + in_index] = weight[in_index * out_dim + out_index];
        }
    }
    transposed
}

fn linear_from(device: &Device, weights: &LinearWeights<'_>) -> Result<Linear> {
    if weights.in_dim == 0 || weights.out_dim == 0 {
        return Err(CortexError::InvalidConfig(
            "candle linear in_dim and out_dim must be non-zero".to_string(),
        ));
    }
    let expected =
        weights
            .in_dim
            .checked_mul(weights.out_dim)
            .ok_or_else(|| CortexError::SizeOverflow {
                shape: vec![weights.in_dim, weights.out_dim],
            })?;
    if weights.weight.len() != expected {
        return Err(CortexError::ShapeMismatch {
            expected: vec![weights.in_dim, weights.out_dim],
            got: vec![weights.weight.len()],
        });
    }
    if let Some(bias) = weights.bias
        && bias.len() != weights.out_dim
    {
        return Err(CortexError::ShapeMismatch {
            expected: vec![weights.out_dim],
            got: vec![bias.len()],
        });
    }
    let transposed = transpose_in_out(weights.weight, weights.in_dim, weights.out_dim);
    let weight = f32_tensor(&transposed, &[weights.out_dim, weights.in_dim], device)?;
    let bias = match weights.bias {
        Some(bias) => Some(f32_tensor(bias, &[weights.out_dim], device)?),
        None => None,
    };
    Ok(Linear::new(weight, bias))
}

fn linear_in_dim(linear: &Linear) -> Result<usize> {
    linear.weight().dims().get(1).copied().ok_or_else(|| {
        CortexError::InvalidConfig(
            "candle linear weight must be rank 2 [out_dim, in_dim]".to_string(),
        )
    })
}

fn linear_out_dim(linear: &Linear) -> Result<usize> {
    match linear.weight().dims() {
        [out_dim, _] => Ok(*out_dim),
        _ => Err(CortexError::InvalidConfig(
            "candle linear weight must be rank 2 [out_dim, in_dim]".to_string(),
        )),
    }
}

fn scale_width(weight: &[f32], eps: f32, op: &'static str) -> Result<usize> {
    if !(eps.is_finite() && eps > 0.0) {
        return Err(CortexError::InvalidEpsilon { eps });
    }
    if weight.is_empty() {
        return Err(CortexError::ZeroWidth { op, axis: 0 });
    }
    Ok(weight.len())
}

fn norm_width(weight: &[f32], bias: &[f32], eps: f32, op: &'static str) -> Result<usize> {
    let hidden = scale_width(weight, eps, op)?;
    if weight.len() != bias.len() {
        return Err(CortexError::ShapeMismatch {
            expected: vec![weight.len()],
            got: vec![bias.len()],
        });
    }
    Ok(hidden)
}

fn prepare_embedding(device: &Device, weights: &EmbeddingWeights<'_>) -> Result<EmbeddingParams> {
    if weights.vocab == 0 || weights.dim == 0 || weights.max_seq_len == 0 {
        return Err(CortexError::InvalidConfig(
            "candle embedding vocab, dim, and max_seq_len must be non-zero".to_string(),
        ));
    }
    expect_matrix(weights.token_weight, weights.vocab, weights.dim)?;
    expect_matrix(weights.position_weight, weights.max_seq_len, weights.dim)?;
    let tokens = Embedding::new(
        f32_tensor(weights.token_weight, &[weights.vocab, weights.dim], device)?,
        weights.dim,
    );
    let positions = Embedding::new(
        f32_tensor(
            weights.position_weight,
            &[weights.max_seq_len, weights.dim],
            device,
        )?,
        weights.dim,
    );
    Ok(EmbeddingParams {
        tokens,
        positions,
        vocab: weights.vocab,
        max_seq_len: weights.max_seq_len,
        dim: weights.dim,
    })
}

fn square_dim(name: &str, weights: &LinearWeights<'_>) -> Result<usize> {
    if weights.in_dim != weights.out_dim {
        return Err(CortexError::InvalidConfig(format!(
            "candle attention {name} projection must be square, got {}x{}",
            weights.in_dim, weights.out_dim
        )));
    }
    Ok(weights.in_dim)
}

fn prepare_attention(device: &Device, weights: &AttentionWeights<'_>) -> Result<AttentionParams> {
    let dim = square_dim("q", &weights.q)?;
    for (name, projection) in [("k", &weights.k), ("v", &weights.v), ("o", &weights.o)] {
        let width = square_dim(name, projection)?;
        if width != dim {
            return Err(CortexError::InvalidConfig(format!(
                "candle attention {name} width {width} disagrees with q width {dim}"
            )));
        }
    }
    if weights.num_heads == 0 || !dim.is_multiple_of(weights.num_heads) {
        return Err(CortexError::InvalidConfig(format!(
            "candle attention num_heads ({}) must be non-zero and divide dim ({dim})",
            weights.num_heads
        )));
    }
    Ok(AttentionParams {
        q: linear_from(device, &weights.q)?,
        k: linear_from(device, &weights.k)?,
        v: linear_from(device, &weights.v)?,
        o: linear_from(device, &weights.o)?,
        num_heads: weights.num_heads,
        head_dim: dim / weights.num_heads,
        dim,
    })
}

fn finish(stage_id: &StageId, tensor: Tensor) -> Result<CandleTensor> {
    CandleTensor::from_candle(tensor, stage_id.as_str()).map_err(|err| stage_failed(stage_id, err))
}

/// Ranks a tensor-input Candle stage accepts: `[seq, dim]` (the reference
/// layout) or `[batch, seq, dim]`.
///
/// Anything else returns [`CortexError::RankMismatch`] with `expected: 2`,
/// the same variant and payload [`ReferenceExecutor`](crate::stage::ReferenceExecutor)
/// returns for rank 0, 1, and 4+. Rank 3 is a Candle-only extension; the
/// reference backend rejects it.
fn check_stage_rank(stage_id: &StageId, dims: &[usize]) -> Result<()> {
    if matches!(dims.len(), 2 | 3) {
        return Ok(());
    }
    Err(rank_mismatch(stage_id, dims.len()))
}

fn rank_mismatch(stage_id: &StageId, got: usize) -> CortexError {
    stage_failed(stage_id, CortexError::RankMismatch { expected: 2, got })
}

fn run_embedding(
    stage_id: &StageId,
    device: &Device,
    params: &EmbeddingParams,
    inputs: &[StageInput<'_, CandleTensor>],
) -> Result<CandleTensor> {
    expect_arity(stage_id, inputs, 1)?;
    let token_ids = tokens_arg(stage_id, inputs, 0)?;
    let seq_len = token_ids.len();
    if seq_len > params.max_seq_len {
        return Err(stage_failed(
            stage_id,
            CortexError::InputLengthMismatch {
                expected: params.max_seq_len,
                got: seq_len,
            },
        ));
    }
    if let Some(&id) = token_ids.iter().find(|&&id| id as usize >= params.vocab) {
        return Err(stage_failed(
            stage_id,
            CortexError::TokenIndex {
                index: id as usize,
                vocab_size: params.vocab,
            },
        ));
    }
    if seq_len == 0 {
        let zeros = Tensor::zeros((0, params.dim), candle_core::DType::F32, device)
            .map_err(|err| fail_candle(stage_id, err))?;
        return finish(stage_id, zeros);
    }
    let pos_ids = (0..seq_len)
        .map(|index| {
            u32::try_from(index).map_err(|_| {
                stage_failed(
                    stage_id,
                    CortexError::InvalidConfig(
                        "sequence length exceeds u32::MAX, cannot generate position ids"
                            .to_string(),
                    ),
                )
            })
        })
        .collect::<Result<Vec<u32>>>()?;
    let tokens =
        Tensor::from_slice(token_ids, seq_len, device).map_err(|err| fail_candle(stage_id, err))?;
    let positions =
        Tensor::from_slice(&pos_ids, seq_len, device).map_err(|err| fail_candle(stage_id, err))?;
    let hidden = params
        .tokens
        .forward(&tokens)
        .and_then(|tok| {
            let pos = params.positions.forward(&positions)?;
            tok.add(&pos)
        })
        .map_err(|err| fail_candle(stage_id, err))?;
    finish(stage_id, hidden)
}

fn run_attention(
    stage_id: &StageId,
    device: &Device,
    params: &AttentionParams,
    inputs: &[StageInput<'_, CandleTensor>],
) -> Result<CandleTensor> {
    expect_arity(stage_id, inputs, 1)?;
    let input = tensor_arg(stage_id, inputs, 0)?;
    let dims = input.inner.dims();
    check_stage_rank(stage_id, dims)?;
    let (batch, seq, keep_batch) = match dims {
        [seq, dim] => {
            check_dim(stage_id, 1, params.dim, *dim)?;
            (1, *seq, false)
        }
        [batch, seq, dim] => {
            check_dim(stage_id, 2, params.dim, *dim)?;
            (*batch, *seq, true)
        }
        // `check_stage_rank` already rejected every other rank.
        _ => return Err(rank_mismatch(stage_id, dims.len())),
    };
    // An empty batch or sequence has nothing to attend over. Candle's
    // `Linear::forward` reshapes through a `()` hole that cannot be resolved
    // when the product is zero, so return the empty output directly.
    if batch == 0 || seq == 0 {
        let out_shape = if keep_batch {
            vec![batch, seq, params.dim]
        } else {
            vec![seq, params.dim]
        };
        candle_shape_fits(&out_shape).map_err(|err| stage_failed(stage_id, err))?;
        let zeros = Tensor::zeros(out_shape, candle_core::DType::F32, device)
            .map_err(|err| fail_candle(stage_id, err))?;
        return finish(stage_id, zeros);
    }

    let batched = if keep_batch {
        input.inner.clone()
    } else {
        input
            .inner
            .unsqueeze(0)
            .map_err(|err| fail_candle(stage_id, err))?
    };
    let project = |linear: &Linear| -> std::result::Result<Tensor, candle_core::Error> {
        linear
            .forward(&batched)?
            .reshape((batch, seq, params.num_heads, params.head_dim))
    };
    let q = project(&params.q).map_err(|err| fail_candle(stage_id, err))?;
    let k = project(&params.k).map_err(|err| fail_candle(stage_id, err))?;
    let v = project(&params.v).map_err(|err| fail_candle(stage_id, err))?;
    let scale = 1.0 / (params.head_dim as f32).sqrt();
    // Candle CPU flash-attention. `ops::sdpa` is Metal-only, so it is not used.
    let context = flash_attn::<f32>(&q, &k, &v, scale, AttnMask::causal(), None, None)
        .map_err(|err| fail_candle(stage_id, err))?;
    // flash_attn returns [batch, num_heads, seq, head_dim]. Transpose restores
    // [batch, seq, num_heads, head_dim] before reshape to [batch, seq, dim].
    let merged = context
        .transpose(1, 2)
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.reshape((batch, seq, params.dim)))
        .and_then(|tensor| params.o.forward(&tensor))
        .map_err(|err| fail_candle(stage_id, err))?;
    let output = if keep_batch {
        merged
    } else {
        merged
            .squeeze(0)
            .map_err(|err| fail_candle(stage_id, err))?
    };
    finish(stage_id, output)
}

fn check_dim(stage_id: &StageId, axis: usize, expected: usize, got: usize) -> Result<()> {
    if expected != got {
        return Err(stage_failed(
            stage_id,
            CortexError::DimMismatch {
                op: "candle attention",
                axis,
                expected,
                got,
            },
        ));
    }
    Ok(())
}

fn run_module(
    stage_id: &StageId,
    inputs: &[StageInput<'_, CandleTensor>],
    in_features: usize,
    out_features: usize,
    forward: impl FnOnce(&Tensor) -> candle_core::Result<Tensor>,
) -> Result<CandleTensor> {
    expect_arity(stage_id, inputs, 1)?;
    let input = tensor_arg(stage_id, inputs, 0)?;
    let dims = input.inner.dims();
    check_stage_rank(stage_id, dims)?;
    let last = dims[dims.len() - 1];
    if last == 0 {
        return Err(stage_failed(
            stage_id,
            CortexError::ZeroWidth {
                op: "candle stage",
                axis: dims.len() - 1,
            },
        ));
    }
    if last != in_features {
        return Err(stage_failed(
            stage_id,
            CortexError::DimMismatch {
                op: "candle stage",
                axis: dims.len() - 1,
                expected: in_features,
                got: last,
            },
        ));
    }
    let leading = &dims[..dims.len() - 1];
    if leading.contains(&0) {
        // Same reason as attention: `Linear::forward` cannot reshape an empty
        // rank-3 batch through its `()` hole, so build the empty output here.
        let mut out_shape = leading.to_vec();
        out_shape.push(out_features);
        candle_shape_fits(&out_shape).map_err(|err| stage_failed(stage_id, err))?;
        let zeros = Tensor::zeros(out_shape, candle_core::DType::F32, input.inner.device())
            .map_err(|err| fail_candle(stage_id, err))?;
        return finish(stage_id, zeros);
    }
    let output = forward(&input.inner).map_err(|err| fail_candle(stage_id, err))?;
    finish(stage_id, output)
}

fn run_dense_mlp(
    stage_id: &StageId,
    params: &MlpParams,
    inputs: &[StageInput<'_, CandleTensor>],
) -> Result<CandleTensor> {
    let in_dim = linear_in_dim(&params.up)?;
    let out_dim = linear_out_dim(&params.down)?;
    run_module(stage_id, inputs, in_dim, out_dim, |x| {
        let hidden = params.up.forward(x)?;
        let hidden = Activation::NewGelu.forward(&hidden)?;
        params.down.forward(&hidden)
    })
}

fn run_gated_mlp(
    stage_id: &StageId,
    params: &GatedParams,
    inputs: &[StageInput<'_, CandleTensor>],
) -> Result<CandleTensor> {
    let in_dim = linear_in_dim(&params.gate)?;
    let out_dim = linear_out_dim(&params.down)?;
    run_module(stage_id, inputs, in_dim, out_dim, |x| {
        let gate = params.gate.forward(x)?;
        let up = params.up.forward(x)?;
        let gated = Tensor::cat(&[&gate, &up], candle_core::D::Minus1)?;
        let hidden = candle_nn::ops::swiglu(&gated)?;
        params.down.forward(&hidden)
    })
}

fn run_add(stage_id: &StageId, inputs: &[StageInput<'_, CandleTensor>]) -> Result<CandleTensor> {
    if inputs.is_empty() {
        return Err(CortexError::StageInputArity {
            stage_id: stage_id.to_string(),
            expected: 1,
            got: 0,
        });
    }
    let first = tensor_arg(stage_id, inputs, 0)?;
    let mut acc = first.inner.clone();
    for index in 1..inputs.len() {
        let rhs = tensor_arg(stage_id, inputs, index)?;
        if acc.dims() != rhs.inner.dims() {
            return Err(stage_failed(
                stage_id,
                CortexError::ShapeMismatch {
                    expected: acc.dims().to_vec(),
                    got: rhs.inner.dims().to_vec(),
                },
            ));
        }
        acc = acc
            .add(&rhs.inner)
            .map_err(|err| fail_candle(stage_id, err))?;
    }
    finish(stage_id, acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_candle_errors_are_prefixed() {
        let err = candle_msg(candle_core::Error::Msg("boom".to_string()));
        assert!(
            matches!(&err, CortexError::Msg(msg) if msg == "candle: boom"),
            "{err:?}"
        );
    }

    #[test]
    fn non_f32_tensor_is_stage_dtype_mismatch() {
        let f64_tensor = Tensor::zeros((2, 2), candle_core::DType::F64, &Device::Cpu).unwrap();
        let err = CandleTensor::from_candle(f64_tensor.clone(), "blocks.0.mlp").unwrap_err();
        assert!(
            matches!(
                &err,
                CortexError::StageDTypeMismatch { stage_id, expected, got }
                    if stage_id == "blocks.0.mlp" && expected == "f32" && got == "f64"
            ),
            "{err:?}"
        );

        let wrapped = CandleTensor { inner: f64_tensor };
        let err = export_hidden_state(&wrapped).unwrap_err();
        assert!(
            matches!(
                &err,
                CortexError::StageDTypeMismatch { stage_id, got, .. }
                    if stage_id == HIDDEN_STATE && got == "f64"
            ),
            "{err:?}"
        );
    }

    #[test]
    fn candle_shape_fits_checks_prefix_and_suffix_products() {
        assert!(candle_shape_fits(&[2, 3, 4]).is_ok());
        assert!(candle_shape_fits(&[]).is_ok());
        assert!(candle_shape_fits(&[usize::MAX, 1, 0]).is_ok());
        for shape in [
            [usize::MAX, usize::MAX, 0],
            [0, usize::MAX, usize::MAX],
            [0, usize::MAX, 2],
        ] {
            assert!(
                matches!(
                    candle_shape_fits(&shape),
                    Err(CortexError::SizeOverflow { .. })
                ),
                "{shape:?}"
            );
        }
    }
}
