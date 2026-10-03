// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral stage-execution contracts for ANN inference.
//!
//! This module defines the vocabulary a backend-agnostic orchestrator uses to
//! describe and run an ANN forward pass as an ordered graph of *stages*:
//!
//! * **Describe:** [`StageKind`] names what a stage computes, [`AnnStage`]
//!   binds an identity ([`StageId`]) and input wiring ([`StageSource`]) to a
//!   kind, and [`AnnTopology`] is a validated ordered list of stages.
//! * **Capabilities:** [`AnnCapabilities`] reports which stage kinds
//!   ([`StageKindTag`]) and [`DType`]s a backend can execute, mirroring the
//!   `snn` module's `SnnCapabilities` contract.
//! * **Execute:** the [`AnnExecutor`] trait runs one stage at a time against
//!   backend-owned tensors, and [`run_topology`] composes an executor over a
//!   topology using an [`ExternalBindings`] source for ingress.
//!
//! The public execution contract is deliberately generic: [`AnnExecutor`]
//! never names [`crate::Tensor`] or `Vec<f32>`, only its associated
//! [`AnnExecutor::Tensor`] type (bounded by [`StageTensor`]). This keeps the
//! contract usable by future accelerated backends.
//!
//! # Intentional omissions
//!
//! A `device` field on [`TensorMeta`]/[`AnnCapabilities`] and a concrete
//! tensor conversion / wire format for crossing the stage boundary are
//! intentionally **not** modelled here; they are owned by RM-1827. Stage
//! descriptors ([`AnnStage`], [`StageSource`], [`AnnTopology`]) carry no serde
//! derives: they describe identity, kind, and wiring only, not a persisted
//! wire format.

use crate::adapter::{self, require_f32_buffer};
use crate::error::{CortexError, Result};
use crate::tensor::Tensor;
use crate::tensor::ops::{try_embedding, try_layer_norm, try_matmul, try_rms_norm};
use crate::transformer::block::FeedForward;
use crate::transformer::{MultiHeadAttention, TransformerLM};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[cfg(feature = "burn")]
pub use crate::adapter::BurnAdapter;
#[cfg(feature = "candle")]
pub use crate::adapter::CandleAdapter;
pub use crate::adapter::{
    AdapterCapabilities, BackendLimitations, DeviceClass, ExternalAdapter, ExternalAdapterMarker,
    HiddenStateBuffer, NegotiationDocument,
};

/// Element type reported through the stage boundary.
///
/// The reference backend is `f32`-only ([`DType::F32`]); the other variants
/// exist so capability negotiation and dtype checks can describe backends that
/// support reduced precision without this crate implementing them.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    /// 32-bit IEEE-754 float (the reference backend's only dtype).
    F32,
    /// 16-bit IEEE-754 half precision.
    F16,
    /// 16-bit brain floating point.
    BF16,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl DType {
    /// Stable negotiation token (`"f32"`, `"f16"`, `"bf16"`).
    pub fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
        }
    }
}

/// Shape + dtype description of a tensor crossing the stage boundary.
///
/// This is metadata only: it never carries data, so an executor can report an
/// output's shape and dtype without exposing its concrete storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    /// Row-major extents, outermost dimension first.
    pub shape: Vec<usize>,
    /// Element type.
    pub dtype: DType,
}

/// A tensor an [`AnnExecutor`] can carry across the stage boundary.
///
/// Implementors expose only shape and dtype through [`TensorMeta`]; the
/// composition driver uses this to validate dtypes without touching storage.
pub trait StageTensor {
    /// Report this tensor's shape and dtype.
    fn meta(&self) -> TensorMeta;
}

impl StageTensor for crate::Tensor {
    fn meta(&self) -> TensorMeta {
        TensorMeta {
            shape: self.shape().to_vec(),
            dtype: DType::F32,
        }
    }
}

/// Which normalization a [`StageKind::Normalization`] stage applies.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// Layer normalization (mean/variance over the feature axis, with bias).
    Layer,
    /// Root-mean-square normalization (no mean subtraction, no bias).
    Rms,
}

/// Which MLP a [`StageKind::Mlp`] stage applies.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpKind {
    /// Dense feed-forward (single up-projection then down-projection).
    Dense,
    /// Gated feed-forward (gate × value, e.g. SwiGLU-style).
    Gated,
}

/// What a stage computes.
///
/// Executable kinds map 1:1 to a [`StageKindTag`] via [`StageKind::tag`];
/// dense and gated MLPs map to *distinct* tags so capability negotiation can
/// advertise them independently.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageKind {
    /// Token (and position) embedding lookup.
    Embedding,
    /// Multi-head self-attention.
    Attention,
    /// Normalization; see [`NormKind`].
    Normalization {
        /// Normalization flavour.
        kind: NormKind,
    },
    /// Feed-forward network; see [`MlpKind`].
    Mlp {
        /// MLP flavour.
        kind: MlpKind,
    },
    /// Mixture-of-Experts router selecting `top_k` of `num_experts`.
    MoeRouter {
        /// Number of experts the router gates over.
        num_experts: usize,
        /// Number of experts selected per token.
        top_k: usize,
    },
    /// A single Mixture-of-Experts expert.
    MoeExpert {
        /// Index of this expert within its router.
        expert_index: usize,
    },
    /// Elementwise addition (e.g. a residual connection).
    Add,
    /// Final projection to logits / outputs.
    Readout,
    /// A backend-specific stage identified by name.
    Custom {
        /// Backend-defined stage name.
        name: String,
    },
}

/// Fieldless tag identifying an executable [`StageKind`] for capability checks.
///
/// Dense and gated MLPs get separate tags ([`StageKindTag::DenseMlp`] /
/// [`StageKindTag::GatedMlp`]), as do layer and RMS normalization, so a backend
/// can advertise support for one flavour without the other.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StageKindTag {
    /// Tag for [`StageKind::Embedding`].
    Embedding,
    /// Tag for [`StageKind::Attention`].
    Attention,
    /// Tag for [`StageKind::Normalization`] with [`NormKind::Layer`].
    LayerNorm,
    /// Tag for [`StageKind::Normalization`] with [`NormKind::Rms`].
    RmsNorm,
    /// Tag for [`StageKind::Mlp`] with [`MlpKind::Dense`].
    DenseMlp,
    /// Tag for [`StageKind::Mlp`] with [`MlpKind::Gated`].
    GatedMlp,
    /// Tag for [`StageKind::MoeRouter`].
    MoeRouter,
    /// Tag for [`StageKind::MoeExpert`].
    MoeExpert,
    /// Tag for [`StageKind::Add`].
    Add,
    /// Tag for [`StageKind::Readout`].
    Readout,
    /// Tag for [`StageKind::Custom`].
    Custom,
}

impl StageKindTag {
    /// Stable negotiation token, independent of [`Debug`] formatting.
    pub fn as_str(self) -> &'static str {
        match self {
            StageKindTag::Embedding => "embedding",
            StageKindTag::Attention => "attention",
            StageKindTag::LayerNorm => "layer_norm",
            StageKindTag::RmsNorm => "rms_norm",
            StageKindTag::DenseMlp => "dense_mlp",
            StageKindTag::GatedMlp => "gated_mlp",
            StageKindTag::MoeRouter => "moe_router",
            StageKindTag::MoeExpert => "moe_expert",
            StageKindTag::Add => "add",
            StageKindTag::Readout => "readout",
            StageKindTag::Custom => "custom",
        }
    }
}

impl fmt::Display for StageKindTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl StageKind {
    /// Map this kind to its fieldless [`StageKindTag`] for capability checks.
    pub fn tag(&self) -> StageKindTag {
        match self {
            StageKind::Embedding => StageKindTag::Embedding,
            StageKind::Attention => StageKindTag::Attention,
            StageKind::Normalization {
                kind: NormKind::Layer,
            } => StageKindTag::LayerNorm,
            StageKind::Normalization {
                kind: NormKind::Rms,
            } => StageKindTag::RmsNorm,
            StageKind::Mlp {
                kind: MlpKind::Dense,
            } => StageKindTag::DenseMlp,
            StageKind::Mlp {
                kind: MlpKind::Gated,
            } => StageKindTag::GatedMlp,
            StageKind::MoeRouter { .. } => StageKindTag::MoeRouter,
            StageKind::MoeExpert { .. } => StageKindTag::MoeExpert,
            StageKind::Add => StageKindTag::Add,
            StageKind::Readout => StageKindTag::Readout,
            StageKind::Custom { .. } => StageKindTag::Custom,
        }
    }
}

/// A validated dotted-path stage identifier.
///
/// A valid id is non-empty and composed of `.`-separated segments, each of
/// which is non-empty and matches `[a-z0-9_]+` (lowercase ASCII letters,
/// digits, and underscores). Construct with [`StageId::new`]; the type is a
/// transparent newtype over the validated `String` and orders/hashes by it so
/// it can key a [`BTreeMap`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StageId(String);

impl StageId {
    /// Validate and construct a [`StageId`].
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidStageId`] when the id is empty, has an
    /// empty segment (leading, trailing, or doubled `.`), or contains any
    /// character outside `[a-z0-9_.]`.
    pub fn new(id: impl Into<String>) -> Result<Self> {
        let id = id.into();
        if id.is_empty() {
            return Err(CortexError::InvalidStageId {
                id,
                reason: "stage id must not be empty".to_string(),
            });
        }
        for segment in id.split('.') {
            if segment.is_empty() {
                return Err(CortexError::InvalidStageId {
                    id: id.clone(),
                    reason: "stage id has an empty '.'-separated segment".to_string(),
                });
            }
            if !segment
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            {
                return Err(CortexError::InvalidStageId {
                    id: id.clone(),
                    reason: format!(
                        "segment '{segment}' must match [a-z0-9_]+ (lowercase ascii, digits, underscore)"
                    ),
                });
            }
        }
        Ok(StageId(id))
    }

    /// Borrow the validated dotted path.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for StageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where a stage input comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageSource {
    /// A named external ingress supplied at run time (e.g. `"tokens"`).
    External(String),
    /// The output of an earlier stage in the topology.
    Stage(StageId),
}

/// A single stage: identity, kind, and input wiring.
///
/// Carries no serde derives — it describes the graph, not a persisted format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnStage {
    /// Unique identity of this stage within its topology.
    pub id: StageId,
    /// What the stage computes.
    pub kind: StageKind,
    /// Ordered inputs; each resolves to an external ingress or an earlier stage.
    pub inputs: Vec<StageSource>,
}

/// A validated, ordered graph of [`AnnStage`]s.
///
/// Construct with [`AnnTopology::new`], which enforces that the graph is a
/// well-formed DAG in topological order (every [`StageSource::Stage`] refers to
/// an *earlier* stage). Carries no serde derives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnTopology {
    stages: Vec<AnnStage>,
}

/// A contiguous sub-span of a topology plus the inputs it depends on from
/// outside the span.
///
/// Returned by [`AnnTopology::sub_span`]. `external_deps` lists, in first-seen
/// order, the stage ids produced *before* the span that the span references,
/// followed by the [`StageSource::External`] ingress names it references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubSpan<'a> {
    /// The contiguous slice of stages, order preserved.
    pub stages: &'a [AnnStage],
    /// Earlier-stage outputs the span consumes but does not itself produce.
    pub external_stage_deps: Vec<StageId>,
    /// Named external ingress the span consumes.
    pub external_inputs: Vec<String>,
}

impl AnnTopology {
    /// Validate and construct a topology from an ordered list of stages.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidTopology`] when the list is empty, when
    /// two stages share an id, or when any [`StageSource::Stage`] references a
    /// stage that is not strictly earlier in the list (a forward or self
    /// reference). [`StageSource::External`] inputs are always allowed.
    pub fn new(stages: Vec<AnnStage>) -> Result<Self> {
        if stages.is_empty() {
            return Err(CortexError::InvalidTopology {
                reason: "topology must contain at least one stage".to_string(),
            });
        }
        let mut seen: BTreeSet<&StageId> = BTreeSet::new();
        for stage in &stages {
            // Validate inputs against *earlier* stages before admitting this
            // stage's own id, so a self-reference is rejected like any other
            // forward reference.
            for source in &stage.inputs {
                if let StageSource::Stage(dep) = source
                    && !seen.contains(dep)
                {
                    return Err(CortexError::InvalidTopology {
                        reason: format!(
                            "stage '{}' references stage '{dep}', which is not an earlier stage",
                            stage.id
                        ),
                    });
                }
            }
            if !seen.insert(&stage.id) {
                return Err(CortexError::InvalidTopology {
                    reason: format!("duplicate stage id '{}'", stage.id),
                });
            }
        }
        Ok(Self { stages })
    }

    /// The stages in topological order.
    pub fn stages(&self) -> &[AnnStage] {
        &self.stages
    }

    /// Index of the stage with `id`, if present.
    pub fn position_by_id(&self, id: &StageId) -> Option<usize> {
        self.stages.iter().position(|stage| &stage.id == id)
    }

    /// The stage with `id`, if present.
    pub fn stage_by_id(&self, id: &StageId) -> Option<&AnnStage> {
        self.stages.iter().find(|stage| &stage.id == id)
    }

    /// Extract the contiguous span from `start` to `end` (inclusive) and the
    /// inputs it depends on from outside the span.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidTopology`] when either id is absent or
    /// `end` precedes `start`.
    pub fn sub_span(&self, start: &StageId, end: &StageId) -> Result<SubSpan<'_>> {
        let start_idx = self
            .position_by_id(start)
            .ok_or_else(|| CortexError::InvalidTopology {
                reason: format!("sub-span start '{start}' is not in the topology"),
            })?;
        let end_idx = self
            .position_by_id(end)
            .ok_or_else(|| CortexError::InvalidTopology {
                reason: format!("sub-span end '{end}' is not in the topology"),
            })?;
        if end_idx < start_idx {
            return Err(CortexError::InvalidTopology {
                reason: format!("sub-span end '{end}' precedes start '{start}'"),
            });
        }

        let span = &self.stages[start_idx..=end_idx];
        let in_span: BTreeSet<&StageId> = span.iter().map(|stage| &stage.id).collect();

        let mut external_stage_deps: Vec<StageId> = Vec::new();
        let mut external_inputs: Vec<String> = Vec::new();
        for stage in span {
            for source in &stage.inputs {
                match source {
                    StageSource::Stage(dep) => {
                        if !in_span.contains(dep) && !external_stage_deps.contains(dep) {
                            external_stage_deps.push(dep.clone());
                        }
                    }
                    StageSource::External(name) => {
                        if !external_inputs.contains(name) {
                            external_inputs.push(name.clone());
                        }
                    }
                }
            }
        }

        Ok(SubSpan {
            stages: span,
            external_stage_deps,
            external_inputs,
        })
    }
}

/// What a backend can execute, mirroring the `snn` module's capabilities style.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnCapabilities {
    /// Human-readable backend identity, e.g. `"cortex::ReferenceExecutor"`.
    pub backend_name: &'static str,
    /// Stage kinds (by tag) the backend can execute.
    pub supported_tags: BTreeSet<StageKindTag>,
    /// Element types the backend accepts and produces.
    pub supported_dtypes: BTreeSet<DType>,
    /// Whether `execute` mutates backend state across calls.
    pub stateful: bool,
}

impl AnnCapabilities {
    /// Whether this backend can execute `kind` (by its [`StageKind::tag`]).
    pub fn supports(&self, kind: &StageKind) -> bool {
        self.supported_tags.contains(&kind.tag())
    }
}

/// One input handed to [`AnnExecutor::execute`].
///
/// Generic over the executor's tensor type so the contract never names a
/// concrete tensor.
#[derive(Debug)]
pub enum StageInput<'a, T> {
    /// Token ids for an embedding ingress.
    Tokens(&'a [u32]),
    /// A tensor produced by an earlier stage or supplied as ingress.
    Tensor(&'a T),
}

/// Executes stages against backend-owned tensors.
///
/// The contract is deliberately generic: it names only the associated
/// [`AnnExecutor::Tensor`] type, never [`crate::Tensor`] or `Vec<f32>`.
///
/// Implementations must, mirroring `snn::SnnBackend`:
///
/// * **Be failure-atomic:** a rejected input (wrong arity, kind, dtype, or an
///   internal error) must leave executor state unmodified, so a caller can fail
///   closed without a half-applied stage.
/// * **Reject unsupported stages structurally:** a stage whose kind is not in
///   [`AnnCapabilities::supported_tags`] returns
///   [`CortexError::UnsupportedStage`] rather than panicking.
/// * **Never panic at the boundary:** all error conditions are reported as
///   [`CortexError`].
pub trait AnnExecutor {
    /// Backend-owned tensor type crossing the stage boundary.
    type Tensor: StageTensor;

    /// Report what this backend can execute.
    fn capabilities(&self) -> AnnCapabilities;

    /// Execute one stage against its resolved inputs, returning its output.
    ///
    /// # Errors
    ///
    /// Returns a [`CortexError`] on unsupported kinds, input arity/kind/dtype
    /// mismatches, or an internal failure. On error, executor state is
    /// unchanged.
    fn execute(
        &mut self,
        stage: &AnnStage,
        inputs: &[StageInput<Self::Tensor>],
    ) -> Result<Self::Tensor>;
}

/// Supplies external ingress for [`run_topology`].
///
/// Keys are [`StageSource::External`] names. Each binding is either token ids
/// (for an embedding ingress) or a tensor (for a hidden-state ingress), and is
/// resolved into the matching [`StageInput`] when a stage references it.
pub struct ExternalBindings<'a, T> {
    tokens: BTreeMap<String, &'a [u32]>,
    tensors: BTreeMap<String, &'a T>,
}

impl<'a, T> Default for ExternalBindings<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> ExternalBindings<'a, T> {
    /// An empty binding set.
    pub fn new() -> Self {
        Self {
            tokens: BTreeMap::new(),
            tensors: BTreeMap::new(),
        }
    }

    /// Bind `name` to token ids. Returns `self` for chaining.
    pub fn with_tokens(mut self, name: impl Into<String>, tokens: &'a [u32]) -> Self {
        self.tokens.insert(name.into(), tokens);
        self
    }

    /// Bind `name` to a tensor. Returns `self` for chaining.
    pub fn with_tensor(mut self, name: impl Into<String>, tensor: &'a T) -> Self {
        self.tensors.insert(name.into(), tensor);
        self
    }

    /// Resolve `name` into a [`StageInput`], preferring a token binding.
    fn resolve(&self, name: &str) -> Option<StageInput<'a, T>> {
        if let Some(tokens) = self.tokens.get(name) {
            Some(StageInput::Tokens(tokens))
        } else {
            self.tensors
                .get(name)
                .map(|&tensor| StageInput::Tensor(tensor))
        }
    }
}

/// Run `executor` over `topology`, resolving ingress from `bindings`.
///
/// Stages run in topological order. For each stage, every [`StageSource`] is
/// resolved — [`StageSource::External`] from `bindings`, [`StageSource::Stage`]
/// from the outputs produced so far — and passed to [`AnnExecutor::execute`].
/// Each produced output's [`DType`] is checked against the executor's
/// [`AnnCapabilities::supported_dtypes`] and the output is stored under its
/// stage id. The first error encountered stops the run.
///
/// # Errors
///
/// * [`CortexError::MissingStageBinding`] when a referenced external ingress
///   has no binding.
/// * [`CortexError::InvalidTopology`] when a stage input references an output
///   that is unexpectedly absent (should not occur for a validated topology).
/// * [`CortexError::StageDTypeMismatch`] when an output dtype is unsupported.
/// * Any error returned by [`AnnExecutor::execute`], unchanged.
pub fn run_topology<E: AnnExecutor>(
    executor: &mut E,
    topology: &AnnTopology,
    bindings: &ExternalBindings<E::Tensor>,
) -> Result<BTreeMap<StageId, E::Tensor>> {
    let supported_dtypes = executor.capabilities().supported_dtypes;
    let mut outputs: BTreeMap<StageId, E::Tensor> = BTreeMap::new();

    for stage in topology.stages() {
        let mut inputs: Vec<StageInput<E::Tensor>> = Vec::with_capacity(stage.inputs.len());
        for source in &stage.inputs {
            match source {
                StageSource::External(name) => {
                    let resolved = bindings
                        .resolve(name)
                        .ok_or_else(|| CortexError::MissingStageBinding { name: name.clone() })?;
                    inputs.push(resolved);
                }
                StageSource::Stage(dep) => {
                    let tensor =
                        outputs
                            .get(dep)
                            .ok_or_else(|| CortexError::InvalidTopology {
                                reason: format!(
                                    "stage '{}' references output of '{dep}', which has not been produced",
                                    stage.id
                                ),
                            })?;
                    inputs.push(StageInput::Tensor(tensor));
                }
            }
        }

        let output = executor.execute(stage, &inputs)?;
        let dtype = output.meta().dtype;
        if !supported_dtypes.contains(&dtype) {
            let expected = supported_dtypes
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(CortexError::StageDTypeMismatch {
                stage_id: stage.id.to_string(),
                expected,
                got: dtype.to_string(),
            });
        }
        outputs.insert(stage.id.clone(), output);
    }

    Ok(outputs)
}

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
        max_seq_len: usize,
    },
    /// Multi-head self-attention weights.
    Attention(&'p MultiHeadAttention),
    /// Normalization weights, optional bias, epsilon, and flavour.
    Normalization {
        /// Scale/gain vector `[dim]`.
        weight: &'p Tensor,
        /// Optional shift vector `[dim]` (LayerNorm only; `None` for RMS).
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
/// [`TransformerLM::try_forward`] uses ([`try_embedding`], [`try_layer_norm`],
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

    /// Embedding stage: token + position lookup, matching
    /// [`TransformerLM::embed_tokens`] exactly.
    fn run_embedding(
        stage_id: &StageId,
        tok_embed: &Tensor,
        pos_embed: &Tensor,
        max_seq_len: usize,
        token_ids: &[u32],
    ) -> Result<Tensor> {
        let seq_len = token_ids.len();
        if seq_len > max_seq_len {
            return Err(Self::wrap_failure(
                stage_id,
                CortexError::InputLengthMismatch {
                    expected: max_seq_len,
                    got: seq_len,
                },
            ));
        }
        let run = || -> Result<Tensor> {
            let tok = try_embedding(tok_embed, token_ids)?;
            let pos_ids: Vec<u32> = (0..seq_len)
                .map(u32::try_from)
                .collect::<std::result::Result<_, _>>()
                .map_err(|_| CortexError::InputLengthMismatch {
                    expected: u32::MAX as usize,
                    got: seq_len,
                })?;
            let pos = try_embedding(pos_embed, &pos_ids)?;
            tok.try_add(&pos)
        };
        run().map_err(|err| Self::wrap_failure(stage_id, err))
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
    /// to borrowed params from `model`, so [`run_topology`] reproduces
    /// [`TransformerLM::try_forward`] bit for bit.
    ///
    /// The returned executor borrows from `model`, so `model` must outlive it.
    ///
    /// # Errors
    ///
    /// Returns the structured error from [`TransformerLM::validate_wire`] when
    /// the model is malformed, or [`CortexError::InvalidStageId`] /
    /// [`CortexError::InvalidTopology`] if id construction or topology
    /// validation fails (not expected for a valid model).
    pub fn from_transformer(model: &'p TransformerLM) -> Result<(Self, AnnTopology)> {
        model.validate_wire()?;

        let eps = crate::transformer::LAYER_NORM_EPS;
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

        // Add stages carry no params; everything else needs a binding.
        if !matches!(stage.kind, StageKind::Add) {
            let params =
                self.params
                    .get(stage_id)
                    .ok_or_else(|| CortexError::MissingStageParams {
                        stage_id: stage_id.to_string(),
                    })?;
            // A params/kind disagreement is a wiring bug; treat it as missing
            // params for the requested kind.
            if params.tag() != stage.kind.tag() {
                return Err(CortexError::MissingStageParams {
                    stage_id: stage_id.to_string(),
                });
            }
        }

        match &stage.kind {
            StageKind::Embedding => {
                Self::expect_arity(stage_id, inputs, 1)?;
                let token_ids = Self::tokens_arg(stage_id, inputs, 0)?;
                let ReferenceStageParams::Embedding {
                    tok_embed,
                    pos_embed,
                    max_seq_len,
                } = self.params[stage_id]
                else {
                    unreachable!("embedding params checked above");
                };
                Self::run_embedding(stage_id, tok_embed, pos_embed, max_seq_len, token_ids)
            }
            StageKind::Attention => {
                Self::expect_arity(stage_id, inputs, 1)?;
                let x = Self::tensor_arg(stage_id, inputs, 0)?;
                let ReferenceStageParams::Attention(attn) = self.params[stage_id] else {
                    unreachable!("attention params checked above");
                };
                attn.try_forward(x)
                    .map_err(|err| Self::wrap_failure(stage_id, err))
            }
            StageKind::Normalization { .. } => {
                Self::expect_arity(stage_id, inputs, 1)?;
                let x = Self::tensor_arg(stage_id, inputs, 0)?;
                let ReferenceStageParams::Normalization {
                    weight,
                    bias,
                    eps,
                    kind,
                } = self.params[stage_id]
                else {
                    unreachable!("normalization params checked above");
                };
                let out = match kind {
                    NormKind::Layer => {
                        let bias = bias.ok_or_else(|| CortexError::MissingStageParams {
                            stage_id: stage_id.to_string(),
                        })?;
                        try_layer_norm(x, weight, bias, eps)
                    }
                    NormKind::Rms => try_rms_norm(x, weight, eps),
                };
                out.map_err(|err| Self::wrap_failure(stage_id, err))
            }
            StageKind::Mlp { .. } => {
                Self::expect_arity(stage_id, inputs, 1)?;
                let x = Self::tensor_arg(stage_id, inputs, 0)?;
                let ReferenceStageParams::Mlp(ffn) = self.params[stage_id] else {
                    unreachable!("mlp params checked above");
                };
                ffn.try_forward(x)
                    .map_err(|err| Self::wrap_failure(stage_id, err))
            }
            StageKind::Readout => {
                Self::expect_arity(stage_id, inputs, 1)?;
                let x = Self::tensor_arg(stage_id, inputs, 0)?;
                let ReferenceStageParams::Readout { lm_head } = self.params[stage_id] else {
                    unreachable!("readout params checked above");
                };
                try_matmul(x, lm_head).map_err(|err| Self::wrap_failure(stage_id, err))
            }
            StageKind::Add => {
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
            // Unsupported kinds were rejected above.
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

    fn stage(id: &str, kind: StageKind, inputs: Vec<StageSource>) -> AnnStage {
        AnnStage {
            id: StageId::new(id).expect("valid id"),
            kind,
            inputs,
        }
    }

    #[test]
    fn stage_id_accepts_valid_dotted_paths() {
        assert_eq!(StageId::new("tokens").unwrap().as_str(), "tokens");
        assert_eq!(
            StageId::new("block_0.attn.q_proj").unwrap().as_str(),
            "block_0.attn.q_proj"
        );
    }

    #[test]
    fn stage_id_rejects_invalid_paths() {
        for bad in ["", "Block0", "a..b", ".lead", "trail.", "has space", "a.B"] {
            let err = StageId::new(bad).unwrap_err();
            assert!(
                matches!(err, CortexError::InvalidStageId { .. }),
                "expected InvalidStageId for {bad:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn topology_rejects_empty() {
        let err = AnnTopology::new(Vec::new()).unwrap_err();
        assert!(matches!(err, CortexError::InvalidTopology { .. }));
    }

    #[test]
    fn topology_rejects_duplicate_ids() {
        let stages = vec![
            stage("embed", StageKind::Embedding, vec![]),
            stage("embed", StageKind::Add, vec![]),
        ];
        let err = AnnTopology::new(stages).unwrap_err();
        assert!(matches!(err, CortexError::InvalidTopology { .. }));
    }

    #[test]
    fn topology_rejects_forward_reference() {
        let stages = vec![
            stage(
                "a",
                StageKind::Add,
                vec![StageSource::Stage(StageId::new("b").unwrap())],
            ),
            stage("b", StageKind::Embedding, vec![]),
        ];
        let err = AnnTopology::new(stages).unwrap_err();
        assert!(matches!(err, CortexError::InvalidTopology { .. }));
    }

    #[test]
    fn topology_rejects_self_reference() {
        let stages = vec![stage(
            "a",
            StageKind::Add,
            vec![StageSource::Stage(StageId::new("a").unwrap())],
        )];
        let err = AnnTopology::new(stages).unwrap_err();
        assert!(matches!(err, CortexError::InvalidTopology { .. }));
    }

    #[test]
    fn topology_accepts_backward_and_external_references() {
        let stages = vec![
            stage(
                "embed",
                StageKind::Embedding,
                vec![StageSource::External("tokens".to_string())],
            ),
            stage(
                "add",
                StageKind::Add,
                vec![StageSource::Stage(StageId::new("embed").unwrap())],
            ),
        ];
        let topo = AnnTopology::new(stages).expect("valid topology");
        assert_eq!(topo.stages().len(), 2);
        assert_eq!(topo.position_by_id(&StageId::new("add").unwrap()), Some(1));
        assert!(topo.stage_by_id(&StageId::new("embed").unwrap()).is_some());
        assert!(
            topo.stage_by_id(&StageId::new("missing").unwrap())
                .is_none()
        );
    }

    #[test]
    fn stage_kind_tag_separates_dense_and_gated_mlp() {
        let dense = StageKind::Mlp {
            kind: MlpKind::Dense,
        };
        let gated = StageKind::Mlp {
            kind: MlpKind::Gated,
        };
        assert_eq!(dense.tag(), StageKindTag::DenseMlp);
        assert_eq!(gated.tag(), StageKindTag::GatedMlp);
        assert_ne!(dense.tag(), gated.tag());

        // Normalization flavours are likewise distinct.
        assert_ne!(
            StageKind::Normalization {
                kind: NormKind::Layer
            }
            .tag(),
            StageKind::Normalization {
                kind: NormKind::Rms
            }
            .tag()
        );
    }

    #[test]
    fn capabilities_supports_reports_expected_booleans() {
        let caps = AnnCapabilities {
            backend_name: "test",
            supported_tags: BTreeSet::from([StageKindTag::Embedding, StageKindTag::DenseMlp]),
            supported_dtypes: BTreeSet::from([DType::F32]),
            stateful: false,
        };
        assert!(caps.supports(&StageKind::Embedding));
        assert!(caps.supports(&StageKind::Mlp {
            kind: MlpKind::Dense
        }));
        // Gated MLP has a distinct tag and is not advertised.
        assert!(!caps.supports(&StageKind::Mlp {
            kind: MlpKind::Gated
        }));
        assert!(!caps.supports(&StageKind::Attention));
    }

    #[test]
    fn sub_span_lists_external_dependencies_in_order() {
        let stages = vec![
            stage(
                "embed",
                StageKind::Embedding,
                vec![StageSource::External("tokens".to_string())],
            ),
            stage(
                "norm",
                StageKind::Normalization {
                    kind: NormKind::Layer,
                },
                vec![StageSource::Stage(StageId::new("embed").unwrap())],
            ),
            stage(
                "attn",
                StageKind::Attention,
                vec![
                    StageSource::Stage(StageId::new("norm").unwrap()),
                    StageSource::External("mask".to_string()),
                ],
            ),
            stage(
                "add",
                StageKind::Add,
                vec![
                    StageSource::Stage(StageId::new("attn").unwrap()),
                    StageSource::Stage(StageId::new("embed").unwrap()),
                ],
            ),
        ];
        let topo = AnnTopology::new(stages).expect("valid topology");

        // Span [attn, add]: depends on 'norm' output (before span), 'embed'
        // output (before span), and external 'mask'. 'attn' is produced inside.
        let span = topo
            .sub_span(
                &StageId::new("attn").unwrap(),
                &StageId::new("add").unwrap(),
            )
            .expect("valid span");
        assert_eq!(span.stages.len(), 2);
        assert_eq!(
            span.external_stage_deps,
            vec![
                StageId::new("norm").unwrap(),
                StageId::new("embed").unwrap()
            ]
        );
        assert_eq!(span.external_inputs, vec!["mask".to_string()]);
    }

    #[test]
    fn sub_span_rejects_reversed_or_missing_bounds() {
        let stages = vec![
            stage("a", StageKind::Embedding, vec![]),
            stage("b", StageKind::Add, vec![]),
        ];
        let topo = AnnTopology::new(stages).expect("valid topology");
        let reversed = topo.sub_span(&StageId::new("b").unwrap(), &StageId::new("a").unwrap());
        assert!(matches!(reversed, Err(CortexError::InvalidTopology { .. })));
        let missing = topo.sub_span(&StageId::new("a").unwrap(), &StageId::new("z").unwrap());
        assert!(matches!(missing, Err(CortexError::InvalidTopology { .. })));
    }

    #[test]
    fn tensor_reports_f32_meta() {
        let tensor = crate::Tensor::try_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let meta = tensor.meta();
        assert_eq!(meta.dtype, DType::F32);
        assert_eq!(meta.shape, vec![2, 2]);
    }

    // ── ReferenceExecutor (FEAT-002) ──────────────────────────────────────

    use crate::tensor::Tensor;
    use crate::tensor::ops::{try_layer_norm, try_matmul};
    use crate::transformer::{TransformerConfig, TransformerLM};

    /// A minimal, valid model so executor tests stay fast.
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
    fn missing_external_binding_surfaces_through_driver() {
        let model = micro_model();
        let (mut exec, topo) = ReferenceExecutor::from_transformer(&model).unwrap();
        // Bind nothing: the "tokens" external ingress is unresolved.
        let bindings: ExternalBindings<Tensor> = ExternalBindings::new();
        let err = run_topology(&mut exec, &topo, &bindings).unwrap_err();
        assert!(matches!(
            err,
            CortexError::MissingStageBinding { name } if name == "tokens"
        ));
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
                eps: crate::transformer::LAYER_NORM_EPS,
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
        let bindings = ExternalBindings::new().with_tokens("tokens", &ids);
        let outputs = run_topology(&mut exec, &topo, &bindings).unwrap();
        let readout = outputs
            .get(&StageId::new("readout").unwrap())
            .expect("readout output");
        let expected = model.try_forward(&ids).unwrap();
        assert_eq!(readout.data(), expected.data());
        assert_eq!(readout.shape(), expected.shape());
    }
}
