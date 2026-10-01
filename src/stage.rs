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

use crate::error::{CortexError, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

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
        let name = match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
        };
        f.write_str(name)
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
#[derive(Debug, Clone)]
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
}
