// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{CortexError, Result};
use std::collections::BTreeSet;
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
/// it can key a [`std::collections::BTreeMap`].
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
/// an *earlier* stage), or with [`AnnTopology::with_external_stage_deps`] /
/// [`SubSpan::to_topology`] to additionally admit references to stage outputs
/// computed outside the topology and supplied at run time through
/// [`crate::stage::ExternalBindings::with_stage_output`]. Carries no serde derives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnTopology {
    stages: Vec<AnnStage>,
    external_stage_deps: Vec<StageId>,
}

/// A contiguous sub-span of a topology plus the inputs it depends on from
/// outside the span.
///
/// Returned by [`AnnTopology::sub_span`]. `external_stage_deps` lists, in
/// first-seen order, the stage ids produced *before* the span that the span
/// references; `external_inputs` lists the [`StageSource::External`] ingress
/// names it references. [`SubSpan::to_topology`] turns the span into a
/// runnable [`AnnTopology`] whose `external_stage_deps` must be satisfied with
/// [`crate::stage::ExternalBindings::with_stage_output`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubSpan<'a> {
    /// The contiguous slice of stages, order preserved.
    pub stages: &'a [AnnStage],
    /// Earlier-stage outputs the span consumes but does not itself produce.
    pub external_stage_deps: Vec<StageId>,
    /// Named external ingress the span consumes.
    pub external_inputs: Vec<String>,
}

impl SubSpan<'_> {
    /// Build a runnable topology from this span.
    ///
    /// The stages are kept exactly as they are (same ids, kinds, and wiring);
    /// the span's `external_stage_deps` become the topology's declared
    /// external stage dependencies, so [`crate::stage::run_topology`] resolves each of them
    /// from [`crate::stage::ExternalBindings::with_stage_output`] instead of from an earlier
    /// stage.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidTopology`] under the same conditions as
    /// [`AnnTopology::with_external_stage_deps`]; not expected for a span taken
    /// from a validated topology.
    pub fn to_topology(&self) -> Result<AnnTopology> {
        AnnTopology::with_external_stage_deps(
            self.stages.to_vec(),
            self.external_stage_deps.clone(),
        )
    }
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
        Self::with_external_stage_deps(stages, Vec::new())
    }

    /// Validate and construct a topology whose stages may also reference the
    /// stage outputs listed in `external_stage_deps`, which are computed
    /// outside this topology and bound at run time with
    /// [`crate::stage::ExternalBindings::with_stage_output`].
    ///
    /// This is how a [`SubSpan`] becomes runnable (see
    /// [`SubSpan::to_topology`]). With an empty `external_stage_deps` it is
    /// exactly [`AnnTopology::new`].
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::InvalidTopology`] when the list is empty, when
    /// two stages share an id, when any [`StageSource::Stage`] references a
    /// stage that is neither strictly earlier in the list nor declared in
    /// `external_stage_deps`, when a declared dependency is listed twice, is
    /// also produced by a stage in the list, or is not referenced by any stage.
    pub fn with_external_stage_deps(
        stages: Vec<AnnStage>,
        external_stage_deps: Vec<StageId>,
    ) -> Result<Self> {
        if stages.is_empty() {
            return Err(CortexError::InvalidTopology {
                reason: "topology must contain at least one stage".to_string(),
            });
        }
        let declared = Self::validate_declared_deps(&external_stage_deps)?;
        let mut referenced: BTreeSet<&StageId> = BTreeSet::new();
        let mut seen: BTreeSet<&StageId> = BTreeSet::new();
        for stage in &stages {
            // Validate inputs against *earlier* stages before admitting this
            // stage's own id, so a self-reference is rejected like any other
            // forward reference.
            Self::check_stage_refs(stage, &seen, &declared, &mut referenced)?;
            Self::check_dep_not_produced(stage, &declared)?;
            Self::admit_stage_id(stage, &mut seen)?;
        }
        Self::check_declared_deps_referenced(&external_stage_deps, &referenced)?;
        Ok(Self {
            stages,
            external_stage_deps,
        })
    }

    /// The declared external dependencies as a set, rejecting a duplicate.
    fn validate_declared_deps(deps: &[StageId]) -> Result<BTreeSet<&StageId>> {
        let mut declared = BTreeSet::new();
        for dep in deps {
            if !declared.insert(dep) {
                return Err(CortexError::InvalidTopology {
                    reason: format!("external stage dependency '{dep}' is declared twice"),
                });
            }
        }
        Ok(declared)
    }

    /// Every [`StageSource::Stage`] input of `stage` must name an already-seen
    /// stage or a declared external dependency; declared ones are recorded in
    /// `referenced`.
    fn check_stage_refs<'s>(
        stage: &'s AnnStage,
        seen: &BTreeSet<&StageId>,
        declared: &BTreeSet<&StageId>,
        referenced: &mut BTreeSet<&'s StageId>,
    ) -> Result<()> {
        for source in &stage.inputs {
            let StageSource::Stage(dep) = source else {
                continue;
            };
            if declared.contains(dep) {
                referenced.insert(dep);
            } else if !seen.contains(dep) {
                return Err(CortexError::InvalidTopology {
                    reason: format!(
                        "stage '{}' references stage '{dep}', which is not an earlier stage",
                        stage.id
                    ),
                });
            }
        }
        Ok(())
    }

    /// A declared external dependency may not also be produced by `stage`.
    fn check_dep_not_produced(stage: &AnnStage, declared: &BTreeSet<&StageId>) -> Result<()> {
        if declared.contains(&stage.id) {
            return Err(CortexError::InvalidTopology {
                reason: format!(
                    "external stage dependency '{}' is also produced by the topology",
                    stage.id
                ),
            });
        }
        Ok(())
    }

    /// Record `stage`'s id as produced, rejecting a duplicate.
    fn admit_stage_id<'s>(stage: &'s AnnStage, seen: &mut BTreeSet<&'s StageId>) -> Result<()> {
        if !seen.insert(&stage.id) {
            return Err(CortexError::InvalidTopology {
                reason: format!("duplicate stage id '{}'", stage.id),
            });
        }
        Ok(())
    }

    /// Every declared external dependency must be referenced by some stage.
    fn check_declared_deps_referenced(
        deps: &[StageId],
        referenced: &BTreeSet<&StageId>,
    ) -> Result<()> {
        if let Some(unused) = deps.iter().find(|dep| !referenced.contains(dep)) {
            return Err(CortexError::InvalidTopology {
                reason: format!(
                    "external stage dependency '{unused}' is not referenced by any stage"
                ),
            });
        }
        Ok(())
    }

    /// The stages in topological order.
    pub fn stages(&self) -> &[AnnStage] {
        &self.stages
    }

    /// Stage outputs this topology consumes but does not produce, in
    /// declaration order. Empty for a topology built with [`AnnTopology::new`];
    /// each entry must be bound with [`crate::stage::ExternalBindings::with_stage_output`]
    /// before [`crate::stage::run_topology`] can run it.
    pub fn external_stage_deps(&self) -> &[StageId] {
        &self.external_stage_deps
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
    fn sub_span_to_topology_keeps_external_stage_deps() {
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
                "add",
                StageKind::Add,
                vec![
                    StageSource::Stage(StageId::new("norm").unwrap()),
                    StageSource::Stage(StageId::new("embed").unwrap()),
                ],
            ),
        ];
        let topo = AnnTopology::new(stages).expect("valid topology");
        assert!(topo.external_stage_deps().is_empty());

        let span = topo
            .sub_span(&StageId::new("add").unwrap(), &StageId::new("add").unwrap())
            .expect("valid span");
        // As a plain topology the span is invalid: it references earlier stages.
        assert!(matches!(
            AnnTopology::new(span.stages.to_vec()),
            Err(CortexError::InvalidTopology { .. })
        ));
        // As a span topology those references are declared external deps.
        let span_topo = span.to_topology().expect("span topology");
        assert_eq!(span_topo.stages(), span.stages);
        assert_eq!(
            span_topo.external_stage_deps(),
            &[
                StageId::new("norm").unwrap(),
                StageId::new("embed").unwrap()
            ]
        );
    }

    #[test]
    fn topology_with_external_stage_deps_validates_declarations() {
        let add = |deps: &[&str]| {
            stage(
                "add",
                StageKind::Add,
                deps.iter()
                    .map(|d| StageSource::Stage(StageId::new(*d).unwrap()))
                    .collect(),
            )
        };
        let id = |s: &str| StageId::new(s).unwrap();

        // Declared dep that no stage references.
        let err = AnnTopology::with_external_stage_deps(vec![add(&["x"])], vec![id("x"), id("y")])
            .unwrap_err();
        assert!(
            matches!(err, CortexError::InvalidTopology { ref reason } if reason.contains("'y'"))
        );

        // Declared dep that is also produced by the topology.
        let err = AnnTopology::with_external_stage_deps(
            vec![stage("x", StageKind::Embedding, vec![]), add(&["x"])],
            vec![id("x")],
        )
        .unwrap_err();
        assert!(
            matches!(err, CortexError::InvalidTopology { ref reason } if reason.contains("'x'"))
        );

        // Duplicate declaration.
        let err = AnnTopology::with_external_stage_deps(vec![add(&["x"])], vec![id("x"), id("x")])
            .unwrap_err();
        assert!(matches!(err, CortexError::InvalidTopology { .. }));

        // Reference that is neither earlier nor declared is still rejected.
        let err = AnnTopology::with_external_stage_deps(vec![add(&["x", "z"])], vec![id("x")])
            .unwrap_err();
        assert!(
            matches!(err, CortexError::InvalidTopology { ref reason } if reason.contains("'z'"))
        );

        // Well-formed.
        let topo =
            AnnTopology::with_external_stage_deps(vec![add(&["x"])], vec![id("x")]).expect("valid");
        assert_eq!(topo.external_stage_deps(), &[id("x")]);
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
