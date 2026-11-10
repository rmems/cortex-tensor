// SPDX-License-Identifier: MIT OR Apache-2.0

use super::vocabulary::{AnnExecutor, AnnTopology, StageId, StageInput, StageSource, StageTensor};
use crate::error::{CortexError, Result};
use std::collections::BTreeMap;
use std::fmt;

/// One external ingress value: token ids or a tensor.
///
/// Tokens and tensors share a single namespace so a name can only ever resolve
/// to one of them; see [`ExternalBindings::with_tokens`].
#[derive(Debug)]
enum Binding<'a, T> {
    Tokens(&'a [u32]),
    Tensor(&'a T),
}

/// Supplies external ingress for [`run_topology`].
///
/// Keys are [`StageSource::External`] names. Each binding is either token ids
/// (for an embedding ingress) or a tensor (for a hidden-state ingress), and is
/// resolved into the matching [`StageInput`] when a stage references it.
///
/// Names live in **one** namespace: binding the same name twice, whether as
/// tokens and a tensor or as two tensors, is rejected with
/// [`CortexError::ConflictingStageBinding`] at bind time rather than letting
/// one binding silently shadow the other.
///
/// Separately, [`ExternalBindings::with_stage_output`] supplies the output of
/// a stage computed *outside* the topology being run, keyed by [`StageId`], for
/// the [`StageSource::Stage`] references a [`SubSpan::to_topology`] topology
/// declares in [`AnnTopology::external_stage_deps`].
pub struct ExternalBindings<'a, T> {
    externals: BTreeMap<String, Binding<'a, T>>,
    stage_outputs: BTreeMap<StageId, &'a T>,
}

impl<'a, T> Default for ExternalBindings<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Reports bound names and their kinds (`tokens` / `tensor`), never values, so
/// `T` need not implement `Debug`.
impl<T> fmt::Debug for ExternalBindings<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        for (name, binding) in &self.externals {
            map.entry(name, &binding_kind(binding));
        }
        for id in self.stage_outputs.keys() {
            map.entry(id, &"stage output");
        }
        map.finish()
    }
}

impl<'a, T> ExternalBindings<'a, T> {
    /// An empty binding set.
    pub fn new() -> Self {
        Self {
            externals: BTreeMap::new(),
            stage_outputs: BTreeMap::new(),
        }
    }

    /// Bind `name` to token ids. Returns `self` for chaining.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::ConflictingStageBinding`] when `name` is already
    /// bound (as tokens or as a tensor).
    pub fn with_tokens(self, name: impl Into<String>, tokens: &'a [u32]) -> Result<Self> {
        self.insert_external(name.into(), Binding::Tokens(tokens))
    }

    /// Bind `name` to a tensor. Returns `self` for chaining.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::ConflictingStageBinding`] when `name` is already
    /// bound (as tokens or as a tensor).
    pub fn with_tensor(self, name: impl Into<String>, tensor: &'a T) -> Result<Self> {
        self.insert_external(name.into(), Binding::Tensor(tensor))
    }

    /// Bind the externally computed output of stage `id` to a tensor, for a
    /// topology that declares `id` in [`AnnTopology::external_stage_deps`].
    /// Returns `self` for chaining.
    ///
    /// Stage outputs are keyed by [`StageId`] and are independent of the
    /// [`StageSource::External`] names bound with
    /// [`ExternalBindings::with_tensor`]. [`run_topology`] rejects a bound
    /// stage output whose id the topology itself produces.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::ConflictingStageBinding`] when `id` already has
    /// a bound output.
    pub fn with_stage_output(mut self, id: StageId, tensor: &'a T) -> Result<Self> {
        if self.stage_outputs.contains_key(&id) {
            return Err(CortexError::ConflictingStageBinding {
                name: id.to_string(),
                reason: "stage output is already bound".to_string(),
            });
        }
        self.stage_outputs.insert(id, tensor);
        Ok(self)
    }

    /// The bound output of stage `id`, if any.
    fn stage_output(&self, id: &StageId) -> Option<&'a T> {
        self.stage_outputs.get(id).copied()
    }

    fn insert_external(mut self, name: String, binding: Binding<'a, T>) -> Result<Self> {
        if let Some(existing) = self.externals.get(&name) {
            let (existing_kind, new_kind) = (binding_kind(existing), binding_kind(&binding));
            return Err(CortexError::ConflictingStageBinding {
                name,
                reason: format!(
                    "already bound as {existing_kind}; cannot also bind it as {new_kind}"
                ),
            });
        }
        self.externals.insert(name, binding);
        Ok(self)
    }

    /// Resolve `name` into a [`StageInput`], if bound.
    fn resolve(&self, name: &str) -> Option<StageInput<'a, T>> {
        self.externals.get(name).map(|binding| match *binding {
            Binding::Tokens(tokens) => StageInput::Tokens(tokens),
            Binding::Tensor(tensor) => StageInput::Tensor(tensor),
        })
    }
}

fn binding_kind<T>(binding: &Binding<'_, T>) -> &'static str {
    match binding {
        Binding::Tokens(_) => "tokens",
        Binding::Tensor(_) => "tensor",
    }
}

/// Run `executor` over `topology`, resolving ingress from `bindings`.
///
/// Stages run in topological order. For each stage, every [`StageSource`] is
/// resolved — [`StageSource::External`] from `bindings`, [`StageSource::Stage`]
/// from the outputs produced so far or, for an id in
/// [`AnnTopology::external_stage_deps`], from
/// [`ExternalBindings::with_stage_output`] — and passed to
/// [`AnnExecutor::execute`]. Each produced output's [`crate::stage::DType`] is checked
/// against the executor's [`crate::stage::AnnCapabilities::supported_dtypes`] and the output
/// is stored under its stage id. Bound stage outputs are inputs, not outputs,
/// and do not appear in the returned map. The first error encountered stops
/// the run. Every binding the topology needs — each [`StageSource::External`]
/// name and each declared external stage dependency — is checked **before the
/// first [`AnnExecutor::execute`]**, so a missing binding on a later stage
/// never leaves a stateful executor half-advanced.
///
/// # Errors
///
/// * [`CortexError::MissingStageBinding`] when a referenced external ingress,
///   or a declared external stage dependency, has no binding.
/// * [`CortexError::ConflictingStageBinding`] when a bound stage output names
///   a stage the topology itself produces (it would shadow that stage).
/// * [`CortexError::InvalidTopology`] when a stage input references an output
///   that is unexpectedly absent (should not occur for a validated topology).
/// * [`CortexError::StageDTypeMismatch`] when an output dtype is unsupported.
/// * Any error returned by [`AnnExecutor::execute`], unchanged.
pub fn run_topology<E: AnnExecutor>(
    executor: &mut E,
    topology: &AnnTopology,
    bindings: &ExternalBindings<E::Tensor>,
) -> Result<BTreeMap<StageId, E::Tensor>> {
    check_bindings(topology, bindings)?;

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
                    let tensor = match outputs.get(dep) {
                        Some(tensor) => tensor,
                        None if topology.external_stage_deps().contains(dep) => bindings
                            .stage_output(dep)
                            .ok_or_else(|| CortexError::MissingStageBinding {
                                name: dep.to_string(),
                            })?,
                        None => {
                            return Err(CortexError::InvalidTopology {
                                reason: format!(
                                    "stage '{}' references output of '{dep}', which has not been produced",
                                    stage.id
                                ),
                            });
                        }
                    };
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

/// Fail closed on bindings before anything runs: every declared external stage
/// dependency and every referenced [`StageSource::External`] name needs a
/// binding, and no bound stage output may shadow a stage the topology produces.
///
/// External names are checked in topology order, so the reported missing name
/// is the first one the run would have needed.
fn check_bindings<T>(topology: &AnnTopology, bindings: &ExternalBindings<T>) -> Result<()> {
    for dep in topology.external_stage_deps() {
        if bindings.stage_output(dep).is_none() {
            return Err(CortexError::MissingStageBinding {
                name: dep.to_string(),
            });
        }
    }
    for stage in topology.stages() {
        if bindings.stage_output(&stage.id).is_some() {
            return Err(CortexError::ConflictingStageBinding {
                name: stage.id.to_string(),
                reason: "a stage output is bound for a stage the topology itself produces"
                    .to_string(),
            });
        }
        for source in &stage.inputs {
            if let StageSource::External(name) = source
                && bindings.resolve(name).is_none()
            {
                return Err(CortexError::MissingStageBinding { name: name.clone() });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::{AnnCapabilities, AnnStage, ReferenceExecutor, StageKind};
    use crate::tensor::Tensor;
    use crate::transformer::{TransformerConfig, TransformerLM};

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
    fn stage_output_bindings_reject_duplicates() {
        let hidden = Tensor::try_from_vec(vec![0.0; 8], &[1, 8]).unwrap();
        let id = StageId::new("blocks.0.residual2").unwrap();
        let err = ExternalBindings::new()
            .with_stage_output(id.clone(), &hidden)
            .unwrap()
            .with_stage_output(id, &hidden)
            .unwrap_err();
        assert!(matches!(
            err,
            CortexError::ConflictingStageBinding { ref name, .. } if name == "blocks.0.residual2"
        ));
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
    fn bindings_reject_same_name_bound_as_tokens_and_tensor() {
        let hidden = Tensor::try_from_vec(vec![0.0; 8], &[1, 8]).unwrap();
        let ids = [0u32, 1];

        // tensor first, then tokens
        let err = ExternalBindings::new()
            .with_tensor("x", &hidden)
            .unwrap()
            .with_tokens("x", &ids)
            .unwrap_err();
        assert!(
            matches!(err, CortexError::ConflictingStageBinding { ref name, .. } if name == "x"),
            "expected ConflictingStageBinding, got {err:?}"
        );

        // tokens first, then tensor
        let err = ExternalBindings::new()
            .with_tokens("x", &ids)
            .unwrap()
            .with_tensor("x", &hidden)
            .unwrap_err();
        assert!(matches!(err, CortexError::ConflictingStageBinding { .. }));

        // same kind twice is a collision too
        let err = ExternalBindings::<Tensor>::new()
            .with_tokens("x", &ids)
            .unwrap()
            .with_tokens("x", &ids)
            .unwrap_err();
        assert!(matches!(err, CortexError::ConflictingStageBinding { .. }));

        // Distinct names resolve to their own kind, whichever was bound first.
        let bindings = ExternalBindings::new()
            .with_tokens("tokens", &ids)
            .unwrap()
            .with_tensor("hidden", &hidden)
            .unwrap();
        assert!(matches!(
            bindings.resolve("tokens"),
            Some(StageInput::Tokens(_))
        ));
        assert!(matches!(
            bindings.resolve("hidden"),
            Some(StageInput::Tensor(_))
        ));
        assert!(bindings.resolve("absent").is_none());
    }

    #[test]
    fn run_topology_requires_and_guards_stage_output_bindings() {
        let model = micro_model();
        let (mut exec, topo) = ReferenceExecutor::from_transformer(&model).unwrap();
        let span = topo
            .sub_span(
                &StageId::new("final_norm").unwrap(),
                &StageId::new("readout").unwrap(),
            )
            .unwrap()
            .to_topology()
            .unwrap();
        let dep = StageId::new("blocks.1.residual2").unwrap();
        assert_eq!(span.external_stage_deps(), std::slice::from_ref(&dep));

        // Nothing bound for the declared dep: MissingStageBinding names it.
        let err = run_topology(&mut exec, &span, &ExternalBindings::new()).unwrap_err();
        assert!(
            matches!(err, CortexError::MissingStageBinding { ref name } if name == "blocks.1.residual2"),
            "got {err:?}"
        );

        // A stage output bound under an id the span itself produces would
        // shadow that stage; it is rejected before anything runs.
        let hidden = Tensor::randn(&[2, model.config.dim], 0.0, 0.1);
        let bindings = ExternalBindings::new()
            .with_stage_output(dep, &hidden)
            .unwrap()
            .with_stage_output(StageId::new("final_norm").unwrap(), &hidden)
            .unwrap();
        let err = run_topology(&mut exec, &span, &bindings).unwrap_err();
        assert!(
            matches!(err, CortexError::ConflictingStageBinding { ref name, .. } if name == "final_norm"),
            "got {err:?}"
        );
    }

    /// Counts `execute` calls so a test can prove nothing ran.
    struct CountingExecutor<'e, E: AnnExecutor> {
        inner: &'e mut E,
        executes: usize,
    }

    impl<E: AnnExecutor> AnnExecutor for CountingExecutor<'_, E> {
        type Tensor = E::Tensor;

        fn capabilities(&self) -> AnnCapabilities {
            self.inner.capabilities()
        }

        fn execute(
            &mut self,
            stage: &AnnStage,
            inputs: &[StageInput<Self::Tensor>],
        ) -> Result<Self::Tensor> {
            self.executes += 1;
            self.inner.execute(stage, inputs)
        }
    }

    #[test]
    fn run_topology_rejects_missing_external_ingress_before_any_execute() {
        // Stage 1 only needs `hidden`; stage 2 also needs `mask`. With `mask`
        // unbound the run must fail before stage 1 executes, so a stateful
        // executor is never left half-advanced.
        let hidden = Tensor::try_from_vec(vec![1.0; 8], &[1, 8]).unwrap();
        let mask = Tensor::try_from_vec(vec![0.5; 8], &[1, 8]).unwrap();
        let topo = AnnTopology::new(vec![
            stage(
                "scale",
                StageKind::Add,
                vec![StageSource::External("hidden".to_string())],
            ),
            stage(
                "masked",
                StageKind::Add,
                vec![
                    StageSource::Stage(StageId::new("scale").unwrap()),
                    StageSource::External("mask".to_string()),
                ],
            ),
        ])
        .unwrap();

        let mut reference = ReferenceExecutor::new();
        let mut exec = CountingExecutor {
            inner: &mut reference,
            executes: 0,
        };
        let partial = ExternalBindings::new()
            .with_tensor("hidden", &hidden)
            .unwrap();
        let err = run_topology(&mut exec, &topo, &partial).unwrap_err();
        assert!(
            matches!(err, CortexError::MissingStageBinding { ref name } if name == "mask"),
            "got {err:?}"
        );
        assert_eq!(
            exec.executes, 0,
            "no stage may execute before preflight passes"
        );

        // Fully bound, both stages run.
        let full = ExternalBindings::new()
            .with_tensor("hidden", &hidden)
            .unwrap()
            .with_tensor("mask", &mask)
            .unwrap();
        let out = run_topology(&mut exec, &topo, &full).unwrap();
        assert_eq!(exec.executes, 2);
        assert_eq!(out[&StageId::new("masked").unwrap()].data(), &[1.5; 8]);
    }
}
