// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral SNN execution contract for ANN/SNN hybrid stages.
//!
//! Cortex stays hybrid-capable without embedding a bespoke SNN runtime inside
//! `MoeRouter`. Neuron dynamics live in focused reusable crates behind the
//! [`SnnBackend`] trait; this module defines the ANN↔SNN boundary:
//!
//! * **Encode:** [`SnnEncoder`] maps ANN-side signals (e.g. MoE gate scores or
//!   hidden activations) into backend stimulus channels.
//! * **Step:** [`SnnBackend::step`] / [`SnnBackend::step_frozen`] advance the
//!   backend one tick. `step_frozen` is the held-out evaluation path: runtime
//!   outputs are produced without retaining plasticity changes.
//! * **Decode:** [`SnnDecoder`] maps [`SnnStepOutput`] back into per-channel
//!   f32 readouts ANN stages can consume.
//! * **Semantics:** [`SnnBackend::reset`] starts a new epoch and
//!   [`SnnBackend::capabilities`] reports what the backend supports so
//!   orchestrators such as `hybrid-fusion` can pick execution paths.
//!
//! Adapters are optional and feature-gated. Enable `neuromod` for
//! [`NeuromodNetwork`], the primary integration candidate backed by the
//! published `neuromod` crate's `SpikingNetwork`.
//!
//! [`SpikingMoeRouter`] composes a [`MoeRouter`] (gate scores + deterministic
//! top-k) with an `SnnBackend`, replacing the removed embedded `SpikingSim`
//! routing mode.

use crate::error::{CortexError, Result};
use crate::moe::MoeRouter;

#[cfg(feature = "neuromod")]
mod neuromod_adapter;

#[cfg(feature = "neuromod")]
pub use neuromod_adapter::NeuromodNetwork;

/// Capabilities a backend advertises to orchestrators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnnCapabilities {
    /// Human-readable backend identity, e.g. `"neuromod::SpikingNetwork"`.
    pub backend_name: &'static str,
    /// Held-out evaluation path that preserves plasticity-controlled state.
    pub frozen_evaluation: bool,
    /// Normal stepping retains plasticity changes (eligibility traces, weights).
    pub plasticity: bool,
    /// Backend accepts caller-injected RNG for deterministic replay.
    pub caller_rng: bool,
    /// Backend consumes neuromodulator state (dopamine, ACh, …) per step.
    pub neuromodulation: bool,
}

/// One backend tick of spiking activity.
#[derive(Debug, Clone, Default)]
pub struct SnnStepOutput {
    /// Indices of channels/neurons that spiked this tick, in backend order.
    /// A channel may appear at most once per tick.
    pub spikes: Vec<usize>,
}

/// Execution contract an SNN crate adapter implements.
///
/// Implementations must be failure-atomic: a rejected input leaves backend
/// state unmodified, so callers can fail closed on NaN/length errors before a
/// tick is consumed.
pub trait SnnBackend {
    /// Stimulus width `step` expects — and the spike-index space of
    /// [`SnnStepOutput::spikes`] (`0..channels`). Backends whose output
    /// width differs must map or reject at construction.
    fn channels(&self) -> usize;

    /// Advance one tick with learning-enabled dynamics.
    fn step(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput>;

    /// Advance one tick for held-out evaluation without retaining
    /// plasticity-controlled state. Backends without a frozen path may
    /// delegate to [`Self::step`] and report `frozen_evaluation = false`.
    fn step_frozen(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput>;

    /// Reset runtime state (membranes, spike times, tick counter) while
    /// keeping learned weights and configured parameters.
    fn reset(&mut self);

    /// Report what this backend supports.
    fn capabilities(&self) -> SnnCapabilities;
}

/// ANN → SNN boundary: map an ANN-side f32 signal into backend stimuli.
pub trait SnnEncoder {
    /// Produce a stimulus vector of exactly `channels` length.
    fn encode(&self, ann_signal: &[f32], channels: usize) -> Vec<f32>;
}

/// SNN → ANN boundary: map one step of spikes into per-channel f32 readouts.
pub trait SnnDecoder {
    /// Produce a readout vector of exactly `channels` length.
    fn decode(&self, output: &SnnStepOutput, channels: usize) -> Vec<f32>;
}

/// Linear encoder: scales each input by `gain` and tiles it across the
/// backend's channel count. Expert *i* occupies channel `i % len`, so with
/// `channels >= len` each signal element owns a contiguous prefix of channels.
#[derive(Debug, Clone, Copy)]
pub struct RateEncoder {
    /// Multiplier applied to every input element.
    pub gain: f32,
}

impl Default for RateEncoder {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

impl SnnEncoder for RateEncoder {
    fn encode(&self, ann_signal: &[f32], channels: usize) -> Vec<f32> {
        let mut stimulus = vec![0.0f32; channels];
        if ann_signal.is_empty() {
            return stimulus;
        }
        for (idx, slot) in stimulus.iter_mut().enumerate() {
            *slot = ann_signal[idx % ann_signal.len()] * self.gain;
        }
        stimulus
    }
}

/// Per-channel spike-count readout: each channel's output is the number of
/// times it spiked in the tick (0 or 1 for index-style backends).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpikeCountDecoder;

impl SnnDecoder for SpikeCountDecoder {
    fn decode(&self, output: &SnnStepOutput, channels: usize) -> Vec<f32> {
        let mut counts = vec![0.0f32; channels];
        for &idx in &output.spikes {
            if idx < channels {
                counts[idx] += 1.0;
            }
        }
        counts
    }
}

/// Spiking MoE routing through an external SNN backend.
///
/// Replaces the removed embedded `RoutingMode::SpikingSim`: gate scores are
/// computed by [`MoeRouter`], encoded into backend stimuli by `E`, stepped by
/// `B`, and decoded by `D`. Routing scores are `gate * (1 + spikes)`, so a
/// spiked channel amplifies its expert without masking the ANN gate signal.
/// NaN gate scores and NaN encoder output are rejected before the
/// backend steps, preserving the fail-closed ordering of the old embedded
/// path. Errors raised after the step (backend step failure, NaN or
/// wrong-length decoder output) consume the tick — `reset()` restores a
/// clean epoch.
///
/// `B` is generic (not `dyn`) so callers keep concrete backend types for
/// checkpointing and backend-specific tuning.
pub struct SpikingMoeRouter<B, E = RateEncoder, D = SpikeCountDecoder>
where
    B: SnnBackend,
    E: SnnEncoder,
    D: SnnDecoder,
{
    router: MoeRouter,
    backend: B,
    encoder: E,
    decoder: D,
}

impl<B, E, D> SpikingMoeRouter<B, E, D>
where
    B: SnnBackend,
    E: SnnEncoder,
    D: SnnDecoder,
{
    /// Every expert must own a dedicated backend channel: expert *i* is
    /// encoded/decoded at channel *i*, so `channels < num_experts` would
    /// silently leave trailing experts permanently ANN-only.
    pub fn new(router: MoeRouter, backend: B, encoder: E, decoder: D) -> Result<Self> {
        if backend.channels() < router.num_experts() {
            return Err(CortexError::SnnChannelMismatch {
                experts: router.num_experts(),
                channels: backend.channels(),
            });
        }
        Ok(Self {
            router,
            backend,
            encoder,
            decoder,
        })
    }

    /// Shared forward path. `frozen` selects [`SnnBackend::step_frozen`].
    fn forward_impl(&mut self, embedding: &[f32], frozen: bool) -> Result<crate::moe::MoeOutput> {
        let gate_scores = self.router.gate_scores(embedding)?;
        // Fail closed before the backend consumes a tick.
        crate::moe::reject_nan_routing_scores(&gate_scores)?;

        // Encoders may inject NaN stimuli; validate before the backend
        // consumes a tick so the whole forward stays fail-closed. ±Inf is
        // rankable per the finite-value policy and passes through.
        let stimulus = self.encoder.encode(&gate_scores, self.backend.channels());
        if stimulus.iter().any(|v| v.is_nan()) {
            return Err(CortexError::SnnNan {
                stage: "encoder stimulus",
            });
        }
        let step_out = if frozen {
            self.backend.step_frozen(&stimulus)?
        } else {
            self.backend.step(&stimulus)?
        };
        // Failures past this point have consumed a backend tick; callers that
        // need a clean epoch should `reset()` after such an error.
        let decoded = self.decoder.decode(&step_out, self.backend.channels());
        if decoded.len() != self.backend.channels() || decoded.iter().any(|v| v.is_nan()) {
            return Err(CortexError::SnnNan {
                stage: "decoder readout",
            });
        }

        let num_experts = self.router.num_experts();
        let routing_scores: Vec<f32> = (0..num_experts)
            .map(|expert_id| {
                let spikes = decoded.get(expert_id).copied().unwrap_or(0.0);
                gate_scores[expert_id] * (1.0 + spikes)
            })
            .collect();
        let (expert_weights, selected_experts) =
            crate::moe::route_top_k(&routing_scores, self.router.top_k())?;

        let spiking_mass: f32 = selected_experts
            .iter()
            .map(|&idx| decoded.get(idx).copied().unwrap_or(0.0))
            .sum();
        let hidden: Vec<f32> = embedding.iter().map(|&v| v * spiking_mass).collect();

        Ok(crate::moe::MoeOutput {
            expert_weights,
            selected_experts,
            hidden,
        })
    }

    /// Learning-enabled forward: backend steps retain plasticity.
    pub fn forward(&mut self, embedding: &[f32]) -> Result<crate::moe::MoeOutput> {
        if embedding.len() != crate::types::EMBEDDING_DIM {
            return Err(CortexError::InputLengthMismatch {
                expected: crate::types::EMBEDDING_DIM,
                got: embedding.len(),
            });
        }
        self.forward_impl(embedding, false)
    }

    /// Held-out forward: backend runs its frozen evaluation path.
    pub fn forward_frozen(&mut self, embedding: &[f32]) -> Result<crate::moe::MoeOutput> {
        if embedding.len() != crate::types::EMBEDDING_DIM {
            return Err(CortexError::InputLengthMismatch {
                expected: crate::types::EMBEDDING_DIM,
                got: embedding.len(),
            });
        }
        self.forward_impl(embedding, true)
    }

    /// Reset the backend epoch; router (ANN) state is untouched.
    pub fn reset(&mut self) {
        self.backend.reset();
    }

    /// Capabilities of the wrapped backend, for orchestrator reporting.
    pub fn capabilities(&self) -> SnnCapabilities {
        self.backend.capabilities()
    }

    pub fn router(&self) -> &MoeRouter {
        &self.router
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

#[cfg(test)]
mod tests;
