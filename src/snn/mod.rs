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
//! `NeuromodNetwork`, the primary integration candidate backed by the
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

#[cfg(feature = "axon-encoder")]
mod axon_adapter;

#[cfg(feature = "axon-encoder")]
pub use axon_adapter::AxonEncoder;

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
///
/// `&mut self` because real encoders are stateful (phase accumulators,
/// streaming windows); one `encode` call is one backend tick. Encoder state
/// advances when `encode` is called — it is NOT failure-atomic with respect
/// to later forward errors; callers needing a clean epoch reset the encoder.
pub trait SnnEncoder {
    /// Produce a stimulus vector of exactly `channels` length, or an error
    /// when the encoding cannot be represented in `channels` (e.g. spikes
    /// outside the channel space).
    fn encode(&mut self, ann_signal: &[f32], channels: usize) -> Result<Vec<f32>>;

    /// Restore the encoder's streaming state (phase/window) to a clean epoch.
    /// Called by [`SpikingMoeRouter::reset`] alongside the backend reset.
    /// Stateless encoders use the no-op default.
    fn reset(&mut self) {}
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
    fn encode(&mut self, ann_signal: &[f32], channels: usize) -> Result<Vec<f32>> {
        let mut stimulus = vec![0.0f32; channels];
        if ann_signal.is_empty() {
            return Ok(stimulus);
        }
        for (idx, slot) in stimulus.iter_mut().enumerate() {
            *slot = ann_signal[idx % ann_signal.len()] * self.gain;
        }
        Ok(stimulus)
    }
}

/// Per-channel spike-count readout: each channel's output is the number of
/// times it spiked in the tick (0 or 1 for index-style backends). Indices
/// outside `0..channels` are ignored here; `SpikingMoeRouter` validates the
/// spike list before decoding so such indices never reach this decoder.
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
/// `B`, and decoded by `D`.
///
/// # Routing contract
///
/// * **Scores:** `gate + spike_boost * activity`, where `activity` is the
///   decoder readout for the expert's channel (≥ 0) and `spike_boost` is
///   [`DEFAULT_SPIKE_BOOST`] unless changed with [`Self::with_spike_boost`].
///   The boost is additive so it is monotone in spike activity for *any*
///   logit sign: a spike can only promote its expert, never demote it (the
///   earlier multiplicative `gate * (1 + spikes)` pushed negative logits
///   further down the ranking). Selection is `route_top_k` over those scores;
///   `expert_weights` is their softmax.
/// * **Hidden state:** `embedding * Σ_{selected} expert_weights[e] * a[e]`,
///   where `a[e] = activity[e] / max(activity)` lies in `[0, 1]`. The gain is
///   therefore bounded by the selected softmax mass (≤ 1), like
///   `MoeRouter::forward`, whatever the decoder's count scale. When none of
///   the selected experts is active (a silent tick, or spikes only on
///   unselected channels) `a[e] = 1` for every selected expert, so the
///   output degrades to the dense selected mass instead of a zero hidden
///   state.
///
/// # Validation
///
/// NaN gate scores and NaN encoder output are rejected before the backend
/// steps, preserving the fail-closed ordering of the old embedded path.
/// After the step, the backend's spike list is checked against its contract
/// — every index in `0..channels` ([`CortexError::SnnSpikeOutOfRange`]) and
/// each channel at most once ([`CortexError::SnnDuplicateSpike`]) — and the
/// decoder readout must have exactly `channels` entries
/// ([`CortexError::SnnReadoutLength`]), be NaN-free ([`CortexError::SnnNan`])
/// and be finite and non-negative ([`CortexError::SnnInvalidReadout`]). Errors raised after the step consume
/// the tick — `reset()` restores a clean epoch.
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
    spike_boost: f32,
}

/// Default additive routing boost per unit of decoded spike activity: one
/// spike on a channel adds `1.0` to its expert's gate logit.
pub const DEFAULT_SPIKE_BOOST: f32 = 1.0;

/// Validate one backend tick against the [`SnnStepOutput`] contract.
fn validate_spikes(step_out: &SnnStepOutput, channels: usize) -> Result<()> {
    let mut seen = vec![false; channels];
    for &channel in &step_out.spikes {
        if channel >= channels {
            return Err(CortexError::SnnSpikeOutOfRange { channel, channels });
        }
        if seen[channel] {
            return Err(CortexError::SnnDuplicateSpike { channel });
        }
        seen[channel] = true;
    }
    Ok(())
}

/// Validate a decoder readout: exactly `channels` entries, each finite and
/// non-negative. NaN keeps its dedicated variant.
fn validate_readout(decoded: &[f32], channels: usize) -> Result<()> {
    if decoded.len() != channels {
        return Err(CortexError::SnnReadoutLength {
            expected: channels,
            got: decoded.len(),
        });
    }
    for (channel, &value) in decoded.iter().enumerate() {
        if value.is_nan() {
            return Err(CortexError::SnnNan {
                stage: "decoder readout",
            });
        }
        if !value.is_finite() || value < 0.0 {
            return Err(CortexError::SnnInvalidReadout { channel, value });
        }
    }
    Ok(())
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
            spike_boost: DEFAULT_SPIKE_BOOST,
        })
    }

    /// Set the additive routing boost per unit of decoded spike activity
    /// (default [`DEFAULT_SPIKE_BOOST`]). Must be finite and non-negative so
    /// the boost stays monotone; `0.0` makes routing purely ANN-driven while
    /// the backend still steps and shapes the hidden state. For a finite gate
    /// the score `gate + spike_boost * activity` saturates at `f32::MAX`
    /// instead of overflowing to `+Inf`; a `±Inf` gate keeps its value, so the
    /// boost never turns a finite score infinite or a `-Inf` gate into NaN.
    pub fn with_spike_boost(mut self, spike_boost: f32) -> Result<Self> {
        if !spike_boost.is_finite() || spike_boost < 0.0 {
            return Err(CortexError::InvalidConfig(format!(
                "spike_boost must be finite and >= 0, got {spike_boost}"
            )));
        }
        self.spike_boost = spike_boost;
        Ok(self)
    }

    /// Additive routing boost per unit of decoded spike activity.
    pub fn spike_boost(&self) -> f32 {
        self.spike_boost
    }

    /// Shared forward path. `frozen` selects [`SnnBackend::step_frozen`].
    fn forward_impl(&mut self, embedding: &[f32], frozen: bool) -> Result<crate::moe::MoeOutput> {
        let gate_scores = self.router.gate_scores(embedding)?;
        // Fail closed before the backend consumes a tick.
        crate::moe::reject_nan_routing_scores(&gate_scores)?;

        // Encoders may inject NaN stimuli; validate before the backend
        // consumes a tick so the whole forward stays fail-closed. ±Inf is
        // rankable per the finite-value policy and passes through.
        // The one-channel-per-expert invariant is checked in `new`, but
        // `channels()` is re-read every tick (a backend may be resized through
        // `backend_mut`, and user backends need not be constant-width).
        // Re-check it before the tick so `decoded[expert_id]` below cannot
        // index past the readout, and fail closed without consuming a step.
        let channels = self.backend.channels();
        let num_experts = self.router.num_experts();
        if channels < num_experts {
            return Err(CortexError::SnnChannelMismatch {
                experts: num_experts,
                channels,
            });
        }
        let stimulus = self.encoder.encode(&gate_scores, channels)?;
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
        // need a clean epoch should `reset()` after such an error. The
        // pre-tick width is reused so encode and decode agree even if the
        // backend changed shape during the step.
        validate_spikes(&step_out, channels)?;
        let decoded = self.decoder.decode(&step_out, channels);
        validate_readout(&decoded, channels)?;

        // Additive boost: monotone in activity for any logit sign (issue #93).
        // Both factors are finite and >= 0, so the bonus is finite once its
        // product is capped; a finite gate plus that bonus is then saturated
        // at f32::MAX so it can never become +Inf, while a ±Inf gate keeps its
        // sign (bonus >= 0, so -Inf + bonus cannot be NaN).
        let routing_scores: Vec<f32> = (0..num_experts)
            .map(|expert_id| {
                let gate = gate_scores[expert_id];
                let bonus = (self.spike_boost * decoded[expert_id]).min(f32::MAX);
                if gate.is_finite() {
                    (gate + bonus).min(f32::MAX)
                } else {
                    gate
                }
            })
            .collect();
        let (expert_weights, selected_experts) =
            crate::moe::route_top_k(&routing_scores, self.router.top_k())?;

        // Hidden gain = selected softmax mass weighted by activity normalised
        // to the tick's peak, so it stays within [0, 1] regardless of the
        // decoder's count scale. When no selected expert is active (silent
        // tick, or spikes only on unselected channels) the gain falls back to
        // the dense selected mass rather than collapsing to zero.
        let peak_activity = decoded[..num_experts]
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let selected_active = selected_experts.iter().any(|&idx| decoded[idx] > 0.0);
        let spiking_mass: f32 = selected_experts
            .iter()
            .map(|&idx| {
                let activity = if selected_active {
                    decoded[idx] / peak_activity
                } else {
                    1.0
                };
                expert_weights[idx] * activity
            })
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

    /// Reset the backend and encoder epochs; router (ANN) state is untouched.
    pub fn reset(&mut self) {
        self.encoder.reset();
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

    pub fn encoder(&self) -> &E {
        &self.encoder
    }

    /// Access the encoder (e.g. to reset its streaming state after an error).
    pub fn encoder_mut(&mut self) -> &mut E {
        &mut self.encoder
    }
}

#[cfg(test)]
mod tests;
