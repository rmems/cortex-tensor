// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adapter from [`SnnBackend`] to `neuromod`'s `SpikingNetwork`.
//!
//! Primary SNN integration candidate per the v0.2 boundary reset: neuron
//! dynamics, plasticity (R-STDP), and neuromodulation stay owned by the
//! `neuromod` crate; this module only translates the boundary types.
//!
//! `neuromod` validates stimulus length against `num_channels` itself and its
//! step preflight is failure-atomic, which satisfies the [`SnnBackend`]
//! failure-atomicity contract. `NeuroModulators::default()` is used for every
//! step; callers that need reward-modulated dynamics should set
//! [`NeuromodNetwork::modulators`] or step the wrapped network directly via
//! [`NeuromodNetwork::network_mut`].
//!
//! `step_frozen` maps onto neuromod 0.7's `SpikingNetwork::step_frozen`, which
//! runs the full runtime pipeline but leaves weights, eligibility traces, LIF
//! thresholds/decay, and the stored modulators bit-for-bit unchanged. With a
//! seed installed ([`NeuromodNetwork::with_seed`] / [`NeuromodNetwork::set_seed`])
//! both paths draw their Bernoulli input encoding from one caller-owned
//! `StdRng` via `step_with_rng` / `step_frozen_with_rng`, so a run replays
//! bit-for-bit; without a seed they use neuromod's thread-local RNG and
//! `capabilities().caller_rng` is `false`.

use super::{SnnBackend, SnnCapabilities, SnnStepOutput};
use crate::error::{CortexError, Result};
use neuromod::{NeuroModulators, SeedableRng, SpikingNetwork, StdRng};

/// Caller-owned RNG for deterministic replay: the seed is kept so
/// [`SnnBackend::reset`] can restart the stream for a new epoch.
struct SeededRng {
    seed: u64,
    rng: StdRng,
}

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

/// [`SnnBackend`] over `neuromod::SpikingNetwork` (LIF + Izhikevich banks).
pub struct NeuromodNetwork {
    network: SpikingNetwork,
    /// Modulator frame applied to every `step`/`step_frozen` call.
    pub modulators: NeuroModulators,
    rng: Option<SeededRng>,
}

impl NeuromodNetwork {
    /// Build a network of `num_lif` LIF and `num_izh` Izhikevich neurons over
    /// `num_channels` stimulus channels.
    ///
    /// The `SnnBackend` contract uses `channels()` as both the stimulus width
    /// and the spike-output width. `neuromod`'s `step` reports **LIF** spikes
    /// only (Izhikevich spikes are not listed in its return value), so
    /// `num_lif` must equal `num_channels`; the Izhikevich bank is auxiliary
    /// and may be any size. A channel backed by an Izhikevich neuron would
    /// never spike and its expert would silently stay ANN-only.
    ///
    /// `SpikingNetwork::with_dimensions` zero-initialises every LIF weight and
    /// the engine's L1 renormalization leaves a blank neuron blank, so an
    /// unseeded network can never spike or learn. `new` therefore wires LIF
    /// neuron *i* to channel *i* at the configured maximum weight (identity
    /// wiring, matching [`super::SpikingMoeRouter`]'s expert-*i*-at-channel-*i*
    /// mapping). Callers wanting a different initialisation adjust the weights
    /// through [`Self::network_mut`] or build the network and use
    /// [`Self::from_network`].
    ///
    /// The backend starts on neuromod's thread-local RNG; see
    /// [`Self::with_seed`] for deterministic replay.
    pub fn new(num_lif: usize, num_izh: usize, num_channels: usize) -> Result<Self> {
        let mut network = SpikingNetwork::with_dimensions(num_lif, num_izh, num_channels);
        let (_, w_max) = network.stdp_config.weight_bounds();
        for (idx, neuron) in network.neurons.iter_mut().enumerate() {
            if let Some(weight) = neuron.weights.get_mut(idx) {
                *weight = w_max;
            }
        }
        Self::checked(network)
    }

    /// Wrap an already-configured `SpikingNetwork` (e.g. a restored checkpoint).
    ///
    /// Same one-LIF-per-channel requirement as [`Self::new`]. Every LIF neuron
    /// must also carry a weight vector of at least `num_channels` finite
    /// entries with at least one strictly positive weight in
    /// `weights[..num_channels]`. The engine integrates only that range, takes
    /// `|stimulus|` (so a negative-only neuron receives non-positive current),
    /// skips channels past a short vector, and (because the renormalization
    /// pass skips a blank neuron) cannot bootstrap such a neuron through STDP:
    /// in each case the channel would be permanently silent.
    pub fn from_network(network: SpikingNetwork) -> Result<Self> {
        Self::checked(network)
    }

    fn checked(network: SpikingNetwork) -> Result<Self> {
        let lif = network.neurons.len();
        if lif != network.num_channels {
            return Err(CortexError::InvalidConfig(format!(
                "neuromod SpikingNetwork has {lif} LIF neurons but {} channels; the SnnBackend contract requires one LIF neuron per channel (the {} Izhikevich neurons do not count: neuromod reports LIF spikes only)",
                network.num_channels,
                network.iz_neurons.len()
            )));
        }
        let channels = network.num_channels;
        for (idx, neuron) in network.neurons.iter().enumerate() {
            // Only `weights[..num_channels]` is ever integrated; a longer
            // vector is tolerated, a shorter one leaves channels the engine
            // silently skips.
            let Some(in_range) = neuron.weights.get(..channels) else {
                return Err(CortexError::InvalidConfig(format!(
                    "neuromod SpikingNetwork LIF neuron {idx} has {} weights but the network has {channels} channels; the engine integrates one weight per channel",
                    neuron.weights.len()
                )));
            };
            if let Some(ch) = in_range.iter().position(|w| !w.is_finite()) {
                return Err(CortexError::InvalidConfig(format!(
                    "neuromod SpikingNetwork LIF neuron {idx} has a non-finite weight on channel {ch}; it would poison the membrane potential"
                )));
            }
            // The engine integrates `|stimulus| * weight`, so only a strictly
            // positive weight can ever drive the membrane toward threshold.
            if !in_range.iter().any(|w| *w > 0.0) {
                return Err(CortexError::InvalidConfig(format!(
                    "neuromod SpikingNetwork LIF neuron {idx} has no positive weight on channels 0..{channels}, so channel {idx} can never spike; seed its weights (e.g. via network_mut()) before wrapping it"
                )));
            }
        }
        Ok(Self {
            network,
            modulators: NeuroModulators::default(),
            rng: None,
        })
    }

    /// Drive the Bernoulli input encoding from a `StdRng` seeded with `seed`
    /// instead of neuromod's thread-local generator, so an identical network,
    /// stimulus sequence, and seed replays bit-for-bit. The generator is
    /// reseeded on [`SnnBackend::reset`] (each epoch replays from its reset)
    /// and is not part of the wrapped network's serde checkpoint.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.set_seed(Some(seed));
        self
    }

    /// Install (`Some`) or remove (`None`) the replay seed; see
    /// [`Self::with_seed`]. Installing a seed restarts the stream from it.
    pub fn set_seed(&mut self, seed: Option<u64>) {
        self.rng = seed.map(SeededRng::new);
    }

    /// The replay seed in effect, or `None` when stepping on the thread-local RNG.
    pub fn seed(&self) -> Option<u64> {
        self.rng.as_ref().map(|r| r.seed)
    }

    pub fn network(&self) -> &SpikingNetwork {
        &self.network
    }

    pub fn network_mut(&mut self) -> &mut SpikingNetwork {
        &mut self.network
    }

    fn into_output(
        result: std::result::Result<Vec<usize>, neuromod::StepError>,
    ) -> Result<SnnStepOutput> {
        result
            .map(|spikes| SnnStepOutput { spikes })
            .map_err(|err| CortexError::SnnBackend(err.to_string()))
    }
}

impl SnnBackend for NeuromodNetwork {
    fn channels(&self) -> usize {
        self.network.num_channels
    }

    fn step(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput> {
        let result = match self.rng.as_mut() {
            Some(seeded) => self
                .network
                .step_with_rng(stimulus, &self.modulators, &mut seeded.rng),
            None => self.network.step(stimulus, &self.modulators),
        };
        Self::into_output(result)
    }

    fn step_frozen(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput> {
        let result = match self.rng.as_mut() {
            Some(seeded) => {
                self.network
                    .step_frozen_with_rng(stimulus, &self.modulators, &mut seeded.rng)
            }
            None => self.network.step_frozen(stimulus, &self.modulators),
        };
        Self::into_output(result)
    }

    fn reset(&mut self) {
        self.network.reset();
        if let Some(seeded) = self.rng.as_mut() {
            *seeded = SeededRng::new(seeded.seed);
        }
    }

    fn capabilities(&self) -> SnnCapabilities {
        SnnCapabilities {
            backend_name: "neuromod::SpikingNetwork",
            frozen_evaluation: true,
            plasticity: true,
            caller_rng: self.rng.is_some(),
            neuromodulation: true,
        }
    }
}
