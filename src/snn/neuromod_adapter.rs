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

use super::{SnnBackend, SnnCapabilities, SnnStepOutput};
use crate::error::{CortexError, Result};
use neuromod::{NeuroModulators, SpikingNetwork};

/// [`SnnBackend`] over `neuromod::SpikingNetwork` (LIF + Izhikevich banks).
pub struct NeuromodNetwork {
    network: SpikingNetwork,
    /// Modulator frame applied to every `step`/`step_frozen` call.
    pub modulators: NeuroModulators,
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
    /// must also carry at least one non-zero finite weight: a blank neuron
    /// receives no input current, never reaches threshold, and (because the
    /// renormalization pass skips it) cannot be bootstrapped by STDP, so its
    /// channel would be permanently silent.
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
        if let Some(idx) = network
            .neurons
            .iter()
            .position(|neuron| !neuron.weights.iter().any(|w| w.is_finite() && *w != 0.0))
        {
            return Err(CortexError::InvalidConfig(format!(
                "neuromod SpikingNetwork LIF neuron {idx} has no non-zero finite weight, so channel {idx} can never spike; seed its weights (e.g. via network_mut()) before wrapping it"
            )));
        }
        Ok(Self {
            network,
            modulators: NeuroModulators::default(),
        })
    }

    pub fn network(&self) -> &SpikingNetwork {
        &self.network
    }

    pub fn network_mut(&mut self) -> &mut SpikingNetwork {
        &mut self.network
    }
}

impl SnnBackend for NeuromodNetwork {
    fn channels(&self) -> usize {
        self.network.num_channels
    }

    fn step(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput> {
        let spikes = self
            .network
            .step(stimulus, &self.modulators)
            .map_err(|err| CortexError::SnnBackend(err.to_string()))?;
        Ok(SnnStepOutput { spikes })
    }

    fn step_frozen(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput> {
        // neuromod 0.6.0 (latest published) has no `step_frozen`; the frozen
        // evaluation path lands in a later release. Until then frozen forwards
        // delegate to `step`, and `capabilities().frozen_evaluation` is false
        // so orchestrators know plasticity state can advance on this path.
        self.step(stimulus)
    }

    fn reset(&mut self) {
        self.network.reset();
    }

    fn capabilities(&self) -> SnnCapabilities {
        SnnCapabilities {
            backend_name: "neuromod::SpikingNetwork",
            frozen_evaluation: false,
            plasticity: true,
            caller_rng: true,
            neuromodulation: true,
        }
    }
}
