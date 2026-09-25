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
    /// and the spike-output width, so the neuron count must equal the channel
    /// count — spike indices would otherwise silently drop or leave channels
    /// unread.
    pub fn new(num_lif: usize, num_izh: usize, num_channels: usize) -> Result<Self> {
        Self::checked(SpikingNetwork::with_dimensions(
            num_lif,
            num_izh,
            num_channels,
        ))
    }

    /// Wrap an already-configured `SpikingNetwork` (e.g. a restored checkpoint).
    /// Same 1:1 neuron-to-channel requirement as [`Self::new`].
    pub fn from_network(network: SpikingNetwork) -> Result<Self> {
        Self::checked(network)
    }

    fn checked(network: SpikingNetwork) -> Result<Self> {
        let neurons = network.neurons.len() + network.iz_neurons.len();
        if neurons != network.num_channels {
            return Err(CortexError::InvalidConfig(format!(
                "neuromod SpikingNetwork has {neurons} neurons but {} channels; the SnnBackend contract requires equal widths",
                network.num_channels
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
