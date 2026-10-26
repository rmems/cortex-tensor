// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adapter from [`SnnEncoder`] to `axon-encoder` spike encoders.
//!
//! `axon-encoder` owns the canonical encoding role in the rmems SNN stack
//! (rate, temporal, predictive, population, neuromodulator-driven). This
//! module turns one encoder call — one backend tick — into the per-channel
//! stimulus vector [`SnnBackend`] expects: each spike contributes `+1.0`
//! (excitatory) or `-1.0` (inhibitory) to its channel.
//!
//! [`RateEncoder`](crate::snn::RateEncoder) remains the deterministic
//! reference fallback: it needs no crate dependency and keeps routing tests
//! self-contained. Use `AxonEncoder` when the encoded spike stream itself
//! matters (rates, timing, plasticity dynamics).

use std::any::Any;

use super::SnnEncoder;
use crate::error::{CortexError, Result};
use axon_encoder::Encoder;
use axon_encoder::prelude::PopulationEncoder;

/// Spike channel ids are `u16` (`axon_encoder`'s `MAX_SPIKE_CHANNELS`, which is
/// `pub(crate)`). `PopulationEncoder::try_new` checks `num_neurons` alone;
/// `encode_step` asserts `input.len() * num_neurons` against this same limit.
const POPULATION_SPIKE_CHANNELS: usize = u16::MAX as usize + 1;

/// Every axon-encoder 0.6 encoder tags spikes with the input index as a `u16`.
/// `EmbeddingRateEncoder` and `DerivativeEncoder` `expect` that conversion
/// (the others `break`), so more inputs than this can panic inside `encode_step`.
const MAX_ENCODER_INPUTS: usize = u16::MAX as usize + 1;

/// [`SnnEncoder`] over any `axon_encoder::Encoder` implementation.
///
/// Generic so callers choose the encoding scheme:
/// `axon_encoder::prelude::RateEncoder`, `LatencyEncoder`, `DeltaEncoder`,
/// `PredictiveEncoder`, `PopulationEncoder`, …
///
/// Streaming is the contract: `SnnEncoder::encode` delegates to
/// [`Encoder::encode_step`], so one forward pass advances the encoder's
/// internal phase/state by exactly one tick. That advancement is NOT
/// failure-atomic: encoder state moves even if a later stage rejects the
/// forward (NaN stimulus, backend step error). [`SnnEncoder::reset`]
/// delegates to the wrapped encoder's `reset` and is called automatically
/// by `SpikingMoeRouter::reset` to restore a clean epoch.
///
/// axon-encoder 0.6 `PopulationEncoder` encodes every input channel
/// (`channel = input_index * num_neurons + neuron`) and panics when that
/// product does not fit the `u16` spike-id space. 0.5 read only
/// `input.first()` and never asserted on the product. [`SnnEncoder::encode`]
/// checks the product first and returns [`CortexError::InvalidConfig`],
/// because the [`Encoder`] trait does not report the width. The `'static`
/// bound exists so that check can see a concrete `PopulationEncoder` inside
/// this generic impl; stable Rust has no specialization. Every encoder also
/// gets a universal `ann_signal.len() <= 65,536` check, because
/// `EmbeddingRateEncoder` and `DerivativeEncoder` panic on an input index past
/// `u16::MAX`.
pub struct AxonEncoder<E> {
    encoder: E,
}

impl<E> AxonEncoder<E> {
    /// Wrap a configured `axon-encoder` encoder.
    pub fn new(encoder: E) -> Self {
        Self { encoder }
    }

    pub fn encoder(&self) -> &E {
        &self.encoder
    }

    /// Access the wrapped encoder (e.g. to `reset()` its streaming state).
    pub fn encoder_mut(&mut self) -> &mut E {
        &mut self.encoder
    }
}

impl<E> SnnEncoder for AxonEncoder<E>
where
    E: Encoder + 'static,
{
    fn encode(&mut self, ann_signal: &[f32], channels: usize) -> Result<Vec<f32>> {
        if ann_signal.len() > MAX_ENCODER_INPUTS {
            return Err(CortexError::InvalidConfig(format!(
                "{} inputs exceed the {MAX_ENCODER_INPUTS}-channel u16 id space of axon-encoder spikes",
                ann_signal.len()
            )));
        }
        if let Some(population) = (&self.encoder as &dyn Any).downcast_ref::<PopulationEncoder>() {
            reject_population_width(population.num_neurons(), ann_signal.len())?;
        }
        let mut stimulus = vec![0.0f32; channels];
        let output = self.encoder.encode_step(ann_signal);
        for spike in &output.spikes {
            let idx = usize::from(spike.channel);
            if idx >= channels {
                return Err(CortexError::SnnSpikeOutOfRange {
                    channel: idx,
                    channels,
                });
            }
            stimulus[idx] += if spike.polarity { 1.0 } else { -1.0 };
        }
        Ok(stimulus)
    }

    fn reset(&mut self) {
        self.encoder.reset();
    }
}

/// `encode_step` asserts this product. Reject it first so a router forward
/// returns a typed error instead of panicking, and so encoder state does not
/// move for a call that cannot be represented.
fn reject_population_width(neurons: usize, inputs: usize) -> Result<()> {
    match inputs.checked_mul(neurons) {
        Some(width) if width <= POPULATION_SPIKE_CHANNELS => Ok(()),
        Some(width) => Err(CortexError::InvalidConfig(format!(
            "PopulationEncoder would address {width} spike channels ({inputs} inputs × {neurons} neurons), exceeding the {POPULATION_SPIKE_CHANNELS}-channel u16 id space"
        ))),
        None => Err(CortexError::InvalidConfig(format!(
            "PopulationEncoder channel count overflows for {inputs} inputs × {neurons} neurons"
        ))),
    }
}
