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

use super::SnnEncoder;
use crate::error::{CortexError, Result};
use axon_encoder::Encoder;

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
    E: Encoder,
{
    fn encode(&mut self, ann_signal: &[f32], channels: usize) -> Result<Vec<f32>> {
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
