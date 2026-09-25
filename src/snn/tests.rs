// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::error::HybridError;
use crate::moe::MoeRouter;
use crate::types::EMBEDDING_DIM;

/// Deterministic in-tree backend for contract tests. Spikes every channel
/// whose stimulus exceeds `threshold`; tracks step count to prove state
/// advances and reset works. Not neuron dynamics — the test boundary.
struct MockBackend {
    channels: usize,
    threshold: f32,
    steps: usize,
    frozen_steps: usize,
}

impl MockBackend {
    fn new(channels: usize, threshold: f32) -> Self {
        Self {
            channels,
            threshold,
            steps: 0,
            frozen_steps: 0,
        }
    }
}

impl SnnBackend for MockBackend {
    fn channels(&self) -> usize {
        self.channels
    }

    fn step(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput> {
        assert_eq!(stimulus.len(), self.channels);
        self.steps += 1;
        Ok(SnnStepOutput {
            spikes: stimulus
                .iter()
                .enumerate()
                .filter(|(_, v)| **v > self.threshold)
                .map(|(idx, _)| idx)
                .collect(),
        })
    }

    fn step_frozen(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput> {
        assert_eq!(stimulus.len(), self.channels);
        self.frozen_steps += 1;
        Ok(SnnStepOutput { spikes: Vec::new() })
    }

    fn reset(&mut self) {
        self.steps = 0;
        self.frozen_steps = 0;
    }

    fn capabilities(&self) -> SnnCapabilities {
        SnnCapabilities {
            backend_name: "mock",
            frozen_evaluation: true,
            plasticity: true,
            caller_rng: false,
            neuromodulation: false,
        }
    }
}

fn spiking_router(backend: MockBackend) -> SpikingMoeRouter<MockBackend> {
    let router = MoeRouter::load_with_mode("", 8, 2, crate::moe::RoutingMode::DenseSim).unwrap();
    SpikingMoeRouter::new(router, backend, RateEncoder::default(), SpikeCountDecoder).unwrap()
}

#[test]
fn rate_encoder_tiles_signal_across_channels() {
    let enc = RateEncoder { gain: 2.0 };
    assert_eq!(enc.encode(&[0.5, -1.0], 5), vec![1.0, -2.0, 1.0, -2.0, 1.0]);
    assert_eq!(enc.encode(&[], 3), vec![0.0; 3]);
}

#[test]
fn spike_count_decoder_counts_per_channel() {
    let out = SnnStepOutput {
        spikes: vec![0, 2, 9],
    };
    assert_eq!(SpikeCountDecoder.decode(&out, 4), vec![1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn spiking_forward_routes_and_steps_backend() {
    let backend = MockBackend::new(8, 0.0);
    let mut model = spiking_router(backend);
    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    let chunk = (EMBEDDING_DIM / 8).max(1);
    embedding[3 * chunk] = f32::INFINITY;
    let out = model.forward(&embedding).unwrap();
    assert_eq!(out.selected_experts[0], 3);
    assert_eq!(model.backend().steps, 1);
}

#[test]
fn spiking_forward_rejects_nan_before_backend_step() {
    let backend = MockBackend::new(8, 0.0);
    let mut model = spiking_router(backend);
    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    let chunk = (EMBEDDING_DIM / 8).max(1);
    embedding[3 * chunk] = f32::NAN;
    assert!(matches!(
        model.forward(&embedding).unwrap_err(),
        HybridError::NanRoutingScore { expert_id: 3 }
    ));
    // Fail-closed: no tick consumed.
    assert_eq!(model.backend().steps, 0);
}

#[test]
fn constructor_rejects_fewer_channels_than_experts() {
    let router = MoeRouter::load_with_mode("", 8, 2, crate::moe::RoutingMode::DenseSim).unwrap();
    assert!(matches!(
        SpikingMoeRouter::new(
            router,
            MockBackend::new(4, 0.0),
            RateEncoder::default(),
            SpikeCountDecoder
        ),
        Err(HybridError::SnnChannelMismatch {
            experts: 8,
            channels: 4
        })
    ));
}

#[test]
fn nan_stimulus_rejected_before_backend_step() {
    let router = MoeRouter::load_with_mode("", 8, 2, crate::moe::RoutingMode::DenseSim).unwrap();
    let mut model = SpikingMoeRouter::new(
        router,
        MockBackend::new(8, 0.0),
        RateEncoder { gain: f32::NAN },
        SpikeCountDecoder,
    )
    .unwrap();
    assert!(matches!(
        model.forward(&[1.0; EMBEDDING_DIM]).unwrap_err(),
        HybridError::SnnNan { .. }
    ));
    // Fail-closed: no tick consumed.
    assert_eq!(model.backend().steps, 0);
}

#[test]
fn forward_frozen_uses_frozen_path_and_reset_clears_state() {
    let backend = MockBackend::new(8, 0.0);
    let mut model = spiking_router(backend);
    let _ = model.forward(&[1.0; EMBEDDING_DIM]).unwrap();
    let _ = model.forward_frozen(&[1.0; EMBEDDING_DIM]).unwrap();
    assert_eq!(model.backend().steps, 1);
    assert_eq!(model.backend().frozen_steps, 1);
    assert!(model.capabilities().frozen_evaluation);
    model.reset();
    assert_eq!(model.backend().steps, 0);
    assert_eq!(model.backend().frozen_steps, 0);
}

#[test]
fn spiking_forward_rejects_wrong_embedding_length() {
    let backend = MockBackend::new(8, 0.0);
    let mut model = spiking_router(backend);
    assert!(matches!(
        model.forward(&[0.0; 8]).unwrap_err(),
        HybridError::InputLengthMismatch { .. }
    ));
}

#[cfg(feature = "neuromod")]
#[test]
fn neuromod_adapter_steps_and_reports_capabilities() {
    let mut net = NeuromodNetwork::new(4, 1, 8);
    assert_eq!(net.channels(), 8);
    let caps = net.capabilities();
    assert_eq!(caps.backend_name, "neuromod::SpikingNetwork");
    // Frozen eval delegates to `step` until a neuromod release ships step_frozen.
    assert!(!caps.frozen_evaluation && caps.plasticity && caps.caller_rng && caps.neuromodulation);

    let out = net.step(&[0.5; 8]).unwrap();
    assert!(out.spikes.iter().all(|&i| i < 4));
    let _ = net.step_frozen(&[0.5; 8]).unwrap();
    net.reset();
    assert_eq!(net.network().global_step, 0);

    // Length validation is failure-atomic via neuromod's preflight.
    assert!(net.step(&[0.5; 4]).is_err());
    assert_eq!(net.network().global_step, 0);
}
