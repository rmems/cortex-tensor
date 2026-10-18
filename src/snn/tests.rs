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
    let mut enc = RateEncoder { gain: 2.0 };
    assert_eq!(
        enc.encode(&[0.5, -1.0], 5).unwrap(),
        vec![1.0, -2.0, 1.0, -2.0, 1.0]
    );
    assert_eq!(enc.encode(&[], 3).unwrap(), vec![0.0; 3]);
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

/// Backend that replays a fixed spike list every tick, so tests can script
/// exactly which channels fire (including out-of-range and duplicate indices
/// a misbehaving backend might emit). Not neuron dynamics — the test boundary.
struct ScriptedBackend {
    channels: usize,
    spikes: Vec<usize>,
}

impl SnnBackend for ScriptedBackend {
    fn channels(&self) -> usize {
        self.channels
    }

    fn step(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput> {
        assert_eq!(stimulus.len(), self.channels);
        Ok(SnnStepOutput {
            spikes: self.spikes.clone(),
        })
    }

    fn step_frozen(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput> {
        self.step(stimulus)
    }

    fn reset(&mut self) {}

    fn capabilities(&self) -> SnnCapabilities {
        SnnCapabilities {
            backend_name: "scripted",
            frozen_evaluation: false,
            plasticity: false,
            caller_rng: false,
            neuromodulation: false,
        }
    }
}

/// Decoder that multiplies each spike count by a large factor, standing in
/// for multi-count readouts (burst counts, rate estimates) whose magnitude
/// is unbounded from the router's point of view.
struct ScaledCountDecoder(f32);

impl SnnDecoder for ScaledCountDecoder {
    fn decode(&self, output: &SnnStepOutput, channels: usize) -> Vec<f32> {
        SpikeCountDecoder
            .decode(output, channels)
            .into_iter()
            .map(|count| count * self.0)
            .collect()
    }
}

/// Decoder that emits a fixed readout regardless of spikes, used to feed the
/// router readouts (negative, non-finite) a well-behaved decoder never would.
struct FixedReadoutDecoder(Vec<f32>);

impl SnnDecoder for FixedReadoutDecoder {
    fn decode(&self, _output: &SnnStepOutput, _channels: usize) -> Vec<f32> {
        self.0.clone()
    }
}

/// Embedding whose stub (`DenseSim`) gate scores equal `logits` exactly:
/// expert *e*'s gate is the sum of embedding chunk *e*, so one non-zero
/// element per chunk pins the logit.
fn embedding_with_gate_logits(logits: &[f32]) -> Vec<f32> {
    let chunk = (EMBEDDING_DIM / logits.len()).max(1);
    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    for (expert, &logit) in logits.iter().enumerate() {
        embedding[expert * chunk] = logit;
    }
    embedding
}

/// Issue #93 repro: eight experts with strictly negative gate logits.
const NEGATIVE_LOGITS: [f32; 8] = [-1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5, -2.75];

fn scripted_router<D: SnnDecoder>(
    spikes: Vec<usize>,
    decoder: D,
) -> SpikingMoeRouter<ScriptedBackend, RateEncoder, D> {
    let router = MoeRouter::load_with_mode("", 8, 2, crate::moe::RoutingMode::DenseSim).unwrap();
    SpikingMoeRouter::new(
        router,
        ScriptedBackend {
            channels: 8,
            spikes,
        },
        RateEncoder::default(),
        decoder,
    )
    .unwrap()
}

/// Issue #93 (1): a spike must never demote its expert. With the old
/// `gate * (1 + spikes)` boost, spiking channel 0 on negative logits evicted
/// expert 0 (`[0, 1]` → `[1, 2]`).
#[test]
fn spike_boost_promotes_spiked_expert_with_negative_gate_logits() {
    let embedding = embedding_with_gate_logits(&NEGATIVE_LOGITS);

    // Silent backend: ranking is the dense ranking.
    let mut silent = scripted_router(Vec::new(), SpikeCountDecoder);
    assert_eq!(
        silent.forward(&embedding).unwrap().selected_experts,
        vec![0, 1]
    );

    // Spiking the already-top expert keeps it on top.
    let mut spike_top = scripted_router(vec![0], SpikeCountDecoder);
    assert_eq!(
        spike_top.forward(&embedding).unwrap().selected_experts,
        vec![0, 1]
    );

    // Spiking a lower-ranked expert promotes it: -1.5 + 1.0 beats -1.0.
    let mut spike_mid = scripted_router(vec![2], SpikeCountDecoder);
    assert_eq!(
        spike_mid.forward(&embedding).unwrap().selected_experts,
        vec![2, 0]
    );
}

/// Issue #93 (1): the boost is monotone in the spike count for either logit
/// sign — more spikes never lower an expert's rank.
#[test]
fn spike_boost_is_monotone_in_spike_count_regardless_of_logit_sign() {
    // Same spread shifted into positive territory (2.75 … 1.0) keeps expert 7 weakest.
    for logits in [NEGATIVE_LOGITS, NEGATIVE_LOGITS.map(|v| v + 3.75)] {
        let embedding = embedding_with_gate_logits(&logits);
        let mut previous_rank = usize::MAX;
        // Expert 7 holds the lowest logit of each set; raise its readout from
        // 0 to 4 "spikes" via a scaled decoder and watch its rank only improve.
        for count in 0..5 {
            let mut model = scripted_router(vec![7], ScaledCountDecoder(count as f32));
            let out = model.forward(&embedding).unwrap();
            let rank = out
                .selected_experts
                .iter()
                .position(|&e| e == 7)
                .unwrap_or(out.selected_experts.len());
            assert!(
                rank <= previous_rank,
                "count {count}: rank {rank} > {previous_rank}"
            );
            previous_rank = rank;
        }
        // Four spikes lift the weakest expert (-2.75 or 1.0) to the top.
        assert_eq!(previous_rank, 0, "logits {logits:?}");
    }
}

/// Issue #93 (1): `with_spike_boost` scales the additive boost and rejects
/// values that would break monotonicity.
#[test]
fn spike_boost_gain_is_configurable_and_validated() {
    let embedding = embedding_with_gate_logits(&NEGATIVE_LOGITS);
    let router = MoeRouter::load_with_mode("", 8, 2, crate::moe::RoutingMode::DenseSim).unwrap();
    // beta = 0.1: -1.5 + 0.1 does not beat -1.25, so spiking expert 2 changes nothing.
    let mut weak = SpikingMoeRouter::new(
        router,
        ScriptedBackend {
            channels: 8,
            spikes: vec![2],
        },
        RateEncoder::default(),
        SpikeCountDecoder,
    )
    .unwrap()
    .with_spike_boost(0.1)
    .unwrap();
    assert_eq!(weak.spike_boost(), 0.1);
    assert_eq!(
        weak.forward(&embedding).unwrap().selected_experts,
        vec![0, 1]
    );

    for bad in [-1.0, f32::NAN, f32::INFINITY] {
        let model = scripted_router(Vec::new(), SpikeCountDecoder);
        assert!(matches!(
            model.with_spike_boost(bad),
            Err(CortexError::InvalidConfig(_))
        ));
    }
}

/// Issue #93 (2): `hidden` is scaled by selected softmax mass weighted by
/// normalised spike activity, so its gain stays in `[0, 1]` even when the
/// decoder reports huge counts, and a silent tick falls back to the dense
/// selected mass instead of zeroing the hidden state.
#[test]
fn hidden_scale_is_bounded_and_silent_tick_keeps_dense_mass() {
    let embedding = embedding_with_gate_logits(&NEGATIVE_LOGITS);

    // Silent backend: same scale as `MoeRouter::forward` (sum of selected softmax weights).
    let mut silent = scripted_router(Vec::new(), SpikeCountDecoder);
    let out = silent.forward(&embedding).unwrap();
    let dense_mass: f32 = out
        .selected_experts
        .iter()
        .map(|&e| out.expert_weights[e])
        .sum();
    assert!(dense_mass > 0.0 && dense_mass <= 1.0);
    assert_eq!(out.hidden[0], embedding[0] * dense_mass);
    assert!(
        out.hidden
            .iter()
            .zip(&embedding)
            .all(|(h, e)| *h == e * dense_mass)
    );

    // Huge multi-count readout on both selected experts: gain is still ≤ 1.
    let mut loud = scripted_router(vec![0, 1], ScaledCountDecoder(1e6));
    let out = loud.forward(&embedding).unwrap();
    assert_eq!(out.selected_experts, vec![0, 1]);
    let mass: f32 = out
        .selected_experts
        .iter()
        .map(|&e| out.expert_weights[e])
        .sum();
    assert_eq!(out.hidden[0], embedding[0] * mass);
    assert!(out.hidden.iter().all(|h| h.is_finite() && h.abs() <= 2.75));

    // One selected expert spiking, the other silent: only the spiking expert's
    // softmax weight contributes.
    let mut partial = scripted_router(vec![0], SpikeCountDecoder);
    let out = partial.forward(&embedding).unwrap();
    assert_eq!(out.selected_experts, vec![0, 1]);
    assert_eq!(out.hidden[0], embedding[0] * out.expert_weights[0]);
}

/// Issue #93 (3): backend spike indices outside `0..channels` are rejected
/// before decoding instead of being silently dropped by the decoder.
#[test]
fn forward_rejects_backend_spike_outside_channel_space() {
    let embedding = embedding_with_gate_logits(&NEGATIVE_LOGITS);
    let mut model = scripted_router(vec![0, 9], SpikeCountDecoder);
    assert!(matches!(
        model.forward(&embedding).unwrap_err(),
        CortexError::SnnSpikeOutOfRange {
            channel: 9,
            channels: 8
        }
    ));
}

/// Issue #93 (3): a channel reported twice in one tick violates the
/// `SnnStepOutput` contract and is rejected instead of double-counted.
#[test]
fn forward_rejects_duplicate_backend_spikes() {
    let embedding = embedding_with_gate_logits(&NEGATIVE_LOGITS);
    let mut model = scripted_router(vec![3, 1, 3], SpikeCountDecoder);
    assert!(matches!(
        model.forward_frozen(&embedding).unwrap_err(),
        CortexError::SnnDuplicateSpike { channel: 3 }
    ));
}

/// Issue #93 (3): decoder readouts must be finite and non-negative — the
/// boost and the hidden-state normalisation both assume activity ≥ 0.
#[test]
fn forward_rejects_negative_or_non_finite_decoder_readout() {
    let embedding = embedding_with_gate_logits(&NEGATIVE_LOGITS);
    for (bad, channel) in [(-1.0, 2usize), (f32::INFINITY, 5), (f32::NEG_INFINITY, 0)] {
        let mut readout = vec![0.0f32; 8];
        readout[channel] = bad;
        let mut model = scripted_router(Vec::new(), FixedReadoutDecoder(readout));
        match model.forward(&embedding).unwrap_err() {
            CortexError::SnnInvalidReadout {
                channel: got,
                value,
            } => {
                assert_eq!(got, channel);
                assert_eq!(value, bad);
            }
            other => panic!("expected SnnInvalidReadout, got {other:?}"),
        }
    }
    // NaN keeps its dedicated variant.
    let mut nan = scripted_router(Vec::new(), FixedReadoutDecoder(vec![f32::NAN; 8]));
    assert!(matches!(
        nan.forward(&embedding).unwrap_err(),
        CortexError::SnnNan {
            stage: "decoder readout"
        }
    ));
}

#[cfg(feature = "axon-encoder")]
#[test]
fn axon_encoder_adapter_counts_spikes_per_channel() {
    let enc = axon_encoder::prelude::RateEncoder::try_new(0.0, 100.0, (0.0, 1.0), 0.1).unwrap();
    let mut adapter = AxonEncoder::new(enc);
    // 100 Hz * 0.1 s = 10 spikes on channel 0 for input 1.0; input 0.0 stays silent.
    let stimulus = adapter.encode(&[1.0, 0.0], 8).unwrap();
    assert_eq!(stimulus[0], 10.0);
    assert!(stimulus[1..].iter().all(|&v| v == 0.0));
}

#[cfg(feature = "axon-encoder")]
#[test]
fn axon_encoder_rejects_spikes_outside_channel_space() {
    let enc = axon_encoder::prelude::RateEncoder::try_new(0.0, 100.0, (0.0, 1.0), 0.1).unwrap();
    let mut adapter = AxonEncoder::new(enc);
    // 8 input channels of spikes cannot be represented in a 4-channel stimulus.
    assert!(matches!(
        adapter.encode(&[1.0; 8], 4),
        Err(HybridError::SnnSpikeOutOfRange {
            channel: 4,
            channels: 4
        })
    ));
}

#[cfg(feature = "axon-encoder")]
#[test]
fn axon_encoder_streams_phase_across_calls() {
    // 5 Hz * 0.1 s = 0.5 expected spikes per step: the first call accumulates
    // phase and emits nothing; the second emits the deferred spike — encoder
    // state advances per call (documented streaming contract, not atomic).
    let enc = axon_encoder::prelude::RateEncoder::try_new(5.0, 5.0, (0.0, 1.0), 0.1).unwrap();
    let mut adapter = AxonEncoder::new(enc);
    assert_eq!(adapter.encode(&[1.0], 1).unwrap()[0], 0.0);
    assert_eq!(adapter.encode(&[1.0], 1).unwrap()[0], 1.0);
    SnnEncoder::reset(&mut adapter);
    // Clean epoch: accumulated phase is gone, next call accumulates again.
    assert_eq!(adapter.encode(&[1.0], 1).unwrap()[0], 0.0);
}

#[cfg(feature = "axon-encoder")]
#[test]
fn axon_encoder_adapter_drives_spiking_router() {
    let enc = axon_encoder::prelude::RateEncoder::try_new(0.0, 100.0, (0.0, 1.0), 0.1).unwrap();
    let router = MoeRouter::load_with_mode("", 8, 2, crate::moe::RoutingMode::DenseSim).unwrap();
    let mut model = SpikingMoeRouter::new(
        router,
        MockBackend::new(8, 0.0),
        AxonEncoder::new(enc),
        SpikeCountDecoder,
    )
    .unwrap();
    model.forward(&[1.0; EMBEDDING_DIM]).unwrap();
    assert_eq!(model.backend().steps, 1);
}

#[cfg(feature = "neuromod")]
#[test]
fn neuromod_adapter_steps_and_reports_capabilities() {
    let mut net = NeuromodNetwork::new(7, 1, 8).unwrap();
    assert_eq!(net.channels(), 8);
    let caps = net.capabilities();
    assert_eq!(caps.backend_name, "neuromod::SpikingNetwork");
    // Frozen eval delegates to `step` until a neuromod release ships step_frozen.
    assert!(!caps.frozen_evaluation && caps.plasticity && caps.caller_rng && caps.neuromodulation);

    let out = net.step(&[0.5; 8]).unwrap();
    assert!(out.spikes.iter().all(|&i| i < 8));
    let _ = net.step_frozen(&[0.5; 8]).unwrap();
    net.reset();
    assert_eq!(net.network().global_step, 0);

    // Length validation is failure-atomic via neuromod's preflight.
    assert!(net.step(&[0.5; 4]).is_err());
    assert_eq!(net.network().global_step, 0);

    // Neuron count must equal channel count for the 1:1 output contract.
    assert!(matches!(
        NeuromodNetwork::new(4, 1, 8),
        Err(CortexError::InvalidConfig(_))
    ));
}
