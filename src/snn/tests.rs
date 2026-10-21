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

/// Issue #89: `SpikingMoeRouter` shares `MoeRouter::gate_scores`, so a
/// checkpoint gate written in GGUF `ne` order must route the same way through
/// the spiking path. Expert 5's row is all ones; input `e_0` selects expert 5.
#[test]
fn spiking_forward_selects_expert_by_gguf_ne_order_row() {
    use crate::moe::test_fixtures::{build_real_size_checkpoint, write_temp_file};

    let experts = 64usize;
    let mut gate = vec![0.0f32; EMBEDDING_DIM * experts];
    gate[5 * EMBEDDING_DIM..6 * EMBEDDING_DIM].fill(1.0);
    let gate_bytes: Vec<u8> = gate.iter().flat_map(|value| value.to_le_bytes()).collect();
    let path = write_temp_file(&build_real_size_checkpoint(gate_bytes), "spiking-row-order");

    let router = MoeRouter::load_with_mode(
        path.to_str().unwrap(),
        8,
        1,
        crate::moe::RoutingMode::DenseSim,
    )
    .unwrap();
    let mut model = SpikingMoeRouter::new(
        router,
        MockBackend::new(8, 0.5),
        RateEncoder::default(),
        SpikeCountDecoder,
    )
    .unwrap();
    let mut embedding = vec![0.0f32; EMBEDDING_DIM];
    embedding[0] = 1.0;
    let out = model.forward(&embedding).unwrap();
    assert_eq!(out.selected_experts, vec![5]);

    let _ = std::fs::remove_file(path);
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
    assert!(!logits.is_empty(), "need at least one gate logit");
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

/// A decoder readout of the wrong length is a structural violation, not a
/// NaN: it gets its own variant carrying the expected and actual lengths.
#[test]
fn forward_rejects_wrong_length_decoder_readout() {
    let embedding = embedding_with_gate_logits(&NEGATIVE_LOGITS);
    for got in [0usize, 7, 9] {
        let mut model = scripted_router(Vec::new(), FixedReadoutDecoder(vec![0.0; got]));
        assert!(matches!(
            model.forward(&embedding).unwrap_err(),
            CortexError::SnnReadoutLength {
                expected: 8,
                got: g
            } if g == got
        ));
    }
}

/// `spike_boost * activity` saturates instead of overflowing: a finite gate
/// stays finite and a `-Inf` gate stays `-Inf` rather than becoming NaN
/// after the backend tick has been consumed.
#[test]
fn spike_boost_saturates_instead_of_overflowing() {
    // Expert 0 gated at -Inf, expert 1 at 0.0; both spike with activity 1e30.
    let mut logits = [0.0f32; 8];
    logits[0] = f32::NEG_INFINITY;
    let embedding = embedding_with_gate_logits(&logits);
    let mut model = scripted_router(vec![0, 1], ScaledCountDecoder(1e30))
        .with_spike_boost(1e30)
        .unwrap();
    let out = model.forward(&embedding).unwrap();
    assert_eq!(out.selected_experts[0], 1);
    assert!(!out.selected_experts.contains(&0));
    assert!(out.expert_weights.iter().all(|w| !w.is_nan()));
    assert_eq!(out.expert_weights[0], 0.0);
    assert!(out.hidden.iter().all(|h| !h.is_nan()));
}

/// A gate already at `f32::MAX` plus a large bonus must saturate rather
/// than overflow to `+Inf`: the score stays finite and the expert wins.
/// Expert 1 sits at `f32::MAX` without spiking, so if the spiked expert 0
/// overflowed to `+Inf` the softmax would hand it all the mass; saturation
/// leaves both at `f32::MAX` and they share it.
#[test]
fn finite_gate_plus_large_bonus_saturates_to_finite_score() {
    let mut logits = [0.0f32; 8];
    logits[0] = f32::MAX;
    logits[1] = f32::MAX;
    let embedding = embedding_with_gate_logits(&logits);
    // Expert 0 spikes with activity 1e30 under a 1e30 boost.
    let mut model = scripted_router(vec![0], ScaledCountDecoder(1e30))
        .with_spike_boost(1e30)
        .unwrap();
    let scores = model.router().gate_scores(&embedding).unwrap();
    assert!(scores[0].is_finite(), "gate must be finite: {}", scores[0]);
    let out = model.forward(&embedding).unwrap();
    assert_eq!(out.selected_experts, vec![0, 1]);
    assert!(out.expert_weights.iter().all(|w| w.is_finite()));
    assert_eq!(
        out.expert_weights[0], out.expert_weights[1],
        "both scores saturate at f32::MAX, so neither may absorb the mass"
    );
    assert!(out.expert_weights[0] > 0.0);
    assert!(out.hidden.iter().all(|h| h.is_finite()));
}

/// Spikes only on unselected experts must not zero the hidden state: the
/// gain falls back to the dense selected mass exactly as on a silent tick.
#[test]
fn hidden_keeps_dense_mass_when_only_unselected_experts_spike() {
    let embedding = embedding_with_gate_logits(&NEGATIVE_LOGITS);
    // Expert 7 spikes but -2.75 + 1.0 still loses to -1.0 / -1.25.
    let mut model = scripted_router(vec![7], SpikeCountDecoder);
    let out = model.forward(&embedding).unwrap();
    assert_eq!(out.selected_experts, vec![0, 1]);
    let dense_mass: f32 = out
        .selected_experts
        .iter()
        .map(|&e| out.expert_weights[e])
        .sum();
    assert!(dense_mass > 0.0);
    assert!(
        out.hidden
            .iter()
            .zip(&embedding)
            .all(|(h, e)| *h == e * dense_mass)
    );
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
    // One LIF neuron per channel; the Izhikevich bank is auxiliary and does
    // not count toward the output width (#91).
    let mut net = NeuromodNetwork::new(8, 1, 8).unwrap();
    assert_eq!(net.channels(), 8);
    let caps = net.capabilities();
    assert_eq!(caps.backend_name, "neuromod::SpikingNetwork");
    // neuromod 0.7 ships `step_frozen`; the thread-local RNG path is the
    // default, so replay is only advertised once a seed is installed (#92).
    assert!(caps.frozen_evaluation && caps.plasticity && caps.neuromodulation);
    assert!(!caps.caller_rng);

    let out = net.step(&[0.5; 8]).unwrap();
    assert!(out.spikes.iter().all(|&i| i < 8));
    let _ = net.step_frozen(&[0.5; 8]).unwrap();
    net.reset();
    assert_eq!(net.network().global_step, 0);

    // Length validation is failure-atomic via neuromod's preflight.
    assert!(net.step(&[0.5; 4]).is_err());
    assert_eq!(net.network().global_step, 0);

    // LIF count must equal channel count for the 1:1 output contract.
    assert!(matches!(
        NeuromodNetwork::new(4, 1, 8),
        Err(CortexError::InvalidConfig(_))
    ));
}

/// Spike counts per channel after driving `net` with `stimulus` for `steps`
/// learning-enabled ticks.
#[cfg(feature = "neuromod")]
fn neuromod_spike_counts(net: &mut NeuromodNetwork, stimulus: &[f32], steps: usize) -> Vec<usize> {
    let mut counts = vec![0usize; net.channels()];
    for _ in 0..steps {
        for idx in net.step(stimulus).unwrap().spikes {
            counts[idx] += 1;
        }
    }
    counts
}

#[cfg(feature = "neuromod")]
#[test]
fn neuromod_adapter_rejects_izhikevich_neurons_as_output_channels() {
    // neuromod's `step` returns LIF indices only, so an Izhikevich neuron
    // counted toward `num_channels` is a channel that can never spike (#91).
    for (lif, izh) in [(7, 1), (0, 8), (4, 4)] {
        let err = NeuromodNetwork::new(lif, izh, 8).err().expect("rejected");
        assert!(
            matches!(err, CortexError::InvalidConfig(_)),
            "{lif}/{izh}: {err}"
        );
        assert!(err.to_string().contains("Izhikevich"), "{err}");
    }
    // Izhikevich neurons are allowed as an auxiliary bank on top of the
    // one-LIF-per-channel requirement.
    assert!(NeuromodNetwork::new(8, 3, 8).is_ok());
}

#[cfg(feature = "neuromod")]
#[test]
fn neuromod_adapter_new_wires_every_channel_to_a_spiking_neuron() {
    // `SpikingNetwork::with_dimensions` zero-initialises LIF weights and the
    // L1 renormalization keeps a blank neuron blank, so a backend built that
    // way never spikes (#91). `new` must seed weights so every channel can.
    let mut net = NeuromodNetwork::new(8, 0, 8).unwrap();
    let counts = neuromod_spike_counts(&mut net, &[1.0; 8], 200);
    assert!(
        counts.iter().all(|&c| c > 0),
        "every channel must spike under strong stimulus: {counts:?}"
    );

    // Identity wiring: driving channel `i` alone spikes neuron `i` alone, so
    // `SpikingMoeRouter`'s expert-to-channel mapping holds.
    for ch in 0..8 {
        let mut net = NeuromodNetwork::new(8, 0, 8).unwrap();
        let mut stimulus = [0.0; 8];
        stimulus[ch] = 1.0;
        let counts = neuromod_spike_counts(&mut net, &stimulus, 50);
        assert!(counts[ch] > 0, "channel {ch} silent: {counts:?}");
        assert!(
            counts.iter().enumerate().all(|(i, &c)| i == ch || c == 0),
            "channel {ch} leaked: {counts:?}"
        );
    }
}

#[cfg(feature = "neuromod")]
#[test]
fn neuromod_adapter_rejects_blank_lif_neurons() {
    // A restored or hand-built network with an all-zero LIF neuron carries a
    // channel that provably cannot spike; fail at construction, not after
    // thousands of silent ticks (#91).
    let blank = neuromod::SpikingNetwork::with_dimensions(8, 0, 8);
    let err = NeuromodNetwork::from_network(blank)
        .err()
        .expect("rejected");
    assert!(matches!(err, CortexError::InvalidConfig(_)), "{err}");
    assert!(err.to_string().contains("network_mut"), "{err}");

    let mut one_blank = neuromod::SpikingNetwork::with_dimensions(8, 0, 8);
    for (i, neuron) in one_blank.neurons.iter_mut().enumerate() {
        if i != 3 {
            neuron.weights[i] = 1.0;
        }
    }
    let err = NeuromodNetwork::from_network(one_blank)
        .err()
        .expect("rejected");
    assert!(err.to_string().contains("neuron 3"), "{err}");

    // Only `weights[..num_channels]` is ever integrated, so a neuron whose
    // sole non-zero weight sits past the channel range is just as silent.
    let mut out_of_range = neuromod::SpikingNetwork::with_dimensions(8, 0, 8);
    for (i, neuron) in out_of_range.neurons.iter_mut().enumerate() {
        if i == 3 {
            neuron.weights.push(1.0);
        } else {
            neuron.weights[i] = 1.0;
        }
    }
    let err = NeuromodNetwork::from_network(out_of_range)
        .err()
        .expect("rejected");
    assert!(err.to_string().contains("neuron 3"), "{err}");

    // neuromod integrates `|stimulus|`, so a negative-only neuron receives
    // non-positive current and never reaches threshold either.
    let mut negative = neuromod::SpikingNetwork::with_dimensions(8, 0, 8);
    for (i, neuron) in negative.neurons.iter_mut().enumerate() {
        neuron.weights[i] = if i == 3 { -1.0 } else { 1.0 };
    }
    let err = NeuromodNetwork::from_network(negative)
        .err()
        .expect("rejected");
    assert!(err.to_string().contains("neuron 3"), "{err}");

    // A non-finite in-range weight poisons the membrane; a short weight vector
    // leaves channels the engine silently skips. Both are rejected by name.
    let mut nan = neuromod::SpikingNetwork::with_dimensions(8, 0, 8);
    for (i, neuron) in nan.neurons.iter_mut().enumerate() {
        neuron.weights[i] = 1.0;
    }
    nan.neurons[3].weights[5] = f32::NAN;
    let err = NeuromodNetwork::from_network(nan).err().expect("rejected");
    assert!(err.to_string().contains("neuron 3"), "{err}");

    // The engine skips channels past a short weight vector, so a sparse
    // checkpoint is accepted when one present weight can drive the neuron...
    let mut short = neuromod::SpikingNetwork::with_dimensions(8, 0, 8);
    for (i, neuron) in short.neurons.iter_mut().enumerate() {
        neuron.weights[i] = 1.0;
    }
    short.neurons[3].weights.truncate(4);
    let mut short = NeuromodNetwork::from_network(short).expect("sparse but drivable");
    let mut one_hot = [0.0; 8];
    one_hot[3] = 1.0;
    assert!(neuromod_spike_counts(&mut short, &one_hot, 50)[3] > 0);

    // ...and rejected when none of the present weights is positive.
    let mut short_blank = neuromod::SpikingNetwork::with_dimensions(8, 0, 8);
    for (i, neuron) in short_blank.neurons.iter_mut().enumerate() {
        neuron.weights[i] = 1.0;
    }
    short_blank.neurons[3].weights = vec![0.0; 2];
    let err = NeuromodNetwork::from_network(short_blank)
        .err()
        .expect("rejected");
    assert!(err.to_string().contains("neuron 3"), "{err}");
    let mut empty = neuromod::SpikingNetwork::with_dimensions(8, 0, 8);
    for (i, neuron) in empty.neurons.iter_mut().enumerate() {
        neuron.weights[i] = 1.0;
    }
    empty.neurons[3].weights.clear();
    let err = NeuromodNetwork::from_network(empty)
        .err()
        .expect("rejected");
    assert!(err.to_string().contains("neuron 3"), "{err}");

    // A fully wired network round-trips through `from_network`.
    let wired = NeuromodNetwork::new(8, 0, 8).unwrap();
    let json = serde_json::to_string(wired.network()).unwrap();
    let restored: neuromod::SpikingNetwork = serde_json::from_str(&json).unwrap();
    assert!(NeuromodNetwork::from_network(restored).is_ok());
}

/// Learning state of every LIF neuron (weights, eligibility traces, threshold
/// and decay parameters) plus the stored modulators and R-STDP config — what
/// `step_frozen` must leave bit-for-bit intact. Runtime state (membranes,
/// spike timestamps, `global_step`) is allowed to advance and is excluded.
#[cfg(feature = "neuromod")]
fn neuromod_learning_state(net: &NeuromodNetwork) -> String {
    let network = net.network();
    let lif: Vec<_> = network
        .neurons
        .iter()
        .map(|n| {
            (
                n.weights.clone(),
                n.eligibility.clone(),
                n.threshold.to_bits(),
                n.base_threshold.to_bits(),
                n.decay_rate.to_bits(),
            )
        })
        .collect();
    serde_json::to_string(&(lif, network.modulators, network.stdp_config)).unwrap()
}

#[cfg(feature = "neuromod")]
#[test]
fn neuromod_adapter_step_frozen_preserves_learning_state() {
    // Before neuromod 0.7 the adapter delegated `step_frozen` to `step`, so a
    // held-out `forward_frozen` kept advancing eligibility traces (#92).
    let mut net = NeuromodNetwork::new(8, 0, 8).unwrap().with_seed(7);
    // Warm up so traces are non-trivial, then snapshot.
    for _ in 0..5 {
        net.step(&[1.0; 8]).unwrap();
    }
    let before = neuromod_learning_state(&net);
    let step_before = net.network().global_step;

    let out = net.step_frozen(&[1.0; 8]).unwrap();
    assert!(!out.spikes.is_empty(), "frozen path still produces spikes");
    assert_eq!(
        net.network().global_step,
        step_before + 1,
        "runtime state advances"
    );
    assert_eq!(
        neuromod_learning_state(&net),
        before,
        "frozen step mutated learning state"
    );

    // Control: a learning-enabled step with the same input does move traces.
    net.step(&[1.0; 8]).unwrap();
    assert_ne!(neuromod_learning_state(&net), before);
}

/// `SpikingNetwork` is not `Clone`; its serde checkpoint is the supported copy.
#[cfg(feature = "neuromod")]
fn neuromod_clone(net: &NeuromodNetwork) -> NeuromodNetwork {
    let json = serde_json::to_string(net.network()).unwrap();
    let mut copy = NeuromodNetwork::from_network(serde_json::from_str(&json).unwrap()).unwrap();
    copy.modulators = net.modulators;
    copy.set_seed(net.seed());
    copy
}

#[cfg(feature = "neuromod")]
fn neuromod_weights(net: &NeuromodNetwork) -> Vec<Vec<f32>> {
    net.network()
        .neurons
        .iter()
        .map(|n| n.weights.clone())
        .collect()
}

#[cfg(feature = "neuromod")]
#[test]
fn neuromod_adapter_seeded_rng_replays_bit_for_bit() {
    // Bernoulli input encoding draws from the thread-local RNG unless the
    // adapter routes a caller seed through `step_with_rng` (#92).
    fn run(net: &mut NeuromodNetwork, frozen_only: bool) -> (Vec<Vec<usize>>, Vec<i64>, String) {
        let mut spikes = Vec::new();
        for i in 0..40 {
            let stimulus = if i % 2 == 0 { [0.5; 8] } else { [0.3; 8] };
            spikes.push(if frozen_only || i % 5 == 0 {
                net.step_frozen(&stimulus).unwrap().spikes
            } else {
                net.step(&stimulus).unwrap().spikes
            });
        }
        (
            spikes,
            net.network().input_spike_times.clone(),
            serde_json::to_string(net.network()).unwrap(),
        )
    }

    let mut a = NeuromodNetwork::new(8, 0, 8)
        .unwrap()
        .with_seed(0xC0FF_EE01);
    let mut b = NeuromodNetwork::new(8, 0, 8)
        .unwrap()
        .with_seed(0xC0FF_EE01);
    assert!(a.capabilities().caller_rng);
    assert_eq!(a.seed(), Some(0xC0FF_EE01));
    let first = run(&mut a, false);
    assert_eq!(
        first,
        run(&mut b, false),
        "same seed, same inputs, different trajectory"
    );

    // Dropping the seed falls back to the thread-local RNG and says so.
    a.set_seed(None);
    assert!(!a.capabilities().caller_rng);
    assert_eq!(a.seed(), None);
    a.step(&[0.5; 8]).unwrap();

    // The seed can also be installed on a wrapped checkpoint after the fact.
    let mut c = NeuromodNetwork::new(8, 0, 8).unwrap();
    assert!(!c.capabilities().caller_rng);
    c.set_seed(Some(0xC0FF_EE01));
    assert!(c.capabilities().caller_rng);
    assert_eq!(run(&mut c, false), first);

    // `reset` rewinds the RNG to the seed. neuromod's `reset` keeps learned
    // weights, so a post-reset epoch replays the previous one only when the
    // learned state at both resets is the same: a frozen-only epoch...
    let mut frozen = NeuromodNetwork::new(8, 0, 8)
        .unwrap()
        .with_seed(0xC0FF_EE01);
    let frozen_first = run(&mut frozen, true);
    frozen.reset();
    assert_eq!(
        run(&mut frozen, true),
        frozen_first,
        "frozen-only epoch did not replay after reset"
    );

    // ...or a clone taken at the previous reset, even with learning on and
    // dopamine converting traces into weight changes.
    let mut learner = NeuromodNetwork::new(8, 0, 8)
        .unwrap()
        .with_seed(0xC0FF_EE01);
    learner.modulators.dopamine = 0.9;
    for (i, neuron) in learner.network_mut().neurons.iter_mut().enumerate() {
        // Off-diagonal room for STDP to move weight within the L1 budget.
        neuron.weights[i] = 1.0;
        neuron.weights[(i + 1) % 8] = 1.0;
    }
    let weights_at_reset = neuromod_weights(&learner);
    let mut twin = neuromod_clone(&learner);
    let learned = run(&mut learner, false);
    assert_ne!(
        neuromod_weights(&learner),
        weights_at_reset,
        "learning epoch must change weights for this case to mean anything"
    );
    twin.reset();
    assert_eq!(
        run(&mut twin, false),
        learned,
        "clone-at-reset replay diverged"
    );

    // The documented limit: once weights have moved, reset alone does not
    // rewind them, so the same stream from the learned state diverges.
    learner.reset();
    assert_eq!(neuromod_weights(&learner), neuromod_weights(&twin));
    assert_ne!(
        run(&mut learner, false),
        learned,
        "reset is not expected to rewind learned weights"
    );
}

#[cfg(feature = "neuromod")]
#[test]
fn neuromod_adapter_drives_spiking_router_on_every_expert() {
    // End to end: a backend that never spikes leaves `SpikingMoeRouter` with
    // no spike signal at all, so routing degrades to the bare ANN gate (#91).
    // Since #93 the hidden gain is bounded and a silent tick falls back to the
    // dense selected mass, so `hidden` can no longer tell "every channel
    // fired" from "nothing fired". Record the spikes the router's own
    // `forward` path draws from the backend instead.
    struct Recording {
        inner: NeuromodNetwork,
        counts: Vec<usize>,
    }
    impl SnnBackend for Recording {
        fn channels(&self) -> usize {
            self.inner.channels()
        }
        fn step(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput> {
            let out = self.inner.step(stimulus)?;
            for &idx in &out.spikes {
                self.counts[idx] += 1;
            }
            Ok(out)
        }
        fn step_frozen(&mut self, stimulus: &[f32]) -> Result<SnnStepOutput> {
            self.inner.step_frozen(stimulus)
        }
        fn reset(&mut self) {
            self.inner.reset();
        }
        fn capabilities(&self) -> SnnCapabilities {
            self.inner.capabilities()
        }
    }

    let router = MoeRouter::load_with_mode("", 8, 8, crate::moe::RoutingMode::DenseSim).unwrap();
    let backend = Recording {
        inner: NeuromodNetwork::new(8, 0, 8).unwrap(),
        counts: vec![0; 8],
    };
    let mut model =
        SpikingMoeRouter::new(router, backend, RateEncoder::default(), SpikeCountDecoder).unwrap();
    // A uniform gate with `top_k == num_experts` stimulates all eight expert
    // channels, so each one must fire at least once through `forward`.
    for _ in 0..200 {
        let out = model.forward(&[1.0; EMBEDDING_DIM]).unwrap();
        assert!(out.hidden.iter().all(|v| v.is_finite()));
    }
    let counts = &model.backend().counts;
    assert!(
        counts.iter().all(|&c| c > 0),
        "not every expert channel spiked: {counts:?}"
    );
}
