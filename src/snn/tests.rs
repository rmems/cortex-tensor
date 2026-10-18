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
    // End to end: the router's `hidden` is `embedding * spiking_mass`, so a
    // backend that never spikes degrades to DenseSim with `hidden == 0` (#91).
    let router = MoeRouter::load_with_mode("", 8, 8, crate::moe::RoutingMode::DenseSim).unwrap();
    let backend = NeuromodNetwork::new(8, 0, 8).unwrap();
    let mut model =
        SpikingMoeRouter::new(router, backend, RateEncoder::default(), SpikeCountDecoder).unwrap();
    // With `top_k == num_experts`, `spiking_mass` is the number of expert
    // channels that fired this tick; a uniform gate drives all eight, so the
    // hidden state must reach `embedding * 8.0` on some tick.
    let mut max_mass = 0.0f32;
    for _ in 0..200 {
        let out = model.forward(&[1.0; EMBEDDING_DIM]).unwrap();
        max_mass = max_mass.max(out.hidden[0]);
    }
    assert_eq!(max_mass, 8.0, "not every expert channel spiked");
}
