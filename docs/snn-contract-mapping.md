# SNN step-contract mapping: cortex-tensor `src/snn` ↔ hybrid-fusion

- **Status:** Accepted (2026-10-26)
- **Tracking:** Linear
  [RM-2360](https://linear.app/rpd-34/issue/RM-2360/snn-keep-cortex-tensors-snn-step-contract-in-sync-with-hybrid-fusions).
  Decision context: [RM-2205](https://linear.app/rpd-34/issue/RM-2205)
  (hybrid-fusion became an owner), [RM-2206](https://linear.app/rpd-34/issue/RM-2206)
  (cortex-tensor **keeps** its SNN code — that crate is meant to handle both
  ANN and SNN; the removal plan, PRs #137 and #141, was closed unmerged).
- **Twin document:** hybrid-fusion
  [`docs/adr/0002-snn-step-contract-parity.md`](https://github.com/rmems/hybrid-fusion/blob/main/docs/adr/0002-snn-step-contract-parity.md)
  ("ADR 0002"). The two documents change together: edit both in the same PR
  series, or record a new intentional difference in both.

## Direction

Each crate keeps its own traits — cortex-tensor does not implement or depend
on hybrid-fusion's contract, and hybrid-fusion does not depend on
cortex-tensor's. The dependency direction forbids it (an orchestrator may
consume a backend, never the reverse), and the two contracts serve different
callers:

- **cortex-tensor `src/snn`** is the ANN+SNN reference path inside one crate:
  `SpikingMoeRouter` composes the concrete `MoeRouter` and returns a
  `MoeOutput` (a scaled hidden state sized by `EMBEDDING_DIM`).
- **hybrid-fusion's contract** is the orchestrator's pluggable boundary:
  generic over any `ExpertRouter` + `SpikingNetwork`, with capability
  negotiation, plan-port metadata, telemetry, and its own error vocabulary.

Drift between the two is prevented mechanically, not by prose: the mapping
below is the record, and hybrid-fusion's
`smoke/cortex-neuromod/tests/snn_contract_parity.rs` (CI: the smoke package
runs in both of hybrid-fusion's CI jobs) drives the **same seeded inputs**
through both trait sets. Agreement is asserted exactly where the contracts
must agree, and every intentional difference is asserted, so a difference can
only enter through an edited test that names it.

## Mapping

| cortex-tensor | hybrid-fusion | Mapping |
|---|---|---|
| `SnnBackend::channels()` | `SpikingNetwork::num_channels()` | stimulus width |
| — (implied: one neuron per channel; `NeuromodNetwork` enforces one LIF per channel) | `SpikingNetwork::num_neurons()` | output population; a **separate axis** in hybrid-fusion |
| `SnnBackend::step(&[f32])` | `SpikingNetwork::step(&[f32], &NeuroModulators)` | modulators are adapter state here (`NeuromodNetwork.modulators`, `neuromod::NeuroModulators::default()`), an explicit per-call argument there (mapped per-field: `aux_dopamine` folded into `dopamine` clamped to `[0, 1]`, `cortisol → norepinephrine`, `acetylcholine` direct, `tempo` dropped, `serotonin` pinned to `0.0`) |
| `SnnBackend::step_frozen` (required; a backend without a frozen path may delegate to `step` and report `frozen_evaluation = false`) | `SpikingNetwork::step_frozen` (default **fails closed**; overriders must advertise the `frozen_evaluation` feature, and `SpikingExpertRouter::forward_frozen` gates on it) | same held-out contract, different enforcement |
| `SnnBackend::reset()` (infallible) | `SpikingNetwork::reset() -> Result<()>` (default: unsupported error) | see reset semantics below |
| `SnnCapabilities { backend_name, frozen_evaluation, plasticity, caller_rng, neuromodulation }` | `BackendCapabilities::snn(..)` + `RequiredFeature` strings `step`, `frozen_evaluation`, `plasticity`, `caller_rng`, `neuromodulation` | two vocabularies for the same facts; both adapters report identical values |
| `SnnStepOutput { spikes: Vec<usize> }` | `Vec<usize>` fired IDs | same index space (`0..` output population) |
| `SnnEncoder::encode(&ann_signal, channels)`, `reset()` | `SpikeEncoder::encode`, `reset()`, plus `encoding()` (`SpikeEncoding` plan-port metadata) | identical semantics; hybrid-fusion adds declared metadata |
| `SnnDecoder::decode(&SnnStepOutput, channels) -> Vec<f32>` (infallible) | `SpikeDecoder::decode(&[usize], num_neurons) -> Result<Vec<f32>>`, plus `decoding()` metadata | sized by channel count vs output population; infallible vs fail-closed |
| `RateEncoder { gain }` | `RateEncoder { gain }` | identical tiling semantics |
| `SpikeCountDecoder` | `SpikeCountDecoder` | counts per channel / per neuron; out-of-range IDs silently dropped vs rejected (see below) |
| `SpikingMoeRouter<B, E, D>` over the concrete `MoeRouter` | `SpikingExpertRouter<R, S, E, D>` over any `ExpertRouter` | same pipeline: gate scores → encode → step → decode → additive spike boost → top-k |
| `DEFAULT_SPIKE_BOOST = 1.0`, `with_spike_boost` (finite, ≥ 0) | identical | shared routing constant and validation |
| `MoeOutput { expert_weights, selected_experts, hidden: Vec<f32> }` | `SpikingRouteOutput { route, fired_neurons, num_neurons, hidden_gain: f32 }` | same boosted softmax weights and selection; scaled hidden state here, scalar gain there (plus entropy and fired IDs there) |
| `NeuromodNetwork` (`new(num_lif, num_izh, num_channels)`, `from_network`, `with_seed`) | `NeuromodSnn` (`with_seed(channels, neurons, seed)`, `identity_wired(channels, seed)`, `reseed`) | both wrap `neuromod` 0.7; wiring/reset differences below |
| `AxonEncoder` (`axon-encoder` feature) | — | cortex-tensor-only; no counterpart, out of parity scope |
| `CortexError::{SnnBackend, SnnChannelMismatch, SnnNan, SnnSpikeOutOfRange, SnnDuplicateSpike, SnnReadoutLength, SnnInvalidReadout}` | `HybridError::{SnnStep, ConfigMismatch, NonFinite, SnnSpikeOutOfRange, SnnDuplicateSpike, InputLengthMismatch, SnnInvalidReadout}` | same failure conditions, different vocabularies; both hosts reject NaN before the tick and validate spikes/readouts after it |

## Intentional differences (each asserted by the parity tests)

1. **Width axes.** Here `channels()` is the whole width contract — the
   adapter requires **one LIF neuron per channel**
   (`NeuromodNetwork::new(8, 1, 4)` is an `InvalidConfig` error) because
   `neuromod` reports LIF spikes only. hybrid-fusion separates stimulus width
   from output population (`NeuromodSnn::with_seed(4, 8, ..)` is fine), and
   its `identity_wired` is the one-neuron-per-channel special case.
2. **Decoder fallibility and sizing.** `SnnDecoder` is infallible, sized by
   the channel count, and `SpikeCountDecoder` silently drops out-of-range
   IDs; `SpikingMoeRouter` validates the spike list before decoding, so a
   compliant flow never reaches the decoder with a bad ID. hybrid-fusion's
   decoder is fallible, sized by `num_neurons`, and **rejects** out-of-range
   fired IDs. Both *hosts* reject out-of-range and duplicate fired IDs with
   equivalent typed errors; only standalone-decoder behavior differs.
3. **Reset and replay.** `NeuromodNetwork::reset` rewinds the replay RNG to
   its seed and keeps learned weights (neuromod's reset does not touch them);
   replay after `reset` is exact for frozen-only stepping.
   `NeuromodSnn::reset` restores construction-time weights/thresholds and
   does **not** reseed — its replay recipe is `reset(); reseed(seed)`.
4. **Output shape.** `SpikingMoeRouter` returns the scaled hidden state
   (`embedding × spiking_mass`). `SpikingExpertRouter` returns the scalar
   `hidden_gain` in `[0, 1]` and the caller scales. Both compute the identical
   bounded gain (selected softmax mass weighted by activity normalized to the
   tick's peak; silent tick falls back to the dense selected mass).
5. **Adapter boundary.** `NeuromodNetwork::from_network` is public, which is
   where the never-spiking-neuron rejection (#91) is observable here.
   hybrid-fusion's equivalent is `pub(crate)` (no `neuromod` type may cross
   its public API) and its public constructors always seed strictly positive
   weights. hybrid-fusion's adapter additionally enforces a 64 MiB matrix
   budget and an exhaustive `StepError` mapping.

## Fix parity (#91–#93)

All four fixes exist on **both** sides; the parity tests keep it that way:

| Fix | cortex-tensor | hybrid-fusion |
|---|---|---|
| #91 never-spiking channel rejection + drivable identity wiring | `NeuromodNetwork::checked` / `new` (PR #131) | `NeuromodSnn::from_network` (`pub(crate)`) + seeded constructors (PR #65) |
| #92 real frozen step + truthful `caller_rng` | `step_frozen_with_rng`, `with_seed`/`set_seed`, `caller_rng` reflects the installed seed (PR #131) | `step_frozen_with_rng`, always-owned seeded RNG, `frozen_evaluation` + `caller_rng` features (PR #65) |
| #92-style held-out never runs the learning path | trait contract: backends without a frozen path may delegate | trait default fails closed + host capability gate |
| #93 additive spike boost on gate logits (`gate + spike_boost × activity`, saturating at `f32::MAX`, `±Inf` gates keep their value) | `SpikingMoeRouter` | `SpikingExpertRouter` (PR #65) |
| #93 bounded hidden gain (activity normalized to the tick peak, silent-tick fallback) | same formula | same formula |

## Consequences

- No document here may claim hybrid-fusion is the *single* owner of the
  ANN↔SNN step contract or that this crate is ANN-only. cortex-tensor is
  deliberately ANN+SNN capable; that is why this module exists.
- A change to one contract that should apply to both must now say so: edit
  both crates, both mapping documents, and extend the parity tests — or
  explicitly record a new intentional difference in both.
- hybrid-fusion's `smoke/cortex-neuromod` package pins this crate by git rev
  (RM-2208 bumped it past the #91–#93 fix series) and runs the parity tests
  against that pin in CI; keep the pinned rev buildable with the `neuromod`
  feature.
