# cortex-tensor

Backend-neutral ANN stage layer and deterministic reference backend for hybrid ANN/SNN systems. The in-tree `Vec<f32>` tensor, transformer, and dense MoE path are the reference backend used for CI, golden vectors, and conformance. They are not a generic ML framework, and this crate does not rank external engines.

[![Rust](https://img.shields.io/badge/rust-edition%202024-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue)](./LICENSE-APACHE-2.0)
[![codecov](https://codecov.io/gh/rmems/cortex-tensor/branch/main/graph/badge.svg)](https://codecov.io/gh/rmems/cortex-tensor)

## Overview

`cortex-tensor` makes ANN engines usable as interchangeable stage executors inside heterogeneous ANN/SNN plans. It does not compete on generic tensor frameworks, attention kernels, training stacks, or GPU backends.

Core owns the stage vocabulary, execution contract, hidden-state ingress/egress, and capability facts. The reference backend is `f32`, CPU-only, and dense. Candle and Burn are optional adapters; their native tensor and device types do not appear in core. Neuron dynamics stay in focused SNN crates behind the `snn` contract.

```text
engram-parser          checkpoint + topology (not this crate)
        |
        v
hybrid-fusion          placement + orchestration (not merged here)
        |
        v
cortex-tensor          ANN stage contracts + reference backend
   |        |        |
reference  candle    burn     + SNN backend contract
```

The `candle` feature is `adapter::CandleExecutor` (RM-1828), a CPU `f32` `AnnExecutor` with its own hidden-state copy. That copy is not the `ExternalAdapter` contract. The `burn` feature is a policy stub on that contract and does not depend on the `burn` crate. Neither engine is a core dependency.

Design boundaries:

- **Reference backend, not the product surface.** `Tensor` is a contiguous row-major `Vec<f32>`. Public stage contracts (`AnnStage`, `AnnExecutor`) do not name it. New framework-generic numerical features are out of scope unless a conformance or reference test needs them.
- **No mandatory ML framework.** Default features do not depend on Candle, Burn, `tch`, or `ort`.
- **No GPU kernels in this crate.** Specialized hybrid GPU kernels belong to `myelin-accelerator`.
- **Checkpoint parsing is not owned here.** Loading goes through published `engram-parser` 0.3.0 (`features = ["safetensors"]`; `mmap` is off). GGUF and Safetensors share one normalized tensor inventory. This crate binds those weights into the reference backend and does not parse either container.
- **Orchestration is not owned here.** ANN↔SNN placement stays in [`hybrid-fusion`](https://github.com/rmems/hybrid-fusion). The closed decision not to merge that crate (issue 35) stands.

## Architecture

```
src/
├── lib.rs            # re-exports Tensor, CortexError, HybridError, Result
├── error.rs          # CortexError + HybridError alias
├── types.rs          # EMBEDDING_DIM, ExtractTokenOptions, RoutingMode
├── tensor/
│   ├── finite.rs     # finite-value policy + stable softmax / LayerNorm / RMSNorm / L2 kernels
│   ├── mod.rs        # row-major Tensor { data, shape } (strides recomputed from shape)
│   └── ops.rs        # matmul, batched_matmul, causal_mask, softmax, layer_norm, rms_norm
├── transformer/
│   ├── attention.rs  # MultiHeadAttention (scaled dot-product, causal prefix softmax)
│   ├── block.rs      # TransformerBlock (attn + MLP + LayerNorm)
│   ├── model.rs      # TransformerConfig + TransformerLM (decoder-only)
│   └── mod.rs
└── moe/
    ├── mod.rs        # MoeRouter public API, RoutingMode
    ├── adapter.rs    # checkpoint adapter resolution + tensor selection
    ├── checkpoint.rs # normalized checkpoint binding (engram-parser shim)
    ├── routing.rs    # softmax, top-k, L2 normalize, embedding resample
    ├── test_fixtures.rs
    └── tests.rs      # router + checkpoint unit tests
```

## Modules

### `tensor`

| Item | Purpose |
|---|---|
| `Tensor` | Row-major `f32` tensor (`data: Vec<f32>`, `shape`), `Serialize`/`Deserialize`. Always contiguous; row-major strides are recomputed via `row_major_strides()`, never stored. |
| `Tensor::try_from_vec` | Fallible constructor: checked `numel` and stride arithmetic, no giant overflow allocations. |
| `ops::try_matmul` / `try_batched_matmul` | Checked CPU matmul; rank, inner-dim, and size errors are typed. |
| `ops::try_embedding` | Embedding lookup that returns `CortexError::TokenIndex` for OOV ids. |
| `ops::try_layer_norm` / `try_rms_norm` | Normalization that rejects zero-width axes and non-positive/non-finite `eps`. |
| `ops::matmul` / `batched_matmul` | Pre-1.0 panic-style wrappers around the `try_*` ops. |
| `ops::causal_mask` | Additive mask for auto-regressive attention (kept for callers and golden tests; `MultiHeadAttention` no longer uses it). |
| `ops::softmax` / `Tensor::softmax_last` | Numerically stable softmax under the [finite-value policy](#finite-value-policy); `try_softmax` / `try_softmax_last` reject rank `> 2`. |
| `ops::layer_norm` / `rms_norm` | Finite-value LayerNorm / RMSNorm (`try_*` for typed errors). |
| `tensor::finite` | Policy docs and public constants (`SOFTMAX_SUM_TOLERANCE`, `L2_NORM_FLOOR`). |

#### Reference JSON fixtures

Serde JSON is a **versioned fixture format for this deterministic CPU reference backend**. It is not a tensor interchange format for Candle, Burn, hybrid-fusion, or other backends. Version 1 tensors use `{"schema_version":1,"data":[1.0,2.0],"shape":[2]}`. Strides are derived from the shape and are never stored. Tensor, attention, feed-forward, block, config, and LM values each carry `schema_version: 1`; unknown versions and fields are rejected. Deserialization checks tensor layout and transformer weight shapes. Serialization refuses NaN and infinity, so JSON never silently writes them as `null`; finite signed zero round-trips.

For untrusted JSON, use `reference_json::from_slice_with_limit::<Tensor>(&bytes, max_bytes)` (or the same call with `TransformerLM`) and choose a maximum for the complete input. The byte check runs before Serde allocates tensor data. Direct `serde_json::from_str` and `from_value` remain available for trusted fixtures but do not enforce an input-size limit. Versioned values must be JSON objects; positional arrays are rejected.

Unversioned pre-1.0 JSON and the older `strides` field must be migrated explicitly before loading. This strict format is for reproducible reference tests and snapshots, not cross-backend persistence.

### `transformer`

| Item | Purpose |
|---|---|
| `MultiHeadAttention` | Multi-head scaled dot-product attention; each query row is normalised over its causal prefix only (no additive mask). Weights are dense `Tensor`s; no external framework needed. |
| `TransformerBlock` | Attention → residual → MLP → residual, with pre-LayerNorm. |
| `TransformerLM` / `TransformerConfig` | Decoder-only transformer LM: token + positional embedding → N × block → LayerNorm → LM head. |

### `moe`

| Item | Purpose |
|---|---|
| `MoeRouter` | MoE router. Opens a GGUF or Safetensors checkpoint through engram-parser and produces top-k expert selections. |
| `RoutingMode` | `StubUniform`, `DenseSim` (default). Spiking routing moved to `snn::SpikingMoeRouter` + an `SnnBackend` adapter. |
| `ExtractTokenOptions` | Optional resample / L2 when extracting token rows (`for_projector_forward()` matches legacy 2048 + L2). |

Top-k selection ranks finite scores descending and breaks ties by ascending expert ID. NaN scores are rejected (`NanRoutingScore`); `+Inf` ranks above all finite values and `-Inf` below them. The helper is pure: the same `(scores, top_k)` pair always produces the same expert IDs.

Supported token-row dtypes, dequantized by engram-parser: `F32`, `F16`, `Q8_0`, `Q5_K`, `Q6_K`. `IQ3_S` is detected and rejected for token embeddings with a clear error so callers can fall back to `llama.cpp` prompt embeddings. Other IQ3 wire types (`IQ3_XXS` and the rest) fail the same way as any dtype engram-parser 0.3.0 does not decode. Shapes are outermost-first (`[vocab, hidden]` for token rows). Gate weights use that same row-major layout.

### `snn`

Hybrid execution/interchange contract — cortex stays ANN/SNN-capable without
embedding neuron dynamics.

| Item | Purpose |
|---|---|
| `SnnBackend` | Execution contract: `step`, `step_frozen` (held-out eval), `reset`, `capabilities`, `channels`. Failure-atomic by contract. |
| `SnnEncoder` / `SnnDecoder` | ANN→SNN stimulus encoding and SNN→ANN readout (`RateEncoder`, `SpikeCountDecoder` defaults). |
| `SnnStepOutput` / `SnnCapabilities` | Per-tick spike output and backend capability reporting for orchestrators (e.g. `hybrid-fusion`). |
| `SpikingMoeRouter` | Composes `MoeRouter` gate scores with an `SnnBackend`: encode → step → decode → deterministic top-k. Replaces the removed embedded `RoutingMode::SpikingSim`. |
| `NeuromodNetwork` | `neuromod`-feature adapter over `neuromod::SpikingNetwork` (LIF + Izhikevich, R-STDP). Frozen eval delegates to `step` until a neuromod release ships `step_frozen`; `capabilities().frozen_evaluation` reports `false` meanwhile. |
| `AxonEncoder` | `axon-encoder`-feature adapter over any `axon_encoder::Encoder` (rate, latency, delta, predictive, population). Spikes net to `+1.0`/`-1.0` per channel per tick. |

```toml
[dependencies]
cortex-tensor = { git = "https://github.com/rmems/cortex-tensor", branch = "main", features = ["neuromod"] }
```

```rust
use cortex_tensor::moe::{MoeRouter, RoutingMode};
use cortex_tensor::snn::{NeuromodNetwork, RateEncoder, SpikingMoeRouter, SpikeCountDecoder};

let router = MoeRouter::load("model.gguf", 0, 2)?;
let backend = NeuromodNetwork::new(/* lif */ 8, /* izh */ 0, /* channels */ 8)?;
let mut hybrid = SpikingMoeRouter::new(router, backend, RateEncoder::default(), SpikeCountDecoder)?;
let out = hybrid.forward(&embedding)?; // or forward_frozen for held-out eval
```

### `stage`

Backend-neutral ANN stage-execution contracts — describe an ANN forward pass as
an ordered graph of stages, then run it through a swappable backend.

| Item | Purpose |
|---|---|
| `StageKind` / `StageKindTag` | What a stage computes (embedding, attention, layer/RMS norm, dense/gated MLP, MoE router/expert, add, readout, custom) and its fieldless tag for capability checks. |
| `AnnStage` / `StageId` / `AnnTopology` | A stage (validated dotted-path id + kind + input wiring) and a validated, topologically-ordered graph of them, with `sub_span` extraction and `SubSpan::to_topology` to run a span against externally supplied stage outputs. |
| `AnnExecutor` / `AnnCapabilities` | The execution contract (generic over the backend's tensor type — it never names `Tensor`) plus capability reporting, mirroring `snn::SnnBackend`. |
| `run_topology` / `ExternalBindings` | Composition driver that runs an executor over a topology, resolving external ingress (tokens or hidden-state tensors; one name, one binding) and, for a sub-span topology, the earlier-stage outputs bound with `with_stage_output`. |
| `ReferenceExecutor` | The reference backend (see below) and `from_transformer`, which emits a block-ordered topology for a `TransformerLM`. |

`ReferenceExecutor` is the reference backend: `f32`, CPU-only, and dense. It
reuses the same kernels as `TransformerLM::try_forward`, so a topology built by
`ReferenceExecutor::from_transformer` reproduces the model's output bit for bit.
`AnnExecutor` is generic over the backend tensor and never names `Tensor` or
`Vec<f32>`.

MoE router, MoE expert, and gated MLP stages are in the vocabulary but are
**rejected** by the reference backend with `CortexError::UnsupportedStage`.
A device field and a concrete cross-backend tensor wire format are
intentionally deferred (RM-1827); stage descriptors carry no serde and
describe only identity, kind, and wiring.

`ReferenceExecutor` implements `ExternalAdapter`. `HiddenStateBuffer` is that
contract's import/export boundary. It is not a checkpoint format and not a
universal tensor wire. `NegotiationDocument` is the cortex-owned fact list
hybrid-fusion can translate once its `BackendCapabilities` type lands
([hybrid-fusion#41](https://github.com/rmems/hybrid-fusion/issues/41)).
This crate does not select, rank, or place backends.

`CandleExecutor` (`features = ["candle"]`, module `adapter`) is an
`AnnExecutor`, not an `ExternalAdapter`. Its tensor type is a Candle CPU
tensor. Row-major `f32` values cross that executor through
`import_hidden_state` / `export_hidden_state` (`HiddenState`), not through
`HiddenStateBuffer`. Bound weights are converted once. Stage bodies call
Candle's `Embedding`, `Linear`, `LayerNorm`, `RmsNorm`, tanh-approximate
GELU, `swiglu`, and CPU flash-attention. This crate does not reimplement
those kernels and does not enable Candle's CUDA, MKL, or Metal features.
MoE router, MoE expert, and custom stages return `UnsupportedStage`.
`BurnAdapter` (`features = ["burn"]`) is the policy stub on
`ExternalAdapter`: it does not depend on the `burn` crate, and it refuses
execution and hidden-state transfer.

### Migration notes

- `RoutingMode::SpikingSim` → `snn::SpikingMoeRouter` + an `SnnBackend`
  (`NeuromodNetwork` via `features = ["neuromod"]`). `RoutingMode::default()`
  is now `DenseSim` (was `SpikingSim`).
- `MoeRouter::reset_state` removed — call `SpikingMoeRouter::reset` or the
  backend's `reset`.
- `preferred_gpu_synapse_tensor_name()`, `real_gpu_synapse_tensor_name()`,
  `synapse_source()`, `RouterMetadata.preferred_gpu_synapse_tensor_name`,
  `RouterMetadata.synapse_source` removed. SAAQ/GPU synapse-source policy now
  lives downstream in `corinth-canal` / `grok-ozempic`. Artifact consumers that
  read a `synapse_source` manifest label (e.g. `Surrogate_Viz.jl`) keep working
  against artifacts emitted by those downstream pipelines; new artifacts should
  source the label there, not from cortex.
- Checkpoint loading uses engram-parser 0.3.0. `MappedCheckpoint` in `src/moe/checkpoint.rs` is a temporary binding shim around `engram_parser::Checkpoint`. Remove it when callers take that trait directly. Do not add format parsers or mmap policy back into this crate.

**Checkpoint substrate (engram-parser 0.3.0, see #47):** GGUF and Safetensors parsing, tensor inventory, raw payloads, and packed Q8_0/Q5_K/Q6_K dequantization live in `engram-parser`. This crate depends on that release with `features = ["safetensors"]` and does not enable `mmap`. `MoeRouter` binds the normalized outermost-first tensors into reference-backend routing, keeps the decoded gate and the token-embedding payload, and releases the rest of the checkpoint. Model topology and tensor-role discovery stay on the engram-parser v0.4 track and are not required here.

```toml
engram-parser = { version = "0.3.0", default-features = false, features = ["safetensors"] }
```

### GGUF adapter (code paths)

- `MoeRouter::load` / `load_with_mode` → `probe_and_map` calls `resolve_adapter` (adapter.rs).
- `resolve_adapter` reads `{architecture}.*` metadata keys (the arch string is opaque checkpoint data). Numeric metadata may be an integer or a numeric string, so the same keys work for GGUF and Safetensors. It validates the routing tensor (rank-2 F32, one outermost-first dim equal to `hidden_size`, the other ≥ expert count) and token embeddings (F32/F16/Q8_0/Q5_K/Q6_K, shape `[vocab, hidden]`).
- Routing always uses `routing_tensor` via `checkpoint_gate_scores` (routing.rs) when a checkpoint is loaded.
- Token-row extraction supports F32/F16/Q8_0/Q5_K/Q6_K through engram-parser dequant. IQ3_S returns an explicit error.
- Checkpoint inventory is reported via `RouterMetadata` / adapter internals. **SAAQ/GPU synapse-source policy is no longer owned here** — it moved downstream to the experimental repos (`corinth-canal` / `grok-ozempic`); see [Migration notes](#migration-notes).

## Scope / Boundaries

This crate **owns**:

- A backend-neutral ANN stage vocabulary and `AnnExecutor` contract (`stage`).
- Hidden-state ingress and egress at stage boundaries (`ExternalBindings`, `run_topology`).
- External adapter capability facts and hidden-state import/export (`adapter`).
  Native Candle and Burn types stay behind optional features.
- The deterministic `Vec<f32>` reference backend: row-major `Tensor`, CPU ops
  (matmul, batched matmul, causal mask, softmax, layer norm, RMSNorm),
  decoder-only transformer blocks, and dense MoE routing math. Kept for CI,
  golden vectors, fuzzing, and backend-semantic comparisons. Not a second
  product surface.
- The ANN↔SNN execution/interchange contract (`snn` module): `SnnBackend`,
  `SnnEncoder`/`SnnDecoder` boundary types, `SnnCapabilities`, and
  `SpikingMoeRouter` composing the MoE router with an external SNN backend.
- Checkpoint adapter and tensor selection on engram-parser's normalized inventory.
- Reference-backend binding of F32/F16/Q8_0/Q5_K/Q6_K token rows. Packed dequantization is engram-parser's.

This crate **does not own**:

- Checkpoint parsing, GGUF mmap policy, or Safetensors parsing — [`engram-parser`](https://github.com/rmems/engram-parser) 0.3.0 (`safetensors` feature). Issue 47 is the consume that landed here. Model topology / tensor-role discovery remains engram-parser v0.4 and is not an acceptance gate for this binding.
- ANN↔SNN placement and session orchestration — [`hybrid-fusion`](https://github.com/rmems/hybrid-fusion).
  Issue 35, the decision not to merge hybrid-fusion into this repo, is closed and stands.
- SNN neuron dynamics.
- Generic CUDA/GPU kernels — existing ML backends, or `myelin-accelerator` for specialized hybrid kernels.
- GOZ1 / SAAQ experiment policy.
- A universal Rust tensor framework.
- Canonical GGUF v3 deserialization and Safetensors manifests — engram-parser, not this crate.
- CUDA / GPU / SIMD execution, and any GPU host registration.
- SNN neuron dynamics — delegated to reusable crates behind `snn::SnnBackend`
  ([`neuromod`](https://github.com/rmems/neuromod) via the optional `neuromod`
  feature) — and ANN→SNN orchestration
  ([`hybrid-fusion`](https://github.com/rmems/hybrid-fusion)).
- SAAQ experimental policy: GPU synapse-source selection and tensor-source
  preference live in the downstream experimental repos (`corinth-canal` /
  `grok-ozempic`), not in the reusable execution core.
- Specialized SNN/hybrid GPU kernels — `myelin-accelerator`.
- Tokenization and automatic differentiation (see [Non-goals](#non-goals)).

**Allowed dependencies:** `serde`, `serde_json`, `thiserror`, `rand`, and `engram-parser` 0.3.0 with the `safetensors` feature — plus optional feature-gated adapters (`neuromod`, `axon-encoder`, and the `candle` feature's `candle-core` and `candle-nn`, CPU only; CUDA, MKL, and Metal stay off), evaluated individually.

**Forbidden dependencies:** GPU backends (`cust`) as core dependencies,
making an inference framework (`candle`, `burn`, `tch`, `ort`) a required
dependency, domain/SNN orchestration crates (including `hybrid-fusion`), and
any dependency on `rmems/corinth-canal`. The `candle` feature may depend on
`candle-core` and `candle-nn` with CUDA, MKL, and Metal disabled. The `burn`
feature must not depend on the `burn` crate. Native engine types stay out of
the `ExternalAdapter` contract. Extraction from corinth-canal is a
**one-way copy**; that repo keeps an unmodified reference copy per its
`PROMOTION_RULES.md`.

| Crate | Role |
|-------|------|
| [`engram-parser`](https://github.com/rmems/engram-parser) | Checkpoint parsing and topology discovery |
| `cortex-tensor` (this crate) | ANN stage contracts + deterministic reference backend |
| [`hybrid-fusion`](https://github.com/rmems/hybrid-fusion) | Placement and session orchestration (separate repo; not merged) |
| [`neuromod`](https://github.com/rmems/neuromod) | SNN neuron dynamics, behind the optional `neuromod` feature |

See [LIM-9](https://linear.app/saaq-spiking-adaptive-activity/issue/LIM-9/plan-rust-runtime-and-deployment-repo-boundary-matrix)
for the full Rust runtime/deployment boundary matrix, and issues #5 (boundary
doc), #8 (GGUF parser coordination), #9 / #32 (Safetensors provider), and #47
(consume published `engram-parser` 0.3.x) for this repo's tracking.

### Checkpoint boundary (see #8 / #47)

`engram-parser` 0.3.0 owns GGUF and Safetensors parsing. Validated dependency:

```toml
engram-parser = { version = "0.3.0", default-features = false, features = ["safetensors"] }
```

`mmap` is intentionally not enabled. `open_checkpoint` reads GGUF into memory and serves Safetensors with bounded per-tensor reads. Cortex copies F32 payloads with `f32::from_le_bytes` because the borrowed bytes are not promised to be aligned, so `src/moe/` has no `unsafe`.

| Layer | Owner |
|---|---|
| GGUF and Safetensors parsing, tensor inventory, raw payloads | `engram-parser` |
| Packed Q8_0 / Q5_K / Q6_K / IQ3_M dequant | `engram-parser` (this crate calls Q8_0, Q5_K, and Q6_K for token rows) |
| mmap policy | `engram-parser` (`mmap` feature; not used here) |
| Router weight binding, top-k, adapter selection | `cortex-tensor` (`src/moe/checkpoint.rs`, `adapter.rs`, `routing.rs`) |

`MappedCheckpoint` is a migration shim. Remove it when `MoeRouter` callers take `engram_parser::Checkpoint` directly. Malformed or unsupported files return the engram-parser error. There is no in-tree parser to fall back to.

Q6_K token rows call `engram_parser::dequantize_q6_k`. That is the same live path `corinth-canal` uses (`dequantize_row_q6_k` while extracting a token). IQ3 is not on that path in either crate. `corinth-canal` rejects IQ3_S token embeddings and its `dequantize_iq3_m_tensor` helper is unused. Engram-parser 0.3.0 can decode an internal IQ3_M block layout, not the GGUF wire types IQ3_S or IQ3_XXS, so those embeddings fail here with an explicit error. This crate does not reimplement those decoders or add a format parser. Topology and tensor-role discovery are engram-parser v0.4 and are not required to bind weights.

## Install

```toml
[dependencies]
cortex-tensor = { git = "https://github.com/rmems/cortex-tensor", branch = "main" }
```

Features (all off by default):

| Feature | Effect |
|---|---|
| `neuromod` | Enables `snn::NeuromodNetwork` — `neuromod::SpikingNetwork` behind `snn::SnnBackend`. Pulls the optional `neuromod` crate dependency. |
| `axon-encoder` | Enables `snn::AxonEncoder` — any `axon_encoder::Encoder` behind `snn::SnnEncoder` (streaming `encode_step` per forward tick). Pulls the optional `axon-encoder` crate dependency. |
| `candle` | Enables `adapter::CandleExecutor` — Candle CPU `f32` modules behind `AnnExecutor`. Values cross that executor at `import_hidden_state` / `export_hidden_state` (`HiddenState`), not through `ExternalAdapter` / `HiddenStateBuffer`. Does not enable Candle's CUDA, MKL, or Metal features. |
| `burn` | Enables `stage::BurnAdapter` as a structured policy stub on `ExternalAdapter`. Does not depend on the `burn` crate. Hidden-state import/export returns `UnsupportedOperation` (category `policy`) and `execute` returns `UnsupportedStage`. `burn` 0.21's rustc 1.92 requirement is a minimum and is compatible with this crate's 1.99.0 pin. |

## Quick start

```rust
use cortex_tensor::tensor::ops::{matmul, try_matmul};
use cortex_tensor::Tensor;

let a = Tensor::try_from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
let b = Tensor::try_from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
let c = try_matmul(&a, &b).unwrap();
assert_eq!(c.shape(), &[2, 2]);
// Panic-style wrappers (`from_vec`, `matmul`) remain for pre-1.0 compatibility.
let _ = matmul(&a, &b);
```

## Finite-value policy

Tensor math and MoE routing helpers share one contract, implemented in
`tensor::finite`. Public storage remains `f32`; softmax partitions, LayerNorm /
RMSNorm moments, and L2 norms accumulate in `f64`.

**Softmax** is max-subtracted and total over the input classes:

- Empty row → empty output.
- Any NaN → the whole row is NaN (NaN is never an argmax or a `partial_cmp` tie).
- One `+Inf` → one-hot; several `+Inf` maxima share probability equally.
- All `-Inf` (fully masked) → uniform `1/n`.
- Finite logits, including values near `±1e30`, produce non-negative probabilities that sum to 1 within `SOFTMAX_SUM_TOLERANCE` (`1e-5` for rows of length `≤ 4096`).

**LayerNorm / RMSNorm** reject non-positive or non-finite `eps` and a zero-width last axis through `try_layer_norm` / `try_rms_norm` (`CortexError::InvalidEpsilon`, `CortexError::ZeroWidth`). A non-finite data row becomes an all-NaN output row. Finite data with finite affine parameters stay finite; values that overflow `f32` saturate to `±f32::MAX`. The existing `layer_norm` / `rms_norm` wrappers panic on those same rejections.

**L2 routing normalize** fills NaN on any non-finite input, leaves vectors at or below `L2_NORM_FLOOR` unchanged, and otherwise divides by the `f64` Euclidean norm.

Kernels are sequential, so a given input bit pattern is deterministic across debug/release and supported platforms.

The deterministic numerical baseline and coverage inventory live in
[Reference goldens](tests/fixtures/REFERENCE_GOLDENS.md). Run it with
`cargo test --test reference_goldens`; intentional expectation changes require
a documented semantic reason and independent arithmetic review.

### Integration notes (RM-1354)

- **Public API:** Prefer `ops::try_softmax`, `Tensor::try_softmax_last`, `try_layer_norm`, and `try_rms_norm` for rank, shape, `eps`, and allocation checks. Row kernels in `tensor::finite` are crate-internal; they assume valid buffer lengths and (for norms) an `eps` already accepted by `validate_norm_eps`.
- **Softmax sum tolerance:** After the `f64` partition, an `f32` residual is folded onto the largest mass so finite rows of length `≤ 4096` meet `SOFTMAX_SUM_TOLERANCE` (`1e-5`), including uniform branches (all `-Inf`, degenerate partition).
- **Causal attention:** `MultiHeadAttention` runs `softmax_row` over the visible prefix `0..=r` of each query row and writes exact zeros for future slots `r+1..seq` (no additive mask). Future scores are never read, so no score value can leak mass onto a future token. If every visible logit is `-Inf` (for example an overflowing prefix), the shared row policy yields uniform `1/(r+1)` over the visible prefix only. The `attn × V` product also accumulates only value rows `0..=r`, so a NaN/`±Inf` value row at a future position cannot reach earlier tokens through `0.0 * NaN`.
- **MoE routing:** Gate softmax uses the same row policy. Expert selection ranks **raw** scores before softmax (RM-1355); NaN gate scores return `NanRoutingScore` instead of routing with NaN weights.

Building a transformer block:

```rust
use cortex_tensor::transformer::{MultiHeadAttention, TransformerBlock};

let attn = MultiHeadAttention::new(/* dim */ 512, /* num_heads */ 8);
let block = TransformerBlock::new(/* dim */ 512, /* num_heads */ 8, /* mlp_dim */ 2048);
```

Loading a MoE GGUF checkpoint and running the router:

```rust
use cortex_tensor::moe::{ExtractTokenOptions, MoeRouter, RoutingMode};

fn main() -> cortex_tensor::Result<()> {
    let mut router = MoeRouter::load_with_mode(
        "path/to/model.gguf",
        /* num_experts */ 0, // 0 → take count from checkpoint metadata
        /* top_k */ 2,
        RoutingMode::DenseSim,
    )?;
    let native = router.extract_token_embedding(0)?; // checkpoint hidden_size
    let _ = native;
    let embedding = router.extract_token_embedding_with_options(
        0,
        ExtractTokenOptions::for_projector_forward(),
    )?; // resampled to EMBEDDING_DIM + L2 for forward()
    let out = router.forward(&embedding)?;
    // out.selected_experts, out.expert_weights, out.hidden
    let _ = out;
    Ok(())
}
```

## Non-goals

- No GPU backend. Ever. If you need CUDA, consume this crate's `Tensor` into your own kernels.
- No automatic differentiation. This is an inference and forward-pass library.
- No tokenizer. Pair it with `tokenizers` or `llama.cpp`'s tokenizer of choice.
- No embedded SNN runtime. SNN execution goes through `snn::SnnBackend`
  adapters over focused crates (`neuromod` feature); orchestration lives in
  `hybrid-fusion`, neuron dynamics in `neuromod`.

## Status

Rust edition 2024, MSRV 1.99.0. Public API is pre-1.0.

Linked work, not implemented by the boundary-reset docs:

- Issue 57 — `AnnStage` / `AnnExecutor` (closed; on `main`).
- Issue 58 — external Rust ML adapter and capability contracts (open). The Candle executor landed under RM-1828; Burn and the core negotiation contract are still open.
- Issue 47 — consume published engram-parser 0.3.0 and retire in-tree GGUF (this tree).
- Issue 35 — do not merge hybrid-fusion (closed; still in force).

## License

This project is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
- MIT license ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))

at your option.
