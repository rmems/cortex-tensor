# cortex-tensor

Pure-Rust tensor, transformer, and Mixture-of-Experts building blocks. No CUDA, no Julia FFI, no framework dependencies — just `Vec<f32>` and honest math.

[![Rust](https://img.shields.io/badge/rust-edition%202024-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue)](./LICENSE-APACHE-2.0)
[![codecov](https://codecov.io/gh/rmems/cortex-tensor/branch/main/graph/badge.svg)](https://codecov.io/gh/rmems/cortex-tensor)

## Overview

`cortex-tensor` is a minimal, framework-free foundation for building transformer-based language models and MoE routers in Rust. It was surgically extracted from a larger hybrid codebase (`corinth-canal`) and stripped of GPU / CUDA / Julia concerns so it can stand alone as a reusable, open-source numerical kernel. It stays ANN/SNN hybrid-capable through a backend-neutral SNN contract (`snn` module) — neuron dynamics come from focused reusable crates (e.g. `neuromod`) behind feature-gated adapters, not an embedded runtime.

Design goals:

- **Zero GPU coupling.** No `cust`, no `libc` pinned-host registration, no `#[cfg(feature = "gpu")]` branches.
- **Zero framework dependency.** No `candle`, no `tch`, no `ort`. The tensor type is a single contiguous, row-major `Vec<f32>` plus a shape; there are no strided views (strides are implied by the shape and recomputed on demand).
- **Small, auditable dependency set.** `serde`, `serde_json`, `thiserror`, `rand`, `rayon`, `memmap2`, `half` — nothing else.
- **Inference-ready MoE.** A GGUF checkpoint bridge with adapter resolution for MoE checkpoints; backend-neutral — `general.architecture` is treated as opaque checkpoint data.

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
│   ├── attention.rs  # MultiHeadAttention (scaled dot-product, causal mask)
│   ├── block.rs      # TransformerBlock (attn + MLP + LayerNorm)
│   ├── model.rs      # TransformerConfig + TransformerLM (decoder-only)
│   └── mod.rs
└── moe/
    ├── mod.rs        # MoeRouter public API, RoutingMode
    ├── adapter.rs    # checkpoint adapter resolution + tensor selection
    ├── checkpoint.rs # GGUF parser, mmap'd F32/F16/Q8_0/Q5_K access
    ├── dequant.rs    # Q8_0 / Q5_K row dequant, f16→f32, row sizing
    ├── gguf.rs       # GGUF magic/version + GGML type constants
    ├── routing.rs    # softmax, top-k, L2 normalize, embedding resample
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
| `ops::causal_mask` | Additive mask for auto-regressive attention. |
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
| `MultiHeadAttention` | Multi-head scaled dot-product attention with causal masking. Weights are dense `Tensor`s; no external framework needed. |
| `TransformerBlock` | Attention → residual → MLP → residual, with pre-LayerNorm. |
| `TransformerLM` / `TransformerConfig` | Decoder-only transformer LM: token + positional embedding → N × block → LayerNorm → LM head. |

### `moe`

| Item | Purpose |
|---|---|
| `MoeRouter` | MoE router. Loads a GGUF checkpoint and produces top-k expert selections. |
| `RoutingMode` | `StubUniform`, `DenseSim` (default). Spiking routing moved to `snn::SpikingMoeRouter` + an `SnnBackend` adapter. |
| `ExtractTokenOptions` | Optional resample / L2 when extracting token rows (`for_projector_forward()` matches legacy 2048 + L2). |

Top-k selection ranks finite scores descending and breaks ties by ascending expert ID. NaN scores are rejected (`NanRoutingScore`); `+Inf` ranks above all finite values and `-Inf` below them. The helper is pure: the same `(scores, top_k)` pair always produces the same expert IDs.

Supported GGUF tensor types: `F32`, `F16`, `Q8_0`, `Q5_K`. `IQ3_S` is detected and rejected (for token embeddings) with a clear error so callers can fall back to `llama.cpp` prompt embeddings.

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

```toml
[dependencies]
cortex-tensor = { git = "https://github.com/rmems/cortex-tensor", branch = "main", features = ["neuromod"] }
```

```rust
use cortex_tensor::moe::{MoeRouter, RoutingMode};
use cortex_tensor::snn::{NeuromodNetwork, RateEncoder, SpikingMoeRouter, SpikeCountDecoder};

let router = MoeRouter::load("model.gguf", 0, 2)?;
let backend = NeuromodNetwork::new(/* lif */ 8, /* izh */ 0, /* channels */ 8);
let mut hybrid = SpikingMoeRouter::new(router, backend, RateEncoder::default(), SpikeCountDecoder)?;
let out = hybrid.forward(&embedding)?; // or forward_frozen for held-out eval
```

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
- GGUF parsing/mmap stays as-is pending #47 (engram-parser 0.3.x consume).

**Parser layer (planning, see #8 / #47):** the canonical home for GGUF v3
deserialization and per-expert raw weight extraction is `engram-parser`, not this
crate. Parser/dtype freeze holds until #47 can wrap `parse_checkpoint_layout`
around engram-parser 0.2.0 — see [GGUF parser boundary](#gguf-parser-boundary-see-8).

**Future formats (planning, see #9 / #32):** Safetensors header inspection,
deterministic manifests, and MoE candidate discovery belong in `engram-parser`
behind the off-by-default `safetensors` cargo feature (engram-parser#10,
corinth-canal#116). This crate still does not own that parse surface. The
eventual dependency is one crate, not two:

```toml
engram-parser = { version = "...", features = ["safetensors"] }
```

No implementation or dependency is present yet — this keeps the reusable parser
boundary clean. Cross-links and notes are maintained for alignment.

### GGUF adapter (code paths)

- `MoeRouter::load` / `load_with_mode` → `probe_and_map` calls `resolve_adapter` (adapter.rs).
- `resolve_adapter` reads `{architecture}.*` metadata keys (the arch string is opaque checkpoint data), validates routing tensor (rank-2 F32 with one dim equal to `hidden_size` and the other ≥ expert count), validates token embeddings (F32/F16/Q8_0/Q5_K, shape `[hidden, vocab]`).
- Routing always uses `routing_tensor` via `checkpoint_gate_scores` (routing.rs) when checkpoint loaded (never synthetic for real loads).
- `extract_named_token_embedding_from_checkpoint` (checkpoint.rs) supports dequant for Q8_0/Q5_K (and F32/F16); IQ3_S errors for embeddings.
- Checkpoint topology (which tensors exist) is reported via `RouterMetadata` / adapter internals. **SAAQ/GPU synapse-source policy is no longer owned here** — it moved downstream to the experimental repos (`corinth-canal` / `grok-ozempic`) that consume these checkpoints; see [Migration notes](#migration-notes).

## Scope / Boundaries

This crate **owns**:

- Row-major `f32` `Tensor` and CPU tensor ops (matmul, batched matmul, causal
  mask, softmax, layer norm, RMSNorm) under a documented finite-value policy.
- Decoder-only transformer building blocks (attention, block, `TransformerLM`).
- MoE routing math — gate scores, softmax, top-k selection, L2 normalization,
  embedding resampling — and the dense simulation routing mode.
- The ANN↔SNN execution/interchange contract (`snn` module): `SnnBackend`,
  `SnnEncoder`/`SnnDecoder` boundary types, `SnnCapabilities`, and
  `SpikingMoeRouter` composing the MoE router with an external SNN backend.
- Checkpoint adapter and tensor selection.
- Dequantization of supported GGUF quants to `f32` (`Q8_0`, `Q5_K`, `F16`).
- The consumer-side GGUF bridge it needs today: mmap'd tensor access and
  token-embedding extraction.

This crate **does not own**:

- Canonical GGUF v3 deserialization (header, KV metadata, tensor directory) and
  per-expert *raw* weight extraction — see
  [`engram-parser`](https://github.com/rmems/engram-parser) and the
  parser-boundary note below.
- Safetensors header inspection, deterministic manifests, and MoE candidate
  discovery — `engram-parser` feature `safetensors` (see #9, #32).
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

**Allowed dependencies:** the current small set — `serde`, `serde_json`,
`thiserror`, `rand`, `rayon`, `memmap2`, `half` — optional feature-gated
SNN backend crates (`neuromod` today, evaluated individually) and,
in future, the zero-dependency rmems parser crates.

**Forbidden dependencies:** GPU backends (`cust`), inference frameworks
(`candle`, `tch`, `ort`), domain/SNN orchestration crates, and any dependency on
`rmems/corinth-canal`. Extraction from corinth-canal is a **one-way copy**; that
repo keeps an unmodified reference copy per its `PROMOTION_RULES.md`.

| Crate | Role |
|-------|------|
| [`engram-parser`](https://github.com/rmems/engram-parser) | GGUF parse + per-expert raw weight extraction |
| `cortex-tensor` (this crate) | Tensor math + MoE routing on extracted weights |
| [`hybrid-fusion`](https://github.com/rmems/hybrid-fusion) | ANN→SNN orchestration |
| [`neuromod`](https://github.com/rmems/neuromod) | SNN neuron dynamics (downstream consumer) |

See [LIM-9](https://linear.app/saaq-spiking-adaptive-activity/issue/LIM-9/plan-rust-runtime-and-deployment-repo-boundary-matrix)
for the full Rust runtime/deployment boundary matrix, and issues #5 (boundary
doc), #8 (GGUF parser coordination), #9 / #32 (Safetensors provider), and #47
(consume `engram-parser` 0.2.0) for this repo's tracking.

### GGUF parser boundary (see #8)

`engram-parser` is the parser-layer provider for this ecosystem: it is the
canonical home for GGUF v3 layout parsing and MoE per-expert raw weight
extraction, extracted from the experimental `rmems/corinth-canal` reference
implementation (see closed
[engram-parser#7](https://github.com/rmems/engram-parser/issues/7) and
[corinth-canal#115](https://github.com/rmems/corinth-canal/issues/115)).
`cortex-tensor` stays the consumer: `f32` math, `Tensor` ops, routing, and model
adapters on top of parsed layout / extracted weights. Consume follow-up: [#47](https://github.com/rmems/cortex-tensor/issues/47).

| Layer | Canonical owner | Where it lives in this crate today |
|---|---|---|
| GGUF magic + v3 header, KV metadata, tensor directory | `engram-parser` | `src/moe/checkpoint.rs` (`parse_checkpoint_layout`) |
| GGML type + GGUF value-type constants | `engram-parser` | `src/moe/gguf.rs` |
| Per-expert raw weight extraction | `engram-parser` | not implemented here |
| mmap'd tensor access for the router | `cortex-tensor` | `src/moe/checkpoint.rs` (`probe_and_map_checkpoint`) |
| Dequantization to `f32` | `cortex-tensor` | `src/moe/dequant.rs` |
| Routing math, top-k, checkpoint adapter | `cortex-tensor` | `src/moe/routing.rs`, `src/moe/adapter.rs` |

**Freeze until consume lands:** no new parser code and no dtype/GGUF format
enhancements in `src/moe/checkpoint.rs`, `src/moe/gguf.rs`, or
`src/moe/dequant.rs` until [#47](https://github.com/rmems/cortex-tensor/issues/47)
can wrap `parse_checkpoint_layout` around engram-parser 0.2.0. That issue is
blocked on [engram-parser#45](https://github.com/rmems/engram-parser/issues/45)
(mmap + K-quant). Known gaps versus the corinth-canal reference — additional
dtypes (`BF16`, `Q6_K`, `IQ3_*`), a `ggml_type_label` helper, and the
"GGUF wire type 31 is `Q4_0_4_4`, not IQ3_M" discipline — stay parked on
engram-parser. **Do not add Q6_K / IQ3_* dequant in this crate.** Cross-repo
planning is tracked in Linear LIM-88 (under LIM-9).

## Install

```toml
[dependencies]
cortex-tensor = { git = "https://github.com/rmems/cortex-tensor", branch = "main" }
```

Features (all off by default):

| Feature | Effect |
|---|---|
| `neuromod` | Enables `snn::NeuromodNetwork` — `neuromod::SpikingNetwork` behind `snn::SnnBackend`. Pulls the optional `neuromod` crate dependency. |

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

### Integration notes (RM-1354)

- **Public API:** Prefer `ops::try_softmax`, `Tensor::try_softmax_last`, `try_layer_norm`, and `try_rms_norm` for rank, shape, `eps`, and allocation checks. Row kernels in `tensor::finite` are crate-internal; they assume valid buffer lengths and (for norms) an `eps` already accepted by `validate_norm_eps`.
- **Softmax sum tolerance:** After the `f64` partition, an `f32` residual is folded onto the largest mass so finite rows of length `≤ 4096` meet `SOFTMAX_SUM_TOLERANCE` (`1e-5`), including uniform branches (all `-Inf`, degenerate partition).
- **Causal attention:** `MultiHeadAttention` applies an additive causal mask, then row-wise `softmax_row` over the full `[seq]` logits for each query position. The policy’s all-`-Inf` case is uniform `1/seq` over **every** index in that row. In normal use at least one causal-prefix logit stays finite after masking, so future (masked) slots do not receive mass. If every logit in the row is `-Inf` (for example pathological scores on the whole prefix), uniform softmax can assign probability to masked future positions; fixing that belongs at the attention/mask layer, not in the shared row kernel.
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

Extracted and compiling cleanly under Rust edition 2024. Public API is subject to change until `0.1.0` is tagged. Tests and benchmarks are incoming.

## License

This project is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
- MIT license ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))

at your option.
