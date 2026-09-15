# cortex-tensor

Pure-Rust tensor, transformer, and Mixture-of-Experts building blocks. No CUDA, no Julia FFI, no framework dependencies — just `Vec<f32>` and honest math.

[![Rust](https://img.shields.io/badge/rust-edition%202024-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue)](./LICENSE-APACHE-2.0)
[![codecov](https://codecov.io/gh/rmems/cortex-tensor/branch/main/graph/badge.svg)](https://codecov.io/gh/rmems/cortex-tensor)

## Overview

`cortex-tensor` is a minimal, framework-free foundation for building transformer-based language models and MoE routers in Rust. It was surgically extracted from a larger hybrid codebase (`corinth-canal`) and then stripped of every GPU / CUDA / Julia / SNN-specific concern so it can stand alone as a reusable, open-source numerical kernel.

Design goals:

- **Zero GPU coupling.** No `cust`, no `libc` pinned-host registration, no `#[cfg(feature = "gpu")]` branches.
- **Zero framework dependency.** No `candle`, no `tch`, no `ort`. The tensor type is a row-major `Vec<f32>` with explicit shape + strides.
- **Small, auditable dependency set.** `serde`, `serde_json`, `thiserror`, `rand`, `rayon`, `memmap2`, `half` — nothing else.
- **Inference-ready MoE.** A GGUF checkpoint bridge with family-aware adapter resolution for OLMoE, Qwen3-MoE, Gemma-4, DeepSeek-2, and Llama-MoE.

## Architecture

```
src/
├── lib.rs            # re-exports Tensor, CortexError, HybridError, Result
├── error.rs          # CortexError + HybridError alias
├── types.rs          # EMBEDDING_DIM, ModelFamily, RoutingMode
├── tensor/
│   ├── finite.rs     # finite-value policy + stable softmax / LayerNorm / RMSNorm / L2 kernels
│   ├── mod.rs        # row-major Tensor { data, shape, strides }
│   └── ops.rs        # matmul, batched_matmul, causal_mask, softmax, layer_norm, rms_norm
├── transformer/
│   ├── attention.rs  # MultiHeadAttention (scaled dot-product, causal mask)
│   ├── block.rs      # TransformerBlock (attn + MLP + LayerNorm)
│   ├── model.rs      # TransformerConfig + TransformerLM (decoder-only)
│   └── mod.rs
└── moe/
    ├── mod.rs        # MoeRouter public API, RoutingMode
    ├── adapter.rs    # model-family detection + tensor selection
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
| `Tensor` | Row-major `f32` tensor (`data: Vec<f32>`, `shape`, `strides`), `Serialize`/`Deserialize`. |
| `ops::matmul` / `batched_matmul` | Cache-friendly tiled CPU matmul. |
| `ops::causal_mask` | Additive mask for auto-regressive attention. |
| `ops::softmax` / `layer_norm` / `rms_norm` | Numerically stable kernels under the crate [finite-value policy](#finite-value-policy). Prefer `try_layer_norm` / `try_rms_norm` / `try_softmax` for typed errors. |
| `tensor::finite` | Policy docs and public constants (`SOFTMAX_SUM_TOLERANCE`, `L2_NORM_FLOOR`). |

### `transformer`

| Item | Purpose |
|---|---|
| `MultiHeadAttention` | Multi-head scaled dot-product attention with causal masking. Weights are dense `Tensor`s; no external framework needed. |
| `TransformerBlock` | Attention → residual → MLP → residual, with pre-LayerNorm. |
| `TransformerLM` / `TransformerConfig` | Decoder-only transformer LM: token + positional embedding → N × block → LayerNorm → LM head. |

### `moe`

| Item | Purpose |
|---|---|
| `MoeRouter` | Family-aware MoE router. Loads a GGUF checkpoint, detects model family, and produces top-k expert selections. |
| `RoutingMode` | `StubUniform`, `DenseSim`, `SpikingSim` (simulation-only; no GPU dispatch). |
| `ModelFamily` | `Olmoe`, `Qwen3Moe`, `Gemma4`, `DeepSeek2`, `LlamaMoe`. |

Supported GGUF tensor types: `F32`, `F16`, `Q8_0`, `Q5_K`. `IQ3_S` is detected and rejected (for token embeddings) with a clear error so callers can fall back to `llama.cpp` prompt embeddings. For the preferred GPU synapse tensor (e.g. attn_q on qwen3_moe_iq3_m), unsupported quants now correctly route to a checkpoint-backed `routing-f32` source (using the F32 routing tensor) instead of synthetic fallback. See `synapse_source()`, `real_gpu_synapse_tensor_name()`, and `MoeRouter` metadata.

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

### GGUF adapter + synapse source + SAAQ flow (code paths)

- `MoeRouter::load` / `load_with_family_and_mode` → `probe_and_map` calls `resolve_adapter` (adapter.rs).
- `resolve_adapter` infers family from arch, validates routing tensor (rank-2 F32 with one dim equal to `hidden_size` and the other ≥ expert count), validates token embeddings (F32/F16/Q8_0/Q5_K, shape `[hidden, vocab]`), sets `preferred_gpu_synapse_tensor` to `blk.0.attn_q.weight` when present.
- Synapse source selection (updated for qwen3 IQ3_S): if attn_q is F16 rank-2 containing hidden_size (relaxed from strict square to support GQA) → `real`; elif attn_q present → `routing-f32` (real name = routing tensor name); else `synthetic-fallback`.
- Routing always uses `routing_tensor` via `checkpoint_gate_scores` (routing.rs) when checkpoint loaded (never synthetic for real loads).
- `extract_named_token_embedding_from_checkpoint` (checkpoint.rs) supports dequant for Q8_0/Q5_K (and F32/F16); IQ3_S errors for embeddings.
- Public metadata exposes `preferred_gpu_synapse_tensor_name`, `real_gpu_synapse_tensor_name`, `synapse_source` for SAAQ experiment / Surrogate_Viz consumers to choose dequant vs. synthetic path and load the right tensor (f16 path or f32 routing path).
- SAAQ artifacts (external): calibration runs consume the router to emit artifacts for viz; see labels on related issues for campaign.

## Scope / Boundaries

This crate **owns**:

- Row-major `f32` `Tensor` and CPU tensor ops (matmul, batched matmul, causal
  mask, softmax, layer norm, RMSNorm) under a documented finite-value policy.
- Decoder-only transformer building blocks (attention, block, `TransformerLM`).
- MoE routing math — gate scores, softmax, top-k selection, L2 normalization,
  embedding resampling — and the simulation routing modes.
- Model-family adapters and tensor selection (`Olmoe`, `Qwen3Moe`, `Gemma4`,
  `DeepSeek2`, `LlamaMoe`), including synapse-source resolution.
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
- SNN neuron dynamics ([`neuromod`](https://github.com/rmems/neuromod))
  and ANN→SNN orchestration
  ([`hybrid-fusion`](https://github.com/rmems/hybrid-fusion)).
- Tokenization and automatic differentiation (see [Non-goals](#non-goals)).

**Allowed dependencies:** the current small set — `serde`, `serde_json`,
`thiserror`, `rand`, `rayon`, `memmap2`, `half`, plus optional `sentry` — and,
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
| Routing math, top-k, family adapters | `cortex-tensor` | `src/moe/routing.rs`, `src/moe/adapter.rs` |

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

## Quick start

```rust
use cortex_tensor::tensor::ops::matmul;
use cortex_tensor::Tensor;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
let c = matmul(&a, &b);
assert_eq!(c.shape(), &[2, 2]);
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

**LayerNorm / RMSNorm** reject non-positive or non-finite `eps` and a zero-width last axis through `try_layer_norm` / `try_rms_norm` (`CortexError::InvalidEpsilon`, `CortexError::ZeroWidthAxis`). A non-finite data row becomes an all-NaN output row. Finite data with finite affine parameters stay finite; values that overflow `f32` saturate to `±f32::MAX`. The existing `layer_norm` / `rms_norm` wrappers panic on those same rejections.

**L2 routing normalize** fills NaN on any non-finite input, leaves vectors at or below `L2_NORM_FLOOR` unchanged, and otherwise divides by the `f64` Euclidean norm.

Kernels are sequential, so a given input bit pattern is deterministic across debug/release and supported platforms.

Building a transformer block:

```rust
use cortex_tensor::transformer::{MultiHeadAttention, TransformerBlock};

let attn = MultiHeadAttention::new(/* dim */ 512, /* num_heads */ 8);
let block = TransformerBlock::new(/* dim */ 512, /* num_heads */ 8, /* mlp_dim */ 2048);
```

Loading a family-aware MoE GGUF and running the router:

```rust
use cortex_tensor::moe::{MoeRouter, RoutingMode};

fn main() -> cortex_tensor::Result<()> {
    let mut router = MoeRouter::load_with_mode(
        "path/to/model.gguf",
        /* num_experts */ 0, // 0 → take count from checkpoint metadata
        /* top_k */ 2,
        RoutingMode::DenseSim,
    )?;
    let embedding = vec![0.0f32; cortex_tensor::types::EMBEDDING_DIM]; // or extract_token_embedding
    let out = router.forward(&embedding)?;
    // out.selected_experts, out.expert_weights, out.hidden
    let _ = out;
    Ok(())
}
```

## Optional Sentry monitoring

Enable with the `sentry` feature (uses sentry-rust 0.48):

```toml
[dependencies]
cortex-tensor = { git = "...", features = ["sentry"] }
```

Init guard example (call early in main or lib init; keep guard alive for duration of process).
Note: the consuming crate/binary must explicitly enable the `sentry` feature on its `cortex-tensor` dependency
(transitive features do not auto-activate). Then use the re-exported path (or add `sentry` as direct dep):

```rust
#[cfg(feature = "sentry")]
let _sentry_guard = cortex_tensor::sentry::init((
    "https://<key>@sentry.io/<project>",
    cortex_tensor::sentry::ClientOptions {
        release: cortex_tensor::sentry::release_name!(),
        environment: Some("production".into()),
        ..Default::default()
    },
));

// Your app code; errors auto captured when panics or cortex_tensor::sentry::capture_message etc used.
// (The re-export brings the full sentry crate API under the feature gate.)
```

When the feature is off the re-export is not present (guarded).

## Non-goals

- No GPU backend. Ever. If you need CUDA, consume this crate's `Tensor` into your own kernels.
- No automatic differentiation. This is an inference and forward-pass library.
- No tokenizer. Pair it with `tokenizers` or `llama.cpp`'s tokenizer of choice.
- No SNN / neuromorphic logic. Those live in upstream projects.

## Status

Extracted and compiling cleanly under Rust edition 2024. Public API is subject to change until `0.1.0` is tagged. Tests and benchmarks are incoming.

## License

This project is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
- MIT license ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))

at your option.
