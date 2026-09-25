# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New `snn` module: backend-neutral ANN↔SNN execution/interchange contract (`SnnBackend`, `SnnEncoder`/`SnnDecoder`, `SnnStepOutput`, `SnnCapabilities`), default `RateEncoder`/`SpikeCountDecoder`, and `SpikingMoeRouter` composing `MoeRouter` gate scores with an external SNN backend (RM-1824).
- Optional `neuromod` cargo feature adding `snn::NeuromodNetwork` — `neuromod::SpikingNetwork` behind `SnnBackend`. Frozen evaluation delegates to `step` until a neuromod release ships `step_frozen`; `capabilities().frozen_evaluation` reports `false` meanwhile.
- `MoeRouter::top_k()` accessor and `CortexError::{SnnBackend, SnnChannelMismatch, SnnNan}` variants.
- Optional `axon-encoder` cargo feature adding `snn::AxonEncoder<E: axon_encoder::Encoder>` — any axon-encoder encoder behind `snn::SnnEncoder`; spikes net to ±1.0 per channel per tick (RM-1824 follow-up).

### Changed

- **Breaking:** `RoutingMode::default()` is now `DenseSim` (was `SpikingSim`) (RM-1824).
- **Breaking:** `SnnEncoder::encode` takes `&mut self` (real encoders are stateful; one call is one backend tick).

### Removed

- **Breaking:** `RoutingMode::SpikingSim` and the embedded SNN simulation in `MoeRouter` — `expert_membranes`, `hidden_membranes`, hard-coded `threshold`/`decay`, `reset_state()`, and `spiking_moe_routing` (RM-1824). Spiking routing lives in `snn::SpikingMoeRouter` over a crate-backed `SnnBackend`.
- **Breaking:** SAAQ/GPU synapse-source policy — `RouterMetadata.preferred_gpu_synapse_tensor_name`, `RouterMetadata.synapse_source`, `MoeRouter::preferred_gpu_synapse_tensor_name()`, `real_gpu_synapse_tensor_name()`, `synapse_source()`, and the crate-internal `synapse_weights_f16` helper (RM-1824). That policy moves downstream to `corinth-canal` / `grok-ozempic`; `synapse_source` manifest labels should be emitted by those pipelines (consumers such as `Surrogate_Viz.jl` keep reading existing artifacts).

### Fixed

- `MoeRouter::extract_token_embedding` returns checkpoint-native `hidden_size` without implicit resample to `EMBEDDING_DIM` or L2 normalization (RM-1519). Use `ExtractTokenOptions::for_projector_forward()` when feeding `MoeRouter::forward`.
- MoE top-k routing now uses a total order: finite scores descending, ascending expert ID on ties, typed NaN rejection, and explicit ±Inf ranking (RM-1355). Selection ranks raw gate/membrane scores before softmax so non-finite logits cannot be masked into uniform weights.

### Added

- Documented finite / non-finite policy for tensor math and MoE routing helpers (`tensor::finite`), including integration notes (public `try_*` vs crate-internal row kernels, `f32` sum residual, causal-attention all-`-Inf` row semantics, MoE vs RM-1355) (RM-1354).
- `Tensor::row_major_strides()` / `try_row_major_strides()` recompute row-major strides on demand from the shape, and `Tensor::reshape_in_place()` reinterprets a contiguous tensor under a new shape without copying (leaving the tensor unchanged on error) (RM-1490).
- Fallible `Tensor::try_softmax_last` / `ops::try_softmax` for rank validation on softmax.
- Fallible, overflow-safe tensor construction (`Tensor::try_from_vec`) and checked ops (`try_matmul`, `try_batched_matmul`, `try_embedding`, `try_layer_norm`, `try_rms_norm`) that validate rank, shape, and size before allocating (RM-1353).
- Structured `CortexError` variants for rank mismatch, axis dimension mismatch, out-of-vocabulary token ids, invalid epsilon, zero-width normalization, and `usize` size overflow.
- CPU routing-tensor orientation check (one dim must equal `hidden_size`) and token-embedding shape/type validation for F32/F16/Q8_0/Q5_K. Does not add Q6_K/IQ3_* dequant.

### Changed

- MSRV / toolchain target: Rust 1.98.1 (`rust-version` in `Cargo.toml`, `rust-toolchain.toml`).
- Softmax (tensor last-axis, attention, MoE routing) is max-subtracted with explicit NaN, `+Inf` split, and all-`-Inf` uniform behavior; partition sums use `f64`, then an `f32` residual so rows of length `≤ 4096` stay within `SOFTMAX_SUM_TOLERANCE` (RM-1354).
- LayerNorm, RMSNorm, and L2 routing normalize accumulate moments in `f64` so large finite magnitudes stay finite; affine overflow saturates to `±f32::MAX` (RM-1354).
- Panic-style constructors and ops (`from_vec`, `matmul`, `batched_matmul`, `embedding`, `layer_norm`, `rms_norm`) are now compatibility wrappers around the fallible APIs. Prefer `try_*` in new code; wrappers stay until a separate SemVer decision.
- Retargeted Safetensors provider docs from the dropped `safetensors-parser` crate to `engram-parser`'s off-by-default `safetensors` feature (#32). Parser/dtype freeze now points at consume follow-up #47 (blocked on engram-parser#45) instead of closed engram-parser#7.
- Switched license from GPL-3.0 to dual MIT/Apache-2.0 for broader adoption and compatibility with other projects in the Limen-Neural organization.

### Removed

- **Breaking:** the `strides` field and `Tensor::strides()` accessor (RM-1490). No kernel ever indexed with strides — the layout is contiguous row-major only — so the stored/exposed strides were misleading. Recompute them from the shape with `row_major_strides()`. Serialized tensors now omit `strides`; a legacy `strides` field in older JSON is accepted and ignored on deserialize.
- **Breaking:** the public `ModelFamily` enum and its surface (RM-1834): `ModelFamily` (incl. `Qwen3Moe`/`Gemma4`/`DeepSeek2`/`LlamaMoe`/`ReferenceMoe` variants and `slug()`), `RouterMetadata.family`, `MoeRouter::family()`, `MoeRouter::load_with_family_and_mode`, and the `family_override` parameter on `MoeRouter::probe_model`. GGUF `general.architecture` stays exposed as an opaque string via `RouterMetadata.architecture` / `MoeRouter::architecture()`.
- Optional Sentry integration (feature, dependency, and re-export). Wire monitoring at the application layer.
- Qodana Cloud scan (`qodana-rust` + `QODANA_TOKEN_128211718`) after membership expiry. Deleted `qodana.yaml` and `.github/workflows/qodana_code_quality.yml`.

## [0.1.0] - 2026-06-25

### Added

- Initial release of cortex-tensor as standalone crate (extracted from corinth-canal).
- Tensor, ops, transformer, and MoE (GGUF) modules.
