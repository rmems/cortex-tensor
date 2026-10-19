# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `MultiHeadAttention` no longer leaks future tokens when every visible score is `-Inf` (e.g. `-1e40` overflow). Softmax now runs over the causal prefix `0..=r` only and future slots are exact zeros; the additive causal mask is no longer materialised. Outputs for rows with at least one finite visible score and finite future scores are unchanged; a non-finite future score no longer affects earlier rows' attention weights. The `attn × V` product now also accumulates only the visible prefix, so a NaN/`±Inf` value row at a future position no longer poisons earlier tokens via `0.0 * NaN`; finite outputs are bit-identical (issue 94).

### Changed

- **Breaking (checkpoint binding):** `MoeRouter` loads GGUF and Safetensors through published `engram-parser` 0.3.0 (`default-features = false`, `features = ["safetensors"]`; `mmap` is not enabled). Tensor shapes are that crate's outermost-first order, and gate weights are row-major on those shapes (GGUF `ne[0]` is the contiguous axis). Q8_0, Q5_K, and Q6_K rows are dequantized by engram-parser, including Q5_K's low-nibble-then-high-nibble group order. IQ3_S and IQ3_XXS stay explicit errors (engram-parser 0.3.0 does not decode those wire types). A checkpoint with no `general.file_type` reports quantization `unknown`. Validated against engram-parser 0.3.0 (RM-1265 / issue 47).
- `MappedCheckpoint` is a temporary router binding shim. Remove it when callers take `engram_parser::Checkpoint` directly. Malformed input returns the engram-parser error; there is no in-tree parser fallback.
- A loaded router decodes the F32 gate once and keeps the token-embedding payload, then releases the rest of the checkpoint. `mmap` stays off. A stored `expert_count` of 0 is rejected. Token lookup continues past an unsupported `token_embd.weight` and also accepts `embed_tokens.weight` names. Gate suffix search orders `layers.N`, `layer.N`, and `block(s).N` numerically, same as `blk.N` (RM-1265).

### Removed

- In-tree GGUF header/KV/tensor-directory parsing, copy-on-write mmap, `src/moe/gguf.rs`, and `src/moe/dequant.rs`. The `memmap2` and `half` dependencies are gone (RM-1265).

### Changed

- Crate identity is a backend-neutral ANN stage layer plus a deterministic `Vec<f32>` reference backend. README and Cargo description say so. Checkpoint parsing stays in engram-parser, orchestration stays in hybrid-fusion, and the closed decision not to merge hybrid-fusion (issue 35) stands. No new framework-generic numerical API (issue 56).
- Parser-boundary docs point at the engram-parser 0.3.0 consume in this unreleased section (RM-1265). The older plan to wrap `parse_checkpoint_layout` around engram-parser 0.2.0 stays superseded.

### Added

- Optional `candle` cargo feature adding `adapter::CandleExecutor`, a CPU `f32` `AnnExecutor` (backend `"candle"`, stateless) for embedding, causal attention, layer/RMS norm, dense GELU MLP, gated SwiGLU MLP, add, and readout. Stage bodies call Candle's `Embedding`, `Linear`, `LayerNorm`, `RmsNorm`, GELU, `swiglu`, and CPU flash-attention rather than reimplementing those kernels. Row-major `f32` hidden state crosses the backend only through `import_hidden_state` / `export_hidden_state` (and reference-tensor wrappers); weights are transposed into Candle layout once at bind time. MoE and custom stages return `UnsupportedStage`. CUDA, MKL, and Metal stay disabled (RM-1828).
- External adapter contract (`adapter` module, re-exported from `stage`): `ExternalAdapter` / `ExternalAdapterMarker`, `AdapterCapabilities`, `BackendLimitations`, `DeviceClass`, `HiddenStateBuffer`, and `NegotiationDocument`. Core traits do not name Candle or Burn types. `ExternalAdapterMarker` includes `import_hidden` and `export_hidden`. `ReferenceExecutor` implements that contract and copies hidden state through `HiddenStateBuffer`. `AdapterCapabilities::require_supported` rejects stage kinds, dtypes, and devices the adapter did not advertise, including requests absent from the explicit refusal lists. The optional `burn` feature is a structured policy stub and does not depend on the `burn` crate; burn 0.21's rustc 1.92 requirement is a minimum and is compatible with the 1.99.0 pin. `CandleExecutor` is not an `ExternalAdapter` hidden-state conversion. **Breaking:** the new `UnsupportedOperation` variant requires downstream exhaustive `CortexError` matches to add an arm (#58).
- New `stage` module: backend-neutral ANN stage-execution contracts (`DType`, `TensorMeta`, `StageTensor`, `StageKind`/`StageKindTag` with separate dense/gated MLP and layer/RMS norm tags, `NormKind`, `MlpKind`, `StageSource`, `AnnStage`, validated `StageId` dotted-path newtype, validated `AnnTopology` with sub-span dependency listing, `AnnCapabilities`, `StageInput`, the generic `AnnExecutor` trait whose contract never names a concrete tensor, and the `run_topology` composition driver over `ExternalBindings`). Descriptors carry no serde; a device field and tensor wire format are intentionally deferred to RM-1827 (RM-1822).
- `stage::ReferenceExecutor` — a dense CPU `f32` `AnnExecutor` (backend `"reference"`, stateless, supporting embedding, attention, layer/RMS norm, dense MLP, add, and readout stages) that composes a `TransformerLM` as a stage graph by reusing the existing reference kernels, so a topology built by `ReferenceExecutor::from_transformer` reproduces `TransformerLM::try_forward` bit for bit. Includes borrowed `ReferenceStageParams` and `from_transformer`, which validates the model once and emits the block-ordered topology (RM-1822).
- **Possible break for exhaustive `CortexError` matches:** new stage-boundary variants `InvalidStageId`, `InvalidTopology`, `UnsupportedStage`, `MissingStageParams`, `StageInputArity`, `StageInputKind`, `StageDTypeMismatch`, `MissingStageBinding`, and `StageFailed` (RM-1822).
- New `snn` module: backend-neutral ANN↔SNN execution/interchange contract (`SnnBackend`, `SnnEncoder`/`SnnDecoder`, `SnnStepOutput`, `SnnCapabilities`), default `RateEncoder`/`SpikeCountDecoder`, and `SpikingMoeRouter` composing `MoeRouter` gate scores with an external SNN backend (RM-1824).
- Optional `neuromod` cargo feature adding `snn::NeuromodNetwork` — `neuromod::SpikingNetwork` behind `SnnBackend`. `step_frozen` maps onto neuromod 0.7's `SpikingNetwork::step_frozen`, and `with_seed` / `set_seed` / `seed` route both paths through `step_with_rng` / `step_frozen_with_rng` for bit-for-bit replay (#92).
- `MoeRouter::top_k()` accessor and `CortexError::{SnnBackend, SnnChannelMismatch, SnnNan, SnnSpikeOutOfRange}` variants.
- Optional `axon-encoder` cargo feature adding `snn::AxonEncoder<E: axon_encoder::Encoder>` — any axon-encoder encoder behind `snn::SnnEncoder`; spikes net to ±1.0 per channel per tick (RM-1824 follow-up).

### Changed

- **Breaking:** `RoutingMode` serde now emits the canonical snake_case slug wire form (`stub_uniform`, `dense_sim`) instead of PascalCase. Legacy PascalCase payloads still deserialize through explicit aliases (RM-1518).
- **Breaking:** `RoutingMode::default()` is now `DenseSim` (was `SpikingSim`) (RM-1824).
- **Breaking:** `SnnEncoder::encode` takes `&mut self` and returns `Result<Vec<f32>>` (real encoders are stateful — one call is one backend tick, not failure-atomic — and encodings that cannot fit the channel space error instead of silently truncating). `SnnEncoder::reset` (no-op default; `AxonEncoder` delegates to `Encoder::reset`) is invoked by `SpikingMoeRouter::reset` so encoder and backend epochs reset together.

### Removed

- **Breaking:** `RoutingMode::SpikingSim` and the embedded SNN simulation in `MoeRouter` — `expert_membranes`, `hidden_membranes`, hard-coded `threshold`/`decay`, `reset_state()`, and `spiking_moe_routing` (RM-1824). Spiking routing lives in `snn::SpikingMoeRouter` over a crate-backed `SnnBackend`.
- **Breaking:** SAAQ/GPU synapse-source policy — `RouterMetadata.preferred_gpu_synapse_tensor_name`, `RouterMetadata.synapse_source`, `MoeRouter::preferred_gpu_synapse_tensor_name()`, `real_gpu_synapse_tensor_name()`, `synapse_source()`, and the crate-internal `synapse_weights_f16` helper (RM-1824). That policy moves downstream to `corinth-canal` / `grok-ozempic`; `synapse_source` manifest labels should be emitted by those pipelines (consumers such as `Surrogate_Viz.jl` keep reading existing artifacts).
- Unused `rayon` dependency: removed from `Cargo.toml` and from the README allowed-dependencies list. No kernel referenced it (kernels are sequential), so downstream builds no longer compile it; re-add only alongside a deterministic parallel kernel (RM-2099).

### Fixed

- `Tensor::relu` and `Tensor::max_val` no longer discard NaN through `f32::max`: NaN inputs now propagate (`max_val` of any NaN-bearing tensor is NaN, so it agrees with `argmax` for positive NaN), and the empty `max_val` result (`-Inf`) is documented. `gelu`, `silu` and `fast_sigmoid` take an explicit `±Inf` branch and return the mathematical limits (`+Inf`/`0`, `+Inf`/`0`, `±1`) instead of NaN from `Inf − Inf`, `Inf × 0` or `Inf / Inf`. `sum` and `mean` accumulate in Neumaier-compensated `f64` and cast once, so the `f32` running sum no longer stalls at `2^24` (`ones(2^24 + 2).sum()` was `16777216`), overflows on a finite total (`[MAX, MAX, −MAX].sum()` was `+Inf`, now `MAX`) or cancels a small term between two large ones (`[MAX, 1e20, −MAX].sum()` is `1e20`); every existing reference golden, including `(sum, mean) = (21, 3.5)`, is unchanged. The `tensor::finite` policy table gains rows for these activations and reductions (issue 95).
- `snn::NeuromodNetwork::step_frozen` calls neuromod 0.7's `SpikingNetwork::step_frozen` instead of delegating to `step`, so `SpikingMoeRouter::forward_frozen` no longer advances STDP weights or eligibility traces during held-out evaluation; `capabilities().frozen_evaluation` is now `true`. The `caller_rng` capability is truthful: it was hard-coded `true` while every step drew from neuromod's thread-local RNG. `NeuromodNetwork::with_seed(u64)` / `set_seed(Option<u64>)` install a caller-owned `StdRng` that `step` / `step_frozen` pass to `step_with_rng` / `step_frozen_with_rng` (`reset` rewinds the generator to the seed; neuromod's `reset` keeps learned weights and thresholds, so a post-reset epoch replays the previous one only when no learning step changed them in between, e.g. frozen-only stepping or a clone taken at the previous reset), and `caller_rng` reports `true` only while a seed is installed (#92).
- `snn::NeuromodNetwork` no longer accepts channels that can never spike. `neuromod::SpikingNetwork::step` reports LIF spikes only, so `new` / `from_network` now require one LIF neuron per channel (`neurons.len() == num_channels`; the Izhikevich bank is auxiliary and no longer counts toward the width) and reject a LIF neuron that carries a non-finite weight on an integrated channel or has no strictly positive weight in `weights[..weights.len().min(num_channels)]` — the only range the engine integrates (channels past a short vector are skipped, so a sparse checkpoint with one positive present weight is accepted), against `|stimulus|`, and one its L1 renormalization would otherwise leave blank forever. `new` seeds identity wiring — LIF neuron *i* listens to channel *i* at the configured maximum weight — instead of the all-zero weights `with_dimensions` produces, so a freshly built backend spikes under stimulus rather than degrading `SpikingMoeRouter` to a `DenseSim` ranking with `hidden == 0` (#91).
- Optional `candle` builds after crates.io yanked `zip` 8.6.0 (2026-10-16). `candle-core` 0.11 requires `zip ^8.6.0`, which cannot select the unyanked 9.0.3 release. `[patch.crates-io]` points at `compat/zip`, a facade of zip 9.0.3 that restores `ZipFile::name() -> &str` and `file_names()` items candle 0.11 calls. The patch applies when this repository is the Cargo root.
- Tensor constructors and JSON deserialization validate shape products and row-major stride arithmetic with checked multiplication, rejecting overflowing extents in release instead of wrapping to `numel == 0` (RM-1488). Documented rank-0 scalar vs zero-axis contracts and `mean()` returning `NaN` on empty tensors.
- `MoeRouter::extract_token_embedding` returns checkpoint-native `hidden_size` without implicit resample to `EMBEDDING_DIM` or L2 normalization (RM-1519). Use `ExtractTokenOptions::for_projector_forward()` when feeding `MoeRouter::forward`.
- MoE top-k routing now uses a total order: finite scores descending, ascending expert ID on ties, typed NaN rejection, and explicit ±Inf ranking (RM-1355). Selection ranks raw gate/membrane scores before softmax so non-finite logits cannot be masked into uniform weights.

### Added

- Documented finite / non-finite policy for tensor math and MoE routing helpers (`tensor::finite`), including integration notes (public `try_*` vs crate-internal row kernels, `f32` sum residual, causal-attention all-`-Inf` row semantics, MoE vs RM-1355) (RM-1354).
- `Tensor::row_major_strides()` / `try_row_major_strides()` recompute row-major strides on demand from the shape, and `Tensor::reshape_in_place()` reinterprets a contiguous tensor under a new shape without copying (leaving the tensor unchanged on error) (RM-1490).
- Fallible `Tensor::try_softmax_last` / `ops::try_softmax` for rank validation on softmax.
- Fallible, overflow-safe tensor construction (`Tensor::try_from_vec`) and checked ops (`try_matmul`, `try_batched_matmul`, `try_embedding`, `try_layer_norm`, `try_rms_norm`) that validate rank, shape, and size before allocating (RM-1353).
- Structured `CortexError` variants for rank mismatch, axis dimension mismatch, out-of-vocabulary token ids, invalid epsilon, zero-width normalization, and `usize` size overflow.
- CPU routing-tensor orientation check (one dim must equal `hidden_size`) and token-embedding shape/type validation for F32/F16/Q8_0/Q5_K/Q6_K. IQ3 wire types are not bound.

### Changed

- MSRV / toolchain target: Rust 1.99.0 (`rust-version` in `Cargo.toml`, `rust-toolchain.toml`, CI, and `.agents/setup`).
- Crate package excludes agent, CI, coverage, and `rust-toolchain.toml` paths. Sources, tests, licenses, README, and CHANGELOG still ship.
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
