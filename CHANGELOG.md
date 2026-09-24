# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `MoeRouter::extract_token_embedding` returns checkpoint-native `hidden_size` without implicit resample to `EMBEDDING_DIM` or L2 normalization (RM-1519). Use `ExtractTokenOptions::for_projector_forward()` when feeding `MoeRouter::forward`.
- MoE top-k routing now uses a total order: finite scores descending, ascending expert ID on ties, typed NaN rejection, and explicit ±Inf ranking (RM-1355). Selection ranks raw gate/membrane scores before softmax so non-finite logits cannot be masked into uniform weights.

### Added

- Documented finite / non-finite policy for tensor math and MoE routing helpers (`tensor::finite`), including integration notes (public `try_*` vs crate-internal row kernels, `f32` sum residual, causal-attention all-`-Inf` row semantics, MoE vs RM-1355) (RM-1354).
- Fallible `Tensor::try_softmax_last` / `ops::try_softmax` for rank validation on softmax.
- Fallible, overflow-safe tensor construction (`Tensor::try_from_vec`) and checked ops (`try_matmul`, `try_batched_matmul`, `try_embedding`, `try_layer_norm`, `try_rms_norm`) that validate rank, shape, and size before allocating (RM-1353).
- Structured `CortexError` variants for rank mismatch, axis dimension mismatch, out-of-vocabulary token ids, invalid epsilon, zero-width normalization, and `usize` size overflow.
- CPU routing-tensor orientation check (one dim must equal `hidden_size`) and token-embedding shape/type validation for F32/F16/Q8_0/Q5_K. Does not add Q6_K/IQ3_* dequant.

### Changed

- Model-family enum uses `ReferenceMoe` for backend-neutral GGUF MoE checkpoints; unknown architecture strings infer `ReferenceMoe`.
- MSRV / toolchain target: Rust 1.98.1 (`rust-version` in `Cargo.toml`, `rust-toolchain.toml`).
- Softmax (tensor last-axis, attention, MoE routing) is max-subtracted with explicit NaN, `+Inf` split, and all-`-Inf` uniform behavior; partition sums use `f64`, then an `f32` residual so rows of length `≤ 4096` stay within `SOFTMAX_SUM_TOLERANCE` (RM-1354).
- LayerNorm, RMSNorm, and L2 routing normalize accumulate moments in `f64` so large finite magnitudes stay finite; affine overflow saturates to `±f32::MAX` (RM-1354).
- Panic-style constructors and ops (`from_vec`, `matmul`, `batched_matmul`, `embedding`, `layer_norm`, `rms_norm`) are now compatibility wrappers around the fallible APIs. Prefer `try_*` in new code; wrappers stay until a separate SemVer decision.
- Retargeted Safetensors provider docs from the dropped `safetensors-parser` crate to `engram-parser`'s off-by-default `safetensors` feature (#32). Parser/dtype freeze now points at consume follow-up #47 (blocked on engram-parser#45) instead of closed engram-parser#7.
- Switched license from GPL-3.0 to dual MIT/Apache-2.0 for broader adoption and compatibility with other projects in the Limen-Neural organization.

### Removed

- Optional Sentry integration (feature, dependency, and re-export). Wire monitoring at the application layer.
- Qodana Cloud scan (`qodana-rust` + `QODANA_TOKEN_128211718`) after membership expiry. Deleted `qodana.yaml` and `.github/workflows/qodana_code_quality.yml`.

## [0.1.0] - 2026-06-25

### Added

- Initial release of cortex-tensor as standalone crate (extracted from corinth-canal).
- Tensor, ops, transformer, and MoE (GGUF) modules.
