# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Documented finite / non-finite policy for tensor math and MoE routing helpers (`tensor::finite`).
- Fallible `try_layer_norm` / `try_rms_norm` / `Tensor::try_softmax_last` with typed `InvalidEpsilon` and `ZeroWidthAxis` errors.
- CPU routing-tensor orientation check (one dim must equal `hidden_size`) and token-embedding shape/type validation for F32/F16/Q8_0/Q5_K. Does not add Q6_K/IQ3_* dequant.

### Changed

- Softmax (tensor last-axis, attention, MoE routing) is max-subtracted with explicit NaN, `+Inf` split, and all-`-Inf` uniform behavior; partition sums use `f64`, then an `f32` residual so rows of length `≤ 4096` stay within `SOFTMAX_SUM_TOLERANCE`.
- LayerNorm, RMSNorm, and L2 routing normalize accumulate moments in `f64` so large finite magnitudes stay finite; affine overflow saturates to `±f32::MAX`.
- Retargeted Safetensors provider docs from the dropped `safetensors-parser` crate to `engram-parser`'s off-by-default `safetensors` feature (#32). Parser/dtype freeze now points at consume follow-up #47 (blocked on engram-parser#45) instead of closed engram-parser#7.
- Switched license from GPL-3.0 to dual MIT/Apache-2.0 for broader adoption and compatibility with other projects in the Limen-Neural organization.

### Removed

- Qodana Cloud scan (`qodana-rust` + `QODANA_TOKEN_128211718`) after membership expiry. Deleted `qodana.yaml` and `.github/workflows/qodana_code_quality.yml`.

## [0.1.0] - 2026-06-25

### Added

- Initial release of cortex-tensor as standalone crate (extracted from corinth-canal).
- Tensor, ops, transformer, and MoE (GGUF) modules.
- Optional Sentry integration feature.
