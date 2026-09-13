# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- Qodana Cloud scan (`qodana-rust` + `QODANA_TOKEN_128211718`) after membership expiry. Deleted `qodana.yaml` and `.github/workflows/qodana_code_quality.yml`.

### Changed

- Switched license from GPL-3.0 to dual MIT/Apache-2.0 for broader adoption and compatibility with other projects in the Limen-Neural organization.

## [0.1.0] - 2026-06-25

### Added

- Initial release of cortex-tensor as standalone crate (extracted from corinth-canal).
- Tensor, ops, transformer, and MoE (GGUF) modules.
- Optional Sentry integration feature.
