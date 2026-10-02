# AGENTS.md

Guidance for coding agents (Amp, Codex, Cursor, Claude Code, and others) working in this repository.

## Purpose

`cortex-tensor` is a pure-Rust library of tensor, transformer, and Mixture-of-Experts building blocks
(see `README.md`). It has no CUDA, no Julia FFI and no framework dependency: the tensor type is a
contiguous row-major `Vec<f32>` plus a shape. It includes a GGUF checkpoint bridge for MoE routing.
The design goals in the README (zero GPU coupling, zero framework dependency, small dependency set)
are constraints: don't add GPU code or heavy ML frameworks.

## Layout

| Path | Contents |
|------|----------|
| `src/tensor/` | Row-major `Tensor`, ops (matmul, softmax, norms), finite-value policy |
| `src/transformer/` | Attention, transformer block, decoder-only LM |
| `src/moe/` | MoE router, GGUF checkpoint parsing/dequant, adapters, routing |
| `src/snn/` | Optional SNN backend/encoder adapters |
| `src/stage.rs`, `src/reference_json.rs`, `src/types.rs`, `src/error.rs` | Stage execution, reference serde, shared types, errors |
| `tests/` (+ `tests/fixtures/`) | Integration tests (public API compatibility, reference serde, stage execution) |

## Toolchain

- Rust **1.98.1** (`rust-toolchain.toml`, `rust-version`), edition 2024. CI installs exactly 1.98.1
  with rustfmt and clippy.
- `Cargo.lock` is gitignored (library crate), so don't pass `--locked` to build/test.
- Features: `neuromod` (SNN backend adapter) and `axon-encoder` (SNN encoder adapter), both off by default.
- No GPU or system packages needed.

## Commands (from `.github/workflows/rust.yml`)

```bash
cargo build --verbose --all-features
cargo test --verbose --all-features
cargo test --verbose
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

Coverage, as CI runs it after the build and test steps:

```bash
cargo llvm-cov --lib --no-default-features --lcov --output-path lcov.info
```

CI passes `--locked` to that step only because the earlier `cargo build` has already generated a
`Cargo.lock` in the runner. In a fresh checkout, leave `--locked` off (or run
`cargo generate-lockfile` first) because the lockfile isn't committed.

## Conventions visible in the repo

- Every Rust source file starts with an SPDX license identifier header. Keep it on new files.
- `CHANGELOG.md` is maintained; breaking API changes are marked `!` in commit subjects
  (e.g. `refactor!: ...`, `api(moe)!: ...`).
- Commit subjects follow Conventional Commits with scopes (`feat(stage):`, `ci(actions):`) and
  often a Linear ID (`RM-1822`) plus the PR number.
- `.coderabbit.yaml` configures automated review.
