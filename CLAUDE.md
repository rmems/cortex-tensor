@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Layout, toolchain, feature list, and commit conventions are in `AGENTS.md`
(imported above); do not duplicate them here.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout on Rust 1.99.0 (Linux, CPU only).

| Command | Status |
|---|---|
| `cargo build --verbose --all-features` | verified |
| `cargo test --verbose --all-features` | verified |
| `cargo test --verbose` (default features) | verified |
| `cargo test --test reference_goldens` | verified |
| `cargo fmt --all -- --check` | verified |
| `cargo clippy --all-targets --all-features -- -D warnings` | verified |
| `cargo test --verbose --manifest-path compat/zip/Cargo.toml` | verified |
| `cargo fmt --manifest-path compat/zip/Cargo.toml -- --check` | verified |
| `cargo clippy --manifest-path compat/zip/Cargo.toml --all-targets -- -D warnings` | verified |
| `cargo llvm-cov --lib --no-default-features --lcov --output-path lcov.info` | not run here (`cargo-llvm-cov` not installed); CI adds `--locked` |

CI (`.github/workflows/rust.yml`) also runs reviewdog, Snyk, and Codecov upload. Those need secrets and are not local gates.
Run the fmt and clippy gates before committing; CI fails on either.

## Architecture map

- `src/stage.rs`: backend-neutral stage vocabulary (`AnnStage`, `AnnTopology`, `AnnExecutor`), `run_topology`, and `ReferenceExecutor`.
- `src/adapter.rs` + `src/adapter/`: `ExternalAdapter` capability contract. `candle.rs` (`CandleExecutor`, feature `candle`) and `burn_adapter.rs` (policy stub, feature `burn`).
- `src/tensor/`: row-major `Vec<f32>` `Tensor`, ops, and the finite-value policy in `finite.rs`.
- `src/transformer/`, `src/moe/`: reference transformer and dense MoE routing. `moe/checkpoint.rs` binds `engram-parser` output.
- `src/snn/`: `SnnBackend`/`SnnEncoder`/`SnnDecoder` contract; `neuromod_adapter.rs` and `axon_adapter.rs` sit behind features.
- `src/reference_json.rs`: size-limited loading (`from_slice_with_limit`) and the object-shape check for reference JSON fixtures. The `schema_version: 1` check and `deny_unknown_fields` live in `src/tensor/mod.rs` and `src/transformer/mod.rs`. Goldens live in `tests/fixtures/` (see `tests/fixtures/REFERENCE_GOLDENS.md`).
- `compat/zip/`: local facade crate wired in through `[patch.crates-io]` in `Cargo.toml`. It is not part of the library API.

## Invariants and non-goals (from README and `Cargo.toml`)

- No GPU code, no autodiff, no tokenizer, no embedded SNN runtime. Do not add GPU backends or make Candle/Burn/`tch`/`ort` a required dependency.
- `candle` feature: `candle-core`/`candle-nn` with default features off (no CUDA, MKL, Metal). `burn` feature must not depend on the `burn` crate.
- Public stage contracts (`AnnStage`, `AnnExecutor`) must not name `Tensor` or `Vec<f32>`. Native Candle/Burn types stay out of the `ExternalAdapter` contract.
- Checkpoint parsing belongs to `engram-parser`; ANN/SNN placement belongs to `hybrid-fusion`. Do not add parsers, mmap policy, or orchestration here.
- Reference JSON is a fixture format for the CPU reference backend, not a cross-backend tensor wire format. Unknown versions and fields are rejected.
- Finite-value policy: softmax and norms accumulate in `f64`; NaN scores are rejected in routing (`NanRoutingScore`). Kernels are sequential so results stay deterministic.
- Prefer `try_*` ops over the panic-style wrappers for new code.
- Every `.rs` file starts with `// SPDX-License-Identifier: MIT OR Apache-2.0`.

## Gotchas

- The first all-features build compiles Candle and takes about a minute; later runs are incremental.
- Never run `cargo publish`.
- Edit only what the task needs.
- `.claude/` is gitignored, so keep shared agent guidance in this file or `AGENTS.md`.
