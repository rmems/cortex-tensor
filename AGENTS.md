# AGENTS.md

Guidance for coding agents (Amp, Codex, Cursor, Claude Code, and others) working in this repository.

## Purpose

`cortex-tensor` is a backend-neutral ANN stage layer (`AnnStage`, `AnnTopology`, `AnnExecutor`) plus a
deterministic CPU `f32` reference backend: a contiguous row-major `Vec<f32>` tensor, a reference transformer,
and dense MoE routing (see `README.md`). It has no CUDA, no Julia FFI and no mandatory framework dependency.
Candle and Burn are optional adapters behind features. Checkpoint bytes come from `engram-parser` 0.3.0; this crate
binds them into MoE routing. Placement and orchestration belong to `hybrid-fusion`.
The design goals in the README (zero GPU coupling, no required ML framework, small dependency set)
are constraints: don't add GPU code or make a heavy ML framework a required dependency.

## Layout

| Path | Contents |
|------|----------|
| `src/tensor/` | Row-major `Tensor`, ops (matmul, softmax, norms), finite-value policy |
| `src/transformer/` | Attention, transformer block, decoder-only LM |
| `src/moe/` | MoE router, engram-parser checkpoint binding, adapters, routing |
| `src/snn/` | Optional SNN backend/encoder adapters |
| `src/adapter.rs` | External-adapter capability contract; `candle` executor and `burn` policy stub live behind features |
| `src/stage.rs`, `src/reference_json.rs`, `src/types.rs` (incl. `RoutingMode`), `src/error.rs` | Stage execution, reference serde, shared types, errors |
| `tests/` (+ `tests/fixtures/`) | Integration tests (public API compatibility, reference serde, stage execution) |

## Toolchain

- Rust **1.99.0** (`rust-toolchain.toml`; `rust-version` is the lower MSRV floor, 1.97.1, checked by the `msrv` CI job), edition 2024. CI installs exactly 1.99.0
  with rustfmt and clippy. `.agents/setup` installs the same pin.
- `Cargo.lock` is gitignored (library crate), so don't pass `--locked` to build/test.
- Features: `neuromod` (SNN backend adapter), `axon-encoder` (SNN encoder adapter), `candle` (CPU `f32` `AnnExecutor`), and `burn` (policy stub, no `burn` crate), all off by default. The `candle` feature must not enable CUDA, MKL, or Metal.
- No GPU or system packages needed.

## Commands (from `.github/workflows/rust.yml`)

```bash
cargo build --verbose --all-features
cargo test --verbose --all-features
cargo test --verbose
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

Coverage in a fresh checkout (leave `--locked` off; `Cargo.lock` is not committed):

```bash
cargo llvm-cov --lib --no-default-features --lcov --output-path lcov.info
```

CI adds `--locked` to that same command, but only after the earlier `cargo build` has generated a
`Cargo.lock` in the runner. Don't copy `--locked` onto a checkout that has no lockfile yet.

## Conventions visible in the repo

- Every Rust source file starts with an SPDX license identifier header. Keep it on new files.
- `CHANGELOG.md` is maintained; breaking API changes are marked `!` in commit subjects
  (e.g. `refactor!: ...`, `api(moe)!: ...`).
- Commit subjects follow Conventional Commits with scopes (`feat(stage):`, `ci(actions):`) and
  often a Linear ID (`RM-1822`) plus the PR number.
- `.coderabbit.yaml` configures automated review.
