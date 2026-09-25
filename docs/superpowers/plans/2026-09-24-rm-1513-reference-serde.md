# RM-1513 Reference Serde Implementation Plan

**Goal:** Make reference backend JSON round-trip only for valid, finite, versioned tensor and transformer values.

**Architecture:** Tensor deserialization rebuilds through `try_from_vec`; serialization rejects non-finite values. Transformer payloads use versioned wire forms and validate every weight shape against their configuration. JSON remains specific to this deterministic reference backend.

**Tech Stack:** Rust 2024, serde, serde_json, cargo test, clippy.

**Spec:** [RM-1513](https://linear.app/rpd-34/issue/RM-1513/reference-backend-serde-validate-tensor-layout-and-keep-serialization)

## Global Constraints

- Reject unknown fields and unknown schema versions.
- Reject malformed tensor shape and any legacy `strides` field.
- Preserve signed zero and support empty and zero-axis tensors.
- Do not change numerical kernels or establish an external tensor interchange format.

## Task 1: Tensor wire boundary

**Files:** `src/tensor/mod.rs`, `tests/reference_serde.rs`, old tests in `src/tensor/ops.rs`.

- [x] Add golden JSON and hostile input tests for tensor layout, version, unknown fields, finite values and signed zero.
- [x] Run focused test and observe the failure.
- [x] Deserialize through `Tensor::try_from_vec`; serialize only finite data with `schema_version: 1`.
- [x] Update obsolete tests that constructed malformed tensors through serde; run focused tests.

## Task 2: Transformer wire boundary

**Files:** `src/transformer/model.rs`, `src/transformer/block.rs`, `src/transformer/attention.rs`, `tests/reference_serde.rs`.

- [x] Add small-model round-trip and corrupt-shape tests for config, LM, block and attention.
- [x] Run focused tests and observe the failure.
- [x] Add versioned, strict deserialize and shape validation at each transformer boundary.
- [x] Run focused tests.

## Task 3: Verification

- [x] Run `cargo fmt --check`.
- [x] Run `cargo test --all-features`.
- [x] Run `cargo clippy --all-features --all-targets -- -D warnings`.
- [x] Review the diff for wire compatibility and report any remaining limits.
