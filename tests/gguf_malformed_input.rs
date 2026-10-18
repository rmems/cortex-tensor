// SPDX-License-Identifier: MIT OR Apache-2.0

//! Malformed-GGUF regression tests over the public `MoeRouter` surface
//! (issues #86 and #87, parent #85).
//!
//! Format parsing lives in `engram-parser` 0.3.0; this crate only binds the
//! decoded tensors. These tests pin the contract the router relies on: a
//! crafted file must come back as `Err`, never as a panic, a process abort,
//! or an out-of-bounds read. Every fixture is assembled in-test from raw
//! bytes so the shapes under test (wrapping sizes, misaligned offsets, deep
//! nesting) are exactly what the GGUF spec forbids or leaves unguarded.

use cortex_tensor::CortexError;
use cortex_tensor::moe::{MoeRouter, RoutingMode};
use std::path::PathBuf;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGUF_VERSION: u32 = 3;
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const VALUE_UINT8: u32 = 0;
const VALUE_UINT32: u32 = 4;
const VALUE_STRING: u32 = 8;
const VALUE_ARRAY: u32 = 9;

const ARCH: &str = "reference_moe";
const HIDDEN: usize = 32;
const EXPERTS: usize = 4;
const VOCAB: usize = 4;

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_string(out: &mut Vec<u8>, value: &str) {
    push_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

/// One metadata KV entry: key, GGUF value type, and the raw value payload.
struct Kv {
    key: &'static str,
    value_type: u32,
    payload: Vec<u8>,
}

fn kv_u32(key: &'static str, value: u32) -> Kv {
    Kv {
        key,
        value_type: VALUE_UINT32,
        payload: value.to_le_bytes().to_vec(),
    }
}

fn kv_string(key: &'static str, value: &str) -> Kv {
    let mut payload = Vec::new();
    push_string(&mut payload, value);
    Kv {
        key,
        value_type: VALUE_STRING,
        payload,
    }
}

/// One tensor-directory entry. `dims` are GGUF wire order (innermost first)
/// and are written verbatim, so a test can declare sizes that overflow.
struct TensorEntry {
    name: &'static str,
    dims: Vec<u64>,
    ggml_type: u32,
    relative_offset: u64,
}

/// Assemble a GGUF v3 file. `data` is the tensor-data section, written after
/// the header is padded to `alignment` (or unpadded when `alignment == 0`).
fn build_gguf(alignment: u32, extra_kv: Vec<Kv>, tensors: &[TensorEntry], data: &[u8]) -> Vec<u8> {
    let mut kvs = vec![
        kv_u32("general.alignment", alignment),
        kv_u32("general.file_type", 0),
        kv_string("general.architecture", ARCH),
        kv_u32("reference_moe.embedding_length", HIDDEN as u32),
        kv_u32("reference_moe.block_count", 1),
        kv_u32("reference_moe.expert_count", EXPERTS as u32),
        kv_u32("reference_moe.expert_used_count", 2),
    ];
    kvs.extend(extra_kv);

    let mut out = Vec::new();
    out.extend_from_slice(GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    push_u64(&mut out, tensors.len() as u64);
    push_u64(&mut out, kvs.len() as u64);
    for kv in &kvs {
        push_string(&mut out, kv.key);
        push_u32(&mut out, kv.value_type);
        out.extend_from_slice(&kv.payload);
    }
    for tensor in tensors {
        push_string(&mut out, tensor.name);
        push_u32(&mut out, tensor.dims.len() as u32);
        for dim in &tensor.dims {
            push_u64(&mut out, *dim);
        }
        push_u32(&mut out, tensor.ggml_type);
        push_u64(&mut out, tensor.relative_offset);
    }
    if alignment > 1 {
        while !out.len().is_multiple_of(alignment as usize) {
            out.push(0);
        }
    }
    out.extend_from_slice(data);
    out
}

fn gate_entry(relative_offset: u64) -> TensorEntry {
    TensorEntry {
        name: "blk.0.ffn_gate_inp.weight",
        dims: vec![HIDDEN as u64, EXPERTS as u64],
        ggml_type: GGML_TYPE_F32,
        relative_offset,
    }
}

fn token_entry(dims: Vec<u64>, ggml_type: u32, relative_offset: u64) -> TensorEntry {
    TensorEntry {
        name: "token_embd.weight",
        dims,
        ggml_type,
        relative_offset,
    }
}

fn f32_payload(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_le_bytes).collect()
}

fn gate_payload() -> Vec<u8> {
    f32_payload((0..HIDDEN * EXPERTS).map(|i| i as f32 * 0.01))
}

fn token_payload() -> Vec<u8> {
    f32_payload((0..VOCAB * HIDDEN).map(|i| i as f32))
}

struct TempGguf(PathBuf);

impl TempGguf {
    fn write(bytes: &[u8], label: &str) -> Self {
        let dir = std::env::temp_dir().join("cortex-tensor-malformed-gguf");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "{label}-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }

    fn as_str(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for TempGguf {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// `MoeRouter` is not `Debug`, so `unwrap_err` cannot be used on a load.
fn assert_load_fails(path: &str) {
    assert!(
        MoeRouter::load(path, 0, 0).is_err(),
        "load of a malformed checkpoint must fail: {path}"
    );
}

/// A well-formed baseline: gate at offset 0, token embedding right after it.
fn well_formed() -> Vec<u8> {
    let gate = gate_payload();
    let mut data = gate.clone();
    data.extend_from_slice(&token_payload());
    build_gguf(
        32,
        Vec::new(),
        &[
            gate_entry(0),
            token_entry(
                vec![HIDDEN as u64, VOCAB as u64],
                GGML_TYPE_F32,
                gate.len() as u64,
            ),
        ],
        &data,
    )
}

#[test]
fn well_formed_fixture_loads_and_extracts() {
    let file = TempGguf::write(&well_formed(), "well-formed");
    let mut router = MoeRouter::load_with_mode(file.as_str(), 0, 0, RoutingMode::DenseSim).unwrap();
    assert_eq!(router.hidden_size(), HIDDEN);
    assert_eq!(router.num_experts(), EXPERTS);
    let row = router.extract_token_embedding(1).unwrap();
    let expected: Vec<f32> = (HIDDEN..2 * HIDDEN).map(|i| i as f32).collect();
    assert_eq!(row, expected);
}

// ---------------------------------------------------------------------------
// Issue #86: offset / size / alignment math
// ---------------------------------------------------------------------------

/// `HIDDEN * (1 << 58)` elements times 4 bytes wraps `usize` to exactly 0 in
/// release (the issue's `[4, 1 << 60]` shape with `hidden_size = 4`). The
/// original in-tree parser then passed its `end > len` check and read past
/// the file (`extract_token_embedding(1 << 28)` segfaulted). Both the probe
/// and the load must refuse the directory entry.
#[test]
fn token_embedding_element_count_overflow_is_an_error() {
    let gate = gate_payload();
    let bytes = build_gguf(
        32,
        Vec::new(),
        &[
            gate_entry(0),
            token_entry(
                vec![HIDDEN as u64, 1 << 58],
                GGML_TYPE_F32,
                gate.len() as u64,
            ),
        ],
        &gate,
    );
    let file = TempGguf::write(&bytes, "f32-size-wrap");

    MoeRouter::probe_model(file.as_str()).unwrap_err();
    assert_load_fails(file.as_str());
}

/// Same wrap through the F16 branch, which used to end in a slice-index panic.
#[test]
fn f16_token_embedding_element_count_overflow_is_an_error() {
    let gate = gate_payload();
    let bytes = build_gguf(
        32,
        Vec::new(),
        &[
            gate_entry(0),
            token_entry(
                vec![HIDDEN as u64, 1 << 58],
                GGML_TYPE_F16,
                gate.len() as u64,
            ),
        ],
        &gate,
    );
    let file = TempGguf::write(&bytes, "f16-size-wrap");

    MoeRouter::probe_model(file.as_str()).unwrap_err();
    assert_load_fails(file.as_str());
}

/// `tensor_data_offset + u64::MAX` used to panic in debug and wrap in release.
#[test]
fn relative_offset_u64_max_is_an_error() {
    let gate = gate_payload();
    let bytes = build_gguf(
        32,
        Vec::new(),
        &[
            gate_entry(u64::MAX),
            token_entry(vec![HIDDEN as u64, VOCAB as u64], GGML_TYPE_F32, 0),
        ],
        &gate,
    );
    let file = TempGguf::write(&bytes, "offset-u64-max");

    MoeRouter::probe_model(file.as_str()).unwrap_err();
    assert_load_fails(file.as_str());
}

/// A tensor whose declared range runs past the end of the file.
#[test]
fn tensor_past_end_of_file_is_an_error() {
    let gate = gate_payload();
    let bytes = build_gguf(
        32,
        Vec::new(),
        &[
            gate_entry(0),
            // Declares VOCAB*HIDDEN f32 values but only the gate is on disk.
            token_entry(
                vec![HIDDEN as u64, VOCAB as u64],
                GGML_TYPE_F32,
                gate.len() as u64,
            ),
        ],
        &gate,
    );
    let file = TempGguf::write(&bytes, "past-eof");

    MoeRouter::probe_model(file.as_str()).unwrap_err();
    assert_load_fails(file.as_str());
}

/// `general.alignment` of 0 and of a non-power-of-two must be rejected the
/// way llama.cpp rejects them (gguf.cpp:632); 0 used to be a divide-by-zero
/// hazard and 3 made every padded offset meaningless.
#[test]
fn zero_or_non_power_of_two_alignment_is_an_error() {
    for alignment in [0u32, 3, 48] {
        let gate = gate_payload();
        let mut data = gate.clone();
        data.extend_from_slice(&token_payload());
        let bytes = build_gguf(
            alignment,
            Vec::new(),
            &[
                gate_entry(0),
                token_entry(
                    vec![HIDDEN as u64, VOCAB as u64],
                    GGML_TYPE_F32,
                    gate.len() as u64,
                ),
            ],
            &data,
        );
        let file = TempGguf::write(&bytes, &format!("alignment-{alignment}"));
        let err = MoeRouter::probe_model(file.as_str()).unwrap_err();
        assert!(
            err.to_string().contains("alignment"),
            "alignment={alignment}: expected an alignment error, got {err}"
        );
        assert_load_fails(file.as_str());
    }
}

/// A gate and token embedding placed at odd byte offsets. The old `f32_tensor`
/// cast the slice to `*const f32` and called `slice::from_raw_parts`, which is
/// UB on a misaligned pointer (debug builds aborted on the precondition
/// check). Payloads are now decoded byte-wise, so a misaligned-but-in-bounds
/// tensor either loads with the correct values or is refused with an error;
/// it never trips an alignment precondition.
#[test]
fn misaligned_tensor_offsets_never_reach_unsafe_alignment_assumptions() {
    let gate = gate_payload();
    let token = token_payload();
    let mut data = vec![0u8; 1];
    data.extend_from_slice(&gate);
    data.extend_from_slice(&[0u8; 2]);
    let token_offset = data.len() as u64;
    data.extend_from_slice(&token);
    let bytes = build_gguf(
        32,
        Vec::new(),
        &[
            gate_entry(1),
            token_entry(
                vec![HIDDEN as u64, VOCAB as u64],
                GGML_TYPE_F32,
                token_offset,
            ),
        ],
        &data,
    );
    let file = TempGguf::write(&bytes, "misaligned-offsets");

    match MoeRouter::load_with_mode(file.as_str(), 0, 0, RoutingMode::DenseSim) {
        Ok(mut router) => {
            let row = router.extract_token_embedding(3).unwrap();
            let expected: Vec<f32> = (3 * HIDDEN..4 * HIDDEN).map(|i| i as f32).collect();
            assert_eq!(
                row, expected,
                "byte-wise decode must not depend on alignment"
            );
            let embedding = vec![1.0f32; cortex_tensor::types::EMBEDDING_DIM];
            let out = router.forward(&embedding).unwrap();
            assert!(out.expert_weights.iter().all(|w| w.is_finite()));
        }
        Err(err) => {
            // A stricter parser may refuse offsets that are not multiples of
            // `general.alignment`. That is an acceptable outcome; UB is not.
            assert!(!err.to_string().is_empty());
        }
    }
}

// ---------------------------------------------------------------------------
// Issue #87: nested metadata arrays
// ---------------------------------------------------------------------------

/// `depth` arrays nested inside one another, each of length 1, the innermost
/// an empty `UINT8` array. Every level costs 12 bytes of file.
fn nested_array_kv(depth: usize) -> Kv {
    let mut payload = Vec::with_capacity(depth * 12);
    for level in 0..depth {
        let innermost = level + 1 == depth;
        push_u32(
            &mut payload,
            if innermost { VALUE_UINT8 } else { VALUE_ARRAY },
        );
        push_u64(&mut payload, if innermost { 0 } else { 1 });
    }
    Kv {
        key: "tokenizer.ggml.nested",
        value_type: VALUE_ARRAY,
        payload,
    }
}

/// 200k nesting levels (2.4 MB) used to recurse `skip_value` <-> `skip_array`
/// once per level and abort the process with "thread has overflowed its
/// stack" (exit 134), a DoS from the public `probe_model`.
///
/// Detection mechanism: a Rust stack overflow is a SIGABRT of the whole test
/// process, not a `panic!`, so `join()` cannot observe it. The 256 KiB thread
/// is an abort-by-process-exit detector: a recursive regression kills the
/// test binary (CI goes red through the non-zero exit), and the assertions
/// after `join()` only run when the parse completed within that stack.
///
/// Expected outcome: engram-parser walks every level with its explicit
/// stack, then this crate's adapter refuses the tensor-less file with
/// `MissingTensor { token_embd.weight }`. Pinning that exact variant proves
/// the metadata section was consumed in full rather than rejected early for
/// an unrelated reason. `moderately_nested_metadata_arrays_are_skipped_and_the_router_loads`
/// shows the same fixture builder is otherwise valid.
#[test]
fn deeply_nested_metadata_arrays_do_not_overflow_the_stack() {
    const DEPTH: usize = 200_000;
    let bytes = build_gguf(32, vec![nested_array_kv(DEPTH)], &[], &[]);
    assert!(bytes.len() > 2 * 1024 * 1024, "fixture should be ~2.4 MB");
    let file = TempGguf::write(&bytes, "nested-arrays");
    let path = file.as_str().to_owned();

    let probe = std::thread::Builder::new()
        .name("nested-array-probe".into())
        .stack_size(256 * 1024)
        .spawn(move || MoeRouter::probe_model(&path))
        .unwrap()
        .join()
        .expect("probe thread must return (a stack overflow aborts the process before this)");

    let err = match probe {
        Ok(metadata) => panic!("tensor-less file must not probe successfully: {metadata:?}"),
        Err(err) => err,
    };
    assert!(
        matches!(&err, CortexError::MissingTensor { name, .. } if name == "token_embd.weight"),
        "expected the adapter's MissingTensor after the metadata walk, got {err:?}"
    );
}

/// The same shape at a depth that comfortably fits any stack, with real
/// tensors behind it, must load: deep-but-valid metadata is not an error.
#[test]
fn moderately_nested_metadata_arrays_are_skipped_and_the_router_loads() {
    let gate = gate_payload();
    let mut data = gate.clone();
    data.extend_from_slice(&token_payload());
    let bytes = build_gguf(
        32,
        vec![nested_array_kv(64)],
        &[
            gate_entry(0),
            token_entry(
                vec![HIDDEN as u64, VOCAB as u64],
                GGML_TYPE_F32,
                gate.len() as u64,
            ),
        ],
        &data,
    );
    let file = TempGguf::write(&bytes, "nested-arrays-64");
    let router = match MoeRouter::load(file.as_str(), 0, 0) {
        Ok(router) => router,
        Err(err) => panic!("router must load past deep-but-valid metadata: {err}"),
    };
    assert_eq!(router.num_experts(), EXPERTS);
}
