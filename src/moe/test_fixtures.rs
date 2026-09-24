// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::types::EMBEDDING_DIM;
use std::io::Write;
use std::path::PathBuf;

pub(crate) fn write_temp_file(bytes: &[u8], label: &str) -> PathBuf {
    let mut path = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("target")
        .join("test-gguf");
    std::fs::create_dir_all(&path).ok();
    path.push(format!(
        "corinth_canal_{label}_{}.gguf",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(bytes).unwrap();
    path
}

pub(crate) fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn push_string(out: &mut Vec<u8>, value: &str) {
    push_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

pub(crate) fn push_kv_raw(out: &mut Vec<u8>, key: &str, value_type: u32, payload: &[u8]) {
    push_string(out, key);
    push_u32(out, value_type);
    out.extend_from_slice(payload);
}

pub(crate) fn push_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
    push_kv_raw(
        out,
        key,
        super::GGUF_VALUE_TYPE_UINT32,
        &value.to_le_bytes(),
    );
}

pub(crate) fn push_kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
    let mut payload = Vec::new();
    push_string(&mut payload, value);
    push_kv_raw(out, key, super::GGUF_VALUE_TYPE_STRING, &payload);
}

pub(crate) fn push_kv_i8(out: &mut Vec<u8>, key: &str, value: i8) {
    push_kv_raw(
        out,
        key,
        super::GGUF_VALUE_TYPE_INT8,
        &(value as u8).to_le_bytes(),
    );
}

pub(crate) fn push_kv_i16(out: &mut Vec<u8>, key: &str, value: i16) {
    push_kv_raw(
        out,
        key,
        super::GGUF_VALUE_TYPE_INT16,
        &(value as u16).to_le_bytes(),
    );
}

pub(crate) fn push_kv_i32(out: &mut Vec<u8>, key: &str, value: i32) {
    push_kv_raw(
        out,
        key,
        super::GGUF_VALUE_TYPE_INT32,
        &(value as u32).to_le_bytes(),
    );
}

pub(crate) fn push_kv_i64(out: &mut Vec<u8>, key: &str, value: i64) {
    push_kv_raw(
        out,
        key,
        super::GGUF_VALUE_TYPE_INT64,
        &(value as u64).to_le_bytes(),
    );
}

pub(crate) fn push_kv_bool(out: &mut Vec<u8>, key: &str, value: bool) {
    push_kv_raw(out, key, super::GGUF_VALUE_TYPE_BOOL, &[value as u8]);
}

pub(crate) fn push_kv_f32(out: &mut Vec<u8>, key: &str, value: f32) {
    push_kv_raw(
        out,
        key,
        super::GGUF_VALUE_TYPE_FLOAT32,
        &value.to_le_bytes(),
    );
}

pub(crate) fn push_kv_f64(out: &mut Vec<u8>, key: &str, value: f64) {
    push_kv_raw(
        out,
        key,
        super::GGUF_VALUE_TYPE_FLOAT64,
        &value.to_le_bytes(),
    );
}

pub(crate) fn push_kv_array_u32(out: &mut Vec<u8>, key: &str, values: &[u32]) {
    let mut payload = Vec::new();
    push_u32(&mut payload, super::GGUF_VALUE_TYPE_UINT32);
    push_u64(&mut payload, values.len() as u64);
    for value in values {
        push_u32(&mut payload, *value);
    }
    push_kv_raw(out, key, super::GGUF_VALUE_TYPE_ARRAY, &payload);
}

pub(crate) fn build_test_gguf(
    tensors: Vec<(&str, Vec<usize>, u32, Vec<u8>)>,
    alignment: u32,
    hidden_size: usize,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&super::GGUF_MAGIC);
    push_u32(&mut out, super::GGUF_VERSION);
    push_u64(&mut out, tensors.len() as u64);
    push_u64(&mut out, 7);
    push_kv_u32(&mut out, "general.alignment", alignment);
    push_kv_u32(&mut out, "general.file_type", 1);
    push_kv_string(&mut out, "general.architecture", "reference_moe");
    push_kv_u32(
        &mut out,
        "reference_moe.embedding_length",
        hidden_size as u32,
    );
    push_kv_u32(&mut out, "reference_moe.block_count", 16);
    push_kv_u32(&mut out, "reference_moe.expert_count", 64);
    push_kv_u32(&mut out, "reference_moe.expert_used_count", 8);

    let mut data_offset = 0usize;
    let mut tensor_payloads = Vec::new();
    for (name, dims, ggml_type, payload) in tensors {
        push_string(&mut out, name);
        push_u32(&mut out, dims.len() as u32);
        for dim in &dims {
            push_u64(&mut out, *dim as u64);
        }
        push_u32(&mut out, ggml_type);
        push_u64(&mut out, data_offset as u64);
        data_offset += payload.len();
        tensor_payloads.push(payload);
    }

    while out.len() % alignment as usize != 0 {
        out.push(0);
    }
    for payload in tensor_payloads {
        out.extend_from_slice(&payload);
    }

    out
}

pub(crate) fn build_real_size_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
    let attn_q_payload = vec![0u8; EMBEDDING_DIM * EMBEDDING_DIM * 2];
    build_test_gguf(
        vec![
            (
                "blk.0.ffn_gate_inp.weight",
                vec![EMBEDDING_DIM, 64],
                super::GGML_TYPE_F32,
                gate_payload,
            ),
            (
                "blk.0.attn_q.weight",
                vec![EMBEDDING_DIM, EMBEDDING_DIM],
                super::GGML_TYPE_F16,
                attn_q_payload,
            ),
            (
                "token_embd.weight",
                vec![EMBEDDING_DIM, 32],
                super::GGML_TYPE_F16,
                vec![0u8; EMBEDDING_DIM * 32 * 2],
            ),
        ],
        32,
        EMBEDDING_DIM,
    )
}
