// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile-time / public API regression test for panic-style wrappers retained
//! for pre-1.0 source compatibility. If a signature here fails to compile, the
//! wrapper was removed or changed without a SemVer decision (see RM-1353).

use cortex_tensor::tensor::ops::{
    batched_matmul, embedding, layer_norm, matmul, rms_norm, try_batched_matmul, try_embedding,
    try_layer_norm, try_matmul, try_rms_norm,
};
use cortex_tensor::{Result, Tensor};

#[test]
fn retained_panic_wrappers_have_stable_signatures() {
    let _: fn(Vec<f32>, &[usize]) -> Tensor = Tensor::from_vec;
    let _: fn(&[usize]) -> Tensor = Tensor::zeros;
    let _: fn(&[usize]) -> Tensor = Tensor::ones;
    let _: fn(&[usize], f32) -> Tensor = Tensor::full;
    let _: fn(&[usize], f32, f32) -> Tensor = Tensor::randn;
    let _: fn(&Tensor, &Tensor) -> Tensor = matmul;
    let _: fn(&Tensor, &Tensor) -> Tensor = batched_matmul;
    let _: fn(&Tensor, &Tensor, &Tensor, f32) -> Tensor = layer_norm;
    let _: fn(&Tensor, &Tensor, f32) -> Tensor = rms_norm;
    let _: fn(&Tensor, &[u32]) -> Tensor = embedding;
}

#[test]
fn fallible_apis_are_public() {
    let _: fn(Vec<f32>, &[usize]) -> Result<Tensor> = Tensor::try_from_vec;
    let _: fn(&Tensor, &Tensor) -> Result<Tensor> = try_matmul;
    let _: fn(&Tensor, &Tensor) -> Result<Tensor> = try_batched_matmul;
    let _: fn(&Tensor, &[u32]) -> Result<Tensor> = try_embedding;
    let _: fn(&Tensor, &Tensor, &Tensor, f32) -> Result<Tensor> = try_layer_norm;
    let _: fn(&Tensor, &Tensor, f32) -> Result<Tensor> = try_rms_norm;
}
