// SPDX-License-Identifier: MIT OR Apache-2.0

//! Criterion benchmarks for the deterministic Vec reference backend (RM-1491).
//!
//! CI only builds these (`cargo bench --no-run`); timings never gate a build.
//! Run locally with `cargo bench`.

use std::hint::black_box;

use cortex_tensor::Tensor;
use cortex_tensor::tensor::ops::matmul;
use cortex_tensor::transformer::{TransformerConfig, TransformerLM};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const SEED: u64 = 0x5EED;

fn seeded(shape: &[usize], salt: u64) -> Tensor {
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(SEED ^ salt);
    Tensor::try_randn_with_rng(shape, 0.0, 1.0, &mut rng).expect("bench tensor")
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for n in [256usize, 1024] {
        let a = seeded(&[n, n], 1);
        let b = seeded(&[n, n], 2);
        group.throughput(Throughput::Elements((2 * n * n * n) as u64));
        if n >= 1024 {
            group.sample_size(10);
        }
        group.bench_function(format!("{n}x{n}"), |bench| {
            bench.iter(|| matmul(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_transformer_forward(c: &mut Criterion) {
    let cfg = TransformerConfig {
        vocab_size: 256,
        dim: 64,
        num_heads: 4,
        num_layers: 2,
        ff_dim: 128,
        max_seq_len: 32,
    };
    let model = TransformerLM::try_new_seeded(cfg, SEED).expect("tiny model");
    let tokens: Vec<u32> = (0..32u32).collect();
    c.bench_function("transformer_lm/tiny_forward_seq32", |bench| {
        bench.iter(|| model.try_forward(black_box(&tokens)).expect("forward"));
    });
}

/// MoE router scoring step: one `[1, H] x [H, E]` GEMV, the shape of the dense
/// gate in `checkpoint_gate_scores` (crate-private, so the public `matmul`
/// stands in with the same dimensions and arithmetic).
fn bench_gate_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("gate_gemv");
    for (hidden, experts) in [(2048usize, 64usize), (2048, 128)] {
        let x = seeded(&[1, hidden], 3);
        let gate = seeded(&[hidden, experts], 4);
        group.throughput(Throughput::Elements((2 * hidden * experts) as u64));
        group.bench_function(format!("h{hidden}_e{experts}"), |bench| {
            bench.iter(|| matmul(black_box(&x), black_box(&gate)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_transformer_forward,
    bench_gate_gemv
);
criterion_main!(benches);
