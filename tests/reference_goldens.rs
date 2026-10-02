// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed reference semantics, independent of stage executors and external adapters.
//! See fixtures/REFERENCE_GOLDENS.md for derivations, inventory and maintenance.
#[path = "support/golden.rs"]
mod golden;

use cortex_tensor::moe::{EMBEDDING_DIM, MoeRouter, RoutingMode};
use cortex_tensor::tensor::ops::*;
use cortex_tensor::transformer::block::FeedForward;
use cortex_tensor::transformer::{
    MultiHeadAttention, TransformerBlock, TransformerConfig, TransformerLM,
};
use cortex_tensor::{CortexError, Tensor};
use golden::Golden;

fn t(shape: &[usize], data: &[f32]) -> Tensor {
    Tensor::try_from_vec(data.to_vec(), shape).unwrap()
}

fn check(name: &str, actual: &Tensor, expected: Golden<'_>) {
    expected
        .compare(actual.shape(), "f32", actual.data())
        .unwrap_or_else(|error| panic!("{name}: {error}"));
}

/// Every numerical path runs twice; same-process reference determinism is
/// bitwise, while comparison to independently derived constants is tolerant.
fn repeat(name: &str, expected: Golden<'_>, run: impl Fn() -> Tensor) {
    let first = run();
    check(name, &first, expected);
    let second = run();
    check(name, &second, expected);
    assert_eq!(
        first.data().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        second
            .data()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        "{name}: repeatability"
    );
}

#[test]
fn comparator_rejects_layout_dtype_and_nonfinite_mismatches() {
    let g = Golden::approximate(&[2], &[1.0, 2.0]);
    assert!(g.compare(&[1, 2], "f32", &[1.0, 2.0]).is_err());
    assert!(g.compare(&[2], "f64", &[1.0, 2.0]).is_err());
    assert!(g.compare(&[2], "f32", &[1.0]).is_err());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.01] {
        assert!(g.compare(&[2], "f32", &[1.0, bad]).is_err());
    }
    let special = Golden::exact(&[3], &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
    assert!(
        special
            .compare(&[3], "f32", &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY])
            .is_ok()
    );
    assert!(
        special
            .compare(&[3], "f32", &[0.0, f32::INFINITY, f32::NEG_INFINITY])
            .is_err()
    );
    assert!(
        special
            .compare(&[3], "f32", &[f32::NAN, f32::NEG_INFINITY, f32::INFINITY])
            .is_err()
    );
    assert!(g.compare(&[2], "f32", &[1.000001, 2.000001]).is_ok());
}

#[test]
fn checked_layout_and_arithmetic() {
    let a = t(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let b = t(&[2, 3], &[6., 5., 4., 3., 2., 1.]);
    assert_eq!(a.try_row_major_strides().unwrap(), [3, 1]);
    assert_eq!((a.ndim(), a.numel()), (2, 6));
    repeat("row", Golden::exact(&[3], &[4., 5., 6.]), || {
        a.try_row(1).unwrap()
    });
    repeat(
        "transpose",
        Golden::exact(&[3, 2], &[1., 4., 2., 5., 3., 6.]),
        || a.try_transpose().unwrap(),
    );
    repeat(
        "reshape",
        Golden::exact(&[3, 2], &[1., 2., 3., 4., 5., 6.]),
        || a.try_reshape(&[3, 2]).unwrap(),
    );
    let mut reshaped = a.clone();
    reshaped.reshape_in_place(&[6]).unwrap();
    check("reshape in place", &reshaped, Golden::exact(&[6], a.data()));
    repeat("add", Golden::exact(&[2, 3], &[7.; 6]), || {
        a.try_add(&b).unwrap()
    });
    repeat(
        "sub",
        Golden::exact(&[2, 3], &[-5., -3., -1., 1., 3., 5.]),
        || a.try_sub(&b).unwrap(),
    );
    repeat(
        "mul",
        Golden::exact(&[2, 3], &[6., 10., 12., 12., 10., 6.]),
        || a.try_mul(&b).unwrap(),
    );
    repeat(
        "scale",
        Golden::exact(&[2, 3], &[0.5, 1., 1.5, 2., 2.5, 3.]),
        || a.scale(0.5),
    );
    repeat(
        "scalar add",
        Golden::exact(&[2, 3], &[0., 1., 2., 3., 4., 5.]),
        || a.add_scalar(-1.),
    );
    assert_eq!(
        (a.sum(), a.mean(), a.max_val(), a.argmax()),
        (21., 3.5, 6., 5)
    );
    repeat("scalar", Golden::exact(&[], &[3.]), || {
        Tensor::try_full(&[], 3.).unwrap()
    });
    repeat("ones", Golden::exact(&[2], &[1., 1.]), || {
        Tensor::try_ones(&[2]).unwrap()
    });
    repeat("zero axis", Golden::exact(&[2, 0, 3], &[]), || {
        Tensor::try_zeros(&[2, 0, 3]).unwrap()
    });
}

#[test]
fn checked_products_embedding_and_mask() {
    let a = t(&[2, 3], &[1., 2., 3., -1., 0., 2.]);
    let b = t(&[3, 2], &[2., 1., 0., -1., 3., 2.]);
    repeat(
        "rectangular matmul",
        Golden::exact(&[2, 2], &[11., 5., 4., 3.]),
        || try_matmul(&a, &b).unwrap(),
    );
    let ba = t(&[2, 1, 3], &[1., 2., 3., -1., 0., 2.]);
    repeat(
        "broadcast right matrix",
        Golden::exact(&[2, 1, 2], &[11., 5., 4., 3.]),
        || try_batched_matmul(&ba, &b).unwrap(),
    );
    repeat(
        "rank two batched fallback",
        Golden::exact(&[2, 2], &[11., 5., 4., 3.]),
        || try_batched_matmul(&a, &b).unwrap(),
    );
    let bb = t(
        &[2, 3, 2],
        &[2., 1., 0., -1., 3., 2., 1., 2., 3., 4., 5., 6.],
    );
    repeat(
        "batched matmul",
        Golden::exact(&[2, 1, 2], &[11., 5., 9., 10.]),
        || try_batched_matmul(&ba, &bb).unwrap(),
    );
    repeat("zero contraction", Golden::exact(&[2, 3], &[0.; 6]), || {
        try_matmul(&t(&[2, 0], &[]), &t(&[0, 3], &[])).unwrap()
    });
    repeat(
        "embedding order and duplicate ids",
        Golden::exact(&[3, 3], &[-1., 0., 2., 1., 2., 3., -1., 0., 2.]),
        || try_embedding(&a, &[1, 0, 1]).unwrap(),
    );
    repeat("empty embedding", Golden::exact(&[0, 3], &[]), || {
        try_embedding(&a, &[]).unwrap()
    });
    repeat(
        "causal mask",
        Golden::exact(
            &[3, 3],
            &[
                0.,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.,
                0.,
                f32::NEG_INFINITY,
                0.,
                0.,
                0.,
            ],
        ),
        || try_causal_mask(3).unwrap(),
    );
    repeat("empty mask", Golden::exact(&[0, 0], &[]), || {
        try_causal_mask(0).unwrap()
    });
}

#[test]
fn activation_and_norm_conventions() {
    let x = t(&[3], &[-1., 0., 1.]);
    repeat("relu", Golden::exact(&[3], &[0., 0., 1.]), || x.relu());
    repeat(
        "tanh GELU",
        Golden::approximate(&[3], &[-0.158808, 0., 0.841192]),
        || x.gelu(),
    );
    repeat(
        "SiLU",
        Golden::approximate(&[3], &[-0.26894143, 0., 0.7310586]),
        || x.silu(),
    );
    repeat(
        "fast sigmoid surrogate",
        Golden::exact(&[3], &[-0.5, 0., 0.5]),
        || x.fast_sigmoid(),
    );
    let x = t(&[2, 2], &[1., 3., 5., 5.]);
    // Population variance 1 and 0; eps=3 gives denominators 2 and sqrt(3).
    repeat(
        "LayerNorm last axis affine",
        Golden::exact(&[2, 2], &[-0.5, 1., 0.5, -1.]),
        || try_layer_norm(&x, &t(&[2], &[2., 4.]), &t(&[2], &[0.5, -1.]), 3.).unwrap(),
    );
    // mean(x^2)=12.5, eps=3.5 => denominator 4, with no mean subtraction.
    repeat(
        "RMSNorm",
        Golden::exact(&[2, 2], &[1.5, -1., 0., 0.]),
        || try_rms_norm(&t(&[2, 2], &[3., 4., 0., 0.]), &t(&[2], &[2., -1.]), 3.5).unwrap(),
    );
    repeat(
        "finite softmax",
        Golden::approximate(&[2, 2], &[0.26894143, 0.7310586, 0.5, 0.5]),
        || try_softmax(&t(&[2, 2], &[0., 1., 4., 4.])).unwrap(),
    );
    repeat("scalar softmax", Golden::exact(&[], &[1.]), || {
        t(&[], &[8.]).try_softmax_last().unwrap()
    });
    repeat("empty softmax", Golden::exact(&[2, 0], &[]), || {
        try_softmax(&t(&[2, 0], &[])).unwrap()
    });
    for (name, input, output) in [
        (
            "positive infinities",
            vec![f32::INFINITY, 1., f32::INFINITY],
            vec![0.5, 0., 0.5],
        ),
        ("fully masked", vec![f32::NEG_INFINITY; 2], vec![0.5; 2]),
        ("NaN row", vec![0., f32::NAN], vec![f32::NAN; 2]),
    ] {
        repeat(name, Golden::exact(&[output.len()], &output), || {
            try_softmax(&t(&[input.len()], &input)).unwrap()
        });
    }
    repeat(
        "nonfinite norm row isolation",
        Golden::exact(&[2, 2], &[f32::NAN, f32::NAN, -0.5, 0.5]),
        || {
            try_layer_norm(
                &t(&[2, 2], &[f32::INFINITY, 0., 1., 3.]),
                &t(&[2], &[1., 1.]),
                &t(&[2], &[0., 0.]),
                3.,
            )
            .unwrap()
        },
    );
}

fn identity(dim: usize) -> Tensor {
    let mut data = vec![0.; dim * dim];
    for i in 0..dim {
        data[i * dim + i] = 1.;
    }
    t(&[dim, dim], &data)
}

fn attention(dim: usize, heads: usize) -> MultiHeadAttention {
    MultiHeadAttention {
        dim,
        num_heads: heads,
        head_dim: dim / heads,
        wq: identity(dim),
        wk: identity(dim),
        wv: identity(dim),
        wo: identity(dim),
        wb_q: Tensor::zeros(&[1, dim]),
        wb_k: Tensor::zeros(&[1, dim]),
        wb_v: Tensor::zeros(&[1, dim]),
        wb_o: Tensor::zeros(&[1, dim]),
    }
}

#[test]
fn causal_attention_scaling_heads_and_sequence_boundaries() {
    // Two contiguous heads of width 2. On row 1, head 0 scores are
    // [0, 1/sqrt(2)]; head 1 scores are [0, 4/sqrt(2)].
    let attn = attention(4, 2);
    let x = t(&[2, 4], &[1., 0., 2., 0., 0., 1., 0., 2.]);
    repeat(
        "two heads",
        Golden::approximate(
            &[2, 4],
            &[
                1., 0., 2., 0., 0.33023846, 0.66976154, 0.11161444, 1.8883855,
            ],
        ),
        || attn.try_forward(&x).unwrap(),
    );
    repeat(
        "singleton",
        Golden::exact(&[1, 4], &[1., 0., 2., 0.]),
        || attn.try_forward(&t(&[1, 4], &[1., 0., 2., 0.])).unwrap(),
    );
    repeat("empty attention", Golden::exact(&[0, 4], &[]), || {
        attn.try_forward(&t(&[0, 4], &[])).unwrap()
    });
    let changed_future = t(&[2, 4], &[1., 0., 2., 0., 9., -2., 4., -3.]);
    check(
        "future cannot affect prefix",
        &attn
            .try_forward(&changed_future)
            .unwrap()
            .try_row(0)
            .unwrap(),
        Golden::exact(&[4], &[1., 0., 2., 0.]),
    );
    let mut biased = attention(2, 1);
    biased.wq = t(&[2, 2], &[1., 2., 0., 1.]);
    biased.wk = t(&[2, 2], &[0., 1., 1., 0.]);
    biased.wv = t(&[2, 2], &[2., 0., 0., -1.]);
    biased.wo = t(&[2, 2], &[0., 1., 2., 0.]);
    biased.wb_q = t(&[1, 2], &[0.5, 0.25]);
    biased.wb_k = t(&[1, 2], &[-0.5, 0.5]);
    biased.wb_v = t(&[1, 2], &[1., 2.]);
    biased.wb_o = t(&[1, 2], &[0.25, -0.25]);
    // Explicit non-identity projections exercise orientation and every bias.
    repeat(
        "projected attention",
        Golden::approximate(&[2, 2], &[4.25, 2.75, 3.5091202, 2.0091202]),
        || biased.try_forward(&t(&[2, 2], &[1., 0., 0., 1.])).unwrap(),
    );
}

fn ffn() -> FeedForward {
    FeedForward {
        w1: t(&[2, 2], &[1., -1., 0.5, 0.25]),
        b1: t(&[1, 2], &[0.1, -0.2]),
        w2: t(&[2, 2], &[0.5, -0.25, 0.25, 0.5]),
        b2: t(&[1, 2], &[0.2, -0.1]),
    }
}

fn block() -> TransformerBlock {
    let mut attn = attention(2, 1);
    attn.wq = t(&[2, 2], &[0.; 4]);
    attn.wk = t(&[2, 2], &[0.; 4]);
    attn.wo = t(&[2, 2], &[0.5, 0., 0., -0.25]);
    attn.wb_o = t(&[1, 2], &[0.1, -0.2]);
    TransformerBlock {
        attn,
        ffn: ffn(),
        dim: 2,
        ln1_w: t(&[2], &[1., 1.]),
        ln1_b: t(&[2], &[0., 0.]),
        ln2_w: t(&[2], &[1., 1.]),
        ln2_b: t(&[2], &[0., 0.]),
    }
}

fn model() -> TransformerLM {
    TransformerLM {
        config: TransformerConfig {
            vocab_size: 3,
            dim: 2,
            num_heads: 1,
            num_layers: 1,
            ff_dim: 2,
            max_seq_len: 3,
        },
        tok_embed: t(&[3, 2], &[1., 3., -2., 1., 2., -1.]),
        pos_embed: t(&[3, 2], &[0.5, -0.5, -0.5, 0.25, 0., 0.5]),
        blocks: vec![block()],
        final_ln_w: t(&[2], &[1.25, 0.75]),
        final_ln_b: t(&[2], &[0.1, -0.2]),
        lm_head: t(&[2, 3], &[1., 0., -1., 0.5, 2., 1.]),
    }
}

#[test]
fn dense_mlp_and_composed_hidden_boundaries() {
    repeat(
        "dense GELU MLP",
        Golden::approximate(
            &[3, 2],
            &[
                0.6407162,
                -0.406_793_9,
                0.42420682,
                -0.19585533,
                0.5204736,
                0.4734688,
            ],
        ),
        || {
            ffn()
                .try_forward(&t(&[3, 2], &[1., 0., 0., 1., -1., 2.]))
                .unwrap()
        },
    );
    let model = model();
    let block = &model.blocks[0];
    // Token order [2,0,1] plus position order [0,1,2]. Fixed f32, row-major.
    let ingress = t(&[3, 2], &[2.5, -1.5, 0.5, 3.25, -2., 1.5]);
    check(
        "embedding ingress",
        &try_embedding(&model.tok_embed, &[2, 0, 1])
            .unwrap()
            .try_add(&model.pos_embed)
            .unwrap(),
        Golden::exact(&[3, 2], ingress.data()),
    );
    let norm1 = try_layer_norm(&ingress, &block.ln1_w, &block.ln1_b, 1e-5).unwrap();
    check(
        "norm1",
        &norm1,
        Golden::approximate(
            &[3, 2],
            &[
                0.99999875,
                -0.99999875,
                -0.9999974,
                0.9999974,
                -0.9999984,
                0.9999984,
            ],
        ),
    );
    let attn = block.attn.try_forward(&norm1).unwrap();
    check(
        "uniform causal prefix attention",
        &attn,
        Golden::approximate(
            &[3, 2],
            &[
                0.59999937,
                0.04999969,
                0.10000035,
                -0.19999982,
                -0.06666616,
                -0.2833331,
            ],
        ),
    );
    let residual = ingress.try_add(&attn).unwrap();
    check(
        "first residual",
        &residual,
        Golden::approximate(
            &[3, 2],
            &[
                3.0999994, -1.4500003, 0.6000004, 3.0500002, -2.0666661, 1.2166669,
            ],
        ),
    );
    let norm2 = try_layer_norm(&residual, &block.ln2_w, &block.ln2_b, 1e-5).unwrap();
    check(
        "norm2",
        &norm2,
        Golden::approximate(
            &[3, 2],
            &[
                0.99999905,
                -0.99999905,
                -0.99999666,
                0.99999666,
                -0.99999815,
                0.99999815,
            ],
        ),
    );
    check(
        "MLP before residual",
        &block.ffn.try_forward(&norm2).unwrap(),
        Golden::approximate(
            &[3, 2],
            &[
                0.39099613,
                -0.26227617,
                0.3549866,
                0.38227132,
                0.35498703,
                0.3822724,
            ],
        ),
    );
    repeat(
        "block egress",
        Golden::approximate(
            &[3, 2],
            &[
                3.4909954, -1.7122765, 0.95498693, 3.4322715, -1.7116791, 1.5989393,
            ],
        ),
        || block.try_forward(&ingress).unwrap(),
    );
    let hidden = Golden::approximate(
        &[3, 2],
        &[
            1.3499991,
            -0.94999945,
            -1.1499959,
            0.54999757,
            -1.1499977,
            0.54999864,
        ],
    );
    repeat("hidden egress includes final norm", hidden, || {
        model.try_hidden_states(&[2, 0, 1]).unwrap()
    });
    let logits = Golden::approximate(
        &[3, 3],
        &[
            0.87499934,
            -1.8999989,
            -2.2999985,
            -0.87499714,
            1.0999951,
            1.6999935,
            -0.8749984,
            1.0999973,
            1.6999964,
        ],
    );
    repeat("composed ANN", logits, || {
        model.try_forward(&[2, 0, 1]).unwrap()
    });
    // Explicit hidden ingress to readout uses the frozen egress, not model output.
    repeat("hidden ingress to readout", logits, || {
        try_matmul(&t(hidden.shape, hidden.data), &model.lm_head).unwrap()
    });
    repeat("empty model hidden", Golden::exact(&[0, 2], &[]), || {
        model.try_hidden_states(&[]).unwrap()
    });
    repeat("empty model logits", Golden::exact(&[0, 3], &[]), || {
        model.try_forward(&[]).unwrap()
    });
}

#[test]
fn dense_moe_selection_weights_and_hidden_combination() {
    assert_eq!(EMBEDDING_DIM, 2048, "existing public ingress width");
    // Sparse encoding of a 2048-element f32 input: every omitted value is zero.
    // Four 512-wide chunks have raw sums [0,1,1,-1]. No learned experts exist.
    let mut input = vec![0.; 2048];
    input[512] = 1.;
    input[1024] = 1.;
    input[1536] = -1.;
    let mut router = MoeRouter::load_with_mode("", 4, 2, RoutingMode::DenseSim).unwrap();
    let expected_weights =
        Golden::approximate(&[4], &[0.14696279, 0.3994863, 0.3994863, 0.05406459]);
    let mut expected_hidden = vec![0.; 2048];
    expected_hidden[512] = 0.7989726;
    expected_hidden[1024] = 0.7989726;
    expected_hidden[1536] = -0.7989726;
    let first = router.forward(&input).unwrap();
    for _ in 0..3 {
        let out = router.forward(&input).unwrap();
        assert_eq!(out.selected_experts, [1, 2]); // exact IDs; ascending tie-break
        expected_weights
            .compare(&[out.expert_weights.len()], "f32", &out.expert_weights)
            .unwrap();
        Golden::approximate(&[2048], &expected_hidden)
            .compare(&[out.hidden.len()], "f32", &out.hidden)
            .unwrap();
        assert_eq!(
            out.expert_weights
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            first
                .expert_weights
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            out.hidden.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            first.hidden.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    // Public constructor clamps k to [1,n], unlike the internal top-k helper.
    for (k, selected) in [(0, vec![1]), (1, vec![1]), (9, vec![1, 2, 0, 3])] {
        let out = MoeRouter::load_with_mode("", 4, k, RoutingMode::DenseSim)
            .unwrap()
            .forward(&input)
            .unwrap();
        assert_eq!(out.selected_experts, selected);
    }
    let tied = router.forward(&vec![0.; 2048]).unwrap();
    assert_eq!(tied.selected_experts, [0, 1]);
    assert_eq!(tied.expert_weights, [0.25; 4]);
    assert_eq!(tied.hidden, vec![0.; 2048]);
    input[512] = f32::NAN;
    assert!(matches!(
        router.forward(&input),
        Err(CortexError::NanRoutingScore { expert_id: 1 })
    ));
    assert!(matches!(
        router.forward(&[0.; 2]),
        Err(CortexError::InputLengthMismatch {
            expected: 2048,
            got: 2
        })
    ));
}

#[test]
fn checked_errors_are_structured_not_panic_expectations() {
    let x = t(&[2, 2], &[1., 2., 3., 4.]);
    assert!(matches!(
        Tensor::try_from_vec(vec![1.], &[2]),
        Err(CortexError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        Tensor::try_zeros(&[usize::MAX, 2]),
        Err(CortexError::SizeOverflow { .. })
    ));
    assert!(matches!(
        x.try_reshape(&[3]),
        Err(CortexError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        x.try_row(2),
        Err(CortexError::IndexOutOfBounds {
            axis: 0,
            index: 2,
            size: 2
        })
    ));
    assert!(matches!(
        t(&[2], &[1., 2.]).try_transpose(),
        Err(CortexError::RankMismatch {
            expected: 2,
            got: 1
        })
    ));
    let wrong_shape = t(&[4], &[1.; 4]);
    for error in [
        x.try_add(&wrong_shape),
        x.try_sub(&wrong_shape),
        x.try_mul(&wrong_shape),
    ] {
        assert!(matches!(error, Err(CortexError::ShapeMismatch { .. })));
    }
    assert!(matches!(
        try_matmul(&x, &t(&[3, 1], &[1.; 3])),
        Err(CortexError::MatmulDim {
            m: 2,
            k1: 2,
            k2: 3,
            n: 1
        })
    ));
    assert!(matches!(
        try_batched_matmul(&t(&[2, 1, 2], &[1.; 4]), &t(&[3, 2, 1], &[1.; 6])),
        Err(CortexError::DimMismatch {
            axis: 0,
            expected: 2,
            got: 3,
            ..
        })
    ));
    assert!(matches!(
        try_embedding(&x, &[2]),
        Err(CortexError::TokenIndex {
            index: 2,
            vocab_size: 2
        })
    ));
    assert!(matches!(
        try_softmax(&t(&[1, 1, 1], &[1.])),
        Err(CortexError::InvalidConfig(_))
    ));
    let w = t(&[2], &[1., 1.]);
    let b = t(&[2], &[0., 0.]);
    for eps in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(matches!(
            try_layer_norm(&x, &w, &b, eps),
            Err(CortexError::InvalidEpsilon { .. })
        ));
        assert!(matches!(
            try_rms_norm(&x, &w, eps),
            Err(CortexError::InvalidEpsilon { .. })
        ));
    }
    for error in [
        try_layer_norm(&t(&[1, 0], &[]), &w, &b, 1e-5),
        try_rms_norm(&t(&[1, 0], &[]), &w, 1e-5),
    ] {
        assert!(matches!(error, Err(CortexError::ZeroWidth { axis: 1, .. })));
    }
    assert!(matches!(
        try_layer_norm(&x, &t(&[1], &[1.]), &b, 1e-5),
        Err(CortexError::DimMismatch { .. })
    ));
    assert!(matches!(
        MultiHeadAttention::try_new(3, 2),
        Err(CortexError::InvalidConfig(_))
    ));
    assert!(matches!(
        attention(2, 1).try_forward(&t(&[1, 3], &[1.; 3])),
        Err(CortexError::DimMismatch { .. })
    ));
    assert!(matches!(
        ffn().try_forward(&t(&[2], &[1., 2.])),
        Err(CortexError::RankMismatch {
            expected: 2,
            got: 1
        })
    ));
    assert!(matches!(
        model().try_forward(&[3]),
        Err(CortexError::TokenIndex {
            index: 3,
            vocab_size: 3
        })
    ));
    assert!(matches!(
        model().try_hidden_states(&[0; 4]),
        Err(CortexError::InputLengthMismatch {
            expected: 3,
            got: 4
        })
    ));
}
