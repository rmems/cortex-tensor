// SPDX-License-Identifier: MIT OR Apache-2.0

//! Finite / non-finite policy for tensor math and MoE routing helpers.
//!
//! Public storage stays `f32`. Reductions that can lose precision or overflow
//! in `f32` (softmax partition functions, LayerNorm / RMSNorm moments, L2
//! norms) accumulate in `f64` and cast back only when writing the output.
//!
//! # Policy
//!
//! These rules are total: every input class has one documented output. Kernels
//! never use [`f32::max`] or [`partial_cmp`](std::cmp::PartialOrd::partial_cmp)
//! to rank values, because both treat NaN as a missing comparison instead of a
//! distinct class (Rust's `f32::max` even *ignores* NaN).
//!
//! ## Softmax (last axis / one logit row)
//!
//! Max-subtracted `exp` with an `f64` partition sum.
//!
//! | Input row | Output |
//! |-----------|--------|
//! | Empty (`n = 0`) | Empty. There is no probability mass. |
//! | Any NaN | Every output is NaN. NaN is never a softmax argmax or a tie. |
//! | One `+Inf` | Probability `1` at that index, `0` elsewhere. |
//! | `k > 1` values of `+Inf` | Those `k` indices share `1/k` each (deterministic equal split); all other positions are `0`. |
//! | All `-Inf` (fully masked) | Uniform `1/n`. A valid simplex; no NaN from `(-Inf) - (-Inf)`. |
//! | Finite (including values near `±1e30`) | Standard stable softmax. Outputs are non-negative and sum to 1 within [`SOFTMAX_SUM_TOLERANCE`] for rows of length `≤ 4096`. |
//!
//! Finite bounded logits therefore produce finite probabilities.
//!
//! ## LayerNorm / RMSNorm
//!
//! Population statistics over the last axis of a rank-2 `[batch, dim]` tensor.
//! Affine parameters stay `f32`; mean / mean-square / variance use Welford or
//! compensated `f64` sums.
//!
//! | Case | Behavior |
//! |----------|----------|
//! | `eps` not finite or `eps ≤ 0` | [`CortexError::InvalidEpsilon`](crate::CortexError::InvalidEpsilon). |
//! | Last-axis width `0` | [`CortexError::ZeroWidth`](crate::CortexError::ZeroWidth). |
//! | Empty batch (`rows = 0`, `dim > 0`) | Empty output with the same shape. |
//! | Any non-finite value in a data row | That output row is all NaN. |
//! | Finite data and finite affine parameters | Finite output. Values that overflow `f32` saturate to `±f32::MAX`. Constant rows yield the bias (LayerNorm) or `0` (RMSNorm with finite weight). |
//!
//! ## L2 routing normalize
//!
//! | Case | Behavior |
//! |------|----------|
//! | Empty | No-op. |
//! | Any non-finite | Fill with NaN. |
//! | Euclidean norm `≤` [`L2_NORM_FLOOR`] | Leave the vector unchanged (zero / tiny embeddings stay zero). |
//! | Otherwise | Divide by the `f64` Euclidean norm. |
//!
//! Behavior is sequential and therefore identical across debug/release and
//! supported platforms for a given input bit pattern.

use crate::error::{CortexError, Result};

/// Declared `|sum − 1|` bound for a finite softmax row of length `1..=4096`.
pub const SOFTMAX_SUM_TOLERANCE: f32 = 1e-5;

/// Vectors whose Euclidean norm is at or below this floor are left unchanged.
pub const L2_NORM_FLOOR: f64 = 1e-8;

/// Reject non-positive or non-finite epsilon used by LayerNorm / RMSNorm.
pub fn validate_norm_eps(eps: f32) -> Result<f64> {
    if eps.is_finite() && eps > 0.0 {
        Ok(f64::from(eps))
    } else {
        Err(CortexError::InvalidEpsilon { eps })
    }
}

/// Fold an `f32` rounding residual onto the largest mass so a finite simplex
/// row of length `≤ 4096` sums to 1 within [`SOFTMAX_SUM_TOLERANCE`].
fn apply_simplex_residual(out: &mut [f32]) {
    if out.is_empty() {
        return;
    }
    if !out.iter().all(|p| p.is_finite() && *p >= 0.0) {
        return;
    }
    if out.len() == 1 {
        out[0] = 1.0;
        return;
    }
    let mut sum = 0.0f32;
    let mut max_i = 0usize;
    for (i, &p) in out.iter().enumerate() {
        sum += p;
        if p > out[max_i] {
            max_i = i;
        }
    }
    if !sum.is_finite() {
        return;
    }
    let corrected = out[max_i] + (1.0 - sum);
    if corrected.is_finite() && corrected >= 0.0 {
        out[max_i] = corrected;
    }
}

/// Cast an affine result to `f32`, saturating overflow to `±f32::MAX`.
fn saturate_f32(y: f64) -> f32 {
    let y32 = y as f32;
    if y32.is_finite() {
        y32
    } else if y.is_nan() {
        f32::NAN
    } else if y.is_sign_positive() {
        f32::MAX
    } else {
        f32::MIN
    }
}

/// Max-subtracted softmax of one row according to the crate policy.
///
/// `logits` and `out` must have the same length. Crate-internal: callers should
/// use [`crate::tensor::ops::softmax`] or [`crate::tensor::Tensor::softmax_last`].
pub(crate) fn softmax_row(logits: &[f32], out: &mut [f32]) {
    assert_eq!(
        logits.len(),
        out.len(),
        "softmax_row: logit/out length mismatch"
    );
    let n = logits.len();
    if n == 0 {
        return;
    }

    let mut has_nan = false;
    let mut pos_inf = 0usize;
    let mut max_v = f32::NEG_INFINITY;
    for &x in logits {
        if x.is_nan() {
            has_nan = true;
            break;
        }
        if x.is_infinite() && x.is_sign_positive() {
            pos_inf += 1;
        }
        if x > max_v {
            max_v = x;
        }
    }

    if has_nan {
        out.fill(f32::NAN);
        return;
    }

    if pos_inf > 0 {
        let p = 1.0 / pos_inf as f32;
        for (dst, &x) in out.iter_mut().zip(logits.iter()) {
            *dst = if x.is_infinite() && x.is_sign_positive() {
                p
            } else {
                0.0
            };
        }
        apply_simplex_residual(out);
        return;
    }

    if max_v.is_infinite() && max_v.is_sign_negative() {
        out.fill(1.0 / n as f32);
        apply_simplex_residual(out);
        return;
    }

    let max64 = f64::from(max_v);
    let mut sum = 0.0f64;
    for &x in logits {
        sum += (f64::from(x) - max64).exp();
    }
    if sum == 0.0 || !sum.is_finite() {
        out.fill(1.0 / n as f32);
        apply_simplex_residual(out);
        return;
    }
    for (dst, &x) in out.iter_mut().zip(logits.iter()) {
        *dst = ((f64::from(x) - max64).exp() / sum) as f32;
    }
    apply_simplex_residual(out);
}

/// Allocate and return [`softmax_row`] over `logits`.
pub(crate) fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; logits.len()];
    softmax_row(logits, &mut out);
    out
}

/// LayerNorm one row with `f64` Welford mean/variance.
///
/// `eps` must already have been accepted by [`validate_norm_eps`]. Lengths
/// must match and `x` must be non-empty; the public path is
/// [`crate::tensor::ops::try_layer_norm`].
pub(crate) fn layer_norm_row(x: &[f32], weight: &[f32], bias: &[f32], eps: f64, out: &mut [f32]) {
    assert_eq!(
        x.len(),
        weight.len(),
        "layer_norm_row: weight length mismatch"
    );
    assert_eq!(x.len(), bias.len(), "layer_norm_row: bias length mismatch");
    assert_eq!(x.len(), out.len(), "layer_norm_row: out length mismatch");
    assert!(!x.is_empty(), "layer_norm_row: empty axis");

    let n = x.len();
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            out.fill(f32::NAN);
            return;
        }
        let v = f64::from(v);
        let delta = v - mean;
        mean += delta / (i + 1) as f64;
        m2 += delta * (v - mean);
    }
    let var = m2 / n as f64;
    let rstd = 1.0 / (var + eps).sqrt();
    if !rstd.is_finite() {
        out.fill(f32::NAN);
        return;
    }
    for (((dst, &xi), &w), &b) in out
        .iter_mut()
        .zip(x.iter())
        .zip(weight.iter())
        .zip(bias.iter())
    {
        *dst = saturate_f32((f64::from(xi) - mean) * rstd * f64::from(w) + f64::from(b));
    }
}

/// RMSNorm one row with an `f64` mean-square.
///
/// `eps` must already have been accepted by [`validate_norm_eps`]. The public
/// path is [`crate::tensor::ops::try_rms_norm`].
pub(crate) fn rms_norm_row(x: &[f32], weight: &[f32], eps: f64, out: &mut [f32]) {
    assert_eq!(
        x.len(),
        weight.len(),
        "rms_norm_row: weight length mismatch"
    );
    assert_eq!(x.len(), out.len(), "rms_norm_row: out length mismatch");
    assert!(!x.is_empty(), "rms_norm_row: empty axis");

    let n = x.len();
    let mut sumsq = 0.0f64;
    for &v in x {
        if !v.is_finite() {
            out.fill(f32::NAN);
            return;
        }
        let v = f64::from(v);
        sumsq += v * v;
    }
    let rstd = 1.0 / (sumsq / n as f64 + eps).sqrt();
    if !rstd.is_finite() {
        out.fill(f32::NAN);
        return;
    }
    for ((dst, &xi), &w) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *dst = saturate_f32(f64::from(xi) * rstd * f64::from(w));
    }
}

/// L2-normalize `values` in place according to the crate policy.
pub(crate) fn l2_normalize(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    if values.iter().any(|v| !v.is_finite()) {
        values.fill(f32::NAN);
        return;
    }
    let mut sumsq = 0.0f64;
    for &v in values.iter() {
        let v = f64::from(v);
        sumsq += v * v;
    }
    let norm = sumsq.sqrt();
    if !norm.is_finite() || norm <= L2_NORM_FLOOR {
        return;
    }
    for v in values.iter_mut() {
        *v = (f64::from(*v) / norm) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::tensor::ops::{try_layer_norm, try_rms_norm};

    const EPS: f32 = 1e-5;

    fn bits_eq(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "bit mismatch at {i}: {x} vs {y}");
        }
    }

    fn all_finite(xs: &[f32]) -> bool {
        xs.iter().all(|x| x.is_finite())
    }

    fn softmax_oracle(logits: &[f32]) -> Vec<f64> {
        if logits.is_empty() {
            return Vec::new();
        }
        if logits.iter().any(|x| x.is_nan()) {
            return vec![f64::NAN; logits.len()];
        }
        let pos_inf: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_infinite() && x.is_sign_positive())
            .map(|(i, _)| i)
            .collect();
        if !pos_inf.is_empty() {
            let p = 1.0 / pos_inf.len() as f64;
            let mut out = vec![0.0f64; logits.len()];
            for i in pos_inf {
                out[i] = p;
            }
            return out;
        }
        if logits
            .iter()
            .all(|x| x.is_infinite() && x.is_sign_negative())
        {
            return vec![1.0 / logits.len() as f64; logits.len()];
        }
        let mut max_v = f32::NEG_INFINITY;
        for &x in logits {
            if x > max_v {
                max_v = x;
            }
        }
        let max64 = f64::from(max_v);
        let exps: Vec<f64> = logits
            .iter()
            .map(|&x| (f64::from(x) - max64).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    }

    fn layer_norm_oracle(x: &[f32], weight: &[f32], bias: &[f32], eps: f64) -> Vec<f64> {
        if x.iter().any(|v| !v.is_finite()) {
            return vec![f64::NAN; x.len()];
        }
        let n = x.len() as f64;
        let mean: f64 = x.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
        let var: f64 = x
            .iter()
            .map(|v| {
                let d = f64::from(*v) - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let rstd = 1.0 / (var + eps).sqrt();
        x.iter()
            .zip(weight.iter())
            .zip(bias.iter())
            .map(|((&xi, &w), &b)| (f64::from(xi) - mean) * rstd * f64::from(w) + f64::from(b))
            .collect()
    }

    fn rms_norm_oracle(x: &[f32], weight: &[f32], eps: f64) -> Vec<f64> {
        if x.iter().any(|v| !v.is_finite()) {
            return vec![f64::NAN; x.len()];
        }
        let n = x.len() as f64;
        let ms: f64 = x
            .iter()
            .map(|v| {
                let v = f64::from(*v);
                v * v
            })
            .sum::<f64>()
            / n;
        let rstd = 1.0 / (ms + eps).sqrt();
        x.iter()
            .zip(weight.iter())
            .map(|(&xi, &w)| f64::from(xi) * rstd * f64::from(w))
            .collect()
    }

    /// Tiny deterministic generator so property tests do not depend on `rand`.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        fn next_unit(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
        }

        fn next_logit(&mut self) -> f32 {
            self.next_unit() * 40.0 - 20.0
        }
    }

    #[test]
    fn softmax_empty_is_empty() {
        assert!(softmax_vec(&[]).is_empty());
    }

    #[test]
    fn softmax_finite_baseline_matches_closed_form() {
        let y = softmax_vec(&[1.0, 2.0, 3.0]);
        let s = 1.0f64 + 1.0f64.exp() + 2.0f64.exp();
        let expected = [1.0 / s, 1.0f64.exp() / s, 2.0f64.exp() / s];
        let sum: f32 = y.iter().sum();
        assert!((sum - 1.0).abs() < SOFTMAX_SUM_TOLERANCE);
        for (got, exp) in y.iter().zip(expected) {
            assert!((f64::from(*got) - exp).abs() < 1e-6);
        }
        assert!(y.iter().all(|p| *p >= 0.0));
    }

    #[test]
    fn softmax_large_magnitudes_stay_finite_and_normalized() {
        let y = softmax_vec(&[1e30, 1e30, 0.0, -1e30]);
        assert!(all_finite(&y));
        assert!(y.iter().all(|p| *p >= 0.0));
        let sum: f32 = y.iter().sum();
        assert!((sum - 1.0).abs() < SOFTMAX_SUM_TOLERANCE);
        assert!((y[0] - 0.5).abs() < 1e-5);
        assert!((y[1] - 0.5).abs() < 1e-5);
        assert!(y[2].abs() < 1e-6);
        assert!(y[3].abs() < 1e-6);
    }

    #[test]
    fn softmax_constant_row_is_uniform() {
        let y = softmax_vec(&[4.0, 4.0, 4.0, 4.0]);
        for p in &y {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_long_uniform_row_respects_sum_tolerance() {
        let n = 4051;
        let y = softmax_vec(&vec![0.0; n]);
        assert_eq!(y.len(), n);
        assert!(y.iter().all(|p| p.is_finite() && *p >= 0.0));
        let sum: f32 = y.iter().copied().sum();
        assert!(
            (sum - 1.0).abs() < SOFTMAX_SUM_TOLERANCE,
            "sum={sum} for n={n}"
        );
        let masked = softmax_vec(&vec![f32::NEG_INFINITY; n]);
        let masked_sum: f32 = masked.iter().copied().sum();
        assert!((masked_sum - 1.0).abs() < SOFTMAX_SUM_TOLERANCE);
        let infs = softmax_vec(&vec![f32::INFINITY; 4051]);
        let inf_sum: f32 = infs.iter().copied().sum();
        assert!((inf_sum - 1.0).abs() < SOFTMAX_SUM_TOLERANCE);
    }

    #[test]
    fn softmax_all_neg_inf_is_uniform() {
        let y = softmax_vec(&[f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        assert!(all_finite(&y));
        for p in &y {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_single_pos_inf_is_one_hot() {
        let y = softmax_vec(&[1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY]);
        bits_eq(&y, &[0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn softmax_multiple_pos_inf_split_equally() {
        let y = softmax_vec(&[f32::INFINITY, 0.0, f32::INFINITY, f32::NEG_INFINITY]);
        bits_eq(&y, &[0.5, 0.0, 0.5, 0.0]);
        let y3 = softmax_vec(&[f32::INFINITY, f32::INFINITY, f32::INFINITY]);
        for p in &y3 {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
        let sum: f32 = y3.iter().sum();
        assert!((sum - 1.0).abs() < SOFTMAX_SUM_TOLERANCE);
    }

    #[test]
    fn softmax_nan_fills_the_row() {
        let y = softmax_vec(&[1.0, f32::NAN, 2.0]);
        assert!(y.iter().all(|p| p.is_nan()));
        // NaN must not be treated as a silent max / tie with f32::max.
        assert!(f32::NAN.max(2.0) == 2.0);
        assert!(y.iter().any(|p| p.is_nan()));
    }

    #[test]
    fn softmax_matches_f64_oracle() {
        let cases: &[&[f32]] = &[
            &[1.0, 2.0, 3.0],
            &[0.0, 0.0, 0.0],
            &[1e30, -1e30, 0.0],
            &[10000.0, 10001.0, 10002.0],
            &[f32::NEG_INFINITY, 0.0, 1.0],
            &[f32::INFINITY, f32::INFINITY],
            &[f32::NEG_INFINITY, f32::NEG_INFINITY],
        ];
        for logits in cases {
            let got = softmax_vec(logits);
            let oracle = softmax_oracle(logits);
            for (g, o) in got.iter().zip(oracle.iter()) {
                if o.is_nan() {
                    assert!(g.is_nan());
                } else {
                    assert!((f64::from(*g) - *o).abs() < 1e-6, "{g} vs {o}");
                }
            }
        }
    }

    #[test]
    fn softmax_is_deterministic_across_repeats() {
        let logits = [1.25f32, -3.5, 8.0, 8.0, -1e30, 1e30];
        let first = softmax_vec(&logits);
        for _ in 0..8 {
            bits_eq(&first, &softmax_vec(&logits));
        }
    }

    #[test]
    fn property_softmax_nonneg_and_normalized() {
        let mut rng = SplitMix64(0xC0FFEE);
        for _ in 0..256 {
            let n = (rng.next_u64() as usize % 32) + 1;
            let logits: Vec<f32> = (0..n).map(|_| rng.next_logit()).collect();
            let y = softmax_vec(&logits);
            assert_eq!(y.len(), n);
            assert!(y.iter().all(|p| p.is_finite() && *p >= 0.0));
            let sum: f32 = y.iter().sum();
            assert!(
                (sum - 1.0).abs() < SOFTMAX_SUM_TOLERANCE,
                "sum={sum} n={n} logits={logits:?} y={y:?}"
            );
        }
    }

    #[test]
    fn try_layer_norm_rejects_invalid_eps() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let w = Tensor::ones(&[3]);
        let b = Tensor::zeros(&[3]);
        for eps in [0.0, -1e-5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = try_layer_norm(&x, &w, &b, eps).unwrap_err();
            assert!(matches!(err, CortexError::InvalidEpsilon { .. }), "{err}");
        }
    }

    #[test]
    fn try_rms_norm_rejects_invalid_eps() {
        let x = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]);
        let w = Tensor::ones(&[2]);
        let err = try_rms_norm(&x, &w, 0.0).unwrap_err();
        assert!(matches!(err, CortexError::InvalidEpsilon { eps } if eps == 0.0));
    }

    #[test]
    fn try_layer_norm_rejects_zero_width() {
        let x = Tensor::from_vec(vec![], &[3, 0]);
        let w = Tensor::from_vec(vec![], &[0]);
        let b = Tensor::from_vec(vec![], &[0]);
        let err = try_layer_norm(&x, &w, &b, EPS).unwrap_err();
        assert!(matches!(
            err,
            CortexError::ZeroWidth {
                op: "layer_norm",
                axis: 1,
            }
        ));
    }

    #[test]
    fn try_rms_norm_rejects_zero_width() {
        let x = Tensor::from_vec(vec![], &[2, 0]);
        let w = Tensor::from_vec(vec![], &[0]);
        let err = try_rms_norm(&x, &w, EPS).unwrap_err();
        assert!(matches!(
            err,
            CortexError::ZeroWidth {
                op: "rms_norm",
                axis: 1
            }
        ));
    }

    #[test]
    fn layer_norm_empty_batch_is_ok() {
        let x = Tensor::from_vec(vec![], &[0, 4]);
        let w = Tensor::ones(&[4]);
        let b = Tensor::zeros(&[4]);
        let y = try_layer_norm(&x, &w, &b, EPS).unwrap();
        assert_eq!(y.shape(), &[0, 4]);
        assert!(y.data().is_empty());
    }

    #[test]
    fn layer_norm_constant_row_is_bias() {
        let x = Tensor::from_vec(vec![3.0, 3.0, 3.0], &[1, 3]);
        let w = Tensor::ones(&[3]);
        let b = Tensor::from_vec(vec![0.1, -0.2, 0.3], &[3]);
        let y = try_layer_norm(&x, &w, &b, EPS).unwrap();
        for (got, want) in y.data().iter().zip(b.data()) {
            assert!((got - want).abs() < 1e-5);
        }
        assert!(all_finite(y.data()));
    }

    #[test]
    fn layer_norm_large_offset_small_spread_is_stable() {
        let x = Tensor::from_vec(vec![10000.0, 10001.0, 10002.0], &[1, 3]);
        let w = Tensor::ones(&[3]);
        let b = Tensor::zeros(&[3]);
        let y = try_layer_norm(&x, &w, &b, EPS).unwrap();
        assert!(all_finite(y.data()));
        let mean: f32 = y.data().iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-4);
        let oracle = layer_norm_oracle(x.data(), w.data(), b.data(), f64::from(EPS));
        for (g, o) in y.data().iter().zip(oracle.iter()) {
            assert!((f64::from(*g) - *o).abs() < 1e-5, "{g} vs {o}");
        }
    }

    #[test]
    fn layer_norm_near_1e30_stays_finite() {
        let x = Tensor::from_vec(vec![1e30, -1e30, 0.0], &[1, 3]);
        let w = Tensor::ones(&[3]);
        let b = Tensor::zeros(&[3]);
        let y = try_layer_norm(&x, &w, &b, EPS).unwrap();
        assert!(all_finite(y.data()));
        let oracle = layer_norm_oracle(x.data(), w.data(), b.data(), f64::from(EPS));
        for (g, o) in y.data().iter().zip(oracle.iter()) {
            assert!((f64::from(*g) - *o).abs() < 1e-4, "{g} vs {o}");
        }
    }

    #[test]
    fn layer_norm_nan_row_is_nan() {
        let x = Tensor::from_vec(vec![1.0, f32::NAN, 3.0], &[1, 3]);
        let w = Tensor::ones(&[3]);
        let b = Tensor::zeros(&[3]);
        let y = try_layer_norm(&x, &w, &b, EPS).unwrap();
        assert!(y.data().iter().all(|v| v.is_nan()));
    }

    #[test]
    fn layer_norm_extreme_affine_stays_finite() {
        let x = Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]);
        let w = Tensor::from_vec(vec![f32::MAX, 1.0, 1.0], &[3]);
        let b = Tensor::zeros(&[3]);
        let y = try_layer_norm(&x, &w, &b, EPS).unwrap();
        assert!(all_finite(y.data()));
        assert_eq!(y.data()[0], f32::MAX);
        let rn = try_rms_norm(&x, &w, EPS).unwrap();
        assert!(all_finite(rn.data()));
        assert_eq!(rn.data()[0], f32::MAX);
    }

    #[test]
    fn layer_norm_inf_row_is_nan() {
        let x = Tensor::from_vec(vec![1.0, f32::INFINITY, -2.0], &[1, 3]);
        let w = Tensor::ones(&[3]);
        let b = Tensor::zeros(&[3]);
        let y = try_layer_norm(&x, &w, &b, EPS).unwrap();
        assert!(y.data().iter().all(|v| v.is_nan()));
    }

    #[test]
    fn rms_norm_large_magnitude_does_not_overflow() {
        // f32 mean-square of 1e30 is 1e60, which overflows f32 (~1e38).
        let x = Tensor::from_vec(vec![1e30, -1e30], &[1, 2]);
        let w = Tensor::ones(&[2]);
        let y = try_rms_norm(&x, &w, EPS).unwrap();
        assert!(all_finite(y.data()));
        let oracle = rms_norm_oracle(x.data(), w.data(), f64::from(EPS));
        for (g, o) in y.data().iter().zip(oracle.iter()) {
            assert!((f64::from(*g) - *o).abs() < 1e-5, "{g} vs {o}");
        }
    }

    #[test]
    fn rms_norm_constant_row_is_finite() {
        let x = Tensor::from_vec(vec![5.0, 5.0, 5.0], &[1, 3]);
        let w = Tensor::ones(&[3]);
        let y = try_rms_norm(&x, &w, EPS).unwrap();
        assert!(all_finite(y.data()));
        let oracle = rms_norm_oracle(x.data(), w.data(), f64::from(EPS));
        for (g, o) in y.data().iter().zip(oracle.iter()) {
            assert!((f64::from(*g) - *o).abs() < 1e-5);
        }
    }

    #[test]
    fn rms_norm_nan_row_is_nan() {
        let x = Tensor::from_vec(vec![1.0, f32::NAN], &[1, 2]);
        let w = Tensor::ones(&[2]);
        let y = try_rms_norm(&x, &w, EPS).unwrap();
        assert!(y.data().iter().all(|v| v.is_nan()));
    }

    #[test]
    fn property_norm_finite_inputs_produce_finite_outputs() {
        let mut rng = SplitMix64(0xDEAD_BEEF);
        for _ in 0..128 {
            let n = (rng.next_u64() as usize % 24) + 1;
            let x: Vec<f32> = (0..n).map(|_| rng.next_logit()).collect();
            let w: Vec<f32> = (0..n).map(|_| rng.next_unit() + 0.25).collect();
            let b: Vec<f32> = (0..n).map(|_| rng.next_logit() * 0.1).collect();
            let xt = Tensor::from_vec(x.clone(), &[1, n]);
            let wt = Tensor::from_vec(w.clone(), &[n]);
            let bt = Tensor::from_vec(b.clone(), &[n]);
            let ln = try_layer_norm(&xt, &wt, &bt, EPS).unwrap();
            let rn = try_rms_norm(&xt, &wt, EPS).unwrap();
            assert!(all_finite(ln.data()), "layer_norm {x:?}");
            assert!(all_finite(rn.data()), "rms_norm {x:?}");
        }
    }

    #[test]
    fn layer_norm_and_rms_norm_are_deterministic() {
        let x = Tensor::from_vec(vec![1.0, -2.0, 3.5, 1e-3], &[1, 4]);
        let w = Tensor::from_vec(vec![0.9, 1.1, 1.0, 0.5], &[4]);
        let b = Tensor::from_vec(vec![0.0, 0.1, -0.1, 0.0], &[4]);
        let ln = try_layer_norm(&x, &w, &b, EPS).unwrap();
        let rn = try_rms_norm(&x, &w, EPS).unwrap();
        for _ in 0..8 {
            bits_eq(ln.data(), try_layer_norm(&x, &w, &b, EPS).unwrap().data());
            bits_eq(rn.data(), try_rms_norm(&x, &w, EPS).unwrap().data());
        }
    }

    #[test]
    fn l2_normalize_zeros_stay_zeros() {
        let mut v = vec![0.0f32; 4];
        l2_normalize(&mut v);
        bits_eq(&v, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_normalize_nan_fills() {
        let mut v = vec![1.0, f32::NAN, 0.0];
        l2_normalize(&mut v);
        assert!(v.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn l2_normalize_large_values_stay_finite() {
        let mut v = vec![1e30, -1e30, 0.0];
        l2_normalize(&mut v);
        assert!(all_finite(&v));
        let n = v
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        assert!((n - 1.0).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_is_deterministic() {
        let src = [0.2f32, -0.5, 0.1, 4.0];
        let mut a = src;
        l2_normalize(&mut a);
        for _ in 0..8 {
            let mut b = src;
            l2_normalize(&mut b);
            bits_eq(&a, &b);
        }
    }

    #[test]
    fn existing_layer_norm_fixture_mean_near_zero() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let w = Tensor::ones(&[3]);
        let b = Tensor::zeros(&[3]);
        let y = try_layer_norm(&x, &w, &b, EPS).unwrap();
        let row0_mean: f32 = y.data()[0..3].iter().sum::<f32>() / 3.0;
        assert!(row0_mean.abs() < 1e-4);
        assert!(all_finite(y.data()));
    }
}
