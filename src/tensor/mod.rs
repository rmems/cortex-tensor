// SPDX-License-Identifier: MIT OR Apache-2.0

pub mod ops;

use crate::error::{CortexError, Result, unwrap_compat};
use serde::{Deserialize, Serialize};
use std::alloc::Layout;
use std::fmt;

/// Row-major dense tensor — candle-core replacement.
///
/// Stores `f32` data in a contiguous `Vec<f32>` with an arbitrary shape.
/// All operations are CPU-first; the `gpu` feature flag will add CUDA kernels later.
///
/// # Construction
///
/// Prefer [`Tensor::try_from_vec`], which validates the element count and
/// row-major strides with checked arithmetic before storing `data`.
/// [`Tensor::from_vec`] and the other panic-style constructors remain as
/// pre-1.0 compatibility wrappers; migrate new call sites to `try_from_vec`.
#[derive(Clone, Serialize, Deserialize)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    // ── Constructors ─────────────────────────────────────────────────

    /// Constructs a tensor from a contiguous row-major buffer and shape.
    ///
    /// # Panics
    ///
    /// Panics if the shape product or stride arithmetic overflows `usize`, or
    /// if `data.len()` does not equal the number of elements implied by
    /// `shape`.
    ///
    /// Prefer [`Tensor::try_from_vec`] in new code. This wrapper is retained
    /// for pre-1.0 source compatibility and will remain until a separate
    /// SemVer decision removes panic-on-invalid-shape constructors.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        unwrap_compat(Self::try_from_vec(data, shape), "Tensor::from_vec")
    }

    /// Fallible constructor: proves `numel` and strides are representable
    /// before taking ownership of `data`.
    ///
    /// Does not allocate a new data buffer. Overflowing shapes therefore fail
    /// without attempting a giant allocation.
    pub fn try_from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let numel = checked_numel(shape)?;
        let strides = try_compute_strides(shape)?;
        if data.len() != numel {
            return Err(CortexError::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![data.len()],
            });
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
            strides,
        })
    }

    /// Fills a tensor with zeros.
    ///
    /// # Panics
    ///
    /// Panics if the shape product or stride arithmetic overflows `usize`.
    /// Prefer [`Tensor::try_from_vec`] with a caller-allocated buffer in new
    /// code.
    pub fn zeros(shape: &[usize]) -> Self {
        unwrap_compat(try_filled(shape, 0.0), "Tensor::zeros")
    }

    /// Fills a tensor with ones.
    ///
    /// # Panics
    ///
    /// Panics if the shape product or stride arithmetic overflows `usize`.
    /// See [`Tensor::from_vec`] for the pre-1.0 compatibility policy.
    pub fn ones(shape: &[usize]) -> Self {
        unwrap_compat(try_filled(shape, 1.0), "Tensor::ones")
    }

    /// Fills a tensor with `val`.
    ///
    /// # Panics
    ///
    /// Panics if the shape product or stride arithmetic overflows `usize`.
    /// See [`Tensor::from_vec`] for the pre-1.0 compatibility policy.
    pub fn full(shape: &[usize], val: f32) -> Self {
        unwrap_compat(try_filled(shape, val), "Tensor::full")
    }

    /// Fills a tensor with i.i.d. normal samples.
    ///
    /// # Panics
    ///
    /// Panics if the shape product or stride arithmetic overflows `usize`.
    /// See [`Tensor::from_vec`] for the pre-1.0 compatibility policy.
    pub fn randn(shape: &[usize], mean: f32, std: f32) -> Self {
        use rand::RngExt;
        let numel = unwrap_compat(checked_numel(shape), "Tensor::randn");
        unwrap_compat(check_f32_alloc(numel, shape), "Tensor::randn");
        let _strides = unwrap_compat(try_compute_strides(shape), "Tensor::randn");
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                // Box-Muller transform
                let u1: f32 = rng.random_range(0.0f32..1.0).max(1e-7);
                let u2: f32 = rng.random_range(0.0f32..1.0);
                let z = (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
                mean + std * z
            })
            .collect();
        Self::from_vec(data, shape)
    }

    // ── Accessors ────────────────────────────────────────────────────

    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    // ── Reshape / view ───────────────────────────────────────────────

    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let numel = unwrap_compat(checked_numel(new_shape), "Tensor::reshape");
        assert_eq!(numel, self.numel(), "reshape: element count mismatch");
        Self::from_vec(self.data.clone(), new_shape)
    }

    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2, "transpose requires 2-D tensor");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let out_len = unwrap_compat(checked_numel(&[rows, cols]), "Tensor::transpose");
        let mut out = vec![0.0f32; out_len];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = self.data[r * cols + c];
            }
        }
        Self::from_vec(out, &[cols, rows])
    }

    // ── Element-wise ops ─────────────────────────────────────────────

    pub fn add(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "add: shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn sub(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "sub: shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn mul(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "mul: shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn scale(&self, s: f32) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x * s).collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn add_scalar(&self, s: f32) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x + s).collect();
        Self::from_vec(data, &self.shape)
    }

    // ── Activation functions ─────────────────────────────────────────

    pub fn relu(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x.max(0.0)).collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn gelu(&self) -> Self {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                let c = (2.0f32 / std::f32::consts::PI).sqrt();
                x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
            })
            .collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn silu(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        Self::from_vec(data, &self.shape)
    }

    /// Fast sigmoid surrogate gradient (from soma-engine's E-prop)
    pub fn fast_sigmoid(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|&x| x / (1.0 + x.abs())).collect();
        Self::from_vec(data, &self.shape)
    }

    // ── Reductions ───────────────────────────────────────────────────

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.numel() as f32
    }

    pub fn max_val(&self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Softmax along the last axis (works on 1-D or flattened last dim of 2-D).
    pub fn softmax_last(&self) -> Self {
        assert!(self.ndim() <= 2, "softmax_last: max 2-D");
        if self.ndim() == 1 {
            let max_v = self.max_val();
            let exp: Vec<f32> = self.data.iter().map(|x| (x - max_v).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let data: Vec<f32> = exp.iter().map(|e| e / sum).collect();
            return Self::from_vec(data, &self.shape);
        }
        // 2-D: softmax each row
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let row_start = r * cols;
            let row = &self.data[row_start..row_start + cols];
            let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = row.iter().map(|x| (x - max_v).exp()).collect();
            let sum: f32 = exp.iter().sum();
            for c in 0..cols {
                data[row_start + c] = exp[c] / sum;
            }
        }
        Self::from_vec(data, &self.shape)
    }

    // ── Row / slice access ───────────────────────────────────────────

    pub fn row(&self, idx: usize) -> Self {
        assert_eq!(self.ndim(), 2);
        let cols = self.shape[1];
        let start = idx * cols;
        Self::from_vec(self.data[start..start + cols].to_vec(), &[cols])
    }
}

/// Product of `shape` using checked arithmetic.
///
/// Any zero axis yields `0` without multiplying the remaining extents, so a
/// trailing zero still produces an empty tensor when an earlier product would
/// overflow (for example `[3, usize::MAX, 0]`).
pub(crate) fn checked_numel(shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| CortexError::SizeOverflow {
                shape: shape.to_vec(),
            })
    })
}

/// Rejects a `Vec<f32>` length whose byte size exceeds `isize::MAX`.
pub(crate) fn check_f32_alloc(numel: usize, shape: &[usize]) -> Result<()> {
    Layout::array::<f32>(numel)
        .map(|_| ())
        .map_err(|_| CortexError::SizeOverflow {
            shape: shape.to_vec(),
        })
}

/// Row-major strides using checked arithmetic.
///
/// The last stride is `1`. Earlier strides are the product of all following
/// dimensions. A zero-sized axis can still overflow a leading stride (for
/// example `[0, usize::MAX, 2]`).
pub(crate) fn try_compute_strides(shape: &[usize]) -> Result<Vec<usize>> {
    if shape.is_empty() {
        return Ok(Vec::new());
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] =
            strides[i + 1]
                .checked_mul(shape[i + 1])
                .ok_or_else(|| CortexError::SizeOverflow {
                    shape: shape.to_vec(),
                })?;
    }
    Ok(strides)
}

/// Allocates a filled buffer only after numel and strides are representable.
fn try_filled(shape: &[usize], val: f32) -> Result<Tensor> {
    let numel = checked_numel(shape)?;
    check_f32_alloc(numel, shape)?;
    let strides = try_compute_strides(shape)?;
    Ok(Tensor {
        data: vec![val; numel],
        shape: shape.to_vec(),
        strides,
    })
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, data=[", self.shape)?;
        let n = self.data.len().min(8);
        for (i, v) in self.data[..n].iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:.4}")?;
        }
        if self.data.len() > 8 {
            write!(f, ", ...")?;
        }
        write!(f, "])")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec_and_shape() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.strides(), &[3, 1]);
    }

    #[test]
    fn try_from_vec_matches_from_vec_on_valid_input() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::from_vec(data.clone(), &[2, 3]);
        let b = Tensor::try_from_vec(data, &[2, 3]).unwrap();
        assert_eq!(a.data(), b.data());
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.strides(), b.strides());
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tt = t.transpose();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
        let s = t.softmax_last();
        let sum: f32 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gelu() {
        let t = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]);
        let g = t.gelu();
        assert!((g.data()[0] - 0.0).abs() < 1e-4);
        assert!(g.data()[1] > 0.8); // GELU(1) ≈ 0.841
    }

    #[test]
    fn try_from_vec_accepts_empty_and_zero_axes() {
        let empty = Tensor::try_from_vec(vec![], &[0, 3]).unwrap();
        assert_eq!(empty.numel(), 0);
        assert_eq!(empty.shape(), &[0, 3]);
        assert_eq!(empty.strides(), &[3, 1]);

        let zero_1d = Tensor::try_from_vec(vec![], &[0]).unwrap();
        assert_eq!(zero_1d.numel(), 0);
        assert_eq!(zero_1d.strides(), &[1]);

        let scalar = Tensor::try_from_vec(vec![3.5], &[]).unwrap();
        assert_eq!(scalar.numel(), 1);
        assert!(scalar.strides().is_empty());
        assert_eq!(scalar.data(), &[3.5]);
    }

    #[test]
    fn try_from_vec_error_table() {
        struct Case {
            name: &'static str,
            data: Vec<f32>,
            shape: Vec<usize>,
            check: fn(&CortexError) -> bool,
        }
        let cases = [
            Case {
                name: "length mismatch",
                data: vec![1.0, 2.0],
                shape: vec![2, 3],
                check: |e| {
                    matches!(
                        e,
                        CortexError::ShapeMismatch {
                            expected,
                            got
                        } if expected == &[2, 3] && got == &[2]
                    )
                },
            },
            Case {
                name: "empty data for non-empty shape",
                data: vec![],
                shape: vec![4],
                check: |e| matches!(e, CortexError::ShapeMismatch { .. }),
            },
            Case {
                name: "scalar requires one element",
                data: vec![],
                shape: vec![],
                check: |e| {
                    matches!(
                        e,
                        CortexError::ShapeMismatch { expected, got }
                            if expected.is_empty() && got == &[0]
                    )
                },
            },
            Case {
                name: "product overflow",
                data: vec![],
                shape: vec![usize::MAX, 2],
                check: |e| {
                    matches!(
                        e,
                        CortexError::SizeOverflow { shape } if shape == &[usize::MAX, 2]
                    )
                },
            },
            Case {
                name: "stride overflow with zero numel",
                data: vec![],
                shape: vec![0, usize::MAX, 2],
                check: |e| matches!(e, CortexError::SizeOverflow { .. }),
            },
        ];
        for case in cases {
            let err = Tensor::try_from_vec(case.data, &case.shape).expect_err(case.name);
            assert!((case.check)(&err), "{}: {err}", case.name);
        }
    }

    #[test]
    fn overflow_shapes_fail_without_giant_allocation() {
        let shapes: &[&[usize]] = &[
            &[usize::MAX, 2],
            &[2, usize::MAX],
            &[usize::MAX, usize::MAX],
            &[usize::MAX / 2 + 1, 2],
            &[0, usize::MAX, 2],
            &[0, 2, usize::MAX],
        ];
        for shape in shapes {
            let err = Tensor::try_from_vec(Vec::new(), shape).unwrap_err();
            assert!(
                matches!(err, CortexError::SizeOverflow { shape: ref got } if got.as_slice() == *shape),
                "shape {shape:?} -> {err}"
            );
        }

        // A zero axis can cancel a huge leading extent so numel is 0 and
        // strides stay representable.
        let cancelled = Tensor::try_from_vec(vec![], &[usize::MAX, 0, 2]).unwrap();
        assert_eq!(cancelled.numel(), 0);
        assert_eq!(cancelled.strides(), &[0, 2, 1]);

        // A trailing zero must not be lost to a left-to-right overflow.
        let trailing = Tensor::try_from_vec(vec![], &[3, usize::MAX, 0]).unwrap();
        assert_eq!(trailing.numel(), 0);
        assert_eq!(trailing.strides(), &[0, 0, 1]);
    }

    #[test]
    fn zeros_overflow_panics_without_allocating() {
        let panicked = std::panic::catch_unwind(|| {
            let _ = Tensor::zeros(&[usize::MAX, 2]);
        });
        assert!(panicked.is_err());
    }

    #[test]
    fn property_accepted_shapes_match_numel_and_strides() {
        fn check(shape: &[usize]) {
            let numel = match checked_numel(shape) {
                Ok(n) => n,
                Err(CortexError::SizeOverflow { shape: got }) => {
                    assert_eq!(got, shape);
                    return;
                }
                Err(other) => panic!("numel check failed unexpectedly: {other}"),
            };
            match try_compute_strides(shape) {
                Ok(strides) => {
                    let t = Tensor::try_from_vec(vec![0.5; numel], shape)
                        .expect("shape with representable numel and strides must construct");
                    assert_eq!(t.numel(), numel);
                    assert_eq!(t.data().len(), numel);
                    assert_eq!(t.shape(), shape);
                    assert_eq!(t.strides(), strides.as_slice());
                    if shape.is_empty() {
                        assert!(strides.is_empty());
                    } else {
                        assert_eq!(*strides.last().unwrap(), 1);
                        for i in (0..shape.len().saturating_sub(1)).rev() {
                            let expected = strides[i + 1]
                                .checked_mul(shape[i + 1])
                                .expect("successful stride vector is overflow-free");
                            assert_eq!(strides[i], expected);
                        }
                    }
                }
                Err(CortexError::SizeOverflow { .. }) => {
                    assert!(Tensor::try_from_vec(vec![0.5; numel], shape).is_err());
                }
                Err(other) => panic!("stride check failed unexpectedly: {other}"),
            }
        }

        fn rec(prefix: &mut Vec<usize>, max_rank: usize, max_dim: usize) {
            check(prefix);
            if prefix.len() >= max_rank {
                return;
            }
            for d in 0..=max_dim {
                prefix.push(d);
                rec(prefix, max_rank, max_dim);
                prefix.pop();
            }
        }

        rec(&mut Vec::new(), 4, 4);
    }
}
