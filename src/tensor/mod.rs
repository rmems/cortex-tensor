// SPDX-License-Identifier: MIT OR Apache-2.0

pub mod finite;
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
    /// Prefer [`Tensor::try_zeros`] in new code.
    pub fn zeros(shape: &[usize]) -> Self {
        unwrap_compat(Self::try_zeros(shape), "Tensor::zeros")
    }

    /// Fallible [`Tensor::zeros`].
    pub fn try_zeros(shape: &[usize]) -> Result<Self> {
        try_filled(shape, 0.0)
    }

    /// Fills a tensor with ones.
    ///
    /// # Panics
    ///
    /// Panics if the shape product or stride arithmetic overflows `usize`.
    /// See [`Tensor::from_vec`] for the pre-1.0 compatibility policy.
    /// Prefer [`Tensor::try_ones`] in new code.
    pub fn ones(shape: &[usize]) -> Self {
        unwrap_compat(Self::try_ones(shape), "Tensor::ones")
    }

    /// Fallible [`Tensor::ones`].
    pub fn try_ones(shape: &[usize]) -> Result<Self> {
        try_filled(shape, 1.0)
    }

    /// Fills a tensor with `val`.
    ///
    /// # Panics
    ///
    /// Panics if the shape product or stride arithmetic overflows `usize`.
    /// See [`Tensor::from_vec`] for the pre-1.0 compatibility policy.
    /// Prefer [`Tensor::try_full`] in new code.
    pub fn full(shape: &[usize], val: f32) -> Self {
        unwrap_compat(Self::try_full(shape, val), "Tensor::full")
    }

    /// Fallible [`Tensor::full`].
    pub fn try_full(shape: &[usize], val: f32) -> Result<Self> {
        try_filled(shape, val)
    }

    /// Fills a tensor with i.i.d. normal samples.
    ///
    /// # Panics
    ///
    /// Panics if the shape product or stride arithmetic overflows `usize`.
    /// See [`Tensor::from_vec`] for the pre-1.0 compatibility policy.
    pub fn randn(shape: &[usize], mean: f32, std: f32) -> Self {
        unwrap_compat(Self::try_randn(shape, mean, std), "Tensor::randn")
    }

    /// Fallible [`Tensor::randn`]: fails on shape/stride overflow before
    /// allocating.
    pub fn try_randn(shape: &[usize], mean: f32, std: f32) -> Result<Self> {
        use rand::RngExt;
        let numel = checked_numel(shape)?;
        check_f32_alloc(numel, shape)?;
        let _strides = try_compute_strides(shape)?;
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
        Self::try_from_vec(data, shape)
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

    /// Reshapes to `new_shape`, which must imply the same element count.
    ///
    /// # Panics
    ///
    /// Panics if the element count changes or the new shape overflows.
    /// Prefer [`Tensor::try_reshape`] in new code.
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        unwrap_compat(self.try_reshape(new_shape), "Tensor::reshape")
    }

    /// Fallible reshape: `new_shape` must imply exactly `self.numel()`
    /// elements and have representable strides.
    pub fn try_reshape(&self, new_shape: &[usize]) -> Result<Self> {
        self.check_storage()?;
        let numel = checked_numel(new_shape)?;
        if numel != self.numel() {
            return Err(CortexError::ShapeMismatch {
                expected: self.shape.clone(),
                got: new_shape.to_vec(),
            });
        }
        Self::try_from_vec(self.data.clone(), new_shape)
    }

    /// Transposes a 2-D tensor: `[rows, cols]` → `[cols, rows]`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not rank-2. Prefer [`Tensor::try_transpose`].
    pub fn transpose(&self) -> Self {
        unwrap_compat(self.try_transpose(), "Tensor::transpose")
    }

    /// Fallible transpose: returns [`CortexError::RankMismatch`] unless
    /// `self` is rank-2.
    pub fn try_transpose(&self) -> Result<Self> {
        self.check_storage()?;
        if self.ndim() != 2 {
            return Err(CortexError::RankMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let out_len = checked_numel(&[rows, cols])?;
        let mut out = vec![0.0f32; out_len];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = self.data[r * cols + c];
            }
        }
        Self::try_from_vec(out, &[cols, rows])
    }

    // ── Element-wise ops ─────────────────────────────────────────────

    /// Element-wise addition. Both tensors must have identical shapes.
    ///
    /// # Panics
    ///
    /// Panics on shape mismatch. Prefer [`Tensor::try_add`] in new code.
    pub fn add(&self, other: &Tensor) -> Self {
        unwrap_compat(self.try_add(other), "Tensor::add")
    }

    /// Fallible element-wise addition.
    pub fn try_add(&self, other: &Tensor) -> Result<Self> {
        self.try_elementwise(other, |a, b| a + b)
    }

    /// Element-wise subtraction. Both tensors must have identical shapes.
    ///
    /// # Panics
    ///
    /// Panics on shape mismatch. Prefer [`Tensor::try_sub`] in new code.
    pub fn sub(&self, other: &Tensor) -> Self {
        unwrap_compat(self.try_sub(other), "Tensor::sub")
    }

    /// Fallible element-wise subtraction.
    pub fn try_sub(&self, other: &Tensor) -> Result<Self> {
        self.try_elementwise(other, |a, b| a - b)
    }

    /// Element-wise multiplication. Both tensors must have identical shapes.
    ///
    /// # Panics
    ///
    /// Panics on shape mismatch. Prefer [`Tensor::try_mul`] in new code.
    pub fn mul(&self, other: &Tensor) -> Self {
        unwrap_compat(self.try_mul(other), "Tensor::mul")
    }

    /// Fallible element-wise multiplication.
    pub fn try_mul(&self, other: &Tensor) -> Result<Self> {
        self.try_elementwise(other, |a, b| a * b)
    }

    fn try_elementwise(&self, other: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Self> {
        if self.shape != other.shape {
            return Err(CortexError::ShapeMismatch {
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Self::try_from_vec(data, &self.shape)
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

    /// Index of the largest element.
    ///
    /// NaN policy: elements are ordered with [`f32::total_cmp`] (IEEE 754
    /// `totalOrder`), which never yields `None`, so this function cannot
    /// panic on NaN inputs. A positive NaN sorts above `+Inf` and therefore
    /// wins; a negative NaN sorts below `-Inf`. Returns `0` for an empty
    /// tensor. Callers that need NaN rejection should screen inputs with
    /// [`f32::is_nan`] first.
    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Softmax along the last axis (1-D, 0-D, or each row of a 2-D tensor).
    ///
    /// See [`finite`] for the NaN / `±Inf` / all-masked policy. Panics if the
    /// rank is greater than 2; use [`Self::try_softmax_last`] to recover.
    pub fn softmax_last(&self) -> Self {
        self.try_softmax_last()
            .unwrap_or_else(|err| panic!("{err}"))
    }

    /// Fallible softmax along the last axis.
    ///
    /// Returns [`CortexError::InvalidConfig`] when `ndim > 2`.
    pub fn try_softmax_last(&self) -> Result<Self> {
        self.check_storage()?;
        if self.ndim() > 2 {
            return Err(CortexError::InvalidConfig(format!(
                "softmax_last: max 2-D, got rank {}",
                self.ndim()
            )));
        }
        if self.ndim() <= 1 {
            let mut data = vec![0.0f32; self.data.len()];
            finite::softmax_row(&self.data, &mut data);
            return Ok(Self::from_vec(data, &self.shape));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0f32; rows * cols];
        if cols == 0 {
            return Ok(Self::from_vec(data, &self.shape));
        }
        for r in 0..rows {
            let row_start = r * cols;
            finite::softmax_row(
                &self.data[row_start..row_start + cols],
                &mut data[row_start..row_start + cols],
            );
        }
        Ok(Self::from_vec(data, &self.shape))
    }

    // ── Row / slice access ───────────────────────────────────────────

    /// Copies row `idx` of a 2-D tensor into a 1-D tensor.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not rank-2 or `idx` is out of bounds. Prefer
    /// [`Tensor::try_row`] in new code.
    pub fn row(&self, idx: usize) -> Self {
        unwrap_compat(self.try_row(idx), "Tensor::row")
    }

    /// Fallible row access: returns [`CortexError::RankMismatch`] unless
    /// `self` is rank-2 and [`CortexError::IndexOutOfBounds`] when
    /// `idx >= shape[0]`.
    pub fn try_row(&self, idx: usize) -> Result<Self> {
        self.check_storage()?;
        if self.ndim() != 2 {
            return Err(CortexError::RankMismatch {
                expected: 2,
                got: self.ndim(),
            });
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        if idx >= rows {
            return Err(CortexError::IndexOutOfBounds {
                axis: 0,
                index: idx,
                size: rows,
            });
        }
        let start = idx * cols;
        Self::try_from_vec(self.data[start..start + cols].to_vec(), &[cols])
    }

    /// Rejects tensors whose stored `data` length disagrees with `shape`.
    ///
    /// `Tensor` derives `Deserialize` without invariant validation, so a
    /// malformed serialized value can reach methods that index `data`
    /// directly. Fallible methods that slice call this first.
    pub(crate) fn check_storage(&self) -> Result<()> {
        let expected = checked_numel(&self.shape).unwrap_or(usize::MAX);
        if self.data.len() != expected {
            return Err(CortexError::ShapeMismatch {
                expected: self.shape.clone(),
                got: vec![self.data.len()],
            });
        }
        Ok(())
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
    fn test_softmax_last_rejects_rank_3() {
        let t = Tensor::from_vec(vec![1.0; 8], &[2, 2, 2]);
        let err = t.try_softmax_last().unwrap_err();
        assert!(matches!(err, CortexError::InvalidConfig(_)));
    }

    #[test]
    fn test_softmax_2d_rows_and_all_masked() {
        let t = Tensor::from_vec(
            vec![
                1.0,
                2.0,
                3.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            &[2, 3],
        );
        let s = t.softmax_last();
        let row0: f32 = s.data()[0..3].iter().sum();
        assert!((row0 - 1.0).abs() < 1e-5);
        for p in &s.data()[3..6] {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn try_elementwise_ops_report_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        for (name, err) in [
            ("add", a.try_add(&b).unwrap_err()),
            ("sub", a.try_sub(&b).unwrap_err()),
            ("mul", a.try_mul(&b).unwrap_err()),
        ] {
            assert!(
                matches!(
                    err,
                    CortexError::ShapeMismatch { ref expected, ref got }
                        if expected == &[2, 2] && got == &[2, 3]
                ),
                "{name}: {err}"
            );
        }
    }

    #[test]
    fn try_reshape_and_transpose_validate() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert!(matches!(
            t.try_reshape(&[4, 2]).unwrap_err(),
            CortexError::ShapeMismatch { .. }
        ));
        assert_eq!(t.try_reshape(&[3, 2]).unwrap().shape(), &[3, 2]);
        assert!(matches!(
            Tensor::from_vec(vec![1.0, 2.0], &[2])
                .try_transpose()
                .unwrap_err(),
            CortexError::RankMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn try_row_bounds_check() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(t.try_row(1).unwrap().data(), &[3.0, 4.0]);
        assert!(matches!(
            t.try_row(9).unwrap_err(),
            CortexError::IndexOutOfBounds {
                axis: 0,
                index: 9,
                size: 2
            }
        ));
        let rank1 = Tensor::from_vec(vec![1.0], &[1]);
        assert!(matches!(
            rank1.try_row(0).unwrap_err(),
            CortexError::RankMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    #[test]
    fn corrupted_storage_is_rejected_not_sliced() {
        // Deserialize skips invariant validation: shape [2,2] with one
        // element must not reach direct indexing in try_row/try_transpose.
        let json = r#"{"data":[1.0],"shape":[2,2],"strides":[2,1]}"#;
        let t: Tensor = serde_json::from_str(json).unwrap();
        assert!(matches!(
            t.try_row(0),
            Err(CortexError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            t.try_transpose(),
            Err(CortexError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn argmax_nan_uses_ieee_total_order_without_panic() {
        // total_cmp: positive NaN sorts above +Inf, so NaN wins argmax.
        let t = Tensor::from_vec(vec![1.0, f32::NAN, 3.0], &[3]);
        assert_eq!(t.argmax(), 1);
        // Negative NaN sorts below -Inf and never wins.
        let neg = f32::from_bits(0xffc0_0000);
        let t = Tensor::from_vec(vec![neg, 1.0, 3.0], &[3]);
        assert_eq!(t.argmax(), 2);
        // All-NaN input is still panic-free.
        let t = Tensor::from_vec(vec![f32::NAN, f32::NAN], &[2]);
        let _ = t.argmax();
        // Empty tensor returns 0.
        let t = Tensor::try_from_vec(vec![], &[0]).unwrap();
        assert_eq!(t.argmax(), 0);
    }

    #[test]
    fn softmax_last_all_neg_inf_is_uniform() {
        let t = Tensor::from_vec(vec![f32::NEG_INFINITY, f32::NEG_INFINITY], &[2]);
        let s = t.softmax_last();
        assert!(s.data().iter().all(|p| (p - 0.5).abs() < 1e-6));
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
