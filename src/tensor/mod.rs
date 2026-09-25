// SPDX-License-Identifier: MIT OR Apache-2.0

pub mod finite;
pub mod ops;

use crate::error::{CortexError, Result, unwrap_compat};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use std::alloc::Layout;
use std::fmt;

/// Row-major dense tensor — a CPU-only reference backend.
///
/// Stores `f32` data in a single contiguous, row-major `Vec<f32>` addressed as
/// `data[i * ncols + j]` (and the rank-`n` generalization). This layout is the
/// one and only invariant: there are no strided views, no broadcasting outside
/// the documented [`ops::batched_matmul`] case, and no device or dtype
/// dispatch. External ML frameworks own general tensor abstractions; this crate
/// deliberately stays a small, honest reference kernel.
///
/// [`Tensor`] does not carry a `strides` field. Because every element is
/// contiguous and row-major, strides are fully determined by [`Self::shape`]
/// and can be recomputed on demand with [`Tensor::row_major_strides`]; storing
/// them would only invite the illusion of stride-aware kernels that do not
/// exist.
///
/// # Layout and copies
///
/// [`Self::reshape`] and [`Self::transpose`] both return owned tensors backed
/// by fresh buffers — they are copies, never aliasing views. [`Self::data_mut`]
/// hands out the raw contiguous storage; callers must preserve the row-major
/// element count for [`Self::shape`] (mutating length or reordering across rows
/// breaks the contiguity invariant).
///
/// [`ops::batched_matmul`]: crate::tensor::ops::batched_matmul
///
/// # Construction
///
/// Prefer [`Tensor::try_from_vec`], which validates the element count and
/// row-major stride arithmetic with checked multiplication before storing
/// `data`. [`Tensor::from_vec`] and the other panic-style constructors remain
/// as pre-1.0 compatibility wrappers; migrate new call sites to `try_from_vec`.
///
/// # Serialization
///
/// JSON is a versioned fixture format for this CPU reference backend, not a
/// universal tensor interchange format. Version 1 carries `schema_version`,
/// `data`, and `shape`; unknown fields, including legacy `strides`, are rejected.
/// Non-finite values cannot be serialized to JSON. Deserialization checks the
/// shape and row-major stride arithmetic through [`Self::try_from_vec`]. Use
/// [`crate::reference_json::from_slice_with_limit`] for untrusted input.
#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorWire {
    schema_version: u32,
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.check_storage().map_err(serde::ser::Error::custom)?;
        if self.data.iter().any(|value| !value.is_finite()) {
            return Err(serde::ser::Error::custom(
                "non-finite tensor data cannot be serialized",
            ));
        }
        let mut wire = serializer.serialize_struct("Tensor", 3)?;
        wire.serialize_field("schema_version", &1u32)?;
        wire.serialize_field("data", &self.data)?;
        wire.serialize_field("shape", &self.shape)?;
        wire.end()
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire: TensorWire = crate::reference_json::deserialize_object(deserializer)?;
        if wire.schema_version != 1 {
            return Err(serde::de::Error::custom(format!(
                "unsupported Tensor schema_version {}",
                wire.schema_version
            )));
        }
        if wire.data.iter().any(|value| !value.is_finite()) {
            return Err(serde::de::Error::custom("non-finite tensor data"));
        }
        Self::try_from_vec(wire.data, &wire.shape).map_err(serde::de::Error::custom)
    }
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

    /// Fallible constructor: proves `numel` and row-major strides are
    /// representable before taking ownership of `data`.
    ///
    /// Strides are not stored; the check exists only to reject shapes whose
    /// row-major stride arithmetic would overflow `usize`, matching the
    /// contiguity guarantees callers of [`Self::data`] rely on. Does not
    /// allocate a new data buffer, so overflowing shapes fail without
    /// attempting a giant allocation.
    pub fn try_from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let numel = checked_numel(shape)?;
        let _ = try_compute_strides(shape)?;
        if data.len() != numel {
            return Err(CortexError::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![data.len()],
            });
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
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
        let _ = try_compute_strides(shape)?;
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

    /// Mutable access to the raw contiguous, row-major storage.
    ///
    /// The slice length equals [`Self::numel`] and is addressed as
    /// `data[i * ncols + j]`. Callers may overwrite values in place but must
    /// not rely on any layout other than contiguous row-major, and must not
    /// assume the length can change: [`Self::shape`] is unaffected by writes
    /// here, so reinterpreting the buffer under a different shape requires
    /// [`Self::reshape`] (which copies).
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Recomputes the row-major strides implied by [`Self::shape`].
    ///
    /// Strides are not stored: this tensor is always contiguous row-major, so
    /// they are a pure function of the shape. The last stride is `1` and each
    /// earlier stride is the product of the following extents. Returns an empty
    /// vector for a scalar (rank-0) shape.
    ///
    /// # Panics
    ///
    /// Panics only if the stride product overflows `usize`, which cannot happen
    /// for a tensor built through the crate's constructors (they reject such
    /// shapes up front). Use [`Self::try_row_major_strides`] to recover instead
    /// of panicking on a hand-corrupted shape.
    #[inline]
    pub fn row_major_strides(&self) -> Vec<usize> {
        unwrap_compat(self.try_row_major_strides(), "Tensor::row_major_strides")
    }

    /// Fallible [`Self::row_major_strides`]: returns
    /// [`CortexError::SizeOverflow`] if the row-major stride product is not
    /// representable in `usize`.
    #[inline]
    pub fn try_row_major_strides(&self) -> Result<Vec<usize>> {
        try_compute_strides(&self.shape)
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    // ── Reshape / transpose (copying) ────────────────────────────────

    /// Reshapes to `new_shape`, which must imply the same element count.
    ///
    /// This **copies**. The returned tensor owns a fresh `Vec<f32>` cloned from
    /// `self`; it does not alias `self`'s storage. Because the layout is always
    /// contiguous row-major, the reinterpretation is a no-op on the element
    /// order — only the shape metadata changes. To avoid the clone entirely,
    /// reinterpret in place with [`Self::reshape_in_place`].
    ///
    /// # Panics
    ///
    /// Panics if the element count changes or the new shape overflows.
    /// Prefer [`Tensor::try_reshape`] in new code.
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        unwrap_compat(self.try_reshape(new_shape), "Tensor::reshape")
    }

    /// Fallible reshape (copying): `new_shape` must imply exactly
    /// `self.numel()` elements and have representable row-major strides.
    ///
    /// See [`Self::reshape`]: the result is a copy, not a view.
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

    /// Reinterprets `self` under `new_shape` in place, without copying.
    ///
    /// Because the storage is already contiguous row-major, a reshape only
    /// rewrites the shape metadata. This variant mutates `self` and reuses the
    /// existing buffer instead of cloning, which [`Self::reshape`] cannot do
    /// behind `&self`. `new_shape` must imply exactly `self.numel()` elements
    /// and have representable row-major strides.
    ///
    /// On error the shape is validated **before** any mutation, so `self` is
    /// left completely unchanged and remains usable: returns
    /// [`CortexError::ShapeMismatch`] on an element-count change and
    /// [`CortexError::SizeOverflow`] on stride overflow.
    pub fn reshape_in_place(&mut self, new_shape: &[usize]) -> Result<()> {
        self.check_storage()?;
        let numel = checked_numel(new_shape)?;
        if numel != self.numel() {
            return Err(CortexError::ShapeMismatch {
                expected: self.shape.clone(),
                got: new_shape.to_vec(),
            });
        }
        try_compute_strides(new_shape)?;
        self.shape = new_shape.to_vec();
        Ok(())
    }

    /// Transposes a 2-D tensor: `[rows, cols]` → `[cols, rows]`.
    ///
    /// This **materializes** a new tensor: the returned buffer is a fresh,
    /// contiguous row-major reordering of the elements, not a stride-swapped
    /// view over `self`'s storage.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not rank-2. Prefer [`Tensor::try_transpose`].
    pub fn transpose(&self) -> Self {
        unwrap_compat(self.try_transpose(), "Tensor::transpose")
    }

    /// Fallible transpose (materializing): returns [`CortexError::RankMismatch`]
    /// unless `self` is rank-2. See [`Self::transpose`]: the result is a copy,
    /// not a view.
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
    /// Deserialization validates this invariant; fallible methods that slice
    /// retain the check as a guard against internal corruption.
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

/// Row-major strides for `shape`, using checked arithmetic.
///
/// Strides are never stored on [`Tensor`]; this helper backs both the
/// public [`Tensor::try_row_major_strides`] recompute path and the
/// construction-time overflow guard that rejects shapes whose stride product
/// is not representable. The last stride is `1` and each earlier stride is the
/// product of all following dimensions. A zero-sized axis can still overflow a
/// leading stride (for example `[0, usize::MAX, 2]`).
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

/// Allocates a filled buffer only after numel and row-major strides are
/// representable.
fn try_filled(shape: &[usize], val: f32) -> Result<Tensor> {
    let numel = checked_numel(shape)?;
    check_f32_alloc(numel, shape)?;
    let _ = try_compute_strides(shape)?;
    Ok(Tensor {
        data: vec![val; numel],
        shape: shape.to_vec(),
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
        assert_eq!(t.row_major_strides(), &[3, 1]);
    }

    #[test]
    fn try_from_vec_matches_from_vec_on_valid_input() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Tensor::from_vec(data.clone(), &[2, 3]);
        let b = Tensor::try_from_vec(data, &[2, 3]).unwrap();
        assert_eq!(a.data(), b.data());
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.row_major_strides(), b.row_major_strides());
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
    fn reshape_and_transpose_are_copies_not_views() {
        // reshape copies: the result owns storage disjoint from the source, so
        // mutating one never aliases the other.
        let mut src = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let reshaped = src.reshape(&[3, 2]);
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.data(), src.data()); // same row-major order
        src.data_mut()[0] = 99.0;
        assert_eq!(reshaped.data()[0], 1.0, "reshape must not alias source");

        // transpose materializes a reordered buffer.
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let tt = t.transpose();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn reshape_in_place_reuses_buffer_without_copy() {
        let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        t.reshape_in_place(&[6]).unwrap();
        assert_eq!(t.shape(), &[6]);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Element-count change is rejected AND leaves the tensor unchanged, so
        // a caller can keep using it after a failed reshape.
        let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(matches!(
            t.reshape_in_place(&[3, 2]).unwrap_err(),
            CortexError::ShapeMismatch { .. }
        ));
        assert_eq!(t.shape(), &[2, 2], "failed reshape must not mutate shape");
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0], "data must be intact");
    }

    #[test]
    fn row_major_strides_recomputes_from_shape() {
        let t = Tensor::from_vec(vec![0.0; 24], &[2, 3, 4]);
        assert_eq!(t.row_major_strides(), &[12, 4, 1]);
        assert_eq!(t.try_row_major_strides().unwrap(), vec![12, 4, 1]);
        let scalar = Tensor::try_from_vec(vec![1.0], &[]).unwrap();
        assert!(scalar.row_major_strides().is_empty());
    }

    #[test]
    fn deserialize_rejects_legacy_strides_field() {
        let json = r#"{"data":[1.0,2.0,3.0,4.0,5.0,6.0],"shape":[2,3],"strides":[3,1]}"#;
        assert!(serde_json::from_str::<Tensor>(json).is_err());
    }

    #[test]
    fn serialized_form_has_no_strides_field() {
        let t = Tensor::from_vec(vec![1.0, 2.0], &[2]);
        let json = serde_json::to_string(&t).unwrap();
        assert!(
            !json.contains("strides"),
            "serialized form leaks strides: {json}"
        );
        assert!(json.contains("\"shape\""));
        assert!(json.contains("\"data\""));
    }

    #[test]
    fn elementwise_ops_require_exact_shape_no_broadcast() {
        // Exact-shape contract: no NumPy broadcasting on add/sub/mul, not even
        // against a size-1 axis that broadcasting would otherwise expand.
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let row = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[1, 3]);
        let scalar_like = Tensor::from_vec(vec![10.0], &[1]);
        for (name, err) in [
            ("add row", a.try_add(&row).unwrap_err()),
            ("sub row", a.try_sub(&row).unwrap_err()),
            ("mul row", a.try_mul(&row).unwrap_err()),
            ("add scalar-like", a.try_add(&scalar_like).unwrap_err()),
        ] {
            assert!(
                matches!(err, CortexError::ShapeMismatch { .. }),
                "{name}: {err}"
            );
        }
        // Exact match succeeds.
        let b = Tensor::from_vec(vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0], &[2, 3]);
        assert_eq!(
            a.try_add(&b).unwrap().data(),
            &[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
        );
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
    fn corrupted_storage_is_rejected_on_deserialize() {
        let json = r#"{"schema_version":1,"data":[1.0],"shape":[2,2]}"#;
        assert!(serde_json::from_str::<Tensor>(json).is_err());
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
        assert_eq!(empty.row_major_strides(), &[3, 1]);

        let zero_1d = Tensor::try_from_vec(vec![], &[0]).unwrap();
        assert_eq!(zero_1d.numel(), 0);
        assert_eq!(zero_1d.row_major_strides(), &[1]);

        let scalar = Tensor::try_from_vec(vec![3.5], &[]).unwrap();
        assert_eq!(scalar.numel(), 1);
        assert!(scalar.row_major_strides().is_empty());
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
        assert_eq!(cancelled.row_major_strides(), &[0, 2, 1]);

        // A trailing zero must not be lost to a left-to-right overflow.
        let trailing = Tensor::try_from_vec(vec![], &[3, usize::MAX, 0]).unwrap();
        assert_eq!(trailing.numel(), 0);
        assert_eq!(trailing.row_major_strides(), &[0, 0, 1]);
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
                    assert_eq!(t.row_major_strides(), strides.as_slice());
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
