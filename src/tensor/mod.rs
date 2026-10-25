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
///
/// # Rank-0 scalars vs zero-sized axes
///
/// - `shape == []` is a rank-0 scalar with exactly one element (`numel == 1`).
/// - Any axis equal to `0` yields `numel == 0`, even when other extents would
///   overflow if multiplied without checked arithmetic (for example
///   `[usize::MAX, 0, 2]`).
/// - Constructors and [`Deserialize`] reject shapes whose element count or
///   row-major stride product is not representable in `usize` (see
///   `checked_numel` and `try_compute_strides`).
/// - [`Self::mean`] on a zero-element tensor is `0.0 / 0.0`, which is `NaN`.
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

/// Portable RNG behind every `*_seeded` constructor. ChaCha8 is a fixed,
/// documented algorithm, unlike `StdRng`, whose stream may change between
/// `rand` releases, so a `(config, seed)` pair keeps rebuilding the same
/// weights. A golden test pins the stream.
pub(crate) fn seeded_rng(seed: u64) -> rand_chacha::ChaCha8Rng {
    use rand::SeedableRng;
    rand_chacha::ChaCha8Rng::seed_from_u64(seed)
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
    ///
    /// Returns [`CortexError::SizeOverflow`] when the element count or strides
    /// are not representable and [`CortexError::AllocationFailed`] when the
    /// allocator refuses the buffer; the process is never aborted for a
    /// shape the machine cannot back. The same holds for every constructor
    /// and op that allocates a fresh output buffer.
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
    /// Panics if the shape product or stride arithmetic overflows `usize`,
    /// if `mean` is not finite, if `std` is not finite or is negative, or if
    /// a sample overflows `f32`. Prefer [`Tensor::try_randn`] in new code.
    /// See [`Tensor::from_vec`] for the pre-1.0 compatibility policy.
    pub fn randn(shape: &[usize], mean: f32, std: f32) -> Self {
        unwrap_compat(Self::try_randn(shape, mean, std), "Tensor::randn")
    }

    /// Fallible [`Tensor::randn`]: fails on shape/stride overflow before
    /// allocating, and with [`CortexError::InvalidConfig`] unless `mean` is
    /// finite and `std` is finite and non-negative. Draws from the thread
    /// RNG; use [`Tensor::try_randn_with_rng`] for reproducible values.
    pub fn try_randn(shape: &[usize], mean: f32, std: f32) -> Result<Self> {
        Self::try_randn_with_rng(shape, mean, std, &mut rand::rng())
    }

    /// Like [`Tensor::try_randn`] but draws from the caller's RNG, so a
    /// seeded RNG reproduces the same tensor.
    pub fn try_randn_with_rng<R: rand::Rng + ?Sized>(
        shape: &[usize],
        mean: f32,
        std: f32,
        rng: &mut R,
    ) -> Result<Self> {
        use rand::RngExt;
        if !mean.is_finite() || !std.is_finite() || std < 0.0 {
            return Err(CortexError::InvalidConfig(format!(
                "randn: mean ({mean}) must be finite and std ({std}) must be finite and >= 0"
            )));
        }
        let numel = checked_numel(shape)?;
        check_f32_alloc(numel, shape)?;
        let _ = try_compute_strides(shape)?;
        let mut data = try_reserve_f32(numel, shape)?;
        // Box-Muller: each (u1, u2) pair yields two independent samples.
        while data.len() < numel {
            // u1 in (0, 1] so ln(u1) is finite without clamping the tails.
            let u1: f32 = 1.0 - rng.random_range(0.0f32..1.0);
            let u2: f32 = rng.random_range(0.0f32..1.0);
            let r = (-2.0f32 * u1.ln()).sqrt();
            let (sin, cos) = (2.0f32 * std::f32::consts::PI * u2).sin_cos();
            data.push(mean + std * r * cos);
            if data.len() < numel {
                data.push(mean + std * r * sin);
            }
        }
        // A finite but huge `std` (or `mean`) can still overflow to inf/NaN.
        if data.iter().any(|v| !v.is_finite()) {
            return Err(CortexError::InvalidConfig(format!(
                "randn: mean ({mean}) and std ({std}) overflow f32 while sampling"
            )));
        }
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
        if rows == 0 || cols == 0 {
            // Nothing to copy. Returning here matters: `[usize::MAX, 0]` has
            // numel 0 but would otherwise iterate the huge axis.
            return Self::try_from_vec(Vec::new(), &[cols, rows]);
        }
        let out_len = checked_numel(&[cols, rows])?;
        let mut out = Vec::with_capacity(out_len);
        // Output row `c` is input column `c`: elements `c, c + cols, ...`.
        // Striding the iterator avoids computing `c * rows + r` offsets, so
        // no index arithmetic can overflow.
        for c in 0..cols {
            out.extend(self.data.iter().skip(c).step_by(cols));
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

    // Elementwise activations follow the finite-value policy in
    // [`finite`](crate::tensor::finite): NaN propagates, `±Inf` maps to the
    // function's mathematical limit, finite inputs give finite outputs.

    /// Rectified linear unit, `max(x, 0)` with NaN propagation.
    ///
    /// `f32::max` is not used because it discards NaN. NaN inputs stay NaN,
    /// `+Inf` stays `+Inf`, and every non-positive input (including `-0.0`
    /// and `-Inf`) is written as canonical `+0.0`.
    pub fn relu(&self) -> Self {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| if x.is_nan() || x > 0.0 { x } else { 0.0 })
            .collect();
        Self::from_vec(data, &self.shape)
    }

    /// Tanh-approximate GELU: `x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 x³)))`.
    ///
    /// `+Inf` gives `+Inf` and `-Inf` gives `0` (the limits), instead of the
    /// NaN that `-Inf * 0` would produce. NaN propagates.
    pub fn gelu(&self) -> Self {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                if x.is_infinite() {
                    return if x.is_sign_positive() { x } else { 0.0 };
                }
                let c = (2.0f32 / std::f32::consts::PI).sqrt();
                x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
            })
            .collect();
        Self::from_vec(data, &self.shape)
    }

    /// SiLU / swish: `x / (1 + exp(-x))`.
    ///
    /// `+Inf` gives `+Inf` and `-Inf` gives `0` (the limits), instead of the
    /// NaN that `-Inf / Inf` would produce. NaN propagates.
    pub fn silu(&self) -> Self {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                if x.is_infinite() {
                    return if x.is_sign_positive() { x } else { 0.0 };
                }
                x / (1.0 + (-x).exp())
            })
            .collect();
        Self::from_vec(data, &self.shape)
    }

    /// Fast sigmoid surrogate gradient (from soma-engine's E-prop): `x / (1 + |x|)`.
    ///
    /// `±Inf` gives `±1` (the limits), instead of the NaN that `Inf / Inf`
    /// would produce. NaN propagates.
    pub fn fast_sigmoid(&self) -> Self {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                if x.is_infinite() {
                    return if x.is_sign_positive() { 1.0 } else { -1.0 };
                }
                x / (1.0 + x.abs())
            })
            .collect();
        Self::from_vec(data, &self.shape)
    }

    // ── Reductions ───────────────────────────────────────────────────

    /// Sum of all elements, accumulated in compensated `f64` and cast back to `f32`.
    ///
    /// Sequential Neumaier (Kahan-Babuška) `f64` accumulation keeps every
    /// `f32` input exact, avoids the `f32` running-sum stall at `2^24` and
    /// spurious intermediate overflow (`[MAX, MAX, -MAX]` sums to `MAX`), and
    /// keeps a small term that plain `f64` would cancel away between two large
    /// ones (`[MAX, 1e20, -MAX]` sums to `1e20`). The final cast follows
    /// IEEE rounding: a true sum outside the `f32` range becomes `±Inf`. NaN
    /// inputs propagate and `+Inf + -Inf` is NaN. An empty tensor sums to `0`.
    pub fn sum(&self) -> f32 {
        self.sum_f64() as f32
    }

    /// Arithmetic mean over all elements: the compensated `f64` total divided
    /// by `numel()` in `f64`, then cast once to `f32` (not `sum() / numel()`).
    ///
    /// Returns `NaN` when `numel() == 0` (zero-sized axis or empty buffer).
    pub fn mean(&self) -> f32 {
        (self.sum_f64() / self.numel() as f64) as f32
    }

    /// Neumaier-compensated sequential `f64` sum (deterministic for a given
    /// input bit pattern). Non-finite inputs fall through IEEE arithmetic:
    /// NaN propagates and `+Inf + -Inf` is NaN.
    fn sum_f64(&self) -> f64 {
        let mut sum = 0.0f64;
        let mut comp = 0.0f64;
        for &x in &self.data {
            let x = f64::from(x);
            let t = sum + x;
            if sum.abs() >= x.abs() {
                comp += (sum - t) + x;
            } else {
                comp += (x - t) + sum;
            }
            sum = t;
        }
        if sum.is_finite() { sum + comp } else { sum }
    }

    /// Largest element.
    ///
    /// NaN policy: any NaN (either sign) makes the result NaN; `f32::max` is
    /// not used because it silently discards NaN. Otherwise this is the
    /// ordinary maximum, so `data()[argmax()] == max_val()` for NaN-free
    /// input and both report NaN when a positive NaN is present. (A negative
    /// NaN sorts below `-Inf` in [`argmax`](Self::argmax)'s total order but
    /// still yields NaN here.) An empty tensor returns `-Inf`, the identity of
    /// `max`.
    pub fn max_val(&self) -> f32 {
        let mut best = f32::NEG_INFINITY;
        for &x in &self.data {
            if x.is_nan() {
                return f32::NAN;
            }
            if x > best {
                best = x;
            }
        }
        best
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

/// Reserves exactly `numel` elements of `f32` storage for `shape`, or returns
/// [`CortexError::AllocationFailed`] when the allocator refuses.
///
/// `vec![x; n]` and `Vec::with_capacity(n)` call `handle_alloc_error` on
/// failure, which aborts the process. A caller-controlled shape that passes
/// the `isize::MAX` layout check (see [`check_f32_alloc`]) can therefore
/// still take the whole process down instead of surfacing an `Err`. Every
/// fresh output buffer sized by a shape goes through this helper so the
/// failure stays a value. Buffers sized by an existing tensor's storage are
/// bounded by an allocation that already succeeded and keep using `vec!`.
pub(crate) fn try_reserve_f32(numel: usize, shape: &[usize]) -> Result<Vec<f32>> {
    let mut data: Vec<f32> = Vec::new();
    data.try_reserve_exact(numel)
        .map_err(|_| CortexError::AllocationFailed {
            shape: shape.to_vec(),
            bytes: numel.saturating_mul(std::mem::size_of::<f32>()),
        })?;
    Ok(data)
}

/// Checks numel, layout and row-major strides for `shape`, then allocates a
/// zero-initialised buffer fallibly. Shared by the constructors and the ops
/// that produce a fresh output tensor.
pub(crate) fn try_alloc_filled(shape: &[usize], val: f32) -> Result<Vec<f32>> {
    let numel = checked_numel(shape)?;
    check_f32_alloc(numel, shape)?;
    let _ = try_compute_strides(shape)?;
    let mut data = try_reserve_f32(numel, shape)?;
    data.resize(numel, val);
    Ok(data)
}

/// Allocates a filled buffer only after numel and row-major strides are
/// representable and the allocation itself succeeds.
fn try_filled(shape: &[usize], val: f32) -> Result<Tensor> {
    let data = try_alloc_filled(shape, val)?;
    Ok(Tensor {
        data,
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

    /// Largest `f32` count that passes the `isize::MAX` layout check, so the
    /// failure must come from the allocator, not from `SizeOverflow`.
    const UNBACKABLE_NUMEL: usize = isize::MAX as usize / std::mem::size_of::<f32>();

    #[test]
    fn unbackable_shape_is_an_error_not_an_abort() {
        let shape = [UNBACKABLE_NUMEL];
        for result in [
            Tensor::try_zeros(&shape),
            Tensor::try_ones(&shape),
            Tensor::try_full(&shape, 2.5),
            Tensor::try_randn(&shape, 0.0, 1.0),
            Tensor::try_randn_with_rng(&shape, 0.0, 1.0, &mut seeded_rng(7)),
        ] {
            match result {
                Err(CortexError::AllocationFailed { shape: got, bytes }) => {
                    assert_eq!(got, shape);
                    assert_eq!(bytes, UNBACKABLE_NUMEL * std::mem::size_of::<f32>());
                }
                Err(other) => panic!("expected AllocationFailed, got {other}"),
                Ok(t) => panic!("allocated {} elements", t.numel()),
            }
        }
        // The two guards stay distinct: one past the layout limit is still
        // `SizeOverflow`, not an allocator failure.
        assert!(matches!(
            Tensor::try_zeros(&[UNBACKABLE_NUMEL + 1]),
            Err(CortexError::SizeOverflow { .. })
        ));
    }

    #[test]
    fn allocation_failed_message_names_shape_and_bytes() {
        let err = Tensor::try_zeros(&[UNBACKABLE_NUMEL]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("allocation of"), "{msg}");
        assert!(msg.contains(&UNBACKABLE_NUMEL.to_string()), "{msg}");
    }

    #[test]
    fn try_randn_rejects_overflowing_std() {
        assert!(matches!(
            Tensor::try_randn(&[64], 0.0, f32::MAX),
            Err(CortexError::InvalidConfig(_))
        ));
    }

    #[test]
    fn try_randn_rejects_bad_params() {
        for (m, s) in [
            (0.0, f32::NAN),
            (0.0, -1.0),
            (0.0, f32::INFINITY),
            (f32::NAN, 1.0),
            (f32::INFINITY, 1.0),
        ] {
            assert!(matches!(
                Tensor::try_randn(&[3], m, s),
                Err(CortexError::InvalidConfig(_))
            ));
        }
        assert!(Tensor::try_randn(&[3], 0.0, 0.0).is_ok());
    }

    #[test]
    fn try_randn_with_rng_is_reproducible() {
        let mk = |seed| {
            let mut rng = seeded_rng(seed);
            Tensor::try_randn_with_rng(&[5, 7], 0.0, 1.0, &mut rng).unwrap()
        };
        assert_eq!(mk(7).data(), mk(7).data());
        assert_ne!(mk(7).data(), mk(8).data());
        assert!(mk(7).data().iter().all(|v| v.is_finite()));
    }

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
    fn try_transpose_zero_axis_returns_without_iterating_huge_extent() {
        // numel is 0 but one axis is enormous: the loops must not run.
        for shape in [
            [0usize, 1 << 40],
            [1 << 40, 0],
            [usize::MAX, 0],
            [0, usize::MAX],
            [0, 0],
        ] {
            let t = Tensor::try_from_vec(vec![], &shape).unwrap();
            let tt = t.try_transpose().unwrap();
            assert_eq!(tt.shape(), &[shape[1], shape[0]]);
            assert_eq!(tt.numel(), 0);
        }
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

    /// Guards against unchecked shape products wrapping to `numel == 0` in release.
    #[test]
    fn equal_power_of_two_extents_reject_without_wrapped_numel() {
        const EXT: usize = 1usize << (usize::BITS / 2);
        let shape = [EXT, EXT];
        for (name, result) in [
            ("try_zeros", Tensor::try_zeros(&shape)),
            ("try_ones", Tensor::try_ones(&shape)),
            ("try_full", Tensor::try_full(&shape, 1.0)),
            ("try_from_vec", Tensor::try_from_vec(Vec::new(), &shape)),
        ] {
            assert!(
                matches!(result, Err(CortexError::SizeOverflow { shape: ref got }) if got.as_slice() == shape),
                "{name}: {result:?}"
            );
        }
        assert!(checked_numel(&shape).is_err());
    }

    /// Empty tensors divide by zero in `mean()`; rank-0 scalars use the sole element.
    #[test]
    fn mean_of_zero_element_tensor_is_nan() {
        let empty = Tensor::try_zeros(&[0, 3]).unwrap();
        assert!(empty.mean().is_nan());
        let scalar = Tensor::try_from_vec(vec![4.0], &[]).unwrap();
        assert_eq!(scalar.mean(), 4.0);
    }

    // ── Finite-value policy: activations and whole-tensor reductions (issue 95) ──

    const NEG_NAN: f32 = f32::from_bits(0xffc0_0000);

    #[test]
    fn relu_propagates_nan_and_keeps_infinities() {
        let t = Tensor::from_vec(
            vec![f32::NAN, 1.0, -2.0, f32::INFINITY, f32::NEG_INFINITY, 0.0],
            &[6],
        );
        let r = t.relu();
        assert!(r.data()[0].is_nan(), "relu must not erase NaN via f32::max");
        assert_eq!(r.data()[1], 1.0);
        assert_eq!(r.data()[2].to_bits(), 0.0f32.to_bits());
        assert_eq!(r.data()[3], f32::INFINITY);
        assert_eq!(r.data()[4].to_bits(), 0.0f32.to_bits());
        assert_eq!(r.data()[5].to_bits(), 0.0f32.to_bits());
        let neg = Tensor::from_vec(vec![NEG_NAN, -0.0], &[2]).relu();
        assert!(neg.data()[0].is_nan());
        // Negative zero is clamped to canonical +0.0 (deterministic bit pattern).
        assert_eq!(neg.data()[1].to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn max_val_propagates_nan_and_documents_empty() {
        let t = Tensor::from_vec(vec![f32::NAN, 1.0, -2.0], &[3]);
        assert!(
            t.max_val().is_nan(),
            "max_val must not drop NaN via f32::max"
        );
        // Positive NaN: argmax picks the NaN and max_val is NaN, so the two agree.
        assert!(t.data()[t.argmax()].is_nan());
        let neg = Tensor::from_vec(vec![NEG_NAN, 1.0, 3.0], &[3]);
        assert!(neg.max_val().is_nan());
        let fin = Tensor::from_vec(vec![-3.0, f32::NEG_INFINITY, 2.5, 2.5], &[4]);
        assert_eq!(fin.max_val(), 2.5);
        assert_eq!(fin.data()[fin.argmax()], fin.max_val());
        let inf = Tensor::from_vec(vec![1.0, f32::INFINITY], &[2]);
        assert_eq!(inf.max_val(), f32::INFINITY);
        let empty = Tensor::try_from_vec(vec![], &[0]).unwrap();
        assert_eq!(empty.max_val(), f32::NEG_INFINITY);
    }

    #[test]
    fn activations_at_infinity_return_their_limits() {
        let t = Tensor::from_vec(vec![f32::NEG_INFINITY, f32::INFINITY, f32::NAN], &[3]);
        let silu = t.silu();
        assert_eq!(silu.data()[0], 0.0, "silu(-Inf) = 0, not -Inf/Inf = NaN");
        assert_eq!(silu.data()[1], f32::INFINITY);
        assert!(silu.data()[2].is_nan());
        let gelu = t.gelu();
        assert_eq!(gelu.data()[0], 0.0, "gelu(-Inf) = 0, not -Inf*0 = NaN");
        assert_eq!(gelu.data()[1], f32::INFINITY);
        assert!(gelu.data()[2].is_nan());
        let fs = t.fast_sigmoid();
        assert_eq!(fs.data()[0], -1.0, "fast_sigmoid(-Inf) = -1, not Inf/Inf");
        assert_eq!(fs.data()[1], 1.0);
        assert!(fs.data()[2].is_nan());
        let relu = t.relu();
        assert_eq!(relu.data()[0], 0.0);
        assert_eq!(relu.data()[1], f32::INFINITY);
        assert!(relu.data()[2].is_nan());
    }

    #[test]
    fn activations_stay_finite_for_large_finite_inputs() {
        // x³ overflows f32 well before x does; the tanh argument must not turn
        // a finite input into NaN, and the result must approach the ±Inf limits.
        let t = Tensor::from_vec(vec![-1e20, 1e20, -3e38, 3e38, -100.0, 100.0], &[6]);
        for (name, out) in [
            ("gelu", t.gelu()),
            ("silu", t.silu()),
            ("fast_sigmoid", t.fast_sigmoid()),
            ("relu", t.relu()),
        ] {
            for (x, y) in t.data().iter().zip(out.data()) {
                assert!(y.is_finite(), "{name}({x}) = {y} is not finite");
            }
        }
        assert_eq!(t.gelu().data()[0], 0.0);
        assert_eq!(t.gelu().data()[1], 1e20);
        assert_eq!(t.silu().data()[0], 0.0);
        assert_eq!(t.silu().data()[1], 1e20);
        // Finite fast sigmoid lies in the closed interval [-1, 1]: once
        // |x| >= 2^24, `1 + |x|` rounds to `|x|` in f32 and the result is ±1.
        let fs = Tensor::from_vec(vec![16_777_216.0, -16_777_216.0, 1e20], &[3]).fast_sigmoid();
        assert_eq!(fs.data(), &[1.0, -1.0, 1.0]);
        let fs = Tensor::from_vec(vec![16_777_215.0], &[1]).fast_sigmoid();
        assert!(fs.data()[0] < 1.0 && fs.data()[0] > 0.99);
    }

    #[test]
    fn sum_and_mean_accumulate_in_f64() {
        // f32 accumulation stalls at 2^24: 16777216 + 1 rounds back to 16777216.
        let t = Tensor::from_vec(vec![16_777_216.0, 1.0, 1.0], &[3]);
        assert_eq!(t.sum(), 16_777_218.0);
        // 2^24 + 2 ones: the f32 running sum would freeze at 2^24.
        let n = (1usize << 24) + 2;
        let ones = Tensor::ones(&[n]);
        assert_eq!(ones.sum(), n as f32);
        assert_eq!(ones.mean(), 1.0);
        // Intermediate f32 overflow must not leak into a finite true sum.
        let t = Tensor::from_vec(vec![f32::MAX, f32::MAX, -f32::MAX], &[3]);
        assert_eq!(t.sum(), f32::MAX);
        // Compensation keeps a term that plain f64 cancels away: 1e20 is below
        // half an f64 ulp of f32::MAX (~7.5e22), so MAX + 1e20 - MAX would be 0.
        let t = Tensor::from_vec(vec![f32::MAX, 1e20, -f32::MAX], &[3]);
        assert_eq!(t.sum(), 1e20);
        assert_eq!(t.mean(), 1e20 / 3.0);
        // A true sum outside the f32 range still rounds to the signed infinity.
        let t = Tensor::from_vec(vec![f32::MAX, f32::MAX], &[2]);
        assert_eq!(t.sum(), f32::INFINITY);
        // Non-finite inputs follow IEEE: NaN propagates, Inf - Inf is NaN.
        assert!(Tensor::from_vec(vec![1.0, f32::NAN], &[2]).sum().is_nan());
        assert_eq!(
            Tensor::from_vec(vec![1.0, f32::INFINITY], &[2]).sum(),
            f32::INFINITY
        );
        assert!(
            Tensor::from_vec(vec![f32::INFINITY, f32::NEG_INFINITY], &[2])
                .sum()
                .is_nan()
        );
        // The reference golden (sum 21, mean 3.5) is unchanged.
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!((a.sum(), a.mean()), (21.0, 3.5));
        // Mean is computed from the f64 sum, not the rounded f32 sum.
        let t = Tensor::from_vec(vec![16_777_216.0, 1.0, 1.0, 0.0], &[4]);
        assert_eq!(t.mean(), 16_777_218.0 / 4.0);
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
