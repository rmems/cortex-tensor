// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral interchange metadata for tensors crossing an ANN stage
//! boundary.
//!
//! [`TensorMeta`] says *what* a stage tensor is (shape and [`DType`]). This
//! module adds *where* it lives and *how* it is laid out, so the reference
//! backend, the Candle adapter, the Burn slot, and `hybrid-fusion` can agree on
//! a hand-off before any bytes move:
//!
//! * [`TensorDevice`]: device class plus ordinal.
//! * [`TensorLayout`]: memory layout of the element storage.
//! * [`InterchangeMeta`]: shape, dtype, device, and layout together.
//! * [`HiddenStateShape`]: the named `[batch?, seq_len, hidden_dim]` axes of a
//!   hidden state.
//! * [`InterchangeTensor`]: a stage tensor that can describe itself with
//!   [`InterchangeMeta`].
//! * [`transfer`]: the conversion contract for moving a tensor between two
//!   [`ExternalAdapterMarker`] backends through [`HiddenStateBuffer`].
//!
//! # Not a wire format
//!
//! Nothing here is serialized. The types carry no serde derives and define no
//! byte encoding; they are in-process facts. An orchestrator that needs to
//! record them translates the stable tokens ([`DType::as_str`],
//! [`TensorDevice`]'s `Display`, [`TensorLayout::as_str`]) into its own
//! format, the same way it translates [`crate::adapter::NegotiationDocument`].
//! Data still moves only by copy through [`HiddenStateBuffer`] (or Candle's
//! `HiddenState`, bridged here behind the `candle` feature).
//!
//! # No backend types
//!
//! No signature in this module names a Candle, Burn, or other engine type.
//! Engine device handles stay inside their adapters and are reported here as
//! a [`DeviceClass`] and ordinal.

use crate::adapter::{AdapterCapabilities, DeviceClass, ExternalAdapterMarker, HiddenStateBuffer};
use crate::error::{CortexError, Result};
use crate::stage::{DType, StageTensor, TensorMeta};
use std::fmt;

/// Stage id used in refusals at the hidden-state boundary, matching
/// [`crate::stage::ReferenceExecutor::import_hidden`].
const HIDDEN_STATE: &str = "hidden_state";

/// Where a tensor's storage lives.
///
/// A placement fact, not a device handle: it never owns or opens a device.
/// `ordinal` distinguishes devices of the same [`DeviceClass`] (for example
/// two CUDA devices). Host memory is [`TensorDevice::HOST`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorDevice {
    /// Kind of device.
    pub class: DeviceClass,
    /// Index among devices of `class`. Always `0` for the host CPU.
    pub ordinal: usize,
}

impl TensorDevice {
    /// Host CPU memory, the only placement the reference backend and the
    /// Candle adapter produce.
    pub const HOST: Self = Self {
        class: DeviceClass::Cpu,
        ordinal: 0,
    };

    /// Device `ordinal` of `class`.
    pub fn new(class: DeviceClass, ordinal: usize) -> Self {
        Self { class, ordinal }
    }
}

/// Renders as `"<class>:<ordinal>"`, for example `"cpu:0"` or `"cuda:1"`.
impl fmt::Display for TensorDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.class, self.ordinal)
    }
}

/// Memory layout of a tensor's element storage.
///
/// Every backend in this crate hands tensors across the boundary as dense
/// row-major storage, so that is the only variant today. A backend holding a
/// strided or transposed view must make it contiguous before export, as the
/// Candle adapter already does.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorLayout {
    /// Dense, contiguous, row-major (C order): the last axis varies fastest
    /// and there is no padding between elements.
    RowMajor,
}

impl TensorLayout {
    /// Stable negotiation token (`"row_major"`).
    pub fn as_str(self) -> &'static str {
        match self {
            TensorLayout::RowMajor => "row_major",
        }
    }
}

impl fmt::Display for TensorLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Shape, dtype, device, and layout of a tensor crossing a stage boundary.
///
/// Metadata only: it never carries or borrows element data. Build it from a
/// stage tensor with [`InterchangeTensor::interchange_meta`], from a copied
/// buffer with [`InterchangeMeta::of_buffer`], or by hand with
/// [`InterchangeMeta::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterchangeMeta {
    /// Extents, outermost dimension first.
    pub shape: Vec<usize>,
    /// Element type.
    pub dtype: DType,
    /// Where the storage lives.
    pub device: TensorDevice,
    /// How the storage is laid out.
    pub layout: TensorLayout,
}

impl InterchangeMeta {
    /// Combine stage metadata with a placement and layout.
    pub fn new(meta: TensorMeta, device: TensorDevice, layout: TensorLayout) -> Self {
        Self {
            shape: meta.shape,
            dtype: meta.dtype,
            device,
            layout,
        }
    }

    /// Stage metadata for a dense row-major tensor in host memory.
    pub fn host(meta: TensorMeta) -> Self {
        Self::new(meta, TensorDevice::HOST, TensorLayout::RowMajor)
    }

    /// Describe a [`HiddenStateBuffer`] after checking its length.
    ///
    /// The buffer is always host-resident and row-major.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::StageFailed`] (stage `"hidden_state"`) wrapping
    /// [`CortexError::SizeOverflow`] when the element count overflows, or
    /// [`CortexError::ShapeMismatch`] when `data.len()` differs from it. A zero
    /// axis yields an element count of `0` even if an earlier product would
    /// overflow, matching [`crate::stage::ReferenceExecutor::import_hidden`].
    pub fn of_buffer(buffer: &HiddenStateBuffer) -> Result<Self> {
        let numel = crate::tensor::checked_numel(&buffer.shape).map_err(hidden_state_failure)?;
        if buffer.data.len() != numel {
            return Err(hidden_state_failure(CortexError::ShapeMismatch {
                expected: buffer.shape.clone(),
                got: vec![buffer.data.len()],
            }));
        }
        Ok(Self::host(TensorMeta {
            shape: buffer.shape.clone(),
            dtype: buffer.dtype,
        }))
    }

    /// The shape and dtype as [`TensorMeta`], dropping placement and layout.
    pub fn tensor_meta(&self) -> TensorMeta {
        TensorMeta {
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }

    /// Number of elements described by [`Self::shape`].
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::SizeOverflow`] when the product overflows
    /// `usize`. Any zero axis short-circuits to `0`.
    pub fn numel(&self) -> Result<usize> {
        crate::tensor::checked_numel(&self.shape)
    }

    /// Read [`Self::shape`] as hidden-state axes; see
    /// [`HiddenStateShape::from_dims`].
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::RankMismatch`] unless the rank is 2 or 3.
    pub fn hidden_state(&self) -> Result<HiddenStateShape> {
        HiddenStateShape::from_dims(&self.shape)
    }

    /// Refuse a tensor that `caps` cannot execute stages on without a
    /// conversion.
    ///
    /// Requires [`Self::dtype`] in [`crate::stage::AnnCapabilities::supported_dtypes`]
    /// and [`Self::device`]'s class in [`AdapterCapabilities::devices`]. Every
    /// [`TensorLayout`] is accepted because every variant is one all adapters
    /// read. Checks run in that order: dtype, then device.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::UnsupportedOperation`] with `category` `"dtype"`
    /// or `"device"` and `stage_id: None` for the first gap.
    pub fn require_supported_by(&self, caps: &AdapterCapabilities) -> Result<()> {
        if !caps.ann.supported_dtypes.contains(&self.dtype) {
            return Err(refusal(
                caps,
                "dtype",
                format!("{} tensors are not accepted", self.dtype),
            ));
        }
        if !caps.devices.contains(&self.device.class) {
            return Err(refusal(
                caps,
                "device",
                format!("tensors on {} are not accepted", self.device),
            ));
        }
        Ok(())
    }

    /// Refuse a hidden state that `caps` cannot take as stage input.
    ///
    /// Runs [`Self::hidden_state`], then rejects a batched hidden state when
    /// [`AdapterCapabilities::batch`] is `false`, then runs
    /// [`Self::require_supported_by`]. Returns the named axes on success.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::RankMismatch`] for a rank other than 2 or 3,
    /// [`CortexError::UnsupportedOperation`] with `category: "batch"` for an
    /// unsupported batch axis, or the errors of [`Self::require_supported_by`].
    pub fn require_hidden_state_supported_by(
        &self,
        caps: &AdapterCapabilities,
    ) -> Result<HiddenStateShape> {
        let axes = self.hidden_state()?;
        if let Some(batch) = axes.batch
            && !caps.batch
        {
            return Err(refusal(
                caps,
                "batch",
                format!("a leading batch axis of {batch} is not accepted"),
            ));
        }
        self.require_supported_by(caps)?;
        Ok(axes)
    }
}

/// Named axes of a hidden state: `[seq_len, hidden_dim]` or
/// `[batch, seq_len, hidden_dim]`.
///
/// The reference backend is unbatched and uses the rank-2 form. Zero-sized
/// axes are allowed, as they are for [`crate::Tensor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HiddenStateShape {
    /// Leading batch extent, or `None` for an unbatched hidden state.
    pub batch: Option<usize>,
    /// Sequence (token) extent.
    pub seq_len: usize,
    /// Feature width of each token's hidden vector.
    pub hidden_dim: usize,
}

impl HiddenStateShape {
    /// Name the axes of a rank-2 or rank-3 shape.
    ///
    /// # Errors
    ///
    /// Returns [`CortexError::RankMismatch`] for any other rank. `expected`
    /// is `2` for a rank below 2 and `3` for a rank above 3.
    pub fn from_dims(dims: &[usize]) -> Result<Self> {
        match *dims {
            [seq_len, hidden_dim] => Ok(Self {
                batch: None,
                seq_len,
                hidden_dim,
            }),
            [batch, seq_len, hidden_dim] => Ok(Self {
                batch: Some(batch),
                seq_len,
                hidden_dim,
            }),
            _ => Err(CortexError::RankMismatch {
                expected: if dims.len() < 2 { 2 } else { 3 },
                got: dims.len(),
            }),
        }
    }

    /// The axes as a shape, outermost dimension first.
    pub fn dims(&self) -> Vec<usize> {
        match self.batch {
            Some(batch) => vec![batch, self.seq_len, self.hidden_dim],
            None => vec![self.seq_len, self.hidden_dim],
        }
    }
}

/// A stage tensor that can report its [`InterchangeMeta`].
///
/// Implementors must report the placement and layout of the storage they
/// actually hold. The reference [`crate::Tensor`] and the Candle adapter's
/// `CandleTensor` (which only ever holds contiguous CPU storage) are always
/// [`TensorDevice::HOST`] and [`TensorLayout::RowMajor`].
pub trait InterchangeTensor: StageTensor {
    /// Shape, dtype, device, and layout of this tensor.
    fn interchange_meta(&self) -> InterchangeMeta;
}

impl InterchangeTensor for crate::Tensor {
    fn interchange_meta(&self) -> InterchangeMeta {
        InterchangeMeta::host(self.meta())
    }
}

#[cfg(feature = "candle")]
impl InterchangeTensor for crate::adapter::CandleTensor {
    fn interchange_meta(&self) -> InterchangeMeta {
        InterchangeMeta::host(self.meta())
    }
}

/// Move `tensor` from `source` to `dest` through a [`HiddenStateBuffer`] copy.
///
/// This is the conversion contract between two [`ExternalAdapterMarker`]
/// backends. Before anything is copied it requires both sides to advertise
/// [`AdapterCapabilities::hidden_state_io`] and `dest` to accept the tensor's
/// dtype. It then exports, checks the buffer with [`InterchangeMeta::of_buffer`],
/// imports, and confirms the imported tensor reports the same shape and dtype.
/// Device is not compared: `dest` places the imported tensor itself.
///
/// # Errors
///
/// * [`CortexError::UnsupportedOperation`] with `category: "hidden_state_io"`
///   when either side cannot import/export, or `category: "dtype"` when
///   `dest` does not support the tensor's dtype. Nothing is copied.
/// * Any error from [`ExternalAdapterMarker::export_hidden`],
///   [`InterchangeMeta::of_buffer`], or
///   [`ExternalAdapterMarker::import_hidden`], unchanged.
/// * [`CortexError::ShapeMismatch`] or [`CortexError::StageDTypeMismatch`]
///   (stage `"hidden_state"`) when the imported tensor does not match the
///   exported buffer.
pub fn transfer<S, D>(source: &S, tensor: &S::Tensor, dest: &D) -> Result<D::Tensor>
where
    S: ExternalAdapterMarker + ?Sized,
    D: ExternalAdapterMarker + ?Sized,
{
    let source_caps = source.adapter_capabilities();
    let dest_caps = dest.adapter_capabilities();
    require_hidden_state_io(&source_caps, "export")?;
    require_hidden_state_io(&dest_caps, "import")?;
    let dtype = tensor.meta().dtype;
    if !dest_caps.ann.supported_dtypes.contains(&dtype) {
        return Err(refusal(
            &dest_caps,
            "dtype",
            format!("cannot import {dtype} hidden state"),
        ));
    }

    let buffer = source.export_hidden(tensor)?;
    let exported = InterchangeMeta::of_buffer(&buffer)?;
    let imported = dest.import_hidden(&buffer)?;
    let got = imported.meta();
    if got.shape != exported.shape {
        return Err(CortexError::ShapeMismatch {
            expected: exported.shape,
            got: got.shape,
        });
    }
    if got.dtype != exported.dtype {
        return Err(CortexError::StageDTypeMismatch {
            stage_id: HIDDEN_STATE.to_string(),
            expected: exported.dtype.to_string(),
            got: got.dtype.to_string(),
        });
    }
    Ok(imported)
}

fn require_hidden_state_io(caps: &AdapterCapabilities, direction: &str) -> Result<()> {
    if caps.hidden_state_io {
        Ok(())
    } else {
        Err(refusal(
            caps,
            "hidden_state_io",
            format!("cannot {direction} hidden state"),
        ))
    }
}

fn refusal(caps: &AdapterCapabilities, category: &'static str, detail: String) -> CortexError {
    CortexError::UnsupportedOperation {
        backend: caps.ann.backend_name,
        stage_id: None,
        category,
        detail,
    }
}

fn hidden_state_failure(err: CortexError) -> CortexError {
    CortexError::StageFailed {
        stage_id: HIDDEN_STATE.to_string(),
        source: Box::new(err),
    }
}

/// Candle's `HiddenState` is always `f32`, so it converts into the shared
/// buffer without a check.
#[cfg(feature = "candle")]
impl From<crate::adapter::HiddenState> for HiddenStateBuffer {
    fn from(state: crate::adapter::HiddenState) -> Self {
        Self {
            data: state.data,
            shape: state.shape,
            dtype: DType::F32,
        }
    }
}

/// Only an `f32` buffer whose length matches its shape converts into Candle's
/// `HiddenState`.
#[cfg(feature = "candle")]
impl TryFrom<HiddenStateBuffer> for crate::adapter::HiddenState {
    type Error = CortexError;

    /// # Errors
    ///
    /// Returns [`CortexError::StageDTypeMismatch`] for a non-`f32` buffer and
    /// [`CortexError::StageFailed`] for a length or size-overflow failure, as
    /// [`crate::stage::ReferenceExecutor::import_hidden`] does.
    fn try_from(buffer: HiddenStateBuffer) -> Result<Self> {
        crate::adapter::require_f32_buffer(&buffer)?;
        Ok(Self {
            data: buffer.data,
            shape: buffer.shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::BackendLimitations;
    use crate::stage::AnnCapabilities;
    use std::collections::BTreeSet;

    fn caps(dtypes: &[DType], devices: &[DeviceClass], batch: bool) -> AdapterCapabilities {
        AdapterCapabilities {
            ann: AnnCapabilities {
                backend_name: "probe",
                supported_tags: BTreeSet::new(),
                supported_dtypes: dtypes.iter().copied().collect(),
                stateful: false,
            },
            devices: devices.iter().copied().collect(),
            batch,
            sequence_cache: false,
            hidden_state_io: true,
            limitations: BackendLimitations {
                unsupported_tags: BTreeSet::new(),
                unsupported_dtypes: BTreeSet::new(),
                unsupported_devices: BTreeSet::new(),
                notes: Vec::new(),
            },
        }
    }

    fn meta(shape: &[usize], dtype: DType, device: TensorDevice) -> InterchangeMeta {
        InterchangeMeta::new(
            TensorMeta {
                shape: shape.to_vec(),
                dtype,
            },
            device,
            TensorLayout::RowMajor,
        )
    }

    #[test]
    fn tokens_are_stable() {
        assert_eq!(TensorDevice::HOST.to_string(), "cpu:0");
        assert_eq!(
            TensorDevice::new(DeviceClass::Cuda, 1).to_string(),
            "cuda:1"
        );
        assert_eq!(TensorLayout::RowMajor.to_string(), "row_major");
    }

    #[test]
    fn hidden_state_shape_names_rank_2_and_3() {
        let unbatched = HiddenStateShape::from_dims(&[5, 8]).unwrap();
        assert_eq!(unbatched.batch, None);
        assert_eq!((unbatched.seq_len, unbatched.hidden_dim), (5, 8));
        assert_eq!(unbatched.dims(), vec![5, 8]);

        let batched = HiddenStateShape::from_dims(&[2, 5, 8]).unwrap();
        assert_eq!(batched.batch, Some(2));
        assert_eq!(batched.dims(), vec![2, 5, 8]);
    }

    #[test]
    fn hidden_state_shape_rejects_other_ranks() {
        for (dims, expected) in [(&[][..], 2), (&[4][..], 2), (&[1, 2, 3, 4][..], 3)] {
            let err = HiddenStateShape::from_dims(dims).unwrap_err();
            assert!(
                matches!(err, CortexError::RankMismatch { expected: e, got } if e == expected && got == dims.len()),
                "{dims:?}: {err:?}"
            );
        }
    }

    #[test]
    fn of_buffer_checks_length_and_reports_host_row_major() {
        let ok = InterchangeMeta::of_buffer(&HiddenStateBuffer {
            data: vec![0.0; 6],
            shape: vec![2, 3],
            dtype: DType::F32,
        })
        .unwrap();
        assert_eq!(ok, meta(&[2, 3], DType::F32, TensorDevice::HOST));
        assert_eq!(ok.numel().unwrap(), 6);

        let err = InterchangeMeta::of_buffer(&HiddenStateBuffer {
            data: vec![0.0; 5],
            shape: vec![2, 3],
            dtype: DType::F32,
        })
        .unwrap_err();
        assert!(
            matches!(&err, CortexError::StageFailed { stage_id, source }
                if stage_id == "hidden_state"
                    && matches!(&**source, CortexError::ShapeMismatch { got, .. } if got == &[5])),
            "{err:?}"
        );

        let err = InterchangeMeta::of_buffer(&HiddenStateBuffer {
            data: Vec::new(),
            shape: vec![usize::MAX, 2],
            dtype: DType::F32,
        })
        .unwrap_err();
        assert!(
            matches!(&err, CortexError::StageFailed { source, .. }
                if matches!(&**source, CortexError::SizeOverflow { .. })),
            "{err:?}"
        );
    }

    #[test]
    fn require_supported_by_checks_dtype_then_device() {
        let cpu_f32 = caps(&[DType::F32], &[DeviceClass::Cpu], false);
        meta(&[2, 4], DType::F32, TensorDevice::HOST)
            .require_supported_by(&cpu_f32)
            .unwrap();

        let both_wrong = meta(
            &[2, 4],
            DType::BF16,
            TensorDevice::new(DeviceClass::Cuda, 0),
        );
        let err = both_wrong.require_supported_by(&cpu_f32).unwrap_err();
        assert!(
            matches!(&err, CortexError::UnsupportedOperation { backend: "probe", stage_id: None, category: "dtype", detail } if detail.contains("bf16")),
            "{err:?}"
        );

        let wrong_device = meta(
            &[2, 4],
            DType::F32,
            TensorDevice::new(DeviceClass::Metal, 1),
        );
        let err = wrong_device.require_supported_by(&cpu_f32).unwrap_err();
        assert!(
            matches!(&err, CortexError::UnsupportedOperation { category: "device", detail, .. } if detail.contains("metal:1")),
            "{err:?}"
        );
    }

    #[test]
    fn hidden_state_check_respects_batch_capability() {
        let unbatched = caps(&[DType::F32], &[DeviceClass::Cpu], false);
        let batched = caps(&[DType::F32], &[DeviceClass::Cpu], true);
        let three_d = meta(&[2, 3, 4], DType::F32, TensorDevice::HOST);

        let err = three_d
            .require_hidden_state_supported_by(&unbatched)
            .unwrap_err();
        assert!(
            matches!(
                err,
                CortexError::UnsupportedOperation {
                    category: "batch",
                    ..
                }
            ),
            "{err:?}"
        );
        let axes = three_d.require_hidden_state_supported_by(&batched).unwrap();
        assert_eq!(axes.batch, Some(2));

        let two_d = meta(&[3, 4], DType::F32, TensorDevice::HOST);
        assert_eq!(
            two_d
                .require_hidden_state_supported_by(&unbatched)
                .unwrap()
                .hidden_dim,
            4
        );
    }

    #[test]
    fn reference_tensor_describes_itself_as_host_row_major() {
        let tensor = crate::Tensor::try_from_vec(vec![1.0; 6], &[2, 3]).unwrap();
        let described = tensor.interchange_meta();
        assert_eq!(described, meta(&[2, 3], DType::F32, TensorDevice::HOST));
        assert_eq!(described.tensor_meta(), tensor.meta());
    }
}
