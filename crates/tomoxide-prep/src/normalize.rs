//! Flat/dark normalization and minus-log (ports tomopy `prep/normalize.py`).
//!
//! Thin, backend-agnostic wrappers over the [`Elementwise`] capability so the
//! same call runs on CPU/CUDA/wgpu.

use tomoxide_core::backend::Backend;
use tomoxide_core::data::{Dataset, Frames, Tomo};
use tomoxide_core::error::{Error, Result};

fn elementwise(backend: &dyn Backend) -> Result<&dyn tomoxide_core::backend::Elementwise> {
    backend.elementwise().ok_or(Error::MissingCapability {
        backend: backend.name(),
        capability: "Elementwise",
    })
}

/// `(data − dark) / (flat − dark)` (tomopy `normalize.py:98`).
pub fn normalize(
    data: &mut Tomo<f32>,
    flat: &Frames<f32>,
    dark: &Frames<f32>,
    backend: &dyn Backend,
) -> Result<()> {
    elementwise(backend)?.darkflat(data, flat, dark)
}

/// In-place `−log` (tomopy `normalize.py:72`).
pub fn minus_log(data: &mut Tomo<f32>, backend: &dyn Backend) -> Result<()> {
    elementwise(backend)?.minus_log(data)
}

/// Full flat-field correction then minus-log on a [`Dataset`], in place.
///
/// No-ops the dark/flat step when either is absent (already-normalized input).
pub fn normalize_dataset(ds: &mut Dataset<f32>, backend: &dyn Backend) -> Result<()> {
    if let (Some(flat), Some(dark)) = (&ds.flat, &ds.dark) {
        normalize(&mut ds.data, flat, dark, backend)?;
    }
    minus_log(&mut ds.data, backend)
}
