//! Flat/dark normalization and minus-log (ports tomopy `prep/normalize.py`).
//!
//! Flat/dark correction and minus-log are thin, backend-agnostic wrappers over
//! the [`Elementwise`] capability so the same call runs on CPU/CUDA/wgpu.
//! Background (air-region) normalization is a per-row reduction, so it is a
//! direct CPU port matching the libtomo C bit-for-bit.

use tomoxide_core::backend::Backend;
use tomoxide_core::data::{Dataset, Frames, Layout, Tomo};
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

/// Background (air-region) normalization — a direct port of tomopy
/// `prep/normalize.py::normalize_bg` / `libtomo/prep/prep.c::normalize_bg`.
///
/// For each projection row the mean of the `air` left-boundary pixels and the
/// `air` right-boundary pixels (typically the air around the object) gives an
/// air baseline that is linearly interpolated across the detector width; every
/// pixel is divided by its local baseline, so the boundaries are scaled to one.
/// Non-positive boundary means are clamped to `1` (matching the C). All
/// arithmetic is f32 in the upstream accumulation order, so the result matches
/// tomopy bit-for-bit (Δ = 0). Projector-independent.
pub fn normalize_bg(data: &mut Tomo<f32>, air: usize) -> Result<()> {
    let target = data.layout;
    // The C indexes `data[m·dz·dy + n·dz + j]` over `(dx=proj, dy=row, dz=col)`,
    // i.e. the `[proj, row, col]` projection layout; the air boundary is the
    // left/right edge of the detector-column (`dz`) axis.
    let mut proj = data.to_layout(Layout::Projection);
    let (dx, dy, dz) = proj.array.dim();
    if dx == 0 || dy == 0 || dz == 0 {
        return Ok(());
    }
    // The C reads `air` pixels in from each boundary; clamp to the row width so a
    // malformed `air > dz` cannot over-read (C's behaviour there is undefined).
    let nair = air.min(dz);
    let arr = &mut proj.array;

    for m in 0..dx {
        for n in 0..dy {
            // Boundary air means, accumulated in f32 in the upstream order.
            let mut air_left = 0.0f32;
            let mut air_right = 0.0f32;
            for j in 0..nair {
                air_left += arr[[m, n, j]];
                air_right += arr[[m, n, dz - 1 - j]];
            }
            air_left /= nair as f32;
            air_right /= nair as f32;
            if air_left <= 0.0 {
                air_left = 1.0;
            }
            if air_right <= 0.0 {
                air_right = 1.0;
            }
            // Linear baseline across the row; divide each pixel by its local air.
            // The C `air_left + air_slope * j` is one statement, which clang
            // contracts to a fused multiply-add (`-ffp-contract=on`, the default);
            // use `mul_add` so the single rounding matches bit-for-bit.
            let air_slope = (air_right - air_left) / (dz - 1) as f32;
            for j in 0..dz {
                let air_val = air_slope.mul_add(j as f32, air_left);
                arr[[m, n, j]] /= air_val;
            }
        }
    }

    *data = proj.to_layout(target);
    Ok(())
}
