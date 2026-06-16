//! Flat/dark normalization and minus-log (ports tomopy `prep/normalize.py`).
//!
//! Flat/dark correction and minus-log are thin, backend-agnostic wrappers over
//! the [`Elementwise`] capability so the same call runs on CPU/CUDA/wgpu.
//! Background (air-region) normalization is a per-row reduction, so it is a
//! direct CPU port matching the libtomo C bit-for-bit.

use ndarray::Array2;
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

/// Dark-frame averaging mode for [`normalize_nf`] (tomopy `averaging=`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Averaging {
    /// `np.mean(dark, axis=0)` (tomopy default).
    Mean,
    /// `np.median(dark, axis=0)` — tomopy passes `dtype=np.float32` to
    /// `np.median`, which raises `TypeError` on modern numpy, so no reference
    /// output exists; [`normalize_nf`] returns a TODO error for this mode.
    Median,
}

/// Per-pixel median over `frames[start..end]` along axis 0 → `[ny, nx]`
/// (`np.median(..., axis=0)` on f32: even counts average the two central values
/// in f32, odd counts select the centre).
fn median_frames(frames: &ndarray::Array3<f32>, start: usize, end: usize) -> Array2<f32> {
    let (_, ny, nx) = frames.dim();
    let g = end - start;
    let mid = g / 2;
    let mut out = Array2::<f32>::zeros((ny, nx));
    let mut col = vec![0.0f32; g];
    for y in 0..ny {
        for x in 0..nx {
            for (k, c) in col.iter_mut().enumerate() {
                *c = frames[[start + k, y, x]];
            }
            col.sort_by(|a, b| a.total_cmp(b));
            out[[y, x]] = if g % 2 == 0 {
                (col[mid - 1] + col[mid]) / 2.0
            } else {
                col[mid]
            };
        }
    }
    out
}

/// Banker's rounding of `diff/2` (numpy `int(np.round(diff/2))`, half-to-even),
/// computed in integers so it is exact.
fn round_half_even_div2(diff: usize) -> usize {
    let k = diff / 2;
    if diff % 2 == 0 || k % 2 == 0 {
        k
    } else {
        k + 1
    }
}

/// Nearest-flat-fields normalization — a port of tomopy
/// `prep/normalize.py::normalize_nf` (default `averaging='mean'`).
///
/// `flats` holds `num_flats = flat_loc.len()` equally-sized flat groups
/// (`flats.len() / num_flats` frames each); each group's per-pixel **median**
/// is the flat for the projections nearest its `flat_loc` position. Each
/// projection is normalized as `(proj − dark) / max(flat − dark, 1e-6)`, with
/// `dark` the per-pixel **mean** of the dark frames; an optional `cutoff` clamps
/// the result from above. Group boundaries fall at the half-sample midpoint
/// between consecutive `flat_loc` entries (`int(np.round(Δ/2)) + loc`,
/// half-to-even). All arithmetic is f32 in the upstream order, so the result
/// matches tomopy bit-for-bit (Δ = 0). Projector-independent.
pub fn normalize_nf(
    data: &mut Tomo<f32>,
    flats: &Frames<f32>,
    dark: &Frames<f32>,
    flat_loc: &[usize],
    cutoff: Option<f32>,
    averaging: Averaging,
) -> Result<()> {
    let target = data.layout;
    let mut proj = data.to_layout(Layout::Projection);
    let (nproj, ny, nx) = proj.array.dim();
    let (nflat, fy, fx) = flats.array.dim();
    let (ndark, ky, kx) = dark.array.dim();
    if fy != ny || fx != nx || ky != ny || kx != nx {
        return Err(Error::InvalidParam(
            "normalize_nf: flats/dark frame shape must match the projection frame".into(),
        ));
    }
    let num_flats = flat_loc.len();
    if num_flats == 0 || nflat < num_flats || ndark == 0 {
        return Err(Error::InvalidParam(
            "normalize_nf: need ≥1 flat group (flat_loc) and ≥1 flat/dark frame".into(),
        ));
    }
    // dark = average of the dark frames over axis 0.
    let dark2d = match averaging {
        Averaging::Mean => {
            // np.mean(dark, axis=0, dtype=float32): f32 accumulation / count.
            let mut d = Array2::<f32>::zeros((ny, nx));
            for y in 0..ny {
                for x in 0..nx {
                    let mut s = 0.0f32;
                    for k in 0..ndark {
                        s += dark.array[[k, y, x]];
                    }
                    d[[y, x]] = s / ndark as f32;
                }
            }
            d
        }
        Averaging::Median => {
            return Err(Error::todo(
                "normalize::normalize_nf (averaging='median')",
                "tomopy prep/normalize.py:305 — np.median(dtype=) raises on modern numpy",
            ));
        }
    };

    let num_per_flat = nflat / num_flats; // floor; trailing flats are unused
    let l = 1e-6f32;
    let mut tend = 0usize;

    for m in 0..num_flats {
        let fstart = m * num_per_flat;
        let fend = fstart + num_per_flat;
        let flat = median_frames(&flats.array, fstart, fend);
        let tstart = if m == 0 { 0 } else { tend };
        tend = if m + 1 >= num_flats {
            nproj
        } else {
            (round_half_even_div2(flat_loc[m + 1] - flat_loc[m]) + flat_loc[m]).min(nproj)
        };
        // denom = max(flat − dark, 1e-6), computed once per group.
        let mut denom = Array2::<f32>::zeros((ny, nx));
        for y in 0..ny {
            for x in 0..nx {
                let d = flat[[y, x]] - dark2d[[y, x]];
                denom[[y, x]] = if d < l { l } else { d };
            }
        }
        for p in tstart..tend {
            for y in 0..ny {
                for x in 0..nx {
                    let mut v = (proj.array[[p, y, x]] - dark2d[[y, x]]) / denom[[y, x]];
                    if let Some(c) = cutoff {
                        if v > c {
                            v = c;
                        }
                    }
                    proj.array[[p, y, x]] = v;
                }
            }
        }
    }

    *data = proj.to_layout(target);
    Ok(())
}
