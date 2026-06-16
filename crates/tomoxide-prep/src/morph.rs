//! Sinogram morphology (ports tomopy `misc/morph.py`).
//!
//! Implements `sino_360_to_180` (folding a 0–360° sinogram into a 0–180° one by
//! stitching the column-reversed second half-rotation onto the first with a
//! linear seam cross-fade) and `downsample`/`upsample` (power-of-two binning /
//! replication along one axis, ports of `libtomo/misc/morph.c`).

use ndarray::Array3;
use tomoxide_core::data::{Layout, Tomo};
use tomoxide_core::error::{Error, Result};

/// Side of the field of view the rotation axis is closest to (tomopy's
/// `rotation='left'`/`'right'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotation {
    /// Rotation centre near the left of the field of view.
    Left,
    /// Rotation centre near the right of the field of view.
    Right,
}

/// `np.linspace(start, stop, num)` (endpoint=True): `arange(num)·step + start`
/// with the last sample forced exactly to `stop`, computed in f64 like numpy.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut y: Vec<f64> = (0..num).map(|k| k as f64 * step + start).collect();
    y[num - 1] = stop; // endpoint forced, matching numpy
    y
}

/// Convert a 0–360° sinogram to a 0–180° sinogram (tomopy
/// `misc/morph.py::sino_360_to_180`).
///
/// The first `n = dx/2` projections cover 0–180° and the next `n` cover
/// 180–360° (an odd final projection is discarded). The second set is reversed
/// along the detector-column axis and stitched onto the first to widen the
/// detector, overlapping by `overlap` columns where the two are linearly
/// cross-faded. For `Rotation::Left` the centre is near the left edge, so the
/// first set occupies the right side of the output (and vice-versa). The output
/// is `[n, dy, 2·dz − overlap]` in the projection layout.
///
/// Direct (non-seam) regions are exact f32 copies; the seam blend is computed in
/// f64 (numpy promotes `float64-weights · float32-data`) and cast back to f32,
/// so the result matches tomopy bit-for-bit (Δ = 0).
pub fn sino_360_to_180(data: &Tomo<f32>, overlap: usize, rotation: Rotation) -> Result<Tomo<f32>> {
    // tomopy indexes `data.shape = (dx, dy, dz) = (proj, row, col)`.
    let src = data.to_layout(Layout::Projection);
    let (dx, dy, dz) = src.array.dim();
    if overlap > dz {
        return Err(Error::InvalidParam(format!(
            "sino_360_to_180: overlap ({overlap}) exceeds detector width ({dz})"
        )));
    }
    let n = dx / 2;
    let width = 2 * dz - overlap;
    let mut out = Array3::<f32>::zeros((n, dy, width));
    if n == 0 || dy == 0 || dz == 0 {
        return Ok(Tomo::new(out, Layout::Projection));
    }
    let a = &src.array;
    let keep = dz - overlap; // width of each direct-copy region

    match rotation {
        Rotation::Left => {
            // weights = linspace(0, 1, overlap).
            let w = linspace(0.0, 1.0, overlap);
            for p in 0..n {
                for r in 0..dy {
                    // Region A [0 : keep): column-reversed second half (cols overlap..dz-1).
                    for t in 0..keep {
                        out[[p, r, t]] = a[[n + p, r, dz - 1 - t]];
                    }
                    // Region C [dz : width): first half (cols overlap..dz-1).
                    for t in 0..keep {
                        out[[p, r, dz + t]] = a[[p, r, overlap + t]];
                    }
                    // Region B [keep : dz): seam cross-fade.
                    for c in 0..overlap {
                        let first = w[c] * a[[p, r, c]] as f64;
                        let second = w[overlap - 1 - c] * a[[n + p, r, overlap - 1 - c]] as f64;
                        out[[p, r, keep + c]] = (first + second) as f32;
                    }
                }
            }
        }
        Rotation::Right => {
            // weights = linspace(1, 0, overlap).
            let w = linspace(1.0, 0.0, overlap);
            for p in 0..n {
                for r in 0..dy {
                    // Region A [0 : keep): first half (cols 0..keep-1).
                    for t in 0..keep {
                        out[[p, r, t]] = a[[p, r, t]];
                    }
                    // Region C [dz : width): column-reversed second half (cols 0..keep-1).
                    for t in 0..keep {
                        out[[p, r, dz + t]] = a[[n + p, r, keep - 1 - t]];
                    }
                    // Region B [keep : dz): seam cross-fade (first/second cols keep..dz-1).
                    for c in 0..overlap {
                        let first = w[c] * a[[p, r, keep + c]] as f64;
                        let second = w[overlap - 1 - c] * a[[n + p, r, dz - 1 - c]] as f64;
                        out[[p, r, keep + c]] = (first + second) as f32;
                    }
                }
            }
        }
    }

    Ok(Tomo::new(out, Layout::Projection))
}

/// Output dimensions and a C-order flat view of `arr`, with `axis` validated.
fn sample_setup(arr: &Array3<f32>, axis: usize) -> Result<((usize, usize, usize), Vec<f32>)> {
    if axis > 2 {
        return Err(Error::InvalidParam(format!(
            "downsample/upsample: axis ({axis}) must be 0, 1 or 2"
        )));
    }
    let std = arr.as_standard_layout(); // C-contiguous copy if needed
    let flat = std.iter().copied().collect();
    Ok((arr.dim(), flat))
}

/// Downsample a 3D array by `2^level` along `axis` — a port of
/// `libtomo/misc/morph.c::downsample` (tomopy `misc/morph.py::downsample`).
///
/// Each output sample is the mean of a `binsize = 2^level` bin, accumulated as
/// `Σ(data / binsize)` in f32 in the upstream order (the C walks a single flat
/// input counter, faithfully reproduced here), so the result matches tomopy
/// bit-for-bit (Δ = 0). The sampled axis shrinks to `⌊dim / binsize⌋`; if it is
/// not divisible the C's running-counter behaviour is preserved exactly.
pub fn downsample(arr: &Array3<f32>, level: u32, axis: usize) -> Result<Array3<f32>> {
    let ((dx, dy, dz), data) = sample_setup(arr, axis)?;
    let binsize = 1usize << level;
    let bs_f = binsize as f32;
    // Output dims: the sampled axis is floor-divided by binsize.
    let (odx, ody, odz) = match axis {
        0 => (dx / binsize, dy, dz),
        1 => (dx, dy / binsize, dz),
        _ => (dx, dy, dz / binsize),
    };
    let mut out = vec![0.0f32; odx * ody * odz];
    let mut ind = 0usize; // flat input counter, exactly as the C
    match axis {
        0 => {
            for m in 0..odx {
                let i = m * (ody * odz); // ody==dy, odz==dz here
                for _p in 0..binsize {
                    for n in 0..dy {
                        let j = n * dz;
                        for k in 0..dz {
                            out[i + j + k] += data[ind] / bs_f;
                            ind += 1;
                        }
                    }
                }
            }
        }
        1 => {
            for m in 0..dx {
                let i = m * (ody * dz);
                for n in 0..ody {
                    let j = n * dz;
                    for _p in 0..binsize {
                        for k in 0..dz {
                            out[i + j + k] += data[ind] / bs_f;
                            ind += 1;
                        }
                    }
                }
            }
        }
        _ => {
            for m in 0..dx {
                let i = m * (dy * odz);
                for n in 0..dy {
                    let j = n * odz;
                    for k in 0..odz {
                        for _p in 0..binsize {
                            out[i + j + k] += data[ind] / bs_f;
                            ind += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(Array3::from_shape_vec((odx, ody, odz), out).expect("downsample shape"))
}

/// Upsample a 3D array by `2^level` along `axis` — a port of
/// `libtomo/misc/morph.c::upsample` (tomopy `misc/morph.py::upsample`).
///
/// Each input value is replicated `binsize = 2^level` times along the axis (no
/// arithmetic), so the result is bit-exact (Δ = 0). The sampled axis grows to
/// `dim · binsize`.
pub fn upsample(arr: &Array3<f32>, level: u32, axis: usize) -> Result<Array3<f32>> {
    let ((dx, dy, dz), data) = sample_setup(arr, axis)?;
    let binsize = 1usize << level;
    let (odx, ody, odz) = match axis {
        0 => (dx * binsize, dy, dz),
        1 => (dx, dy * binsize, dz),
        _ => (dx, dy, dz * binsize),
    };
    let mut out = vec![0.0f32; odx * ody * odz];
    let mut ind = 0usize; // flat output counter, exactly as the C
    match axis {
        0 => {
            for m in 0..dx {
                let i = m * (dy * dz);
                for _p in 0..binsize {
                    for n in 0..dy {
                        let j = n * dz;
                        for k in 0..dz {
                            out[ind] = data[i + j + k];
                            ind += 1;
                        }
                    }
                }
            }
        }
        1 => {
            for m in 0..dx {
                let i = m * (dy * dz);
                for n in 0..dy {
                    let j = n * dz;
                    for _p in 0..binsize {
                        for k in 0..dz {
                            out[ind] = data[i + j + k];
                            ind += 1;
                        }
                    }
                }
            }
        }
        _ => {
            for m in 0..dx {
                let i = m * (dy * dz);
                for n in 0..dy {
                    let j = n * dz;
                    for k in 0..dz {
                        for _p in 0..binsize {
                            out[ind] = data[i + j + k];
                            ind += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(Array3::from_shape_vec((odx, ody, odz), out).expect("upsample shape"))
}
