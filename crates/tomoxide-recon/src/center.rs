//! Rotation-center finding (ports tomopy `recon/rotation.py` + tomocupy
//! `find_center.py`). `find_center_vo` (Nghia Vo's method) is implemented; the
//! others remain stubs. See `docs/PORTING.md` §C.

use ndarray::{Array2, Axis, Slice};
use tomoxide_core::backend::{Backend, Fft};
use tomoxide_core::data::{Layout, Tomo};
use tomoxide_core::dtype::Complex32;
use tomoxide_core::error::{Error, Result};

/// Entropy-based center finding (tomopy `rotation.py:82`).
pub fn find_center(
    _sino: &Tomo<f32>,
    _theta: &[f32],
    _init: Option<f32>,
    _tol: f32,
) -> Result<f32> {
    Err(Error::todo(
        "center::find_center",
        "tomopy recon/rotation.py:82",
    ))
}

/// Nghia Vo's coarse+fine center search — the workhorse (tomopy `rotation.py:205`).
///
/// Sinogram-domain Fourier method (Vo et al. 2014, doi:10.1364/OE.22.019078):
/// pick one sinogram slice, denoise it, then minimise a double-wedge-masked
/// `mean(|fftshift(fft2(·))|)` metric over candidate centers — a coarse
/// integer-grid pass (step 0.5) followed by a fine subpixel pass (step `step`).
/// Because the metric lives entirely in the sinogram's Fourier domain it is
/// projector-independent, so it matches tomopy numerically (unlike the
/// projector-coupled analytic/iterative methods documented in PORTING).
///
/// `ind` selects the slice (default `n_rows/2`, averaging ±5 rows for SNR when
/// there are more than 10); `smin`/`smax` bound the coarse search around the
/// detector midline; `srad`/`step` set the fine radius/step; `ratio` sizes the
/// double-wedge mask (FOV/object) and `drop` blanks rows around its DC line.
#[allow(clippy::too_many_arguments)]
pub fn find_center_vo(
    tomo: &Tomo<f32>,
    backend: &dyn Backend,
    ind: Option<usize>,
    smin: f32,
    smax: f32,
    srad: f32,
    step: f32,
    ratio: f32,
    drop: i32,
) -> Result<f32> {
    let fft = backend.fft().ok_or_else(|| Error::MissingCapability {
        backend: backend.name(),
        capability: "Fft",
    })?;

    let sino = extract_slice(tomo, ind); // (nang, ncol)
    let (nrow, ncol) = sino.dim();

    // Denoise. The coarse and fine passes deliberately use different window
    // sizes (tomopy rotation.py:251-255).
    let tomo_cs = gaussian_filter2d(&sino, 3.0, 1.0);
    let tomo_fs = gaussian_filter2d(&sino, 2.0, 2.0);

    // For large data (>2k×2k) coarse-search a 4× column-downsampled copy.
    let fine_cen = if nrow * ncol > 4_000_000 {
        let coarse = downsample_cols(&tomo_cs, 2);
        let init_cen = search_coarse(&coarse, smin / 4.0, smax / 4.0, ratio, drop, fft)? * 4.0;
        search_fine(&tomo_fs, srad, step, init_cen, ratio, drop, fft)?
    } else {
        let init_cen = search_coarse(&tomo_cs, smin, smax, ratio, drop, fft)?;
        search_fine(&tomo_fs, srad, step, init_cen, ratio, drop, fft)?
    };
    Ok(fine_cen)
}

/// Phase-correlation between a 0°/180° pair (tomopy `rotation.py:391`).
pub fn find_center_pc(_proj0: &[f32], _proj180: &[f32], _tol: f32) -> Result<f32> {
    Err(Error::todo(
        "center::find_center_pc",
        "tomopy recon/rotation.py:391",
    ))
}

/// SIFT-feature center detection (tomocupy `find_center.py:99`).
pub fn find_center_sift(_proj0: &[f32], _proj180: &[f32]) -> Result<f32> {
    Err(Error::todo(
        "center::find_center_sift",
        "tomocupy find_center.py:99",
    ))
}

// ----------------------------------------------------------------------------
// Vo's method internals (private). Mirrors tomopy rotation.py:236-388.
// ----------------------------------------------------------------------------

/// Pull a single 2-D sinogram slice `(angle, col)`, averaging ±5 rows around
/// `ind` for SNR when there are more than 10 slices (tomopy rotation.py:240-249).
fn extract_slice(tomo: &Tomo<f32>, ind: Option<usize>) -> Array2<f32> {
    let sino = tomo.to_layout(Layout::Sinogram); // [row, angle, col]
    let nrows = sino.array.dim().0;
    let ind = ind.unwrap_or(nrows / 2).min(nrows.saturating_sub(1));
    if nrows > 10 {
        let lo = ind.saturating_sub(5);
        let hi = (ind + 5).min(nrows);
        sino.array
            .slice_axis(Axis(0), Slice::from(lo..hi))
            .mean_axis(Axis(0))
            .unwrap()
    } else {
        sino.array.index_axis(Axis(0), ind).to_owned()
    }
}

/// scipy `gaussian_filter` weights for one axis: a normalised Gaussian truncated
/// at `truncate=4.0` standard deviations (`radius = int(4·σ + 0.5)`).
fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (4.0 * sigma + 0.5) as isize;
    let mut k = Vec::with_capacity((2 * radius + 1) as usize);
    let mut sum = 0.0f32;
    for i in -radius..=radius {
        let w = (-0.5 * (i as f32 / sigma).powi(2)).exp();
        k.push(w);
        sum += w;
    }
    for w in &mut k {
        *w /= sum;
    }
    k
}

/// scipy `'reflect'` boundary (half-sample symmetric: `d c b a | a b c d`).
fn reflect(p: isize, n: isize) -> usize {
    if n == 1 {
        return 0;
    }
    let m = 2 * n;
    let mut q = p % m;
    if q < 0 {
        q += m;
    }
    if q >= n {
        q = m - 1 - q;
    }
    q as usize
}

/// Separable 2-D Gaussian blur, `sigma0` along the angle axis and `sigma1`
/// along the column axis, `'reflect'` boundary (tomopy uses `mode='reflect'`).
fn gaussian_filter2d(img: &Array2<f32>, sigma0: f32, sigma1: f32) -> Array2<f32> {
    let (nr, nc) = img.dim();

    // Pass 1: columns (axis 1).
    let k1 = gaussian_kernel(sigma1);
    let r1 = (k1.len() / 2) as isize;
    let mut tmp = Array2::<f32>::zeros((nr, nc));
    for i in 0..nr {
        for j in 0..nc {
            let mut acc = 0.0f32;
            for (t, &w) in k1.iter().enumerate() {
                let jj = reflect(j as isize + t as isize - r1, nc as isize);
                acc += w * img[[i, jj]];
            }
            tmp[[i, j]] = acc;
        }
    }

    // Pass 2: rows (axis 0).
    let k0 = gaussian_kernel(sigma0);
    let r0 = (k0.len() / 2) as isize;
    let mut out = Array2::<f32>::zeros((nr, nc));
    for j in 0..nc {
        for i in 0..nr {
            let mut acc = 0.0f32;
            for (t, &w) in k0.iter().enumerate() {
                let ii = reflect(i as isize + t as isize - r0, nr as isize);
                acc += w * tmp[[ii, j]];
            }
            out[[i, j]] = acc;
        }
    }
    out
}

/// Column block-mean downsample by `2^level` (tomopy `downsample(level, axis=2)`).
/// argmin of the metric is scale-invariant, so mean vs sum is immaterial here.
fn downsample_cols(img: &Array2<f32>, level: u32) -> Array2<f32> {
    let f = 1usize << level;
    let (nr, nc) = img.dim();
    let nc2 = nc / f;
    let mut out = Array2::<f32>::zeros((nr, nc2));
    for i in 0..nr {
        for k in 0..nc2 {
            let mut s = 0.0f32;
            for d in 0..f {
                s += img[[i, k * f + d]];
            }
            out[[i, k]] = s / f as f32;
        }
    }
    out
}

/// Reverse columns (`np.fliplr`).
fn fliplr(a: &Array2<f32>) -> Array2<f32> {
    let (nr, nc) = a.dim();
    Array2::from_shape_fn((nr, nc), |(i, j)| a[[i, nc - 1 - j]])
}

/// Reverse rows (`np.flipud`).
fn flipud(a: &Array2<f32>) -> Array2<f32> {
    let (nr, nc) = a.dim();
    Array2::from_shape_fn((nr, nc), |(i, j)| a[[nr - 1 - i, j]])
}

fn clipf(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

/// Coarse search on the 0.5-pixel grid (tomopy `_search_coarse`).
fn search_coarse(
    sino: &Array2<f32>,
    smin: f32,
    smax: f32,
    ratio: f32,
    drop: i32,
    fft: &dyn Fft,
) -> Result<f32> {
    let (nrow, ncol) = sino.dim();
    let ncolf = ncol as f32;
    let cen_fliplr = (ncolf - 1.0) / 2.0;
    // np.int16(np.clip(s + cen, 0, ncol-1) - cen): truncates toward zero.
    let smin_i = (clipf(smin + cen_fliplr, 0.0, ncolf - 1.0) - cen_fliplr) as i32;
    let smax_i = (clipf(smax + cen_fliplr, 0.0, ncolf - 1.0) - cen_fliplr) as i32;
    let start_cor = (ncol / 2) as i32 + smin_i;
    let stop_cor = (ncol / 2) as i32 + smax_i;

    let flip = fliplr(sino);
    let comp = flipud(sino);
    let mask = create_mask(2 * nrow, ncol, 0.5 * ratio * ncolf, drop);

    // list_cor = np.arange(start_cor, stop_cor + 0.5, 0.5)
    let stop = stop_cor as f64 + 0.5;
    let (mut best, mut best_cor) = (f32::INFINITY, start_cor as f32);
    let mut k = 0i64;
    loop {
        let cor = start_cor as f64 + 0.5 * k as f64;
        if cor >= stop {
            break;
        }
        let shift = 2.0 * (cor - cen_fliplr as f64);
        let m = calculate_metric(shift, sino, &flip, &comp, &mask, fft)?;
        if m < best {
            best = m;
            best_cor = cor as f32;
        }
        k += 1;
    }
    Ok(best_cor)
}

/// Fine subpixel search around `init_cen` (tomopy `_search_fine`).
fn search_fine(
    sino: &Array2<f32>,
    srad: f32,
    step: f32,
    init_cen: f32,
    ratio: f32,
    drop: i32,
    fft: &dyn Fft,
) -> Result<f32> {
    let (nrow, ncol) = sino.dim();
    let ncolf = ncol as f32;
    let cen_fliplr = (ncolf - 1.0) / 2.0;
    let srad = clipf(srad.abs(), 1.0, ncolf / 4.0);
    let step = clipf(step.abs(), 0.1, srad);
    let init_cen = clipf(init_cen, srad, ncolf - srad - 1.0);

    let flip = fliplr(sino);
    let comp = flipud(sino);
    let mask = create_mask(2 * nrow, ncol, 0.5 * ratio * ncolf, drop);

    // list_cor = init_cen + np.arange(-srad, srad + step, step)
    let stop = (srad + step) as f64;
    let (mut best, mut best_cor) = (f32::INFINITY, init_cen - srad);
    let mut k = 0i64;
    loop {
        let off = -(srad as f64) + step as f64 * k as f64;
        if off >= stop {
            break;
        }
        let cor = init_cen as f64 + off;
        let shift = 2.0 * (cor - cen_fliplr as f64);
        let m = calculate_metric(shift, sino, &flip, &comp, &mask, fft)?;
        if m < best {
            best = m;
            best_cor = cor as f32;
        }
        k += 1;
    }
    Ok(best_cor)
}

/// The double-wedge binary mask, Eq.(3) of doi:10.1364/OE.22.019078
/// (tomopy `_create_mask`). `nrow` here is `2 ·` the sinogram angle count.
fn create_mask(nrow: usize, ncol: usize, radius: f32, drop: i32) -> Array2<f32> {
    let nrowf = nrow as f32;
    let ncolf = ncol as f32;
    let du = 1.0 / ncolf;
    let dv = (nrowf - 1.0) / (nrowf * 2.0 * std::f32::consts::PI);
    let cen_row = (nrowf / 2.0).ceil() as i32 - 1;
    let cen_col = (ncolf / 2.0).ceil() as i32 - 1;
    let drop = drop.min((0.05 * nrowf).ceil() as i32);

    let mut mask = Array2::<f32>::zeros((nrow, ncol));
    let ncol_i = ncol as i32;
    for i in 0..nrow as i32 {
        let val = (((i - cen_row) as f32) * dv / radius) / du;
        let pos = val.ceil() as i32; // np.int16(np.ceil(..)): toward zero.
        let (a, b) = (-pos + cen_col, pos + cen_col);
        let (p1, p2) = if a <= b { (a, b) } else { (b, a) };
        let p1 = p1.clamp(0, ncol_i - 1);
        let p2 = p2.clamp(0, ncol_i - 1);
        for j in p1..=p2 {
            mask[[i as usize, j as usize]] = 1.0;
        }
    }
    // Blank ±drop rows around the DC line and the three central columns.
    let r0 = (cen_row - drop).max(0);
    let r1 = (cen_row + drop).min(nrow as i32 - 1);
    for i in r0..=r1 {
        for j in 0..ncol {
            mask[[i as usize, j]] = 0.0;
        }
    }
    let c0 = (cen_col - 1).max(0);
    let c1 = (cen_col + 1).min(ncol_i - 1);
    for mut row in mask.rows_mut() {
        for j in c0..=c1 {
            row[j as usize] = 0.0;
        }
    }
    mask
}

/// The metric for one candidate `shift` (tomopy `_calculate_metric`): stack the
/// sinogram on its flipped-and-shifted copy, FFT, and average the
/// double-wedge-masked spectrum. Integer shifts use a circular roll; fractional
/// shifts use a cubic B-spline column shift. The wrapped/zero-filled boundary
/// columns are overwritten with the row-flipped sinogram (`comp`).
fn calculate_metric(
    shift: f64,
    sino: &Array2<f32>,
    flip: &Array2<f32>,
    comp: &Array2<f32>,
    mask: &Array2<f32>,
    fft: &dyn Fft,
) -> Result<f32> {
    let (nrow, ncol) = sino.dim();
    let nc = ncol as i32;

    let ss = if (shift - shift.round()).abs() < 1e-9 {
        let s = shift.round() as i32;
        let mut ss = Array2::<f32>::zeros((nrow, ncol));
        for i in 0..nrow {
            for j in 0..ncol {
                let src = ((j as i32 - s) % nc + nc) % nc;
                ss[[i, j]] = flip[[i, src as usize]];
            }
        }
        if s >= 0 {
            for i in 0..nrow {
                for j in 0..(s as usize).min(ncol) {
                    ss[[i, j]] = comp[[i, j]];
                }
            }
        } else {
            let from = (nc + s).max(0) as usize;
            for i in 0..nrow {
                for j in from..ncol {
                    ss[[i, j]] = comp[[i, j]];
                }
            }
        }
        ss
    } else {
        let mut ss = cubic_shift_cols(flip, shift);
        if shift >= 0.0 {
            let si = (shift.ceil() as i32).max(0) as usize;
            for i in 0..nrow {
                for j in 0..si.min(ncol) {
                    ss[[i, j]] = comp[[i, j]];
                }
            }
        } else {
            let from = (nc + shift.floor() as i32).max(0) as usize;
            for i in 0..nrow {
                for j in from..ncol {
                    ss[[i, j]] = comp[[i, j]];
                }
            }
        }
        ss
    };

    // mat = vstack((sino, ss)) -> (2·nrow, ncol), then fft2.
    let rr = 2 * nrow;
    let mut buf = vec![Complex32::new(0.0, 0.0); rr * ncol];
    for i in 0..nrow {
        for j in 0..ncol {
            buf[i * ncol + j] = Complex32::new(sino[[i, j]], 0.0);
            buf[(nrow + i) * ncol + j] = Complex32::new(ss[[i, j]], 0.0);
        }
    }
    fft.fft_2d(&mut buf, rr, ncol, 1, false)?;

    // metric = mean(|fftshift(fft2(mat))| * mask). Fold the fftshift into the
    // mask index so no shifted copy is materialised.
    let mut acc = 0.0f64;
    for i in 0..rr {
        let mi = (i + rr / 2) % rr;
        for j in 0..ncol {
            let mj = (j + ncol / 2) % ncol;
            acc += buf[i * ncol + j].norm() as f64 * mask[[mi, mj]] as f64;
        }
    }
    Ok((acc / (rr * ncol) as f64) as f32)
}

/// Cubic B-spline column shift, content moved right by `shift`, zero outside
/// (scipy `ndimage.shift(order=3, mode='constant')`). Each row is prefiltered
/// to B-spline coefficients (Unser/Thévenaz, mirror boundary) then resampled.
/// The boundary columns this routine zero-fills are overwritten by the caller,
/// so the prefilter's boundary convention has no effect on the metric.
fn cubic_shift_cols(img: &Array2<f32>, shift: f64) -> Array2<f32> {
    let (nr, nc) = img.dim();
    let mut out = Array2::<f32>::zeros((nr, nc));
    let mut coeff = vec![0.0f64; nc];
    for i in 0..nr {
        for j in 0..nc {
            coeff[j] = img[[i, j]] as f64;
        }
        prefilter_cubic(&mut coeff);
        for j in 0..nc {
            out[[i, j]] = eval_cubic(&coeff, j as f64 - shift) as f32;
        }
    }
    out
}

/// In-place cubic B-spline prefilter (single pole `√3 − 2`, gain 6),
/// Thévenaz "Interpolation Revisited" recursion with a truncated mirror init.
fn prefilter_cubic(c: &mut [f64]) {
    let n = c.len();
    if n < 2 {
        return;
    }
    let z = 3.0f64.sqrt() - 2.0;
    let lambda = (1.0 - z) * (1.0 - 1.0 / z); // = 6
    for v in c.iter_mut() {
        *v *= lambda;
    }
    // Causal initialisation, horizon truncated at tol 1e-9.
    let horizon = ((1e-9f64.ln() / z.abs().ln()).ceil() as usize).min(n);
    let mut zn = z;
    let mut sum = c[0];
    for v in c.iter().take(horizon).skip(1) {
        sum += zn * v;
        zn *= z;
    }
    c[0] = sum;
    for k in 1..n {
        c[k] += z * c[k - 1];
    }
    // Anticausal initialisation and recursion.
    c[n - 1] = (z / (z * z - 1.0)) * (z * c[n - 2] + c[n - 1]);
    for k in (0..n - 1).rev() {
        c[k] = z * (c[k + 1] - c[k]);
    }
}

/// Evaluate the cubic B-spline with coefficients `c` at position `x`,
/// zero outside `[0, n)` (mode `'constant'`).
fn eval_cubic(c: &[f64], x: f64) -> f64 {
    let n = c.len() as i64;
    let xf = x.floor();
    let base = xf as i64;
    let mut val = 0.0;
    for k in -1..=2 {
        let idx = base + k;
        if idx >= 0 && idx < n {
            val += beta3((x - (xf + k as f64)).abs()) * c[idx as usize];
        }
    }
    val
}

/// Cubic B-spline kernel β₃.
fn beta3(t: f64) -> f64 {
    if t < 1.0 {
        2.0 / 3.0 - t * t + 0.5 * t * t * t
    } else if t < 2.0 {
        let a = 2.0 - t;
        a * a * a / 6.0
    } else {
        0.0
    }
}
