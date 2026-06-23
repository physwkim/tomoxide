//! Stripe-artifact removal (ports tomopy `prep/stripe.py` + tomocupy
//! `processing/remove_stripe.py`). The Fourier-Wavelet (`Fw`), smoothing-filter
//! (`Sf`), Titarenko (`Ti`), Vo all-stripe (`VoAll`), Vo sorting-based
//! (`VoSort`), Vo filtering-based (`VoFilter`), Vo large-stripe (`VoLarge`), Vo
//! dead-stripe (`VoDead`), and Vo fitting-based (`VoFit`) methods are
//! implemented. See `docs/PORTING.md` §D. Dispatch on [`StripeMethod`].

use ndarray::Array2;
use tomoxide_core::data::{Layout, Tomo};
use tomoxide_core::error::Result;
use tomoxide_core::params::StripeMethod;

use crate::{fft, wavelet};

/// Remove stripes from a sinogram stack using the selected method.
pub fn remove_stripe(data: &mut Tomo<f32>, method: StripeMethod) -> Result<()> {
    match method {
        StripeMethod::None => Ok(()),
        StripeMethod::Fw { sigma, level } => remove_stripe_fw(data, sigma, level),
        StripeMethod::Ti { nblock, beta } => remove_stripe_ti(data, beta, nblock),
        StripeMethod::Sf { size } => remove_stripe_sf(data, size),
        StripeMethod::VoAll {
            snr,
            la_size,
            sm_size,
        } => remove_all_stripe(data, snr, la_size, sm_size),
        StripeMethod::VoSort { size, dim } => remove_stripe_based_sorting(data, size, dim),
        StripeMethod::VoFilter { sigma, size, dim } => {
            remove_stripe_based_filtering(data, sigma, size, dim)
        }
        StripeMethod::VoLarge {
            snr,
            size,
            drop_ratio,
            norm,
        } => remove_large_stripe(data, snr, size, drop_ratio, norm),
        StripeMethod::VoDead { snr, size, norm } => remove_dead_stripe(data, snr, size, norm),
        StripeMethod::VoFit { order, sigma } => remove_stripe_based_fitting(data, order, sigma),
    }
}

/// Smoothing-filter stripe removal — a direct port of tomopy
/// `libtomo/prep/stripe.c::remove_stripe_sf`.
///
/// For each reconstruction slice (the `row` axis) the average sinogram row
/// (column-wise mean over projections) is computed, smoothed by a width-`size`
/// moving average with clamp-to-edge boundaries, and the residual
/// `average − smoothed` is subtracted from every projection in that column. All
/// arithmetic is f32 in the upstream summation order, so the result matches
/// tomopy bit-for-bit. Projector-independent.
fn remove_stripe_sf(data: &mut Tomo<f32>, size: usize) -> Result<()> {
    let target = data.layout;
    // tomopy's `remove_stripe_sf` indexes `data[j + s*dz + p*dy*dz]` over
    // `(dx=proj, dy=row, dz=col)` — i.e. the `[angle, row, col]` projection
    // layout.
    let mut proj = data.to_layout(Layout::Projection);
    let (dx, dy, dz) = proj.array.dim();
    if dx == 0 || dy == 0 || dz == 0 || size == 0 {
        return Ok(());
    }
    let arr = &mut proj.array;
    let half = (size / 2) as isize; // C: `size / 2`, integer division
    let last = dz as isize - 1;
    let dxf = dx as f32;
    let sizef = size as f32;
    let mut average_row = vec![0.0f32; dz];
    let mut smooth_row = vec![0.0f32; dz];

    for s in 0..dy {
        // Average row: column-wise mean over projections (each term divided by
        // `dx` before summing, exactly as the C does, to match rounding).
        for j in 0..dz {
            let mut acc = 0.0f32;
            for p in 0..dx {
                acc += arr[[p, s, j]] / dxf;
            }
            average_row[j] = acc;
        }
        // Smooth the average row with a width-`size` moving average, clamping
        // out-of-range taps to the nearest edge.
        for (i, sv) in smooth_row.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for jj in 0..size {
                let mut k = i as isize + jj as isize - half;
                if k < 0 {
                    k = 0;
                }
                if k > last {
                    k = last;
                }
                acc += average_row[k as usize];
            }
            *sv = acc / sizef;
        }
        // Subtract the column residual from every projection in this slice.
        for p in 0..dx {
            for j in 0..dz {
                arr[[p, s, j]] -= average_row[j] - smooth_row[j];
            }
        }
    }

    *data = proj.to_layout(target);
    Ok(())
}

// ----------------------------------------------------------------------------
// Fourier-Wavelet stripe removal (tomopy `prep/stripe.py::_remove_stripe_fw`,
// Münch 2009). Per sinogram slice: pad the projection axis, run a `level`-deep
// `db5` 2-D wavelet decomposition, damp the vertical-detail bands in Fourier
// space along the projection direction, reconstruct, and crop back. The forward
// decomposition mirrors tomopy's float32 `pywt` path (each band rounded to f32),
// while damping and reconstruction run in f64 exactly as tomopy's numpy/pywt
// promotion does, so the result matches to the f32 round-off floor.
// Projector-independent. The wavelet kernels live in `crate::wavelet`; the
// arbitrary-length column FFT for the damping lives in `crate::fft`.
// ----------------------------------------------------------------------------

/// Round every element of an `f64` band to `f32` precision (emulating tomopy's
/// float32 `pywt` forward pass), keeping the `f64` storage type.
fn round_to_f32(a: &Array2<f64>) -> Array2<f64> {
    a.mapv(|v| v as f32 as f64)
}

/// Münch damping factor `D = ifftshift(damp)` for a band with `my` rows, where
/// `damp[k] = 1 − exp(−ŷ²/(2σ²))` and `ŷ = (arange(−my, my, 2) + 1)/2`. Computed
/// in f32 like tomopy (`y_hat`/`damp` are float32 there), returned as f64.
fn damp_vector(my: usize, sigma: f32) -> Vec<f64> {
    let two_sig2 = 2.0f32 * sigma * sigma;
    // damp in natural (fftshift) order.
    let damp: Vec<f32> = (0..my)
        .map(|k| {
            // arange(-my, my, 2)[k] == -my + 2k.
            let y_hat = ((-(my as i64) + 2 * k as i64) as f32 + 1.0) / 2.0;
            1.0f32 - (-(y_hat * y_hat) / two_sig2).exp()
        })
        .collect();
    // ifftshift: D[i] = damp[(i + my/2) mod my]  (np.roll(damp, -(my//2))).
    let half = my / 2;
    (0..my).map(|i| damp[(i + half) % my] as f64).collect()
}

/// Damp the vertical-detail band `cv` (shape `[my, mx]`) along axis 0, in place
/// on a fresh array. Matches tomopy `fcV = fftshift(fft(cV, axis=0)); fcV *=
/// damp; cV = real(ifft(ifftshift(fcV), axis=0))`.
///
/// The `fftshift`/`ifftshift` pair cancels around the elementwise `damp`
/// multiply (a permutation distributes over an elementwise product), so each
/// column reduces to `real(ifft(fft(col) · D))` with `D = ifftshift(damp)` —
/// computed by the `O(n log n)` arbitrary-length FFT in `crate::fft`.
fn damp_vertical(cv: &Array2<f64>, sigma: f32) -> Array2<f64> {
    let (my, mx) = cv.dim();
    let d = damp_vector(my, sigma);
    let mut out = Array2::<f64>::zeros((my, mx));
    let mut col = vec![0.0f64; my];
    for c in 0..mx {
        for (r, v) in col.iter_mut().enumerate() {
            *v = cv[[r, c]];
        }
        let damped = fft::filter_real_column(&col, &d);
        for r in 0..my {
            out[[r, c]] = damped[r];
        }
    }
    out
}

/// Crop `a` to its top-left `[rows, cols]` sub-block (tomopy `sli[0:r, 0:c]`).
fn crop(a: &Array2<f64>, rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(r, c)| a[[r, c]])
}

/// Process one `[nproj, ncol]` sinogram through the Fourier-Wavelet filter.
fn fw_slice(sino: &Array2<f64>, sigma: f32, level: usize, nx: usize, xshift: usize) -> Array2<f64> {
    let (nproj, ncol) = sino.dim();
    // Pad the projection axis to `nx`, sinogram placed at rows [xshift, xshift+nproj).
    let mut approx = Array2::<f64>::zeros((nx, ncol));
    for p in 0..nproj {
        for c in 0..ncol {
            approx[[xshift + p, c]] = sino[[p, c]];
        }
    }
    // Forward: `level`-deep db5 decomposition, each band rounded to f32.
    let mut chs = Vec::with_capacity(level);
    let mut cvs = Vec::with_capacity(level);
    let mut cds = Vec::with_capacity(level);
    for _ in 0..level {
        let (ca, ch, cv, cd) = wavelet::dwt2(&approx);
        approx = round_to_f32(&ca);
        chs.push(round_to_f32(&ch));
        cvs.push(round_to_f32(&cv));
        cds.push(round_to_f32(&cd));
    }
    // Damp every vertical-detail band (f64, exactly as tomopy after numpy promotion).
    for cv in cvs.iter_mut() {
        *cv = damp_vertical(cv, sigma);
    }
    // Reconstruct: crop the running approximation to each level's band shape,
    // then inverse-transform with the (damped) details.
    let mut sli = approx;
    for n in (0..level).rev() {
        let (hr, hc) = chs[n].dim();
        let cropped = crop(&sli, hr, hc);
        sli = wavelet::idwt2(&cropped, &chs[n], &cvs[n], &cds[n]);
    }
    // Crop back to the original sinogram region.
    Array2::from_shape_fn((nproj, ncol), |(p, c)| sli[[xshift + p, c]])
}

/// Fourier-Wavelet stripe removal (`remove_stripe_fw`). `level = None` selects
/// `ceil(log2(max(nproj, nrows, ncol)))`, matching tomopy. `pad` is always on
/// (tomopy's default): the projection axis is padded to `nproj + nproj/8`.
fn remove_stripe_fw(data: &mut Tomo<f32>, sigma: f32, level: Option<usize>) -> Result<()> {
    let target = data.layout;
    // tomopy operates on `tomo[:, m, :]` = `[proj, col]` slices of the
    // `[proj, row, col]` projection-layout stack.
    let mut proj = data.to_layout(Layout::Projection);
    let (nproj, nrows, ncol) = proj.array.dim();
    if nproj == 0 || nrows == 0 || ncol == 0 {
        return Ok(());
    }
    let level = level.unwrap_or_else(|| {
        let size = nproj.max(nrows).max(ncol);
        (size as f64).log2().ceil() as usize
    });
    if level == 0 {
        return Ok(());
    }
    let nx = nproj + nproj / 8; // pad=True
    let xshift = (nx - nproj) / 2;

    for m in 0..nrows {
        let mut sino = Array2::<f64>::zeros((nproj, ncol));
        for p in 0..nproj {
            for c in 0..ncol {
                sino[[p, c]] = proj.array[[p, m, c]] as f64;
            }
        }
        let out = fw_slice(&sino, sigma, level, nx, xshift);
        for p in 0..nproj {
            for c in 0..ncol {
                proj.array[[p, m, c]] = out[[p, c]] as f32;
            }
        }
    }

    *data = proj.to_layout(target);
    Ok(())
}

// ----------------------------------------------------------------------------
// Titarenko stripe removal (tomopy `prep/stripe.py::remove_stripe_ti`, Miqueles
// 2014). Per sinogram slice the per-detector-column background offset is the CG
// solution of a finite-difference normal-equations system; the corrected first-
// and second-difference sinograms are combined as `sqrt(d1·d2 + β·|min|)`. The
// CG runs in f64 and each `_ring` rounds to f32, so it matches tomopy to the
// f32 round-off floor (projector-independent). All reductions and convolutions
// follow the upstream operation order.
// ----------------------------------------------------------------------------

/// Finite-difference kernels (`_kernel(m, n)` — the `[m-1][n-1]` table entry).
/// Only the `(1, 1)` and `(2, 1)` rows are reachable from `remove_stripe_ti`.
fn ti_kernel(m: usize, n: usize) -> Vec<f64> {
    let table: [&[&[f64]]; 3] = [
        &[
            &[1.0, -1.0],
            &[-3.0 / 2.0, 2.0, -1.0 / 2.0],
            &[-11.0 / 6.0, 3.0, -3.0 / 2.0, 1.0 / 3.0],
        ],
        &[&[-1.0, 2.0, -1.0], &[2.0, -5.0, 4.0, -1.0]],
        &[&[-1.0, 3.0, -3.0, 1.0]],
    ];
    table[m - 1][n - 1].to_vec()
}

/// Full discrete convolution (`np.convolve(a, b)`, `mode='full'`), length
/// `a.len() + b.len() - 1`.
fn convolve_full(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0f64; a.len() + b.len() - 1];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    out
}

/// `_ringMatXvec(h, x)`: `y = conv(conv(x, flip(h))[|h|-1 .. |x|], h)`, the
/// symmetric finite-difference operator `Hᵀ H x`. Returns a vector of `x.len()`.
fn ti_matxvec(h: &[f64], x: &[f64]) -> Vec<f64> {
    let hf: Vec<f64> = h.iter().rev().copied().collect();
    let s = convolve_full(x, &hf);
    // s[|h|-1 : |x|] — Python slice (end-exclusive).
    let u = &s[h.len() - 1..x.len()];
    convolve_full(u, h)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `_ringCGM(h, alpha, f)`: conjugate-gradient solve of `(HᵀH + α I) x = f`.
fn ti_cgm(h: &[f64], alpha: f64, f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let x0 = vec![0.0f64; n];
    // r = f - (matxvec(h, x0) + alpha*x0) = f  (x0 == 0).
    let mut r = f.to_vec();
    let mut w: Vec<f64> = r.iter().map(|v| -v).collect();
    let apply = |w: &[f64]| -> Vec<f64> {
        let mv = ti_matxvec(h, w);
        mv.iter()
            .zip(w)
            .map(|(m, wv)| m + alpha * wv)
            .collect::<Vec<f64>>()
    };
    let mut z = apply(&w);
    let mut a = dot(&r, &w) / dot(&w, &z);
    let mut x: Vec<f64> = x0.iter().zip(&w).map(|(x0v, wv)| x0v + a * wv).collect();
    for _ in 0..1_000_000 {
        for (rv, zv) in r.iter_mut().zip(&z) {
            *rv -= a * zv;
        }
        let norm = dot(&r, &r).sqrt();
        if norm < 0.0000001 {
            break;
        }
        let bb = dot(&r, &z) / dot(&w, &z);
        for (wv, rv) in w.iter_mut().zip(&r) {
            *wv = -rv + bb * *wv;
        }
        z = apply(&w);
        a = dot(&r, &w) / dot(&w, &z);
        for (xv, wv) in x.iter_mut().zip(&w) {
            *xv += a * wv;
        }
    }
    x
}

/// `_get_parameter(x)`: `1 / (2·(max - min))` of the column sums (`x.sum(0)`),
/// where `x` is the transposed sinogram laid out as `mysino[col][angle]`.
fn ti_parameter(mysino: &[Vec<f64>], r: usize, n: usize) -> f64 {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for j in 0..n {
        let mut s = 0.0f64;
        for row in mysino.iter().take(r) {
            s += row[j];
        }
        if s < min {
            min = s;
        }
        if s > max {
            max = s;
        }
    }
    1.0 / (2.0 * (max - min))
}

/// `_ring(sino, m, n)`: solve for the per-column offset `q` and add it to every
/// angle, rounding to f32. `sino` is `[nproj, ncol]`; works on the transpose
/// `mysino[col][angle]` like tomopy.
fn ti_ring(sino: &Array2<f32>, m: usize, n: usize) -> Array2<f32> {
    let (nproj, ncol) = sino.dim();
    let r = ncol; // mysino rows  (R = transposed axis 0)
    let nn = nproj; // mysino cols  (N)
                    // mysino[col][angle] with NaN → 0.
    let mut mysino = vec![vec![0.0f64; nn]; r];
    for (col, row) in mysino.iter_mut().enumerate() {
        for (angle, cell) in row.iter_mut().enumerate() {
            let v = sino[[angle, col]] as f64;
            *cell = if v.is_nan() { 0.0 } else { v };
        }
    }
    let alpha = ti_parameter(&mysino, r, nn);
    // pp = mysino.mean(1) — per-column mean over angles, length R.
    let pp: Vec<f64> = mysino
        .iter()
        .map(|row| row.iter().sum::<f64>() / nn as f64)
        .collect();
    let h = ti_kernel(m, n);
    let f: Vec<f64> = ti_matxvec(&h, &pp).iter().map(|v| -v).collect();
    let q = ti_cgm(&h, alpha, &f);
    // new[col][angle] = mysino + q[col]; transpose back to [nproj, ncol] as f32.
    let mut out = Array2::<f32>::zeros((nproj, ncol));
    for col in 0..ncol {
        for angle in 0..nproj {
            out[[angle, col]] = (mysino[col][angle] + q[col]) as f32;
        }
    }
    out
}

/// `_ringb(sino, m, n, step)`: block-wise variant of [`ti_ring`]. The transposed
/// sinogram `mysino[col][angle]` is split into `floor(nproj/step)` blocks of
/// `step` angles; each block gets its own regularization parameter and
/// per-column offset `q`. Angles past the last full block keep tomopy's
/// `np.ones` fill (`1.0`). Faithful to tomopy `_ringb`, including its no-op NaN
/// guard (`np.where(np.isnan(x) is True)` never matches, so NaNs are *not*
/// zeroed here, unlike `_ring`).
fn ti_ringb(sino: &Array2<f32>, m: usize, n: usize, step: usize) -> Array2<f32> {
    let (nproj, ncol) = sino.dim();
    let r = ncol; // mysino rows (R)
    let nn = nproj; // mysino cols (N = angles)
    let mut mysino = vec![vec![0.0f64; nn]; r];
    for (col, row) in mysino.iter_mut().enumerate() {
        for (angle, cell) in row.iter_mut().enumerate() {
            *cell = sino[[angle, col]] as f64;
        }
    }
    let h = ti_kernel(m, n);
    let nblock = if step == 0 { 0 } else { nn / step };
    // new[col][angle], initialized to ones (tomopy `np.ones((R, N))`).
    let mut newm = vec![vec![1.0f64; nn]; r];
    for k in 0..nblock {
        let j0 = k * step;
        let j1 = j0 + step;
        // alpha = 1/(2·(max−min)) of the block's column sums (sum over R rows).
        let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
        for j in j0..j1 {
            let mut s = 0.0f64;
            for row in mysino.iter().take(r) {
                s += row[j];
            }
            if s < min {
                min = s;
            }
            if s > max {
                max = s;
            }
        }
        let alpha = 1.0 / (2.0 * (max - min));
        // pp = block.mean(1): per-row mean over the `step` block angles.
        let pp: Vec<f64> = mysino
            .iter()
            .map(|row| row[j0..j1].iter().sum::<f64>() / step as f64)
            .collect();
        let f: Vec<f64> = ti_matxvec(&h, &pp).iter().map(|v| -v).collect();
        let q = ti_cgm(&h, alpha, &f);
        for (col, row) in mysino.iter().enumerate() {
            for j in j0..j1 {
                newm[col][j] = row[j] + q[col];
            }
        }
    }
    let mut out = Array2::<f32>::zeros((nproj, ncol));
    for col in 0..ncol {
        for angle in 0..nproj {
            out[[angle, col]] = newm[col][angle] as f32;
        }
    }
    out
}

/// Titarenko stripe removal (`remove_stripe_ti`, default `nblock = 0`): combine
/// the first- and second-difference corrected sinograms as
/// `sqrt(d1·d2 + β·|min(d1·d2)|)`.
fn remove_stripe_ti(data: &mut Tomo<f32>, beta: f32, nblock: usize) -> Result<()> {
    let target = data.layout;
    let mut proj = data.to_layout(Layout::Projection);
    let (nproj, nrows, ncol) = proj.array.dim();
    if nproj == 0 || nrows == 0 || ncol == 0 {
        return Ok(());
    }
    // tomopy `_remove_stripe_ti`: nblock==0 → `_ring`, else `_ringb` with
    // block size `int(nproj / nblock)` (the transposed N axis).
    let step = if nblock == 0 { 0 } else { nproj / nblock };
    for m in 0..nrows {
        let mut sino = Array2::<f32>::zeros((nproj, ncol));
        for p in 0..nproj {
            for c in 0..ncol {
                sino[[p, c]] = proj.array[[p, m, c]];
            }
        }
        let (d1, d2) = if nblock == 0 {
            (ti_ring(&sino, 1, 1), ti_ring(&sino, 2, 1))
        } else {
            (ti_ringb(&sino, 1, 1, step), ti_ringb(&sino, 2, 1, step))
        };
        // p = d1 * d2 (f32); d = sqrt(p + β·|min(p)|) (f32).
        let mut p = Array2::<f32>::zeros((nproj, ncol));
        let mut pmin = f32::INFINITY;
        for r in 0..nproj {
            for c in 0..ncol {
                let v = d1[[r, c]] * d2[[r, c]];
                p[[r, c]] = v;
                if v < pmin {
                    pmin = v;
                }
            }
        }
        let shift = beta * pmin.abs();
        for c in 0..ncol {
            for r in 0..nproj {
                proj.array[[r, m, c]] = (p[[r, c]] + shift).sqrt();
            }
        }
    }
    *data = proj.to_layout(target);
    Ok(())
}

// ----------------------------------------------------------------------------
// Vo all-stripe removal (tomopy `prep/stripe.py::remove_all_stripe`, Vo
// algorithms 3+5+6). Per sinogram slice: `_rs_dead` (dead/large stripe removal
// via detection + bilinear column fill + `_rs_large`) followed by `_rs_sort`
// (sorting-based small-stripe removal). Projector-independent, but it composes
// several scipy primitives (uniform_filter1d, median_filter, binary_dilation,
// polyfit, RectBivariateSpline) whose summation/fit numerics differ slightly
// from this reimplementation, so it is held to a tolerance, not bit-exactness.
// ----------------------------------------------------------------------------

/// scipy half-sample `reflect` index mapping into `[0, n)`.
fn reflect_index(i: isize, n: isize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * n;
    let mut j = i % period;
    if j < 0 {
        j += period;
    }
    if j >= n {
        j = period - 1 - j;
    }
    j as usize
}

/// `uniform_filter1d` along axis 0 (the projection axis), `mode='reflect'`.
/// The window for output `i` is `[i - size/2, i - size/2 + size - 1]`.
fn uniform_filter1d_axis0(sino: &Array2<f32>, size: usize) -> Array2<f32> {
    let (nrow, ncol) = sino.dim();
    let half = (size / 2) as isize;
    let n = nrow as isize;
    let inv = 1.0f64 / size as f64;
    let mut out = Array2::<f32>::zeros((nrow, ncol));
    for c in 0..ncol {
        for i in 0..nrow {
            let mut sum = 0.0f64;
            for k in 0..size {
                let r = reflect_index(i as isize - half + k as isize, n);
                sum += sino[[r, c]] as f64;
            }
            out[[i, c]] = (sum * inv) as f32;
        }
    }
    out
}

/// `median_filter` over a 1-D list, `mode='reflect'`, window
/// `[i - size/2, i - size/2 + size - 1]`, value at sorted index `size/2`.
fn median_filter_1d(list: &[f32], size: usize) -> Vec<f32> {
    let n = list.len();
    let half = (size / 2) as isize;
    let mid = size / 2;
    let ni = n as isize;
    let mut out = vec![0.0f32; n];
    let mut win = vec![0.0f32; size];
    for i in 0..n {
        for (k, w) in win.iter_mut().enumerate() {
            *w = list[reflect_index(i as isize - half + k as isize, ni)];
        }
        win.sort_by(|a, b| a.total_cmp(b));
        out[i] = win[mid];
    }
    out
}

/// `median_filter` with footprint `(size, 1)`: median along axis 0 for each column.
fn median_filter_axis0(arr: &Array2<f32>, size: usize) -> Array2<f32> {
    let (nrow, ncol) = arr.dim();
    let mut out = Array2::<f32>::zeros((nrow, ncol));
    let mut col = vec![0.0f32; nrow];
    for c in 0..ncol {
        for (r, v) in col.iter_mut().enumerate() {
            *v = arr[[r, c]];
        }
        let med = median_filter_1d(&col, size);
        for r in 0..nrow {
            out[[r, c]] = med[r];
        }
    }
    out
}

/// `median_filter` with footprint `(1, size)`: median along axis 1 for each row.
fn median_filter_axis1(arr: &Array2<f32>, size: usize) -> Array2<f32> {
    let (nrow, ncol) = arr.dim();
    let mut out = Array2::<f32>::zeros((nrow, ncol));
    let mut row = vec![0.0f32; ncol];
    for r in 0..nrow {
        for (c, v) in row.iter_mut().enumerate() {
            *v = arr[[r, c]];
        }
        let med = median_filter_1d(&row, size);
        for c in 0..ncol {
            out[[r, c]] = med[c];
        }
    }
    out
}

/// `median_filter` with a square footprint `(size, size)`, scipy `reflect`
/// boundary, rank `size²/2` (the `dim=2` branch of tomopy `_rs_sort`).
fn median_filter_2d(arr: &Array2<f32>, size: usize) -> Array2<f32> {
    let (nrow, ncol) = arr.dim();
    let half = (size / 2) as isize;
    let (nr, nc) = (nrow as isize, ncol as isize);
    let mid = (size * size) / 2;
    let mut out = Array2::<f32>::zeros((nrow, ncol));
    let mut win = vec![0.0f32; size * size];
    for i in 0..nrow {
        for j in 0..ncol {
            let mut t = 0;
            for di in 0..size {
                let ri = reflect_index(i as isize - half + di as isize, nr);
                for dj in 0..size {
                    let cj = reflect_index(j as isize - half + dj as isize, nc);
                    win[t] = arr[[ri, cj]];
                    t += 1;
                }
            }
            win.sort_by(|a, b| a.total_cmp(b));
            out[[i, j]] = win[mid];
        }
    }
    out
}

/// `binary_dilation` with one iteration of the default 3-element structuring
/// element (border value 0).
fn binary_dilation_1d(mask: &[f32]) -> Vec<f32> {
    let n = mask.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let l = i > 0 && mask[i - 1] > 0.0;
        let m = mask[i] > 0.0;
        let r = i + 1 < n && mask[i + 1] > 0.0;
        if l || m || r {
            out[i] = 1.0;
        }
    }
    out
}

/// Degree-1 least-squares fit `y ≈ slope·x + intercept` (closed form, the
/// `np.polyfit(x, y, 1)` path in `_detect_stripe`).
fn polyfit1(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
    let denom = n * sxx - sx * sx;
    let slope = if denom != 0.0 {
        (n * sxy - sx * sy) / denom
    } else {
        0.0
    };
    let intercept = (sy - slope * sx) / n;
    (slope, intercept)
}

/// `_detect_stripe` (Vo algorithm 4): mark columns whose `listdata` value sits
/// far enough above/below a robust linear baseline of the sorted profile.
fn detect_stripe(listdata: &[f32], snr: f32) -> Vec<f32> {
    let numdata = listdata.len();
    let mut listmask = vec![0.0f32; numdata];
    if numdata == 0 {
        return listmask;
    }
    // Descending sort.
    let mut listsorted: Vec<f32> = listdata.to_vec();
    listsorted.sort_by(|a, b| b.total_cmp(a));
    let ndrop = (0.25 * numdata as f64) as i16 as usize; // np.int16(0.25*numdata)
                                                         // Fit over xlist[ndrop : numdata-ndrop-1] (Python's `[ndrop:-ndrop-1]`).
    if numdata < 2 * ndrop + 2 {
        return listmask; // not enough interior points to fit
    }
    let lo = ndrop;
    let hi = numdata - ndrop - 1; // exclusive
    let xs: Vec<f64> = (lo..hi).map(|v| v as f64).collect();
    let ys: Vec<f64> = (lo..hi).map(|v| listsorted[v] as f64).collect();
    let (slope, intercept) = polyfit1(&xs, &ys);

    let numt1 = intercept + slope * (numdata - 1) as f64;
    let mut noiselevel = (numt1 - intercept).abs();
    if noiselevel < 1e-6 {
        noiselevel = 1e-6;
    }
    let snr = snr as f64;
    let val1 = (listsorted[0] as f64 - intercept).abs() / noiselevel;
    let val2 = (listsorted[numdata - 1] as f64 - numt1).abs() / noiselevel;
    if val1 >= snr {
        let upper = intercept + noiselevel * snr * 0.5;
        for (j, &v) in listdata.iter().enumerate() {
            if v as f64 > upper {
                listmask[j] = 1.0;
            }
        }
    }
    if val2 >= snr {
        let lower = numt1 - noiselevel * snr * 0.5;
        for (j, &v) in listdata.iter().enumerate() {
            if v as f64 <= lower {
                listmask[j] = 1.0;
            }
        }
    }
    listmask
}

/// Sort each column of `sino` ascending; return `(sorted_values[rank, col],
/// original_row[col][rank])`.
fn argsort_columns(sino: &Array2<f32>) -> (Array2<f32>, Vec<Vec<usize>>) {
    let (nrow, ncol) = sino.dim();
    let mut sorted = Array2::<f32>::zeros((nrow, ncol));
    let mut perm = vec![vec![0usize; nrow]; ncol];
    for c in 0..ncol {
        let mut idx: Vec<usize> = (0..nrow).collect();
        idx.sort_by(|&a, &b| sino[[a, c]].total_cmp(&sino[[b, c]]));
        for (rank, &row) in idx.iter().enumerate() {
            sorted[[rank, c]] = sino[[row, c]];
            perm[c][rank] = row;
        }
    }
    (sorted, perm)
}

/// `_rs_large` (Vo algorithm 5): replace detected large-stripe columns with the
/// rank-smoothed profile, optionally normalising by the per-column intensity
/// factor first.
fn rs_large(sino: &Array2<f32>, snr: f32, size: usize, drop_ratio: f32, norm: bool) -> Array2<f32> {
    let (nrow, ncol) = sino.dim();
    let dr = drop_ratio.clamp(0.0, 0.8) as f64;
    let ndrop = (0.5 * dr * nrow as f64) as usize;

    // sinosort = sort each column; sinosmooth = per-row median along columns.
    let (sinosort, _) = argsort_columns(sino);
    let sinosmooth = median_filter_axis1(&sinosort, size);

    // Per-column means of the central rows.
    let cnt = (nrow.saturating_sub(2 * ndrop)).max(1) as f64;
    let mut listfact = vec![1.0f64; ncol];
    for (c, lf) in listfact.iter_mut().enumerate() {
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for r in ndrop..(nrow - ndrop) {
            s1 += sinosort[[r, c]] as f64;
            s2 += sinosmooth[[r, c]] as f64;
        }
        let (m1, m2) = (s1 / cnt, s2 / cnt);
        *lf = if m2 != 0.0 { m1 / m2 } else { 1.0 };
    }

    let listfact_f32: Vec<f32> = listfact.iter().map(|&v| v as f32).collect();
    let listmask = binary_dilation_1d(&detect_stripe(&listfact_f32, snr));

    // Normalised working copy (each column scaled by 1/listfact).
    let mut work = sino.clone();
    if norm {
        for c in 0..ncol {
            let f = listfact[c];
            for r in 0..nrow {
                work[[r, c]] = (work[[r, c]] as f64 / f) as f32;
            }
        }
    }

    // Map the rank-smoothed sorted profile back through the (normalised) sort
    // order, and overwrite only the masked columns.
    let (_, perm) = argsort_columns(&work);
    let mut out = work;
    for c in 0..ncol {
        if listmask[c] > 0.0 {
            for rank in 0..nrow {
                out[[perm[c][rank], c]] = sinosmooth[[rank, c]];
            }
        }
    }
    out
}

/// Linear-interpolation bracket of `xmiss` within the ascending good-column
/// list `goodx`: returns `(lower_index, t)` for `goodx[i] ≤ xmiss ≤ goodx[i+1]`.
fn bracket(goodx: &[usize], xmiss: usize) -> (usize, f64) {
    let mut i0 = 0;
    for k in 0..goodx.len().saturating_sub(1) {
        if goodx[k] <= xmiss && xmiss <= goodx[k + 1] {
            i0 = k;
            break;
        }
    }
    let (g0, g1) = (goodx[i0] as f64, goodx[i0 + 1] as f64);
    let t = if g1 != g0 {
        (xmiss as f64 - g0) / (g1 - g0)
    } else {
        0.0
    };
    (i0, t)
}

/// `_rs_dead` (Vo algorithm 6): detect unresponsive/fluctuating columns, fill
/// them by per-row linear interpolation across good columns (the `kx=ky=1`
/// `RectBivariateSpline`), then — only when `norm` is set — pass through
/// `_rs_large` for residual stripes (tomopy gates the residual pass on `norm`).
fn rs_dead(sino: &Array2<f32>, snr: f32, size: usize, norm: bool) -> Array2<f32> {
    let (nrow, ncol) = sino.dim();
    let sinosmooth = uniform_filter1d_axis0(sino, 10);
    let mut listdiff = vec![0.0f32; ncol];
    for (c, ld) in listdiff.iter_mut().enumerate() {
        let mut s = 0.0f64;
        for r in 0..nrow {
            s += (sino[[r, c]] - sinosmooth[[r, c]]).abs() as f64;
        }
        *ld = s as f32;
    }
    let listdiffbck = median_filter_1d(&listdiff, size);
    let listfact: Vec<f32> = (0..ncol)
        .map(|c| {
            if listdiffbck[c] != 0.0 {
                listdiff[c] / listdiffbck[c]
            } else {
                1.0
            }
        })
        .collect();
    let mut listmask = binary_dilation_1d(&detect_stripe(&listfact, snr));
    // Never treat the two border columns on each side as dead.
    listmask[..ncol.min(2)].fill(0.0);
    listmask[ncol.saturating_sub(2)..].fill(0.0);

    let goodx: Vec<usize> = (0..ncol).filter(|&c| listmask[c] < 1.0).collect();
    let badx: Vec<usize> = (0..ncol).filter(|&c| listmask[c] > 0.0).collect();

    let mut work = sino.clone();
    if !badx.is_empty() && goodx.len() >= 2 {
        for &xmiss in &badx {
            let (i0, t) = bracket(&goodx, xmiss);
            let (c0, c1) = (goodx[i0], goodx[i0 + 1]);
            for r in 0..nrow {
                let v0 = sino[[r, c0]] as f64;
                let v1 = sino[[r, c1]] as f64;
                work[[r, xmiss]] = ((1.0 - t) * v0 + t * v1) as f32;
            }
        }
    }

    // Residual large-stripe pass — tomopy runs it only when `norm is True`; the
    // inner `_rs_large` always uses its own defaults (drop_ratio=0.1, norm=True).
    if norm {
        rs_large(&work, snr, size, 0.1, true)
    } else {
        work
    }
}

/// `_rs_sort` (Vo algorithm 3, `dim = 1`): sort each column, median-smooth the
/// sorted profiles across columns, then unsort.
fn rs_sort_with(sino: &Array2<f32>, smooth: impl Fn(&Array2<f32>) -> Array2<f32>) -> Array2<f32> {
    let (nrow, ncol) = sino.dim();
    // Build the sorted-value matrix laid out as [ncol, nrow] (the transpose the
    // tomopy code filters over) plus the per-column permutation.
    let mut sortedv = Array2::<f32>::zeros((ncol, nrow));
    let mut perm = vec![vec![0usize; nrow]; ncol];
    for c in 0..ncol {
        let mut idx: Vec<usize> = (0..nrow).collect();
        idx.sort_by(|&a, &b| sino[[a, c]].total_cmp(&sino[[b, c]]));
        for (rank, &row) in idx.iter().enumerate() {
            sortedv[[c, rank]] = sino[[row, c]];
            perm[c][rank] = row;
        }
    }
    // Smooth the sorted [ncol, nrow] matrix (footprint differs by `dim`), then
    // scatter back to the original projection order.
    let smoothed = smooth(&sortedv);

    let mut out = Array2::<f32>::zeros((nrow, ncol));
    for c in 0..ncol {
        for rank in 0..nrow {
            out[[perm[c][rank], c]] = smoothed[[c, rank]];
        }
    }
    out
}

/// `_rs_sort` with `dim=1` (median footprint `(size, 1)`) — the `VoAll` path.
fn rs_sort(sino: &Array2<f32>, size: usize) -> Array2<f32> {
    rs_sort_with(sino, |s| median_filter_axis0(s, size))
}

/// Vo all-stripe removal (`remove_all_stripe`): `_rs_dead` then `_rs_sort` per
/// sinogram slice.
fn remove_all_stripe(data: &mut Tomo<f32>, snr: f32, la_size: usize, sm_size: usize) -> Result<()> {
    let target = data.layout;
    // tomopy operates on `tomo[:, m, :]` = `[proj, col]` slices of the
    // `[proj, row, col]` projection-layout stack.
    let mut proj = data.to_layout(Layout::Projection);
    let (nproj, nrows, ncol) = proj.array.dim();
    if nproj < 2 || nrows == 0 || ncol < 4 || la_size == 0 || sm_size == 0 {
        return Ok(());
    }

    for m in 0..nrows {
        let mut sino = Array2::<f32>::zeros((nproj, ncol));
        for p in 0..nproj {
            for c in 0..ncol {
                sino[[p, c]] = proj.array[[p, m, c]];
            }
        }
        // VoAll always runs the residual large-stripe pass (tomopy default norm=True).
        let sino = rs_dead(&sino, snr, la_size, true);
        let sino = rs_sort(&sino, sm_size);
        for p in 0..nproj {
            for c in 0..ncol {
                proj.array[[p, m, c]] = sino[[p, c]];
            }
        }
    }

    *data = proj.to_layout(target);
    Ok(())
}

/// Vo large-stripe removal (tomopy `remove_large_stripe`, Vo 2018 algorithm 5).
///
/// For each sinogram slice apply `_rs_large`: sort each detector column over
/// projections, median-smooth the sorted profile, estimate a per-column
/// intensity factor from the central rows (`drop_ratio` of the extremes
/// dropped), detect the wide-stripe columns (`_detect_stripe` + 1-px binary
/// dilation), and overwrite *only* those columns with the rank-smoothed profile
/// mapped back through the (optionally intensity-normalised) sort order. The
/// smoothed values are pure rank-filter selections of existing f32 samples and
/// the per-column factor is computed in the upstream op order, so this matches
/// tomopy to the f32 round-off floor on tie-free columns. Shares the `rs_large`
/// helper with `VoAll`.
fn remove_large_stripe(
    data: &mut Tomo<f32>,
    snr: f32,
    size: usize,
    drop_ratio: f32,
    norm: bool,
) -> Result<()> {
    let target = data.layout;
    // tomopy operates on `tomo[:, m, :]` = `[proj, col]` slices of the
    // `[proj, row, col]` projection-layout stack.
    let mut proj = data.to_layout(Layout::Projection);
    let (nproj, nrows, ncol) = proj.array.dim();
    if nproj < 2 || nrows == 0 || ncol < 4 || size == 0 {
        return Ok(());
    }

    for m in 0..nrows {
        let mut sino = Array2::<f32>::zeros((nproj, ncol));
        for p in 0..nproj {
            for c in 0..ncol {
                sino[[p, c]] = proj.array[[p, m, c]];
            }
        }
        let sino = rs_large(&sino, snr, size, drop_ratio, norm);
        for p in 0..nproj {
            for c in 0..ncol {
                proj.array[[p, m, c]] = sino[[p, c]];
            }
        }
    }

    *data = proj.to_layout(target);
    Ok(())
}

/// Vo dead-stripe removal (tomopy `remove_dead_stripe`, Vo 2018 algorithm 6).
///
/// For each sinogram slice apply `_rs_dead`: smooth each detector column over
/// projections (`uniform_filter1d` width 10), score each column by its summed
/// deviation from that smooth, detect the unresponsive/fluctuating columns
/// (`_detect_stripe` + 1-px dilation, the two border columns never flagged), and
/// fill the flagged columns by per-row linear interpolation across the good
/// columns (the `kx=ky=1` `RectBivariateSpline`). When `norm` is set a residual
/// `_rs_large` pass then removes wide stripes. The bilinear fill (and the
/// `norm`-gated residual factor division) are arithmetic, so this is held to the
/// f32 round-off floor. Shares the `rs_dead`/`rs_large` helpers with `VoAll`.
fn remove_dead_stripe(data: &mut Tomo<f32>, snr: f32, size: usize, norm: bool) -> Result<()> {
    let target = data.layout;
    // tomopy operates on `tomo[:, m, :]` = `[proj, col]` slices of the
    // `[proj, row, col]` projection-layout stack.
    let mut proj = data.to_layout(Layout::Projection);
    let (nproj, nrows, ncol) = proj.array.dim();
    if nproj < 2 || nrows == 0 || ncol < 4 || size == 0 {
        return Ok(());
    }

    for m in 0..nrows {
        let mut sino = Array2::<f32>::zeros((nproj, ncol));
        for p in 0..nproj {
            for c in 0..ncol {
                sino[[p, c]] = proj.array[[p, m, c]];
            }
        }
        let sino = rs_dead(&sino, snr, size, norm);
        for p in 0..nproj {
            for c in 0..ncol {
                proj.array[[p, m, c]] = sino[[p, c]];
            }
        }
    }

    *data = proj.to_layout(target);
    Ok(())
}

/// Vo sorting-based stripe removal (tomopy `remove_stripe_based_sorting`,
/// Vo 2018 algorithm 3) — good for partial stripes.
///
/// For each sinogram slice apply `_rs_sort`: sort each detector column's values
/// over projections, median-smooth the sorted matrix, then unsort. The median is
/// a pure rank-filter selection of an existing f32 value (no arithmetic), so this
/// matches tomopy bit-for-bit (Δ = 0) on tie-free columns (exact ties sort in
/// numpy-quicksort order, which is not portable).
///
/// `size = None` → tomopy default `max(5, ⌊0.01·ncol⌋)` (`21` for `ncol > 2000`);
/// `dim` selects the median footprint (`1` → `(size, 1)`, any other value →
/// `(size, size)`, matching tomopy's `if dim == 1 … else` branch).
fn remove_stripe_based_sorting(data: &mut Tomo<f32>, size: Option<usize>, dim: u8) -> Result<()> {
    let target = data.layout;
    // tomopy operates on `tomo[:, m, :]` = `[proj, col]` slices of the
    // `[proj, row, col]` projection-layout stack.
    let mut proj = data.to_layout(Layout::Projection);
    let (nproj, nrows, ncol) = proj.array.dim();
    if nproj < 2 || nrows == 0 || ncol == 0 {
        return Ok(());
    }
    // tomopy `_remove_stripe_based_sorting` default window (stripe.py:427-431).
    let size = size.unwrap_or_else(|| {
        if ncol > 2000 {
            21
        } else {
            5.max((0.01 * ncol as f64) as usize)
        }
    });
    if size == 0 {
        return Ok(());
    }

    for m in 0..nrows {
        let mut sino = Array2::<f32>::zeros((nproj, ncol));
        for p in 0..nproj {
            for c in 0..ncol {
                sino[[p, c]] = proj.array[[p, m, c]];
            }
        }
        let corrected = if dim == 1 {
            rs_sort(&sino, size)
        } else {
            rs_sort_with(&sino, |s| median_filter_2d(s, size))
        };
        for p in 0..nproj {
            for c in 0..ncol {
                proj.array[[p, m, c]] = corrected[[p, c]];
            }
        }
    }

    *data = proj.to_layout(target);
    Ok(())
}

// ----------------------------------------------------------------------------
// Vo filtering-based stripe removal (tomopy `prep/stripe.py`
// ::remove_stripe_based_filtering, Vo 2018 algorithm 2). Per sinogram slice:
// separate a low-pass (smooth) component with a Gaussian Fourier filter along
// the projection axis (`_rs_filter`), apply the sorting-based correction
// (`_rs_sort`) to that smooth component, then add back the high-pass residual.
// The Fourier filter runs in f64 (numpy promotes `float32 · float64-window` to
// float64) before casting the smooth component to f32, so — like the
// Fourier-Wavelet path — it matches tomopy to the f32 round-off floor, not
// bit-exactly. Projector-independent.
// ----------------------------------------------------------------------------

/// `np.pad(..., mode='reflect')` index map: whole-sample symmetric reflection
/// (the array edge is *not* repeated, unlike scipy.ndimage `reflect`), period
/// `2(n-1)`. Used to reflect-pad the projection axis before the Fourier filter.
fn reflect_whole_sample(i: isize, n: isize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut m = i % period;
    if m < 0 {
        m += period;
    }
    if m >= n {
        m = period - m;
    }
    m as usize
}

/// `scipy.signal.windows.gaussian(len, std=sigma)` (symmetric): `exp(-n²/(2σ²))`
/// with `n = arange(len) − (len−1)/2`. Computed in f64 like scipy.
fn gaussian_window(len: usize, sigma: f64) -> Vec<f64> {
    let center = (len as f64 - 1.0) / 2.0;
    let sig2 = 2.0 * sigma * sigma;
    (0..len)
        .map(|k| {
            let n = k as f64 - center;
            (-(n * n) / sig2).exp()
        })
        .collect()
}

/// `(-1)^k` (tomopy `_create_listsign`), the spatial-domain modulation that
/// turns the low-pass Fourier multiply into a band centred at the spectrum mid.
#[inline]
fn listsign(k: usize) -> f64 {
    if k % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// `_rs_filter` smooth component: for each detector column, reflect-pad the
/// projection profile, modulate by `listsign`, low-pass it with the Gaussian
/// `window` in Fourier space, demodulate, and crop back. Returns the smooth
/// sinogram `[nproj, ncol]` cast to f32 (tomopy stores it in a float32 array).
///
/// `window` has length `nproj + 2·pad`. The Fourier core
/// `real(ifft(fft(col·listsign) · window))` is the `crate::fft` f64 column
/// filter; the surrounding `· listsign` (real) commutes through `real(·)`, so
/// the post-filter demodulation is a plain elementwise sign flip.
fn rs_filter_smooth(sino: &Array2<f32>, window: &[f64], pad: usize) -> Array2<f32> {
    let (nproj, ncol) = sino.dim();
    let len = nproj + 2 * pad;
    debug_assert_eq!(window.len(), len);
    let n = nproj as isize;
    let mut smooth = Array2::<f32>::zeros((nproj, ncol));
    let mut signed = vec![0.0f64; len];
    for c in 0..ncol {
        // Reflect-padded, listsign-modulated column (f64).
        for (k, sv) in signed.iter_mut().enumerate() {
            let src = reflect_whole_sample(k as isize - pad as isize, n);
            *sv = sino[[src, c]] as f64 * listsign(k);
        }
        let filtered = fft::filter_real_column(&signed, window);
        // Demodulate and crop the padding: output index p ↔ filtered[pad + p].
        for p in 0..nproj {
            smooth[[p, c]] = (filtered[pad + p] * listsign(pad + p)) as f32;
        }
    }
    smooth
}

/// Vo filtering-based stripe removal (tomopy `remove_stripe_based_filtering`,
/// Vo 2018 algorithm 2).
///
/// `pad = min(150, ⌊0.1·nproj⌋)`; the Gaussian window length is `nproj + 2·pad`.
/// `size = None` → tomopy default `max(5, ⌊0.01·ncol⌋)` (`21` for `ncol > 2000`);
/// `dim` selects the inner `_rs_sort` median footprint exactly as `VoSort`.
fn remove_stripe_based_filtering(
    data: &mut Tomo<f32>,
    sigma: f32,
    size: Option<usize>,
    dim: u8,
) -> Result<()> {
    let target = data.layout;
    // tomopy operates on `tomo[:, m, :]` = `[proj, col]` slices of the
    // `[proj, row, col]` projection-layout stack.
    let mut proj = data.to_layout(Layout::Projection);
    let (nproj, nrows, ncol) = proj.array.dim();
    if nproj < 2 || nrows == 0 || ncol == 0 {
        return Ok(());
    }
    // tomopy `_remove_stripe_based_filtering` (stripe.py:506-517).
    let pad = 150.min((0.1 * nproj as f64) as usize);
    let window = gaussian_window(nproj + 2 * pad, sigma as f64);
    let size = size.unwrap_or_else(|| {
        if ncol > 2000 {
            21
        } else {
            5.max((0.01 * ncol as f64) as usize)
        }
    });
    if size == 0 {
        return Ok(());
    }

    for m in 0..nrows {
        let mut sino = Array2::<f32>::zeros((nproj, ncol));
        for p in 0..nproj {
            for c in 0..ncol {
                sino[[p, c]] = proj.array[[p, m, c]];
            }
        }
        // Smooth (low-pass) component, then its sorting-based correction.
        let smooth = rs_filter_smooth(&sino, &window, pad);
        let smooth_cor = if dim == 1 {
            rs_sort(&smooth, size)
        } else {
            rs_sort_with(&smooth, |s| median_filter_2d(s, size))
        };
        // out = smooth_cor + (sino − smooth)  (the high-pass residual added back).
        for p in 0..nproj {
            for c in 0..ncol {
                let sharp = sino[[p, c]] - smooth[[p, c]];
                proj.array[[p, m, c]] = smooth_cor[[p, c]] + sharp;
            }
        }
    }

    *data = proj.to_layout(target);
    Ok(())
}

// ----------------------------------------------------------------------------
// Vo fitting-based stripe removal (algorithm 1) — tomopy
// `remove_stripe_based_fitting`. Divides the sinogram by its Savitzky–Golay
// polynomial fit along the projection axis, then re-multiplies by a 2-D
// Gaussian-smoothed (mean-matched) copy of that fit. The 2-D Fourier filter
// runs in f64 (the `fft.rs` 2-D primitive), so — like the Fourier-Wavelet and
// VoFilter paths — it matches tomopy to the f32 round-off floor. The
// Savitzky–Golay weights are computed from scaled normal equations (the fit
// nodes mapped to ≈[-1, 1]), which reproduce scipy's SVD `lstsq` to the f64
// floor without an external linear-algebra dependency.
// ----------------------------------------------------------------------------

/// In-place Gaussian elimination with partial pivoting: solve `a · x = b`,
/// leaving the solution in `b`. `a` is `n×n`; used only for the tiny
/// `(order+1)` Savitzky–Golay normal-equations system.
fn solve_dense(a: &mut [Vec<f64>], b: &mut [f64]) {
    let n = b.len();
    for col in 0..n {
        // Partial pivot: pick the largest-magnitude entry at/below the diagonal.
        let piv = a
            .iter()
            .enumerate()
            .skip(col)
            .max_by(|(_, x), (_, y)| x[col].abs().total_cmp(&y[col].abs()))
            .map(|(r, _)| r)
            .unwrap_or(col);
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        // Eliminate below the pivot. Clone the pivot row so the borrow of `a[r]`
        // does not alias `a[col]`.
        let pivot_row = a[col].clone();
        let bcol = b[col];
        let d = pivot_row[col];
        for (r, ar) in a.iter_mut().enumerate().skip(col + 1) {
            let f = ar[col] / d;
            if f != 0.0 {
                for (arc, &pc) in ar.iter_mut().zip(pivot_row.iter()).skip(col) {
                    *arc -= f * pc;
                }
                b[r] -= f * bcol;
            }
        }
    }
    // Back-substitution.
    for col in (0..n).rev() {
        let s: f64 = a[col]
            .iter()
            .zip(b.iter())
            .skip(col + 1)
            .map(|(ac, bc)| ac * bc)
            .sum();
        b[col] = (b[col] - s) / a[col][col];
    }
}

/// Savitzky–Golay smoothing weights (deriv 0) for an odd `window` and polynomial
/// `order` (`order < window`) — the weights scipy's `savgol_coeffs` returns. The
/// fit nodes are scaled to `≈[-1, 1]` (`u = (k − pos)/pos`) so the normal-
/// equations solve is well-conditioned, reproducing scipy's SVD `lstsq` to the
/// f64 floor.
fn savgol_coeffs(window: usize, order: usize) -> Vec<f64> {
    if window <= 1 {
        return vec![1.0; window];
    }
    let pos = window / 2; // window is odd
    let posf = pos as f64;
    let m = order + 1;
    // powers[j][k] = u_k^j, u_k = (k − pos)/pos.
    let mut powers = vec![vec![0.0f64; window]; m];
    for k in 0..window {
        let u = (k as f64 - posf) / posf;
        let mut p = 1.0;
        for row in powers.iter_mut() {
            row[k] = p;
            p *= u;
        }
    }
    // Normal matrix M[i][j] = Σ_k u_k^i u_k^j; solve M z = e0.
    let mut mat = vec![vec![0.0f64; m]; m];
    for (i, mrow) in mat.iter_mut().enumerate() {
        for (j, cell) in mrow.iter_mut().enumerate() {
            *cell = (0..window).map(|k| powers[i][k] * powers[j][k]).sum();
        }
    }
    let mut z = vec![0.0f64; m];
    z[0] = 1.0;
    solve_dense(&mut mat, &mut z);
    // c_k = Σ_j z_j u_k^j.
    (0..window)
        .map(|k| (0..m).map(|j| z[j] * powers[j][k]).sum())
        .collect()
}

/// `scipy.signal.savgol_filter(sino, window, order, axis=0, mode='mirror')`: for
/// each detector column convolve the projection profile with the Savitzky–Golay
/// weights, reflecting at the boundary (whole-sample 'mirror', the same map as
/// numpy `reflect`). tomopy runs this on the float32 sinogram, so each output is
/// rounded to f32.
fn savgol_filter_axis0(sino: &Array2<f32>, window: usize, order: usize) -> Array2<f32> {
    let (nrow, ncol) = sino.dim();
    let coeffs = savgol_coeffs(window, order);
    let pos = (window / 2) as isize;
    let n = nrow as isize;
    let mut out = Array2::<f32>::zeros((nrow, ncol));
    for c in 0..ncol {
        for i in 0..nrow {
            let mut s = 0.0f64;
            for (k, &ck) in coeffs.iter().enumerate() {
                let idx = reflect_whole_sample(i as isize + k as isize - pos, n);
                s += ck * sino[[idx, c]] as f64;
            }
            out[[i, c]] = s as f32;
        }
    }
    out
}

/// tomopy `_create_2d_window`: a 2-D Gaussian over the padded grid
/// `(nrow+2·pad) × (ncol+2·pad)`, centred at `((H−1)/2, (W−1)/2)`,
/// `win[y][x] = exp(−((x−cx)²/(2σx²) + (y−cy)²/(2σy²)))`.
fn create_2d_window(nrow: usize, ncol: usize, sigma: (f64, f64), pad: usize) -> Array2<f64> {
    let (sigmax, sigmay) = sigma;
    let height = nrow + 2 * pad;
    let width = ncol + 2 * pad;
    let centerx = (width as f64 - 1.0) / 2.0;
    let centery = (height as f64 - 1.0) / 2.0;
    let numx = 2.0 * sigmax * sigmax;
    let numy = 2.0 * sigmay * sigmay;
    let mut w = Array2::<f64>::zeros((height, width));
    for y in 0..height {
        let dy = y as f64 - centery;
        for x in 0..width {
            let dx = x as f64 - centerx;
            w[[y, x]] = (-(dx * dx / numx + dy * dy / numy)).exp();
        }
    }
    w
}

/// tomopy `_create_matsign`: `(-1)^(x+y)` over the padded grid — the spatial
/// modulation that recentres the low-pass Gaussian on the spectrum mid.
fn create_matsign(nrow: usize, ncol: usize, pad: usize) -> Array2<f64> {
    let height = nrow + 2 * pad;
    let width = ncol + 2 * pad;
    let mut s = Array2::<f64>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            s[[y, x]] = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
    s
}

/// tomopy `_2d_filter`: edge-pad the columns and mean-pad the rows by `pad`,
/// band-pass with the 2-D Gaussian `win2d` via
/// `real(ifft2(fft2(matpad·matsign)·win2d)·matsign)`, then crop the padding. The
/// FFT path runs in f64 (numpy promotes the float32·float64-sign product),
/// matching tomopy to the f32 floor.
fn two_d_filter(
    mat: &Array2<f32>,
    win2d: &Array2<f64>,
    matsign: &Array2<f64>,
    pad: usize,
) -> Array2<f64> {
    let (nrow, ncol) = mat.dim();
    let height = nrow + 2 * pad;
    let width = ncol + 2 * pad;
    let mut matpad = Array2::<f64>::zeros((height, width));
    // Centre rows + column edges (replicate the first/last column = mode='edge').
    for r in 0..nrow {
        for x in 0..width {
            let sc = if x < pad {
                0
            } else if x >= pad + ncol {
                ncol - 1
            } else {
                x - pad
            };
            matpad[[r + pad, x]] = mat[[r, sc]] as f64;
        }
    }
    // Mean-pad the rows: each pad row = the per-column mean of the edge-padded
    // matrix (mode='mean', over the existing nrow rows), top and bottom alike.
    for x in 0..width {
        let mut sum = 0.0f64;
        for r in 0..nrow {
            sum += matpad[[r + pad, x]];
        }
        let colmean = sum / nrow as f64;
        for r in 0..pad {
            matpad[[r, x]] = colmean;
        }
        for r in (pad + nrow)..height {
            matpad[[r, x]] = colmean;
        }
    }
    // real(ifft2(fft2(matpad·matsign)·win2d)·matsign), cropped.
    let mut modulated = Array2::<f64>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            modulated[[y, x]] = matpad[[y, x]] * matsign[[y, x]];
        }
    }
    let filt = fft::filter_real_2d(&modulated, win2d);
    let mut out = Array2::<f64>::zeros((nrow, ncol));
    for r in 0..nrow {
        for c in 0..ncol {
            let (y, x) = (r + pad, c + pad);
            out[[r, c]] = filt[[y, x]] * matsign[[y, x]];
        }
    }
    out
}

/// tomopy `_rs_fit`: divide the sinogram by its Savitzky–Golay polynomial fit
/// along the projection axis, then re-multiply by the mean-matched 2-D
/// Gaussian-smoothed fit — suppressing the low-pass stripe component.
fn rs_fit(
    sino: &Array2<f32>,
    order: usize,
    win2d: &Array2<f64>,
    matsign: &Array2<f64>,
    pad: usize,
) -> Array2<f32> {
    let (nrow, ncol) = sino.dim();
    // window = nrow made odd; order clamped below window.
    let mut window = nrow;
    if window % 2 == 0 {
        window -= 1;
    }
    let order = if order >= window { window - 1 } else { order };

    let sinofit = savgol_filter_axis0(sino, window, order);
    let sinofitsmooth = two_d_filter(&sinofit, win2d, matsign, pad);
    // num1 = mean(sinofit), num2 = mean(sinofitsmooth); rescale the smooth fit.
    let denom = (nrow * ncol) as f64;
    let num1 = sinofit.iter().map(|&v| v as f64).sum::<f64>() / denom;
    let num2 = sinofitsmooth.iter().sum::<f64>() / denom;
    let mut out = Array2::<f32>::zeros((nrow, ncol));
    for r in 0..nrow {
        for c in 0..ncol {
            let fit = sinofit[[r, c]] as f64;
            let smooth = num1 * sinofitsmooth[[r, c]] / num2;
            out[[r, c]] = (sino[[r, c]] as f64 / fit * smooth) as f32;
        }
    }
    out
}

/// Vo fitting-based stripe removal (tomopy `remove_stripe_based_fitting`,
/// Vo 2018 algorithm 1) — suitable for low-pass stripes.
///
/// `pad = min(150, ⌊0.1·nproj⌋)`; the 2-D Gaussian window and `(-1)^(x+y)` sign
/// matrix (which depend only on the stack dims, `sigma`, and `pad`) are built
/// once and reused for every slice. Each `[proj, col]` slice goes through
/// `_rs_fit`. Held to the f32 round-off floor (the 2-D Fourier smoothing runs in
/// f64).
fn remove_stripe_based_fitting(
    data: &mut Tomo<f32>,
    order: usize,
    sigma: (f32, f32),
) -> Result<()> {
    let target = data.layout;
    // tomopy operates on `tomo[:, m, :]` = `[proj, col]` slices of the
    // `[proj, row, col]` projection-layout stack.
    let mut proj = data.to_layout(Layout::Projection);
    let (nproj, nrows, ncol) = proj.array.dim();
    if nproj < 2 || nrows == 0 || ncol == 0 {
        return Ok(());
    }
    let pad = 150.min((0.1 * nproj as f64) as usize);
    let sigma = (sigma.0 as f64, sigma.1 as f64);
    let win2d = create_2d_window(nproj, ncol, sigma, pad);
    let matsign = create_matsign(nproj, ncol, pad);

    for m in 0..nrows {
        let mut sino = Array2::<f32>::zeros((nproj, ncol));
        for p in 0..nproj {
            for c in 0..ncol {
                sino[[p, c]] = proj.array[[p, m, c]];
            }
        }
        let fixed = rs_fit(&sino, order, &win2d, &matsign, pad);
        for p in 0..nproj {
            for c in 0..ncol {
                proj.array[[p, m, c]] = fixed[[p, c]];
            }
        }
    }

    *data = proj.to_layout(target);
    Ok(())
}
