//! Stripe-artifact removal (ports tomopy `prep/stripe.py` + tomocupy
//! `processing/remove_stripe.py`). The smoothing-filter method (`Sf`) and the
//! Vo all-stripe method (`VoAll`) are implemented; `Fw`/`Ti` are stubs. See
//! `docs/PORTING.md` §D. Dispatch on [`StripeMethod`].

use ndarray::Array2;
use tomoxide_core::data::{Layout, Tomo};
use tomoxide_core::error::{Error, Result};
use tomoxide_core::params::StripeMethod;

/// Remove stripes from a sinogram stack using the selected method.
pub fn remove_stripe(data: &mut Tomo<f32>, method: StripeMethod) -> Result<()> {
    match method {
        StripeMethod::None => Ok(()),
        StripeMethod::Fw { .. } => Err(Error::todo(
            "stripe::remove_stripe_fw",
            "tomopy prep/stripe.py:88 (Fourier-Wavelet)",
        )),
        StripeMethod::Ti { .. } => Err(Error::todo(
            "stripe::remove_stripe_ti",
            "tomopy prep/stripe.py:179 (Titarenko)",
        )),
        StripeMethod::Sf { size } => remove_stripe_sf(data, size),
        StripeMethod::VoAll {
            snr,
            la_size,
            sm_size,
        } => remove_all_stripe(data, snr, la_size, sm_size),
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
/// `RectBivariateSpline`), then pass through `_rs_large` for residual stripes.
fn rs_dead(sino: &Array2<f32>, snr: f32, size: usize) -> Array2<f32> {
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

    rs_large(&work, snr, size, 0.1, true)
}

/// `_rs_sort` (Vo algorithm 3, `dim = 1`): sort each column, median-smooth the
/// sorted profiles across columns, then unsort.
fn rs_sort(sino: &Array2<f32>, size: usize) -> Array2<f32> {
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
    // median_filter footprint (size, 1) on the [ncol, nrow] array: median along
    // axis 0 (across columns) at each sorted rank.
    let smoothed = median_filter_axis0(&sortedv, size);

    let mut out = Array2::<f32>::zeros((nrow, ncol));
    for c in 0..ncol {
        for rank in 0..nrow {
            out[[perm[c][rank], c]] = smoothed[[c, rank]];
        }
    }
    out
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
        let sino = rs_dead(&sino, snr, la_size);
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
