//! Misc filters & corrections (ports tomopy `misc/corr.py` + `libtomo/misc`).
//! `circ_mask`/`remove_nan`/`remove_neg`/`adjust_range`/`median_filter_nonfinite`/
//! `median_filter`/`remove_outlier1d`/`remove_outlier`/`gaussian_filter`/
//! `sobel_filter` are real; the 3-D rank filters (`median_filter3d`,
//! `remove_outlier3d`) route through the backend (stubbed). See
//! `docs/PORTING.md` §E.

use ndarray::{Array2, Axis};
use tomoxide_core::backend::Backend;
use tomoxide_core::data::{Tomo, Volume};
use tomoxide_core::error::{Error, Result};

/// Zero everything outside a centred circle of radius `ratio · (min_dim/2)` in
/// every slice (tomopy `misc/corr.py:852`).
pub fn circ_mask(vol: &mut Volume<f32>, ratio: f32, val: f32) -> Result<()> {
    if !(0.0..=1.0).contains(&ratio) {
        return Err(Error::InvalidParam(
            "circ_mask ratio must be in [0,1]".into(),
        ));
    }
    let (nz, ny, nx) = vol.dims();
    let cy = (ny as f32 - 1.0) / 2.0;
    let cx = (nx as f32 - 1.0) / 2.0;
    let radius = ratio * (ny.min(nx) as f32) / 2.0;
    let r2 = radius * radius;
    for z in 0..nz {
        for y in 0..ny {
            let dy = y as f32 - cy;
            for x in 0..nx {
                let dx = x as f32 - cx;
                if dy * dy + dx * dx > r2 {
                    vol.array[[z, y, x]] = val;
                }
            }
        }
    }
    Ok(())
}

/// Replace non-finite values (NaN/±inf) with `val` (tomopy `misc/corr.py:506`).
pub fn remove_nan(data: &mut Tomo<f32>, val: f32) -> Result<()> {
    data.array
        .mapv_inplace(|v| if v.is_finite() { v } else { val });
    Ok(())
}

/// Replace negative values with `val` (tomopy `misc/corr.py:533`).
pub fn remove_neg(data: &mut Tomo<f32>, val: f32) -> Result<()> {
    data.array.mapv_inplace(|v| if v < 0.0 { val } else { v });
    Ok(())
}

/// Clip the dynamic range of `data` to `[dmin, dmax]` (tomopy `misc/corr.py:90`
/// `adjust_range`). A `None` bound defaults to the data's own min/max, and a
/// bound is applied only when it is *strictly* tighter than the data range
/// (strict `>`/`<`), so both-`None` and looser-than-data bounds are no-ops,
/// exactly as upstream. The data is assumed finite (numpy's NaN propagation in
/// `np.max`/`np.min` is not replicated; compose with [`remove_nan`] first).
pub fn adjust_range(data: &mut Tomo<f32>, dmin: Option<f32>, dmax: Option<f32>) -> Result<()> {
    let arr = &mut data.array;
    if arr.is_empty() {
        return Ok(());
    }
    // np.max / np.min on the unmodified array (tomopy lines 108-111).
    let data_max = arr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let data_min = arr.iter().copied().fold(f32::INFINITY, f32::min);
    let dmax = dmax.unwrap_or(data_max);
    let dmin = dmin.unwrap_or(data_min);
    // Clip high only if dmax lies below the data max (tomopy lines 111-112).
    if dmax < data_max {
        arr.mapv_inplace(|v| if v > dmax { dmax } else { v });
    }
    // np.min is recomputed after the high clip (tomopy line 113); the high clip
    // cannot lower the min, so this equals data_min unless dmax < data_min.
    let cur_min = arr.iter().copied().fold(f32::INFINITY, f32::min);
    if dmin > cur_min {
        arr.mapv_inplace(|v| if v < dmin { dmin } else { v });
    }
    Ok(())
}

/// Replace every non-finite value (NaN/±inf) with the median of the finite
/// values in its `size×size` neighbourhood along the last two axes (tomopy
/// `misc/corr.py:281` `median_filter_nonfinite`).
///
/// Each 2-D slice (axis 0 = projection) is corrected against a snapshot taken
/// before any write, so two adjacent bad pixels do not see each other's fixes
/// (tomopy's `projection_copy`). The window is clamped to the slice (the bounds
/// use `i ± size/2`, so an even `size` gives an odd `2·(size/2)+1` width, as
/// upstream). `np.median` on float32 returns float32 (the even-count midpoint is
/// the f32 mean of the two middle order statistics), reproduced here. Errors if
/// any kernel contains no finite value (tomopy raises `ValueError`).
pub fn median_filter_nonfinite(data: &mut Tomo<f32>, size: usize) -> Result<()> {
    if size == 0 {
        return Err(Error::InvalidParam(
            "median_filter_nonfinite size must be > 0".into(),
        ));
    }
    let (n0, n1, n2) = data.array.dim();
    let h = size / 2;
    let mut window: Vec<f32> = Vec::new();
    for i0 in 0..n0 {
        // Snapshot: medians read pre-correction values (tomopy projection_copy).
        let snap = data.array.index_axis(Axis(0), i0).to_owned();
        for i1 in 0..n1 {
            for i2 in 0..n2 {
                if snap[[i1, i2]].is_finite() {
                    continue;
                }
                let x_lo = i1.saturating_sub(h);
                let x_hi = (i1 + h + 1).min(n1);
                let y_lo = i2.saturating_sub(h);
                let y_hi = (i2 + h + 1).min(n2);
                window.clear();
                for x in x_lo..x_hi {
                    for y in y_lo..y_hi {
                        let v = snap[[x, y]];
                        if v.is_finite() {
                            window.push(v);
                        }
                    }
                }
                if window.is_empty() {
                    return Err(Error::InvalidParam(
                        "median_filter_nonfinite: kernel contains only non-finite \
                         values; increase size"
                            .into(),
                    ));
                }
                data.array[[i0, i1, i2]] = median_f32(&mut window);
            }
        }
    }
    Ok(())
}

/// `np.median` of finite `vals` (no NaN): the middle order statistic for an odd
/// count, the f32 mean of the two middle order statistics for an even count.
fn median_f32(vals: &mut [f32]) -> f32 {
    vals.sort_by(|a, b| a.partial_cmp(b).expect("vals are finite"));
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    }
}

/// 3-D median filter, dispatched to the backend's [`RankFilter`] (stub).
pub fn median_filter3d(vol: &mut Volume<f32>, size: usize, backend: &dyn Backend) -> Result<()> {
    backend
        .rank_filter()
        .ok_or(Error::MissingCapability {
            backend: backend.name(),
            capability: "RankFilter",
        })?
        .median3d(vol, size)
}

/// 3-D-cube outlier (zinger) removal, dispatched to the backend's
/// [`RankFilter`] (tomopy `misc/corr.py:413` `remove_outlier3d`). Distinct from
/// [`remove_outlier`], which is the axis-chunked 2-D dezinger (corr.py:559).
pub fn remove_outlier3d(
    data: &mut Tomo<f32>,
    diff: f32,
    size: usize,
    backend: &dyn Backend,
) -> Result<()> {
    backend
        .rank_filter()
        .ok_or(Error::MissingCapability {
            backend: backend.name(),
            capability: "RankFilter",
        })?
        .remove_outlier3d(data, diff, size)
}

/// Median-filter every 2-D slice along `axis` with a `size×size` footprint
/// (tomopy `misc/corr.py:167` `median_filter`). For each index along `axis` the
/// orthogonal 2-D image is replaced by its `size×size` median taken with
/// scipy.ndimage's default `mode='reflect'` (half-sample reflection, the edge
/// sample repeated). `axis` indexes the underlying 3-D array (0/1/2), matching
/// tomopy's `axis` on the raw ndarray.
///
/// scipy's median filter selects a single order statistic (rank
/// `size·size/2`, never an average — even for an even footprint), so the result
/// is bit-exact (Δ=0) vs tomopy on finite input. Unlike [`remove_outlier1d`]
/// there is no threshold: every pixel is replaced by its local median. Input is
/// assumed finite (compose with [`remove_nan`] first if needed).
pub fn median_filter(data: &mut Tomo<f32>, size: usize, axis: usize) -> Result<()> {
    if size == 0 {
        return Err(Error::InvalidParam("median_filter size must be > 0".into()));
    }
    if axis > 2 {
        return Err(Error::InvalidParam(
            "median_filter axis must be 0, 1, or 2".into(),
        ));
    }
    data.array = median2d_reflect(&data.array, size, axis);
    Ok(())
}

/// Remove bright outliers with a per-slice 2-D median along `axis` (tomopy
/// `misc/corr.py:559` `remove_outlier`). For each index along `axis` the
/// orthogonal 2-D image's `size×size` median is taken with scipy.ndimage's
/// default `mode='reflect'`, then a pixel is replaced by that median only where
/// it exceeds it by at least `diff` (`arr − median ≥ diff`, strict `<` keeps the
/// pixel). `axis` indexes the underlying 3-D array (0/1/2), matching tomopy's
/// `axis` on the raw ndarray.
///
/// Shares the 2-D median primitive with [`median_filter`]: the median is a
/// single order statistic and the `where` test is a plain f32 subtraction, so
/// the result is bit-exact (Δ=0) vs tomopy on finite input. This is the 2-D
/// per-slice dezinger; [`remove_outlier3d`] is the 3-D-cube variant and
/// [`remove_outlier1d`] the 1-D (`mode='mirror'`) variant.
pub fn remove_outlier(data: &mut Tomo<f32>, diff: f32, size: usize, axis: usize) -> Result<()> {
    if size == 0 {
        return Err(Error::InvalidParam(
            "remove_outlier size must be > 0".into(),
        ));
    }
    if axis > 2 {
        return Err(Error::InvalidParam(
            "remove_outlier axis must be 0, 1, or 2".into(),
        ));
    }
    let tmp = median2d_reflect(&data.array, size, axis);
    ndarray::Zip::from(&mut data.array)
        .and(&tmp)
        .for_each(|a, &t| {
            if *a - t >= diff {
                *a = t;
            }
        });
    Ok(())
}

/// Remove bright outliers with a 1-D median filter along `axis` (tomopy
/// `misc/corr.py:615` `remove_outlier1d`). For each element the local
/// `size`-tap median along `axis` is taken with scipy.ndimage `mode='mirror'`
/// (whole-sample reflection); a pixel is then replaced by that median only when
/// it exceeds it by at least `diff` (`arr − median ≥ diff`, strict `<` keeps the
/// pixel), all others pass through unchanged. `axis` indexes the underlying 3-D
/// array (0/1/2), matching tomopy's `axis` on the raw ndarray.
///
/// scipy's median filter selects a single order statistic (rank `size/2`, never
/// an average — even for even `size`), and the `where` test is a plain f32
/// subtraction, so the result is bit-exact (Δ=0) vs tomopy on finite input.
/// Input is assumed finite (the dezinger operates on real projection data;
/// compose with [`remove_nan`] first if needed).
pub fn remove_outlier1d(data: &mut Tomo<f32>, diff: f32, size: usize, axis: usize) -> Result<()> {
    if size == 0 {
        return Err(Error::InvalidParam(
            "remove_outlier1d size must be > 0".into(),
        ));
    }
    if axis > 2 {
        return Err(Error::InvalidParam(
            "remove_outlier1d axis must be 0, 1, or 2".into(),
        ));
    }
    let dims = data.array.dim();
    let shape = [dims.0, dims.1, dims.2];
    let len = shape[axis] as isize;
    if len == 0 {
        return Ok(());
    }
    // scipy.ndimage.median_filter footprint origin = size/2; median rank = size/2
    // (filter_size // 2). For even `size` this picks one element, not a mean.
    let orgn = (size / 2) as isize;
    let rank = size / 2;
    // The two axes orthogonal to `axis`, iterated as independent 1-D lines.
    let (a1, a2) = match axis {
        0 => (1usize, 2usize),
        1 => (0usize, 2usize),
        _ => (0usize, 1usize),
    };
    // Fill the median-filtered array `tmp` from the unmodified `data.array`,
    // then apply the `where` replacement in place (the comparison reads the
    // original values, which are still intact because `tmp` is separate).
    let mut tmp = ndarray::Array3::<f32>::zeros(dims);
    let mut window: Vec<f32> = Vec::with_capacity(size);
    for p1 in 0..shape[a1] {
        for p2 in 0..shape[a2] {
            for i in 0..shape[axis] {
                window.clear();
                for k in 0..size {
                    let src = mirror_index(i as isize + k as isize - orgn, len);
                    let mut idx = [0usize; 3];
                    idx[axis] = src;
                    idx[a1] = p1;
                    idx[a2] = p2;
                    window.push(data.array[[idx[0], idx[1], idx[2]]]);
                }
                window.sort_by(|a, b| a.partial_cmp(b).expect("input is finite"));
                let mut oidx = [0usize; 3];
                oidx[axis] = i;
                oidx[a1] = p1;
                oidx[a2] = p2;
                tmp[[oidx[0], oidx[1], oidx[2]]] = window[rank];
            }
        }
    }
    ndarray::Zip::from(&mut data.array)
        .and(&tmp)
        .for_each(|a, &t| {
            if *a - t >= diff {
                *a = t;
            }
        });
    Ok(())
}

/// Gaussian-filter every 2-D slice along `axis` (tomopy `misc/corr.py:118`
/// `gaussian_filter`). For each index along `axis` the orthogonal 2-D image is
/// convolved with a separable Gaussian (1-D pass along each slice axis, the
/// intermediate stored in f32 between passes, exactly as scipy.ndimage). `sigma`
/// is the standard deviation (applied to both slice axes), `order` the
/// derivative order (0 = plain Gaussian; ≥1 = that derivative of a Gaussian),
/// and `axis` (0/1/2) indexes the underlying 3-D array, matching tomopy's `axis`.
///
/// Faithful to scipy.ndimage: the kernel radius is `⌊4σ + 0.5⌋` (`truncate=4`),
/// the kernel is `exp(−x²/2σ²)` normalised by numpy's f64 pairwise sum then
/// reversed for correlation, and the convolution accumulates in f64 with
/// scipy's exact symmetric / anti-symmetric summation branch and `mode='reflect'`
/// (half-sample) boundaries. Because the kernel uses `exp` (a transcendental,
/// where numpy's vectorised f64 `exp` and libm differ by ≤1 ULP), the result is
/// held to the **f32 round-off floor** (≤1 ULP), like the Fourier stripe ports.
/// `σ ≤ 1e-15` is a no-op copy (scipy skips such axes). Input is assumed finite.
pub fn gaussian_filter(data: &mut Tomo<f32>, sigma: f64, order: usize, axis: usize) -> Result<()> {
    if axis > 2 {
        return Err(Error::InvalidParam(
            "gaussian_filter axis must be 0, 1, or 2".into(),
        ));
    }
    if sigma < 0.0 || sigma.is_nan() {
        return Err(Error::InvalidParam(
            "gaussian_filter sigma must be >= 0".into(),
        ));
    }
    // scipy skips axes with sigma <= 1e-15; both slice axes use the same sigma,
    // so the whole filter is then the identity.
    if sigma <= 1e-15 {
        return Ok(());
    }
    // radius lw = int(truncate*sigma + 0.5), truncate = 4.0 (scipy default).
    let lw = (4.0 * sigma + 0.5) as usize;
    let weights = gaussian_kernel1d(sigma, order, lw);
    let nslices = [data.array.dim().0, data.array.dim().1, data.array.dim().2][axis];
    for s in 0..nslices {
        let slice = data.array.index_axis(Axis(axis), s).to_owned();
        let filtered = gaussian_filter2d(&slice, &weights);
        data.array.index_axis_mut(Axis(axis), s).assign(&filtered);
    }
    Ok(())
}

/// Sobel-filter every 2-D slice along `axis` (tomopy `misc/corr.py:474`
/// `sobel_filter`). For each index along `axis` the orthogonal 2-D image is run
/// through scipy.ndimage's Sobel transform: a `[−1, 0, 1]` central-difference
/// correlation along the slice's last axis, then a `[1, 2, 1]` smoothing
/// correlation along the other axis (both `mode='reflect'`). `axis` (0/1/2)
/// indexes the underlying 3-D array, selecting which 2-D slices are taken,
/// exactly like tomopy (which leaves scipy's `axis=-1` default, so the gradient
/// is always along the slice's last axis).
///
/// Reuses the f64 [`correlate1d_2d`] primitive shared with [`gaussian_filter`].
/// The weights are exact small integers (no transcendental), and f32 inputs are
/// exact in the f64 accumulator, so the result is **bit-exact (Δ=0)** vs tomopy
/// on finite input.
pub fn sobel_filter(data: &mut Tomo<f32>, axis: usize) -> Result<()> {
    if axis > 2 {
        return Err(Error::InvalidParam(
            "sobel_filter axis must be 0, 1, or 2".into(),
        ));
    }
    // scipy.ndimage.sobel weights (passed to correlate1d unreversed).
    const DERIV: [f64; 3] = [-1.0, 0.0, 1.0];
    const SMOOTH: [f64; 3] = [1.0, 2.0, 1.0];
    let nslices = [data.array.dim().0, data.array.dim().1, data.array.dim().2][axis];
    for s in 0..nslices {
        let slice = data.array.index_axis(Axis(axis), s).to_owned();
        // axis=-1 → derivative along slice-axis 1; smoothing along slice-axis 0.
        let deriv = correlate1d_2d(&slice, &DERIV, 1);
        let out = correlate1d_2d(&deriv, &SMOOTH, 0);
        data.array.index_axis_mut(Axis(axis), s).assign(&out);
    }
    Ok(())
}

/// Separable 2-D Gaussian filter on a single slice: a 1-D `correlate1d` pass
/// along slice-axis 0 then slice-axis 1, the intermediate stored in f32 between
/// passes (scipy.ndimage keeps intermediates in the output dtype). `weights` is
/// the reversed f64 kernel shared by both axes.
fn gaussian_filter2d(slice: &Array2<f32>, weights: &[f64]) -> Array2<f32> {
    let mut cur = correlate1d_2d(slice, weights, 0);
    cur = correlate1d_2d(&cur, weights, 1);
    cur
}

/// Per-axis symmetry class of a correlation kernel, mirroring scipy's
/// `NI_Correlate1D` test (`symmetric` = 1 / −1 / 0).
enum Sym {
    Even,
    Odd,
    None,
}

/// scipy's `NI_Correlate1D` symmetry test: only odd-length kernels can be
/// symmetric; `|fw[i+size1] − fw[size1−i]| ≤ DBL_EPSILON` ⇒ even-symmetric,
/// `|fw[size1+i] + fw[size1−i]| ≤ DBL_EPSILON` ⇒ anti-symmetric.
fn filter_symmetry(weights: &[f64], size1: usize) -> Sym {
    let fsize = weights.len();
    if fsize % 2 == 0 {
        return Sym::None;
    }
    let half = fsize / 2;
    if (1..=half).all(|i| (weights[i + size1] - weights[size1 - i]).abs() <= f64::EPSILON) {
        return Sym::Even;
    }
    if (1..=half).all(|i| (weights[size1 + i] + weights[size1 - i]).abs() <= f64::EPSILON) {
        return Sym::Odd;
    }
    Sym::None
}

/// One scipy.ndimage `correlate1d` pass over `img` along slice-axis `sax`
/// (0 = down columns / 1 = across rows), accumulating in f64 (scipy's line
/// buffers are `double`) with `mode='reflect'` (half-sample) boundaries and
/// `origin = 0`. The symmetric / anti-symmetric / general branch — and its exact
/// summation order — matches `NI_Correlate1D`, so integer-weight kernels (sobel)
/// are bit-exact and Gaussian kernels reach the f32 round-off floor. `weights`
/// is the f64 kernel already reversed for correlation by the caller.
fn correlate1d_2d(img: &Array2<f32>, weights: &[f64], sax: usize) -> Array2<f32> {
    let (nr, nc) = img.dim();
    let fsize = weights.len();
    let size1 = fsize / 2;
    let size2 = fsize - size1 - 1;
    let sym = filter_symmetry(weights, size1);
    let (nr_i, nc_i) = (nr, nc);
    let len = if sax == 0 { nr_i } else { nc_i } as isize;
    let lines = if sax == 0 { nc_i } else { nr_i };
    let mut out = Array2::<f32>::zeros((nr, nc));
    for line in 0..lines {
        // Value at position `p` along `sax` (reflect-extended), other index = line.
        let get = |p: isize| -> f64 {
            let q = reflect_index(p, len);
            (if sax == 0 {
                img[[q, line]]
            } else {
                img[[line, q]]
            }) as f64
        };
        for ll in 0..len {
            let acc: f64 = match sym {
                // oline[ll] = iline[0]*fw[0] + Σ_{jj=-size1..-1}(iline[jj]±iline[-jj])*fw[jj]
                // fw was advanced by size1, so fw[jj] = weights[size1+jj]; the loop runs
                // jj = -size1..-1 (outermost pair first), matching scipy's order.
                Sym::Even => {
                    let mut a = get(ll) * weights[size1];
                    let mut jj = -(size1 as isize);
                    while jj < 0 {
                        a +=
                            (get(ll + jj) + get(ll - jj)) * weights[(size1 as isize + jj) as usize];
                        jj += 1;
                    }
                    a
                }
                Sym::Odd => {
                    let mut a = get(ll) * weights[size1];
                    let mut jj = -(size1 as isize);
                    while jj < 0 {
                        a +=
                            (get(ll + jj) - get(ll - jj)) * weights[(size1 as isize + jj) as usize];
                        jj += 1;
                    }
                    a
                }
                // oline[ll] = iline[size2]*fw[size2] + Σ_{jj=-size1..size2-1} iline[jj]*fw[jj]
                Sym::None => {
                    let mut a = get(ll + size2 as isize) * weights[size1 + size2];
                    let mut jj = -(size1 as isize);
                    while jj < size2 as isize {
                        a += get(ll + jj) * weights[(size1 as isize + jj) as usize];
                        jj += 1;
                    }
                    a
                }
            };
            let o = acc as f32;
            if sax == 0 {
                out[[ll as usize, line]] = o;
            } else {
                out[[line, ll as usize]] = o;
            }
        }
    }
    out
}

/// scipy.ndimage `gaussian_filter1d`'s kernel: `_gaussian_kernel1d(sigma, order,
/// lw)` reversed for correlation. For `order 0` it is the normalised Gaussian
/// `exp(−x²/2σ²)/Σ`; for `order ≥ 1` it is multiplied by the polynomial from
/// scipy's derivative recurrence (`q' + q·p'`, `p'=−1/σ²`). The Gaussian is
/// normalised with numpy's f64 pairwise sum so the weights match numpy bit-for-bit
/// up to the `exp` round-off floor. Returns a `2·lw+1`-length f64 kernel.
fn gaussian_kernel1d(sigma: f64, order: usize, lw: usize) -> Vec<f64> {
    let sigma2 = sigma * sigma;
    let n = 2 * lw + 1;
    let f = -0.5 / sigma2;
    // phi_x = exp(-0.5/sigma2 * x**2), x = arange(-lw, lw+1) (x**2 exact in int).
    let mut phi: Vec<f64> = (0..n)
        .map(|i| {
            let xi = i as i64 - lw as i64;
            let xx = (xi * xi) as f64;
            (f * xx).exp()
        })
        .collect();
    let s = pairwise_sum_f64(&phi);
    for v in &mut phi {
        *v /= s;
    }
    let mut kernel = if order == 0 {
        phi
    } else {
        // q(x): start q = [1, 0, ..., 0]; apply Q_deriv = D + P `order` times.
        // D[i][i+1] = i+1 (q' operator); P[i+1][i] = -1/sigma2 (q·p' operator).
        let mut q = vec![0.0f64; order + 1];
        q[0] = 1.0;
        for _ in 0..order {
            let mut nq = vec![0.0f64; order + 1];
            for i in 0..=order {
                // (D@q)[i] = (i+1)*q[i+1]
                if i < order {
                    nq[i] += ((i + 1) as f64) * q[i + 1];
                }
                // (P@q)[i] = (-1/sigma2)*q[i-1]
                if i >= 1 {
                    nq[i] += (-1.0 / sigma2) * q[i - 1];
                }
            }
            q = nq;
        }
        // q_poly(x_i) = Σ_j x_i^j * q[j], then multiply by phi_x elementwise.
        (0..n)
            .map(|i| {
                let xi = (i as i64 - lw as i64) as f64;
                let mut acc = 0.0f64;
                let mut xpow = 1.0f64;
                for &qj in &q {
                    acc += xpow * qj;
                    xpow *= xi;
                }
                acc * phi[i]
            })
            .collect()
    };
    kernel.reverse();
    kernel
}

/// numpy's f64 pairwise summation (the f64 analogue of `normalize`'s
/// `pairwise_sum_f32`): sequential for `n < 8`, an 8-accumulator unrolled base
/// case for `n ≤ 128`, otherwise split at `n/2` rounded down to a multiple of 8
/// and recurse. Reproduces `np.sum`/`ndarray.mean` over f64 exactly, so a
/// Gaussian kernel's normalisation matches numpy bit-for-bit.
fn pairwise_sum_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    if n < 8 {
        let mut res = 0.0f64;
        for &v in a {
            res += v;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i + 8 <= n {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum_f64(&a[..n2]) + pairwise_sum_f64(&a[n2..])
}

/// Per-slice 2-D median over the two axes orthogonal to `axis`, with a
/// `size×size` footprint and scipy.ndimage's default `mode='reflect'`
/// (half-sample reflection). Selects a single order statistic (rank
/// `size·size/2`, never an average), so it is bit-exact. The caller validates
/// `size > 0` and `axis ≤ 2`. Shared by [`median_filter`] and [`remove_outlier`].
fn median2d_reflect(arr: &ndarray::Array3<f32>, size: usize, axis: usize) -> ndarray::Array3<f32> {
    let dims = arr.dim();
    let shape = [dims.0, dims.1, dims.2];
    // scipy.ndimage.median_filter footprint origin = size/2 per axis; median rank
    // = filter_size // 2 = size·size // 2 (a single element, not a mean).
    let orgn = (size / 2) as isize;
    let rank = (size * size) / 2;
    // The two axes orthogonal to `axis` form each 2-D slice (row = a1, col = a2).
    let (a1, a2) = match axis {
        0 => (1usize, 2usize),
        1 => (0usize, 2usize),
        _ => (0usize, 1usize),
    };
    let nr = shape[a1] as isize;
    let nc = shape[a2] as isize;
    let mut out = ndarray::Array3::<f32>::zeros(dims);
    let mut window: Vec<f32> = Vec::with_capacity(size * size);
    for s in 0..shape[axis] {
        for r in 0..shape[a1] {
            for c in 0..shape[a2] {
                window.clear();
                for dr in 0..size {
                    let sr = reflect_index(r as isize + dr as isize - orgn, nr);
                    for dc in 0..size {
                        let sc = reflect_index(c as isize + dc as isize - orgn, nc);
                        let mut idx = [0usize; 3];
                        idx[axis] = s;
                        idx[a1] = sr;
                        idx[a2] = sc;
                        window.push(arr[[idx[0], idx[1], idx[2]]]);
                    }
                }
                window.sort_by(|a, b| a.partial_cmp(b).expect("input is finite"));
                let mut oidx = [0usize; 3];
                oidx[axis] = s;
                oidx[a1] = r;
                oidx[a2] = c;
                out[[oidx[0], oidx[1], oidx[2]]] = window[rank];
            }
        }
    }
    out
}

/// scipy.ndimage `mode='reflect'` index map: half-sample symmetric reflection
/// (period `2n`, the edge sample *is* repeated), matching `NI_EXTEND_REFLECT`
/// in scipy `ni_support.c`.
fn reflect_index(i: isize, n: isize) -> usize {
    if n <= 1 {
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

/// scipy.ndimage `mode='mirror'` index map: whole-sample symmetric reflection
/// (period `2(n−1)`, the edge sample is *not* repeated), matching
/// `NI_EXTEND_MIRROR` in scipy `ni_support.c`.
fn mirror_index(i: isize, n: isize) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use tomoxide_core::data::Layout;

    #[test]
    fn circ_mask_zeros_corners_keeps_center() {
        let mut v = Volume::new(Array3::from_elem((1, 5, 5), 1.0f32));
        circ_mask(&mut v, 1.0, 0.0).unwrap();
        assert_eq!(v.array[[0, 0, 0]], 0.0); // corner masked
        assert_eq!(v.array[[0, 2, 2]], 1.0); // center kept
    }

    #[test]
    fn remove_nan_and_neg() {
        let arr =
            Array3::from_shape_vec((1, 1, 4), vec![f32::NAN, -2.0, 3.0, f32::INFINITY]).unwrap();
        let mut t = Tomo::new(arr, Layout::Projection);
        remove_nan(&mut t, 0.0).unwrap();
        remove_neg(&mut t, 0.0).unwrap();
        assert_eq!(t.array.as_slice().unwrap(), &[0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn median_filter_rejects_bad_params() {
        let mut t = Tomo::new(Array3::<f32>::zeros((1, 4, 4)), Layout::Projection);
        assert!(matches!(
            median_filter(&mut t, 0, 0),
            Err(Error::InvalidParam(_))
        ));
        assert!(matches!(
            median_filter(&mut t, 3, 3),
            Err(Error::InvalidParam(_))
        ));
    }

    #[test]
    fn median_filter_replaces_spike_with_2d_median() {
        // One 3×3 slice (axis 0) with a single bright spike at the centre.
        // size=3 → 9-tap median; the centre window is the whole slice, whose
        // median (rank 4 of {0,1,2,3,4,5,6,7,100} sorted) is 4.
        #[rustfmt::skip]
        let slice = vec![
            0.0f32, 1.0, 2.0,
            3.0,  100.0, 4.0,
            5.0,    6.0, 7.0,
        ];
        let arr = Array3::from_shape_vec((1, 3, 3), slice).unwrap();
        let mut t = Tomo::new(arr, Layout::Projection);
        median_filter(&mut t, 3, 0).unwrap();
        // Centre: sorted {0,1,2,3,4,5,6,7,100}, rank 9/2=4 → 4.0 (spike removed).
        assert_eq!(t.array[[0, 1, 1]], 4.0);
    }

    #[test]
    fn remove_outlier_rejects_bad_params() {
        let mut t = Tomo::new(Array3::<f32>::zeros((1, 4, 4)), Layout::Projection);
        assert!(matches!(
            remove_outlier(&mut t, 0.5, 0, 0),
            Err(Error::InvalidParam(_))
        ));
        assert!(matches!(
            remove_outlier(&mut t, 0.5, 3, 3),
            Err(Error::InvalidParam(_))
        ));
    }

    #[test]
    fn remove_outlier_replaces_only_above_threshold() {
        // 3×3 slice (axis 0) with a bright spike at the centre; 2-D median is 4.0.
        #[rustfmt::skip]
        let slice = vec![
            0.0f32, 1.0, 2.0,
            3.0,  100.0, 4.0,
            5.0,    6.0, 7.0,
        ];
        // Small threshold: spike deviates 100−4=96 ≥ 1 → replaced; others kept
        // (their |value − local median| stays below 1).
        let mut t = Tomo::new(
            Array3::from_shape_vec((1, 3, 3), slice.clone()).unwrap(),
            Layout::Projection,
        );
        remove_outlier(&mut t, 1.0, 3, 0).unwrap();
        assert_eq!(t.array[[0, 1, 1]], 4.0); // spike replaced by median
        assert_eq!(t.array[[0, 0, 0]], 0.0); // non-outlier kept

        // Huge threshold: nothing exceeds it → input unchanged.
        let mut t = Tomo::new(
            Array3::from_shape_vec((1, 3, 3), slice).unwrap(),
            Layout::Projection,
        );
        remove_outlier(&mut t, 1000.0, 3, 0).unwrap();
        assert_eq!(t.array[[0, 1, 1]], 100.0);
    }

    #[test]
    fn gaussian_filter_rejects_bad_params() {
        let mut t = Tomo::new(Array3::<f32>::zeros((1, 4, 4)), Layout::Projection);
        assert!(matches!(
            gaussian_filter(&mut t, 1.0, 0, 3),
            Err(Error::InvalidParam(_))
        ));
        assert!(matches!(
            gaussian_filter(&mut t, -1.0, 0, 0),
            Err(Error::InvalidParam(_))
        ));
    }

    #[test]
    fn gaussian_filter_zero_sigma_is_noop() {
        // scipy skips axes with sigma <= 1e-15, so the filter is the identity.
        let arr = Array3::from_shape_vec((1, 2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut t = Tomo::new(arr.clone(), Layout::Projection);
        gaussian_filter(&mut t, 0.0, 0, 0).unwrap();
        assert_eq!(t.array, arr);
    }

    #[test]
    fn gaussian_filter_smooths_spike() {
        // One 7×7 slice (axis 0): a unit spike on a zero field. A Gaussian blur
        // must lower the centre below 1, raise its 4-neighbours above 0, leave
        // the centre as the global max, and (kernel sums to ~1) roughly conserve
        // the total mass.
        let mut slice = vec![0.0f32; 49];
        slice[3 * 7 + 3] = 1.0;
        let arr = Array3::from_shape_vec((1, 7, 7), slice).unwrap();
        let before: f64 = arr.iter().map(|&v| v as f64).sum();
        let mut t = Tomo::new(arr, Layout::Projection);
        gaussian_filter(&mut t, 1.0, 0, 0).unwrap();
        let c = t.array[[0, 3, 3]];
        assert!(c > 0.0 && c < 1.0, "centre {c} not in (0,1)");
        assert!(t.array[[0, 3, 2]] > 0.0, "neighbour not raised");
        assert!(t.array[[0, 2, 3]] > 0.0, "neighbour not raised");
        // Centre stays the maximum of the blurred field.
        let max = t.array.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(c, max);
        // Mass roughly conserved (truncation + reflect edges, so only approx).
        let after: f64 = t.array.iter().map(|&v| v as f64).sum();
        assert!((after - before).abs() < 0.05, "mass {before} -> {after}");
    }

    #[test]
    fn sobel_filter_rejects_bad_axis() {
        let mut t = Tomo::new(Array3::<f32>::zeros((1, 4, 4)), Layout::Projection);
        assert!(matches!(
            sobel_filter(&mut t, 3),
            Err(Error::InvalidParam(_))
        ));
    }

    #[test]
    fn sobel_filter_responds_to_ramp_edge() {
        // One 3×5 slice (axis 0), each row a column ramp [0,1,2,3,4].
        // scipy.ndimage.sobel: central diff [-1,0,1] along the last axis (cols)
        // with reflect edges → per row [1,2,2,2,1]; then [1,2,1] smoothing along
        // rows (all rows identical → ×4) → [4,8,8,8,4].
        #[rustfmt::skip]
        let slice = vec![
            0.0f32, 1.0, 2.0, 3.0, 4.0,
            0.0,    1.0, 2.0, 3.0, 4.0,
            0.0,    1.0, 2.0, 3.0, 4.0,
        ];
        let arr = Array3::from_shape_vec((1, 3, 5), slice).unwrap();
        let mut t = Tomo::new(arr, Layout::Projection);
        sobel_filter(&mut t, 0).unwrap();
        let want = [4.0f32, 8.0, 8.0, 8.0, 4.0];
        for r in 0..3 {
            for (c, &w) in want.iter().enumerate() {
                assert_eq!(t.array[[0, r, c]], w, "mismatch at ({r},{c})");
            }
        }
    }

    #[test]
    fn remove_outlier1d_rejects_bad_params() {
        let mut t = Tomo::new(Array3::<f32>::zeros((1, 1, 4)), Layout::Projection);
        assert!(matches!(
            remove_outlier1d(&mut t, 0.5, 0, 2),
            Err(Error::InvalidParam(_))
        ));
        assert!(matches!(
            remove_outlier1d(&mut t, 0.5, 3, 3),
            Err(Error::InvalidParam(_))
        ));
    }

    #[test]
    fn remove_outlier1d_replaces_spike_with_mirror_median() {
        // One line of length 5 along axis 2 with a single bright spike at i=2.
        // size=3, mode='mirror' (whole-sample reflection). Medians of the
        // centred 3-taps: i0 [b,a,b]; the spike (10) is the max in its windows
        // so it never enters a median as the middle value while neighbours stay.
        let line = vec![1.0f32, 2.0, 10.0, 2.0, 1.0];
        let arr = Array3::from_shape_vec((1, 1, 5), line).unwrap();
        let mut t = Tomo::new(arr, Layout::Projection);
        remove_outlier1d(&mut t, 0.5, 3, 2).unwrap();
        // i=2 window {2,10,2} median 2; 10-2=8 >= 0.5 -> replaced by 2.
        // i=1 window {1,2,10} median 2; 2-2=0  < 0.5 -> kept.
        // i=3 window {10,2,1} median 2; 2-2=0  < 0.5 -> kept.
        assert_eq!(t.array[[0, 0, 2]], 2.0);
        assert_eq!(t.array[[0, 0, 1]], 2.0);
        assert_eq!(t.array[[0, 0, 3]], 2.0);
        // Mirror edges: i=0 window {a,a,b}={1,1,2} median 1; kept.
        assert_eq!(t.array[[0, 0, 0]], 1.0);
        assert_eq!(t.array[[0, 0, 4]], 1.0);
    }
}
