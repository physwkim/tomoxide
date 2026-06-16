//! Misc filters & corrections (ports tomopy `misc/corr.py` + `libtomo/misc`).
//! `circ_mask`/`remove_nan`/`remove_neg`/`adjust_range`/`median_filter_nonfinite`/
//! `remove_outlier1d` are real; the 3-D rank filters (`median_filter3d`,
//! `remove_outlier`) route through the backend (stubbed). See
//! `docs/PORTING.md` §E.

use ndarray::Axis;
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

/// Outlier (zinger) removal, dispatched to the backend's [`RankFilter`] (stub).
pub fn remove_outlier(
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
        .remove_outlier(data, diff, size)
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
