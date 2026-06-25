//! Rotation-center finding (ports tomopy `recon/rotation.py` + tomocupy
//! `find_center.py`). `find_center` (entropy), `find_center_vo` (Nghia Vo's
//! method), `find_center_pc` (phase correlation), and `write_center`
//! (reconstruct a slice across a range of centers) are implemented;
//! `find_center_sift` is implemented behind the `sift-center` feature (it links
//! OpenCV). See `docs/PORTING.md` §C.

use ndarray::{Array2, Array3, ArrayViewMut2, Axis, Slice};
use crate::backend::{Backend, Fft};
use crate::data::{Layout, Tomo};
use crate::dtype::Complex32;
use crate::error::{Error, Result};
use crate::geometry::{Angles, Beam, Center, Detector, Geometry};

/// Entropy-based center finding (tomopy `rotation.py:82`).
///
/// Reconstructs a single slice with gridrec at candidate centers and minimises
/// the Shannon entropy of the masked reconstruction's 64-bin histogram with a
/// Nelder-Mead simplex search (Donath et al. 2006, doi:10.1364/JOSAA.23.001048),
/// exactly as tomopy does — the systematic artifact a wrong center introduces
/// raises the reconstruction's entropy, so the entropy minimum near a good
/// initial guess marks the axis.
///
/// Unlike `find_center_vo`/`find_center_pc` this goes *through* the projector
/// (gridrec), so it inherits the linear-interp-vs-Siddon gridrec gap (see
/// PORTING): the entropy surface is a near-replica of tomopy's but not bit-exact,
/// and the result is the local basin Nelder-Mead reaches from `init`. The center
/// is therefore held to ±~1 px of the injected/tomopy value, not bit parity.
///
/// `ind` selects the slice (default `n_rows/2`); `init` is the optimiser start
/// (default `n_cols/2`, tomopy's `dx//2`); `tol` sets both the x- and
/// f-tolerance of the simplex termination (tomopy `tol=0.5`).
pub fn find_center(
    tomo: &Tomo<f32>,
    theta: &[f32],
    backend: &dyn Backend,
    ind: Option<usize>,
    init: Option<f32>,
    tol: f32,
) -> Result<f32> {
    let fft = backend.fft().ok_or_else(|| Error::MissingCapability {
        backend: backend.name(),
        capability: "Fft",
    })?;
    let sino = tomo.to_layout(Layout::Sinogram); // [row, angle, col]
    let (nrows, nang, ncol) = sino.array.dim();
    if theta.len() != nang {
        return Err(Error::ShapeMismatch {
            expected: format!("{nang} angles"),
            found: format!("{} theta", theta.len()),
        });
    }
    let ind = ind.unwrap_or(nrows / 2).min(nrows.saturating_sub(1));

    // Single-slice sinogram [1, nang, ncol] for the per-center reconstructions.
    let slc = Tomo::new(
        sino.array
            .index_axis(Axis(0), ind)
            .to_owned()
            .insert_axis(Axis(0)),
        Layout::Sinogram,
    );
    let n = ncol; // reconstruction grid = detector width (tomopy num_gridx = dx)
    let init = init.unwrap_or((ncol / 2) as f32) as f64; // dx // 2

    // Histogram limits from a default-center (`dx/2`) reconstruction, masked
    // (tomopy `_adjust_hist_limits`).
    let mut rec0 = recon_at(&slc, theta, ncol as f32 / 2.0, n, fft)?;
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    {
        let mut img = rec0.index_axis_mut(Axis(0), 0);
        circ_mask_inplace(&mut img);
        for &v in img.iter() {
            mn = mn.min(v);
            mx = mx.max(v);
        }
    }
    let hmin = adjust_hist_min(mn) as f64;
    let hmax = adjust_hist_max(mx) as f64;

    // Entropy of the masked reconstruction at a candidate center (tomopy casts
    // the center to f32 before reconstructing).
    let cost = |center: f64| -> Result<f64> {
        let mut rec = recon_at(&slc, theta, center as f32, n, fft)?;
        let mut img = rec.index_axis_mut(Axis(0), 0);
        circ_mask_inplace(&mut img);
        Ok(entropy64(&img.view(), hmin, hmax))
    };
    let center = nelder_mead_1d(&cost, init, tol as f64, tol as f64)?;
    Ok(center as f32)
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

/// Phase-correlation center from a 0°/180° projection pair (tomopy
/// `rotation.py:391`).
///
/// Registers `proj0` against the mirrored `proj180` by subpixel phase
/// cross-correlation (a port of skimage `phase_cross_correlation` with
/// `normalization="phase"` and `upsample_factor = 1/tol`) and maps the recovered
/// column shift to a rotation center `(ncol + shift_col − 1)/2`. Because it is
/// pure Fourier-domain image registration it never touches a projector, so it
/// matches tomopy numerically. With the default `tol = 0.5` (upsample 2) the
/// shift is quantized to half a pixel, so the center lands on a quarter-pixel
/// grid exactly as tomopy's does.
///
/// `rotc_guess` (tomopy's pre-alignment shift) pre-shifts both projections by
/// `[0, −imgshift]` with `imgshift = rotc_guess − (ncol−1)/2` through a faithful
/// `scipy.ndimage.shift` (order-3 cubic spline, `mode='constant'`, `cval=0`; see
/// `ndimage_shift_spline3_constant`) and adds `imgshift` back to the recovered
/// center, exactly as tomopy `rotation.py:419-435`. With the default `None`,
/// `imgshift == 0`, where scipy's spline shift is the identity to f32 precision,
/// so the round-trip is skipped (bit-for-bit unchanged from before).
pub fn find_center_pc(
    proj0: &Array2<f32>,
    proj180: &Array2<f32>,
    backend: &dyn Backend,
    tol: f32,
    rotc_guess: Option<f32>,
) -> Result<f32> {
    let fft = backend.fft().ok_or_else(|| Error::MissingCapability {
        backend: backend.name(),
        capability: "Fft",
    })?;
    let (nrow, ncol) = proj0.dim();
    if proj180.dim() != (nrow, ncol) {
        return Err(Error::ShapeMismatch {
            expected: format!("{nrow}x{ncol}"),
            found: format!("{}x{}", proj180.dim().0, proj180.dim().1),
        });
    }
    // Pre-alignment shift about the detector midline (tomopy rotation.py:419).
    let imgshift = match rotc_guess {
        Some(g) => g as f64 - (ncol as f64 - 1.0) / 2.0,
        None => 0.0,
    };
    let upsample = 1.0f64 / tol as f64;
    // reference = proj0, moving = fliplr(proj180). When imgshift != 0 both
    // projections are spline-shifted by [0, -imgshift] first (rotation.py:422-423);
    // at imgshift == 0 the spline shift is f32-identity, so use the inputs as-is.
    let shift_col = if imgshift != 0.0 {
        let p0 = ndimage_shift_spline3_constant(proj0, 0.0, -imgshift);
        let p180 = ndimage_shift_spline3_constant(proj180, 0.0, -imgshift);
        let mov = fliplr(&p180);
        phase_cross_correlation(&p0, &mov, upsample, fft)?.1
    } else {
        let mov = fliplr(proj180);
        phase_cross_correlation(proj0, &mov, upsample, fft)?.1
    };
    let center = (ncol as f64 + shift_col - 1.0) / 2.0;
    Ok((center + imgshift) as f32)
}

/// SIFT-feature center detection (tomocupy `find_center.py:99`).
///
/// Available only with the **`sift-center`** feature (it links OpenCV via the
/// `opencv` crate). See [`sift`] for the implementation.
#[cfg(not(feature = "sift-center"))]
pub fn find_center_sift(_proj0: &Array2<f32>, _proj180: &Array2<f32>, _threshold: f32) -> Result<f32> {
    Err(Error::todo(
        "center::find_center_sift (build with the `sift-center` feature)",
        "tomocupy find_center.py:99",
    ))
}

#[cfg(feature = "sift-center")]
pub use sift::{find_center_sift, register_shift_sift};

/// SIFT-feature rotation-center finding (tomocupy `find_center.py`
/// `_register_shift_sift` + the `n//2 - shift_x/2` center formula). Requires the
/// `sift-center` feature.
#[cfg(feature = "sift-center")]
pub mod sift {
    use super::{fliplr, Error, Result};
    use ndarray::Array2;
    use opencv::core::{DMatch, KeyPoint, Mat, Vector, CV_8UC1, NORM_L2};
    use opencv::features2d::{BFMatcher, SIFT};
    use opencv::prelude::*;

    /// numpy `np.histogram(data, 1000)` peak-thresholded robust min/max
    /// (`_find_min_max`): 1000 uniform bins over `[min, max]`, keep bins whose
    /// count exceeds 0.5 % of the peak, return the outermost surviving edges.
    /// Replicates numpy's uniform-bin indexing (including the floating-point
    /// decrement/increment corrections) so the 0–255 normalization is bit-exact.
    fn find_min_max(data: &[f32]) -> (f32, f32) {
        const NBINS: usize = 1000;
        let (mut dmin, mut dmax) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in data {
            if v < dmin {
                dmin = v;
            }
            if v > dmax {
                dmax = v;
            }
        }
        let (first, last) = (dmin as f64, dmax as f64);
        if last <= first {
            return (dmin, dmax);
        }
        // bin_edges = linspace(first, last, NBINS+1) (last forced exact).
        let edges: Vec<f64> = (0..=NBINS)
            .map(|i| {
                if i == NBINS {
                    last
                } else {
                    first + (last - first) * (i as f64) / (NBINS as f64)
                }
            })
            .collect();
        let norm = NBINS as f64 / (last - first);
        let mut h = vec![0u64; NBINS];
        for &v in data {
            let x = v as f64;
            let mut idx = ((x - first) * norm) as i64;
            if idx == NBINS as i64 {
                idx -= 1;
            }
            let mut idx = idx.clamp(0, NBINS as i64 - 1) as usize;
            // numpy corrections (both applied, sequentially — not else-if).
            if x < edges[idx] && idx > 0 {
                idx -= 1;
            }
            if idx != NBINS - 1 && x >= edges[idx + 1] {
                idx += 1;
            }
            h[idx] += 1;
        }
        let hmax = *h.iter().max().unwrap_or(&0);
        let thr = hmax as f64 * 0.005;
        let st = h.iter().position(|&c| c as f64 > thr).unwrap_or(0);
        let end = h.iter().rposition(|&c| c as f64 > thr).unwrap_or(NBINS - 1);
        (edges[st] as f32, edges[end + 1] as f32)
    }

    /// `(img - mmin)/(mmax-mmin)*255`, clipped to 0–255, as a `CV_8UC1` Mat
    /// (numpy f32 arithmetic then `astype(uint8)` truncation).
    fn to_u8_mat(img: &Array2<f32>, mmin: f32, mmax: f32) -> Result<Mat> {
        let (rows, cols) = img.dim();
        let scale = mmax - mmin;
        let mut buf = Vec::with_capacity(rows * cols);
        for &v in img.iter() {
            // numpy clips >255→255 then <0→0; clamp is equivalent for finite v.
            let t = ((v - mmin) / scale * 255.0).clamp(0.0, 255.0);
            buf.push(t as u8);
        }
        let mut m = Mat::new_rows_cols_with_default(
            rows as i32,
            cols as i32,
            CV_8UC1,
            opencv::core::Scalar::all(0.0),
        )
        .map_err(cv_err)?;
        m.data_bytes_mut().map_err(cv_err)?.copy_from_slice(&buf);
        Ok(m)
    }

    fn cv_err(e: opencv::Error) -> Error {
        Error::Backend(format!("opencv: {e}"))
    }

    /// `(img normalized by its own robust min/max) → uint8` (tomocupy's
    /// per-image 0–255 mapping). Exposed for parity testing of the
    /// histogram-based normalization in isolation.
    pub fn normalize_to_u8(img: &Array2<f32>) -> Vec<u8> {
        let (mmin, mmax) = find_min_max(img.as_slice().expect("contiguous image"));
        let scale = mmax - mmin;
        img.iter()
            .map(|&v| ((v - mmin) / scale * 255.0).clamp(0.0, 255.0) as u8)
            .collect()
    }

    /// Per-pair SIFT shift estimate (tomocupy `_register_shift_sift`): SIFT
    /// detect+describe on the normalized `datap2`/`datap1` images, BFMatcher
    /// knn (k=2) with Lowe ratio test (`threshold`), and the mean keypoint
    /// displacement, returned as `[dy, dx]` per pair. min/max for normalization
    /// come from `datap1` (matching upstream). Also returns the number of good
    /// matches in the last pair.
    pub fn register_shift_sift(
        datap1: &[Array2<f32>],
        datap2: &[Array2<f32>],
        threshold: f32,
    ) -> Result<(Array2<f32>, usize)> {
        if datap1.len() != datap2.len() || datap1.is_empty() {
            return Err(Error::InvalidParam(
                "register_shift_sift: need equal, non-empty datap1/datap2".into(),
            ));
        }
        let mut sift = SIFT::create_def().map_err(cv_err)?;
        let mut shifts = Array2::<f32>::zeros((datap1.len(), 2));
        let mut ngood = 0usize;
        for (id, (p1, p2)) in datap1.iter().zip(datap2).enumerate() {
            let (mmin, mmax) = find_min_max(p1.as_slice().expect("contiguous datap1"));
            let tmp1 = to_u8_mat(p2, mmin, mmax)?; // datap2 → query
            let tmp2 = to_u8_mat(p1, mmin, mmax)?; // datap1 → train
            let no_mask = Mat::default();
            let (mut kp1, mut des1) = (Vector::<KeyPoint>::new(), Mat::default());
            let (mut kp2, mut des2) = (Vector::<KeyPoint>::new(), Mat::default());
            sift.detect_and_compute(&tmp1, &no_mask, &mut kp1, &mut des1, false)
                .map_err(cv_err)?;
            sift.detect_and_compute(&tmp2, &no_mask, &mut kp2, &mut des2, false)
                .map_err(cv_err)?;

            let bf = BFMatcher::new(NORM_L2, false).map_err(cv_err)?;
            let mut knn = Vector::<Vector<DMatch>>::new();
            bf.knn_train_match(&des1, &des2, &mut knn, 2, &no_mask, false)
                .map_err(cv_err)?;

            let (mut sum_x, mut sum_y, mut n) = (0.0f64, 0.0f64, 0usize);
            for pair in knn.iter() {
                if pair.len() < 2 {
                    continue;
                }
                let m = pair.get(0).map_err(cv_err)?;
                let nn = pair.get(1).map_err(cv_err)?;
                if m.distance < threshold * nn.distance {
                    let src = kp1.get(m.query_idx as usize).map_err(cv_err)?.pt();
                    let dst = kp2.get(m.train_idx as usize).map_err(cv_err)?.pt();
                    sum_x += (src.x - dst.x) as f64;
                    sum_y += (src.y - dst.y) as f64;
                    n += 1;
                }
            }
            if n == 0 {
                return Err(Error::InvalidParam(format!(
                    "register_shift_sift: no good SIFT matches for pair {id}"
                )));
            }
            ngood = n;
            // np.mean(shift)[::-1] = [mean_y, mean_x].
            shifts[[id, 0]] = (sum_y / n as f64) as f32;
            shifts[[id, 1]] = (sum_x / n as f64) as f32;
        }
        Ok((shifts, ngood))
    }

    /// Rotation center from a 0°/180° projection pair via SIFT feature matching
    /// (tomocupy `find_center_sift`). The 180° projection is flipped left-right
    /// (`data[..., ::-1]`), matched against the 0° projection, and the center is
    /// `ncol/2 - mean_horizontal_shift/2`. `threshold` is the Lowe ratio (0.5
    /// upstream).
    pub fn find_center_sift(
        proj0: &Array2<f32>,
        proj180: &Array2<f32>,
        threshold: f32,
    ) -> Result<f32> {
        if proj0.dim() != proj180.dim() {
            return Err(Error::ShapeMismatch {
                expected: format!("{:?}", proj0.dim()),
                found: format!("{:?}", proj180.dim()),
            });
        }
        let ncol = proj0.dim().1;
        let datap1 = [proj0.clone()];
        let datap2 = [fliplr(proj180)];
        let (shifts, _) = register_shift_sift(&datap1, &datap2, threshold)?;
        Ok(ncol as f32 / 2.0 - shifts[[0, 1]] / 2.0)
    }
}

/// Reconstruct one slice across a range of rotation centers (tomopy
/// `rotation.py:438` `write_center`).
///
/// Helps pick the rotation axis by eye: the `ind`-th sinogram is reconstructed
/// with gridrec at every center in `cen_range = (start, stop, step)` (numpy
/// `arange` semantics — values `start, start+step, …` while `< stop`; default
/// `(ncol/2 − 5, ncol/2 + 5, 0.5)`), optionally circular-masked, and returned as a
/// `[len(centers), n, n]` stack (`n = ncol`) alongside the center values. tomopy
/// writes each slice to `{center:.2f}.tiff`; persist the returned stack the same
/// way (e.g. via `tomoxide-io`) if those files are wanted — this is the
/// I/O-free core so `tomoxide-recon` stays backend/`tomoxide-core`-only.
///
/// Parity scope: only the **center enumeration** is held to tomopy (Δ = 0 — it is
/// pure `np.arange`). The reconstruction *content* goes through tomoxide's
/// gridrec, a gridrec-*family* method (Kaiser–Bessel kernel, ramp weight; see
/// `gridrec.rs`), so the slice pixels are self-consistent gridrec
/// reconstructions, **not** bit-identical to tomopy's PSWF + `parzen` `gridrec`.
///
/// `ind` selects the slice (default `n_rows/2`, tomopy `dy//2`); `mask` applies a
/// `ratio`-scaled circular mask (tomopy `mask`/`ratio`, default `ratio = 1`).
pub fn write_center(
    tomo: &Tomo<f32>,
    theta: &[f32],
    backend: &dyn Backend,
    cen_range: Option<(f32, f32, f32)>,
    ind: Option<usize>,
    mask: bool,
    ratio: f32,
) -> Result<(Vec<f32>, Array3<f32>)> {
    let fft = backend.fft().ok_or_else(|| Error::MissingCapability {
        backend: backend.name(),
        capability: "Fft",
    })?;
    let sino = tomo.to_layout(Layout::Sinogram); // [row, angle, col]
    let (nrows, nang, ncol) = sino.array.dim();
    if theta.len() != nang {
        return Err(Error::ShapeMismatch {
            expected: format!("{nang} angles"),
            found: format!("{} theta", theta.len()),
        });
    }
    if nrows == 0 || nang == 0 || ncol == 0 {
        return Ok((Vec::new(), Array3::zeros((0, ncol, ncol))));
    }
    let ind = ind.unwrap_or(nrows / 2).min(nrows - 1);

    // Center range (numpy `arange`). Default: `arange(ncol/2 − 5, ncol/2 + 5, 0.5)`
    // (tomopy `rotation.py:548`).
    let (start, stop, step) = match cen_range {
        Some((a, b, s)) => (a as f64, b as f64, s as f64),
        None => {
            let half = ncol as f64 / 2.0;
            (half - 5.0, half + 5.0, 0.5)
        }
    };
    let centers = arange(start, stop, step);

    // Reconstruct the same slice at each center (tomopy replicates `tomo[:, ind, :]`
    // into a stack and reconstructs with per-slice centers).
    let slc = Tomo::new(
        sino.array
            .index_axis(Axis(0), ind)
            .to_owned()
            .insert_axis(Axis(0)),
        Layout::Sinogram,
    );
    let n = ncol; // tomopy num_gridx = dx
    let mut stack = Array3::<f32>::zeros((centers.len(), n, n));
    for (m, &c) in centers.iter().enumerate() {
        let mut rec = recon_at(&slc, theta, c as f32, n, fft)?; // [1, n, n]
        if mask {
            let mut img = rec.index_axis_mut(Axis(0), 0);
            circ_mask_inplace_ratio(&mut img, ratio as f64);
        }
        stack
            .index_axis_mut(Axis(0), m)
            .assign(&rec.index_axis(Axis(0), 0));
    }
    let centers: Vec<f32> = centers.iter().map(|&c| c as f32).collect();
    Ok((centers, stack))
}

/// numpy `arange(start, stop, step)` for a positive `step`: length
/// `⌈(stop − start)/step⌉`, value `start + i·step`.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    if step <= 0.0 {
        return Vec::new();
    }
    let len = ((stop - start) / step).ceil();
    let len = if len.is_finite() && len > 0.0 {
        len as usize
    } else {
        0
    };
    (0..len).map(|i| start + i as f64 * step).collect()
}

// ----------------------------------------------------------------------------
// Entropy center-finding internals (private). Mirrors tomopy
// rotation.py:82-202 (`find_center`, `_adjust_hist_limits`, `_find_center_cost`)
// plus scipy's scalar Nelder-Mead simplex (`_minimize_neldermead`).
// ----------------------------------------------------------------------------

/// gridrec a single-slice sinogram at rotation `center` (tomopy default grid =
/// detector width, parallel beam, unit pixel).
fn recon_at(
    sino: &Tomo<f32>,
    theta: &[f32],
    center: f32,
    n: usize,
    fft: &dyn Fft,
) -> Result<Array3<f32>> {
    let geom = Geometry {
        angles: Angles(theta.to_vec()),
        center: Center::Scalar(center),
        beam: Beam::Parallel,
        detector: Detector {
            width: sino.n_cols(),
            height: 1,
            pixel_size: 1.0,
        },
    };
    crate::recon::gridrec::gridrec(sino, &geom, n, fft)
}

/// Apply tomopy's `circ_mask(rec, axis=0, ratio, val=0)` to one reconstruction
/// slice: keep where `x²+y² < (ratio·min(ny,nx)/2)²`, zero outside (tomopy
/// `_get_mask`: `x²+y² < ratio²·r²`, `r = min/2`). Coordinates match
/// `np.ogrid[0.5−n/2 : 0.5+n/2]`, i.e. cell `(i, j)` sits at
/// `(i + 0.5 − ny/2, j + 0.5 − nx/2)`.
fn circ_mask_inplace_ratio(img: &mut ArrayViewMut2<f32>, ratio: f64) {
    let (ny, nx) = img.dim();
    let half_y = ny as f64 / 2.0;
    let half_x = nx as f64 / 2.0;
    let r2 = (ratio * (ny.min(nx) as f64 / 2.0)).powi(2);
    for i in 0..ny {
        let y = i as f64 + 0.5 - half_y;
        for j in 0..nx {
            let x = j as f64 + 0.5 - half_x;
            if x * x + y * y >= r2 {
                img[[i, j]] = 0.0;
            }
        }
    }
}

/// `circ_mask` with tomopy's default `ratio = 1`.
fn circ_mask_inplace(img: &mut ArrayViewMut2<f32>) {
    circ_mask_inplace_ratio(img, 1.0);
}

/// tomopy `_adjust_hist_min`: stretch the lower histogram bound away from zero.
fn adjust_hist_min(v: f32) -> f32 {
    if v < 0.0 {
        2.0 * v
    } else {
        0.5 * v
    }
}

/// tomopy `_adjust_hist_max`: stretch the upper histogram bound away from zero.
fn adjust_hist_max(v: f32) -> f32 {
    if v < 0.0 {
        0.5 * v
    } else {
        2.0 * v
    }
}

/// Shannon entropy of the 64-bin histogram of `img` over `[hmin, hmax]`,
/// `−Σ p·log2(p)` with `p = count/size + 1e-12` (tomopy `_find_center_cost`).
/// `size` counts every pixel (including masked and out-of-range ones), and the
/// `1e-12` floor is added to all 64 bins, exactly as numpy does.
fn entropy64(img: &ndarray::ArrayView2<f32>, hmin: f64, hmax: f64) -> f64 {
    const BINS: usize = 64;
    let size = img.len() as f64;
    let mut counts = [0.0f64; BINS];
    let span = hmax - hmin;
    for &v in img.iter() {
        let v = v as f64;
        if v < hmin || v > hmax {
            continue; // numpy histogram drops out-of-range values
        }
        let mut idx = ((v - hmin) / span * BINS as f64) as usize;
        if idx >= BINS {
            idx = BINS - 1; // v == hmax lands in the last bin
        }
        counts[idx] += 1.0;
    }
    let mut val = 0.0f64;
    for &c in counts.iter() {
        let p = c / size + 1e-12;
        val -= p * p.log2();
    }
    val
}

/// scipy's `_minimize_neldermead` specialised to one variable (`tol` sets both
/// `xatol` and `fatol`). The simplex is `{x0, 1.05·x0}`; reflect/expand/
/// contract/shrink use scipy's `ρ=1, χ=2, ψ=σ=0.5` coefficients. Returns the
/// best vertex when the simplex shrinks within tolerance (default 200 evals).
fn nelder_mead_1d<F>(f: &F, x0: f64, xatol: f64, fatol: f64) -> Result<f64>
where
    F: Fn(f64) -> Result<f64>,
{
    let (rho, chi, psi, sigma) = (1.0, 2.0, 0.5, 0.5);
    let (nonzdelt, zdelt) = (0.05, 0.00025);
    let mut s0 = x0;
    let mut s1 = if x0 != 0.0 {
        (1.0 + nonzdelt) * x0
    } else {
        zdelt
    };
    let mut f0 = f(s0)?;
    let mut f1 = f(s1)?;
    if f1 < f0 {
        std::mem::swap(&mut s0, &mut s1);
        std::mem::swap(&mut f0, &mut f1);
    }
    let (maxiter, maxfun) = (200usize, 200usize);
    let mut fcalls = 2usize;
    let mut iters = 1usize;
    while fcalls < maxfun && iters < maxiter {
        if (s1 - s0).abs() <= xatol && (f0 - f1).abs() <= fatol {
            break;
        }
        let xbar = s0; // centroid of the best vertex (N = 1)
        let xr = (1.0 + rho) * xbar - rho * s1;
        let fxr = f(xr)?;
        fcalls += 1;
        let mut doshrink = false;
        if fxr < f0 {
            // Reflection improved on the best — try to expand.
            let xe = (1.0 + rho * chi) * xbar - rho * chi * s1;
            let fxe = f(xe)?;
            fcalls += 1;
            if fxe < fxr {
                s1 = xe;
                f1 = fxe;
            } else {
                s1 = xr;
                f1 = fxr;
            }
        } else if fxr < f1 {
            // Reflection between best and worst — outside contraction.
            let xc = (1.0 + psi * rho) * xbar - psi * rho * s1;
            let fxc = f(xc)?;
            fcalls += 1;
            if fxc <= fxr {
                s1 = xc;
                f1 = fxc;
            } else {
                doshrink = true;
            }
        } else {
            // Reflection worse than the worst — inside contraction.
            let xcc = (1.0 - psi) * xbar + psi * s1;
            let fxcc = f(xcc)?;
            fcalls += 1;
            if fxcc < f1 {
                s1 = xcc;
                f1 = fxcc;
            } else {
                doshrink = true;
            }
        }
        if doshrink {
            s1 = s0 + sigma * (s1 - s0);
            f1 = f(s1)?;
            fcalls += 1;
        }
        if f1 < f0 {
            std::mem::swap(&mut s0, &mut s1);
            std::mem::swap(&mut f0, &mut f1);
        }
        iters += 1;
    }
    Ok(s0)
}

// ----------------------------------------------------------------------------
// Phase-correlation internals (private). Port of skimage
// `registration/_phase_cross_correlation.py` for the 2-D, real-input,
// `normalization="phase"`, `disambiguate=False` path that tomopy's
// `find_center_pc` uses. Only the registration shift is needed (tomopy discards
// the error/phasediff), so `CCmax`/amplitudes are not computed.
// ----------------------------------------------------------------------------

/// `numpy.fft.fftfreq(n, d)` — sample frequencies for an `n`-point DFT.
fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let nn = n as f64;
    let cut = (n - 1) / 2; // (n-1)//2 — last positive-frequency index
    (0..n)
        .map(|k| {
            let idx = if k <= cut { k as f64 } else { k as f64 - nn };
            idx / (nn * d)
        })
        .collect()
}

/// Register `moving` to `reference` by phase cross-correlation. Returns the
/// `(row, col)` shift; with `upsample_factor > 1` the half-pixel refinement is a
/// direct matrix-multiply upsampled DFT (skimage `_upsampled_dft`).
fn phase_cross_correlation(
    reference: &Array2<f32>,
    moving: &Array2<f32>,
    upsample_factor: f64,
    fft: &dyn Fft,
) -> Result<(f64, f64)> {
    use std::f64::consts::PI;
    let (nrow, ncol) = reference.dim();
    let n = nrow * ncol;

    // Forward 2-D FFTs (unnormalized — matches scipy `fftn`).
    let mut rf = vec![Complex32::new(0.0, 0.0); n];
    let mut mf = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..nrow {
        for j in 0..ncol {
            rf[i * ncol + j] = Complex32::new(reference[[i, j]], 0.0);
            mf[i * ncol + j] = Complex32::new(moving[[i, j]], 0.0);
        }
    }
    fft.fft_2d(&mut rf, nrow, ncol, 1, false)?;
    fft.fft_2d(&mut mf, nrow, ncol, 1, false)?;

    // image_product = rf · conj(mf), phase-normalized: divide by
    // max(|·|, 100·eps) with f32 eps (scipy uses the complex64 real dtype).
    let eps100 = 100.0 * f32::EPSILON as f64;
    let mut pr = vec![0.0f64; n];
    let mut pi = vec![0.0f64; n];
    for k in 0..n {
        let (ar, ai) = (rf[k].re as f64, rf[k].im as f64);
        let (br, bi) = (mf[k].re as f64, mf[k].im as f64);
        let re = ar * br + ai * bi; // a · conj(b)
        let im = ai * br - ar * bi;
        let denom = (re * re + im * im).sqrt().max(eps100);
        pr[k] = re / denom;
        pi[k] = im / denom;
    }

    // Whole-pixel peak: argmax|ifft2(image_product)| (C-order, first max wins).
    let mut cc = vec![Complex32::new(0.0, 0.0); n];
    for k in 0..n {
        cc[k] = Complex32::new(pr[k] as f32, pi[k] as f32);
    }
    fft.fft_2d(&mut cc, nrow, ncol, 1, true)?;
    let (mut best, mut peak) = (-1.0f64, 0usize);
    for (k, c) in cc.iter().enumerate() {
        let m = (c.re as f64).hypot(c.im as f64);
        if m > best {
            best = m;
            peak = k;
        }
    }
    let (r0, c0) = ((peak / ncol) as f64, (peak % ncol) as f64);

    // Wrap to a signed shift about the midpoint `fix(axis/2)`.
    let mut shift_r = if r0 > (nrow as f64 / 2.0).trunc() {
        r0 - nrow as f64
    } else {
        r0
    };
    let mut shift_c = if c0 > (ncol as f64 / 2.0).trunc() {
        c0 - ncol as f64
    } else {
        c0
    };

    if upsample_factor > 1.0 {
        // Refine on an upsampled grid via the matrix-multiply DFT of
        // conj(image_product) — data = (pr, −pi).
        shift_r = (shift_r * upsample_factor).round() / upsample_factor;
        shift_c = (shift_c * upsample_factor).round() / upsample_factor;
        let region_f = (upsample_factor * 1.5).ceil();
        let region = region_f as usize;
        let dftshift = (region_f / 2.0).trunc();
        let off_r = dftshift - shift_r * upsample_factor;
        let off_c = dftshift - shift_c * upsample_factor;
        let freq_c = fftfreq(ncol, upsample_factor);
        let freq_r = fftfreq(nrow, upsample_factor);

        // D1[a][r] = Σ_c exp(−2πi (a−off_c) freq_c[c]) · conj(prod)[r][c].
        let mut d1r = vec![0.0f64; region * nrow];
        let mut d1i = vec![0.0f64; region * nrow];
        for a in 0..region {
            for r in 0..nrow {
                let (mut sre, mut sim) = (0.0f64, 0.0f64);
                for (c, &fc) in freq_c.iter().enumerate() {
                    let ang = -2.0 * PI * (a as f64 - off_c) * fc;
                    let (ks, kc) = ang.sin_cos();
                    let idx = r * ncol + c;
                    let (dre, dim) = (pr[idx], -pi[idx]); // conj(image_product)
                    sre += kc * dre - ks * dim;
                    sim += kc * dim + ks * dre;
                }
                d1r[a * nrow + r] = sre;
                d1i[a * nrow + r] = sim;
            }
        }
        // D2[b][a] = Σ_r exp(−2πi (b−off_r) freq_r[r]) · D1[a][r]; argmax|D2|.
        let (mut bestup, mut mb, mut ma) = (-1.0f64, 0usize, 0usize);
        for b in 0..region {
            for a in 0..region {
                let (mut sre, mut sim) = (0.0f64, 0.0f64);
                for (r, &fr) in freq_r.iter().enumerate() {
                    let ang = -2.0 * PI * (b as f64 - off_r) * fr;
                    let (ks, kc) = ang.sin_cos();
                    let (dre, dim) = (d1r[a * nrow + r], d1i[a * nrow + r]);
                    sre += kc * dre - ks * dim;
                    sim += kc * dim + ks * dre;
                }
                let m = (sre * sre + sim * sim).sqrt();
                if m > bestup {
                    bestup = m;
                    mb = b;
                    ma = a;
                }
            }
        }
        shift_r += (mb as f64 - dftshift) / upsample_factor;
        shift_c += (ma as f64 - dftshift) / upsample_factor;
    }

    // A unit-length axis carries no shift information.
    if nrow == 1 {
        shift_r = 0.0;
    }
    if ncol == 1 {
        shift_c = 0.0;
    }
    Ok((shift_r, shift_c))
}

// ----------------------------------------------------------------------------
// scipy.ndimage.shift (order=3 cubic spline, mode='constant', cval=0) — the
// exact port used by `find_center_pc`'s `rotc_guess` pre-alignment. tomopy
// shifts both projections by `[0, -imgshift]` (rotation.py:422-423) before phase
// correlation, so the spline output feeds straight into the FFT registration and
// must match scipy to the f64 floor — unlike the deliberately-approximate
// `cubic_shift_cols` above (truncated-horizon prefilter, taps dropped at the
// boundary), whose zero-filled boundary columns the Vo metric overwrites.
//
// Mirrors scipy 1.17.1: `_interpolation.py::shift` (prefilter both axes to
// float64 with `spline_filter`, negate the shift, call `zoom_shift`),
// `ni_splines.c::apply_filter`/`_init_causal_mirror`/`_init_anticausal_mirror`/
// `get_spline_interpolation_weights`, and `ni_interpolation.c::NI_ZoomShift`
// (mode='constant' ⇒ an out-of-bounds output centre collapses to `cval`, while
// in-bounds taps are whole-sample mirror-reflected) + `map_coordinate`.
// ----------------------------------------------------------------------------

/// `scipy.ndimage.shift(img, [shift_row, shift_col], order=3, mode='constant',
/// cval=0)` for a 2-D float32 image. The public scipy API negates the shift
/// internally, so output pixel `(i, j)` samples input coordinate
/// `(i − shift_row, j − shift_col)`; out-of-bounds centres yield 0.
fn ndimage_shift_spline3_constant(
    img: &Array2<f32>,
    shift_row: f64,
    shift_col: f64,
) -> Array2<f32> {
    let (nr, nc) = img.dim();
    let mut out = Array2::<f32>::zeros((nr, nc));
    if nr == 0 || nc == 0 {
        return out;
    }
    let coeff = spline_prefilter_2d(img); // float64 B-spline coefficients
                                          // scipy negates the user shift (`shift = [-ii for ii in shift]`).
    let (s_row, s_col) = (-shift_row, -shift_col);
    let (nri, nci) = (nr as isize, nc as isize);
    let (row_hi, col_hi) = (nr as f64 - 1.0, nc as f64 - 1.0);
    for i in 0..nr {
        let cc_row = i as f64 + s_row;
        // map_coordinate(.., CONSTANT): out-of-bounds centre ⇒ cval (0).
        if cc_row < 0.0 || cc_row > row_hi {
            continue; // whole row is cval
        }
        let w_row = spline_weights3(cc_row);
        let start_row = (cc_row.floor() as isize) - 1; // order odd ⇒ floor(cc) − order/2
        let row_taps = [
            mirror_index(start_row, nri),
            mirror_index(start_row + 1, nri),
            mirror_index(start_row + 2, nri),
            mirror_index(start_row + 3, nri),
        ];
        for j in 0..nc {
            let cc_col = j as f64 + s_col;
            if cc_col < 0.0 || cc_col > col_hi {
                continue; // out-of-bounds centre ⇒ cval (0)
            }
            let w_col = spline_weights3(cc_col);
            let start_col = (cc_col.floor() as isize) - 1;
            let col_taps = [
                mirror_index(start_col, nci),
                mirror_index(start_col + 1, nci),
                mirror_index(start_col + 2, nci),
                mirror_index(start_col + 3, nci),
            ];
            // 16-tap separable sum, axis-0 (rows) outer and axis-1 (cols) inner
            // with the `coeff * w_row * w_col` multiply order, matching
            // NI_ZoomShift's fcoordinates enumeration and accumulation.
            let mut t = 0.0f64;
            for a in 0..4 {
                let cr = coeff.row(row_taps[a]);
                let wr = w_row[a];
                for b in 0..4 {
                    t += cr[col_taps[b]] * wr * w_col[b];
                }
            }
            out[[i, j]] = t as f32;
        }
    }
    out
}

/// Prefilter `img` (float32) to cubic B-spline coefficients (float64) over both
/// axes with mirror-boundary initialisation — scipy `spline_filter` runs
/// `spline_filter1d` along axis 0 then axis 1, and for `mode='constant'` the
/// order-3 `apply_filter` uses the mirror init (`ni_splines.c:295-300`).
fn spline_prefilter_2d(img: &Array2<f32>) -> Array2<f64> {
    let (nr, nc) = img.dim();
    let z = 3.0f64.sqrt() - 2.0; // the single order-3 pole √3 − 2
    let mut c = Array2::<f64>::from_shape_fn((nr, nc), |(i, j)| img[[i, j]] as f64);
    // axis 0: filter each column (a line along axis 0). `len == 1` ⇒ no filter.
    if nr > 1 {
        let mut line = vec![0.0f64; nr];
        for j in 0..nc {
            for (i, v) in line.iter_mut().enumerate() {
                *v = c[[i, j]];
            }
            apply_spline_filter_mirror(&mut line, z);
            for (i, &v) in line.iter().enumerate() {
                c[[i, j]] = v;
            }
        }
    }
    // axis 1: filter each row (a line along axis 1).
    if nc > 1 {
        let mut line = vec![0.0f64; nc];
        for i in 0..nr {
            for (j, v) in line.iter_mut().enumerate() {
                *v = c[[i, j]];
            }
            apply_spline_filter_mirror(&mut line, z);
            for (j, &v) in line.iter().enumerate() {
                c[[i, j]] = v;
            }
        }
    }
    c
}

/// scipy `ni_splines.c::apply_filter` for a single order-3 pole with mirror
/// initialisation: gain `(1−z)(1−1/z)` applied first, then one causal/anticausal
/// pass — `_init_causal_mirror`, forward recursion, `_init_anticausal_mirror`,
/// backward recursion. Caller guarantees `len ≥ 2`.
fn apply_spline_filter_mirror(c: &mut [f64], z: f64) {
    let n = c.len();
    // _apply_filter_gain.
    let gain = (1.0 - z) * (1.0 - 1.0 / z);
    for v in c.iter_mut() {
        *v *= gain;
    }
    // _init_causal_mirror: c[0] = (c[0] + Σ_{i} z^i·(c[i] + z^{n-1}·c[n-1-i])) / (1 − z^{2(n-1)}).
    // `powf` (libm `pow`) matches scipy's `pow(z, n-1)` to the last bit; `powi`
    // (exponentiation by squaring) can differ in the final ULP.
    let z_n_1 = z.powf((n - 1) as f64);
    let mut acc = c[0] + z_n_1 * c[n - 1];
    let mut z_i = z;
    for i in 1..n - 1 {
        acc += z_i * (c[i] + z_n_1 * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] = acc / (1.0 - z_n_1 * z_n_1);
    // Forward (causal) recursion.
    for i in 1..n {
        c[i] += z * c[i - 1];
    }
    // _init_anticausal_mirror.
    c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1.0);
    // Backward (anticausal) recursion.
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

/// scipy `ni_splines.c::get_spline_interpolation_weights` for order 3: the cubic
/// B-spline weights at fractional offset `frac(cc)`, for the four taps
/// `floor(cc)−1 … floor(cc)+2`. The last weight is `1 − Σ` (in scipy's order).
fn spline_weights3(cc: f64) -> [f64; 4] {
    let x = cc - cc.floor(); // order is odd ⇒ x -= floor(x)
    let y = x;
    let z = 1.0 - x;
    let mut w = [0.0f64; 4];
    w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
    w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
    w[0] = z * z * z / 6.0;
    w[3] = 1.0 - w[0] - w[1] - w[2];
    w
}

/// scipy `map_coordinate(idx, len, NI_EXTEND_MIRROR)` for an integer tap index:
/// whole-sample reflection (period `2·len−2`, edges not repeated).
fn mirror_index(idx: isize, len: isize) -> usize {
    if len == 1 {
        return 0;
    }
    let mut v = idx;
    if v < 0 {
        let sz2 = 2 * len - 2;
        v = sz2 * ((-v) / sz2) + v;
        v = if v <= 1 - len { v + sz2 } else { -v };
    } else if v > len - 1 {
        let sz2 = 2 * len - 2;
        v -= sz2 * (v / sz2);
        if v >= len {
            v = sz2 - v;
        }
    }
    v as usize
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

#[cfg(test)]
mod tests {
    use super::ndimage_shift_spline3_constant;
    use ndarray::{Array2, Array3};
    use ndarray_npy::read_npy;

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

    /// The cubic-spline `ndimage.shift` port reproduces scipy 1.17.1 bit-for-bit
    /// (order=3, mode='constant', cval=0) across fractional/integer shifts of
    /// both signs, including out-of-bounds (cval) and mirror-tap edge cases.
    #[test]
    fn ndimage_shift_matches_scipy() {
        let input: Array2<f32> = read_npy(format!("{FIXTURES}/ndimage_shift_input.npy")).unwrap();
        let outputs: Array3<f32> =
            read_npy(format!("{FIXTURES}/ndimage_shift_outputs.npy")).unwrap();
        let params: Array2<f64> = read_npy(format!("{FIXTURES}/ndimage_shift_params.npy")).unwrap();
        let ncases = params.dim().0;
        assert_eq!(outputs.dim().0, ncases);

        for k in 0..ncases {
            let (sr, sc) = (params[[k, 0]], params[[k, 1]]);
            let got = ndimage_shift_spline3_constant(&input, sr, sc);
            let want = outputs.index_axis(ndarray::Axis(0), k);
            let mut max_abs = 0.0f32;
            let mut n_mismatch = 0usize;
            for (g, w) in got.iter().zip(want.iter()) {
                let d = (g - w).abs();
                if d > max_abs {
                    max_abs = d;
                }
                if g.to_bits() != w.to_bits() {
                    n_mismatch += 1;
                }
            }
            assert_eq!(
                n_mismatch, 0,
                "case {k} shift=({sr},{sc}): {n_mismatch} f32 bit-mismatches vs scipy, max|Δ|={max_abs:e}"
            );
        }
    }
}
