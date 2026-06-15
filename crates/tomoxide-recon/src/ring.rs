//! Ring-artifact removal on reconstructed slices (ports tomopy
//! `misc/corr.py::remove_ring` + `libtomo/misc/remove_ring.c`, original author
//! Justin Blair). Operates on the reconstructed image (projector-independent),
//! so it matches tomopy numerically. See `docs/PORTING.md` §E.
//!
//! The pipeline per slice: forward polar transform (nearest-pixel lookup with
//! `thresh_min`/`thresh_max` clamping) → radial median filter in three radial
//! bands → subtract + `thresh` thresholding → azimuthal mean filter (WRAP) in
//! three bands → inverse polar transform → subtract the resulting ring image
//! from the original. Only tomopy's default `int_mode="WRAP"` is implemented
//! (the public API exposes no `int_mode`, matching the stub signature).

use tomoxide_core::data::Volume;
use tomoxide_core::error::Result;

// tomopy's literal PI (`libtomo/misc/remove_ring.c:54`). Matching the C `#define`
// keeps the float/double cast chain — and therefore the integer pixel indices
// produced by `iroundf` — identical to tomopy's. `std::f64::consts::PI` would be
// more precise and could round a borderline index the other way.
#[allow(clippy::approx_constant)]
const RING_PI: f64 = 3.14159265359;

/// `iroundf` from the C: `(x != 0) ? floor((double)x + 0.5) : 0`. The argument
/// is the already-`f32`-narrowed sum, promoted to `f64` for the `+ 0.5` exactly
/// as C promotes for `floor`.
fn iroundf(x: f32) -> i32 {
    if x != 0.0 {
        ((x as f64) + 0.5).floor() as i32
    } else {
        0
    }
}

/// `min_distance_to_edge`: the four edge distances are truncated to `int`
/// (toward zero) exactly as the C assigns `float`/`int - float` into `int`.
fn min_distance_to_edge(cx: f32, cy: f32, width: i32, height: i32) -> i32 {
    let d0 = (cx + 1.0) as i32;
    let d1 = (cy + 1.0) as i32;
    let d2 = (width as f32 - cx) as i32;
    let d3 = (height as f32 - cy) as i32;
    d0.min(d1).min(d2).min(d3)
}

/// Forward polar transform (`r_scale = ang_scale = 1`). Returns the polar image
/// (`pol_height` rows × `pol_width` cols, row-major) plus its extents.
fn polar_transform(
    image: &[f32],
    cx: f32,
    cy: f32,
    width: i32,
    height: i32,
    thresh_max: f32,
    thresh_min: f32,
) -> (Vec<f32>, usize, usize) {
    let max_r = min_distance_to_edge(cx, cy, width, height);
    let pol_width = max_r.max(0) as usize; // r_scale = 1
    let pol_height = iroundf((2.0 * RING_PI * max_r as f64) as f32).max(0) as usize;
    let (w, h) = (pol_width, pol_height);
    let dxu = width as usize;
    let mut polar = vec![0.0f32; w * h];

    for row in 0..h {
        // theta depends only on the row; the C recomputes it (and the trig) for
        // every r, but the value is identical — hoist it.
        let theta = ((row as f64) * 2.0 * RING_PI / (h as f64)) as f32;
        let ang = theta as f64 + RING_PI / (h as f64);
        let (ct, st) = (ang.cos(), ang.sin());
        for r in 0..w {
            let fl_x = ((r as f64) * ct) as f32;
            let fl_y = ((r as f64) * st) as f32;
            let x = iroundf(fl_x + cx) as usize;
            let y = iroundf(fl_y + cy) as usize;
            let mut v = image[y * dxu + x];
            if v > thresh_max {
                v = thresh_max;
            } else if v < thresh_min {
                v = thresh_min;
            }
            polar[row * w + r] = v;
        }
    }
    (polar, w, h)
}

/// Inverse polar transform: nearest-neighbour mapping back to Cartesian, zero
/// outside the polar grid.
fn inverse_polar_transform(
    polar: &[f32],
    pol_width: usize,
    pol_height: usize,
    cx: f32,
    cy: f32,
    width: i32,
    height: i32,
) -> Vec<f32> {
    let (w, h) = (width as usize, height as usize);
    let (pw, ph) = (pol_width, pol_height);
    let offset = RING_PI / (ph as f64);
    let mut cart = vec![0.0f32; w * h];

    for row in 0..h {
        for col in 0..w {
            let arg1 = (row as f32 - cy) as f64;
            let arg2 = (col as f32 - cx) as f64 - offset;
            let mut theta = arg1.atan2(arg2);
            if theta < 0.0 {
                theta += 2.0 * RING_PI;
            }
            let pol_row = iroundf((theta * (ph as f64) / (2.0 * RING_PI)) as f32);
            let dyv = row as f32 - cy;
            let dxv = col as f32 - cx;
            let pol_col = iroundf((dyv * dyv + dxv * dxv).sqrt());
            if pol_row < ph as i32 && pol_col < pw as i32 {
                cart[row * w + col] = polar[pol_row as usize * pw + pol_col as usize];
            }
        }
    }
    cart
}

/// Radial (`axis = 'x'`) median filter over the column band `[start_col,
/// end_col]`, window radius `kernel_rad`, writing into `filtered`.
///
/// Reconstructs each window directly: this yields the same multiset — and hence
/// the same median value — as the C rolling filter, given `start_col +
/// kernel_rad < pol_width` (so the initial window never over-reads the
/// contiguous polar block). Negative columns wrap to the opposite angle (the
/// C's `-col` / `±height/2` rule); columns past `pol_width` read 0 (the rolling
/// filter's `next_value = 0` boundary).
fn median_filter_band(
    polar: &[f32],
    filtered: &mut [f32],
    start_col: usize,
    end_col: usize,
    kernel_rad: usize,
    pol_width: usize,
    pol_height: usize,
) {
    let kr = kernel_rad as isize;
    let half_h = pol_height / 2;
    let mut window = vec![0.0f32; 2 * kernel_rad + 1];
    for row in 0..pol_height {
        for col in start_col..=end_col {
            for (idx, n) in (-kr..=kr).enumerate() {
                let c = col as isize + n;
                window[idx] = if c < 0 {
                    let ac = (-c) as usize;
                    let ar = if row < half_h {
                        row + half_h
                    } else {
                        row - half_h
                    };
                    polar[ar * pol_width + ac]
                } else if c as usize >= pol_width {
                    0.0
                } else {
                    polar[row * pol_width + c as usize]
                };
            }
            window.sort_by(|a, b| a.total_cmp(b));
            filtered[row * pol_width + col] = window[kernel_rad];
        }
    }
}

/// Azimuthal mean filter (`int_mode = WRAP`) over the column band `[start_col,
/// end_col]`, window radius `kernel_rad`, writing into `filtered`. The running
/// sum is kept in `f64` (the C uses `long double`, which is `double` on this
/// 64-bit platform) in the C's exact subtract-then-add order. Rows whose source
/// value is exactly 0 stay 0, matching the C's guard.
fn mean_filter_band_wrap(
    src: &[f32],
    filtered: &mut [f32],
    start_col: usize,
    end_col: usize,
    kernel_rad: usize,
    pol_width: usize,
    pol_height: usize,
) {
    let h = pol_height as isize;
    let kr = kernel_rad as isize;
    let num_elems = (2 * kernel_rad + 1) as f64;
    for col in start_col..=end_col {
        let mut sum = 0.0f64;
        for n in -kr..=kr {
            let mut row = n;
            if row < 0 {
                row += h;
            } else if row >= h {
                row -= h;
            }
            sum += src[row as usize * pol_width + col] as f64;
        }
        filtered[col] = (sum / num_elems) as f32;
        let mut previous_sum = sum;
        for row in 1..pol_height {
            let ri = row as isize;
            let mut last_row = (ri - 1) - kr;
            let mut next_row = ri + kr;
            if last_row < 0 {
                last_row += h;
            }
            if next_row >= h {
                next_row -= h;
            }
            let s = previous_sum - src[last_row as usize * pol_width + col] as f64
                + src[next_row as usize * pol_width + col] as f64;
            filtered[row * pol_width + col] = if src[row * pol_width + col] != 0.0 {
                (s / num_elems) as f32
            } else {
                0.0
            };
            previous_sum = s;
        }
    }
}

/// Full ring filter on a polar image: three-band radial median, subtract +
/// threshold, three-band azimuthal mean (WRAP). Overwrites `polar` with the
/// fully filtered result.
fn ring_filter(
    polar: &mut [f32],
    pol_width: usize,
    pol_height: usize,
    threshold: f32,
    m_rad: i32,
    m_azi: i32,
) {
    let (pw, ph) = (pol_width, pol_height);
    let mut filtered = vec![0.0f32; pw * ph];
    // Band boundaries use C integer division: `pol_width/3` and `2*pol_width/3`
    // (the latter is `(2*pw)/3`, which differs from `2*(pw/3)` when `pw % 3 != 0`).
    let b1 = pw / 3;
    let b2 = 2 * pw / 3;

    median_filter_band(
        polar,
        &mut filtered,
        0,
        b1 - 1,
        (m_rad / 3) as usize,
        pw,
        ph,
    );
    median_filter_band(
        polar,
        &mut filtered,
        b1,
        b2 - 1,
        (2 * m_rad / 3) as usize,
        pw,
        ph,
    );
    median_filter_band(polar, &mut filtered, b2, pw - 1, m_rad as usize, pw, ph);

    // Difference image, with the final thresholding.
    for (p, f) in polar.iter_mut().zip(filtered.iter()) {
        *p -= *f;
        if *p > threshold || *p < -threshold {
            *p = 0.0;
        }
    }

    mean_filter_band_wrap(
        polar,
        &mut filtered,
        0,
        b1 - 1,
        (m_azi / 3) as usize,
        pw,
        ph,
    );
    mean_filter_band_wrap(
        polar,
        &mut filtered,
        b1,
        b2 - 1,
        (2 * m_azi / 3) as usize,
        pw,
        ph,
    );
    mean_filter_band_wrap(polar, &mut filtered, b2, pw - 1, m_azi as usize, pw, ph);

    polar.copy_from_slice(&filtered);
}

/// Polar-transform ring removal.
///
/// `thresh`/`thresh_min`/`thresh_max` bound the correction, `rwidth` is the
/// smoothing width, `theta_min` the minimum arc, matching the C signature
/// `remove_ring(rec, center_x, center_y, dx, dy, dz, thresh_max, thresh_min,
/// thresh, theta_min, rwidth, int_mode, istart, iend)` with `int_mode = WRAP`
/// (tomopy's default).
#[allow(clippy::too_many_arguments)]
pub fn remove_ring(
    vol: &mut Volume<f32>,
    center_x: f32,
    center_y: f32,
    thresh: f32,
    thresh_min: f32,
    thresh_max: f32,
    theta_min: i32,
    rwidth: i32,
) -> Result<()> {
    let (dz, dy, dx) = vol.array.dim();
    if dz == 0 || dy == 0 || dx == 0 {
        return Ok(());
    }
    let (width, height) = (dx as i32, dy as i32);
    let m_rad = 2 * rwidth + 1;

    for s in 0..dz {
        // Copy the slice into a contiguous row-major buffer (the C's `image`).
        let mut image = vec![0.0f32; dy * dx];
        for r in 0..dy {
            for c in 0..dx {
                image[r * dx + c] = vol.array[[s, r, c]];
            }
        }

        let (mut polar, pw, ph) = polar_transform(
            &image, center_x, center_y, width, height, thresh_max, thresh_min,
        );
        // Need at least three radial columns and a defined half-height for the
        // band split and the median wrap; smaller polar grids cannot be filtered.
        if pw < 3 || ph < 2 {
            continue;
        }
        let m_azi = ((ph as f64) / 360.0 * (theta_min as f64)).floor() as i32;

        ring_filter(&mut polar, pw, ph, thresh, m_rad, m_azi);
        let ring_image = inverse_polar_transform(&polar, pw, ph, center_x, center_y, width, height);

        for r in 0..dy {
            for c in 0..dx {
                vol.array[[s, r, c]] = image[r * dx + c] - ring_image[r * dx + c];
            }
        }
    }
    Ok(())
}
