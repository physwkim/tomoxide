//! Projection-alignment helpers (ports tomopy `prep/alignment.py`).
//!
//! `scale` and `blur_edges` are real (bit-exact, Δ=0); the rest of the alignment
//! family (`shift_images`, `add_jitter`, `align_seq`/`align_joint`) is unported —
//! see `docs/PORTING.md` §D. [`warp_frames`] and [`move_frames`] are not tomopy
//! ports: the first resamples each frame by its own affine map, the second by a
//! measured per-frame shift and a detector roll, both in one resampling.

use rayon::prelude::*;

use crate::data::{Layout, Tomo};
use crate::error::{Error, Result};
use crate::recon::lprec::{bspline_weights, convert_to_coeffs};

/// Edge-replicated margin the spline prefilter runs over around a frame
/// (scipy's `npad` for mode `nearest`): its boundary term has decayed by
/// `|pole|^12 ≈ 1.4e-7` where the frame starts.
pub(crate) const SPLINE_PAD: usize = 12;

/// Linearly scale a projection stack into `[−1, 1]` by its peak magnitude
/// (tomopy `prep/alignment.py:460` `scale`). Divides every pixel by
/// `scl = max(|max|, |min|)` in place and returns `scl` (needed to invert the
/// scaling later).
///
/// The scale factor is pure order statistics (`max`/`min`/`abs`) and the divide
/// is an elementwise f32 operation, so the result is bit-exact (Δ=0) vs tomopy.
/// Matches tomopy's lack of a zero guard: an all-zero stack gives `scl = 0` and
/// NaN pixels, exactly as numpy's `prj /= 0`. Errors only on an empty stack
/// (tomopy's `prj.max()` raises there).
pub fn scale(data: &mut Tomo<f32>) -> Result<f32> {
    let arr = &mut data.array;
    if arr.is_empty() {
        return Err(Error::InvalidParam("scale: empty projection stack".into()));
    }
    let mx = arr.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mn = arr.iter().copied().fold(f32::INFINITY, f32::min);
    // tomopy: scl = max(abs(prj.max()), abs(prj.min())).
    let scl = mx.abs().max(mn.abs());
    arr.mapv_inplace(|v| v / scl);
    Ok(scl)
}

/// Feather the edges of every projection image by a radial mask (tomopy
/// `prep/alignment.py:482` `blur_edges`). Within each projection (axes
/// `[row, col]`), `rad = √((row − rows/2)² + (col − cols/2)²)`; the mask is `1`
/// where `rad < low·rad_max`, `0` where `rad > high·rad_max`, and a linear ramp
/// `(rmax − rad)/(rmax − rmin)` in between (`rmin = low·rad_max`,
/// `rmax = high·rad_max`). Every projection is multiplied by this mask. Defaults
/// upstream are `low = 0`, `high = 0.8`.
///
/// `√` is IEEE-correctly-rounded (matches numpy bit-for-bit) and the rest of the
/// mask is plain f64 arithmetic in numpy's order; the final `prj *= mask` is the
/// f64 product cast to f32 (numpy's in-place `float32 *= float64`), so the result
/// is **bit-exact (Δ=0)** vs tomopy. The mask is built by the same *sequential*
/// assignment as upstream, so even a degenerate `low > high` matches.
pub fn blur_edges(data: &mut Tomo<f32>, low: f64, high: f64) -> Result<()> {
    let target = data.layout;
    let mut proj = data.to_layout(Layout::Projection);
    let (dx, dy, dz) = proj.array.dim();
    if dy == 0 || dz == 0 {
        *data = proj.to_layout(target);
        return Ok(());
    }
    // rad[row, col] = sqrt((row - dy/2)^2 + (col - dz/2)^2); centre is dy/2, dz/2.
    let (cy, cz) = (dy as f64 / 2.0, dz as f64 / 2.0);
    let mut rad = vec![0.0f64; dy * dz];
    let mut rad_max = f64::NEG_INFINITY;
    for row in 0..dy {
        let dr = row as f64 - cy;
        for col in 0..dz {
            let dc = col as f64 - cz;
            let r = (dr * dr + dc * dc).sqrt();
            rad[row * dz + col] = r;
            if r > rad_max {
                rad_max = r;
            }
        }
    }
    let (rmin, rmax) = (low * rad_max, high * rad_max);
    // numpy builds the mask by sequential assignment (order matters if low>high):
    // zeros; rad<rmin -> 1; rad>rmax -> 0; rmin<=rad<=rmax -> (rmax-rad)/(rmax-rmin).
    let mut mask = vec![0.0f64; dy * dz];
    for (idx, &r) in rad.iter().enumerate() {
        let mut m = 0.0f64;
        if r < rmin {
            m = 1.0;
        }
        if r > rmax {
            m = 0.0;
        }
        if r >= rmin && r <= rmax {
            m = (rmax - r) / (rmax - rmin);
        }
        mask[idx] = m;
    }
    // _prj *= mask, broadcast across the projection axis; float32 *= float64 is
    // computed in f64 then cast back to f32.
    for i in 0..dx {
        for row in 0..dy {
            for col in 0..dz {
                let m = mask[row * dz + col];
                proj.array[[i, row, col]] = (proj.array[[i, row, col]] as f64 * m) as f32;
            }
        }
    }
    *data = proj.to_layout(target);
    Ok(())
}

/// One frame's resampling map on `(row, col)`: the output at `o` reads the
/// input at `a·o + t`, as `scipy.ndimage.affine_transform(matrix=a, offset=t)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameAffine {
    pub a: [[f64; 2]; 2],
    pub t: [f64; 2],
}

/// Resample every projection by its own map, one cubic-B-spline resampling per
/// frame: `out[o] = in[a·o + t]` on `(row, col)` with `maps[k]`'s `a` and `t`.
/// Off the frame it reads the spline of the frame extended by its edges.
///
/// This is scipy's `affine_transform(p, a, t, order=3, mode='nearest')`.
pub fn warp_frames(data: &mut Tomo<f32>, maps: &[FrameAffine]) -> Result<()> {
    let nproj = data.n_angles();
    if maps.len() != nproj {
        return Err(Error::InvalidParam(format!(
            "warp_frames: {} maps for {nproj} projections",
            maps.len()
        )));
    }
    if maps
        .iter()
        .any(|m| m.a.iter().flatten().chain(&m.t).any(|v| !v.is_finite()))
    {
        return Err(Error::InvalidParam("warp_frames: non-finite map".into()));
    }
    let (rows, cols) = (data.n_rows(), data.n_cols());
    if nproj == 0 || rows == 0 || cols == 0 {
        return Ok(());
    }
    let target = data.layout;
    if target != Layout::Projection || !data.array.is_standard_layout() {
        *data = data.to_layout(Layout::Projection);
    }
    let (h, w) = (rows + 2 * SPLINE_PAD, cols + 2 * SPLINE_PAD);
    let frames = data.array.as_slice_mut().expect("standard layout");
    frames
        .par_chunks_mut(rows * cols)
        .zip(maps.par_iter())
        .for_each_init(
            || (vec![0.0f32; h * w], vec![0.0f32; h]),
            |(coef, line), (frame, m)| {
                spline_coeffs(frame, rows, cols, coef, line);
                let [[a00, a01], [a10, a11]] = m.a;
                for (i, out) in frame.chunks_exact_mut(cols).enumerate() {
                    let (r0, c0) = (a00 * i as f64 + m.t[0], a10 * i as f64 + m.t[1]);
                    for (j, v) in out.iter_mut().enumerate() {
                        let j = j as f64;
                        *v = spline_at(coef, h, w, r0 + a01 * j, c0 + a11 * j);
                    }
                }
            },
        );
    if target != Layout::Projection {
        *data = data.to_layout(target);
    }
    Ok(())
}

/// Move every projection by its own shift and one common roll, in a single
/// cubic-B-spline resampling per frame: `out[o] = in[M·(o + d − c) + c]` on
/// `(row, col)`, with `d = (dy[k], dx[k])`, `c = ((rows−1)/2, (cols−1)/2)` and
/// `M = [[cos r, sin r], [−sin r, cos r]]` for `r = roll_deg`. What sat at
/// `o + d` in the rolled frame lands at `o`, so a measured displacement `d` is
/// removed. Off the frame it reads the spline of the frame extended by its
/// edges, as scipy does.
///
/// This is scipy's `shift(rotate(p, roll_deg, reshape=False), (−dy, −dx))`
/// with `order=3, mode='nearest'`, made as one resampling instead of two: the
/// [`warp_frames`] map `a = M`, `t = M·(d − c) + c`.
pub fn move_frames(data: &mut Tomo<f32>, dy: &[f32], dx: &[f32], roll_deg: f32) -> Result<()> {
    let nproj = data.n_angles();
    if dy.len() != nproj || dx.len() != nproj {
        return Err(Error::InvalidParam(format!(
            "move_frames: {} dy and {} dx for {nproj} projections",
            dy.len(),
            dx.len()
        )));
    }
    if !roll_deg.is_finite() || dy.iter().chain(dx).any(|v| !v.is_finite()) {
        return Err(Error::InvalidParam(
            "move_frames: non-finite shift or roll".into(),
        ));
    }
    let (sn, cs) = (roll_deg as f64).to_radians().sin_cos();
    let (rows, cols) = (data.n_rows(), data.n_cols());
    let (cy, cx) = (
        rows.saturating_sub(1) as f64 / 2.0,
        cols.saturating_sub(1) as f64 / 2.0,
    );
    let maps: Vec<FrameAffine> = dy
        .iter()
        .zip(dx)
        .map(|(&dy, &dx)| {
            let (a, b) = (dy as f64 - cy, dx as f64 - cx);
            FrameAffine {
                a: [[cs, sn], [-sn, cs]],
                t: [cs * a + sn * b + cy, cs * b - sn * a + cx],
            }
        })
        .collect();
    warp_frames(data, &maps)
}

/// The cubic-B-spline coefficients of `frame` (`rows × cols`) padded by
/// [`SPLINE_PAD`] edge-replicated samples on every side, into `coef`
/// (`(rows + 2·pad) × (cols + 2·pad)`); `line` holds one padded column.
pub(crate) fn spline_coeffs(
    frame: &[f32],
    rows: usize,
    cols: usize,
    coef: &mut [f32],
    line: &mut [f32],
) {
    let w = cols + 2 * SPLINE_PAD;
    for (r, out) in coef.chunks_exact_mut(w).enumerate() {
        let src = &frame[r.saturating_sub(SPLINE_PAD).min(rows - 1) * cols..][..cols];
        out[..SPLINE_PAD].fill(src[0]);
        out[SPLINE_PAD..SPLINE_PAD + cols].copy_from_slice(src);
        out[SPLINE_PAD + cols..].fill(src[cols - 1]);
        convert_to_coeffs(out);
    }
    for c in 0..w {
        for (r, v) in line.iter_mut().enumerate() {
            *v = coef[r * w + c];
        }
        convert_to_coeffs(line);
        for (r, v) in line.iter().enumerate() {
            coef[r * w + c] = *v;
        }
    }
}

/// The spline of padded coefficients `coef` (`h × w`) at frame position
/// `(r, c)`. scipy neither clamps the position nor stops at the padding: a
/// tap past it reads the padding's edge coefficient.
fn spline_at(coef: &[f32], h: usize, w: usize, r: f64, c: f64) -> f32 {
    let (r, c) = (r + SPLINE_PAD as f64, c + SPLINE_PAD as f64);
    let (fr, fc) = (r.floor(), c.floor());
    let wr = bspline_weights((r - fr) as f32);
    let wc = bspline_weights((c - fc) as f32);
    let tap = |f: f64, k: isize, n: usize| (f as isize - 1 + k).clamp(0, n as isize - 1) as usize;
    let cols = [0, 1, 2, 3].map(|k| tap(fc, k, w));
    let mut sum = 0.0f32;
    for (k, &wk) in wr.iter().enumerate() {
        let row = &coef[tap(fr, k as isize, h) * w..][..w];
        sum += wk
            * (wc[0] * row[cols[0]]
                + wc[1] * row[cols[1]]
                + wc[2] * row[cols[2]]
                + wc[3] * row[cols[3]]);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Layout;
    use ndarray::Array3;

    #[test]
    fn scale_normalizes_by_peak_magnitude() {
        // Negative-dominated stack: |min| = 4 is the peak magnitude.
        let arr = Array3::from_shape_vec((1, 1, 4), vec![2.0f32, -4.0, 1.0, -3.0]).unwrap();
        let mut t = Tomo::new(arr, Layout::Projection);
        let scl = scale(&mut t).unwrap();
        assert_eq!(scl, 4.0);
        assert_eq!(t.array.as_slice().unwrap(), &[0.5, -1.0, 0.25, -0.75]);
    }

    #[test]
    fn scale_rejects_empty() {
        let mut t = Tomo::new(Array3::<f32>::zeros((0, 0, 0)), Layout::Projection);
        assert!(matches!(scale(&mut t), Err(Error::InvalidParam(_))));
    }

    #[test]
    fn blur_edges_feathers_radially() {
        // One 4×4 projection of ones. centre (2,2) has rad=0 → mask 1 (kept);
        // corner (0,0) has rad=√8 > 0.8·rad_max → mask 0 (blurred away); an
        // edge-midpoint is a partial feather in (0,1).
        let arr = Array3::from_elem((1, 4, 4), 1.0f32);
        let mut t = Tomo::new(arr, Layout::Projection);
        blur_edges(&mut t, 0.0, 0.8).unwrap();
        assert_eq!(t.array[[0, 2, 2]], 1.0, "centre not kept");
        assert_eq!(t.array[[0, 0, 0]], 0.0, "corner not blurred to 0");
        let mid = t.array[[0, 0, 2]];
        assert!(mid > 0.0 && mid < 1.0, "edge pixel {mid} not feathered");
    }

    /// A frame with no smooth structure, so a sample read from the wrong
    /// pixel cannot pass for the right one.
    fn rough(n: usize, rows: usize, cols: usize) -> Array3<f32> {
        Array3::from_shape_fn((n, rows, cols), |(k, r, c)| {
            ((k * 5 + r * 7 + c * 13) % 11) as f32
        })
    }

    /// A whole-pixel move lands every sample on a node, where the spline
    /// equals the data; the edge reads past the frame as the nearest pixel.
    /// A zero move is the identity.
    #[test]
    fn move_frames_moves_by_whole_pixels() {
        let (rows, cols) = (16, 20);
        let src = rough(3, rows, cols);
        let mut t = Tomo::new(src.clone(), Layout::Projection);
        let (dy, dx) = ([2.0, -1.0, 0.0], [-3.0, 0.0, 0.0]);
        move_frames(&mut t, &dy, &dx, 0.0).unwrap();
        for ((k, i, j), &v) in t.array.indexed_iter() {
            let si = (i as isize + dy[k] as isize).clamp(0, rows as isize - 1) as usize;
            let sj = (j as isize + dx[k] as isize).clamp(0, cols as isize - 1) as usize;
            let want = src[[k, si, sj]];
            assert!((v - want).abs() < 1e-4, "[{k},{i},{j}] {v} want {want}");
        }
    }

    /// Past the edge it reads what scipy 1.13.1's
    /// `shift(f, (−0.4, 2.7), order=3, mode='nearest')` reads.
    #[test]
    fn move_frames_reads_past_the_edge_as_scipy_shift() {
        let src = Array3::from_shape_fn((1, 4, 6), |(_, r, c)| {
            ((3 * r + 7 * c) % 5) as f32 + 0.25 * r as f32
        });
        let mut t = Tomo::new(src, Layout::Projection);
        move_frames(&mut t, &[0.4], &[-2.7], 0.0).unwrap();
        #[rustfmt::skip]
        let want = [
            1.472188, 1.445359, 1.545486, 1.159575, 1.725208, 3.519763,
            2.657169, 2.63813, 2.709184, 2.411628, 1.016459, 2.02057,
            2.455581, 2.480741, 2.386841, 2.758071, 2.931692, 1.194648,
            5.197221, 5.093313, 5.481104, 3.933387, 2.036161, 3.506029,
        ];
        for (k, (&v, &w)) in t.array.iter().zip(&want).enumerate() {
            assert!((v - w).abs() < 1e-5, "[{}][{}] {v} want {w}", k / 6, k % 6);
        }
    }

    /// A 90° roll of a square frame is `np.rot90`, which scipy's
    /// `rotate(p, 90, reshape=False)` also gives: `out[i][j] = in[j][n−1−i]`.
    #[test]
    fn move_frames_rolls_a_quarter_turn_as_scipy_rotate() {
        let n = 12;
        let src = rough(1, n, n);
        let mut t = Tomo::new(src.clone(), Layout::Projection);
        move_frames(&mut t, &[0.0], &[0.0], 90.0).unwrap();
        for ((_, i, j), &v) in t.array.indexed_iter() {
            let want = src[[0, j, n - 1 - i]];
            assert!((v - want).abs() < 1e-4, "[{i},{j}] {v} want {want}");
        }
    }

    /// A cubic spline reproduces a linear ramp away from the edge, so a
    /// sub-pixel shift under an odd roll reads the ramp exactly where
    /// `M·(o + d − c) + c` says, on a non-square frame in sinogram layout.
    #[test]
    fn move_frames_samples_at_the_rolled_and_shifted_position() {
        let (rows, cols) = (30, 40);
        let ramp = |r: f64, c: f64| 0.5 * r + 0.25 * c;
        let src =
            Array3::from_shape_fn((1, rows, cols), |(_, r, c)| ramp(r as f64, c as f64) as f32);
        let mut t = Tomo::new(src, Layout::Projection).to_layout(Layout::Sinogram);
        let (dy, dx, roll) = (0.5f64, -1.25f64, 30.0f64);
        move_frames(&mut t, &[dy as f32], &[dx as f32], roll as f32).unwrap();
        assert_eq!(t.layout, Layout::Sinogram);
        let (sn, cs) = roll.to_radians().sin_cos();
        let (cy, cx) = ((rows - 1) as f64 / 2.0, (cols - 1) as f64 / 2.0);
        let mut checked = 0;
        for ((i, _, j), &v) in t.array.indexed_iter() {
            let (a, b) = (i as f64 + dy - cy, j as f64 + dx - cx);
            let (r, c) = (cs * a + sn * b + cy, cs * b - sn * a + cx);
            if r < 4.0 || c < 4.0 || r > (rows - 5) as f64 || c > (cols - 5) as f64 {
                continue;
            }
            let want = ramp(r, c) as f32;
            assert!((v - want).abs() < 1e-3, "[{i},{j}] {v} want {want}");
            checked += 1;
        }
        assert!(checked > 200, "only {checked} interior samples");
    }

    /// Under a map with scale, shear and a sub-pixel offset it reads what
    /// scipy 1.13.1's `affine_transform(f, [[0.9, 0.1], [−0.2, 1.1]],
    /// (0.3, −0.7), order=3, mode='nearest')` reads, past the edge too.
    #[test]
    fn warp_frames_reads_as_scipy_affine_transform() {
        let src = Array3::from_shape_fn((1, 4, 6), |(_, r, c)| {
            ((3 * r + 7 * c) % 5) as f32 + 0.25 * r as f32
        });
        let mut t = Tomo::new(src, Layout::Projection);
        let map = FrameAffine {
            a: [[0.9, 0.1], [-0.2, 1.1]],
            t: [0.3, -0.7],
        };
        warp_frames(&mut t, &[map]).unwrap();
        #[rustfmt::skip]
        let want = [
            1.080359, 1.050139, 1.991373, 3.632939, 1.759985, 2.545119,
            3.153204, 2.626173, 1.016459, 2.002381, 3.929068, 2.499835,
            1.64036, 1.808876, 3.331622, 1.304466, 2.108485, 3.86987,
            4.644784, 5.325908, 1.588449, 4.123327, 0.4809509, 3.176043,
        ];
        for (k, (&v, &w)) in t.array.iter().zip(&want).enumerate() {
            assert!((v - w).abs() < 1e-5, "[{}][{}] {v} want {w}", k / 6, k % 6);
        }
    }

    #[test]
    fn warp_frames_rejects_a_map_count_mismatch_or_a_non_finite_map() {
        let mut t = Tomo::new(rough(2, 4, 4), Layout::Projection);
        let id = FrameAffine {
            a: [[1.0, 0.0], [0.0, 1.0]],
            t: [0.0, 0.0],
        };
        let r = warp_frames(&mut t, &[id]);
        assert!(matches!(r, Err(Error::InvalidParam(_))));
        let nan = FrameAffine {
            t: [f64::NAN, 0.0],
            ..id
        };
        let r = warp_frames(&mut t, &[id, nan]);
        assert!(matches!(r, Err(Error::InvalidParam(_))));
    }

    #[test]
    fn move_frames_rejects_a_shift_per_projection_mismatch() {
        let mut t = Tomo::new(rough(2, 4, 4), Layout::Projection);
        let r = move_frames(&mut t, &[0.0], &[0.0, 0.0], 0.0);
        assert!(matches!(r, Err(Error::InvalidParam(_))));
    }
}
