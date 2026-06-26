//! Projection-alignment helpers (ports tomopy `prep/alignment.py`).
//!
//! `scale` and `blur_edges` are real (bit-exact, Δ=0); the rest of the alignment
//! family (`shift_images`, `add_jitter`, `align_seq`/`align_joint`) is unported —
//! see `docs/PORTING.md` §D.

use crate::data::{Layout, Tomo};
use crate::error::{Error, Result};

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
}
