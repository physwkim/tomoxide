//! Projection-alignment helpers (ports tomopy `prep/alignment.py`).
//!
//! `scale` is real (bit-exact, Δ=0); the rest of the alignment family
//! (`blur_edges`, `shift_images`, `add_jitter`, `align_seq`/`align_joint`) is
//! unported — see `docs/PORTING.md` §D.

use tomoxide_core::data::Tomo;
use tomoxide_core::error::{Error, Result};

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use tomoxide_core::data::Layout;

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
}
