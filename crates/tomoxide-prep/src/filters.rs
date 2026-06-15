//! Misc filters & corrections (ports tomopy `misc/corr.py` + `libtomo/misc`).
//! `circ_mask`/`remove_nan`/`remove_neg` are real; rank filters route through
//! the backend (stubbed). See `docs/PORTING.md` §E.

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
}
