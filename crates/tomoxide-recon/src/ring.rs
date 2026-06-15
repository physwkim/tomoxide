//! Ring-artifact removal on reconstructed slices (ports tomopy
//! `misc/corr.py::remove_ring` + `libtomo/misc/remove_ring.c`). Stub in this
//! scaffold; see `docs/PORTING.md` §E.

use tomoxide_core::data::Volume;
use tomoxide_core::error::{Error, Result};

/// Polar-transform ring removal.
///
/// `thresh`/`thresh_min`/`thresh_max` bound the correction, `rwidth` is the
/// smoothing width, `theta_min` the minimum arc, matching the C signature
/// `remove_ring(rec, center_x, center_y, dx, dy, dz, thresh_max, thresh_min,
/// thresh, theta_min, rwidth, int_mode, istart, iend)`.
#[allow(clippy::too_many_arguments)]
pub fn remove_ring(
    _vol: &mut Volume<f32>,
    _center_x: f32,
    _center_y: f32,
    _thresh: f32,
    _thresh_min: f32,
    _thresh_max: f32,
    _theta_min: i32,
    _rwidth: i32,
) -> Result<()> {
    Err(Error::todo(
        "ring::remove_ring",
        "tomopy libtomo/misc/remove_ring.c; misc/corr.py:751",
    ))
}
