//! Test phantoms (ports tomopy `misc/phantom.py`). `shepp2d` is a real
//! analytic rasterizer; the rest are stubs.

use ndarray::Array2;
use tomoxide_core::data::Slice2D;
use tomoxide_core::error::{Error, Result};

/// One ellipse: `(intensity, a, b, x0, y0, angle_deg)` in normalized `[-1,1]`
/// coordinates. These are the modified (Toft) Shepp-Logan parameters.
const SHEPP_ELLIPSES: [(f32, f32, f32, f32, f32, f32); 10] = [
    (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
    (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0.0),
    (-0.2, 0.11, 0.31, 0.22, 0.0, -18.0),
    (-0.2, 0.16, 0.41, -0.22, 0.0, 18.0),
    (0.1, 0.21, 0.25, 0.0, 0.35, 0.0),
    (0.1, 0.046, 0.046, 0.0, 0.1, 0.0),
    (0.1, 0.046, 0.046, 0.0, -0.1, 0.0),
    (0.1, 0.046, 0.023, -0.08, -0.605, 0.0),
    (0.1, 0.023, 0.023, 0.0, -0.606, 0.0),
    (0.1, 0.023, 0.046, 0.06, -0.605, 0.0),
];

/// A `size × size` modified Shepp-Logan head phantom (tomopy `phantom.py:246`).
pub fn shepp2d(size: usize) -> Result<Slice2D<f32>> {
    if size == 0 {
        return Err(Error::InvalidParam("phantom size must be > 0".into()));
    }
    let mut img = Array2::<f32>::zeros((size, size));
    let half = (size as f32 - 1.0) / 2.0;
    for ((iy, ix), px) in img.indexed_iter_mut() {
        // Map pixel to [-1, 1]; image row 0 is the top (y = +1).
        let x = (ix as f32 - half) / half;
        let y = (half - iy as f32) / half;
        let mut val = 0.0f32;
        for &(intensity, a, b, x0, y0, angle_deg) in &SHEPP_ELLIPSES {
            let phi = angle_deg.to_radians();
            let (s, c) = phi.sin_cos();
            let xc = x - x0;
            let yc = y - y0;
            let xr = c * xc + s * yc;
            let yr = -s * xc + c * yc;
            if (xr * xr) / (a * a) + (yr * yr) / (b * b) <= 1.0 {
                val += intensity;
            }
        }
        *px = val;
    }
    Ok(img)
}
