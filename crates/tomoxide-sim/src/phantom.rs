//! Test phantoms (ports tomopy `misc/phantom.py`). `shepp2d` is a real
//! analytic rasterizer; `shepp3d` is a faithful f64 parity port of the 3-D
//! ellipsoid phantom; the rest are stubs.

use ndarray::{Array2, Array3};
use tomoxide_core::data::{Slice2D, Volume};
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

/// Modified 3-D Shepp-Logan ellipsoid parameters (tomopy
/// `phantom.py:_get_shepp_array`): each row is
/// `[A, a, b, c, x0, y0, z0, phi, theta, psi]` — amplitude, the three semi-axes,
/// the centre, and the Euler angles (degrees).
#[rustfmt::skip]
const SHEPP3D_ELLIPSOIDS: [[f64; 10]; 10] = [
    [ 1.0,  0.6900, 0.920, 0.810,  0.0,    0.0,    0.0,    90.0, 90.0,  90.0],
    [-0.8,  0.6624, 0.874, 0.780,  0.0,   -0.0184, 0.0,    90.0, 90.0,  90.0],
    [-0.2,  0.1100, 0.310, 0.220,  0.22,   0.0,    0.0,  -108.0, 90.0, 100.0],
    [-0.2,  0.1600, 0.410, 0.280, -0.22,   0.0,    0.0,   108.0, 90.0, 100.0],
    [ 0.1,  0.2100, 0.250, 0.410,  0.0,    0.35,  -0.15,   90.0, 90.0,  90.0],
    [ 0.1,  0.0460, 0.046, 0.050,  0.0,    0.1,    0.25,   90.0, 90.0,  90.0],
    [ 0.1,  0.0460, 0.046, 0.050,  0.0,   -0.1,    0.25,   90.0, 90.0,  90.0],
    [ 0.1,  0.0460, 0.023, 0.050, -0.08,  -0.605,  0.0,    90.0, 90.0,  90.0],
    [ 0.1,  0.0230, 0.023, 0.020,  0.0,   -0.606,  0.0,    90.0, 90.0,  90.0],
    [ 0.1,  0.0230, 0.046, 0.020,  0.06,  -0.605,  0.0,    90.0, 90.0,  90.0],
];

/// Euler rotation matrix from `phi`/`theta`/`psi` (degrees), tomopy
/// `phantom.py:_rotation_matrix`. Each trig term is `sin`/`cos` of `to_radians`
/// (== numpy's `np.radians`, verified bit-exact) called separately (not
/// `sin_cos`, to match numpy's distinct `np.sin`/`np.cos` scalar libm calls).
fn euler_matrix(phi: f64, theta: f64, psi: f64) -> [[f64; 3]; 3] {
    let (rphi, rtheta, rpsi) = (phi.to_radians(), theta.to_radians(), psi.to_radians());
    let (cphi, sphi) = (rphi.cos(), rphi.sin());
    let (ctheta, stheta) = (rtheta.cos(), rtheta.sin());
    let (cpsi, spsi) = (rpsi.cos(), rpsi.sin());
    [
        [
            cpsi * cphi - ctheta * sphi * spsi,
            cpsi * sphi + ctheta * cphi * spsi,
            spsi * stheta,
        ],
        [
            -spsi * cphi - ctheta * sphi * cpsi,
            -spsi * sphi + ctheta * cphi * cpsi,
            cpsi * stheta,
        ],
        [stheta * sphi, -stheta * cphi, ctheta],
    ]
}

/// A `size³` modified 3-D Shepp-Logan phantom (tomopy `phantom.py:284`
/// `shepp3d`), then `clip(0, +∞)`.
///
/// Faithful f64 parity port of tomopy's ellipsoid rasterizer: the cube is
/// sampled on `np.mgrid[-1:1:size·j]` per axis (`= arange(size)·step − 1`,
/// `step = 2/(size−1)`), each ellipsoid rotates the coords by an Euler matrix
/// (`np.tensordot`), shifts by its centre, scales by its semi-axes, and a voxel
/// is inside when `Σ((R·r − c)/s)² ≤ 1`. Inside voxels accumulate the amplitude
/// `A` in f32 (the f64 add cast back to f32 after each ellipsoid, matching
/// numpy's in-place `obj[mask] += A`).
///
/// All coordinate math is f64 with libm `sin`/`cos` (bit-exact vs numpy's scalar
/// trig) and a fixed accumulation order, so the inclusion mask matches tomopy
/// **bit-for-bit (Δ=0)**: the only non-reproduced step is numpy's BLAS dot order
/// in `tensordot`, which differs by ≤1 ULP but never flips a voxel because none
/// land within f64-ULP of the `≤ 1` boundary (verified by the parity test).
pub fn shepp3d(size: usize) -> Result<Volume<f32>> {
    if size == 0 {
        return Err(Error::InvalidParam("phantom size must be > 0".into()));
    }
    let n = size;
    // np.mgrid[-1:1:n·j] = arange(n)·step + (-1); n == 1 collapses to [-1].
    let lin: Vec<f64> = if n == 1 {
        vec![-1.0]
    } else {
        let step = 2.0 / ((n - 1) as f64);
        (0..n).map(|i| (i as f64) * step - 1.0).collect()
    };
    let mut obj = Array3::<f32>::zeros((n, n, n));
    for ell in &SHEPP3D_ELLIPSOIDS {
        let amp = ell[0];
        let (a, b, c) = (ell[1], ell[2], ell[3]);
        let (x0, y0, z0) = (ell[4], ell[5], ell[6]);
        let alpha = euler_matrix(ell[7], ell[8], ell[9]);
        for i in 0..n {
            let x = lin[i];
            for j in 0..n {
                let y = lin[j];
                for k in 0..n {
                    let z = lin[k];
                    // tensordot row m = alpha[m,0]·x + alpha[m,1]·y + alpha[m,2]·z,
                    // then −centre, ÷semi-axis, square, sum over m (numpy order).
                    let u0 = ((alpha[0][0] * x + alpha[0][1] * y) + alpha[0][2] * z - x0) / a;
                    let u1 = ((alpha[1][0] * x + alpha[1][1] * y) + alpha[1][2] * z - y0) / b;
                    let u2 = ((alpha[2][0] * x + alpha[2][1] * y) + alpha[2][2] * z - z0) / c;
                    let s = (u0 * u0 + u1 * u1) + u2 * u2;
                    if s <= 1.0 {
                        obj[[i, j, k]] = (obj[[i, j, k]] as f64 + amp) as f32;
                    }
                }
            }
        }
    }
    // .clip(0, np.inf): negatives → 0 (no NaN/inf in this phantom).
    obj.mapv_inplace(|v| if v < 0.0 { 0.0 } else { v });
    Ok(Volume::new(obj))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shepp3d_rejects_zero_size() {
        assert!(matches!(shepp3d(0), Err(Error::InvalidParam(_))));
    }

    #[test]
    fn shepp3d_shape_clip_and_structure() {
        let v = shepp3d(16).unwrap();
        assert_eq!(v.array.dim(), (16, 16, 16));
        // clip(0, inf): no negatives survive.
        assert!(v.array.iter().all(|&x| x >= 0.0), "clip left a negative");
        // The phantom has interior structure (not an all-zero cube).
        assert!(v.array.iter().any(|&x| x > 0.0), "phantom is empty");
        // The skull/brain region sits near the centre; the central voxel is
        // inside the outermost ellipsoid, so it is strictly positive.
        assert!(v.array[[8, 8, 8]] > 0.0, "centre voxel not inside the head");
    }
}
