//! Zone-plate magnification correction (XANES prep step, docs/GUI.md §6 #14).
//!
//! A transmission X-ray microscope focuses with a Fresnel zone plate whose focal
//! length grows with photon energy, so each energy in a XANES scan images the
//! sample at a slightly different magnification. Left uncorrected this is a
//! per-energy spatial scale drift that smears the fitted edge; the correction
//! resamples every energy volume back onto the first energy's scale before
//! registration and fitting.
//!
//! Direct port of the reference `magnification_correction.py` (`xanes_tools`):
//! the focal length drives a per-energy correction factor (normalised to the
//! first energy), and each volume is affine-scaled about its centre by that
//! factor. **Axis convention matches the reference and the `reconstructions/
//! {energy}` stacks the [`super::reader`] loads**: volumes are `(z, y, x)` with
//! `y` the rotation axis — the correction scales `z` and `x` and leaves `y`
//! untouched.

use ndarray::{Array2, Array3, ArrayView3};
use rayon::prelude::*;

/// Zone-plate geometry for the focal-length model. Defaults match the reference
/// tool (`--magnification 23.08`, 300 µm diameter, 30 nm outermost zone).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagnificationParams {
    /// Nominal system magnification.
    pub magnification: f64,
    /// Zone-plate diameter, metres.
    pub zp_diameter_m: f64,
    /// Outermost zone width, metres.
    pub zp_outermost_width_m: f64,
    /// Number of zones. Retained for parity with the reference tool's knobs; the
    /// focal-length model below does not use it (neither does the reference).
    pub num_zones: u32,
}

impl Default for MagnificationParams {
    fn default() -> Self {
        Self {
            magnification: 23.08,
            zp_diameter_m: 300e-6,
            zp_outermost_width_m: 30e-9,
            num_zones: 2500,
        }
    }
}

/// Zone-plate focal length at `energy` (eV). Units follow the reference exactly
/// (the absolute value is irrelevant — only ratios survive into the factors).
fn focal_length(energy: f64, p: &MagnificationParams) -> f64 {
    let wl_nm = 1.239842 / energy * 1000.0; // photon wavelength, nm
    p.zp_outermost_width_m * p.zp_diameter_m / wl_nm / 1000.0
}

/// Per-energy magnification correction factors, normalised so the first energy
/// is `1.0`. `energies` need not be sorted, but the factors follow the given
/// order (the caller is expected to pass energies in the volumes' order). A
/// factor `cf` means each energy volume is resampled with input coordinate
/// `cf·out + (1−cf)·n/2` — `cf < 1` magnifies (zooms in), `cf > 1` shrinks.
///
/// Port of `magnification_corr_factors`.
pub fn magnification_corr_factors(energies: &[f64], p: &MagnificationParams) -> Vec<f64> {
    if energies.is_empty() {
        return Vec::new();
    }
    let a: Vec<f64> = energies
        .iter()
        .map(|&e| focal_length(e, p) * (1.0 + 1.0 / p.magnification))
        .collect();

    // b_real[0] = a[0]·m; thereafter accumulate the negative first difference of
    // `a` onto the running value (the reference's `bRealValues` recurrence).
    let mut b_real = vec![0.0f64; a.len()];
    b_real[0] = a[0] * p.magnification;
    for i in 1..a.len() {
        b_real[i] = b_real[i - 1] - (a[i] - a[i - 1]);
    }

    let mag: Vec<f64> = a.iter().zip(&b_real).map(|(&ai, &bi)| ai / bi).collect();
    let mag0 = mag[0];
    mag.iter().map(|&m| mag0 / m).collect()
}

/// Resample one `(z, y, x)` volume by the magnification factor `cf`, scaling `z`
/// and `x` about the volume centre with bilinear interpolation and zero fill
/// outside the source — `scipy.ndimage.affine_transform(diag(cf, 1, cf),
/// offset=(1−cf)·n/2, order=1, mode='constant', cval=0)`. `y` (the rotation
/// axis) is untouched. `cf == 1` returns a copy.
pub fn apply_magnification(vol: ArrayView3<f32>, cf: f64) -> Array3<f32> {
    let (nz, ny, nx) = vol.dim();
    if (cf - 1.0).abs() < 1e-8 || nz == 0 || nx == 0 {
        return vol.to_owned();
    }
    let off_z = (1.0 - cf) * nz as f64 / 2.0;
    let off_x = (1.0 - cf) * nx as f64 / 2.0;

    // Sample a source voxel, treating out-of-range indices as 0 (constant fill).
    let at = |z: isize, y: usize, x: isize| -> f32 {
        if z < 0 || x < 0 || z as usize >= nz || x as usize >= nx {
            0.0
        } else {
            vol[[z as usize, y, x as usize]]
        }
    };

    // Build each output z-plane independently (matches the fit driver's rayon
    // granularity; `AxisIterMut` is not itself a parallel iterator).
    let planes: Vec<Array2<f32>> = (0..nz)
        .into_par_iter()
        .map(|oz| {
            let mut plane = Array2::<f32>::zeros((ny, nx));
            let src_z = cf * oz as f64 + off_z;
            let z0f = src_z.floor();
            let fz = (src_z - z0f) as f32;
            let z0 = z0f as isize;
            for ox in 0..nx {
                let src_x = cf * ox as f64 + off_x;
                let x0f = src_x.floor();
                let fx = (src_x - x0f) as f32;
                let x0 = x0f as isize;
                for oy in 0..ny {
                    // Bilinear over (z, x); y is the identity axis.
                    let v00 = at(z0, oy, x0);
                    let v01 = at(z0, oy, x0 + 1);
                    let v10 = at(z0 + 1, oy, x0);
                    let v11 = at(z0 + 1, oy, x0 + 1);
                    let top = v00 * (1.0 - fx) + v01 * fx;
                    let bot = v10 * (1.0 - fx) + v11 * fx;
                    plane[[oy, ox]] = top * (1.0 - fz) + bot * fz;
                }
            }
            plane
        })
        .collect();

    let mut out = Array3::<f32>::zeros((nz, ny, nx));
    for (oz, plane) in planes.into_iter().enumerate() {
        out.slice_mut(ndarray::s![oz, .., ..]).assign(&plane);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn first_energy_factor_is_unity() {
        let energies = [8300.0, 8325.0, 8350.0, 8400.0];
        let cf = magnification_corr_factors(&energies, &MagnificationParams::default());
        assert_eq!(cf.len(), 4);
        assert!(
            (cf[0] - 1.0).abs() < 1e-12,
            "cf[0] must be 1, got {}",
            cf[0]
        );
        // Focal length grows with energy ⇒ factors drift monotonically away from 1.
        assert!(cf.windows(2).all(|w| w[1] < w[0]), "factors: {cf:?}");
    }

    #[test]
    fn empty_energies_gives_no_factors() {
        assert!(magnification_corr_factors(&[], &MagnificationParams::default()).is_empty());
    }

    #[test]
    fn unit_factor_is_identity() {
        let mut vol = Array3::<f32>::zeros((3, 2, 4));
        vol[[1, 0, 2]] = 5.0;
        let out = apply_magnification(vol.view(), 1.0);
        assert_eq!(out, vol);
    }

    #[test]
    fn scaling_preserves_the_centre_voxel() {
        // The transform's fixed coordinate is `n/2`. With an even extent that is
        // the integer index `n/2`, so a bright voxel there maps exactly to itself
        // (fz = fx = 0) regardless of `cf`.
        let (nz, ny, nx) = (4, 1, 4);
        let mut vol = Array3::<f32>::zeros((nz, ny, nx));
        vol[[2, 0, 2]] = 1.0; // == (n/2) on both scaled axes
        let out = apply_magnification(vol.view(), 0.5);
        assert!(
            (out[[2, 0, 2]] - 1.0).abs() < 1e-6,
            "centre lost: {}",
            out[[2, 0, 2]]
        );
    }

    #[test]
    fn y_axis_is_not_scaled() {
        // Two y-planes with distinct constant values: scaling z/x must leave the
        // y separation intact (each y-plane maps only within itself).
        let (nz, ny, nx) = (4, 2, 4);
        let mut vol = Array3::<f32>::zeros((nz, ny, nx));
        for z in 0..nz {
            for x in 0..nx {
                vol[[z, 0, x]] = 1.0;
                vol[[z, 1, x]] = 7.0;
            }
        }
        let out = apply_magnification(vol.view(), 0.8);
        // Interior voxels (away from the zero-filled border) keep their plane value.
        assert!((out[[2, 0, 2]] - 1.0).abs() < 1e-4, "y=0 plane bled");
        assert!((out[[2, 1, 2]] - 7.0).abs() < 1e-4, "y=1 plane bled");
    }
}
