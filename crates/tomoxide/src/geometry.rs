//! Acquisition geometry: angles, rotation center, beam type, detector.
//!
//! See `docs/ARCHITECTURE.md` §1.

use crate::error::{Error, Result};

/// Projection angles in radians.
#[derive(Clone, Debug)]
pub struct Angles(pub Vec<f32>);

impl Angles {
    /// `nang` uniformly spaced angles over `[ang1, ang2)` radians.
    ///
    /// Mirrors tomopy `sim/project.py::angles` (default `0..π`).
    pub fn uniform(nang: usize, ang1: f32, ang2: f32) -> Self {
        if nang == 0 {
            return Angles(Vec::new());
        }
        let step = (ang2 - ang1) / nang as f32;
        Angles((0..nang).map(|i| ang1 + step * i as f32).collect())
    }

    /// Number of angles.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are no angles.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Rotation-axis position, in detector-column coordinates.
///
/// tomopy accepts a per-slice center array; tomocupy searches it per chunk.
#[derive(Clone, Debug)]
pub enum Center {
    /// One center shared by every slice.
    Scalar(f32),
    /// One center per detector row (slice).
    PerRow(Vec<f32>),
}

impl Center {
    /// The center for row `row`.
    pub fn at(&self, row: usize) -> f32 {
        match self {
            Center::Scalar(c) => *c,
            Center::PerRow(v) => v[row],
        }
    }
}

/// Beam geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Beam {
    /// Parallel-beam (synchrotron); slices reconstruct independently.
    Parallel,
    /// Cone-beam with the given source-to-sample distance.
    Cone {
        /// Source-to-sample distance (same units as `pixel_size`).
        source_dist: f32,
    },
    /// Laminography with the given pitch angle `phi` in radians (tomocupy).
    Laminography {
        /// Laminographic tilt angle in radians.
        phi: f32,
    },
}

impl Beam {
    /// Laminography beam for a rotation-axis tilt of `tilt_deg` degrees.
    pub fn laminography(tilt_deg: f32) -> Self {
        Beam::Laminography {
            phi: lamino_phi(tilt_deg),
        }
    }
}

/// Beam pitch `phi = π/2 + tilt_deg·π/180` (radians) for a rotation-axis tilt
/// in degrees — tomocupy's `--lamino_angle` convention. The only tilt→phi
/// conversion; [`lamino_tilt_deg`] is its inverse.
pub fn lamino_phi(tilt_deg: f32) -> f32 {
    std::f32::consts::FRAC_PI_2 + tilt_deg * std::f32::consts::PI / 180.0
}

/// Rotation-axis tilt in degrees for beam pitch `phi` (inverse of
/// [`lamino_phi`]).
pub fn lamino_tilt_deg(phi: f32) -> f32 {
    (phi - std::f32::consts::FRAC_PI_2) * 180.0 / std::f32::consts::PI
}

/// Detector dimensions and pixel size.
#[derive(Clone, Copy, Debug)]
pub struct Detector {
    /// Width in pixels (detector columns).
    pub width: usize,
    /// Height in pixels (detector rows).
    pub height: usize,
    /// Physical pixel size (cm), used by phase retrieval and cone-beam.
    pub pixel_size: f32,
}

/// Full acquisition geometry passed to projectors/backprojectors.
#[derive(Clone, Debug)]
pub struct Geometry {
    /// Projection angles (radians).
    pub angles: Angles,
    /// Rotation-axis position.
    pub center: Center,
    /// Beam type.
    pub beam: Beam,
    /// Detector description.
    pub detector: Detector,
}

impl Geometry {
    /// A minimal parallel-beam geometry centred on the detector midline.
    pub fn parallel(angles: Angles, width: usize, height: usize, pixel_size: f32) -> Self {
        Geometry {
            center: Center::Scalar(width as f32 / 2.0),
            angles,
            beam: Beam::Parallel,
            detector: Detector {
                width,
                height,
                pixel_size,
            },
        }
    }

    /// Where every projection saw the sample, slice by slice, in a `depth`-deep
    /// reconstruction. A voxel at radius `r` from the rotation axis and height
    /// `h = z − depth/2` reaches detector row `h·cos t ± r·sin t` from the
    /// middle and column `c ± r`, so it stayed on the detector all the way round
    /// iff `r ≤ min(c, W − c)` and `|h|·cos t + r·sin t ≤ H/2`. Outside that
    /// radius a slice is built from a partial orbit, and its texture is not the
    /// sample's. Parallel beam is the `t = 0` case. Cone beam is refused: its
    /// magnification makes the bound depend on the source distance.
    pub fn valid_fov(&self, depth: usize) -> Result<ValidFov> {
        let tilt = match self.beam {
            Beam::Parallel => 0.0,
            Beam::Laminography { phi } => f64::from(lamino_tilt_deg(phi)).to_radians(),
            Beam::Cone { .. } => {
                return Err(Error::InvalidParam(
                    "valid_fov: cone beam has no parallel-projection field of view".into(),
                ))
            }
        };
        let w = self.detector.width as f64;
        let lateral = |c: f32| f64::from(c).min(w - f64::from(c));
        let r_lateral = match &self.center {
            Center::Scalar(c) => lateral(*c),
            Center::PerRow(v) => v.iter().map(|&c| lateral(c)).fold(f64::INFINITY, f64::min),
        }
        .max(0.0);
        let (sin, cos) = (tilt.sin().abs(), tilt.cos());
        let half_h = self.detector.height as f64 / 2.0;
        let radius = (0..depth)
            .map(|z| {
                let reach = half_h - (z as f64 - depth as f64 / 2.0).abs() * cos;
                let r = if sin > 0.0 {
                    reach / sin
                } else if reach >= 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                };
                r.clamp(0.0, r_lateral)
            })
            .collect();
        Ok(ValidFov { r_lateral, radius })
    }
}

/// From [`Geometry::valid_fov`]: the disk about the rotation axis — the
/// reconstruction's centre — that every projection saw, per slice.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidFov {
    /// `min(c, W − c)`: the radius the detector width allows at any height
    /// (the tightest row of a per-row centre).
    pub r_lateral: f64,
    /// Per slice `z`, the valid radius in pixels: `r_lateral` near the middle,
    /// shrinking to 0 where `|h|·cos t` reaches `H/2`.
    pub radius: Vec<f64>,
}

impl ValidFov {
    /// The inclusive slice range whose whole `r_lateral` disk is valid; `None`
    /// when no slice is.
    pub fn full_z(&self) -> Option<(usize, usize)> {
        let full = |r: &f64| *r >= self.r_lateral;
        Some((
            self.radius.iter().position(full)?,
            self.radius.iter().rposition(full)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lamino_phi_is_tomocupy_convention_and_inverts() {
        assert_eq!(lamino_phi(0.0), std::f32::consts::FRAC_PI_2);
        assert!((lamino_phi(20.0) - (90.0f32 + 20.0).to_radians()).abs() < 1e-6);
        assert_eq!(
            Beam::laminography(20.0),
            Beam::Laminography {
                phi: lamino_phi(20.0)
            }
        );
        for d in [-30.0f32, 0.0, 17.5, 20.0, 35.0] {
            assert!((lamino_tilt_deg(lamino_phi(d)) - d).abs() < 1e-4, "{d}");
        }
    }

    fn lamino(center: f32, n: usize, tilt_deg: f32) -> Geometry {
        Geometry {
            beam: Beam::laminography(tilt_deg),
            center: Center::Scalar(center),
            ..Geometry::parallel(Angles(vec![0.0]), n, n, 1.0)
        }
    }

    /// The 261003 scan at bin 2 (axis 504 of 1024, 20°) is whole for
    /// `|h| ≤ (512 − 504·sin t)/cos t = 361.4` and blind at `512/cos t = 544.9`;
    /// at full resolution the bound is 722.8.
    #[test]
    fn valid_fov_bounds_the_laminography_disk_by_height() {
        let fov = lamino(504.0, 1024, 20.0).valid_fov(1090).unwrap();
        assert_eq!(fov.r_lateral, 504.0);
        assert_eq!(fov.full_z(), Some((545 - 361, 545 + 361)));
        assert_eq!(fov.radius[545], 504.0);
        assert!(fov.radius[545 - 362] < 504.0);
        assert_eq!(fov.radius[0], 0.0);
        let full = lamino(1008.0, 2048, 20.0).valid_fov(2180).unwrap();
        assert_eq!(full.full_z(), Some((1090 - 722, 1090 + 722)));
    }

    #[test]
    fn valid_fov_parallel_is_the_lateral_disk_on_every_detector_row() {
        let mut g = Geometry::parallel(Angles(vec![0.0]), 100, 50, 1.0);
        g.center = Center::PerRow(vec![40.0, 70.0]);
        let fov = g.valid_fov(60).unwrap();
        assert_eq!(fov.r_lateral, 30.0);
        assert_eq!(fov.full_z(), Some((5, 55)));
        assert_eq!(fov.radius[4], 0.0);
        g.beam = Beam::Cone { source_dist: 1.0 };
        assert!(matches!(g.valid_fov(60), Err(Error::InvalidParam(_))));
    }
}
