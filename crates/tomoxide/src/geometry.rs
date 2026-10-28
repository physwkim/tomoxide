//! Acquisition geometry: angles, rotation center, beam type, detector.
//!
//! See `docs/ARCHITECTURE.md` §1.

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
}
