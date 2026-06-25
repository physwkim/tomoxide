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
