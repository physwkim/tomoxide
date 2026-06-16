//! # tomoxide-sim
//!
//! Phantoms and forward simulation (ports tomopy `sim/` + `misc/phantom.py`).
//! `angles`, `phantom::shepp2d`, and the noise models are real; `project` is a
//! thin backend wrapper (see `docs/PORTING.md` §F).
#![forbid(unsafe_code)]

mod noise;
pub mod phantom;

use tomoxide_core::backend::Backend;
use tomoxide_core::data::{Layout, Tomo, Volume};
use tomoxide_core::error::{Error, Result};
use tomoxide_core::geometry::{Angles, Geometry};

pub use noise::{add_drift, add_gaussian, add_poisson, add_rings, add_salt_pepper, add_zingers};
pub use phantom::shepp2d;

/// `nang` uniformly spaced angles over `[ang1, ang2)` radians (tomopy
/// `sim/project.py:241`).
pub fn angles(nang: usize, ang1: f32, ang2: f32) -> Vec<f32> {
    Angles::uniform(nang, ang1, ang2).0
}

/// Forward project a volume into a sinogram (the Radon transform) via a
/// backend's [`ForwardProject`](tomoxide_core::backend::ForwardProject)
/// capability.
///
/// A thin convenience wrapper for round-trip testing — the projection math
/// lives in the backend (tomoxide-cpu ports tomopy `libtomo/recon/project.c`).
/// Returns the `[row, angle, col]` sinogram, or
/// [`Error::MissingCapability`](tomoxide_core::error::Error::MissingCapability)
/// if `backend` cannot forward-project.
pub fn project(vol: &Volume<f32>, geom: &Geometry, backend: &dyn Backend) -> Result<Tomo<f32>> {
    let proj = backend.projector().ok_or(Error::MissingCapability {
        backend: backend.name(),
        capability: "ForwardProject",
    })?;
    // project() overwrites `out` with the correctly shaped sinogram.
    let mut out = Tomo::new(ndarray::Array3::zeros((0, 0, 0)), Layout::Sinogram);
    proj.project(vol, geom, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shepp2d_has_expected_skull_intensity() {
        let p = shepp2d(64).unwrap();
        assert_eq!(p.dim(), (64, 64));
        // The center pixel sits inside the big skull (1.0) minus the inner
        // ellipse (-0.8) plus the lower mid ellipse offsets → ~0.2 region.
        let center = p[[32, 32]];
        assert!(center > 0.0, "center intensity was {center}");
        // Corners are outside every ellipse → 0.
        assert_eq!(p[[0, 0]], 0.0);
    }

    #[test]
    fn angles_span_half_turn() {
        let a = angles(180, 0.0, std::f32::consts::PI);
        assert_eq!(a.len(), 180);
        assert_eq!(a[0], 0.0);
    }

    #[test]
    fn project_reports_missing_capability() {
        use tomoxide_core::backend::DeviceKind;
        use tomoxide_core::dtype::Dtype;

        // A backend that advertises no ForwardProject capability.
        struct NullBackend;
        impl Backend for NullBackend {
            fn name(&self) -> &'static str {
                "null"
            }
            fn device(&self) -> DeviceKind {
                DeviceKind::Cpu
            }
            fn supports(&self, _dt: Dtype) -> bool {
                false
            }
        }

        let v = Volume::new(ndarray::Array3::<f32>::zeros((1, 4, 4)));
        let g = Geometry::parallel(Angles::uniform(4, 0.0, std::f32::consts::PI), 4, 1, 1.0);
        assert!(matches!(
            project(&v, &g, &NullBackend),
            Err(Error::MissingCapability {
                capability: "ForwardProject",
                ..
            })
        ));
    }
}
