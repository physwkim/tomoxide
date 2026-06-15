//! # tomoxide-sim
//!
//! Phantoms and forward simulation (ports tomopy `sim/` + `misc/phantom.py`).
//! `angles` and `phantom::shepp2d` are real; `project` and the noise models are
//! stubs (see `docs/PORTING.md` §F).
#![forbid(unsafe_code)]

pub mod phantom;

use tomoxide_core::data::{Tomo, Volume};
use tomoxide_core::error::{Error, Result};
use tomoxide_core::geometry::{Angles, Geometry};

pub use phantom::shepp2d;

/// `nang` uniformly spaced angles over `[ang1, ang2)` radians (tomopy
/// `sim/project.py:241`).
pub fn angles(nang: usize, ang1: f32, ang2: f32) -> Vec<f32> {
    Angles::uniform(nang, ang1, ang2).0
}

/// Forward project a volume into a projection stack (the Radon transform).
///
/// Stub — the real projector ports tomopy `libtomo/recon/project.c`. Once the
/// CPU [`ForwardProject`](tomoxide_core::backend::ForwardProject) capability
/// lands (M1), this becomes a thin wrapper for round-trip testing.
pub fn project(_vol: &Volume<f32>, _geom: &Geometry) -> Result<Tomo<f32>> {
    Err(Error::todo(
        "sim::project",
        "tomopy libtomo/recon/project.c",
    ))
}

/// Add Gaussian noise (tomopy `sim/project.py:110`). Stub.
pub fn add_gaussian(_data: &mut Tomo<f32>, _mean: f32, _std: f32) -> Result<()> {
    Err(Error::todo(
        "sim::add_gaussian",
        "tomopy sim/project.py:110",
    ))
}

/// Add Poisson noise (tomopy `sim/project.py:136`). Stub.
pub fn add_poisson(_data: &mut Tomo<f32>) -> Result<()> {
    Err(Error::todo("sim::add_poisson", "tomopy sim/project.py:136"))
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
}
