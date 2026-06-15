//! # tomoxide-recon
//!
//! Backend-agnostic reconstruction. The single [`recon`] entry point dispatches
//! every [`Algorithm`] to backend capability traits, so the same code path runs
//! on CPU, CUDA, or wgpu. It depends on `tomoxide-core` only — never on a
//! concrete backend.
//!
//! Analytic methods (`fbp`, `gridrec`, `fourierrec`, `lprec`, `linerec`) are a
//! filter + back-projection pass; iterative methods compose forward projection
//! and back-projection in a loop (see `docs/ARCHITECTURE.md` §3).
#![forbid(unsafe_code)]

pub mod center;
pub mod ring;

use ndarray::Array3;
use tomoxide_core::backend::Backend;
use tomoxide_core::data::{Tomo, Volume};
use tomoxide_core::error::{Error, Result};
use tomoxide_core::geometry::Geometry;
use tomoxide_core::params::{Algorithm, ReconParams};

fn missing(capability: &'static str, backend: &dyn Backend) -> Error {
    Error::MissingCapability {
        backend: backend.name(),
        capability,
    }
}

/// Reconstruct a volume from a sinogram stack with the given algorithm.
///
/// `sino` is consumed by neither path (it is cloned where filtering mutates a
/// copy). The output volume is `[n_rows, n_grid, n_grid]`.
pub fn recon(
    sino: &Tomo<f32>,
    geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    backend: &dyn Backend,
) -> Result<Volume<f32>> {
    if algorithm.is_analytic() {
        analytic(sino, geom, algorithm, params, backend)
    } else {
        iterative(sino, geom, algorithm, params, backend)
    }
}

/// Grid size for the output slice (defaults to the detector width).
fn grid_size(sino: &Tomo<f32>, params: &ReconParams) -> usize {
    params.num_gridx.unwrap_or_else(|| sino.n_cols())
}

fn analytic(
    sino: &Tomo<f32>,
    geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    backend: &dyn Backend,
) -> Result<Volume<f32>> {
    let bp = backend
        .backprojector()
        .ok_or_else(|| missing("FilteredBackproject", backend))?;
    let n = grid_size(sino, params);
    let nz = sino.n_rows();
    let mut vol = Volume::new(Array3::zeros((nz, n, n)));

    // gridrec filters internally; the others take an explicit FBP filter pass.
    if algorithm == Algorithm::Gridrec {
        bp.backproject(sino, geom, &mut vol)?;
    } else {
        let filt = backend
            .fbp_filter()
            .ok_or_else(|| missing("FbpFilter", backend))?;
        let kernel = filt.make_filter(params.filter_name, sino.n_cols())?;
        let mut filtered = sino.clone();
        filt.apply(&mut filtered, &kernel, geom)?;
        bp.backproject(&filtered, geom, &mut vol)?;
    }
    Ok(vol)
}

fn iterative(
    sino: &Tomo<f32>,
    geom: &Geometry,
    _algorithm: Algorithm,
    params: &ReconParams,
    backend: &dyn Backend,
) -> Result<Volume<f32>> {
    let proj = backend
        .projector()
        .ok_or_else(|| missing("ForwardProject", backend))?;
    let bp = backend
        .backprojector()
        .ok_or_else(|| missing("FilteredBackproject", backend))?;
    let n = grid_size(sino, params);
    let nz = sino.n_rows();

    let mut vol = Volume::new(Array3::zeros((nz, n, n)));
    let mut sim = sino.clone(); // forward-projection workspace / residual

    // The generic SIRT/ART/MLEM skeleton: project, form a residual, back-project
    // the correction. The per-algorithm residual/regularization math lands in M2
    // (see docs/PORTING.md §B); the loop structure is backend-agnostic already.
    for _ in 0..params.num_iter.max(1) {
        proj.project(&vol, geom, &mut sim)?;
        bp.backproject(&sim, geom, &mut vol)?;
    }
    Ok(vol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tomoxide_core::backend::{Backend, DeviceKind};
    use tomoxide_core::data::Layout;
    use tomoxide_core::dtype::Dtype;
    use tomoxide_core::geometry::Angles;

    /// A backend that advertises no capabilities, to exercise dispatch.
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

    fn tiny_sino() -> (Tomo<f32>, Geometry) {
        let s = Tomo::new(Array3::<f32>::zeros((2, 4, 8)), Layout::Sinogram); // [row,angle,col]
        let g = Geometry::parallel(Angles::uniform(4, 0.0, std::f32::consts::PI), 8, 2, 1.0);
        (s, g)
    }

    #[test]
    fn analytic_without_backprojector_reports_missing_capability() {
        let (s, g) = tiny_sino();
        let err = recon(
            &s,
            &g,
            Algorithm::Fbp,
            &ReconParams::default(),
            &NullBackend,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::MissingCapability {
                capability: "FilteredBackproject",
                ..
            }
        ));
    }

    #[test]
    fn iterative_without_projector_reports_missing_capability() {
        let (s, g) = tiny_sino();
        let err = recon(
            &s,
            &g,
            Algorithm::Sirt,
            &ReconParams::default(),
            &NullBackend,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::MissingCapability {
                capability: "ForwardProject",
                ..
            }
        ));
    }
}
