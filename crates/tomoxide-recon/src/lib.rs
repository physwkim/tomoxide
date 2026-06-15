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
mod gridrec;
pub mod ring;

use ndarray::Array3;
use tomoxide_core::backend::{Backend, FilteredBackproject, ForwardProject};
use tomoxide_core::data::{Layout, Tomo, Volume};
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
    let n = grid_size(sino, params);
    let nz = sino.n_rows();

    // gridrec is a Fourier-grid method (needs only the Fft capability); the
    // others are an FBP filter pass + back-projection.
    if algorithm == Algorithm::Gridrec {
        let fft = backend.fft().ok_or_else(|| missing("Fft", backend))?;
        return Ok(Volume::new(gridrec::gridrec(sino, geom, n, fft)?));
    }
    let bp = backend
        .backprojector()
        .ok_or_else(|| missing("FilteredBackproject", backend))?;
    let filt = backend
        .fbp_filter()
        .ok_or_else(|| missing("FbpFilter", backend))?;
    let kernel = filt.make_filter(params.filter_name, sino.n_cols())?;
    let mut filtered = sino.clone();
    filt.apply(&mut filtered, &kernel, geom)?;
    let mut vol = Volume::new(Array3::zeros((nz, n, n)));
    bp.backproject(&filtered, geom, &mut vol)?;
    Ok(vol)
}

fn iterative(
    sino: &Tomo<f32>,
    geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    backend: &dyn Backend,
) -> Result<Volume<f32>> {
    let proj = backend
        .projector()
        .ok_or_else(|| missing("ForwardProject", backend))?;
    let bp = backend
        .backprojector()
        .ok_or_else(|| missing("FilteredBackproject", backend))?;
    match algorithm {
        Algorithm::Sirt => sirt(sino, geom, params, proj, bp),
        // ART/BART/OSEM and the regularized family (ospml_*, pml_*, tv, tikh,
        // grad, vector) land in M2; the SIRT loop above is the shared skeleton.
        _ => Err(Error::todo(
            "recon iterative (ART/BART/OSEM + regularized family)",
            "tomopy libtomo/recon/{art,bart,osem,ospml_hybrid,ospml_quad,pml_hybrid,pml_quad,tv,tikh,grad,vector}.c",
        )),
    }
}

/// Simultaneous Iterative Reconstruction Technique.
///
/// R/C-weighted update `x ← x + C ∘ Aᵀ(R ∘ (b − A x))` with `R = 1/A(1)`
/// (per-ray length) and `C = 1/Aᵀ(1)` (per-pixel sensitivity). This is the
/// parameter-free, convergent form of tomopy's rotation-based SIRT (which
/// distributes the per-ray residual by `1/nx` and averages over angles —
/// exactly `R` and `C` here). `A` = forward projector, `Aᵀ` = back-projector.
fn sirt(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    proj: &dyn ForwardProject,
    bp: &dyn FilteredBackproject,
) -> Result<Volume<f32>> {
    let n = grid_size(sino, params);
    let nz = sino.n_rows();
    let b = sino.to_layout(Layout::Sinogram);
    let nang = b.n_angles();
    let ncols = b.n_cols();

    // Ray-length weights R = 1 / A(1).
    let ones_img = Volume::new(Array3::from_elem((nz, n, n), 1.0));
    let mut ray = Tomo::new(Array3::zeros((nz, nang, ncols)), Layout::Sinogram);
    proj.project(&ones_img, geom, &mut ray)?;
    let rw = ray
        .array
        .mapv(|v| if v.abs() > 1e-6 { 1.0 / v } else { 0.0 });

    // Sensitivity weights C = 1 / Aᵀ(1).
    let ones_sino = Tomo::new(Array3::from_elem((nz, nang, ncols), 1.0), Layout::Sinogram);
    let mut sens = Volume::new(Array3::zeros((nz, n, n)));
    bp.backproject(&ones_sino, geom, &mut sens)?;
    let cw = sens
        .array
        .mapv(|v| if v.abs() > 1e-6 { 1.0 / v } else { 0.0 });

    let mut vol = Volume::new(Array3::zeros((nz, n, n)));
    let mut ax = Tomo::new(Array3::zeros((nz, nang, ncols)), Layout::Sinogram);
    let mut corr = Volume::new(Array3::zeros((nz, n, n)));
    for _ in 0..params.num_iter.max(1) {
        proj.project(&vol, geom, &mut ax)?; // A x
        let mut resid = &b.array - &ax.array; // b − A x
        resid *= &rw; // R ∘ (b − A x)
        bp.backproject(&Tomo::new(resid, Layout::Sinogram), geom, &mut corr)?;
        vol.array += &(&cw * &corr.array); // x += C ∘ Aᵀ(…)
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
