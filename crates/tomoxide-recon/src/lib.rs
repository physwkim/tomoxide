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

use ndarray::{Array3, Axis};
use tomoxide_core::backend::{Backend, FilteredBackproject, ForwardProject};
use tomoxide_core::data::{Layout, Tomo, Volume};
use tomoxide_core::error::{Error, Result};
use tomoxide_core::geometry::{Angles, Geometry};
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
        Algorithm::Mlem => mlem(sino, geom, params, proj, bp),
        Algorithm::Osem => osem(sino, geom, params, proj, bp),
        // ART/BART and the regularized family (ospml_*, pml_*, tv, tikh, grad,
        // vector) land later in M2; SIRT/MLEM/OSEM above are the shared skeleton.
        _ => Err(Error::todo(
            "recon iterative (ART/BART + regularized family)",
            "tomopy libtomo/recon/{art,bart,ospml_hybrid,ospml_quad,pml_hybrid,pml_quad,tv,tikh,grad,vector}.c",
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

/// Maximum-Likelihood Expectation-Maximization.
///
/// Multiplicative update `x ← x ∘ Aᵀ(b ⊘ A x) ⊘ Aᵀ(1)`, positivity-preserving
/// from a positive initial guess; requires a non-negative sinogram. Ports the
/// EM update of tomopy `accel/cxx/mlem.cc`. `A` = forward projector,
/// `Aᵀ` = back-projector.
fn mlem(
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

    // Sensitivity Aᵀ(1).
    let ones_sino = Tomo::new(Array3::from_elem((nz, nang, ncols), 1.0), Layout::Sinogram);
    let mut sens = Volume::new(Array3::zeros((nz, n, n)));
    bp.backproject(&ones_sino, geom, &mut sens)?;

    let mut vol = Volume::new(Array3::from_elem((nz, n, n), 1.0)); // positive init
    let mut ax = Tomo::new(Array3::zeros((nz, nang, ncols)), Layout::Sinogram);
    let mut corr = Volume::new(Array3::zeros((nz, n, n)));
    for _ in 0..params.num_iter.max(1) {
        proj.project(&vol, geom, &mut ax)?; // A x
        let mut ratio = b.array.clone();
        ndarray::Zip::from(&mut ratio)
            .and(&ax.array)
            .for_each(|r, &a| {
                *r = if a.abs() > 1e-6 { *r / a } else { 0.0 }; // b ⊘ A x
            });
        bp.backproject(&Tomo::new(ratio, Layout::Sinogram), geom, &mut corr)?;
        ndarray::Zip::from(&mut vol.array)
            .and(&corr.array)
            .and(&sens.array)
            .for_each(|x, &c, &s| {
                if s.abs() > 1e-6 {
                    *x = *x * c / s; // x ∘ Aᵀ(ratio) ⊘ sens
                }
            });
    }
    Ok(vol)
}

/// Partition the `nang` angle indices into ordered subsets, matching tomopy
/// `osem.c`: each subset is a contiguous slice of the angle *ordering* — the
/// caller's `ind_block` permutation when it has length `nang`, else the identity
/// `0..nang`. Subset `os` has `nang/num_block + (os < nang % num_block)` angles,
/// so the blocks tile `0..nang` exactly. `num_block` is clamped to `1..=nang`,
/// and `1` yields the single full-angle subset (i.e. plain MLEM).
fn ordered_subsets(nang: usize, params: &ReconParams) -> Vec<Vec<usize>> {
    let num_block = params.num_block.clamp(1, nang.max(1));
    let order: Vec<usize> = if params.ind_block.len() == nang {
        params.ind_block.iter().map(|&i| i as usize).collect()
    } else {
        (0..nang).collect()
    };
    let blocksize = nang / num_block;
    let remainder = nang % num_block;
    let mut subsets = Vec::with_capacity(num_block);
    let mut start = 0;
    for os in 0..num_block {
        let len = blocksize + usize::from(os < remainder);
        subsets.push(order[start..start + len].to_vec());
        start += len;
    }
    subsets
}

/// Ordered-Subset Expectation-Maximization.
///
/// MLEM restricted to ordered angle-subsets: each subset `s` applies one
/// multiplicative `x ← x ∘ Aₛᵀ(b ⊘ Aₛ x) ⊘ Aₛᵀ(1)` update over only its own
/// angles, so a single outer iteration performs `num_block` updates (faster
/// early convergence than MLEM). With `num_block ≤ 1` it is exactly [`mlem`].
/// The per-subset sensitivity `Aₛᵀ(1)` is geometry-only, so it is precomputed
/// once. Ports tomopy `libtomo/recon/osem.c`. `Aₛ` = forward projector over
/// subset `s`'s angles, `Aₛᵀ` = its back-projector.
fn osem(
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

    /// One ordered subset: its sub-geometry, measured sinogram slice, and the
    /// (iteration-invariant) sensitivity `Aₛᵀ(1)`.
    struct Subset {
        geom: Geometry,
        b: Array3<f32>,    // [nz, len, ncols]
        sens: Array3<f32>, // [nz, n, n]
    }
    let mut subsets = Vec::new();
    for idx in ordered_subsets(nang, params) {
        let len = idx.len();
        let mut sub_geom = geom.clone();
        sub_geom.angles = Angles(idx.iter().map(|&p| geom.angles.0[p]).collect());
        let sub_b = b.array.select(Axis(1), &idx);
        let ones = Tomo::new(Array3::from_elem((nz, len, ncols), 1.0), Layout::Sinogram);
        let mut sens = Volume::new(Array3::zeros((nz, n, n)));
        bp.backproject(&ones, &sub_geom, &mut sens)?;
        subsets.push(Subset {
            geom: sub_geom,
            b: sub_b,
            sens: sens.array,
        });
    }

    let mut vol = Volume::new(Array3::from_elem((nz, n, n), 1.0)); // positive init
    let mut corr = Volume::new(Array3::zeros((nz, n, n)));
    for _ in 0..params.num_iter.max(1) {
        for sub in &subsets {
            let len = sub.geom.angles.0.len();
            let mut ax = Tomo::new(Array3::zeros((nz, len, ncols)), Layout::Sinogram);
            proj.project(&vol, &sub.geom, &mut ax)?; // Aₛ x
            let mut ratio = sub.b.clone();
            ndarray::Zip::from(&mut ratio)
                .and(&ax.array)
                .for_each(|r, &a| {
                    *r = if a.abs() > 1e-6 { *r / a } else { 0.0 }; // b ⊘ Aₛ x
                });
            bp.backproject(&Tomo::new(ratio, Layout::Sinogram), &sub.geom, &mut corr)?;
            ndarray::Zip::from(&mut vol.array)
                .and(&corr.array)
                .and(&sub.sens)
                .for_each(|x, &c, &s| {
                    if s.abs() > 1e-6 {
                        *x = *x * c / s; // x ∘ Aₛᵀ(ratio) ⊘ Aₛᵀ(1)
                    }
                });
        }
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
