//! Verifies iterative reconstruction (SIRT/CGLS) runs on a laminography geometry
//! through the generic host solvers composed over the tilted CUDA projector pair
//! (`project`/`backproject` with phi ≠ π/2). The device-resident
//! `IterativeReconstruct::solve` returns `None` for a tilted beam, so lamino
//! falls through to the generic solvers, which drive the lamino-capable
//! capabilities.
//!
//! Data-consistent recovery: a phantom volume `[nz,n,n]` is forward-projected
//! through the *same* tilted forward operator the solver's `A` uses, so a
//! converged least-squares solve must recover the phantom (an inverse crime that
//! isolates solver+projector correctness — physical fidelity vs tomocupy is the
//! analytic path's job). Reconstructs at volume-height = detector rows (`nz`);
//! a distinct recon-height `rh` is a separate solver change.
//!
//! Own test binary (touches CUDA device state) per the suite convention.
#![cfg(feature = "cuda")]

use tomoxide::{
    recon, Algorithm, Angles, Beam, Center, CudaBackend, Detector, ForwardProject, Geometry,
    Layout, ReconParams, Tomo, Volume,
};

fn pearson(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mb = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va.sqrt() * vb.sqrt()).max(1e-12)
}

// (cuda, geom, data-consistent sinogram, phantom) for a small tilted problem.
// The sinogram is the phantom forward-projected through the tilted operator the
// solver's `A` uses, so a converged solve must recover the phantom.
type Setup = (CudaBackend, Geometry, Tomo<f32>, Volume<f32>);
fn setup() -> Option<Setup> {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return None;
        }
    };

    let (n, nproj, nz) = (64usize, 120usize, 32usize);
    let lamino_angle_deg = 20.0f32;
    let phi = std::f32::consts::FRAC_PI_2 + lamino_angle_deg * std::f32::consts::PI / 180.0;
    let angles = Angles::uniform(nproj, 0.0, std::f32::consts::PI);
    let geom = Geometry {
        angles,
        center: Center::Scalar(n as f32 / 2.0),
        beam: Beam::Laminography { phi },
        detector: Detector {
            width: n,
            height: nz,
            pixel_size: 1.0,
        },
    };

    // Phantom volume [nz,n,n]: two offset solid cubes (bounded support well
    // inside the field so the tilt keeps it on the detector).
    let mut vol = ndarray::Array3::<f32>::zeros((nz, n, n));
    let mut cube = |cz: usize, cy: usize, cx: usize, r: usize, val: f32| {
        for z in cz.saturating_sub(r)..(cz + r).min(nz) {
            for y in cy.saturating_sub(r)..(cy + r).min(n) {
                for x in cx.saturating_sub(r)..(cx + r).min(n) {
                    vol[[z, y, x]] = val;
                }
            }
        }
    };
    cube(nz / 2, n / 2 - 6, n / 2 - 6, 5, 1.0);
    cube(nz / 2 + 4, n / 2 + 6, n / 2 + 4, 4, 0.6);
    let phantom = Volume::new(vol);

    let mut sino = Tomo::new(ndarray::Array3::zeros((nz, nproj, n)), Layout::Sinogram);
    cuda.project(&phantom, &geom, &mut sino).unwrap();
    Some((cuda, geom, sino, phantom))
}

fn run(algorithm: Algorithm, num_iter: usize) -> Option<f64> {
    let (cuda, geom, sino, phantom) = setup()?;
    let params = ReconParams {
        num_iter,
        ..Default::default()
    };
    let rec = recon::recon(&sino, &geom, algorithm, &params, &cuda).unwrap();
    assert_eq!(rec.array.dim(), phantom.array.dim());
    Some(pearson(
        rec.array.as_slice().unwrap(),
        phantom.array.as_slice().unwrap(),
    ))
}

// The structurally-global solvers (per-voxel/per-ray diagonal weights, no
// per-slice scalar step size) are geometry-agnostic and recover the phantom
// through the tilted projector pair. The gradient/Krylov solvers (CGLS/GRAD/
// TIKH) use a per-slice scalar step — a parallel-beam slice-separability
// optimization — and do NOT converge for the non-separable lamino geometry, so
// `recon` rejects them for laminography (see `cuda_lamino_rejects_per_slice`).
#[test]
fn cuda_lamino_sirt_converges() {
    let Some(r) = run(Algorithm::Sirt, 60) else {
        return;
    };
    eprintln!("lamino SIRT recovery Pearson = {r:.4}");
    assert!(r > 0.9, "lamino SIRT did not converge: Pearson = {r:.4}");
}

#[test]
fn cuda_lamino_mlem_converges() {
    let Some(r) = run(Algorithm::Mlem, 60) else {
        return;
    };
    eprintln!("lamino MLEM recovery Pearson = {r:.4}");
    assert!(r > 0.8, "lamino MLEM did not converge: Pearson = {r:.4}");
}

#[test]
fn cuda_lamino_tv_converges() {
    let Some(r) = run(Algorithm::Tv, 60) else {
        return;
    };
    eprintln!("lamino TV recovery Pearson = {r:.4}");
    assert!(r > 0.9, "lamino TV did not converge: Pearson = {r:.4}");
}

// The per-slice solvers are rejected for laminography (not silently wrong).
#[test]
fn cuda_lamino_rejects_per_slice() {
    let Some((cuda, geom, sino, _)) = setup() else {
        return;
    };
    let params = ReconParams {
        num_iter: 5,
        ..Default::default()
    };
    for algo in [Algorithm::Cgls, Algorithm::Grad, Algorithm::Tikh] {
        let r = recon::recon(&sino, &geom, algo, &params, &cuda);
        assert!(
            r.is_err(),
            "{algo:?} must be rejected for laminography, got Ok"
        );
    }
}
