//! Absolute-amplitude pins for the iterative projector pair.
//!
//! The roundtrip suite generates its sinograms with the SAME forward operator
//! it reconstructs with, so any operator gain cancels and a scale defect is
//! invisible there (the [[fourierrec-amplitude-scale]] lesson: Pearson and
//! roundtrip designs are scale-blind). These tests pin the *absolute* scale
//! against operator-independent ground truth instead: an analytic disk
//! sinogram `p(θ, s) = 2√(R² − s²)` (the exact line integral of a unit disk).
//!
//! Pinned convention: the forward projector is the plain line-integral Radon
//! transform `W` (unit pixel spacing, no gain) and the back-projector its pure
//! adjoint `Wᵀ`, so a converged solve of `W x = p` yields the physical μ —
//! `x ≈ 1.0` inside the disk, consistent with ART/BART's ungained ray rows and
//! tomopy `project.c`. A regression to the old `(π/nproj)·W` pair would show
//! up here as core means of `nproj/π` (≈ 30 at these sizes), and a stray
//! factor 2 as 0.5/2.0 — both far outside the tolerances.

use ndarray::Array3;
use tomoxide::backend::Backend;
use tomoxide::data::{Layout, Tomo};
use tomoxide::recon::recon;
use tomoxide::{Algorithm, Angles, CpuBackend, Geometry, ReconParams, Volume};

const N: usize = 128;
const NANG: usize = 96;
const RADIUS: f32 = 0.3 * N as f32;

/// Unit disk of radius [`RADIUS`] centred on the grid centre (n/2, n/2).
fn disk_phantom() -> Volume<f32> {
    let c = N as f32 / 2.0;
    let mut v = Array3::<f32>::zeros((1, N, N));
    for iy in 0..N {
        for ix in 0..N {
            let (dx, dy) = (ix as f32 - c, iy as f32 - c);
            if dx * dx + dy * dy <= RADIUS * RADIUS {
                v[[0, iy, ix]] = 1.0;
            }
        }
    }
    Volume::new(v)
}

/// Exact line integrals of the unit disk: `p(θ, s) = 2√(R² − s²)`, identical
/// for every angle (the disk is centred), with `s` measured from the rotation
/// axis at column n/2.
fn analytic_disk_sino() -> Tomo<f32> {
    let c = N as f32 / 2.0;
    let mut s = Array3::<f32>::zeros((1, NANG, N));
    for ia in 0..NANG {
        for ix in 0..N {
            let d = ix as f32 - c;
            if d.abs() < RADIUS {
                s[[0, ia, ix]] = 2.0 * (RADIUS * RADIUS - d * d).sqrt();
            }
        }
    }
    Tomo::new(s, Layout::Sinogram)
}

fn geom() -> Geometry {
    Geometry::parallel(Angles::uniform(NANG, 0.0, std::f32::consts::PI), N, 1, 1.0)
}

/// Mean over the disk core (radius < R/2), well inside any edge blur.
fn core_mean(vol: &Volume<f32>) -> f32 {
    let c = N as f32 / 2.0;
    let (mut sum, mut cnt) = (0.0f64, 0u32);
    for iy in 0..N {
        for ix in 0..N {
            let (dx, dy) = (ix as f32 - c, iy as f32 - c);
            if dx * dx + dy * dy < (RADIUS / 2.0) * (RADIUS / 2.0) {
                sum += vol.array[[0, iy, ix]] as f64;
                cnt += 1;
            }
        }
    }
    (sum / cnt as f64) as f32
}

/// The forward projector is the true line integral. Two probes:
///
/// - Radon mass invariant, every angle: each projection carries exactly the
///   phantom's total mass (the linear splat conserves mass per angle; a
///   `π/nproj` gain regression fails this by ~30×).
/// - Centre-ray value at θ = 0 only: with the axis-aligned angle the splat
///   hits integer columns exactly, so the centre column is the exact pixel
///   count along the diameter ≈ `2R`. (At oblique angles a *single* detector
///   bin of a pixel-driven splat ripples by ~10% while neighbouring bins
///   compensate — that is a discretization property, not a scale, so only the
///   exact angle is pinned per-bin.)
#[test]
fn forward_project_is_true_line_integral() {
    let phantom = disk_phantom();
    let g = geom();
    let mut sino = Tomo::new(Array3::zeros((0, 0, 0)), Layout::Sinogram);
    CpuBackend::new()
        .projector()
        .unwrap()
        .project(&phantom, &g, &mut sino)
        .unwrap();

    let mass: f32 = phantom.array.sum();
    for ia in 0..NANG {
        let proj_mass: f32 = (0..N).map(|ix| sino.array[[0, ia, ix]]).sum();
        assert!(
            (proj_mass / mass - 1.0).abs() < 0.01,
            "ia={ia}: projection mass {proj_mass} ≠ phantom mass {mass}"
        );
    }
    // θ = angles[0] = 0: exact column sums.
    let center_ray = sino.array[[0, 0, N / 2]];
    assert!(
        (center_ray / (2.0 * RADIUS) - 1.0).abs() < 0.02,
        "θ=0 centre ray {center_ray} ≠ diameter {}",
        2.0 * RADIUS
    );
}

/// Converged SIRT on the analytic sinogram reconstructs the physical μ = 1.0
/// inside the disk (not `nproj/π ≈ 30.6`, the old gained-pair fixed point).
#[test]
fn sirt_converges_to_physical_mu() {
    let params = ReconParams {
        num_iter: 150,
        ..Default::default()
    };
    let vol = recon(
        &analytic_disk_sino(),
        &geom(),
        Algorithm::Sirt,
        &params,
        &CpuBackend::new(),
    )
    .unwrap();
    let m = core_mean(&vol);
    assert!(
        (m - 1.0).abs() < 0.05,
        "SIRT core mean {m} ≠ 1.0 (physical μ)"
    );
}

/// Converged CGLS likewise lands on μ = 1.0 (CGLS is scale-free in structure,
/// so this pins the operator pair itself).
#[test]
fn cgls_converges_to_physical_mu() {
    let params = ReconParams {
        num_iter: 20,
        ..Default::default()
    };
    let vol = recon(
        &analytic_disk_sino(),
        &geom(),
        Algorithm::Cgls,
        &params,
        &CpuBackend::new(),
    )
    .unwrap();
    let m = core_mean(&vol);
    assert!(
        (m - 1.0).abs() < 0.05,
        "CGLS core mean {m} ≠ 1.0 (physical μ)"
    );
}
