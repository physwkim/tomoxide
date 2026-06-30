//! CUDA forward projection (`ForwardProject for CudaBackend`) — the GPU
//! parallel-beam Radon transform that unlocks the iterative reconstruction suite
//! (SIRT/MLEM/OSEM/OSPML/PML/GRAD/TIKH/TV) on CUDA via the backend-generic
//! solvers in `recon`.
//!
//! Three checks:
//!  1. **Adjoint identity** `⟨A x, y⟩ = ⟨x, Aᵀ y⟩` — the forward projector is the
//!     exact discrete transpose of the back-projector (`cfunc_linerec`). This is
//!     the invariant the iterative solvers rely on; it holds independently of any
//!     geometry/centre convention.
//!  2. **Forward parity vs CPU** — the CUDA forward reads the volume with the
//!     documented y-flip, so `cuda_forward(P) ≈ scale · cpu_forward(flipud(P))`.
//!     Pins the geometry against the CPU reference and the orientation against a
//!     flip regression.
//!  3. **SIRT self-consistency** — reconstruct a phantom from its own CUDA
//!     forward projection and recover it.
//!
//! Only built under `cuda`; needs a real CUDA device (skipped otherwise).
//! Run: `cargo test -p tomoxide --features cuda`.
#![cfg(feature = "cuda")]

use ndarray::{Array3, Axis};
use std::f32::consts::PI;
use tomoxide::{
    recon, sim, Algorithm, Angles, Backend, CpuBackend, CudaBackend, Geometry, Layout, ReconParams,
    Tomo, Volume,
};

/// Deterministic pseudo-random fill in roughly `[-1, 1)` (LCG; no RNG dependency
/// and reproducible across runs).
fn rand_arr(shape: (usize, usize, usize), seed: u64) -> Array3<f32> {
    let mut s = seed;
    Array3::from_shape_fn(shape, |_| {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    })
}

/// Pearson correlation between two equal-length value streams.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x - ma, y - mb);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Stack a 2-D phantom into `nz` identical slices.
fn stack(slice2d: &ndarray::Array2<f32>, nz: usize) -> Array3<f32> {
    let (h, w) = slice2d.dim();
    let mut v = Array3::<f32>::zeros((nz, h, w));
    for z in 0..nz {
        v.index_axis_mut(Axis(0), z).assign(slice2d);
    }
    v
}

/// Flip the y (row) axis of every slice — the documented CUDA handedness.
fn flipud(v: &Array3<f32>) -> Array3<f32> {
    let (nz, n, w) = v.dim();
    let mut out = Array3::<f32>::zeros((nz, n, w));
    for z in 0..nz {
        for r in 0..n {
            for c in 0..w {
                out[[z, n - 1 - r, c]] = v[[z, r, c]];
            }
        }
    }
    out
}

/// Values of a sinogram over interior slices `1..nz` (CUDA drops slice 0 by the
/// shared `vr < nz-1` boundary guard — the documented ≥2-slice rule).
fn interior(sino: &Tomo<f32>) -> Vec<f64> {
    let nz = sino.array.dim().0;
    let mut out = Vec::new();
    for z in 1..nz {
        for v in sino.array.index_axis(Axis(0), z).iter() {
            out.push(*v as f64);
        }
    }
    out
}

#[test]
fn cuda_forward_is_adjoint_of_backproject() {
    let cuda = match CudaBackend::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };
    let (n, nproj, nz) = (64usize, 90usize, 4usize);
    let geom = Geometry::parallel(Angles::uniform(nproj, 0.0, PI), n, nz, 1.0);
    let proj = cuda.projector().expect("cuda projector");
    let bp = cuda.backprojector().expect("cuda backprojector");

    let x = Volume::new(rand_arr((nz, n, n), 0x1234_5678));
    let y = Tomo::new(rand_arr((nz, nproj, n), 0x9abc_def0), Layout::Sinogram);

    // A x  and  Aᵀ y
    let mut ax = Tomo::new(Array3::zeros((nz, nproj, n)), Layout::Sinogram);
    proj.project(&x, &geom, &mut ax).unwrap();
    let mut aty = Volume::new(Array3::zeros((nz, n, n)));
    bp.backproject(&y, &geom, &mut aty).unwrap();

    let lhs: f64 = ax
        .array
        .iter()
        .zip(y.array.iter())
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();
    let rhs: f64 = x
        .array
        .iter()
        .zip(aty.array.iter())
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();
    let rel = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-12);
    eprintln!("adjoint: <Ax,y> = {lhs:.6}, <x,Aᵀy> = {rhs:.6}, rel = {rel:.2e}");
    assert!(
        rel < 2e-3,
        "forward projector is not the transpose of the back-projector: \
         <Ax,y>={lhs:.6}, <x,Aᵀy>={rhs:.6}, rel={rel:.2e}"
    );
}

#[test]
fn cuda_forward_matches_cpu_under_flip() {
    let cuda = match CudaBackend::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (n, nproj, nz) = (128usize, 180usize, 4usize);
    let geom = Geometry::parallel(Angles::uniform(nproj, 0.0, PI), n, nz, 1.0);

    // Shepp–Logan is y-asymmetric, so the flip is detectable.
    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(stack(&phantom, nz));
    let vol_flip = Volume::new(flipud(&vol.array));

    let cuda_sino = sim::project(&vol, &geom, &cuda).unwrap();
    let cpu_sino_flip = sim::project(&vol_flip, &geom, &cpu).unwrap(); // cuda(P) ≈ s·cpu(flipud P)
    let cpu_sino = sim::project(&vol, &geom, &cpu).unwrap(); // wrong handedness

    let g = interior(&cuda_sino);
    let r_ok = pearson(&g, &interior(&cpu_sino_flip));
    let r_wrong = pearson(&g, &interior(&cpu_sino));

    // Best-fit scale cuda/cpu over the interior (≈ 4/nproj; logged, not asserted).
    let p = interior(&cpu_sino_flip);
    let (num, den): (f64, f64) = g
        .iter()
        .zip(p.iter())
        .fold((0.0, 0.0), |(num, den), (&a, &b)| {
            (num + a * b, den + b * b)
        });
    let scale = if den > 0.0 { num / den } else { 0.0 };
    eprintln!(
        "forward parity: flipped r = {r_ok:.6}, wrong-handedness r = {r_wrong:.6}, \
         scale (cuda/cpu) = {scale:.6} (4/nproj = {:.6})",
        4.0 / nproj as f64
    );

    assert!(
        r_ok > 0.999,
        "CUDA forward disagrees with CPU forward (flipud): r = {r_ok:.6} (expected > 0.999)"
    );
    assert!(
        r_wrong < 0.99,
        "orientation pin ineffective: wrong-handedness r = {r_wrong:.6} (expected < 0.99)"
    );
}

#[test]
fn cuda_sirt_recovers_phantom() {
    let cuda = match CudaBackend::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };
    let (n, nproj, nz) = (64usize, 90usize, 4usize);
    let geom = Geometry::parallel(Angles::uniform(nproj, 0.0, PI), n, nz, 1.0);

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(stack(&phantom, nz));

    // Generate the sinogram with the CUDA forward, then reconstruct with CUDA
    // SIRT — the recon lives in the same space as `vol` (the solver feeds its
    // iterate straight back through the same forward), so it recovers it.
    let sino = sim::project(&vol, &geom, &cuda).unwrap();
    let params = ReconParams {
        num_iter: 150,
        num_gridx: Some(n),
        ..Default::default()
    };
    let rec = recon::recon(&sino, &geom, Algorithm::Sirt, &params, &cuda).unwrap();

    // Interior slices only (slice 0 has a zero sinogram from the ≥2-slice rule).
    let mut rv = Vec::new();
    let mut pv = Vec::new();
    for z in 1..nz {
        for (a, b) in rec
            .array
            .index_axis(Axis(0), z)
            .iter()
            .zip(vol.array.index_axis(Axis(0), z).iter())
        {
            rv.push(*a as f64);
            pv.push(*b as f64);
        }
    }
    let r = pearson(&rv, &pv);
    eprintln!("cuda SIRT recovery: r = {r:.6}");
    assert!(
        r > 0.95,
        "CUDA SIRT did not recover the phantom: r = {r:.6} (expected > 0.95)"
    );
}

/// Every iterative algorithm tomocupy lacks (tomopy-only) now runs on CUDA via
/// the single `ForwardProject` primitive. The backend-generic solvers in `recon`
/// pick up `projector()`/`backprojector()` with no dispatch change, so this
/// asserts the whole set reconstructs without error and produces a finite,
/// non-degenerate volume (recovery quality is method-specific; SIRT's is pinned
/// separately above).
#[test]
fn cuda_iterative_suite_runs() {
    let cuda = match CudaBackend::new() {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };
    let (n, nproj, nz) = (64usize, 90usize, 4usize);
    let geom = Geometry::parallel(Angles::uniform(nproj, 0.0, PI), n, nz, 1.0);

    // Non-negative phantom so the line integrals stay ≥ 0 — the multiplicative
    // methods (MLEM/OSEM) and the Poisson-prior methods require it.
    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let vol = Volume::new(stack(&phantom, nz));
    let sino = sim::project(&vol, &geom, &cuda).unwrap();

    let algos = [
        Algorithm::Mlem,
        Algorithm::Osem,
        Algorithm::OspmlQuad,
        Algorithm::PmlQuad,
        Algorithm::OspmlHybrid,
        Algorithm::PmlHybrid,
        Algorithm::Grad,
        Algorithm::Tikh,
        Algorithm::Tv,
    ];
    for algo in algos {
        let params = ReconParams {
            num_iter: 20,
            num_gridx: Some(n),
            // OSPML/PML hybrid use reg_par[1] as the edge threshold; give the
            // priors a benign non-zero regularization so they exercise it.
            reg_par: vec![0.1, 0.01],
            num_block: 2,
            ..Default::default()
        };
        let rec = recon::recon(&sino, &geom, algo, &params, &cuda)
            .unwrap_or_else(|e| panic!("{algo:?} failed on CUDA: {e}"));

        // Finite everywhere, and non-degenerate (nonzero variance) on the
        // interior slices the ≥2-slice rule actually reconstructs.
        assert!(
            rec.array.iter().all(|v| v.is_finite()),
            "{algo:?}: produced non-finite values on CUDA"
        );
        let interior: Vec<f64> = (1..nz)
            .flat_map(|z| {
                rec.array
                    .index_axis(Axis(0), z)
                    .iter()
                    .map(|v| *v as f64)
                    .collect::<Vec<_>>()
            })
            .collect();
        let mean = interior.iter().sum::<f64>() / interior.len() as f64;
        let var = interior.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / interior.len() as f64;
        assert!(
            var > 1e-12,
            "{algo:?}: degenerate (near-constant) recon on CUDA (var = {var:.3e})"
        );
        eprintln!("{algo:?}: ran on CUDA, interior var = {var:.3e}");
    }
}
