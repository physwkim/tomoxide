//! End-to-end GPU↔CPU iterative-reconstruction parity (M6).
//!
//! SIRT applies the forward projector *and* the back-projector every iteration,
//! so this is the first test that exercises the wgpu `ForwardProject` kernel in
//! a real recon composition (the per-kernel test only runs it in isolation), and
//! the first to run project∘back-project in a loop on the GPU. Driven through
//! `recon::recon(Algorithm::Sirt, &dyn Backend)` with no recon-crate changes:
//! the solver, its R/C weight maps, and the residual updates all dispatch their
//! projections to whichever backend is passed.
//!
//! Only built under `gpu-wgpu`; needs a real GPU adapter (skipped by the default
//! workspace run). Run: `cargo test -p tomoxide --features gpu-wgpu`.
#![cfg(feature = "gpu-wgpu")]

use ndarray::{Array2, Axis};
use tomoxide::{recon, sim, Algorithm, Angles, CpuBackend, Geometry, ReconParams, Volume};
use tomoxide_wgpu::WgpuBackend;

/// Pearson correlation between two slices over a centered disk (amplitude-scale
/// invariant), kept inside the phantom support away from clipped corners.
fn pearson_disk(a: &Array2<f32>, b: &Array2<f32>, n: usize, radius_frac: f32) -> f32 {
    let c = (n as f32 - 1.0) / 2.0;
    let r2 = (radius_frac * (n as f32 / 2.0)).powi(2);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for iy in 0..n {
        for ix in 0..n {
            let (dy, dx) = (iy as f32 - c, ix as f32 - c);
            if dx * dx + dy * dy <= r2 {
                xs.push(a[[iy, ix]]);
                ys.push(b[[iy, ix]]);
            }
        }
    }
    let nn = xs.len() as f32;
    let mx = xs.iter().sum::<f32>() / nn;
    let my = ys.iter().sum::<f32>() / nn;
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let (dx, dy) = (x - mx, y - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Normalized RMS error and max abs difference of `a` vs reference `b` over a
/// centered disk. NRMSE = sqrt(mean((a−b)²)) / sqrt(mean(b²)).
fn disk_nrmse(a: &Array2<f32>, b: &Array2<f32>, n: usize, radius_frac: f32) -> (f32, f32) {
    let c = (n as f32 - 1.0) / 2.0;
    let r2 = (radius_frac * (n as f32 / 2.0)).powi(2);
    let (mut se, mut sb, mut maxabs, mut cnt) = (0.0f32, 0.0f32, 0.0f32, 0usize);
    for iy in 0..n {
        for ix in 0..n {
            let (dy, dx) = (iy as f32 - c, ix as f32 - c);
            if dx * dx + dy * dy <= r2 {
                let d = a[[iy, ix]] - b[[iy, ix]];
                se += d * d;
                sb += b[[iy, ix]] * b[[iy, ix]];
                maxabs = maxabs.max(d.abs());
                cnt += 1;
            }
        }
    }
    let nn = cnt as f32;
    ((se / nn).sqrt() / (sb / nn).sqrt(), maxabs)
}

#[test]
fn sirt_recon_matches_cpu_on_gpu() {
    let n = 64;
    let nang = 90;
    let iters = 100;
    let cpu = CpuBackend::new();
    let gpu = WgpuBackend::new().expect("wgpu device init");

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        num_iter: iters,
        ..Default::default()
    };
    let rc = recon::recon(&sino, &geom, Algorithm::Sirt, &params, &cpu).unwrap();
    let rg = recon::recon(&sino, &geom, Algorithm::Sirt, &params, &gpu).unwrap();
    assert_eq!(rg.array.dim(), (1, n, n));

    let sc = rc.array.index_axis(Axis(0), 0).to_owned();
    let sg = rg.array.index_axis(Axis(0), 0).to_owned();

    // (1) The GPU reconstruction is as good as the CPU one — this is a parity
    //     test, so the bar is "GPU does not degrade quality", not an absolute
    //     convergence threshold (that is the CPU iterative_roundtrip test's job).
    //     Both forward/back-project on their respective backend each iteration.
    let corr_gpu = pearson_disk(&sg, &phantom, n, 0.8);
    let corr_cpu_phantom = pearson_disk(&sc, &phantom, n, 0.8);
    eprintln!("SIRT Pearson vs phantom: GPU = {corr_gpu:.4}, CPU = {corr_cpu_phantom:.4}");
    assert!(
        corr_gpu > corr_cpu_phantom - 0.02,
        "GPU SIRT reconstructs worse than CPU: GPU r = {corr_gpu:.4} vs CPU r = {corr_cpu_phantom:.4}"
    );

    // (2) GPU and CPU SIRT agree. Unlike the single-pass analytic methods, SIRT
    //     applies project∘back-project every iteration, so per-step f32
    //     differences (~1e-4) accumulate over {iters} iterations — the tolerance
    //     is necessarily looser than the analytic 1e-4, but the two
    //     reconstructions must still track closely. Observed on Metal: corr
    //     1.00000, NRMSE 1.8e-4 (the per-step differences largely cancel as SIRT
    //     converges to the same fixed point); the 5e-3 bar leaves ~28× headroom
    //     for iteration-count/adapter variation, far tighter than a wiring bug.
    let corr_cpu = pearson_disk(&sg, &sc, n, 0.8);
    let (nrmse, maxabs) = disk_nrmse(&sg, &sc, n, 0.8);
    eprintln!("GPU vs CPU SIRT: corr = {corr_cpu:.5}, NRMSE = {nrmse:.3e}, max|Δ| = {maxabs:.3e}");
    assert!(
        corr_cpu > 0.999,
        "GPU vs CPU SIRT correlation too low: {corr_cpu:.5}"
    );
    assert!(nrmse < 5e-3, "GPU vs CPU SIRT NRMSE too large: {nrmse:.3e}");
}
