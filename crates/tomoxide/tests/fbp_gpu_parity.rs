//! End-to-end GPU↔CPU FBP parity (M6).
//!
//! The per-kernel tests prove each wgpu capability correct in isolation; this
//! proves they *compose* — that `FbpFilter::apply` then `FilteredBackproject`,
//! driven through `recon::recon(Algorithm::Fbp, &dyn Backend)`, produce a full
//! reconstruction on the GPU that (1) actually reconstructs the phantom and
//! (2) matches the CPU reconstruction within f32 tolerance. This is the real
//! validation that the full GPU FBP path is closed.
//!
//! Only built under `gpu-wgpu`; needs a real GPU adapter (skipped by the
//! default workspace run). Run: `cargo test -p tomoxide --features gpu-wgpu`.
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
fn fbp_recon_matches_cpu_on_gpu() {
    let n = 128;
    let nang = 180;
    let cpu = CpuBackend::new();
    let gpu = WgpuBackend::new().expect("wgpu device init");

    // Single-slice Shepp-Logan phantom, forward-projected once on the CPU; both
    // backends reconstruct the identical sinogram (ncols = 128 → filter pads to
    // 256, a power of two the radix-2 GPU FFT handles).
    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };
    let rc = recon::recon(&sino, &geom, Algorithm::Fbp, &params, &cpu).unwrap();
    let rg = recon::recon(&sino, &geom, Algorithm::Fbp, &params, &gpu).unwrap();
    assert_eq!(rg.array.dim(), (1, n, n));

    let sc = rc.array.index_axis(Axis(0), 0).to_owned();
    let sg = rg.array.index_axis(Axis(0), 0).to_owned();

    // (1) The GPU reconstruction is itself a faithful reconstruction (not just a
    //     match to a possibly-wrong CPU output).
    let corr = pearson_disk(&sg, &phantom, n, 0.85);
    eprintln!("GPU FBP Pearson vs phantom = {corr:.4}");
    assert!(
        corr > 0.9,
        "GPU FBP correlates poorly with phantom: r = {corr:.4}"
    );

    // (2) GPU and CPU reconstructions agree over an interior disk (radius 0.8 ⇒
    //     all rays stay inside the detector, away from the edge-inclusion cutoff
    //     where GPU/CPU t-rounding can flip a whole boundary sample). Proves the
    //     filter→back-project composition is bit-faithful up to f32 tolerance.
    // Observed NRMSE ≈ 1.2e-6 on Metal (the radix-2 GPU FFT and rustfft, plus
    // the back-projection sum, agree to ~6 digits); the 1e-4 bar leaves ~80×
    // headroom for cross-adapter twiddle differences while still being ~100×
    // tighter than any composition bug (layout swap, wrong crop) would produce.
    let (nrmse, maxabs) = disk_nrmse(&sg, &sc, n, 0.8);
    eprintln!("GPU vs CPU FBP: NRMSE = {nrmse:.3e}, max|Δ| = {maxabs:.3e}");
    assert!(nrmse < 1e-4, "GPU vs CPU FBP NRMSE too large: {nrmse:.3e}");
}
