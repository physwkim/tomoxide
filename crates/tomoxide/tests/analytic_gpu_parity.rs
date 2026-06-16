//! End-to-end GPU↔CPU analytic-reconstruction parity (M6).
//!
//! The per-kernel tests prove each wgpu capability correct in isolation; these
//! prove they *compose* into full reconstructions driven through
//! `recon::recon(.., &dyn Backend)`:
//!   - **FBP** exercises `FbpFilter::apply` then `FilteredBackproject`.
//!   - **gridrec** exercises only the `Fft` capability (the Kaiser-Bessel
//!     gridding/deapodization is host code shared by both backends); it runs on
//!     the GPU for free because every gridrec transform length is power-of-two
//!     (`pad = (2·ncols).next_power_of_two()`, grid `m = pad`).
//!   - **fourierrec** (tomocupy Gaussian-USFFT gridding) also composes through
//!     `Fft` alone. At `n = 128` its transforms are power-of-two; at `n = 96`
//!     the radial 1-D FFT is length 96 (Bluestein) and the 2-D FFT is 192×192
//!     (separable Bluestein), so it additionally exercises the arbitrary-length
//!     GPU FFT paths end to end.
//!
//! Each asserts the GPU reconstruction (1) actually reconstructs the phantom
//! and (2) matches the CPU reconstruction within f32 tolerance.
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

/// Reconstruct a Shepp-Logan phantom with `algorithm` on both backends.
///
/// The phantom is forward-projected once on the CPU; both backends reconstruct
/// the identical sinogram (`ncols = n`). Returns `(gpu_slice, cpu_slice,
/// phantom)`. Output grid is `n×n` (`num_gridx = n`).
fn recon_both(
    algorithm: Algorithm,
    n: usize,
    nang: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let cpu = CpuBackend::new();
    let gpu = WgpuBackend::new().expect("wgpu device init");

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };
    let rc = recon::recon(&sino, &geom, algorithm, &params, &cpu).unwrap();
    let rg = recon::recon(&sino, &geom, algorithm, &params, &gpu).unwrap();
    assert_eq!(rg.array.dim(), (1, n, n));

    let sc = rc.array.index_axis(Axis(0), 0).to_owned();
    let sg = rg.array.index_axis(Axis(0), 0).to_owned();
    (sg, sc, phantom)
}

#[test]
fn fbp_recon_matches_cpu_on_gpu() {
    // ncols = 128 → the ramp filter pads to 256, a power of two the radix-2 GPU
    // FFT handles. Exercises FbpFilter::apply ∘ FilteredBackproject on the GPU.
    let n = 128;
    let (sg, sc, phantom) = recon_both(Algorithm::Fbp, n, 180);

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

#[test]
fn gridrec_recon_matches_cpu_on_gpu() {
    // gridrec needs only the Fft capability; every transform length is
    // power-of-two (pad = (2·128).next = 256, grid m = 256), so it runs on the
    // GPU with no extra kernels — only the FFT backend differs from CPU.
    let n = 128;
    let (sg, sc, phantom) = recon_both(Algorithm::Gridrec, n, 180);

    let corr = pearson_disk(&sg, &phantom, n, 0.85);
    eprintln!("GPU gridrec Pearson vs phantom = {corr:.4}");
    assert!(
        corr > 0.9,
        "GPU gridrec correlates poorly with phantom: r = {corr:.4}"
    );

    // Both backends run identical host gridding/deapodization, differing only in
    // the FFT (wgpu radix-2 vs rustfft) on a 256-pt radial 1-D and a 256×256
    // 2-D transform. Observed NRMSE ≈ 3.4e-7 on Metal (max|Δ| ≈ 3e-10 — the host
    // gridding dominates, the FFT-backend difference is negligible); the 1e-4
    // bar gives generous cross-adapter headroom yet is orders of magnitude
    // tighter than any wiring bug (wrong FFT size/direction) would produce.
    let (nrmse, maxabs) = disk_nrmse(&sg, &sc, n, 0.8);
    eprintln!("GPU vs CPU gridrec: NRMSE = {nrmse:.3e}, max|Δ| = {maxabs:.3e}");
    assert!(
        nrmse < 1e-4,
        "GPU vs CPU gridrec NRMSE too large: {nrmse:.3e}"
    );
}

#[test]
fn fourierrec_recon_matches_cpu_on_gpu() {
    // fourierrec needs only the Fft capability (FBP filter then Gaussian-USFFT
    // gridding, all host code shared by both backends). At n=128 the radial 1-D
    // FFT is length 128 and the 2-D inverse FFT is 256×256 — both power-of-two,
    // so the radix-2 GPU path runs it for free, only the FFT backend differs.
    let n = 128;
    let (sg, sc, phantom) = recon_both(Algorithm::Fourierrec, n, 180);

    let corr = pearson_disk(&sg, &phantom, n, 0.85);
    eprintln!("GPU fourierrec Pearson vs phantom = {corr:.4}");
    assert!(
        corr > 0.9,
        "GPU fourierrec correlates poorly with phantom: r = {corr:.4}"
    );

    let (nrmse, maxabs) = disk_nrmse(&sg, &sc, n, 0.8);
    eprintln!("GPU vs CPU fourierrec: NRMSE = {nrmse:.3e}, max|Δ| = {maxabs:.3e}");
    assert!(
        nrmse < 1e-4,
        "GPU vs CPU fourierrec NRMSE too large: {nrmse:.3e}"
    );
}

#[test]
fn lprec_recon_matches_cpu_on_gpu() {
    // lprec (log-polar method) needs only the Fft capability: the precompute
    // 1-D FFTs (lengths ntheta=128, nrho=256, Nthetalarge=512) and the runtime
    // 2-D convolution (256×128) are all power-of-two at n=128, so the radix-2 GPU
    // path runs the whole reconstruction with no extra kernels — only the FFT
    // backend (wgpu vs rustfft) differs from CPU. The cubic prefilter / gather /
    // resample are host code shared by both backends.
    let n = 128;
    let (sg, sc, phantom) = recon_both(Algorithm::Lprec, n, 180);

    let corr = pearson_disk(&sg, &phantom, n, 0.85);
    eprintln!("GPU lprec Pearson vs phantom = {corr:.4}");
    assert!(
        corr > 0.9,
        "GPU lprec correlates poorly with phantom: r = {corr:.4}"
    );

    // The only backend difference is the FFT (several power-of-two 1-D transforms
    // in precompute plus the 256×128 2-D convolution per span), so GPU↔CPU agree
    // to f32-FFT tolerance. The 1e-4 bar matches the other analytic GPU tests and
    // is far tighter than any wiring bug (wrong FFT size/direction/layout).
    let (nrmse, maxabs) = disk_nrmse(&sg, &sc, n, 0.8);
    eprintln!("GPU vs CPU lprec: NRMSE = {nrmse:.3e}, max|Δ| = {maxabs:.3e}");
    assert!(nrmse < 1e-4, "GPU vs CPU lprec NRMSE too large: {nrmse:.3e}");
}

#[test]
fn fourierrec_non_power_of_two_recon_matches_cpu_on_gpu() {
    // n=96 drives fourierrec through both arbitrary-length GPU FFT paths: the
    // radial 1-D transform is length 96 (Bluestein chirp-z) and the 2-D inverse
    // transform is 192×192 (separable Bluestein with a host transpose). This is
    // the end-to-end proof that the non-power-of-two FFT generalization composes
    // into a full reconstruction, not just isolated FFT round-trips.
    let n = 96;
    let (sg, sc, phantom) = recon_both(Algorithm::Fourierrec, n, 150);

    let corr = pearson_disk(&sg, &phantom, n, 0.85);
    eprintln!("GPU fourierrec(96) Pearson vs phantom = {corr:.4}");
    assert!(
        corr > 0.9,
        "GPU fourierrec(96) correlates poorly with phantom: r = {corr:.4}"
    );

    // Although the GPU runs Bluestein here (≈1e-6 rel error vs rustfft, vs the
    // radix-2 path's ≈1e-7), the host gridding dominates the reconstruction, so
    // the GPU↔CPU difference stays at the pow2 level — observed NRMSE ≈ 3.2e-7,
    // essentially identical to the n=128 case. The 1e-4 bar therefore holds the
    // same ~300× headroom while staying far tighter than any wiring bug.
    let (nrmse, maxabs) = disk_nrmse(&sg, &sc, n, 0.8);
    eprintln!("GPU vs CPU fourierrec(96): NRMSE = {nrmse:.3e}, max|Δ| = {maxabs:.3e}");
    assert!(
        nrmse < 1e-4,
        "GPU vs CPU fourierrec(96) NRMSE too large: {nrmse:.3e}"
    );
}
