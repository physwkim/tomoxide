//! fourierrec (tomocupy USFFT Gaussian-gridding) reconstruction parity.
//!
//! tomocupy's `fourierrec` runs on CUDA, so a bit-for-bit golden is unavailable
//! offline. Instead this verifies correctness two ways that need no CUDA:
//!   1. Round-trip — forward-project a Shepp-Logan phantom, reconstruct with
//!      `fourierrec`, and require high Pearson correlation with the phantom over
//!      a central disk (scale-invariant, so the gridding amplitude constants
//!      don't matter).
//!   2. Cross-method — `fourierrec` and the already-verified `gridrec` are both
//!      central-slice-theorem methods (Gaussian vs Kaiser-Bessel gridding), so
//!      on the same data they must agree closely. This catches a gather /
//!      deapodize / centring bug that a single-method round-trip might mask.

use ndarray::{Array2, Axis};
use tomoxide::{
    recon, sim, Algorithm, Angles, CpuBackend, FilterName, Geometry, ReconParams, Volume,
};

/// Pearson correlation between two slices over a centered disk of the given
/// radius fraction (kept inside the phantom support, away from clipped corners).
fn pearson_disk(a: &Array2<f32>, b: &Array2<f32>, n: usize, radius_frac: f32) -> f32 {
    let c = (n as f32 - 1.0) / 2.0;
    let r = radius_frac * (n as f32 / 2.0);
    let r2 = r * r;
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for iy in 0..n {
        for ix in 0..n {
            let dy = iy as f32 - c;
            let dx = ix as f32 - c;
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

/// Reconstruct a forward-projected Shepp-Logan phantom with `algorithm`.
fn recon_slice(algorithm: Algorithm, n: usize, nang: usize) -> (Array2<f32>, Array2<f32>) {
    let cpu = CpuBackend::new();
    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();
    let params = ReconParams {
        num_gridx: Some(n),
        // Pin ramp: these tests check fourierrec recovery/agreement, calibrated
        // for the sharp filter (the default is parzen); gridrec ignores the filter.
        filter_name: FilterName::Ramp,
        ..Default::default()
    };
    let recon = recon::recon(&sino, &geom, algorithm, &params, &cpu).unwrap();
    let slice = recon.array.index_axis(Axis(0), 0).to_owned();
    (slice, phantom)
}

#[test]
fn fourierrec_reconstructs_shepp_logan_phantom() {
    let n = 128;
    let nang = 180;
    let (slice, phantom) = recon_slice(Algorithm::Fourierrec, n, nang);
    assert_eq!(slice.dim(), (n, n));
    let corr = pearson_disk(&slice, &phantom, n, 0.85);
    eprintln!("fourierrec round-trip Pearson correlation = {corr:.4}");
    assert!(
        corr > 0.9,
        "fourierrec reconstruction correlates poorly with phantom: r = {corr:.4}"
    );
}

/// Best-fit scale `a` minimizing `‖x − a·y‖` over the centered disk.
fn fit_scale(x: &Array2<f32>, y: &Array2<f32>, n: usize, radius_frac: f32) -> f64 {
    let c = (n as f32 - 1.0) / 2.0;
    let r = radius_frac * (n as f32 / 2.0);
    let r2 = r * r;
    let (mut sxy, mut syy) = (0.0f64, 0.0f64);
    for iy in 0..n {
        for ix in 0..n {
            let dy = iy as f32 - c;
            let dx = ix as f32 - c;
            if dx * dx + dy * dy <= r2 {
                sxy += (x[[iy, ix]] as f64) * (y[[iy, ix]] as f64);
                syy += (y[[iy, ix]] as f64) * (y[[iy, ix]] as f64);
            }
        }
    }
    sxy / syy
}

#[test]
fn fourierrec_matches_fbp_amplitude() {
    // AMPLITUDE pin — every check above is Pearson (scale-invariant), which is
    // exactly how a π·nd²-sized scale defect hid in fourierrec: its output was
    // uniformly ~2×10⁶× smaller than fbp on 800-wide data (read as "all zeros")
    // while every parity test passed. All methods emit the tomopy/fbp amplitude
    // (Phase 2 scale convention), so the best-fit fbp/fourierrec scale must be ≈1;
    // the residual is the ramp-shape + USFFT-deapodization spectral gap.
    let n = 128;
    let nang = 180;
    let (fr, _) = recon_slice(Algorithm::Fourierrec, n, nang);
    let (fbp, _) = recon_slice(Algorithm::Fbp, n, nang);
    let scale = fit_scale(&fbp, &fr, n, 0.85);
    eprintln!("fbp/fourierrec amplitude scale = {scale:.5}");
    assert!(
        (scale - 1.0).abs() < 0.05,
        "fourierrec amplitude differs from fbp: fbp/fourierrec = {scale:.5} \
         (expected ≈ 1; a fourierrec normalization/quadrature factor regressed)"
    );
}

#[test]
fn fourierrec_agrees_with_gridrec() {
    // Both are central-slice-theorem direct methods; on the same forward-projected
    // phantom their reconstructions must be nearly identical (only the gridding
    // kernel differs). A gather / wrap / deapodize / centring bug in fourierrec
    // would break this agreement even while the single-method round-trip passes.
    let n = 128;
    let nang = 180;
    let (fr, phantom) = recon_slice(Algorithm::Fourierrec, n, nang);
    let (gr, _) = recon_slice(Algorithm::Gridrec, n, nang);
    let cross = pearson_disk(&fr, &gr, n, 0.85);
    let fr_corr = pearson_disk(&fr, &phantom, n, 0.85);
    let gr_corr = pearson_disk(&gr, &phantom, n, 0.85);
    eprintln!(
        "fourierrec↔gridrec r = {cross:.4} (fourierrec↔phantom {fr_corr:.4}, gridrec↔phantom {gr_corr:.4})"
    );
    assert!(
        cross > 0.97,
        "fourierrec and gridrec disagree: r = {cross:.4}"
    );
}
