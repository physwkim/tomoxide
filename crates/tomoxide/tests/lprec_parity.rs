//! lprec (log-polar Andersson–Carlsson–Nikitin) reconstruction parity.
//!
//! tomocupy's `lprec` runs on CUDA, so a bit-for-bit golden is unavailable
//! offline. This verifies correctness two ways that need no CUDA:
//!   1. Round-trip — forward-project a Shepp-Logan phantom, reconstruct with
//!      `lprec`, and require high Pearson correlation with the phantom over a
//!      central disk (scale-invariant, so the kernel amplitude constants don't
//!      matter).
//!   2. Cross-method — `lprec` and the already-verified `gridrec` are different
//!      analytic inversions (log-polar FFT convolution vs central-slice
//!      gridding), so on the same data they must agree closely. This catches a
//!      grid / kernel / interpolation bug a single round-trip might mask.

use ndarray::{Array2, Axis};
use tomoxide::{recon, sim, Algorithm, Angles, CpuBackend, Geometry, ReconParams, Volume};

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
        ..Default::default()
    };
    let recon = recon::recon(&sino, &geom, algorithm, &params, &cpu).unwrap();
    let slice = recon.array.index_axis(Axis(0), 0).to_owned();
    (slice, phantom)
}

#[test]
fn lprec_reconstructs_shepp_logan_phantom() {
    let n = 128;
    let nang = 180;
    let (slice, phantom) = recon_slice(Algorithm::Lprec, n, nang);
    assert_eq!(slice.dim(), (n, n));
    let corr = pearson_disk(&slice, &phantom, n, 0.85);
    eprintln!("lprec round-trip Pearson correlation = {corr:.4}");
    assert!(
        corr > 0.9,
        "lprec reconstruction correlates poorly with phantom: r = {corr:.4}"
    );
}

#[test]
fn lprec_agrees_with_gridrec() {
    // lprec (log-polar FFT convolution) and gridrec (Kaiser-Bessel central-slice
    // gridding) are *genuinely different* analytic inversions, so unlike the
    // fourierrec↔gridrec pair (two gridding methods, r ≈ 1.0) they agree only to
    // ~0.97, not bit-for-bit: the log-polar resampling and the differing
    // apodization diverge at high spatial frequency (the disk periphery). The
    // 0.95 bar confirms the inversion is correct and co-registered while leaving
    // room for that legitimate method difference; observed r ≈ 0.968 over the
    // 0.85 disk (≈0.99 central, ≈0.96 peripheral). A grid / kernel / orientation
    // / interpolation bug drops this far below 0.95 (e.g. the pre-fix theta-order
    // bug gave 0.01, the vertical-flip bug 0.58).
    let n = 128;
    let nang = 180;
    let (lp, phantom) = recon_slice(Algorithm::Lprec, n, nang);
    let (gr, _) = recon_slice(Algorithm::Gridrec, n, nang);
    let cross = pearson_disk(&lp, &gr, n, 0.85);
    let lp_corr = pearson_disk(&lp, &phantom, n, 0.85);
    let gr_corr = pearson_disk(&gr, &phantom, n, 0.85);
    eprintln!(
        "lprec↔gridrec r = {cross:.4} (lprec↔phantom {lp_corr:.4}, gridrec↔phantom {gr_corr:.4})"
    );
    assert!(cross > 0.95, "lprec and gridrec disagree: r = {cross:.4}");
}
