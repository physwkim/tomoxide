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
        // Pin ramp: these tests check lprec recovery/agreement, calibrated for
        // the sharp filter (the default is parzen); gridrec ignores the filter.
        filter_name: FilterName::Ramp,
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
    // Observed r ≈ 0.9348 (deterministic: fixed phantom, angles, pow2 grid). The
    // 0.93 bar is tighter than the original 0.90 while leaving headroom for f32
    // FFT-rounding variance across rustfft versions/platforms.
    assert!(
        corr > 0.93,
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
    // 0.96 bar confirms the inversion is correct and co-registered while leaving
    // room for that legitimate method difference; observed r ≈ 0.968 over the
    // 0.85 disk (≈0.99 central, ≈0.96 peripheral). A grid / kernel / orientation
    // / interpolation bug drops this far below the 0.96 bar (e.g. the pre-fix theta-order
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
    // Observed r ≈ 0.9676 (deterministic). Tightened from 0.95 to 0.96: still
    // below the genuine method divergence at high frequency, but close enough to
    // catch a regression the 0.95 bar would have let through.
    assert!(cross > 0.96, "lprec and gridrec disagree: r = {cross:.4}");
}

#[test]
fn lprec_reconstructs_at_n256() {
    // Larger power-of-two grid (n=256 → nrho=512, ntheta=128). The other tests
    // exercise n=128; this confirms the precompute (zeta kernel, log-polar grids)
    // and the per-slice pipeline scale to a 4× larger grid without a sizing bug.
    let n = 256;
    let nang = 256;
    let (slice, phantom) = recon_slice(Algorithm::Lprec, n, nang);
    assert_eq!(slice.dim(), (n, n));
    let corr = pearson_disk(&slice, &phantom, n, 0.85);
    eprintln!("lprec n=256 round-trip Pearson = {corr:.4}");
    assert!(
        corr > 0.93,
        "lprec n=256 reconstruction correlates poorly with phantom: r = {corr:.4}"
    );
}

#[test]
fn lprec_handles_non_power_of_two_size() {
    // n = 192 is not a power of two. lprec rounds its *internal* FFT lengths to
    // powers of two by construction (ntheta = 2^round(log2 nproj),
    // nrho = 2·2^round(log2 n)), so this does NOT exercise a non-pow2/Bluestein
    // FFT — the FFT path is byte-identical to the pow2 case. What a non-pow2 size
    // does exercise is the detector/grid sampling at an odd width: the
    // cubic-B-spline prefilter (`convert_to_coeffs`), the 4×4 wrap-addressed
    // `cubic_interp2d`, the `lin[]` Cartesian coordinate ramp, and the unit-disk
    // mask all run at width 192. An off-by-one or stride bug there (which the
    // pow2 widths can mask) drops the correlation below the bar.
    let n = 192;
    let nang = 180;
    let (lp, phantom) = recon_slice(Algorithm::Lprec, n, nang);
    assert_eq!(lp.dim(), (n, n));
    let (gr, _) = recon_slice(Algorithm::Gridrec, n, nang);
    let rt = pearson_disk(&lp, &phantom, n, 0.85);
    let cross = pearson_disk(&lp, &gr, n, 0.85);
    eprintln!("lprec n=192 round-trip = {rt:.4}, lprec↔gridrec = {cross:.4}");
    assert!(rt > 0.92, "lprec n=192 round-trip poor: r = {rt:.4}");
    assert!(
        cross > 0.95,
        "lprec n=192 disagrees with gridrec: r = {cross:.4}"
    );
}

#[test]
fn lprec_rejects_non_square_geometry() {
    // lprec's log-polar grid sizing assumes the reconstruction grid width equals
    // the detector width (square geometry). A request with num_gridx ≠ detector
    // width must be rejected up front, not silently mis-reconstructed onto a
    // mismatched grid.
    let n = 128;
    let nang = 180;
    let cpu = CpuBackend::new();
    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();
    let params = ReconParams {
        num_gridx: Some(100), // ≠ detector width 128 → non-square
        ..Default::default()
    };
    let result = recon::recon(&sino, &geom, Algorithm::Lprec, &params, &cpu);
    assert!(
        result.is_err(),
        "lprec must reject a non-square (num_gridx ≠ detector) geometry"
    );
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("square") || msg.contains("detector width"),
        "unexpected error for non-square geometry: {msg}"
    );
}
