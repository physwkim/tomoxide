//! Rotation-centre-in-filter parity (tomocupy `fbp_filter_center`).
//!
//! `FbpFilter::apply` folds the sub-pixel rotation-centre shift into the shared
//! filter as a per-row Fourier phase, so every analytic back-projector and
//! Fourier grid reconstructs against a centre = `ncols/2` geometry. The
//! centre = `ncols/2` goldens cannot catch a wrong off-centre shift — the phase
//! is unity there — so this test drives a deliberately off-centre acquisition:
//! a Shepp-Logan phantom forward-projected with the rotation axis at a
//! non-midpoint, *sub-pixel* detector column, then reconstructed with fbp /
//! fourierrec / lprec at that same centre.
//!
//! Two checks per method:
//!   1. The off-centre reconstruction still recovers the phantom (the filter
//!      shift co-registers the data with the centre=`ncols/2` back-projector).
//!   2. It matches the centred reconstruction of the same object almost exactly
//!      (the filter shift exactly compensates the acquisition offset; a wrong
//!      sign/magnitude or the raw-vs-signed-frequency trap drops this sharply).

use ndarray::{Array2, Axis};
use std::f32::consts::PI;
use tomoxide::{recon, sim, Algorithm, Angles, Center, CpuBackend, Geometry, ReconParams, Volume};

/// Pearson correlation over a centered disk (scale-invariant; ignores the
/// clipped corners outside the phantom support).
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

/// Forward-project the phantom with the rotation axis at detector column
/// `center`, then reconstruct with `algorithm` at that same center.
fn recon_at_center(algorithm: Algorithm, n: usize, nang: usize, center: f32) -> Array2<f32> {
    let cpu = CpuBackend::new();
    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.insert_axis(Axis(0)));
    let mut geom = Geometry::parallel(Angles::uniform(nang, 0.0, PI), n, 1, 1.0);
    geom.center = Center::Scalar(center);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();
    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };
    let recon = recon::recon(&sino, &geom, algorithm, &params, &cpu).unwrap();
    recon.array.index_axis(Axis(0), 0).to_owned()
}

#[test]
fn analytic_methods_recover_offcenter_phantom() {
    let n = 128;
    let nang = 180;
    // 66.5: off the midpoint (64.0) by 2.5 px — a half-integer (sub-pixel) offset
    // that exercises the Hermitian signed-frequency phase, not just integer shifts.
    let offcenter = n as f32 / 2.0 + 2.5;
    let phantom = sim::shepp2d(n).unwrap();

    for alg in [Algorithm::Fbp, Algorithm::Fourierrec, Algorithm::Lprec] {
        let off = recon_at_center(alg, n, nang, offcenter);
        let centered = recon_at_center(alg, n, nang, n as f32 / 2.0);

        let recovery = pearson_disk(&off, &phantom, n, 0.8);
        let vs_centered = pearson_disk(&off, &centered, n, 0.8);
        eprintln!(
            "{alg:?} off-centre(c={offcenter}): recovery r = {recovery:.4}, vs centred r = {vs_centered:.4}"
        );

        // 1. Off-centre reconstruction recovers the phantom.
        assert!(
            recovery > 0.85,
            "{alg:?} failed to recover the off-centre phantom: r = {recovery:.4}"
        );
        // 2. The filter shift makes it agree with the centred reconstruction of
        //    the same object. Observed: fbp 0.986, lprec 0.985, fourierrec 0.972
        //    (its filter-shift-then-nd-refft path resamples twice, so it is a
        //    touch noisier — but its phantom recovery above matches centred
        //    fourierrec, so it is correct). The 0.96 bar leaves headroom there
        //    while still decisively catching the raw-vs-signed-frequency bug,
        //    which collapses this metric to ~0.
        assert!(
            vs_centered > 0.96,
            "{alg:?} off-centre disagrees with centred recon: r = {vs_centered:.4}"
        );
    }
}
