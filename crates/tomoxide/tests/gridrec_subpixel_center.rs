//! Regression: gridrec must not collapse at sub-pixel (non-integer) centers.
//!
//! The Fourier-domain recentering shift `exp(2πi·ρ·center/pad)` must key off the
//! SIGNED frequency `ρ`, not the raw FFT bin index. The two agree for integer
//! centers (so the gridrec/tomopy parity test never exercised this), but a raw
//! index multiplies the negative-frequency half by `exp(2πi·center)` = −1 at a
//! half-integer center, cancelling it and collapsing the reconstruction to ~0.
//! Entropy-based `find_center` evaluates gridrec at fractional centers, so it
//! depends on this being correct. Here we assert a half-pixel center change
//! barely perturbs the slice (comparable energy, high correlation) rather than
//! collapsing it.

use ndarray::{Array1, Array3};
use ndarray_npy::read_npy;
use tomoxide::{
    recon, Algorithm, Angles, Beam, Center, CpuBackend, Detector, Geometry, Layout, ReconParams,
    Tomo,
};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn gridrec_at(sino: &Tomo<f32>, theta: &[f32], center: f32, cpu: &CpuBackend) -> Array3<f32> {
    let geom = Geometry {
        angles: Angles(theta.to_vec()),
        center: Center::Scalar(center),
        beam: Beam::Parallel,
        detector: Detector {
            width: sino.n_cols(),
            height: 1,
            pixel_size: 1.0,
        },
    };
    recon::recon(
        sino,
        &geom,
        Algorithm::Gridrec,
        &ReconParams::default(),
        cpu,
    )
    .unwrap()
    .array
}

fn l2(a: &Array3<f32>) -> f64 {
    a.iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt()
}

fn pearson(a: &Array3<f32>, b: &Array3<f32>) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    sab / (saa.sqrt() * sbb.sqrt())
}

#[test]
fn gridrec_subpixel_center_does_not_collapse() {
    let sino: Array3<f32> = read_npy(format!("{FIXTURES}/sino.npy")).unwrap(); // (nang, 1, ncol)
    let theta: Array1<f32> = read_npy(format!("{FIXTURES}/angles.npy")).unwrap();
    let theta = theta.to_vec();
    let cpu = CpuBackend::new();
    let sino = Tomo::new(sino, Layout::Projection); // (nang, row=1, ncol), tomopy proj order

    let r_int = gridrec_at(&sino, &theta, 64.0, &cpu);
    let r_half = gridrec_at(&sino, &theta, 64.5, &cpu);

    let (e_int, e_half) = (l2(&r_int), l2(&r_half));
    let ratio = e_half / e_int;
    let corr = pearson(&r_int, &r_half);

    // Pre-fix the half-integer slice collapsed to ~0 (energy ratio ~4e-3, and
    // the residual noise was uncorrelated with the real slice, r≈0). The fix
    // makes 64.5 a genuine half-pixel shift of 64.0: energy preserved (≈1×) and
    // still well-correlated (a half-pixel shift moves ~9% of the variance).
    assert!(
        (0.7..=1.4).contains(&ratio),
        "half-integer-center energy ratio {ratio} (collapse?) e_int={e_int} e_half={e_half}"
    );
    assert!(
        corr > 0.85,
        "half-integer-center reconstruction decorrelated from integer center: r={corr}"
    );
}
