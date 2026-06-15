//! Numeric parity against tomopy for Vo all-stripe removal.
//!
//! `remove_all_stripe` (Vo algorithms 3+5+6) is projector-independent but
//! composes several scipy primitives (uniform_filter1d, median_filter, polyfit,
//! RectBivariateSpline) whose summation/fit numerics differ slightly from this
//! reimplementation, so it is held to a tolerance plus a stripe-reduction
//! metric, not to bit-exactness. Goldens from tomopy 1.15.3 `remove_all_stripe`
//! (`tools/gen_tomopy_stripe_voall_golden.py`, la_size=61, sm_size=21).
//!
//! Two cases over the same input exercise the distinct code paths:
//!   * `snr = 3` (tomopy default) — large-stripe removal (cols 30/75/100) +
//!     sorting. The dead-column `val2` gate caps below 3, so the bilinear fill
//!     does not fire (matching tomopy).
//!   * `snr = 2` — the above plus the dead-column path: cols 54/55/56 are
//!     detected and filled by the `kx = ky = 1` RectBivariateSpline, exercising
//!     that branch.

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, Layout, StripeMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

/// Column-roughness proxy for stripe energy: the variance of the
/// column-to-column differences of the per-column mean (over angles and rows).
/// A stripe is a column whose mean is offset from its neighbours, so injected
/// stripes inflate this and their removal must shrink it.
fn stripe_roughness(a: &Array3<f32>) -> f64 {
    let (np, nr, nc) = a.dim();
    let mut col_mean = vec![0.0f64; nc];
    for p in 0..np {
        for r in 0..nr {
            for c in 0..nc {
                col_mean[c] += a[[p, r, c]] as f64;
            }
        }
    }
    let denom = (np * nr) as f64;
    for v in col_mean.iter_mut() {
        *v /= denom;
    }
    let diffs: Vec<f64> = col_mean.windows(2).map(|w| w[1] - w[0]).collect();
    let mean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
    diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64
}

fn run_case(snr: f32, golden_name: &str) {
    let input = load("stripe_voall_input.npy");
    let golden = load(golden_name);

    let mut tomo = Tomo::new(input.clone(), Layout::Projection);
    prep::stripe::remove_stripe(
        &mut tomo,
        StripeMethod::VoAll {
            snr,
            la_size: 61,
            sm_size: 21,
        },
    )
    .unwrap();
    let got = tomo.array;

    // (1) Agreement with the tomopy golden, to the f32 round-off / scipy-numeric
    //     floor. Measured max|Δ| ≈ 1e-6 absolute, ~6e-7 relative on this fixture.
    let scale = golden.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_abs = golden
        .iter()
        .zip(got.iter())
        .fold(0.0f32, |m, (&g, &p)| m.max((g - p).abs()));
    let max_rel = max_abs / scale;
    eprintln!("Vo-all snr={snr}: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})");
    // Hold to 1e-5 relative — a small safety margin over the observed ~6e-7
    // floor, still far below anything that would mask a logic divergence (an
    // argsort/scatter mismatch lands at ~1e-1+).
    assert!(
        max_rel <= 1e-5,
        "Vo-all snr={snr} golden parity: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})"
    );

    // (2) The injected stripes are actually suppressed: the column-roughness
    //     metric drops by a large factor versus the striped input.
    let before = stripe_roughness(&input);
    let after = stripe_roughness(&got);
    assert!(
        after < before * 0.2,
        "Vo-all snr={snr} did not reduce stripe roughness enough: before = {before}, after = {after}"
    );
}

#[test]
fn remove_all_stripe_matches_tomopy_snr3() {
    run_case(3.0, "tomopy_stripe_voall_snr3.npy");
}

#[test]
fn remove_all_stripe_matches_tomopy_snr2() {
    run_case(2.0, "tomopy_stripe_voall_snr2.npy");
}
