//! Parity against tomopy for Vo large-stripe removal
//! (`remove_large_stripe`, Vo 2018 algorithm 5).
//!
//! `_rs_large` sorts each detector column over projections, median-smooths the
//! sorted profile, detects the wide-stripe columns, and overwrites only those
//! with the rank-smoothed profile mapped back through the sort order. The
//! smoothed values are pure rank-filter *selections* of existing f32 samples, so
//! the parity bar depends on `norm`:
//!   * `norm = false` — the whole result is selections/copies of the input, so
//!     it matches tomopy bit-for-bit (Δ = 0).
//!   * `norm = true`  — the unmasked columns are additionally divided by their
//!     per-column intensity factor (arithmetic), so it is held to the f32
//!     round-off floor, like VoAll.
//!
//! Goldens from the real tomopy 1.15.3 (`tools/gen_tomopy_stripe_volarge_golden.py`,
//! snr=3, size=51, drop_ratio=0.1).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, Layout, StripeMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

/// Column-roughness proxy for stripe energy: the variance of the
/// column-to-column differences of the per-column mean (over angles and rows).
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

fn run_case(golden_name: &str, norm: bool, max_rel_bound: f32) {
    let input = load("stripe_volarge_input.npy");
    let golden = load(golden_name);

    // Input is tomopy projection order [nproj, nslices, ncol].
    let mut tomo = Tomo::new(input.clone(), Layout::Projection);
    prep::stripe::remove_stripe(
        &mut tomo,
        StripeMethod::VoLarge {
            snr: 3.0,
            size: 51,
            drop_ratio: 0.1,
            norm,
        },
    )
    .unwrap();
    let got = tomo.to_layout(Layout::Projection).array;
    assert_eq!(got.dim(), golden.dim());

    let scale = golden.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_abs = golden
        .iter()
        .zip(got.iter())
        .fold(0.0f32, |m, (&g, &p)| m.max((g - p).abs()));
    let max_rel = max_abs / scale;
    eprintln!("Vo-large norm={norm}: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})");
    assert!(
        max_rel <= max_rel_bound,
        "Vo-large norm={norm} golden parity: max|Δ| = {max_abs}, max relative = {max_rel} (bound {max_rel_bound})"
    );

    // The injected stripes (cols 30/75/100) are actually suppressed.
    let before = stripe_roughness(&input);
    let after = stripe_roughness(&got);
    assert!(
        after < before * 0.2,
        "Vo-large norm={norm} did not reduce stripe roughness enough: before = {before}, after = {after}"
    );
}

#[test]
fn volarge_norm_true_matches_tomopy() {
    // norm=True divides the unmasked columns by their f32 factor → f32 floor.
    // 1e-5 relative is a margin over the observed floor, far below a logic
    // divergence (an argsort/scatter mismatch lands at ~1e-1+).
    run_case("tomopy_stripe_volarge_norm.npy", true, 1e-5);
}

#[test]
fn volarge_norm_false_matches_tomopy_bit_exact() {
    // norm=False is pure selection/copy → bit-exact.
    run_case("tomopy_stripe_volarge_raw.npy", false, 0.0);
}
