//! Parity against tomopy for Vo fitting-based stripe removal
//! (`remove_stripe_based_fitting`, Vo 2018 algorithm 1).
//!
//! `_rs_fit` divides each sinogram by its Savitzky-Golay polynomial fit along
//! the projection axis, then re-multiplies by a mean-matched 2-D Gaussian-
//! smoothed copy of that fit (`_2d_filter`, an `ifft2(fft2(.)·win2d)` band-pass
//! with `(-1)^(x+y)` modulation and edge/mean padding). The Savitzky-Golay
//! weights are reproduced from scaled normal equations (matching scipy's SVD
//! `lstsq` to the f64 floor) and the 2-D Fourier filter runs in f64, so — like
//! the Fourier-Wavelet and VoFilter paths — it is held to the f32 round-off
//! floor, not bit-exactness.
//!
//! Two cases: order=3 sigma=(5,20) (tomopy defaults) and order=1 sigma=(3,10).
//! Goldens from the real tomopy 1.15.3 (`tools/gen_tomopy_stripe_vofit_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, Layout, StripeMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

/// Column-roughness proxy for stripe energy: the variance of the column-to-column
/// differences of the per-column mean (over angles and rows). A detector-gain
/// stripe is a column offset from its neighbours, so it inflates this and its
/// removal shrinks it.
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

fn run_case(golden_name: &str, order: usize, sigma: (f32, f32)) {
    let input = load("stripe_vofit_input.npy");
    let golden = load(golden_name);

    let mut tomo = Tomo::new(input.clone(), Layout::Projection);
    prep::stripe::remove_stripe(&mut tomo, StripeMethod::VoFit { order, sigma }).unwrap();
    let got = tomo.to_layout(Layout::Projection).array;
    assert_eq!(got.dim(), golden.dim());

    // (1) Agreement with the tomopy golden, to the f32 round-off floor.
    let scale = golden.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_abs = golden
        .iter()
        .zip(got.iter())
        .fold(0.0f32, |m, (&g, &p)| m.max((g - p).abs()));
    let max_rel = max_abs / scale;
    eprintln!("Vo-fit order={order}: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})");
    assert!(
        max_rel <= 1e-5,
        "Vo-fit order={order} golden parity: max|Δ| = {max_abs}, max relative = {max_rel}"
    );

    // (2) The injected detector-gain stripes (cols 30/75/100/101) are suppressed.
    let before = stripe_roughness(&input);
    let after = stripe_roughness(&got);
    assert!(
        after < before * 0.2,
        "Vo-fit order={order} did not reduce stripe roughness enough: before = {before}, after = {after}"
    );
}

#[test]
fn vofit_default_matches_tomopy() {
    run_case("tomopy_stripe_vofit_def.npy", 3, (5.0, 20.0));
}

#[test]
fn vofit_order1_matches_tomopy() {
    run_case("tomopy_stripe_vofit_o1.npy", 1, (3.0, 10.0));
}
