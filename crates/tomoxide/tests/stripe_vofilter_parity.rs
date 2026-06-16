//! Numeric parity against tomopy for Vo filtering-based stripe removal
//! (`remove_stripe_based_filtering`, Vo 2018 algorithm 2).
//!
//! `_rs_filter` separates a low-pass (smooth) component with a Gaussian Fourier
//! filter along the projection axis, applies the sorting-based correction
//! (`_rs_sort`) to that component, then adds back the high-pass residual. tomopy
//! runs the Fourier filter in float64 (numpy promotes `float32 · float64-window`)
//! and casts the smooth component back to float32; tomoxide reuses the crate's
//! self-contained f64 column FFT (which matches `fft_impl = numpy.fft` pocketfft
//! to f64 round-off) with the same dtype flow, so it is held to the f32 round-off
//! floor (projector-independent), not bit-exactness — the same contract as the
//! Fourier-Wavelet path. Both window dimensionalities are checked: `dim=1`
//! (footprint `(size, 1)`, sigma=3, size auto → 5) and `dim=2` (footprint
//! `(size, size)`, sigma=5, size=5). Golden from the real tomopy 1.15.3
//! (`tools/gen_tomopy_stripe_vofilter_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, Layout, StripeMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn run_case(golden_file: &str, method: StripeMethod) {
    let input: Array3<f32> = read_npy(format!("{FIXTURES}/stripe_vofilter_input.npy")).unwrap();
    let golden: Array3<f32> = read_npy(format!("{FIXTURES}/{golden_file}")).unwrap();

    // Input is tomopy projection order [nproj, nslices, ncol].
    let mut tomo = Tomo::new(input, Layout::Projection);
    prep::stripe::remove_stripe(&mut tomo, method).unwrap();

    let got = tomo.to_layout(Layout::Projection);
    assert_eq!(got.array.dim(), golden.dim());
    let scale = golden.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_abs = golden
        .iter()
        .zip(got.array.iter())
        .fold(0.0f32, |m, (&g, &p)| m.max((g - p).abs()));
    let max_rel = max_abs / scale;
    eprintln!("{golden_file}: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})");
    assert!(
        max_rel <= 1e-5,
        "{golden_file}: Vo-filter parity to the f32 floor, got max|Δ| = {max_abs}, \
         max relative = {max_rel} (scale {scale})"
    );
}

#[test]
fn vofilter_dim1_matches_tomopy() {
    // sigma=3, size=None → tomopy default max(5, 0.01·ncol) = 5 for ncol=64.
    run_case(
        "tomopy_stripe_vofilter_dim1.npy",
        StripeMethod::VoFilter {
            sigma: 3.0,
            size: None,
            dim: 1,
        },
    );
}

#[test]
fn vofilter_dim2_matches_tomopy() {
    run_case(
        "tomopy_stripe_vofilter_dim2.npy",
        StripeMethod::VoFilter {
            sigma: 5.0,
            size: Some(5),
            dim: 2,
        },
    );
}
