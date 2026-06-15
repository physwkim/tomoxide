//! Numeric parity against tomopy for Fourier-Wavelet stripe removal (`remove_stripe_fw`).
//!
//! The Fourier-Wavelet method (Münch 2009) runs a `level`-deep `db5` 2-D wavelet
//! decomposition per slice, damps the vertical-detail bands along the projection
//! axis in Fourier space, reconstructs, and casts back to f32. tomopy's `pywt`
//! forward pass is float32 (each band quantized), while numpy promotes the FFT
//! damping and `pywt` inverse to float64; tomoxide reimplements the same db5
//! transform (validated against `pywt`) with the matching float32-forward /
//! float64-damp+inverse dtype flow, so it is held to the f32 round-off floor
//! (projector-independent), not bit-exactness. Golden from tomopy 1.15.3
//! `remove_stripe_fw` (`tools/gen_tomopy_stripe_fw_golden.py`, db5, sigma=2,
//! pad=True, level=auto).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, Layout, StripeMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

#[test]
fn remove_stripe_fw_matches_tomopy() {
    let input = load("stripe_fw_input.npy");
    let golden = load("tomopy_stripe_fw_db5.npy");

    let mut tomo = Tomo::new(input, Layout::Projection);
    prep::stripe::remove_stripe(
        &mut tomo,
        StripeMethod::Fw {
            sigma: 2.0,
            level: None,
        },
    )
    .unwrap();
    let got = tomo.array;

    // Agreement with the tomopy golden, to the f32 round-off floor. The db5
    // transform is bit-reproducible against `pywt` (see the wavelet unit tests);
    // the residual is the f32 quantization of each forward band plus the
    // pocketfft-vs-direct-DFT difference in the damping, both below ~1e-5 rel.
    let scale = golden.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_abs = golden
        .iter()
        .zip(got.iter())
        .fold(0.0f32, |m, (&g, &p)| m.max((g - p).abs()));
    let max_rel = max_abs / scale;
    eprintln!("Fw db5: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})");
    assert!(
        max_rel <= 1e-5,
        "Fw golden parity: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})"
    );
}
