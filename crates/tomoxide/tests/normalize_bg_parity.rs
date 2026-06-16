//! Bit-exact parity against tomopy for background (air-region) normalization
//! (`normalize_bg` → `libtomo/prep/prep.c::normalize_bg`).
//!
//! For each projection row the mean of the `air` left-boundary pixels and the
//! `air` right-boundary pixels defines an air baseline linearly interpolated
//! across the detector width; every pixel is divided by its local baseline. The
//! C does all arithmetic in float32 in a fixed accumulation order, so the port
//! matches tomopy bit-for-bit (Δ = 0). Two boundary widths are checked: `air=1`
//! (single-pixel boundary, the tomopy default) and `air=4` (multi-pixel mean).
//! Golden from the real tomopy 1.15.3 (`tools/gen_tomopy_normalize_bg_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn run_case(golden_file: &str, air: usize) {
    let input: Array3<f32> = read_npy(format!("{FIXTURES}/normalize_bg_input.npy")).unwrap();
    let golden: Array3<f32> = read_npy(format!("{FIXTURES}/{golden_file}")).unwrap();

    // Input is tomopy projection order [nproj, nrows, ncol].
    let mut tomo = Tomo::new(input, Layout::Projection);
    prep::normalize::normalize_bg(&mut tomo, air).unwrap();

    let got = tomo.to_layout(Layout::Projection);
    assert_eq!(got.array.dim(), golden.dim());
    let mut max_abs = 0.0f32;
    for (&g, &p) in golden.iter().zip(got.array.iter()) {
        max_abs = max_abs.max((g - p).abs());
    }
    eprintln!("{golden_file}: max|Δ| = {max_abs}");
    assert_eq!(
        max_abs, 0.0,
        "{golden_file}: normalize_bg must match tomopy bit-for-bit, got max|Δ| = {max_abs}"
    );
}

#[test]
fn normalize_bg_air1_matches_tomopy() {
    run_case("tomopy_normalize_bg_air1.npy", 1);
}

#[test]
fn normalize_bg_air4_matches_tomopy() {
    run_case("tomopy_normalize_bg_air4.npy", 4);
}
