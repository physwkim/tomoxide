//! Bit-exact parity against tomopy for `sino_360_to_180` (misc/morph.py).
//!
//! Folds a 0–360° sinogram into 0–180° by stitching the column-reversed second
//! half-rotation onto the first with a linear cross-fade across an `overlap`-wide
//! seam. Direct regions are exact f32 copies; the seam blend is computed in f64
//! (numpy promotes float64-weights · float32-data) and cast to f32, so it matches
//! tomopy bit-for-bit (Δ = 0). Both rotation sides and a seam-less `overlap=0`
//! case are checked. Golden from the real tomopy 1.15.3
//! (`tools/gen_tomopy_sino360_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::prep::Rotation;
use tomoxide::{prep, Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn run_case(golden_file: &str, overlap: usize, rotation: Rotation) {
    let input: Array3<f32> = read_npy(format!("{FIXTURES}/sino360_input.npy")).unwrap();
    let golden: Array3<f32> = read_npy(format!("{FIXTURES}/{golden_file}")).unwrap();

    let tomo = Tomo::new(input, Layout::Projection);
    let got = prep::morph::sino_360_to_180(&tomo, overlap, rotation).unwrap();

    assert_eq!(
        got.array.dim(),
        golden.dim(),
        "{golden_file}: shape mismatch"
    );
    let mut max_abs = 0.0f32;
    for (&g, &p) in golden.iter().zip(got.array.iter()) {
        max_abs = max_abs.max((g - p).abs());
    }
    eprintln!("{golden_file}: max|Δ| = {max_abs}");
    assert_eq!(
        max_abs, 0.0,
        "{golden_file}: sino_360_to_180 must match tomopy bit-for-bit, got max|Δ| = {max_abs}"
    );
}

#[test]
fn sino360_left_overlap4_matches_tomopy() {
    run_case("tomopy_sino360_left_o4.npy", 4, Rotation::Left);
}

#[test]
fn sino360_right_overlap4_matches_tomopy() {
    run_case("tomopy_sino360_right_o4.npy", 4, Rotation::Right);
}

#[test]
fn sino360_left_overlap0_matches_tomopy() {
    run_case("tomopy_sino360_left_o0.npy", 0, Rotation::Left);
}
