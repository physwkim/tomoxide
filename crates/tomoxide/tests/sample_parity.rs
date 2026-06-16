//! Bit-exact parity against tomopy for `downsample`/`upsample` (misc/morph.py →
//! libtomo/misc/morph.c `c_sample`).
//!
//! downsample bins by `2^level` along an axis (each output = mean of the bin,
//! accumulated as Σ(data/binsize) in f32); upsample replicates each value
//! `2^level` times along the axis. Both are f32 in the upstream order, so they
//! match tomopy bit-for-bit (Δ = 0). Checked across axes 0/1/2 at level 1.
//! Golden from the real tomopy 1.15.3 (`tools/gen_tomopy_sample_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::prep::morph;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

fn assert_bit_exact(golden_file: &str, got: &Array3<f32>) {
    let golden = load(golden_file);
    assert_eq!(got.dim(), golden.dim(), "{golden_file}: shape mismatch");
    let mut max_abs = 0.0f32;
    for (&g, &p) in golden.iter().zip(got.iter()) {
        max_abs = max_abs.max((g - p).abs());
    }
    eprintln!("{golden_file}: max|Δ| = {max_abs}");
    assert_eq!(
        max_abs, 0.0,
        "{golden_file}: must match tomopy bit-for-bit, got max|Δ| = {max_abs}"
    );
}

#[test]
fn downsample_axis2_level1_matches_tomopy() {
    let got = morph::downsample(&load("sample_input.npy"), 1, 2).unwrap();
    assert_bit_exact("tomopy_downsample_ax2_l1.npy", &got);
}

#[test]
fn downsample_axis0_level1_matches_tomopy() {
    let got = morph::downsample(&load("sample_input.npy"), 1, 0).unwrap();
    assert_bit_exact("tomopy_downsample_ax0_l1.npy", &got);
}

#[test]
fn upsample_axis2_level1_matches_tomopy() {
    let got = morph::upsample(&load("sample_input.npy"), 1, 2).unwrap();
    assert_bit_exact("tomopy_upsample_ax2_l1.npy", &got);
}

#[test]
fn upsample_axis1_level1_matches_tomopy() {
    let got = morph::upsample(&load("sample_input.npy"), 1, 1).unwrap();
    assert_bit_exact("tomopy_upsample_ax1_l1.npy", &got);
}
