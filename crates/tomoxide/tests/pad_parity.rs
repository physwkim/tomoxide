//! Bit-exact parity against tomopy for `morph.pad` (misc/morph.py).
//!
//! `pad` widens an axis by `npad` on each side (npad=None → ⌈(dim·√2−dim)/2⌉);
//! flanks are a constant or the replicated edge slab. Pure copy/fill, so it
//! matches tomopy bit-for-bit (Δ = 0). Checked: axis=2 constant/edge at default
//! npad, axis=0 constant npad=3 value=0.5, axis=1 edge npad=2. Golden from the
//! real tomopy 1.15.3 (`tools/gen_tomopy_pad_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::prep::morph::{self, PadMode};

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
        "{golden_file}: pad must match tomopy bit-for-bit, got max|Δ| = {max_abs}"
    );
}

#[test]
fn pad_axis2_constant_default_matches_tomopy() {
    let got = morph::pad(&load("pad_input.npy"), 2, None, PadMode::Constant(0.0)).unwrap();
    assert_bit_exact("tomopy_pad_ax2_const_def.npy", &got);
}

#[test]
fn pad_axis2_edge_default_matches_tomopy() {
    let got = morph::pad(&load("pad_input.npy"), 2, None, PadMode::Edge).unwrap();
    assert_bit_exact("tomopy_pad_ax2_edge_def.npy", &got);
}

#[test]
fn pad_axis0_constant_npad3_matches_tomopy() {
    let got = morph::pad(&load("pad_input.npy"), 0, Some(3), PadMode::Constant(0.5)).unwrap();
    assert_bit_exact("tomopy_pad_ax0_const_n3.npy", &got);
}

#[test]
fn pad_axis1_edge_npad2_matches_tomopy() {
    let got = morph::pad(&load("pad_input.npy"), 1, Some(2), PadMode::Edge).unwrap();
    assert_bit_exact("tomopy_pad_ax1_edge_n2.npy", &got);
}
