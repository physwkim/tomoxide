//! Numeric parity against tomopy for smoothing-filter stripe removal.
//!
//! `remove_stripe_sf` is pure per-column f32 arithmetic on the projection stack
//! (column-wise mean over angles → clamp-to-edge moving average → subtract the
//! residual) — projector-independent, performed in the same summation order as
//! tomopy — so it is held to bit-exact parity. Goldens from tomopy 1.15.3
//! (`tools/gen_tomopy_stripe_sf_golden.py`) on the SAME input this test feeds.

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, Layout, StripeMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

fn assert_bit_equal(got: &Array3<f32>, want: &Array3<f32>, label: &str) {
    assert_eq!(got.dim(), want.dim(), "{label}: shape mismatch");
    let mut diffs = 0usize;
    let mut max_abs = 0.0f32;
    for (&g, &w) in got.iter().zip(want.iter()) {
        if g != w {
            diffs += 1;
            max_abs = max_abs.max((g - w).abs());
        }
    }
    assert!(
        diffs == 0,
        "{label}: {diffs} voxel(s) differ from tomopy (max|Δ| = {max_abs})"
    );
}

#[test]
fn remove_stripe_sf_matches_tomopy() {
    let input = load("stripe_sf_input.npy");

    let mut tomo = Tomo::new(input.clone(), Layout::Projection);
    prep::stripe::remove_stripe(&mut tomo, StripeMethod::Sf { size: 3 }).unwrap();
    assert_bit_equal(&tomo.array, &load("tomopy_stripe_sf3.npy"), "sf size=3");

    let mut tomo = Tomo::new(input.clone(), Layout::Projection);
    prep::stripe::remove_stripe(&mut tomo, StripeMethod::Sf { size: 5 }).unwrap();
    assert_bit_equal(&tomo.array, &load("tomopy_stripe_sf5.npy"), "sf size=5");
}
