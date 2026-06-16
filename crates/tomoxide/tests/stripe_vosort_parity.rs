//! Bit-exact parity against tomopy for Vo sorting-based stripe removal
//! (`remove_stripe_based_sorting`, Vo 2018 algorithm 3).
//!
//! `_rs_sort` sorts each detector column's values over projections, median-filters
//! the sorted matrix, and unsorts. The median is a pure rank-filter *selection* of
//! an existing f32 value (no arithmetic), so on tie-free columns the result is
//! identical to tomopy bit-for-bit (Δ = 0). Both window dimensionalities are
//! checked: `dim=1` (footprint `(size, 1)`, size auto = `max(5, 0.01·ncol)` → 5)
//! and `dim=2` (footprint `(size, size)`, size = 5). Golden from the real tomopy
//! 1.15.3 (`tools/gen_tomopy_stripe_vosort_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, Layout, StripeMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn run_case(golden_file: &str, method: StripeMethod) {
    let input: Array3<f32> = read_npy(format!("{FIXTURES}/stripe_vosort_input.npy")).unwrap();
    let golden: Array3<f32> = read_npy(format!("{FIXTURES}/{golden_file}")).unwrap();

    // Input is tomopy projection order [nproj, nslices, ncol].
    let mut tomo = Tomo::new(input, Layout::Projection);
    prep::stripe::remove_stripe(&mut tomo, method).unwrap();

    let got = tomo.to_layout(Layout::Projection);
    assert_eq!(got.array.dim(), golden.dim());
    let mut max_abs = 0.0f32;
    for (&g, &p) in golden.iter().zip(got.array.iter()) {
        max_abs = max_abs.max((g - p).abs());
    }
    eprintln!("{golden_file}: max|Δ| = {max_abs}");
    assert_eq!(
        max_abs, 0.0,
        "{golden_file}: Vo-sort must match tomopy bit-for-bit, got max|Δ| = {max_abs}"
    );
}

#[test]
fn vosort_dim1_matches_tomopy() {
    // size=None → tomopy default max(5, 0.01·ncol) = 5 for ncol=64.
    run_case(
        "tomopy_stripe_vosort_dim1.npy",
        StripeMethod::VoSort { size: None, dim: 1 },
    );
}

#[test]
fn vosort_dim2_matches_tomopy() {
    run_case(
        "tomopy_stripe_vosort_dim2.npy",
        StripeMethod::VoSort {
            size: Some(5),
            dim: 2,
        },
    );
}
