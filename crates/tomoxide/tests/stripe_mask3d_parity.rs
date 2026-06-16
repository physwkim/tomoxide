//! Parity against tomopy for 3-D stripe masking (`stripes_mask3d`,
//! Kazantsev 2023), backed by libtomo `stripesmask3d_main_float`.
//!
//! `stripes_mask3d` consumes a `stripes_detect3d` weights volume and returns a
//! binary `bool` mask: threshold the weights, enforce stripe consistency in
//! depth and along-angle, drop short stripes, and iteratively merge nearby ones.
//! It is pure integer/bool logic with a single `f32` threshold compare and
//! `(int)(0.01*sensitivity*len)` thresholds, so the port is held to
//! BIT-EXACTNESS — the bool mask matches tomopy element-for-element.
//!
//! Two parameter sets: tomopy defaults (threshold=0.6, length=20, depth=10,
//! width=5, sensitivity=85) and a looser set (0.5, 10, 4, 3, 60). The weights
//! fixture is itself the real-tomopy `stripes_detect3d` output. Goldens from
//! real tomopy 1.15.3 (`tools/gen_tomopy_stripe_mask3d_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::prep;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load_f32(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

fn load_bool(name: &str) -> Array3<bool> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    golden_name: &str,
    threshold: f32,
    length: usize,
    depth: usize,
    width: usize,
    sensitivity: f32,
) {
    let weights = load_f32("stripe_mask3d_weights.npy");
    let golden = load_bool(golden_name);

    let got = prep::stripes_mask3d(&weights, threshold, length, depth, width, sensitivity).unwrap();
    assert_eq!(got.dim(), golden.dim());

    // (1) Bit-exact bool agreement with the tomopy golden.
    let mismatches = golden
        .iter()
        .zip(got.iter())
        .filter(|(&g, &p)| g != p)
        .count();
    let true_count = got.iter().filter(|&&v| v).count();
    eprintln!(
        "mask3d thr={threshold} len={length}: mismatches = {mismatches}, got True = {true_count}, golden True = {}",
        golden.iter().filter(|&&v| v).count()
    );
    assert_eq!(
        mismatches, 0,
        "mask3d thr={threshold} len={length} is not bit-exact: {mismatches} voxels differ"
    );

    // (2) The mask is non-trivial (the synthetic stripes produce some True).
    assert!(
        true_count > 0,
        "mask3d thr={threshold} len={length} produced an all-false mask"
    );
}

#[test]
fn mask3d_default_matches_tomopy() {
    run_case("tomopy_mask3d_def.npy", 0.6, 20, 10, 5, 85.0);
}

#[test]
fn mask3d_alt_matches_tomopy() {
    run_case("tomopy_mask3d_alt.npy", 0.5, 10, 4, 3, 60.0);
}
