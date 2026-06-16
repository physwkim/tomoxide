//! Bit-exact parity against tomopy for nearest-flat-fields normalization
//! (`normalize_nf`, averaging='mean').
//!
//! Each flat group's per-pixel median is the flat for the projections nearest
//! its `flat_loc`; `dark` is the per-pixel mean of the dark frames. Each
//! projection is `(proj − dark) / max(flat − dark, 1e-6)`, optionally clamped
//! above by `cutoff`. All arithmetic is f32 in the upstream order, so the port
//! matches tomopy bit-for-bit (Δ = 0). Two cases: an even group (median averages
//! two) without cutoff, and an odd group (median selects) with cutoff=1.5; both
//! force the (0,0) flat to the dark mean to exercise the `< 1e-6` denom clamp.
//! Golden from the real tomopy 1.15.3 (`tools/gen_tomopy_normalize_nf_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::prep::Averaging;
use tomoxide::{prep, Frames, Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

fn run_case(flats_file: &str, golden_file: &str, flat_loc: &[usize], cutoff: Option<f32>) {
    let mut tomo = Tomo::new(load("normalize_nf_tomo.npy"), Layout::Projection);
    let flats = Frames::new(load(flats_file));
    let dark = Frames::new(load("normalize_nf_dark.npy"));
    let golden = load(golden_file);

    prep::normalize::normalize_nf(&mut tomo, &flats, &dark, flat_loc, cutoff, Averaging::Mean)
        .unwrap();

    let got = tomo.to_layout(Layout::Projection);
    assert_eq!(got.array.dim(), golden.dim());
    let mut max_abs = 0.0f32;
    for (&g, &p) in golden.iter().zip(got.array.iter()) {
        max_abs = max_abs.max((g - p).abs());
    }
    eprintln!("{golden_file}: max|Δ| = {max_abs}");
    assert_eq!(
        max_abs, 0.0,
        "{golden_file}: normalize_nf must match tomopy bit-for-bit, got max|Δ| = {max_abs}"
    );
}

#[test]
fn normalize_nf_even_group_no_cutoff_matches_tomopy() {
    run_case(
        "normalize_nf_flatsA.npy",
        "tomopy_normalize_nf_A.npy",
        &[0, 7],
        None,
    );
}

#[test]
fn normalize_nf_odd_group_with_cutoff_matches_tomopy() {
    run_case(
        "normalize_nf_flatsB.npy",
        "tomopy_normalize_nf_B.npy",
        &[1, 6],
        Some(1.5),
    );
}
