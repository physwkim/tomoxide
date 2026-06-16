//! Parity against tomopy for 3-D stripe detection (`stripes_detect3d`,
//! Kazantsev 2023), backed by libtomo `stripesdetect3d_main_float`.
//!
//! The kernel is pure `f32` arithmetic with no FFT (6-stencil mean smoothing →
//! horizontal forward gradient with step 2 → parallel/orthogonal mean-ratio map
//! → vertical median filter), so the port is held to BIT-EXACTNESS (Δ = 0), not
//! the f32 round-off floor that the FFT-based stripe paths use.
//!
//! Two parameter sets: size=10 radius=3 (tomopy defaults) and size=5 radius=2.
//! Goldens from real tomopy 1.15.3 (`tools/gen_tomopy_stripe_detect3d_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

fn run_case(golden_name: &str, size: usize, radius: usize) {
    let input = load("stripe_detect3d_input.npy");
    let golden = load(golden_name);

    let tomo = Tomo::new(input.clone(), Layout::Projection);
    let got = prep::stripes_detect3d(&tomo, size, radius).unwrap();
    assert_eq!(got.dim(), golden.dim());

    // (1) Bit-exact agreement with the tomopy golden.
    let max_abs = golden
        .iter()
        .zip(got.iter())
        .fold(0.0f32, |m, (&g, &p)| m.max((g - p).abs()));
    eprintln!("detect3d size={size} radius={radius}: max|Δ| = {max_abs}");
    assert_eq!(
        max_abs, 0.0,
        "detect3d size={size} radius={radius} is not bit-exact: max|Δ| = {max_abs}"
    );

    // (2) The weights are valid probabilities in [0, 1].
    let (lo, hi) = got
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    assert!(lo >= 0.0 && hi <= 1.0, "weights out of [0,1]: [{lo}, {hi}]");

    // (3) Injected stripes (constant-across-angle gain at detX 20/45/50) are
    // highlighted: their minimum weight is well below a clean reference column.
    let col_min = |c: usize| {
        got.slice(ndarray::s![.., .., c])
            .iter()
            .fold(f32::INFINITY, |m, &v| m.min(v))
    };
    let clean_col_min = col_min(10);
    for stripe_col in [20usize, 45, 50] {
        let smin = col_min(stripe_col);
        assert!(
            smin < 0.5,
            "stripe col {stripe_col} not highlighted: min weight {smin} (clean col10 min {clean_col_min})"
        );
        assert!(
            smin < clean_col_min,
            "stripe col {stripe_col} min weight {smin} not below clean col10 min {clean_col_min}"
        );
    }
}

#[test]
fn detect3d_default_matches_tomopy() {
    run_case("tomopy_detect3d_def.npy", 10, 3);
}

#[test]
fn detect3d_alt_matches_tomopy() {
    run_case("tomopy_detect3d_alt.npy", 5, 2);
}
