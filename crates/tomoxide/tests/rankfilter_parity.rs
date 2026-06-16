//! Numeric parity against tomopy for the CPU rank filters.
//!
//! `median_filter3d` and `remove_outlier3d` are pure `(2·radius+1)³`
//! neighbourhood operations (clamp-to-center boundary, sorted median) — no
//! projector, no FFT — so they are held to exact tomopy parity (bit-for-bit).
//! Goldens from tomopy 1.15.3 (`tools/gen_tomopy_rankfilter_golden.py`) on the
//! SAME input volume this test feeds offline.

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::prep::filters::{median_filter3d, remove_outlier3d};
use tomoxide::{CpuBackend, Layout, Tomo, Volume};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

/// Every voxel must match tomopy exactly (these ops are integer-index +
/// comparison only — no floating-point arithmetic that could diverge).
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
fn median_filter3d_matches_tomopy() {
    let input = load("rankfilter_input.npy");
    let cpu = CpuBackend::new();

    // size = 3 (radius 1).
    let mut vol = Volume::new(input.clone());
    median_filter3d(&mut vol, 3, &cpu).unwrap();
    assert_bit_equal(&vol.array, &load("tomopy_median3.npy"), "median3d size=3");

    // size = 5 (radius 2).
    let mut vol = Volume::new(input.clone());
    median_filter3d(&mut vol, 5, &cpu).unwrap();
    assert_bit_equal(&vol.array, &load("tomopy_median5.npy"), "median3d size=5");
}

#[test]
fn remove_outlier3d_matches_tomopy() {
    let input = load("rankfilter_input.npy");
    let cpu = CpuBackend::new();

    // Small threshold: every injected spike exceeds it → all replaced.
    let mut tomo = Tomo::new(input.clone(), Layout::Projection);
    remove_outlier3d(&mut tomo, 0.5, 3, &cpu).unwrap();
    assert_bit_equal(
        &tomo.array,
        &load("tomopy_dezinger_small.npy"),
        "remove_outlier3d dif=0.5",
    );

    // Larger threshold: spikes (±10) still exceed 5.0, smooth structure stays.
    let mut tomo = Tomo::new(input.clone(), Layout::Projection);
    remove_outlier3d(&mut tomo, 5.0, 3, &cpu).unwrap();
    assert_bit_equal(
        &tomo.array,
        &load("tomopy_dezinger_large.npy"),
        "remove_outlier3d dif=5.0",
    );
}
