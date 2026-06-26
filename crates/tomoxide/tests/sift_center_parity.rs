//! SIFT center-finding parity against tomocupy's cv2 implementation.
//!
//! Golden from `tools/gen_sift_center_golden.py` (cv2 4.13 SIFT + BFMatcher).
//! The Rust port links the SAME OpenCV (conda libopencv 4.13), so the uint8
//! normalization, keypoints, matches, and recovered center match it closely
//! (the small residual is f32 keypoint coords / match ordering).
//!
//! Runs only with the `sift-center` feature (needs OpenCV at build + run time).
#![cfg(feature = "sift-center")]

use ndarray::{Array1, Array2, Array3};
use ndarray_npy::read_npy;
use tomoxide::recon::center::sift::{find_center_sift, normalize_to_u8, register_shift_sift};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn slices(a: &Array3<f32>) -> Vec<Array2<f32>> {
    (0..a.dim().0)
        .map(|i| a.index_axis(ndarray::Axis(0), i).to_owned())
        .collect()
}

fn fliplr(a: &Array2<f32>) -> Array2<f32> {
    let (nr, nc) = a.dim();
    Array2::from_shape_fn((nr, nc), |(i, j)| a[[i, nc - 1 - j]])
}

#[test]
fn uint8_normalization_matches_numpy() {
    // tmp2 in the golden = datap1 normalized by its own robust min/max.
    let datap1: Array3<f32> = read_npy(format!("{FIXTURES}/sift_datap1.npy")).unwrap();
    let u2: Array3<u8> = read_npy(format!("{FIXTURES}/sift_u2.npy")).unwrap();
    for i in 0..datap1.dim().0 {
        let img = datap1.index_axis(ndarray::Axis(0), i).to_owned();
        let got = normalize_to_u8(&img);
        let want: Vec<u8> = u2.index_axis(ndarray::Axis(0), i).iter().copied().collect();
        let diffs = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(
            diffs, 0,
            "uint8 normalization mismatch in image {i}: {diffs} px"
        );
    }
}

#[test]
fn shifts_and_center_match_cv2() {
    let datap1: Array3<f32> = read_npy(format!("{FIXTURES}/sift_datap1.npy")).unwrap();
    let datap2: Array3<f32> = read_npy(format!("{FIXTURES}/sift_datap2.npy")).unwrap();
    let g_shifts: Array2<f32> = read_npy(format!("{FIXTURES}/sift_shifts.npy")).unwrap();
    let g_center: Array1<f32> = read_npy(format!("{FIXTURES}/sift_center.npy")).unwrap();

    let d1 = slices(&datap1);
    let d2 = slices(&datap2);
    let (shifts, _ngood) = register_shift_sift(&d1, &d2, 0.5).unwrap();

    let mut max_d = 0.0f32;
    for i in 0..g_shifts.dim().0 {
        for j in 0..2 {
            max_d = max_d.max((shifts[[i, j]] - g_shifts[[i, j]]).abs());
        }
    }
    // Same OpenCV library both sides → measured ≈ 5e-7 (f32 floor).
    assert!(max_d < 1e-4, "SIFT shift max abs diff {max_d}");

    // center = ncol/2 - mean(shift_x)/2 (mean over pairs), matching the golden.
    let ncol = datap1.dim().2 as f32;
    let mean_dx = (0..shifts.dim().0).map(|i| shifts[[i, 1]]).sum::<f32>() / shifts.dim().0 as f32;
    let center = ncol / 2.0 - mean_dx / 2.0;
    assert!(
        (center - g_center[0]).abs() < 1e-4,
        "center {center} vs {}",
        g_center[0]
    );

    // find_center_sift on pair 0: it flips proj180 internally, so feed
    // proj180 = fliplr(datap2[0]) to reconstruct the golden datap2[0].
    let proj0 = d1[0].clone();
    let proj180 = fliplr(&d2[0]);
    let c0 = find_center_sift(&proj0, &proj180, 0.5).unwrap();
    let want0 = ncol / 2.0 - g_shifts[[0, 1]] / 2.0;
    assert!(
        (c0 - want0).abs() < 1e-4,
        "find_center_sift {c0} vs {want0}"
    );
}
