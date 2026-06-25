//! Bit-exact parity against tomopy for `normalize_roi` (normalize.py:168).
//!
//! Each projection is divided by the mean `bg` of its ROI window
//! `proj[r0:r2, r1:r3]` (when `bg != 0`). The ROI mean reproduces numpy's f32
//! pairwise summation, so `bg` and the elementwise f32 divide are bit-exact
//! (Δ=0). A plain sequential sum would diverge by ~1 ULP and fail.
//!
//! Golden from `tools/gen_tomopy_normalize_roi_golden.py`, which applies
//! tomopy's verbatim per-projection kernel `_normalize_roi` in-process (the
//! macOS `mproc.distribute_jobs` pool is flaky; the chunking is per-projection
//! independent, so this is numerically identical). Cases: the default 10×10
//! ROI (base case), a >128-element ROI (recursion path), and an offset
//! non-square ROI.

use ndarray::{Array2, Array3, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::prep::normalize_roi;
use tomoxide::data::{Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn normalize_roi_matches_tomopy() {
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/normalize_roi_input.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/normalize_roi_output.npy")).unwrap();
    let params: Array2<f64> = read_npy(format!("{FIXTURES}/normalize_roi_params.npy")).unwrap();

    let ncase = inputs.dim().0;
    for k in 0..ncase {
        let roi = [
            params[[k, 0]] as usize,
            params[[k, 1]] as usize,
            params[[k, 2]] as usize,
            params[[k, 3]] as usize,
        ];
        let input: Array3<f32> = inputs.index_axis(Axis(0), k).to_owned();
        let mut tomo = Tomo::new(input, Layout::Projection);
        normalize_roi(&mut tomo, roi).unwrap();

        let want = outputs.index_axis(Axis(0), k);
        let mut mismatch = 0usize;
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                mismatch += 1;
            }
        }
        assert_eq!(
            mismatch, 0,
            "case {k} (roi={roi:?}): {mismatch} f32 bit-mismatches vs tomopy"
        );
    }
}
