//! Bit-exact parity against tomopy for `sobel_filter` (misc/corr.py:474).
//!
//! Sobel-filters every 2-D slice along `axis` via scipy.ndimage.sobel: a
//! `[-1,0,1]` central-difference correlation along the slice's last axis then a
//! `[1,2,1]` smoothing correlation along the other (both `mode='reflect'`),
//! reusing the f64 `correlate1d` primitive shared with `gaussian_filter`. The
//! weights are exact small integers and f32 inputs are exact in the f64
//! accumulator, so the result is bit-exact (Δ=0).
//!
//! tomopy's published `sobel_filter` cannot run (a bare `filters.sobel`
//! NameError plus the numpy-2.x `arr[slc]` list-index IndexError), so the golden
//! `tools/gen_tomopy_sobel_filter_golden.py` inlines tomopy's verbatim body with
//! exactly those two one-token compat fixes — same dtype cast and per-slice
//! scipy call. Cases: a 2-D slice taken along each of the three axes.

use ndarray::{Array2, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::prep::filters::sobel_filter;
use tomoxide::data::{Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn sobel_filter_matches_tomopy() {
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/sobel_filter_input.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/sobel_filter_output.npy")).unwrap();
    let params: Array2<f64> = read_npy(format!("{FIXTURES}/sobel_filter_params.npy")).unwrap();

    let ncase = inputs.dim().0;
    for k in 0..ncase {
        let axis = params[[k, 0]] as usize;
        // sobel_filter's `axis` indexes the raw 3-D array; Projection layout
        // keeps `.array` identical to the numpy array, so axes map directly.
        let mut tomo = Tomo::new(inputs.index_axis(Axis(0), k).to_owned(), Layout::Projection);
        sobel_filter(&mut tomo, axis).unwrap();

        let want = outputs.index_axis(Axis(0), k);
        let mut mismatch = 0usize;
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                mismatch += 1;
            }
        }
        assert_eq!(
            mismatch, 0,
            "case {k} (axis={axis}): {mismatch} f32 bit-mismatches vs tomopy"
        );
    }
}
