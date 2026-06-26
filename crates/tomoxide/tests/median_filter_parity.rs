//! Bit-exact parity against tomopy for `median_filter` (corr.py:167).
//!
//! Median-filters every 2-D slice along `axis` with a `size×size` footprint
//! (scipy.ndimage.median_filter, default `mode='reflect'`, half-sample
//! reflection). Every pixel is replaced by its local median (no threshold). The
//! filter selects a single order statistic (rank `size·size/2`, never an average
//! even for an even footprint), so the result is bit-exact (Δ=0).
//!
//! Golden from `tools/gen_tomopy_median_filter_golden.py` (real tomopy 1.15.3;
//! this wrapper uses `arr[tuple(slc)]` so it runs unmodified on numpy 2.x).
//! Cases cover odd/even `size` and all three axes.

use ndarray::{Array3, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::data::{Layout, Tomo};
use tomoxide::prep::filters::median_filter;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn median_filter_matches_tomopy() {
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/median_filter_input.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/median_filter_output.npy")).unwrap();
    let params: ndarray::Array2<f64> =
        read_npy(format!("{FIXTURES}/median_filter_params.npy")).unwrap();

    let ncase = inputs.dim().0;
    for k in 0..ncase {
        let size = params[[k, 0]] as usize;
        let axis = params[[k, 1]] as usize;

        // `axis` indexes the raw 3-D array, matching tomopy; layout is irrelevant
        // to the math, so it merely carries the array.
        let input: Array3<f32> = inputs.index_axis(Axis(0), k).to_owned();
        let mut tomo = Tomo::new(input, Layout::Projection);
        median_filter(&mut tomo, size, axis).unwrap();

        let want = outputs.index_axis(Axis(0), k);
        let mut mismatch = 0usize;
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                mismatch += 1;
            }
        }
        assert_eq!(
            mismatch, 0,
            "case {k} (size={size}, axis={axis}): {mismatch} f32 bit-mismatches vs tomopy"
        );
    }
}
