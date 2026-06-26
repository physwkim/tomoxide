//! Bit-exact parity against tomopy for `remove_outlier1d`.
//!
//! Dezinger via a 1-D median filter along `axis` (scipy.ndimage.median_filter,
//! `mode='mirror'`), replacing a pixel by the local median only where `arr -
//! median >= dif`. The median filter selects a single order statistic (rank
//! `size/2`, never an average even for even `size`) and the `where` test is a
//! plain f32 subtraction, so the result is bit-exact (Δ=0).
//!
//! Golden from `tools/gen_tomopy_remove_outlier1d_golden.py`, which runs
//! tomopy's verbatim `misc/corr.py:615` body with the single numpy-2.x compat
//! fix (`arr[slc]` → `arr[tuple(slc)]`; tomopy 1.15.3's published wrapper
//! raises `IndexError` on numpy 2.x, while its sibling `remove_outlier` already
//! uses the tupled index). Cases cover odd/even `size`, all three axes, and a
//! `dif=0` threshold.

use ndarray::{Array3, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::data::{Layout, Tomo};
use tomoxide::prep::filters::remove_outlier1d;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn remove_outlier1d_matches_tomopy() {
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/remove_outlier1d_input.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/remove_outlier1d_output.npy")).unwrap();
    let params: ndarray::Array2<f64> =
        read_npy(format!("{FIXTURES}/remove_outlier1d_params.npy")).unwrap();

    let ncase = inputs.dim().0;
    let mut total_mismatch = 0usize;
    for k in 0..ncase {
        let dif = params[[k, 0]] as f32;
        let size = params[[k, 1]] as usize;
        let axis = params[[k, 2]] as usize;

        // `axis` indexes the raw 3-D array, matching tomopy; layout is irrelevant
        // to the math, so any layout works as a carrier.
        let input: Array3<f32> = inputs.index_axis(Axis(0), k).to_owned();
        let mut tomo = Tomo::new(input, Layout::Projection);
        remove_outlier1d(&mut tomo, dif, size, axis).unwrap();

        let want = outputs.index_axis(Axis(0), k);
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                total_mismatch += 1;
            }
        }
        assert_eq!(
            total_mismatch, 0,
            "case {k} (dif={dif}, size={size}, axis={axis}): {total_mismatch} f32 bit-mismatches vs tomopy"
        );
    }
}
