//! Bit-exact parity against tomopy for the 2-D `remove_outlier` (corr.py:559).
//!
//! The axis-chunked 2-D dezinger: for each index along `axis` the orthogonal
//! 2-D image's `size×size` median is taken (scipy.ndimage default
//! `mode='reflect'`), then a pixel is replaced by that median only where `arr −
//! median ≥ dif`. The filter selects a single order statistic (rank
//! `size·size/2`, never an average) and the `where` test is a plain f32
//! subtraction, so the result is bit-exact (Δ=0).
//!
//! Distinct from `remove_outlier3d` (3-D cube) and `remove_outlier1d` (1-D
//! mirror). Golden from `tools/gen_tomopy_remove_outlier_golden.py` (real
//! tomopy 1.15.3; this wrapper uses `arr[tuple(slc)]`, so it runs unmodified on
//! numpy 2.x). Cases cover odd/even `size`, all three axes, and `dif=0`.

use ndarray::{Array3, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::data::{Layout, Tomo};
use tomoxide::prep::filters::remove_outlier;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn remove_outlier_matches_tomopy() {
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/remove_outlier_input.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/remove_outlier_output.npy")).unwrap();
    let params: ndarray::Array2<f64> =
        read_npy(format!("{FIXTURES}/remove_outlier_params.npy")).unwrap();

    let ncase = inputs.dim().0;
    for k in 0..ncase {
        let dif = params[[k, 0]] as f32;
        let size = params[[k, 1]] as usize;
        let axis = params[[k, 2]] as usize;

        let input: Array3<f32> = inputs.index_axis(Axis(0), k).to_owned();
        let mut tomo = Tomo::new(input, Layout::Projection);
        remove_outlier(&mut tomo, dif, size, axis).unwrap();

        let want = outputs.index_axis(Axis(0), k);
        let mut mismatch = 0usize;
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                mismatch += 1;
            }
        }
        assert_eq!(
            mismatch, 0,
            "case {k} (dif={dif}, size={size}, axis={axis}): {mismatch} f32 bit-mismatches vs tomopy"
        );
    }
}
