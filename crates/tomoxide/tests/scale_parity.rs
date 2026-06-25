//! Bit-exact parity against tomopy for `scale` (alignment.py:460).
//!
//! Scales a projection stack into `[−1, 1]` by `scl = max(|max|, |min|)`, then
//! divides in place. Pure order statistics (`max`/`min`/`abs`) plus an
//! elementwise f32 divide — no summation or transcendental — so both the scaled
//! array and the returned `scl` are bit-exact (Δ=0).
//!
//! Golden from `tools/gen_tomopy_scale_golden.py` (real tomopy 1.15.3). Cases:
//! positive-dominated, negative-dominated (|min| is the peak), symmetric, and
//! small-magnitude stacks.

use ndarray::{Array1, Array3, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::prep::scale;
use tomoxide::data::{Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn scale_matches_tomopy() {
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/scale_input.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/scale_output.npy")).unwrap();
    let scls: Array1<f64> = read_npy(format!("{FIXTURES}/scale_scl.npy")).unwrap();

    let ncase = inputs.dim().0;
    for k in 0..ncase {
        let input: Array3<f32> = inputs.index_axis(Axis(0), k).to_owned();
        let mut tomo = Tomo::new(input, Layout::Projection);
        let scl = scale(&mut tomo).unwrap();

        assert_eq!(
            scl.to_bits(),
            (scls[k] as f32).to_bits(),
            "case {k}: scl {scl} != tomopy {}",
            scls[k] as f32
        );

        let want = outputs.index_axis(Axis(0), k);
        let mut mismatch = 0usize;
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                mismatch += 1;
            }
        }
        assert_eq!(
            mismatch, 0,
            "case {k}: {mismatch} f32 bit-mismatches vs tomopy"
        );
    }
}
