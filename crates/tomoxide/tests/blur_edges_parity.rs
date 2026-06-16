//! Bit-exact parity against tomopy for `blur_edges` (prep/alignment.py:482).
//!
//! Multiplies every projection image by a radial feather mask. `rad` uses `√`
//! (IEEE-correctly-rounded, so bit-exact vs numpy), the mask is plain f64
//! arithmetic in numpy's order, and the in-place `float32 *= float64` is the f64
//! product cast to f32 — so the result matches tomopy **bit-for-bit (Δ=0)**.
//!
//! Golden from the real tomopy `tools/gen_tomopy_blur_edges_golden.py`: the
//! upstream default `(low=0, high=0.8)` and a non-zero inner radius `(0.2, 0.9)`.

use ndarray::{Array2, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::prep::blur_edges;
use tomoxide_core::data::{Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn blur_edges_matches_tomopy() {
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/blur_edges_input.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/blur_edges_output.npy")).unwrap();
    let params: Array2<f64> = read_npy(format!("{FIXTURES}/blur_edges_params.npy")).unwrap();

    let ncase = inputs.dim().0;
    for k in 0..ncase {
        let (low, high) = (params[[k, 0]], params[[k, 1]]);
        // blur_edges treats axis 0 as the projection axis → Projection layout.
        let mut tomo = Tomo::new(inputs.index_axis(Axis(0), k).to_owned(), Layout::Projection);
        blur_edges(&mut tomo, low, high).unwrap();

        let want = outputs.index_axis(Axis(0), k);
        let mut mismatch = 0usize;
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                mismatch += 1;
            }
        }
        assert_eq!(
            mismatch, 0,
            "case {k} (low={low}, high={high}): {mismatch} f32 bit-mismatches vs tomopy"
        );
    }
}
