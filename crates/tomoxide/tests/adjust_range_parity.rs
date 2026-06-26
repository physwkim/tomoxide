//! Numeric parity against tomopy for `adjust_range`.
//!
//! Clips the dynamic range to `[dmin, dmax]`, with `None` bounds defaulting to
//! the data min/max and each bound applied only when strictly tighter than the
//! data range. Pure NumPy (`np.max`/`np.min` + boolean-mask assignment) with no
//! summation, so the port reproduces tomopy 1.15.3 **bit-for-bit (Δ = 0)**.
//! Golden from the real tomopy `tools/gen_tomopy_adjust_range_golden.py` (both
//! None, high-only, low-only, both, and looser-than-data no-op cases; `None`
//! encoded as NaN in the param arrays).

use ndarray::{Array1, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::data::{Layout, Tomo};
use tomoxide::prep::filters::adjust_range;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn opt(v: f64) -> Option<f32> {
    if v.is_nan() {
        None
    } else {
        Some(v as f32)
    }
}

#[test]
fn adjust_range_matches_tomopy() {
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/adjust_range_input.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/adjust_range_output.npy")).unwrap();
    let dmins: Array1<f64> = read_npy(format!("{FIXTURES}/adjust_range_dmin.npy")).unwrap();
    let dmaxs: Array1<f64> = read_npy(format!("{FIXTURES}/adjust_range_dmax.npy")).unwrap();

    let ncase = inputs.dim().0;
    for k in 0..ncase {
        let mut tomo = Tomo::new(inputs.index_axis(Axis(0), k).to_owned(), Layout::Projection);
        adjust_range(&mut tomo, opt(dmins[k]), opt(dmaxs[k])).unwrap();

        let want = outputs.index_axis(Axis(0), k);
        let mut n_mismatch = 0usize;
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                n_mismatch += 1;
            }
        }
        assert_eq!(
            n_mismatch,
            0,
            "case {k} (dmin={:?}, dmax={:?}): {n_mismatch} f32 bit-mismatches vs tomopy",
            opt(dmins[k]),
            opt(dmaxs[k])
        );
    }
}
