//! Numeric parity against tomopy for `median_filter_nonfinite`.
//!
//! Each non-finite value (NaN/±inf) is replaced by the median of the finite
//! values in its `size×size` neighbourhood along the last two axes, with the
//! medians read from a per-projection snapshot taken before any correction. The
//! op is pure NumPy (`np.isfinite`/`np.nonzero`/`np.median`) with order-free
//! medians, so the port reproduces tomopy 1.15.3 **bit-for-bit (Δ = 0)**. Golden
//! from the real tomopy `tools/gen_tomopy_median_nonfinite_golden.py` (size 3
//! and 5, with NaN, +inf and −inf scattered and boundary-clamped corner
//! kernels).

use ndarray::{Array1, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::prep::filters::median_filter_nonfinite;
use tomoxide_core::data::{Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn median_filter_nonfinite_matches_tomopy() {
    // (ncase, n0, n1, n2)
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/median_nonfinite_input.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/median_nonfinite_output.npy")).unwrap();
    let sizes: Array1<i64> = read_npy(format!("{FIXTURES}/median_nonfinite_sizes.npy")).unwrap();

    let ncase = inputs.dim().0;
    assert_eq!(outputs.dim().0, ncase);
    assert_eq!(sizes.len(), ncase);

    for k in 0..ncase {
        let size = sizes[k] as usize;
        let mut tomo = Tomo::new(inputs.index_axis(Axis(0), k).to_owned(), Layout::Projection);
        median_filter_nonfinite(&mut tomo, size).unwrap();

        let want = outputs.index_axis(Axis(0), k);
        let mut n_mismatch = 0usize;
        let mut max_abs = 0.0f32;
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                n_mismatch += 1;
                let d = (g - w).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
        }
        assert_eq!(
            n_mismatch, 0,
            "case {k} (size={size}): {n_mismatch} f32 bit-mismatches vs tomopy, max|Δ|={max_abs:e}"
        );
    }
}

#[test]
fn median_filter_nonfinite_rejects_all_nonfinite_kernel() {
    // A whole 1×N slice of NaN leaves no finite neighbour → must error, not
    // silently keep NaN (tomopy raises ValueError).
    let arr = ndarray::Array3::from_elem((1, 3, 3), f32::NAN);
    let mut tomo = Tomo::new(arr, Layout::Projection);
    assert!(median_filter_nonfinite(&mut tomo, 3).is_err());
}
