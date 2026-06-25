//! Numeric parity against tomopy for `gaussian_filter` (misc/corr.py:118).
//!
//! Gaussian-filters every 2-D slice along `axis` with scipy.ndimage's separable
//! Gaussian (reflect boundaries, truncate=4). The convolution accumulates in f64
//! exactly like scipy's `NI_Correlate1D` (symmetric branch for order 0/2,
//! anti-symmetric for order 1) with the intermediate cast to f32 between the two
//! passes, and the kernel is normalised by numpy's f64 pairwise sum. The only
//! divergence is the kernel's `exp` — numpy's vectorised f64 `exp` can differ
//! from libm by <=1 ULP — so the result is held to the **f32 round-off floor**,
//! the same precision class as add_drift and the Fourier stripe ports. (For
//! these fixtures numpy's `exp` matched libm exactly, so the realised
//! divergence is Δ=0 / 0 mismatches; the tolerance documents the class.)
//!
//! Golden from the real tomopy `tools/gen_tomopy_gaussian_filter_golden.py`
//! (default sigma=3 order=0 on each axis, a small-radius sigma, and the
//! derivative orders 1 and 2).

use ndarray::{Array2, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::prep::filters::gaussian_filter;
use tomoxide::data::{Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn gaussian_filter_matches_tomopy() {
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/gaussian_filter_input.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/gaussian_filter_output.npy")).unwrap();
    let params: Array2<f64> = read_npy(format!("{FIXTURES}/gaussian_filter_params.npy")).unwrap();

    let ncase = inputs.dim().0;
    let mut worst_abs = 0.0f32;
    let mut worst_rel = 0.0f32;
    let mut total_mismatch = 0usize;
    let mut total = 0usize;
    for k in 0..ncase {
        let sigma = params[[k, 0]];
        let order = params[[k, 1]] as usize;
        let axis = params[[k, 2]] as usize;
        // gaussian_filter's `axis` indexes the raw 3-D array; Projection layout
        // keeps `.array` identical to the numpy array, so axes map directly.
        let mut tomo = Tomo::new(inputs.index_axis(Axis(0), k).to_owned(), Layout::Projection);
        gaussian_filter(&mut tomo, sigma, order, axis).unwrap();

        let want = outputs.index_axis(Axis(0), k);
        let mut case_mismatch = 0usize;
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            total += 1;
            if g.to_bits() != w.to_bits() {
                total_mismatch += 1;
                case_mismatch += 1;
                let d = (g - w).abs();
                worst_abs = worst_abs.max(d);
                // Relative to the slice's own scale (the derivative cases have
                // tiny magnitudes, so divide by the case's peak |output|).
                let scale = want.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-30);
                worst_rel = worst_rel.max(d / scale);
            }
        }
        // Per case: the divergence stays at the f32 round-off floor (<=1 ULP of
        // the slice's scale). 4 ULP gives margin for the exp floor compounding
        // across the two separable passes.
        let scale = want.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-30);
        let tol = 4.0 * f32::EPSILON * scale;
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            let d = (g - w).abs();
            assert!(
                d <= tol,
                "case {k} (sigma={sigma}, order={order}, axis={axis}): |Δ|={d:e} > {tol:e} at value {w}"
            );
        }
        let _ = case_mismatch;
    }
    println!(
        "gaussian_filter: {total_mismatch}/{total} pixels differ (<=f32 round-off floor); \
         worst |Δ|={worst_abs:e}, worst rel={worst_rel:e}"
    );
}
