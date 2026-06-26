//! Numeric parity against tomopy for `add_drift`.
//!
//! Scales each projection angle `i` by `drift[i] = amp·sin(2π·i/period) + mean +
//! linspace(0,1)[i]`, constant across the detector. This is deterministic (no
//! RNG), so it is held to true numeric parity rather than the distribution
//! parity of the add_gaussian/poisson/... models — but only to the **f32
//! round-off floor, not Δ = 0**: numpy 2.x evaluates `np.sin` on the f64 angle
//! array with its own vectorized routine, which differs from Rust's libm
//! `f64::sin` by ≤ 1 ULP for some angles, and that f64 difference survives the
//! f32 cast in a small fraction of pixels. (The f64·f32 product is commutative,
//! so the drift's `sin` is the only divergence — everything else matches
//! exactly.) Each pixel is therefore held to ≤ 1 f32 ULP, the same precision
//! class as the single-precision phase-retrieval ports. Golden from the real
//! tomopy `tools/gen_tomopy_add_drift_golden.py` (defaults plus shorter and
//! fractional periods).

use ndarray::{Array2, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::data::{Layout, Tomo};
use tomoxide::sim::add_drift;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn add_drift_matches_tomopy() {
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/add_drift_input.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/add_drift_output.npy")).unwrap();
    let params: Array2<f64> = read_npy(format!("{FIXTURES}/add_drift_params.npy")).unwrap();

    let ncase = inputs.dim().0;
    let mut worst_rel = 0.0f32;
    let mut worst_abs = 0.0f32;
    let mut total_mismatch = 0usize;
    let mut total = 0usize;
    for k in 0..ncase {
        let (amp, period, mean) = (
            params[[k, 0]] as f32,
            params[[k, 1]] as f32,
            params[[k, 2]] as f32,
        );
        // Golden axis 0 is the rotation/angle axis → Projection layout.
        let mut tomo = Tomo::new(inputs.index_axis(Axis(0), k).to_owned(), Layout::Projection);
        add_drift(&mut tomo, amp, period, mean).unwrap();

        let want = outputs.index_axis(Axis(0), k);
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            total += 1;
            let d = (g - w).abs();
            // Per-pixel tolerance: 1 f32 ULP relative to the magnitude.
            let tol = f32::EPSILON * w.abs().max(1.0);
            assert!(
                d <= tol,
                "case {k} (amp={amp}, period={period}, mean={mean}): |Δ|={d:e} > 1 ULP ({tol:e}) at value {w}"
            );
            if g.to_bits() != w.to_bits() {
                total_mismatch += 1;
                worst_abs = worst_abs.max(d);
                worst_rel = worst_rel.max(d / w.abs().max(1.0));
            }
        }
    }
    // Document the realised floor: a small fraction of pixels differ by ≤ 1 ULP
    // (numpy's vectorized f64 sin vs libm), the rest are bit-identical.
    println!(
        "add_drift: {total_mismatch}/{total} pixels differ (≤1 ULP); worst |Δ|={worst_abs:e}, worst rel={worst_rel:e}"
    );
}
