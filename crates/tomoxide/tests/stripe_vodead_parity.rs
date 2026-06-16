//! Parity against tomopy for Vo dead-stripe removal
//! (`remove_dead_stripe`, Vo 2018 algorithm 6).
//!
//! `_rs_dead` smooths each detector column over projections, scores each column
//! by its summed deviation from that smooth, detects the unresponsive/fluctuating
//! columns, and fills them by per-row linear interpolation across the good
//! columns (`RectBivariateSpline`, kx=ky=1). When `norm=true` a residual
//! `_rs_large` pass then removes wide stripes. The bilinear fill is arithmetic, so
//! both cases are held to the f32 round-off floor (max rel ≤ 1e-5).
//!
//! `snr=2` makes the dead-column detection fire (col 55, a near-dead ramp). The
//! two cases differ structurally on the injected large stripes (cols 30/75/100):
//!   * `norm=true`  — dead column filled AND large stripes removed,
//!   * `norm=false` — dead column filled only; the large stripes stay untouched.
//!
//! Goldens from the real tomopy 1.15.3 (`tools/gen_tomopy_stripe_vodead_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, Layout, StripeMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

fn run(norm: bool) -> (Array3<f32>, Array3<f32>) {
    let input = load("stripe_vodead_input.npy");
    let mut tomo = Tomo::new(input.clone(), Layout::Projection);
    prep::stripe::remove_stripe(
        &mut tomo,
        StripeMethod::VoDead {
            snr: 2.0,
            size: 51,
            norm,
        },
    )
    .unwrap();
    (input, tomo.to_layout(Layout::Projection).array)
}

/// Max |Δ| over a single detector column `c` (across all projections and rows).
fn col_max_abs(a: &Array3<f32>, b: &Array3<f32>, c: usize) -> f32 {
    let (np, nr, _) = a.dim();
    let mut m = 0.0f32;
    for p in 0..np {
        for r in 0..nr {
            m = m.max((a[[p, r, c]] - b[[p, r, c]]).abs());
        }
    }
    m
}

fn assert_golden_parity(golden_name: &str, got: &Array3<f32>) {
    let golden = load(golden_name);
    assert_eq!(got.dim(), golden.dim());
    let scale = golden.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_abs = golden
        .iter()
        .zip(got.iter())
        .fold(0.0f32, |m, (&g, &p)| m.max((g - p).abs()));
    let max_rel = max_abs / scale;
    eprintln!("{golden_name}: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})");
    assert!(
        max_rel <= 1e-5,
        "{golden_name} golden parity: max|Δ| = {max_abs}, max relative = {max_rel}"
    );
}

#[test]
fn vodead_norm_true_matches_tomopy() {
    let (input, got) = run(true);
    assert_golden_parity("tomopy_stripe_vodead_norm.npy", &got);
    // The dead column is filled and the residual pass removes the large stripes.
    assert!(
        col_max_abs(&input, &got, 55) > 0.1,
        "dead col 55 should be filled"
    );
    for c in [30usize, 75, 100] {
        assert!(
            col_max_abs(&input, &got, c) > 0.1,
            "norm=true should remove the large stripe at col {c}"
        );
    }
}

#[test]
fn vodead_norm_false_fills_dead_only() {
    let (input, got) = run(false);
    assert_golden_parity("tomopy_stripe_vodead_raw.npy", &got);
    // The dead column is filled, but with no residual pass the large stripes
    // (and every other column) are left exactly as the input — Δ = 0.
    assert!(
        col_max_abs(&input, &got, 55) > 0.1,
        "dead col 55 should be filled"
    );
    for c in [30usize, 75, 100] {
        assert_eq!(
            col_max_abs(&input, &got, c),
            0.0,
            "norm=false must leave the large stripe at col {c} untouched"
        );
    }
}
