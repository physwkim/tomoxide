//! Numeric parity against tomopy for polar-transform ring removal.
//!
//! `remove_ring` operates on the reconstructed image (projector-independent):
//! nearest-pixel polar transforms (integer `iroundf` indices), a radial median
//! filter (pure selection), and an azimuthal mean filter (f64 running sum,
//! matching the C's `long double` on this 64-bit platform). The trig / `sqrt`
//! go through the same libm as tomopy, so it is held to tomopy parity at the
//! f32 round-off floor. Goldens from tomopy 1.15.3 `remove_ring` for both
//! `int_mode='WRAP'` and `'REFLECT'` (`tools/gen_tomopy_ring_golden.py`).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::recon::ring::RingIntMode;
use tomoxide::{recon, Volume};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

fn run(input: &Array3<f32>, rwidth: i32, int_mode: RingIntMode) -> Array3<f32> {
    let mut vol = Volume::new(input.clone());
    // center_x/center_y = (nx-1)/2, (ny-1)/2 (= 39.5 for 80-wide), tomopy
    // defaults; thresh/thresh_max/thresh_min/theta_min = tomopy defaults.
    recon::ring::remove_ring(
        &mut vol, 39.5, 39.5, 300.0, -100.0, 300.0, 30, rwidth, int_mode,
    )
    .unwrap();
    vol.array
}

fn max_abs(got: &Array3<f32>, want: &Array3<f32>) -> f32 {
    got.iter()
        .zip(want.iter())
        .fold(0.0f32, |m, (&g, &w)| m.max((g - w).abs()))
}

#[test]
fn remove_ring_matches_tomopy() {
    let input = load("ring_input.npy");

    let scale = input.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

    let got2 = run(&input, 2, RingIntMode::Wrap);
    let d2 = max_abs(&got2, &load("tomopy_ring_rw2.npy"));

    let got4 = run(&input, 4, RingIntMode::Wrap);
    let d4 = max_abs(&got4, &load("tomopy_ring_rw4.npy"));

    // Measured bit-exact (Δ = 0) here: integer polar indices and the median
    // filter are platform-independent, and the mean filter's f64 running sum
    // equals the C's `long double` on this 64-bit-`long double` platform. The
    // small tolerance only guards a host whose `long double` is 80-bit (where
    // the azimuthal mean could differ at the f32 round-off floor).
    let tol = 1e-5 * scale;
    assert!(
        d2 <= tol && d4 <= tol,
        "ring parity (WRAP): rw2 max|Δ| = {d2}, rw4 max|Δ| = {d4}, tol = {tol} (scale {scale})"
    );
}

#[test]
fn remove_ring_reflect_matches_tomopy() {
    let input = load("ring_input.npy");
    let scale = input.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

    let got2 = run(&input, 2, RingIntMode::Reflect);
    let d2 = max_abs(&got2, &load("tomopy_ring_reflect_rw2.npy"));

    let got4 = run(&input, 4, RingIntMode::Reflect);
    let d4 = max_abs(&got4, &load("tomopy_ring_reflect_rw4.npy"));

    // Same f64-running-sum reasoning as WRAP; the REFLECT branch mirrors each
    // polar half at its 0/π and π/2π edges instead of wrapping. The fixture's
    // WRAP and REFLECT goldens differ (max|Δ| ≈ 0.018), so this exercises the
    // reflect path rather than re-testing WRAP.
    let tol = 1e-5 * scale;
    assert!(
        d2 <= tol && d4 <= tol,
        "ring parity (REFLECT): rw2 max|Δ| = {d2}, rw4 max|Δ| = {d4}, tol = {tol} (scale {scale})"
    );
}
