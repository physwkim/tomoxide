//! Distribution-parity tests for the noise models.
//!
//! tomopy draws Gaussian/Poisson noise from numpy's global MT19937 generator,
//! whose bit-stream we cannot reproduce from Rust. So instead of an array Δ
//! against a golden, these assert the samples follow the *same distribution*
//! tomopy's numpy calls produce — matched by their moments:
//!
//! * `add_gaussian` → mean and standard deviation of the added noise.
//! * `add_poisson`  → mean, variance, *and skewness* (≈ 1/√λ). The skewness
//!   check is the load-bearing one for `λ ≥ 10`: it fails for a normal
//!   approximation (skew → 0) and passes only for a genuine Poisson, so it
//!   verifies the Hörmann PTRS path, not just the first two moments.
//! * `add_rings`    → mean/std of the per-pixel sensitivity, plus the *load-
//!   bearing structural* check that the sensitivity is constant across the
//!   angle axis (a ring, not per-element noise).
//! * `add_zingers`  → the saturated fraction matches `f`, saturated cells equal
//!   `sat`, and untouched cells keep their value.
//!
//! All tolerances are many standard errors wide of the sampling noise at the
//! sample sizes used, so the bound is on a real distributional defect, not on
//! a lucky seed.

use ndarray::Array3;
use tomoxide_core::data::{Layout, Tomo};

fn tomo_filled(n: usize, value: f32) -> Tomo<f32> {
    Tomo::new(Array3::from_elem((1, n, n), value), Layout::Sinogram)
}

/// Population mean of all elements.
fn mean(a: &Array3<f32>) -> f64 {
    let n = a.len() as f64;
    a.iter().map(|&v| v as f64).sum::<f64>() / n
}

/// Population central moments (m2, m3) about the sample mean.
fn central_moments(a: &Array3<f32>) -> (f64, f64, f64) {
    let m = mean(a);
    let n = a.len() as f64;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &v in a.iter() {
        let d = v as f64 - m;
        m2 += d * d;
        m3 += d * d * d;
    }
    (m, m2 / n, m3 / n)
}

#[test]
fn add_gaussian_matches_distribution() {
    // 65536 samples → standard error of the mean = std/256 ≈ 0.008; the 0.05
    // tolerance is ~6 SE, so it bounds a real bias, not sampling noise.
    let mut t = tomo_filled(256, 0.0);
    tomoxide_sim::add_gaussian(&mut t, 0.5, Some(2.0), 0xC0FFEE).unwrap();

    let (m, m2, _) = central_moments(&t.array);
    let std = m2.sqrt();
    assert!((m - 0.5).abs() < 0.05, "gaussian mean = {m} (want 0.5)");
    assert!((std - 2.0).abs() < 0.05, "gaussian std = {std} (want 2.0)");
}

#[test]
fn add_gaussian_default_std_is_five_percent_of_max() {
    // std=None → tomopy uses data.max() * 0.05. Constant array of 100 → max
    // 100 → noise std 5; the resulting array's std should track 5.
    let mut t = tomo_filled(256, 100.0);
    tomoxide_sim::add_gaussian(&mut t, 0.0, None, 7).unwrap();

    let (_, m2, _) = central_moments(&t.array);
    let std = m2.sqrt();
    assert!(
        (std - 5.0).abs() < 0.2,
        "default-std gaussian std = {std} (want 100*0.05 = 5)"
    );
}

#[test]
fn add_gaussian_is_deterministic_per_seed() {
    let mut a = tomo_filled(32, 1.0);
    let mut b = tomo_filled(32, 1.0);
    let mut c = tomo_filled(32, 1.0);
    tomoxide_sim::add_gaussian(&mut a, 0.0, Some(1.0), 42).unwrap();
    tomoxide_sim::add_gaussian(&mut b, 0.0, Some(1.0), 42).unwrap();
    tomoxide_sim::add_gaussian(&mut c, 0.0, Some(1.0), 43).unwrap();

    assert_eq!(a.array, b.array, "same seed must reproduce the same draw");
    assert_ne!(a.array, c.array, "a different seed must change the draw");
}

/// Drive one λ through `add_poisson` and return (mean, variance, skewness).
fn poisson_moments(lam: f32, side: usize, seed: u64) -> (f64, f64, f64) {
    let mut t = tomo_filled(side, lam);
    tomoxide_sim::add_poisson(&mut t, seed).unwrap();
    // Every sample must be a non-negative integer count.
    for &v in t.array.iter() {
        assert!(
            v >= 0.0 && v.fract() == 0.0,
            "poisson sample {v} is not a count"
        );
    }
    let (m, m2, m3) = central_moments(&t.array);
    (m, m2, m3 / m2.powf(1.5))
}

#[test]
fn add_poisson_small_lambda_matches_distribution() {
    // λ = 2 exercises the Knuth multiplication path (λ < 10). 262144 samples.
    let lam = 2.0;
    let (m, var, skew) = poisson_moments(lam, 512, 0xABCD);
    assert!(
        (m - lam as f64).abs() < 0.05,
        "poisson(2) mean = {m} (want 2)"
    );
    assert!(
        (var - lam as f64).abs() < 0.1,
        "poisson(2) var = {var} (want 2)"
    );
    // skew of Poisson(λ) = 1/√λ = 0.7071 here.
    let want_skew = 1.0 / (lam as f64).sqrt();
    assert!(
        (skew - want_skew).abs() < 0.1,
        "poisson(2) skew = {skew} (want {want_skew})"
    );
}

#[test]
fn add_poisson_large_lambda_matches_distribution() {
    // λ = 50 exercises the Hörmann PTRS path (λ ≥ 10). The skew check is the
    // point: 1/√50 ≈ 0.1414, ~30 SE away from the 0 a normal approximation
    // would give, so passing it proves the sampler is a true Poisson.
    let lam = 50.0;
    let (m, var, skew) = poisson_moments(lam, 512, 0x1234);
    assert!(
        (m - lam as f64).abs() < 0.2,
        "poisson(50) mean = {m} (want 50)"
    );
    assert!(
        (var - lam as f64).abs() < 1.0,
        "poisson(50) var = {var} (want 50)"
    );
    // skew of Poisson(λ) = 1/√λ = 0.1414 here; ~30 SE from the 0 of a normal.
    let want_skew = 1.0 / (lam as f64).sqrt();
    assert!(
        (skew - want_skew).abs() < 0.05,
        "poisson(50) skew = {skew} (want {want_skew})"
    );
}

#[test]
fn add_poisson_is_deterministic_and_rejects_negative() {
    let mut a = tomo_filled(32, 10.0);
    let mut b = tomo_filled(32, 10.0);
    tomoxide_sim::add_poisson(&mut a, 99).unwrap();
    tomoxide_sim::add_poisson(&mut b, 99).unwrap();
    assert_eq!(a.array, b.array, "same seed must reproduce the same draw");

    // A negative intensity is not a valid Poisson mean: error, no mutation.
    let mut neg = tomo_filled(4, 5.0);
    neg.array[[0, 1, 1]] = -3.0;
    let before = neg.array.clone();
    assert!(tomoxide_sim::add_poisson(&mut neg, 1).is_err());
    assert_eq!(
        neg.array, before,
        "a rejected call must not mutate the data"
    );
}

#[test]
fn add_rings_is_constant_across_angles_and_matches_std() {
    // Ring artifacts come from a *fixed* per-pixel sensitivity: every angle
    // sees the same (row, col) value. With a 1.0 input the output equals the
    // sensitivity itself, drawn from N(1, std).
    let (n_ang, n_row, n_col) = (8usize, 64usize, 64usize);
    let std = 0.05f32;
    let mut t = Tomo::new(
        Array3::from_elem((n_ang, n_row, n_col), 1.0f32),
        Layout::Projection,
    );
    tomoxide_sim::add_rings(&mut t, std, 0xBEEF).unwrap();

    // Structural (load-bearing): the sensitivity does not vary with angle. This
    // is what makes it a ring rather than the per-element noise of add_gaussian.
    for a in 0..n_ang {
        for r in 0..n_row {
            for c in 0..n_col {
                assert_eq!(
                    t.array[[a, r, c]],
                    t.array[[0, r, c]],
                    "ring sensitivity must be constant across the angle axis"
                );
            }
        }
    }

    // Distributional: 4096 distinct pixels → SE of the mean ≈ 0.05/64 ≈ 8e-4;
    // the 0.01 tolerance is ~13 SE, so it bounds a real bias, not the seed.
    let (m, m2, _) = central_moments(&t.array);
    let s = m2.sqrt();
    assert!(
        (m - 1.0).abs() < 0.01,
        "ring sensitivity mean = {m} (want 1.0)"
    );
    assert!(
        (s - std as f64).abs() < 0.01,
        "ring sensitivity std = {s} (want {std})"
    );
}

#[test]
fn add_rings_is_deterministic_per_seed() {
    let base = Tomo::new(Array3::from_elem((4, 16, 16), 1.0f32), Layout::Projection);
    let (mut a, mut b, mut c) = (base.clone(), base.clone(), base.clone());
    tomoxide_sim::add_rings(&mut a, 0.05, 7).unwrap();
    tomoxide_sim::add_rings(&mut b, 0.05, 7).unwrap();
    tomoxide_sim::add_rings(&mut c, 0.05, 8).unwrap();
    assert_eq!(a.array, b.array, "same seed must reproduce the same draw");
    assert_ne!(a.array, c.array, "a different seed must change the draw");
}

#[test]
fn add_zingers_saturates_fraction_f() {
    // Each element is independently set to `sat` with probability f.
    let f = 0.02f32;
    let sat = 65536.0f32;
    let mut t = tomo_filled(512, 0.0); // 262144 elements, all 0.0
    tomoxide_sim::add_zingers(&mut t, f, sat, 0x5A17).unwrap();

    let mut n_sat = 0usize;
    for &v in t.array.iter() {
        if v == sat {
            n_sat += 1;
        } else {
            assert_eq!(v, 0.0, "a non-zinger element must keep its original value");
        }
    }
    let frac = n_sat as f64 / t.array.len() as f64;
    // SE of the fraction = sqrt(f(1-f)/N) ≈ sqrt(0.02·0.98/262144) ≈ 2.7e-4;
    // the 0.002 tolerance is ~7 SE.
    assert!(
        (frac - f as f64).abs() < 0.002,
        "zinger fraction = {frac} (want {f})"
    );
}

#[test]
fn add_zingers_is_deterministic_per_seed() {
    let mut a = tomo_filled(32, 1.0);
    let mut b = tomo_filled(32, 1.0);
    let mut c = tomo_filled(32, 1.0);
    tomoxide_sim::add_zingers(&mut a, 0.1, 9.0, 3).unwrap();
    tomoxide_sim::add_zingers(&mut b, 0.1, 9.0, 3).unwrap();
    tomoxide_sim::add_zingers(&mut c, 0.1, 9.0, 4).unwrap();
    assert_eq!(a.array, b.array, "same seed must reproduce the same draw");
    assert_ne!(a.array, c.array, "a different seed must change the draw");
}
