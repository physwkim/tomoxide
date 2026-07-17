//! `find_center_rings` — the ring/bullseye rotation-axis estimator.
//!
//! The physics (a 360° mean projection is a bullseye centred on the axis) is
//! validated against real scans by `examples/ring_center_validate.rs`, which
//! reproduces `docs/LAMINOGRAPHY_ALIGNMENT.md`'s reference numbers on the two
//! pouch datasets. These tests cover what CI can hold: that a bullseye is
//! registered at its true column, and — the part that matters more — that a
//! pattern which is *not* a bullseye is reported as untrustworthy instead of
//! returning a confident wrong number.

use ndarray::{Array3, Axis};
use tomoxide::data::{Layout, Tomo};
use tomoxide::recon::center::find_center_rings;
use tomoxide::CpuBackend;

const NY: usize = 96;
const NX: usize = 128;

/// A bullseye: concentric rings centred on column `cx`, row `cy`.
fn bullseye(cx: f32, cy: f32) -> Array3<f32> {
    let mut a = Array3::<f32>::zeros((4, NY, NX));
    for p in 0..4 {
        for y in 0..NY {
            for x in 0..NX {
                let r = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                // Rings every 8 px, fading out — a mean projection's look.
                let v = (r * std::f32::consts::TAU / 8.0).cos() * (-r / 40.0).exp();
                a[[p, y, x]] = v.max(0.0);
            }
        }
    }
    a
}

fn probe(a: Array3<f32>) -> tomoxide::recon::center::RingCenter {
    let cpu = CpuBackend::new();
    let tomo = Tomo::new(a, Layout::Projection);
    find_center_rings(&tomo, &cpu, 1).unwrap()
}

/// A bullseye registers at its own column, off-centre included — the whole point
/// is that the axis need not sit at nx/2.
#[test]
fn ring_center_finds_the_bullseye_column() {
    for &cx in &[64.0f32, 50.0, 78.0, 44.5] {
        let r = probe(bullseye(cx, NY as f32 / 2.0));
        eprintln!(
            "true {cx} -> centre {:.2} (Δ {:+.2}), prominence {:.1}, trustworthy {}",
            r.center,
            r.center - cx,
            r.prominence,
            r.trustworthy
        );
        assert!(
            (r.center - cx).abs() < 1.0,
            "bullseye at column {cx}: got {:.2}",
            r.center
        );
        assert!(
            r.trustworthy,
            "a clean bullseye at {cx} should be trustworthy, prominence {:.1}",
            r.prominence
        );
    }
}

/// The row of the bullseye is irrelevant — only the column is a free parameter,
/// which is the reason this estimator only reports one.
#[test]
fn ring_center_ignores_the_bullseye_row() {
    let a = probe(bullseye(70.0, NY as f32 / 2.0));
    let b = probe(bullseye(70.0, NY as f32 / 2.0 - 15.0));
    eprintln!("centred row {:.2}, offset row {:.2}", a.center, b.center);
    assert!(
        (a.center - b.center).abs() < 1.0,
        "moving the rings vertically moved the estimate: {:.2} vs {:.2}",
        a.center,
        b.center
    );
}

/// The alibi. Structure with no ring symmetry must come back untrustworthy: on
/// the real mis-aligned scan the estimator returns 281 against a true 138, and
/// the only thing standing between that number and a wasted reconstruction is
/// this flag.
#[test]
fn ring_center_flags_a_pattern_that_is_not_a_bullseye() {
    // A one-sided ramp plus stripes: plenty of signal, no concentric rings.
    let mut a = Array3::<f32>::zeros((4, NY, NX));
    for p in 0..4 {
        for y in 0..NY {
            for x in 0..NX {
                let ramp = x as f32 / NX as f32;
                let stripe = ((y as f32 / 5.0).sin() * 0.2).abs();
                a[[p, y, x]] = ramp + stripe;
            }
        }
    }
    let r = probe(a);
    eprintln!(
        "non-bullseye -> centre {:.2}, prominence {:.2}, trustworthy {}",
        r.center, r.prominence, r.trustworthy
    );
    assert!(
        !r.trustworthy,
        "a pattern with no ring symmetry was reported trustworthy (prominence {:.2}) — \
         the mis-alignment flag is what keeps a confident wrong centre out of a recon",
        r.prominence
    );
}

/// A trustworthy verdict has to mean *more* than a non-trustworthy one: the two
/// classes must not overlap on the same input scale.
#[test]
fn ring_center_prominence_separates_rings_from_noise() {
    let rings = probe(bullseye(64.0, NY as f32 / 2.0));
    let mut flat = Array3::<f32>::zeros((4, NY, NX));
    // Deterministic pseudo-noise: no rings, no symmetry.
    let mut s = 12345u32;
    for p in 0..4 {
        for y in 0..NY {
            for x in 0..NX {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                flat[[p, y, x]] = (s >> 16) as f32 / 65535.0;
            }
        }
    }
    let noise = probe(flat);
    eprintln!(
        "rings prominence {:.1} vs noise prominence {:.1}",
        rings.prominence, noise.prominence
    );
    assert!(
        rings.prominence > noise.prominence * 2.0,
        "rings {:.1} should stand well clear of noise {:.1}",
        rings.prominence,
        noise.prominence
    );
}

/// `step` subsamples projections; on a stack whose frames are identical it must
/// not change the answer. (Guards the mean's divisor against the stride.)
#[test]
fn ring_center_step_does_not_shift_the_estimate() {
    let cpu = CpuBackend::new();
    let mut a = Array3::<f32>::zeros((20, NY, NX));
    let one = bullseye(58.0, NY as f32 / 2.0);
    for p in 0..20 {
        a.index_axis_mut(Axis(0), p)
            .assign(&one.index_axis(Axis(0), 0));
    }
    let tomo = Tomo::new(a, Layout::Projection);
    let full = find_center_rings(&tomo, &cpu, 1).unwrap();
    let strided = find_center_rings(&tomo, &cpu, 7).unwrap();
    eprintln!(
        "step 1 -> {:.3}, step 7 -> {:.3}",
        full.center, strided.center
    );
    assert!(
        (full.center - strided.center).abs() < 1e-3,
        "step changed the estimate: {:.3} vs {:.3}",
        full.center,
        strided.center
    );
}
