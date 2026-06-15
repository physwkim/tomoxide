//! Numeric parity against tomopy for Paganin phase retrieval.
//!
//! `retrieve_phase` is a Fourier low-pass on each (zero/edge-padded) radiograph
//! — projector-independent — so it is held to tomopy parity. Golden from tomopy
//! 1.15.3 `retrieve_phase` (`tools/gen_tomopy_phase_golden.py`), default params
//! (pixel_size=1e-4, dist=50, energy=20, alpha=1e-3).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, CpuBackend, Layout, PhaseMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn paganin_matches_tomopy() {
    let input: Array3<f32> = read_npy(format!("{FIXTURES}/phase_input.npy")).unwrap();
    let golden: Array3<f32> = read_npy(format!("{FIXTURES}/tomopy_phase.npy")).unwrap();

    let mut tomo = Tomo::new(input, Layout::Projection);
    let cpu = CpuBackend::new();
    prep::retrieve_phase(
        &mut tomo,
        PhaseMethod::Paganin {
            pixel_size: 1e-4,
            dist: 50.0,
            energy: 20.0,
            alpha: 1e-3,
        },
        &cpu,
    )
    .unwrap();

    let scale = golden.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let mut max_abs = 0.0f32;
    for (&g, &p) in golden.iter().zip(tomo.array.iter()) {
        max_abs = max_abs.max((g - p).abs());
    }
    let max_rel = max_abs / scale;
    // Same single-precision FFT/filter on both sides — agreement is at the f32
    // round-off floor (measured max relative ≈ 2.4e-7).
    assert!(
        max_rel <= 1e-5,
        "Paganin parity: max|Δ| = {max_abs}, max relative = {max_rel}"
    );
}
