//! Numeric parity against tomocupy for Farago phase retrieval.
//!
//! Farago (2024) is the same single-step padded-Fourier retrieval as Paganin but
//! with the filter `1/(cos θ + db·sin θ)`, `θ = π·λ·dist·(ix² + iy²)` over the
//! squared reciprocal grid (tomocupy `_reciprocal_grid` + `_farago_filter_factor`).
//! It is projector-independent (a Fourier filter on each padded radiograph), so
//! it is held to the f32 round-off floor.
//!
//! The filter is evaluated in f32 to mirror cupy: `db ≈ 1e3` multiplies `sin θ`,
//! amplifying any rounding in `θ` ~1e3×, and the squared reciprocal grid must be
//! built from the *exact* f32 reciprocal coordinate (numpy/cupy round the
//! `0.5/((n−1)·ps)` scale to f32 before the multiply) — an f64 grid cast down
//! diverges ~1e-3.
//!
//! The reference is tomocupy `retrieve_phase.farago_filter`, which runs on the
//! GPU; the golden is a faithful CPU/numpy transcription of tomocupy's exact
//! functions (`tools/gen_tomocupy_farago_golden.py`) since `cupy.fft` and
//! `scipy.fft` implement the same single-precision DFT. Default params
//! (pixel_size=1e-4, dist=50, energy=20, db=1000).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, CpuBackend, Layout, PhaseMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn farago_matches_tomocupy() {
    let input: Array3<f32> = read_npy(format!("{FIXTURES}/farago_input.npy")).unwrap();
    let golden: Array3<f32> = read_npy(format!("{FIXTURES}/tomocupy_farago.npy")).unwrap();

    let mut tomo = Tomo::new(input, Layout::Projection);
    let cpu = CpuBackend::new();
    prep::retrieve_phase(
        &mut tomo,
        PhaseMethod::Farago {
            pixel_size: 1e-4,
            dist: 50.0,
            energy: 20.0,
            db: 1000.0,
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
    eprintln!("Farago: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})");
    // The filter is bit-identical to numpy's (exact f32 reciprocal grid); the
    // residual is the single-precision FFT difference (scipy complex64 golden vs
    // the Rust f32 FFT, and cupy-vs-scipy DFT), which sits below ~1e-5 relative.
    assert!(
        max_rel <= 1e-5,
        "Farago parity: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})"
    );
}
