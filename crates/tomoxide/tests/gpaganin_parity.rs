//! Numeric parity against tomocupy for generalized-Paganin (`GPaganin`) phase retrieval.
//!
//! Generalized Paganin (Paganin et al. 2020) is the same single-step Fourier
//! retrieval as standard Paganin but with a `cos`-based reciprocal grid
//! `kf = cos(ix·2π·ps) + cos(iy·2π·ps)` and the filter
//! `1/(1 − (2·aph/W²)·(kf − 2))`, `aph = db·dist·λ/(4π)`. It is
//! projector-independent (a Fourier low-pass on each padded radiograph), so it is
//! held to the f32 round-off floor.
//!
//! The reference is tomocupy `retrieve_phase.paganin_filter(method='Gpaganin')`,
//! which runs on the GPU; the golden is a faithful CPU/numpy transcription of
//! tomocupy's exact functions (`tools/gen_tomocupy_gpaganin_golden.py`) since
//! `cupy.fft` and `numpy.fft` implement the same DFT. Default params
//! (pixel_size=1e-4, dist=50, energy=20, db=1000, W=2e-4).

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, CpuBackend, Layout, PhaseMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn gpaganin_matches_tomocupy() {
    let input: Array3<f32> = read_npy(format!("{FIXTURES}/gpaganin_input.npy")).unwrap();
    let golden: Array3<f32> = read_npy(format!("{FIXTURES}/tomocupy_gpaganin.npy")).unwrap();

    let mut tomo = Tomo::new(input, Layout::Projection);
    let cpu = CpuBackend::new();
    prep::retrieve_phase(
        &mut tomo,
        PhaseMethod::GPaganin {
            pixel_size: 1e-4,
            dist: 50.0,
            energy: 20.0,
            db: 1000.0,
            w: 2e-4,
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
    eprintln!("GPaganin: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})");
    // Same single-precision FFT/filter on both sides — agreement is at the f32
    // round-off floor (the numpy golden's complex128 FFT vs the Rust f32 FFT and
    // the cupy-vs-numpy DFT difference all sit below ~1e-5 relative).
    assert!(
        max_rel <= 1e-5,
        "GPaganin parity: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})"
    );
}
