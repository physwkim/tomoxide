//! Numeric parity against tomopy for the phase-correlation center finder.
//!
//! `find_center_pc` registers the 0°/mirrored-180° projection pair by subpixel
//! phase cross-correlation (a port of skimage `phase_cross_correlation`,
//! `normalization="phase"`, `upsample_factor = 1/tol`). It is pure Fourier-domain
//! image registration — no projector — so it is held to TRUE cross-implementation
//! parity. With `tol = 0.5` the recovered shift is quantized to half a pixel and
//! the center to a quarter pixel, and the whole-pixel + 3×3 upsampled-DFT argmax
//! are robust to f32 FFT round-off, so the port reproduces tomopy's center
//! exactly. Goldens from tomopy 1.15.3 `find_center_pc`
//! (`tools/gen_tomopy_center_pc_golden.py`); cases 2 and 4 land off the integer
//! grid (centers 77.25 / 78.25), exercising the subpixel refinement.

use ndarray::{Array1, Array3, Axis};
use ndarray_npy::read_npy;
use tomoxide::{recon, CpuBackend};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load3(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

fn load1(name: &str) -> Array1<f64> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

#[test]
fn find_center_pc_matches_tomopy() {
    let proj0 = load3("center_pc_proj0.npy"); // (ncase, nrow, ncol)
    let proj180 = load3("center_pc_proj180.npy");
    let tols = load1("center_pc_tols.npy");
    let centers = load1("center_pc_centers.npy");
    let cpu = CpuBackend::new();

    let ncase = proj0.dim().0;
    for i in 0..ncase {
        let p0 = proj0.index_axis(Axis(0), i).to_owned();
        let p180 = proj180.index_axis(Axis(0), i).to_owned();
        let tol = tols[i] as f32;
        let want = centers[i] as f32;
        let got = recon::center::find_center_pc(&p0, &p180, &cpu, tol, None).unwrap();
        // Quantized to a quarter pixel and argmax-robust → exact parity.
        assert!(
            (got - want).abs() <= 1e-4,
            "case {i}: find_center_pc = {got}, tomopy = {want} (|Δ| = {})",
            (got - want).abs()
        );
    }
}

#[test]
fn find_center_pc_rejects_rotc_guess() {
    // The `rotc_guess` pre-alignment (ndimage.shift) is not yet ported; the
    // function must reject it rather than silently ignore it.
    let proj0 = load3("center_pc_proj0.npy");
    let p0 = proj0.index_axis(Axis(0), 0).to_owned();
    let cpu = CpuBackend::new();
    let err = recon::center::find_center_pc(&p0, &p0, &cpu, 0.5, Some(80.0));
    assert!(
        err.is_err(),
        "rotc_guess=Some must be rejected, got {err:?}"
    );
}
