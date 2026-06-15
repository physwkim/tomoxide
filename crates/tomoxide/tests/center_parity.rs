//! Numeric parity against tomopy for Nghia Vo's center finder.
//!
//! `find_center_vo` is a sinogram-domain Fourier method — it never touches a
//! projector — so, unlike `fbp`, it is held to TRUE cross-implementation parity.
//! Golden centers come from tomopy 1.15.3 `find_center_vo`
//! (`tools/gen_tomopy_center_golden.py`), computed on the SAME sinograms this
//! test feeds the port. Case order is fixed by the generator:
//!   [0] base sino,       default params      -> 63.5
//!   [1] base sino,       ratio=0.7, drop=10  -> 63.5
//!   [2] base sino,       smin=-30, smax=30   -> 63.5
//!   [3] left-pad-8 sino, default params      -> 71.5  (axis tracks the pad)

use ndarray::{Array1, Array3};
use ndarray_npy::read_npy;
use tomoxide::{recon, CpuBackend, Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load_tomo(name: &str) -> Tomo<f32> {
    let a: Array3<f32> = read_npy(format!("{FIXTURES}/{name}")).unwrap(); // (nang, 1, ncol)
    Tomo::new(a, Layout::Projection)
}

#[test]
fn find_center_vo_matches_tomopy() {
    let golden: Array1<f32> = read_npy(format!("{FIXTURES}/tomopy_center_vo.npy")).unwrap();
    let base = load_tomo("sino.npy");
    let pad = load_tomo("center_sino_pad.npy");
    let cpu = CpuBackend::new();

    // tomopy default fine radius/step (srad=6, step=0.25) for every case; the
    // coarse range, mask ratio, and drop vary per case.
    let run = |tomo: &Tomo<f32>, smin, smax, ratio, drop| {
        recon::center::find_center_vo(tomo, &cpu, None, smin, smax, 6.0, 0.25, ratio, drop).unwrap()
    };
    let got = [
        run(&base, -50.0, 50.0, 0.5, 20),
        run(&base, -50.0, 50.0, 0.7, 10),
        run(&base, -30.0, 30.0, 0.5, 20),
        run(&pad, -50.0, 50.0, 0.5, 20),
    ];

    for (i, &got) in got.iter().enumerate() {
        let want = golden[i];
        // The metric optima sit exactly on the 0.5-pixel coarse grid here, so
        // the port reproduces tomopy's center exactly (Δ = 0 on all four cases);
        // ≤0.25 (one fine step) leaves headroom for cubic-spline subpixel drift.
        assert!(
            (got - want).abs() <= 0.25,
            "case {i}: find_center_vo = {got}, tomopy = {want} (|Δ| = {})",
            (got - want).abs()
        );
    }
}
