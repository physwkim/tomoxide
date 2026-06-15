//! Numeric parity against tomopy for the entropy center finder.
//!
//! `find_center` reconstructs a single slice with gridrec at candidate centers
//! and minimises the Shannon entropy of the masked reconstruction (Nelder-Mead),
//! exactly as tomopy does. Unlike `find_center_vo`/`find_center_pc` it goes
//! THROUGH the projector, so it inherits the linear-interp-vs-Siddon gridrec gap
//! (see PORTING): the entropy surface is a near-replica of tomopy's but not
//! bit-exact, and the result is the local basin Nelder-Mead reaches from `init`.
//! It is therefore checked two ways: it recovers the true rotation axis (the
//! projector-independent `find_center_vo`, which matches tomopy exactly) within
//! ±0.5 px, and it agrees with tomopy's own `find_center` within ±1 px.
//! Goldens from tomopy 1.15.3 (`tools/gen_tomopy_center_entropy_golden.py`):
//! base sinogram (true center 63.5, tomopy find_center 64.0) and an off-center
//! left-pad-8 variant (true 71.5, tomopy 71.4).

use ndarray::{Array1, Array3};
use ndarray_npy::read_npy;
use tomoxide::{recon, CpuBackend, Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load3(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

fn load1(name: &str) -> Array1<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

#[test]
fn find_center_entropy_matches_tomopy() {
    // Both fixtures are tomopy projection order (angle, row, col) with one row.
    let base = load3("sino.npy"); // (nang, 1, ncol)
    let pad = load3("center_fc_pad.npy"); // (nang, 1, ncol+8)
    let theta = load1("angles.npy");
    let golden = load1("tomopy_center_fc.npy"); // tomopy find_center [base, pad8]
    let truth = load1("center_fc_true.npy"); // tomopy find_center_vo (true axis)
    let theta = theta.to_vec();
    let cpu = CpuBackend::new();

    for (i, arr) in [base, pad].into_iter().enumerate() {
        let tomo = Tomo::new(arr, Layout::Projection);
        let got = recon::center::find_center(&tomo, &theta, &cpu, None, None, 0.5).unwrap();
        let (want, real) = (golden[i], truth[i]);
        // Recovery: the entropy minimum sits on the true rotation axis to ±0.5 px.
        assert!(
            (got - real).abs() <= 0.5,
            "case {i}: find_center = {got} misses true axis {real} by {}",
            (got - real).abs()
        );
        // Cross-impl: projector-coupled, so the Nelder-Mead basin agrees with
        // tomopy's find_center to ~1 px (tomopy's own result is ±0.5 px of truth).
        assert!(
            (got - want).abs() <= 1.0,
            "case {i}: find_center = {got}, tomopy = {want} (|Δ| = {})",
            (got - want).abs()
        );
    }
}
