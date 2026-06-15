//! Numeric parity against tomopy for Titarenko stripe removal (`remove_stripe_ti`).
//!
//! The Titarenko method (Miqueles 2014) solves a finite-difference
//! normal-equations system per slice by conjugate gradient in f64 and combines
//! the first/second-difference corrected sinograms as `sqrt(d1·d2 + β·|min|)`,
//! rounding each `_ring` to f32. tomoxide reimplements the same f64 CG + f32
//! cast in the upstream operation order, so it is held to the f32 round-off
//! floor (projector-independent), not bit-exactness. Golden from tomopy 1.15.3
//! `remove_stripe_ti` (`tools/gen_tomopy_stripe_ti_golden.py`, nblock=0,
//! alpha=1.5).
//!
//! Only the default `nblock=0` path is covered: tomopy's block path `_ringb`
//! (nblock>0) is unrunnable on modern numpy (its NaN guard
//! `np.where(np.isnan(...) is True)` raises), so no reference output exists —
//! tomoxide returns `NotImplemented` for nblock>0.

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::{prep, Layout, StripeMethod, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load(name: &str) -> Array3<f32> {
    read_npy(format!("{FIXTURES}/{name}")).unwrap()
}

#[test]
fn remove_stripe_ti_matches_tomopy() {
    let input = load("stripe_ti_input.npy");
    let golden = load("tomopy_stripe_ti_nblock0.npy");

    let mut tomo = Tomo::new(input, Layout::Projection);
    prep::stripe::remove_stripe(
        &mut tomo,
        StripeMethod::Ti {
            nblock: 0,
            beta: 1.5,
        },
    )
    .unwrap();
    let got = tomo.array;

    // Agreement with the tomopy golden, to the f32 round-off floor. The f64 CG
    // converges to a 1e-7 residual and each `_ring` casts to f32, so the f32
    // quantization dominates any CG/BLAS summation-order difference.
    let scale = golden.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_abs = golden
        .iter()
        .zip(got.iter())
        .fold(0.0f32, |m, (&g, &p)| m.max((g - p).abs()));
    let max_rel = max_abs / scale;
    eprintln!("Ti nblock=0: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})");
    assert!(
        max_rel <= 1e-5,
        "Ti golden parity: max|Δ| = {max_abs}, max relative = {max_rel} (scale {scale})"
    );
}

#[test]
fn remove_stripe_ti_block_path_is_unsupported() {
    // tomopy's `_ringb` cannot run on modern numpy, so the block path has no
    // verifiable reference; tomoxide rejects it rather than guessing.
    let input = load("stripe_ti_input.npy");
    let mut tomo = Tomo::new(input, Layout::Projection);
    let err = prep::stripe::remove_stripe(
        &mut tomo,
        StripeMethod::Ti {
            nblock: 3,
            beta: 1.5,
        },
    );
    assert!(err.is_err(), "nblock>0 must be rejected, got {err:?}");
}
