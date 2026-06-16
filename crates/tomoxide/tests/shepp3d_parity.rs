//! Bit-exact parity against tomopy for `shepp3d` (misc/phantom.py:284).
//!
//! The modified 3-D Shepp-Logan phantom: 10 ellipsoids rasterised on
//! `np.mgrid[-1:1:size·j]`. The inclusion test `Σ((R·r − c)/s)² ≤ 1` is computed
//! in f64 with libm `sin`/`cos` (bit-exact vs numpy's scalar trig) and a fixed
//! accumulation order, and the amplitudes accumulate in f32 like numpy's
//! `obj[mask] += A`, so the cube matches tomopy **bit-for-bit (Δ=0)**. The one
//! unreproduced step is numpy's BLAS dot order in `tensordot` (≤1 ULP), which
//! never flips a voxel — no sample lands within f64-ULP of the boundary.
//!
//! Golden from the real tomopy `tools/gen_tomopy_shepp3d_golden.py` at sizes 16
//! (even), 17 (odd), and 32.

use ndarray::Array3;
use ndarray_npy::read_npy;
use tomoxide::sim::shepp3d;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn shepp3d_matches_tomopy() {
    for size in [16usize, 17, 32] {
        let want: Array3<f32> = read_npy(format!("{FIXTURES}/shepp3d_{size}.npy")).unwrap();
        let got = shepp3d(size).unwrap();
        assert_eq!(got.array.dim(), want.dim(), "size {size}: shape");

        let mut mismatch = 0usize;
        for (g, w) in got.array.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                mismatch += 1;
            }
        }
        assert_eq!(mismatch, 0, "size {size}: {mismatch} f32 bit-mismatches vs tomopy");
    }
}
