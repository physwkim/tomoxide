//! Bit-exact parity against tomopy for `inpainter_morph` (misc/corr.py:996,
//! C `libtomo/misc/inpainter.c` `Inpainter_morph_main`).
//!
//! Morphological inpainter over a boolean mask. Only the deterministic modes are
//! tested for bit-parity: `InpaintingType::Mean` (Gaussian-distance-weighted
//! mean) and `InpaintingType::Median` (median order statistic). The Gaussian
//! weights use `exp`/`powf` (which match macOS libm bit-for-bit) and the
//! fixed-order f32 multiply-accumulate is fused (`mul_add`) to match libtomo's
//! FMA-contracted build, so the cube matches tomopy **bit-for-bit (Δ=0)** —
//! including the 3-D median's zero-padding quirk (one case below
//! deliberately leaves a masked voxel at exactly `0.0`, the C buffer-sort
//! artefact, which the port reproduces). `InpaintingType::Random` is excluded:
//! upstream's C `rand()` under OpenMP is not reproducible run-to-run, so it has
//! no golden reference (it is covered structurally by the unit tests instead).
//!
//! Golden from the real tomopy `tools/gen_tomopy_inpainter_morph_golden.py`:
//! mean/median × {3-D `axis=None`, 2-D-per-slice `axis=0`} × `iterations` 0/1/2.

use ndarray::{Array2, Array3, Array4, Axis};
use ndarray_npy::read_npy;
use tomoxide::prep::filters::{inpainter_morph, InpaintingType};
use tomoxide_core::data::{Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

#[test]
fn inpainter_morph_matches_tomopy() {
    let inputs: Array4<f32> = read_npy(format!("{FIXTURES}/inpainter_morph_input.npy")).unwrap();
    let masks: Array4<u8> = read_npy(format!("{FIXTURES}/inpainter_morph_mask.npy")).unwrap();
    let outputs: Array4<f32> = read_npy(format!("{FIXTURES}/inpainter_morph_output.npy")).unwrap();
    let params: Array2<f64> = read_npy(format!("{FIXTURES}/inpainter_morph_params.npy")).unwrap();

    let ncase = inputs.dim().0;
    for k in 0..ncase {
        let size = params[[k, 0]] as usize;
        let iterations = params[[k, 1]] as usize;
        let inpainting = match params[[k, 2]] as i64 {
            0 => InpaintingType::Mean,
            1 => InpaintingType::Median,
            other => panic!("case {k}: unexpected type code {other}"),
        };
        let axis_code = params[[k, 3]];
        let axis = if axis_code < 0.0 {
            None
        } else {
            Some(axis_code as usize)
        };

        let mask: Array3<bool> = masks.index_axis(Axis(0), k).mapv(|v| v != 0);
        let mut tomo = Tomo::new(inputs.index_axis(Axis(0), k).to_owned(), Layout::Projection);
        inpainter_morph(&mut tomo, &mask, size, iterations, inpainting, axis).unwrap();

        let want = outputs.index_axis(Axis(0), k);
        let mut mismatch = 0usize;
        for (g, w) in tomo.array.iter().zip(want.iter()) {
            if g.to_bits() != w.to_bits() {
                mismatch += 1;
            }
        }
        assert_eq!(
            mismatch, 0,
            "case {k} (size={size}, iter={iterations}, {inpainting:?}, axis={axis:?}): \
             {mismatch} f32 bit-mismatches vs tomopy"
        );
    }
}
