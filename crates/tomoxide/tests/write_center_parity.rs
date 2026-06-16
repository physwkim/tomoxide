//! `write_center` — reconstruct one slice across a range of rotation centers
//! (tomopy `recon/rotation.py:438`).
//!
//! Parity scope. tomoxide's gridrec is a gridrec-*family* method (Kaiser–Bessel
//! kernel + ramp weight, not tomopy's PSWF + `parzen`), so the per-center
//! reconstruction *pixels* are not bit-identical to tomopy and are not compared
//! here. The model-independent, portable part is the **center enumeration**
//! (`np.arange`), which is held to tomopy exactly (Δ = 0) against a numpy golden
//! (`tools/gen_tomopy_write_center_golden.py`). The orchestration — slice
//! selection (`ind`), one reconstruction per center, and the circular mask — is
//! checked by self-consistency against an independent `recon(Gridrec)` call and
//! by the mask geometry.

use ndarray::{Array1, Array3, Axis};
use ndarray_npy::read_npy;
use tomoxide::{
    recon, Algorithm, Angles, Beam, Center, CpuBackend, Detector, Geometry, Layout, ReconParams,
    Tomo,
};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// Deterministic non-degenerate sinogram `[nrows, nang, ncol]` (sinogram layout).
fn make_sino(nrows: usize, nang: usize, ncol: usize) -> Array3<f32> {
    let mut arr = Array3::<f32>::zeros((nrows, nang, ncol));
    for r in 0..nrows {
        for a in 0..nang {
            for c in 0..ncol {
                let x = (c as f32 - ncol as f32 / 2.0) / 8.0;
                arr[[r, a, c]] =
                    x.cos() * (1.0 + 0.3 * (a as f32 * 0.1).sin()) + 0.5 * (r as f32 + 1.0);
            }
        }
    }
    arr
}

/// Independent gridrec of one sinogram row at `center` — replicates the geometry
/// `write_center` uses internally (parallel beam, unit pixel, grid = detector
/// width) so the two must agree bit-for-bit.
fn recon_row_at(sino: &Array3<f32>, theta: &[f32], ind: usize, center: f32) -> Array3<f32> {
    let (_nrows, nang, ncol) = sino.dim();
    let slc = Tomo::new(
        sino.index_axis(Axis(0), ind)
            .to_owned()
            .insert_axis(Axis(0)),
        Layout::Sinogram,
    );
    let geom = Geometry {
        angles: Angles(theta.to_vec()),
        center: Center::Scalar(center),
        beam: Beam::Parallel,
        detector: Detector {
            width: ncol,
            height: 1,
            pixel_size: 1.0,
        },
    };
    let _ = nang;
    let cpu = CpuBackend::new();
    recon::recon(
        &slc,
        &geom,
        Algorithm::Gridrec,
        &ReconParams::default(),
        &cpu,
    )
    .unwrap()
    .array
}

#[test]
fn write_center_enumeration_matches_numpy_default() {
    let (nrows, nang, ncol) = (3, 64, 64);
    let theta: Vec<f32> = (0..nang)
        .map(|i| i as f32 * std::f32::consts::PI / nang as f32)
        .collect();
    let tomo = Tomo::new(make_sino(nrows, nang, ncol), Layout::Sinogram);
    let cpu = CpuBackend::new();

    let (centers, stack) =
        recon::center::write_center(&tomo, &theta, &cpu, None, None, false, 1.0).unwrap();

    let golden: Array1<f32> =
        read_npy(format!("{FIXTURES}/write_center_centers_default.npy")).unwrap();
    assert_eq!(
        centers.len(),
        golden.len(),
        "default center count: got {}, numpy {}",
        centers.len(),
        golden.len()
    );
    for (i, (&c, &g)) in centers.iter().zip(golden.iter()).enumerate() {
        assert_eq!(c, g, "default center[{i}]: got {c}, numpy {g}");
    }
    // Stack is [len(centers), ncol, ncol].
    assert_eq!(stack.dim(), (centers.len(), ncol, ncol));
}

#[test]
fn write_center_enumeration_matches_numpy_explicit_range() {
    let (nrows, nang, ncol) = (3, 64, 64);
    let theta: Vec<f32> = (0..nang)
        .map(|i| i as f32 * std::f32::consts::PI / nang as f32)
        .collect();
    let tomo = Tomo::new(make_sino(nrows, nang, ncol), Layout::Sinogram);
    let cpu = CpuBackend::new();

    let (centers, _stack) = recon::center::write_center(
        &tomo,
        &theta,
        &cpu,
        Some((28.0, 36.0, 0.5)),
        None,
        false,
        1.0,
    )
    .unwrap();

    let golden: Array1<f32> =
        read_npy(format!("{FIXTURES}/write_center_centers_range.npy")).unwrap();
    assert_eq!(centers.len(), golden.len(), "explicit-range center count");
    for (i, (&c, &g)) in centers.iter().zip(golden.iter()).enumerate() {
        assert_eq!(c, g, "range center[{i}]: got {c}, numpy {g}");
    }
}

#[test]
fn write_center_stack_is_per_center_gridrec_of_the_indexed_slice() {
    let (nrows, nang, ncol) = (3, 64, 64);
    let theta: Vec<f32> = (0..nang)
        .map(|i| i as f32 * std::f32::consts::PI / nang as f32)
        .collect();
    let sino = make_sino(nrows, nang, ncol);
    let tomo = Tomo::new(sino.clone(), Layout::Sinogram);
    let cpu = CpuBackend::new();

    // ind defaults to nrows/2; reconstruct each center independently and compare.
    let ind = nrows / 2;
    let (centers, stack) =
        recon::center::write_center(&tomo, &theta, &cpu, None, None, false, 1.0).unwrap();

    let mut max_abs = 0.0f32;
    for (m, &c) in centers.iter().enumerate() {
        let indep = recon_row_at(&sino, &theta, ind, c);
        for (&a, &b) in stack
            .index_axis(Axis(0), m)
            .iter()
            .zip(indep.index_axis(Axis(0), 0).iter())
        {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    assert_eq!(
        max_abs, 0.0,
        "write_center stack must equal an independent per-center gridrec of slice {ind} (Δ={max_abs})"
    );
}

#[test]
fn write_center_mask_zeroes_corners_and_preserves_interior() {
    let (nrows, nang, ncol) = (3, 64, 64);
    let theta: Vec<f32> = (0..nang)
        .map(|i| i as f32 * std::f32::consts::PI / nang as f32)
        .collect();
    let tomo = Tomo::new(make_sino(nrows, nang, ncol), Layout::Sinogram);
    let cpu = CpuBackend::new();

    let (_c0, plain) =
        recon::center::write_center(&tomo, &theta, &cpu, None, None, false, 1.0).unwrap();
    let (_c1, masked) =
        recon::center::write_center(&tomo, &theta, &cpu, None, None, true, 1.0).unwrap();
    assert_eq!(plain.dim(), masked.dim());

    let nc = plain.dim().0;
    for m in 0..nc {
        // Corner (0,0) is outside the inscribed disk (ratio=1) → zeroed.
        assert_eq!(masked[[m, 0, 0]], 0.0, "corner not masked at slice {m}");
        // Center pixel is inside the disk → identical to the unmasked recon.
        let cc = ncol / 2;
        assert_eq!(
            masked[[m, cc, cc]],
            plain[[m, cc, cc]],
            "interior changed by mask at slice {m}"
        );
    }
    // The mask must actually remove energy somewhere (corners are nonzero unmasked
    // for this sinogram), so masked ≠ plain overall.
    let changed = plain
        .iter()
        .zip(masked.iter())
        .any(|(&a, &b)| (a - b).abs() > 0.0);
    assert!(changed, "mask had no effect");
}
