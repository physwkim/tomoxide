//! Beam-hardening parity against an xraylib-based reference.
//!
//! Golden from `tools/gen_beamhardening_golden.py` — a faithful translation of
//! the `beamhardening` package paths tomocupy drives, using **xraylib** (the
//! Rust port's cross-section source; `beamhardening` itself uses xraydb, which
//! has no Rust port). Since both sides call the same xraylib C library and the
//! rest of the algorithm (Simpson integration, thickness grid, `np.interp`,
//! flat-field angle finding) is reproduced exactly, the port matches to the f64
//! floor. The only residual is numpy's vectorised `exp`/`log` and pairwise
//! `np.sum` vs scalar libm, well under the tolerances below.
//!
//! Runs only with the `beam-hardening` feature (it needs xraylib).
#![cfg(feature = "beam-hardening")]

use ndarray::{Array1, Array2, Array3};
use ndarray_npy::read_npy;
use tomoxide::prep::hardening::{
    default_aps_bm_spectra, BeamCorrector, BeamHardeningConfig, Layer, Material,
};
use tomoxide::{Layout, Tomo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn f64v(name: &str) -> Vec<f64> {
    let a: Array1<f64> =
        read_npy(format!("{FIXTURES}/{name}")).unwrap_or_else(|e| panic!("{name}: {e}"));
    a.to_vec()
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / (y.abs().max(1e-12)))
        .fold(0.0f64, f64::max)
}

/// The fixed config the golden generator used (mirrored verbatim).
fn config() -> BeamHardeningConfig {
    BeamHardeningConfig {
        scintillator: Layer {
            material: Material {
                formula: "Lu3Al5O12".into(),
                density: 6.73,
            },
            thickness_um: 100.0,
        },
        sample: Material {
            formula: "Fe".into(),
            density: 7.87,
        },
        filters: vec![
            Layer {
                material: Material {
                    formula: "Al".into(),
                    density: 2.7,
                },
                thickness_um: 750.0,
            },
            Layer {
                material: Material {
                    formula: "Cu".into(),
                    density: 8.96,
                },
                thickness_um: 50.0,
            },
            Layer {
                material: Material {
                    formula: "Be".into(),
                    density: 1.85,
                },
                thickness_um: 250.0,
            },
        ],
        ref_trans: 0.1,
        threshold_trans: 1e-5,
        d_source_m: 36.0,
        pixel_size_um: 10.0,
    }
}

#[test]
fn luts_match_reference() {
    let corr = BeamCorrector::new(&config(), &default_aps_bm_spectra()).unwrap();
    let (ext, path) = corr.centerline_lut();
    let (ang, fac) = corr.angular_lut();
    // Measured residuals are at the f64 floor: ext ≈ 4e-12, path ≈ 2e-15,
    // angular factor ≈ 3e-14 (numpy pairwise-sum / vectorised exp vs scalar libm).
    assert!(
        max_rel(ext, &f64v("bh_centerline_ext.npy")) < 1e-10,
        "centerline ext"
    );
    assert!(
        max_rel(path, &f64v("bh_centerline_path.npy")) < 1e-12,
        "centerline path"
    );
    assert!(
        max_rel(ang, &f64v("bh_angular_angles.npy")) < 1e-12,
        "angular angles"
    );
    assert!(
        max_rel(fac, &f64v("bh_angular_corr.npy")) < 1e-11,
        "angular corr"
    );
}

#[test]
fn find_angles_match_reference() {
    let mut corr = BeamCorrector::new(&config(), &default_aps_bm_spectra()).unwrap();
    let flat: Array2<f32> = read_npy(format!("{FIXTURES}/bh_flat.npy")).unwrap();
    corr.find_angles(&flat);
    assert!(
        max_rel(corr.row_angles(), &f64v("bh_row_angles.npy")) < 1e-12,
        "row angles"
    );
}

#[test]
fn correction_matches_reference() {
    let mut corr = BeamCorrector::new(&config(), &default_aps_bm_spectra()).unwrap();
    let flat: Array2<f32> = read_npy(format!("{FIXTURES}/bh_flat.npy")).unwrap();
    corr.find_angles(&flat);

    let data_in: Array3<f32> = read_npy(format!("{FIXTURES}/bh_data_in.npy")).unwrap();
    let data_out: Array3<f32> = read_npy(format!("{FIXTURES}/bh_data_out.npy")).unwrap();
    let mut t = Tomo::new(data_in, Layout::Projection);
    corr.correct(&mut t, 28, 36).unwrap();

    let got: Vec<f64> = t.array.iter().map(|&v| v as f64).collect();
    let want: Vec<f64> = data_out.iter().map(|&v| v as f64).collect();
    // f32 output: residual ≈ 1.2e-7 (single-precision rounding floor).
    let rel = max_rel(&got, &want);
    assert!(rel < 1e-6, "beam-hardening correction max rel diff {rel}");
}
