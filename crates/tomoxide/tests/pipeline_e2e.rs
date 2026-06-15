//! M3 end-to-end pipeline integration test: HDF5 in → preprocess → center →
//! FBP → TIFF out.
//!
//! Unlike the per-stage parity tests (each gated against a tomopy golden), this
//! wires every M3 stage together on one realistic DXchange acquisition and
//! checks the *whole chain* end to end with no projector golden:
//!
//!   open_dxchange  → raw intensity [nproj, nz, nx], flat=1000, dark=10, θ in deg
//!   normalize      → (data-dark)/(flat-dark) recovers transmission, then −log
//!   remove_stripe  → Titarenko (near-identity on this stripe-free sinogram)
//!   find_center_vo → Vo's Fourier rotation axis  (tomopy-default params → 63.5)
//!   fbp            → reconstruct the slice at the found center
//!   create_writer  → per-slice float32 TIFF, read back bit-exact (Δ=0)
//!
//! The fixture (`tools/gen_dxchange_pipeline_fixture.py`) turns the committed
//! `sino.npy` back into a Beer-Lambert acquisition, rescaled into a realistic
//! attenuation range. FBP is linear and `pearson_disk` is amplitude-scale
//! invariant, so recovering the rescaled sinogram recovers the SAME phantom the
//! unscaled `tomopy_parity` FBP test recovers — the end-to-end gate is that the
//! reconstruction still correlates with the phantom after the full chain.

use std::fs::File;
use std::io::BufReader;

use ndarray::{Array2, Axis};
use ndarray_npy::read_npy;
use tiff::decoder::{Decoder, DecodingResult};
use tomoxide::{
    io, prep, recon, Algorithm, Angles, Center, CpuBackend, Geometry, Layout, ReconParams,
    StripeMethod,
};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// tomopy's grid is a vertical flip of tomoxide's (grid↔detector handedness).
/// Same fixed re-indexing the `tomopy_parity` test uses.
fn row_flip(a: &Array2<f32>) -> Array2<f32> {
    let n = a.dim().0;
    Array2::from_shape_fn((n, n), |(iy, ix)| a[[n - 1 - iy, ix]])
}

/// Pearson correlation over a centered disk, amplitude-scale invariant.
fn pearson_disk(a: &Array2<f32>, b: &Array2<f32>, n: usize, radius_frac: f32) -> f32 {
    let c = (n as f32 - 1.0) / 2.0;
    let r2 = (radius_frac * n as f32 / 2.0).powi(2);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for iy in 0..n {
        for ix in 0..n {
            let (dy, dx) = (iy as f32 - c, ix as f32 - c);
            if dx * dx + dy * dy <= r2 {
                xs.push(a[[iy, ix]]);
                ys.push(b[[iy, ix]]);
            }
        }
    }
    let nn = xs.len() as f32;
    let mx = xs.iter().sum::<f32>() / nn;
    let my = ys.iter().sum::<f32>() / nn;
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let (dx, dy) = (x - mx, y - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Unique scratch directory for this test process (no tempfile dependency).
fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tomoxide_e2e_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn read_tiff_f32(path: &std::path::Path) -> ((u32, u32), Vec<f32>) {
    let mut dec = Decoder::new(BufReader::new(File::open(path).unwrap())).unwrap();
    let dims = dec.dimensions().unwrap();
    match dec.read_image().unwrap() {
        DecodingResult::F32(v) => (dims, v),
        other => panic!("expected F32 image, got {other:?}"),
    }
}

#[test]
fn m3_pipeline_hdf_to_tiff() {
    let cpu = CpuBackend::new();

    // 1. Read the raw DXchange acquisition (intensity data, flat=1000, dark=10,
    //    θ stored in degrees → reader converts to radians).
    let mut reader = io::open_dxchange(&format!("{FIXTURES}/pipeline_dxchange.h5")).unwrap();
    let mut ds = reader.read_all().unwrap();
    let theta = ds.theta.clone();
    let nx = ds.data.n_cols();
    assert_eq!(theta.len(), 180, "expected 180 projections");
    assert_eq!(nx, 128, "expected 128-wide detector");

    // 2. Flat-field correction + minus-log → the (rescaled) line-integral
    //    sinogram. Both flat and dark are present, so real (data−dark)/(flat−dark)
    //    division runs, not an identity.
    prep::normalize_dataset(&mut ds, &cpu).unwrap();
    assert!(
        ds.data.array.iter().all(|v| v.is_finite()),
        "normalize/minus_log produced non-finite values"
    );

    // 3. Stripe removal on the sinogram. The data is stripe-free, so Titarenko is
    //    near-identity here — it must not break the downstream center/recovery.
    let mut sino = ds.data.to_layout(Layout::Sinogram);
    prep::remove_stripe(
        &mut sino,
        StripeMethod::Ti {
            nblock: 0,
            beta: 1.5,
        },
    )
    .unwrap();

    // 4. Rotation-axis center via Vo's Fourier method. tomopy-default params give
    //    63.5 on this sinogram (see center_parity.rs case [0]); the full
    //    preprocess chain must not drift it off that optimum.
    let center =
        recon::center::find_center_vo(&sino, &cpu, None, -50.0, 50.0, 6.0, 0.25, 0.5, 20).unwrap();
    eprintln!("find_center_vo = {center:.3} (Vo golden 63.5)");
    assert!(
        (center - 63.5).abs() <= 0.5,
        "pipeline center {center} drifted from Vo's 63.5"
    );

    // 5. FBP at the found center.
    let mut geom = Geometry::parallel(Angles(theta), nx, 1, 1.0);
    geom.center = Center::Scalar(center);
    let params = ReconParams {
        num_gridx: Some(nx),
        ..Default::default()
    };
    let vol = recon::recon(&sino, &geom, Algorithm::Fbp, &params, &cpu).unwrap();

    // 6. Recover the phantom (scale-invariant Pearson over the centered disk).
    let phantom: Array2<f32> = read_npy(format!("{FIXTURES}/phantom.npy")).unwrap();
    let slice = vol.array.index_axis(Axis(0), 0).to_owned();
    let r = pearson_disk(&row_flip(&slice), &phantom, nx, 0.85);
    eprintln!("end-to-end FBP recovery vs phantom: Pearson r = {r:.4}");
    assert!(
        r >= 0.8,
        "end-to-end pipeline failed to recover the phantom: r = {r:.4}"
    );

    // 7. Persist the reconstruction to per-slice float32 TIFF and read it back.
    //    The writer applies no numeric transform, so the round-trip is bit-exact.
    let dir = scratch("hdf_to_tiff");
    let prefix = dir.join("recon");
    let mut w = io::create_writer(prefix.to_str().unwrap(), io::SaveFormat::Tiff).unwrap();
    w.write_chunk(&vol, 0, 1).unwrap();

    let ((wpx, hpx), buf) = read_tiff_f32(&dir.join("recon_00000.tiff"));
    assert_eq!((wpx, hpx), (nx as u32, nx as u32), "TIFF dimensions");
    let src: Vec<f32> = slice.iter().copied().collect();
    assert_eq!(buf, src, "TIFF round-trip not bit-exact");
    assert!(
        buf.iter().all(|v| v.is_finite()),
        "TIFF holds non-finite pixels"
    );

    std::fs::remove_dir_all(&dir).ok();
}
