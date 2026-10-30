//! `cuda::reconstruct_lamino_streaming` stops on its `cancel` token: before the
//! first tile when the token has already fired (Fourierrec emits no tile until
//! all three of its stages are done, so its stage loops are what stop it), and
//! at the next round when it fires mid-run.
//!
//! Some tests set process-global env vars; nextest runs each in its own process.

use ndarray::Array3;
use tomoxide::{
    cuda, Algorithm, Angles, Beam, CancelToken, Center, CudaBackend, Detector, Error, Geometry,
    Layout, ReconParams, Tomo,
};

const N: usize = 64;
const NPROJ: usize = 90;
const NZ: usize = 32;

fn geom() -> Geometry {
    Geometry {
        angles: Angles::uniform(NPROJ, 0.0, 2.0 * std::f32::consts::PI),
        center: Center::Scalar(N as f32 / 2.0),
        beam: Beam::laminography(20.0),
        detector: Detector {
            width: N,
            height: NZ,
            pixel_size: 1.0,
        },
    }
}

/// Tiles handed to `on_tile` and the result, with `token` fired by
/// `fire_after` tiles (`Some(0)`: before the call).
fn run(
    algorithm: Algorithm,
    fire_after: Option<usize>,
) -> (usize, tomoxide::Result<(usize, usize, usize)>) {
    let sino = Tomo::new(Array3::<f32>::ones((NPROJ, NZ, N)), Layout::Projection);
    let token = CancelToken::new();
    if fire_after == Some(0) {
        token.cancel();
    }
    let mut tiles = 0;
    let r = cuda::reconstruct_lamino_streaming(
        &sino,
        &geom(),
        algorithm,
        &ReconParams::default(),
        None,
        Some(&token),
        &mut |_, _| {
            tiles += 1;
            if fire_after == Some(tiles) {
                token.cancel();
            }
            Ok(())
        },
    );
    (tiles, r)
}

#[test]
fn a_fired_token_stops_linerec_before_its_first_tile() {
    if CudaBackend::new().is_err() {
        eprintln!("skipping CUDA test: no CUDA device");
        return;
    }
    let (tiles, r) = run(Algorithm::Linerec, Some(0));
    assert!(matches!(r, Err(Error::Cancelled)), "{r:?}");
    assert_eq!(tiles, 0);
}

#[test]
fn a_token_fired_mid_run_stops_linerec_at_the_next_round() {
    if CudaBackend::new().is_err() {
        eprintln!("skipping CUDA test: no CUDA device");
        return;
    }
    // A small budget splits the output into many rh tiles; one device makes
    // every round a single tile.
    std::env::set_var("TOMOXIDE_CUDA_MAX_FREE_BYTES", "500000");
    std::env::set_var("TOMOXIDE_CUDA_DEVICES", "0");
    let (all, r) = run(Algorithm::Linerec, None);
    r.unwrap();
    assert!(all > 2, "the budget left {all} tiles");
    let (tiles, r) = run(Algorithm::Linerec, Some(1));
    assert!(matches!(r, Err(Error::Cancelled)), "{r:?}");
    assert_eq!(tiles, 1);
}

#[test]
fn a_fired_token_stops_device_fourierrec_before_its_first_tile() {
    if CudaBackend::new().is_err() {
        eprintln!("skipping CUDA test: no CUDA device");
        return;
    }
    let (tiles, r) = run(Algorithm::Fourierrec, Some(0));
    assert!(matches!(r, Err(Error::Cancelled)), "{r:?}");
    assert_eq!(tiles, 0);
}

#[test]
fn a_fired_token_stops_host_fourierrec_before_its_first_tile() {
    if CudaBackend::new().is_err() {
        eprintln!("skipping CUDA test: no CUDA device");
        return;
    }
    std::env::set_var("TOMOXIDE_LAM_FOURIERREC_HOST", "1");
    let (tiles, r) = run(Algorithm::Fourierrec, Some(0));
    assert!(matches!(r, Err(Error::Cancelled)), "{r:?}");
    assert_eq!(tiles, 0);
}
