//! The CUDA projector edges `projector_edge_taps.rs` cannot reach through the
//! parallel-beam reference. Taps outside the detector are zero, so:
//!  - a tilted projection onto `nz` rows is the middle of the same projection
//!    onto `nz + 2` rows, and back-projecting `g` equals back-projecting `g`
//!    with a zero row added on each side;
//!  - the tilted kernels' column taps are the parallel kernels' (rows summed out);
//!  - an analytic reconstruction of identical rows has identical slices, the
//!    last one included, in f32 (gather kernel) and f16 (texture kernel).
//!
//! Needs a CUDA device (skipped otherwise).
#![cfg(feature = "cuda")]

use ndarray::{s, Array3, Axis};
use tomoxide::{
    recon, Algorithm, Angles, Beam, Center, CudaBackend, Detector, Dtype, FilteredBackproject,
    ForwardProject, Geometry, Layout, ReconParams, Tomo, Volume,
};

fn fill(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn cuda() -> Option<CudaBackend> {
    match CudaBackend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            None
        }
    }
}

fn lamino(ang: &[f32], n: usize, nz: usize) -> Geometry {
    Geometry {
        angles: Angles(ang.to_vec()),
        center: Center::Scalar(n as f32 / 2.0),
        beam: Beam::Laminography {
            phi: std::f32::consts::FRAC_PI_2 + 20f32.to_radians(),
        },
        detector: Detector {
            width: n,
            height: nz,
            pixel_size: 1.0,
        },
    }
}

fn angles(nproj: usize) -> Vec<f32> {
    (0..nproj).map(|t| 0.1 + t as f32 * 0.29).collect()
}

/// max |a − b| over max |b|.
fn rel_err(a: ndarray::ArrayView3<f32>, b: ndarray::ArrayView3<f32>) -> f32 {
    assert_eq!(a.dim(), b.dim());
    let scale = b.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let err = a
        .iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (&x, &y)| m.max((x - y).abs()));
    err / scale
}

#[test]
fn cuda_lamino_detector_edge_rows_are_zero_padded() {
    let Some(cuda) = cuda() else { return };
    let (n, nproj, nz, rh) = (16usize, 12usize, 16usize, 24usize);
    let ang = angles(nproj);
    // Detector centre nz/2 moves by one row with two more rows, so small-detector
    // row r is big-detector row r + 1.
    let (small, big) = (lamino(&ang, n, nz), lamino(&ang, n, nz + 2));

    let vol = Volume::new(Array3::from_shape_vec((rh, n, n), fill(3, rh * n * n)).unwrap());
    let mut gs = Tomo::new(Array3::zeros((nz, nproj, n)), Layout::Sinogram);
    let mut gb = Tomo::new(Array3::zeros((nz + 2, nproj, n)), Layout::Sinogram);
    cuda.project(&vol, &small, &mut gs).unwrap();
    cuda.project(&vol, &big, &mut gb).unwrap();
    let e = rel_err(gs.array.view(), gb.array.slice(s![1..nz + 1, .., ..]));
    eprintln!("forward: small vs cropped big, rel {e:.2e}");
    assert!(e < 1e-5, "forward edge rows differ: rel {e:.2e}");

    let g = Array3::from_shape_vec((nz, nproj, n), fill(4, nz * nproj * n)).unwrap();
    let mut gpad = Array3::zeros((nz + 2, nproj, n));
    gpad.slice_mut(s![1..nz + 1, .., ..]).assign(&g);
    let mut fs = Volume::new(Array3::zeros((rh, n, n)));
    let mut fb = Volume::new(Array3::zeros((rh, n, n)));
    cuda.backproject(&Tomo::new(g, Layout::Sinogram), &small, &mut fs)
        .unwrap();
    cuda.backproject(&Tomo::new(gpad, Layout::Sinogram), &big, &mut fb)
        .unwrap();
    let e = rel_err(fs.array.view(), fb.array.view());
    eprintln!("back: g vs zero-padded g, rel {e:.2e}");
    assert!(e < 1e-5, "back-projection edge rows differ: rel {e:.2e}");
}

#[test]
fn cuda_lamino_columns_match_parallel() {
    let Some(cuda) = cuda() else { return };
    let (n, nproj, nz) = (16usize, 12usize, 32usize);
    // θ = 0 and π/2 put voxels exactly on the last column; the rest put the
    // grid corners in (−1, 0) and (n − 1, n).
    let mut ang = angles(nproj - 2);
    ang.extend([0.0, std::f32::consts::FRAC_PI_2]);
    let geom = lamino(&ang, n, nz);
    let par = Geometry::parallel(Angles(ang.clone()), n, 1, 1.0);
    // Slices [10, 22] land wholly inside the 32 detector rows at this tilt, so
    // each voxel's row weights sum to 1.
    let (z0, z1) = (10usize, 23usize);

    let mut vol = Array3::<f32>::zeros((nz, n, n));
    let body = (z1 - z0) * n * n;
    vol.slice_mut(s![z0..z1, .., ..])
        .assign(&Array3::from_shape_vec((z1 - z0, n, n), fill(5, body)).unwrap());
    let mut g = Tomo::new(Array3::zeros((nz, nproj, n)), Layout::Sinogram);
    cuda.project(&Volume::new(vol.clone()), &geom, &mut g)
        .unwrap();
    let summed = vol.sum_axis(Axis(0)).insert_axis(Axis(0));
    let mut gp = Tomo::new(Array3::zeros((1, nproj, n)), Layout::Sinogram);
    cuda.project(&Volume::new(summed), &par, &mut gp).unwrap();
    let rows = g.array.sum_axis(Axis(0)).insert_axis(Axis(0));
    let e = rel_err(rows.view(), gp.array.view());
    eprintln!("forward: rows summed vs parallel, rel {e:.2e}");
    assert!(
        e < 1e-5,
        "tilted forward columns differ from parallel: rel {e:.2e}"
    );

    let h = Array3::from_shape_vec((1, nproj, n), fill(6, nproj * n)).unwrap();
    let every_row = h.broadcast((nz, nproj, n)).unwrap().to_owned();
    let mut f = Volume::new(Array3::zeros((nz, n, n)));
    cuda.backproject(&Tomo::new(every_row, Layout::Sinogram), &geom, &mut f)
        .unwrap();
    let mut fp = Volume::new(Array3::zeros((1, n, n)));
    cuda.backproject(&Tomo::new(h, Layout::Sinogram), &par, &mut fp)
        .unwrap();
    let want = fp.array.broadcast((z1 - z0, n, n)).unwrap();
    let e = rel_err(f.array.slice(s![z0..z1, .., ..]), want);
    eprintln!("back: every row = parallel row, rel {e:.2e}");
    assert!(
        e < 1e-5,
        "tilted back columns differ from parallel: rel {e:.2e}"
    );
}

#[test]
fn cuda_fbp_keeps_the_last_slice() {
    let Some(cuda) = cuda() else { return };
    // More than 256 rows in every chunk, however many GPUs split the stack.
    let (n, nproj, nz) = (16usize, 16usize, 1040usize);
    let geom = Geometry::parallel(
        Angles::uniform(nproj, 0.0, std::f32::consts::PI),
        n,
        nz,
        1.0,
    );
    let h = Array3::from_shape_vec((1, nproj, n), fill(8, nproj * n)).unwrap();
    let sino = Tomo::new(
        h.broadcast((nz, nproj, n)).unwrap().to_owned(),
        Layout::Sinogram,
    );
    for (dtype, tol) in [(Dtype::F32, 1e-5f32), (Dtype::F16, 1e-2)] {
        let params = ReconParams {
            num_gridx: Some(n),
            dtype,
            ..Default::default()
        };
        let vol = recon::recon(&sino, &geom, Algorithm::Fbp, &params, &cuda).unwrap();
        let first = vol.array.slice(s![0..1, .., ..]);
        let want = first.broadcast((nz, n, n)).unwrap();
        let e = rel_err(vol.array.view(), want);
        eprintln!("{dtype:?}: every slice vs slice 0, rel {e:.2e}");
        assert!(
            e < tol,
            "{dtype:?}: slices differ from slice 0: rel {e:.2e}"
        );
    }
}
