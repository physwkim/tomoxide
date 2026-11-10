//! `cuda::LaminoBandFocus` scores a band of the linerec reconstruction of
//! moved frames without leaving the device: its profile is `focus_by_z` of
//! that band as `move_frames` and `recon::recon` make it, chunked or not, its
//! gradient is the profile mean's derivative by every shift and the centre,
//! and `band` is that band.
//!
//! Own test binary (touches CUDA device state and sets
//! `TOMOXIDE_CUDA_MAX_FREE_BYTES`) per the suite convention.
#![cfg(feature = "cuda")]

use ndarray::{s, Array3};
use tomoxide::prep::alignment::move_frames;
use tomoxide::recon::center::focus_by_z;
use tomoxide::{
    cuda, prep, recon, sim, Algorithm, Angles, Beam, Center, CpuBackend, CudaBackend, Detector,
    FilterName, Geometry, Layout, ReconParams, Tomo, Volume,
};

const N: usize = 64;
const NPROJ: usize = 61;
const ROWS: usize = 40;
const TILT: f32 = 20.0;
const FILTER: FilterName = FilterName::Shepp;

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            None
        }
    }
}

fn geometry(center: f32, depth_rows: usize) -> Geometry {
    Geometry {
        angles: Angles::uniform(NPROJ, 0.0, 2.0 * std::f32::consts::PI),
        center: Center::Scalar(center),
        beam: Beam::laminography(TILT),
        detector: Detector {
            width: N,
            height: depth_rows,
            pixel_size: 1.0,
        },
    }
}

/// `[NPROJ, ROWS, N]` `−log` frames of Gaussian blobs, their angles, and the
/// reconstruction depth.
fn frames(cudab: &CudaBackend) -> (Array3<f32>, Vec<f32>, usize) {
    let depth = cuda::lamino_recon_height(ROWS, TILT);
    let at = [
        (0.3, 30.0, 40.0, 2.5),
        (0.5, 22.0, 28.0, 3.0),
        (0.7, 40.0, 30.0, 2.0),
    ];
    let vol = Array3::from_shape_fn((depth, N, N), |(z, y, x)| {
        at.iter()
            .map(|&(fz, by, bx, sg): &(f32, f32, f32, f32)| {
                let d2 = (z as f32 - fz * depth as f32).powi(2)
                    + (y as f32 - by).powi(2)
                    + (x as f32 - bx).powi(2);
                0.1 * (-d2 / (2.0 * sg * sg)).exp()
            })
            .sum()
    });
    let geom = geometry(N as f32 / 2.0 + 3.5, ROWS);
    let sino = sim::project(&Volume::new(vol), &geom, cudab).unwrap();
    (
        sino.to_layout(Layout::Projection).array,
        geom.angles.0.clone(),
        depth,
    )
}

/// Smooth, nonzero per-frame shifts.
fn shifts(theta: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let dy = theta.iter().map(|t| 0.7 * (3.0 * t).sin() + 0.2).collect();
    let dx = theta.iter().map(|t| 1.1 * (2.0 * t).cos() - 0.4).collect();
    (dy, dx)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    let peak = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        / peak
}

/// The profile is `focus_by_z` of the band of the linerec reconstruction of
/// the frames as `move_frames` moves them, with the angles in one chunk or in
/// many of two lengths; the chunked gradient is the single-chunk one.
#[test]
fn band_focus_profile_is_the_moved_linerec_band() {
    let Some(cudab) = cuda_or_skip() else { return };
    let (proj, theta, depth) = frames(&cudab);
    let (dy, dx) = shifts(&theta);
    let center = N as f32 / 2.0 + 3.8;
    let z = (depth / 4, 3 * depth / 4);

    let mut moved = Tomo::new(proj.mapv(|p| (-p).exp()), Layout::Projection);
    move_frames(&mut moved, &dy, &dx, 0.0).unwrap();
    prep::minus_log(&mut moved, &CpuBackend::new()).unwrap();
    let params = ReconParams {
        lamino_rh: Some(depth),
        filter_name: FILTER,
        ..Default::default()
    };
    let rec = recon::recon(
        &moved.to_layout(Layout::Sinogram),
        &geometry(center, ROWS),
        Algorithm::Linerec,
        &params,
        &cudab,
    )
    .unwrap();
    let want = focus_by_z(rec.array.slice(s![z.0..=z.1, .., ..]));

    let mut one = cuda::LaminoBandFocus::new(proj.view(), &theta, FILTER, TILT, z).unwrap();
    let got = one.profile(&dy, &dx, center).unwrap();
    let band = one.band(&dy, &dx, center).unwrap();
    assert_eq!(band.dim(), (z.1 - z.0 + 1, N, N));
    let d = max_rel(&focus_by_z(band.view()), &got);
    assert!(
        d < 1e-6,
        "the band's focus differs from the profile by {d:e}"
    );
    let peak = rec.array.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = (&band - &rec.array.slice(s![z.0..=z.1, .., ..]))
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()))
        / peak;
    eprintln!("band vs recon band: {d:.3e} of peak");
    assert!(
        d < 1e-5,
        "the band differs from the reconstruction's by {d:e} of peak"
    );
    assert_eq!(got.len(), z.1 - z.0 + 1);
    let d = max_rel(&got, &want);
    eprintln!("single chunk vs recon band: {d:.3e} of peak");
    assert!(d < 1e-4, "single-chunk profile differs by {d:e} of peak");

    // 61 angles in chunks of at most two: 30 of two and one of one.
    std::env::set_var("TOMOXIDE_CUDA_MAX_FREE_BYTES", "500000");
    let many = cuda::LaminoBandFocus::new(proj.view(), &theta, FILTER, TILT, z);
    std::env::remove_var("TOMOXIDE_CUDA_MAX_FREE_BYTES");
    let mut many = many.unwrap();
    let got = many.profile(&dy, &dx, center).unwrap();
    let d = max_rel(&got, &want);
    eprintln!("chunked vs recon band: {d:.3e} of peak");
    assert!(d < 1e-4, "chunked profile differs by {d:e} of peak");

    let g1 = one.gradient(&dy, &dx, center).unwrap();
    let gm = many.gradient(&dy, &dx, center).unwrap();
    let d = max_rel(&got, &gm.profile);
    assert!(d < 1e-6, "gradient's profile differs by {d:e} of peak");
    let d = max_rel(gm.shift.as_slice().unwrap(), g1.shift.as_slice().unwrap());
    eprintln!("chunked vs single-chunk shift gradient: {d:.3e} of peak");
    assert!(d < 1e-4, "chunked shift gradient differs by {d:e} of peak");
    let d = (gm.center - g1.center).abs() / g1.center.abs();
    assert!(d < 1e-4, "chunked centre gradient differs by {d:e}");
}

/// The gradient is the profile mean's central difference in a few frames'
/// shifts and in the centre.
#[test]
fn band_focus_gradient_matches_finite_differences() {
    let Some(cudab) = cuda_or_skip() else { return };
    let (proj, theta, depth) = frames(&cudab);
    let (dy, dx) = shifts(&theta);
    let center = N as f32 / 2.0 + 3.8;
    let mut bf = cuda::LaminoBandFocus::new(
        proj.view(),
        &theta,
        FILTER,
        TILT,
        (depth / 4, 3 * depth / 4),
    )
    .unwrap();
    let g = bf.gradient(&dy, &dx, center).unwrap();
    let mut score = |dy: &[f32], dx: &[f32], c: f32| mean(&bf.profile(dy, dx, c).unwrap());
    let h = 0.05f32;

    // (analytic, central difference) per checked parameter.
    let mut pairs = Vec::new();
    for t in [0, NPROJ / 3, 2 * NPROJ / 3, NPROJ - 1] {
        for (k, base) in [(0, &dx), (1, &dy)] {
            let (mut lo, mut hi) = (base.clone(), base.clone());
            lo[t] -= h;
            hi[t] += h;
            let step = (hi[t] - lo[t]) as f64;
            let fd = if k == 0 {
                (score(&dy, &hi, center) - score(&dy, &lo, center)) / step
            } else {
                (score(&hi, &dx, center) - score(&lo, &dx, center)) / step
            };
            pairs.push((
                format!("frame {t} {}", ["dx", "dy"][k]),
                g.shift[[t, k]],
                fd,
            ));
        }
    }
    let step = ((center + h) - (center - h)) as f64;
    let fd = (score(&dy, &dx, center + h) - score(&dy, &dx, center - h)) / step;
    pairs.push(("centre".into(), g.center, fd));

    let peak = pairs.iter().fold(0.0f64, |m, (_, a, _)| m.max(a.abs()));
    assert!(peak > 0.0, "gradient is zero");
    for (name, a, fd) in &pairs {
        eprintln!("{name}: analytic {a:.6e}, difference {fd:.6e}");
        assert!(
            (a - fd).abs() <= 0.02 * a.abs() + 1e-3 * peak,
            "{name}: analytic {a:e} vs difference {fd:e}"
        );
    }
}

#[test]
fn band_focus_refuses_bad_shapes() {
    let Some(cudab) = cuda_or_skip() else { return };
    let (proj, theta, depth) = frames(&cudab);
    assert!(cuda::LaminoBandFocus::new(proj.view(), &theta, FILTER, TILT, (0, depth)).is_err());
    assert!(cuda::LaminoBandFocus::new(proj.view(), &theta[1..], FILTER, TILT, (0, 4)).is_err());
    let mut bf = cuda::LaminoBandFocus::new(proj.view(), &theta, FILTER, TILT, (0, 4)).unwrap();
    let (dy, dx) = shifts(&theta);
    assert!(bf.profile(&dy[1..], &dx, N as f32 / 2.0).is_err());
    assert!(bf.gradient(&dy, &dx[1..], N as f32 / 2.0).is_err());
    assert!(bf.band(&dy[1..], &dx, N as f32 / 2.0).is_err());
}
