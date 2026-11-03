//! `cuda::lamino_reproject` gives the frames a laminography reconstruction at
//! a given centre reads a band of its volume from: it is the existing
//! projector, a band projects where it sits in the volume, the
//! centre moves the frames along the detector, and Fourierrec at that centre
//! puts what was projected back in place, sharpest at that centre.
//!
//! Own test binary (touches CUDA device state) per the suite convention.
#![cfg(feature = "cuda")]

use ndarray::{s, Array3, Axis};
use tomoxide::{
    cuda, recon, sim, Algorithm, Angles, Beam, Center, CudaBackend, Detector, Geometry, Layout,
    ReconParams, Tomo, Volume,
};

/// Gaussian blobs `(z, y, x, sigma)` in a `[depth, n, n]` volume.
fn blobs(depth: usize, n: usize, at: &[(f32, f32, f32, f32)]) -> Array3<f32> {
    Array3::from_shape_fn((depth, n, n), |(z, y, x)| {
        at.iter()
            .map(|&(bz, by, bx, sg)| {
                let d2 =
                    (z as f32 - bz).powi(2) + (y as f32 - by).powi(2) + (x as f32 - bx).powi(2);
                (-d2 / (2.0 * sg * sg)).exp()
            })
            .sum()
    })
}

fn geometry(nproj: usize, tilt_deg: f32, center: f32, n: usize, rows: usize) -> Geometry {
    Geometry {
        angles: Angles::uniform(nproj, 0.0, 2.0 * std::f32::consts::PI),
        center: Center::Scalar(center),
        beam: Beam::laminography(tilt_deg),
        detector: Detector {
            width: n,
            height: rows,
            pixel_size: 1.0,
        },
    }
}

fn max_rel_diff(a: &Array3<f32>, b: &Array3<f32>) -> f32 {
    let peak = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        / peak
}

fn pearson(a: &Array3<f32>, b: &Array3<f32>) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.sum() as f64 / n, b.sum() as f64 / n);
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64 - ma, y as f64 - mb);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt()
}

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            None
        }
    }
}

/// At a centre off `n/2` the whole volume reprojects as `sim::project`
/// projects it, and two bands at their own `z0` add up to the whole.
#[test]
fn cuda_lamino_reproject_is_the_projector_and_bands_add_up() {
    let Some(cudab) = cuda_or_skip() else { return };
    let (n, nproj, rows, depth, tilt) = (64usize, 60usize, 40usize, 48usize, 20.0f32);
    let vol = blobs(
        depth,
        n,
        &[
            (14.0, 30.0, 40.0, 2.5),
            (30.0, 22.0, 28.0, 3.0),
            (36.0, 40.0, 30.0, 2.0),
        ],
    );
    let center = n as f32 / 2.0 + 3.5;
    let geom = geometry(nproj, tilt, center, n, rows);
    let theta = geom.angles.0.clone();
    let whole = cuda::lamino_reproject(vol.view(), 0, depth, &theta, tilt, center, rows).unwrap();
    assert_eq!(whole.dim(), (nproj, rows, n));

    let sino = sim::project(&Volume::new(vol.clone()), &geom, &cudab).unwrap();
    let projected = sino.to_layout(Layout::Projection).array;
    let d = max_rel_diff(&whole, &projected);
    assert!(
        d < 1e-5,
        "reprojection differs from sim::project by {d:e} of peak"
    );

    let z = 20;
    let low = cuda::lamino_reproject(
        vol.slice(s![..z, .., ..]),
        0,
        depth,
        &theta,
        tilt,
        center,
        rows,
    );
    let high = cuda::lamino_reproject(
        vol.slice(s![z.., .., ..]),
        z,
        depth,
        &theta,
        tilt,
        center,
        rows,
    );
    let sum = low.unwrap() + high.unwrap();
    let d = max_rel_diff(&sum, &whole);
    assert!(
        d < 1e-5,
        "bands at their z0 sum to the whole within {d:e} of peak"
    );
}

/// One `LaminoReprojector` projecting angle chunks gives the frames one call
/// over every angle gives, and refuses an empty chunk.
#[test]
fn cuda_lamino_reprojector_chunks_are_the_whole() {
    if cuda_or_skip().is_none() {
        return;
    }
    let (n, nproj, rows, depth, tilt, center) =
        (64usize, 50usize, 40usize, 48usize, 20.0f32, 33.5f32);
    let vol = blobs(
        depth,
        n,
        &[(20.0, 26.0, 38.0, 2.5), (28.0, 36.0, 24.0, 3.0)],
    );
    let band = vol.slice(s![10..40, .., ..]);
    let theta = Angles::uniform(nproj, 0.0, 2.0 * std::f32::consts::PI).0;
    let whole = cuda::lamino_reproject(band, 10, depth, &theta, tilt, center, rows).unwrap();
    let rp = cuda::LaminoReprojector::new(band, 10, depth, tilt, center, rows).unwrap();
    let parts: Vec<_> = theta.chunks(16).map(|t| rp.project(t).unwrap()).collect();
    let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
    // The kernel scatters with atomicAdd, so two launches agree to rounding.
    let d = max_rel_diff(&ndarray::concatenate(Axis(0), &views).unwrap(), &whole);
    assert!(d < 1e-5, "chunks differ from the whole by {d:e} of peak");
    assert!(rp.project(&[]).is_err());
}

/// A centre `k` columns right of `n/2` is the same frames moved `k` columns
/// right.
#[test]
fn cuda_lamino_reproject_centre_moves_the_frames_along_the_detector() {
    if cuda_or_skip().is_none() {
        return;
    }
    let (n, nproj, rows, depth, tilt) = (64usize, 36usize, 40usize, 48usize, 20.0f32);
    let vol = blobs(depth, n, &[(22.0, 28.0, 36.0, 3.0)]);
    let theta = Angles::uniform(nproj, 0.0, 2.0 * std::f32::consts::PI).0;
    let at = |c: f32| cuda::lamino_reproject(vol.view(), 0, depth, &theta, tilt, c, rows).unwrap();
    let (base, moved) = (at(32.0), at(35.0));
    let peak = base.iter().fold(0.0f32, |m, v| m.max(*v));
    for ((k, r, j), &v) in moved.indexed_iter() {
        if (4..n - 4).contains(&j) {
            let want = base[[k, r, j - 3]];
            assert!(
                (v - want).abs() < 1e-5 * peak,
                "[{k},{r},{j}] {v} want {want}"
            );
        }
    }
}

/// A voxel that projects left of column 0 or above row 0 adds nothing. `(int)`
/// truncates toward zero, so such a sample once landed on column (row) 0 with
/// weight 1.5 and on column (row) 1 with −0.5.
#[test]
fn cuda_lamino_reproject_drops_a_sample_left_of_column_0_or_above_row_0() {
    if cuda_or_skip().is_none() {
        return;
    }
    let (n, rows, depth) = (32usize, 24usize, 32usize);
    // At θ = 0 voxel (z, n/2, x) lands on column x − n/2 + center and row
    // cos(tilt)·(z − depth/2) + rows/2.
    let one = |z: usize, x: usize, tilt: f32, center: f32| {
        let mut vol = Array3::zeros((depth, n, n));
        vol[[z, n / 2, x]] = 1.0f32;
        cuda::lamino_reproject(vol.view(), 0, depth, &[0.0], tilt, center, rows).unwrap()
    };
    let left = one(depth / 2, 0, 20.0, n as f32 / 2.0 - 0.5);
    assert!(
        left.iter().all(|&v| v == 0.0),
        "column −0.5 scattered {:?}",
        left.slice(s![0, rows / 2, ..2])
    );
    // cos(tilt) = 12.5/16 puts z = 0 on row −0.5.
    let tilt = (12.5f32 / 16.0).acos().to_degrees();
    let above = one(0, n / 2, tilt, n as f32 / 2.0);
    assert!(
        above.iter().all(|&v| v == 0.0),
        "row −0.5 scattered {:?}",
        above.slice(s![0, ..2, n / 2])
    );
}

/// Every frame's centroid is where the blob's centroid projects: column
/// `cosθ·(x−n/2) + sinθ·(y−n/2) + center`, row
/// `sinθ·cosφ·(x−n/2) − cosθ·cosφ·(y−n/2) + sinφ·(z−depth/2) + rows/2`.
/// Bilinear splatting keeps first moments, so this holds to rounding at a
/// fractional centre.
#[test]
fn cuda_lamino_reproject_puts_a_blob_where_the_geometry_says() {
    if cuda_or_skip().is_none() {
        return;
    }
    let (n, nproj, rows, depth, tilt, center) =
        (64usize, 24usize, 40usize, 48usize, 20.0f32, 34.5f32);
    let band = blobs(12, n, &[(5.0, 26.0, 38.0, 2.0)]);
    let z0 = 18;
    let (mut m, mut mz, mut my, mut mx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for ((z, y, x), &v) in band.indexed_iter() {
        let v = v as f64;
        m += v;
        mz += v * (z + z0) as f64;
        my += v * y as f64;
        mx += v * x as f64;
    }
    let (bz, by, bx) = (mz / m, my / m, mx / m);
    let theta = Angles::uniform(nproj, 0.0, 2.0 * std::f32::consts::PI).0;
    let frames =
        cuda::lamino_reproject(band.view(), z0, depth, &theta, tilt, center, rows).unwrap();
    let phi = (90.0 + tilt as f64).to_radians();
    let h = (n / 2) as f64;
    for (k, frame) in frames.axis_iter(Axis(0)).enumerate() {
        let (st, ct) = (theta[k] as f64).sin_cos();
        let u = ct * (bx - h) + st * (by - h) + center as f64;
        let v = st * phi.cos() * (bx - h) - ct * phi.cos() * (by - h)
            + phi.sin() * (bz - (depth / 2) as f64)
            + (rows / 2) as f64;
        let (mut s0, mut sr, mut sc) = (0.0f64, 0.0f64, 0.0f64);
        for ((r, c), &p) in frame.indexed_iter() {
            s0 += p as f64;
            sr += p as f64 * r as f64;
            sc += p as f64 * c as f64;
        }
        let (gr, gc) = (sr / s0, sc / s0);
        assert!(
            (gc - u).abs() < 0.02 && (gr - v).abs() < 0.02,
            "angle {k}: centroid (row {gr:.3}, col {gc:.3}), geometry says ({v:.3}, {u:.3})"
        );
    }
}

/// Fourierrec at the centre the frames were reprojected at reconstructs the
/// band where it was, and agrees with it best at that centre: the sub-pixel
/// conventions of the two meet.
#[test]
fn cuda_fourierrec_inverts_lamino_reproject_at_its_centre() {
    let Some(cudab) = cuda_or_skip() else { return };
    let (n, nproj, rows, tilt, center) = (64usize, 360usize, 48usize, 20.0f32, 34.5f32);
    let depth = cuda::lamino_recon_height(rows, tilt);
    let (z0, nb) = (18usize, 14usize);
    let band = blobs(
        nb,
        n,
        &[
            (6.0, 26.0, 40.0, 2.0),
            (7.0, 36.0, 24.0, 2.0),
            (5.0, 40.0, 36.0, 1.5),
        ],
    );
    let theta = Angles::uniform(nproj, 0.0, 2.0 * std::f32::consts::PI).0;
    let frames =
        cuda::lamino_reproject(band.view(), z0, depth, &theta, tilt, center, rows).unwrap();
    let sino = Tomo::new(frames, Layout::Projection).to_layout(Layout::Sinogram);
    let params = ReconParams {
        lamino_rh: Some(depth),
        ..Default::default()
    };
    let offsets = [-1.0f32, -0.5, 0.0, 0.5, 1.0];
    let mut score = Vec::new();
    for &d in &offsets {
        let geom = geometry(nproj, tilt, center + d, n, rows);
        let v = recon::recon(&sino, &geom, Algorithm::Fourierrec, &params, &cudab).unwrap();
        let got = v.array.slice(s![z0..z0 + nb, .., ..]).to_owned();
        if d == 0.0 {
            // Each blob's centroid over its own neighbourhood, positive part.
            for &(bz, by, bx, _) in &[(6.0f32, 26.0f32, 40.0f32, 0.0f32), (7.0, 36.0, 24.0, 0.0)] {
                let (mut m, mut mz, mut my, mut mx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for ((z, y, x), &p) in got.indexed_iter() {
                    let near = (z as f32 - bz).abs() <= 3.0
                        && (y as f32 - by).abs() <= 4.0
                        && (x as f32 - bx).abs() <= 4.0;
                    if near && p > 0.0 {
                        let p = p as f64;
                        m += p;
                        mz += p * z as f64;
                        my += p * y as f64;
                        mx += p * x as f64;
                    }
                }
                let (gz, gy, gx) = (mz / m, my / m, mx / m);
                eprintln!("blob ({bz},{by},{bx}) reconstructed at ({gz:.3},{gy:.3},{gx:.3})");
                assert!(
                    (gz - bz as f64).abs() < 0.25
                        && (gy - by as f64).abs() < 0.25
                        && (gx - bx as f64).abs() < 0.25,
                    "blob ({bz},{by},{bx}) reconstructed at ({gz:.3},{gy:.3},{gx:.3})"
                );
            }
        }
        score.push(pearson(&got, &band));
    }
    eprintln!("correlation with the band at centre offsets {offsets:?}: {score:?}");
    // Parabola through the three around the best.
    let (a, b, c) = (score[1], score[2], score[3]);
    let vertex = 0.5 * 0.5 * (a - c) / (a - 2.0 * b + c);
    assert!(
        score[2] > score[0] && score[2] > score[4] && vertex.abs() < 0.2,
        "best agreement at offset {vertex:.3} px, scores {score:?}"
    );
}
