//! End-to-end CPU FBP round-trip (M1 milestone gate).
//!
//! Forward-project a Shepp-Logan phantom, reconstruct it with parallel-beam
//! FBP (ramp filter), and assert the reconstruction matches the phantom. The
//! metric is the Pearson correlation over a central disk, which is invariant to
//! the absolute amplitude scale (exact tomopy numeric parity needs golden data
//! and is deferred). This is the real proof that the FFT filter and the
//! forward/back-projector adjoint pair are correct together.

use ndarray::{Array2, Axis};
use tomoxide::{recon, sim, Algorithm, Angles, CpuBackend, Geometry, ReconParams, Volume};

/// Pearson correlation between two slices over a centered disk of the given
/// radius fraction (kept inside the phantom support, away from clipped corners).
fn pearson_disk(a: &Array2<f32>, b: &Array2<f32>, n: usize, radius_frac: f32) -> f32 {
    let c = (n as f32 - 1.0) / 2.0;
    let r = radius_frac * (n as f32 / 2.0);
    let r2 = r * r;
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for iy in 0..n {
        for ix in 0..n {
            let dy = iy as f32 - c;
            let dx = ix as f32 - c;
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

#[test]
fn fbp_reconstructs_shepp_logan_phantom() {
    let n = 128;
    let nang = 180;
    let cpu = CpuBackend::new();

    // Phantom as a single-slice volume [1, n, n].
    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));

    // Parallel-beam geometry: detector width == grid width, 0..π.
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);

    // Forward project, then FBP reconstruct.
    let sino = sim::project(&vol, &geom, &cpu).unwrap();
    assert_eq!(sino.array.dim(), (1, nang, n));

    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };
    let recon = recon::recon(&sino, &geom, Algorithm::Fbp, &params, &cpu).unwrap();
    assert_eq!(recon.array.dim(), (1, n, n));

    let slice = recon.array.index_axis(Axis(0), 0).to_owned();
    let corr = pearson_disk(&slice, &phantom, n, 0.85);
    eprintln!("FBP round-trip Pearson correlation = {corr:.4}");
    assert!(
        corr > 0.9,
        "FBP reconstruction correlates poorly with phantom: r = {corr:.4}"
    );
}

#[test]
fn gridrec_reconstructs_shepp_logan_phantom() {
    let n = 128;
    let nang = 180;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };
    let recon = recon::recon(&sino, &geom, Algorithm::Gridrec, &params, &cpu).unwrap();
    assert_eq!(recon.array.dim(), (1, n, n));

    let slice = recon.array.index_axis(Axis(0), 0).to_owned();
    let corr = pearson_disk(&slice, &phantom, n, 0.85);
    eprintln!("gridrec round-trip Pearson correlation = {corr:.4}");
    assert!(
        corr > 0.9,
        "gridrec reconstruction correlates poorly with phantom: r = {corr:.4}"
    );
}
