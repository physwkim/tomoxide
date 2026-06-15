//! End-to-end CPU iterative round-trips (SIRT, MLEM).
//!
//! Forward-project a Shepp-Logan phantom, reconstruct it, and assert the result
//! correlates strongly with the phantom. SIRT additionally must drive the data
//! residual down monotonically (convergence on consistent data); MLEM must
//! preserve non-negativity.

use ndarray::{Array2, Axis};
use tomoxide::{recon, sim, Algorithm, Angles, CpuBackend, Geometry, ReconParams, Volume};

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

/// Sum of squared sinogram residual ‖b − A·recon‖² (forward-project the
/// reconstruction and compare to the measured sinogram).
fn residual_norm(
    recon: &Volume<f32>,
    b: &tomoxide::Tomo<f32>,
    geom: &Geometry,
    cpu: &CpuBackend,
) -> f32 {
    let ax = sim::project(recon, geom, cpu).unwrap();
    ax.array
        .iter()
        .zip(b.array.iter())
        .map(|(&a, &m)| (m - a).powi(2))
        .sum()
}

#[test]
fn sirt_reconstructs_and_converges() {
    let n = 96;
    let nang = 150;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let p = |iters| ReconParams {
        num_gridx: Some(n),
        num_iter: iters,
        ..Default::default()
    };

    // Residual must shrink as iterations grow (SIRT convergence).
    let r10 = residual_norm(
        &recon::recon(&sino, &geom, Algorithm::Sirt, &p(10), &cpu).unwrap(),
        &sino,
        &geom,
        &cpu,
    );
    let rec = recon::recon(&sino, &geom, Algorithm::Sirt, &p(120), &cpu).unwrap();
    let r120 = residual_norm(&rec, &sino, &geom, &cpu);
    eprintln!("SIRT residual: 10 iters = {r10:.3}, 120 iters = {r120:.3}");
    assert!(r120 < r10, "residual did not decrease: {r10} -> {r120}");

    let slice = rec.array.index_axis(Axis(0), 0).to_owned();
    let corr = pearson_disk(&slice, &phantom, n, 0.85);
    eprintln!("SIRT (120 iters) Pearson correlation = {corr:.4}");
    assert!(
        corr > 0.9,
        "SIRT correlates poorly with phantom: r = {corr:.4}"
    );
}

#[test]
fn mlem_reconstructs_nonnegative_phantom() {
    let n = 96;
    let nang = 150;
    let cpu = CpuBackend::new();

    // MLEM is multiplicative/positivity-preserving and needs a non-negative
    // object (hence sinogram), so clamp the phantom's negative ellipses to 0.
    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        num_iter: 120,
        ..Default::default()
    };
    let rec = recon::recon(&sino, &geom, Algorithm::Mlem, &params, &cpu).unwrap();
    let slice = rec.array.index_axis(Axis(0), 0).to_owned();

    // Positivity is preserved by construction.
    assert!(slice.iter().all(|&v| v >= -1e-6), "MLEM produced negatives");
    let corr = pearson_disk(&slice, &phantom, n, 0.85);
    eprintln!("MLEM (120 iters) Pearson correlation = {corr:.4}");
    assert!(
        corr > 0.9,
        "MLEM correlates poorly with phantom: r = {corr:.4}"
    );
}
