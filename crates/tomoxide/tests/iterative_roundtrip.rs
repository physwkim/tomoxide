//! End-to-end CPU iterative round-trips (SIRT, MLEM, OSEM).
//!
//! Forward-project a Shepp-Logan phantom, reconstruct it, and assert the result
//! correlates strongly with the phantom. SIRT additionally must drive the data
//! residual down monotonically (convergence on consistent data); MLEM and OSEM
//! must preserve non-negativity. OSEM with a single block must equal MLEM (the
//! ordered-subset generalization collapses to plain EM at `num_block = 1`).

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

/// Interleaved ordered-subset angle order: `[0, B, 2B, …, 1, 1+B, …]`, so each
/// contiguous block of `nang/B` angles is angularly distributed (good subsets).
fn interleaved_ind_block(nang: usize, num_block: usize) -> Vec<i32> {
    let mut ind = Vec::with_capacity(nang);
    for s in 0..num_block {
        let mut a = s;
        while a < nang {
            ind.push(a as i32);
            a += num_block;
        }
    }
    ind
}

#[test]
fn osem_reconstructs_nonnegative_phantom() {
    let n = 96;
    let nang = 150;
    let num_block = 10;
    let cpu = CpuBackend::new();

    // OSEM, like MLEM, is multiplicative/positivity-preserving — clamp the
    // phantom's negative ellipses so the object (and sinogram) stay non-negative.
    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    // 10 blocks → 10 sub-updates per outer iteration, so 18 iters ≈ 180 EM
    // sub-updates: OSEM reaches MLEM-quality in far fewer outer iterations.
    let params = ReconParams {
        num_gridx: Some(n),
        num_iter: 18,
        num_block,
        ind_block: interleaved_ind_block(nang, num_block),
        ..Default::default()
    };
    let rec = recon::recon(&sino, &geom, Algorithm::Osem, &params, &cpu).unwrap();
    let slice = rec.array.index_axis(Axis(0), 0).to_owned();

    assert!(slice.iter().all(|&v| v >= -1e-6), "OSEM produced negatives");
    let corr = pearson_disk(&slice, &phantom, n, 0.85);
    eprintln!("OSEM (18 iters × {num_block} blocks) Pearson correlation = {corr:.4}");
    assert!(
        corr > 0.9,
        "OSEM correlates poorly with phantom: r = {corr:.4}"
    );
}

#[test]
fn osem_with_one_block_equals_mlem() {
    // Boundary invariant: a single ordered subset is the full angle set, so
    // OSEM(num_block=1) performs exactly MLEM's update each iteration.
    let n = 64;
    let nang = 90;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let vol = Volume::new(phantom.insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let base = ReconParams {
        num_gridx: Some(n),
        num_iter: 15,
        ..Default::default()
    };
    let mlem = recon::recon(&sino, &geom, Algorithm::Mlem, &base, &cpu).unwrap();
    let osem = recon::recon(
        &sino,
        &geom,
        Algorithm::Osem,
        &ReconParams {
            num_block: 1,
            ..base
        },
        &cpu,
    )
    .unwrap();

    let max_abs = mlem
        .array
        .iter()
        .zip(osem.array.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max |MLEM − OSEM(1 block)| = {max_abs:e}");
    assert!(
        max_abs < 1e-4,
        "OSEM(num_block=1) diverges from MLEM: max abs diff = {max_abs:e}"
    );
}
