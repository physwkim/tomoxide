//! End-to-end CPU iterative round-trips for the full scalar family (SIRT, MLEM,
//! OSEM, PML/OSPML quad & hybrid, grad, tikh, tv, ART, BART).
//!
//! Forward-project a Shepp-Logan phantom, reconstruct it, and assert the result
//! correlates strongly with the phantom. The convergent/least-squares methods
//! (SIRT, grad, ART, BART) must drive the data residual down monotonically; MLEM
//! and OSEM must preserve non-negativity. Boundary invariants are bit-identical:
//! OSEM with one block equals MLEM, and the penalized-ML methods reduce to their
//! unpenalized counterparts at `reg_par = 0` (pml_quad → MLEM, ospml_quad → OSEM,
//! ospml_hybrid without δ → ospml_quad) — likewise tikh without a Tikhonov weight
//! equals grad. A positive `reg_par` smooths (quadratic prior, tv) or shrinks
//! energy (tikh ridge), and ordered subsets accelerate convergence (OSEM, BART).

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

/// Regression: multi-slice (`nz > 1`) OSEM / OSPML on the CPU must not error.
/// `build_subsets` gathers each subset with `select(Axis(1))`, whose owned result
/// is C-contiguous only for `nz == 1` and non-contiguous for `nz > 1`; the CPU
/// back-projector consumes it via `as_slice()` and errored ("non-contiguous
/// sinogram") on the multi-slice, multi-block path — invisible to the `nz == 1`
/// tests above. Both algorithms share `build_subsets`, so one guards the family.
#[test]
fn osem_ospml_multislice_cpu_runs() {
    let (n, nang, nz, num_block) = (48usize, 60usize, 3usize, 4usize);
    let cpu = CpuBackend::new();
    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let mut vol3 = ndarray::Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        vol3.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let vol = Volume::new(vol3);
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, nz, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    for (algo, reg_par) in [
        (Algorithm::Osem, vec![]),
        (Algorithm::OspmlQuad, vec![0.1f32]),
    ] {
        let params = ReconParams {
            num_gridx: Some(n),
            num_iter: 5,
            num_block,
            reg_par,
            ..Default::default()
        };
        let rec = recon::recon(&sino, &geom, algo, &params, &cpu)
            .unwrap_or_else(|e| panic!("{algo:?} multi-slice CPU errored: {e}"));
        assert_eq!(rec.dims(), (nz, n, n));
        assert!(
            rec.array.iter().all(|v| v.is_finite()),
            "{algo:?} produced non-finite values"
        );
    }
}

/// Warm-start correctness: splitting SIRT via `init` reproduces the continuous
/// run. SIRT(20) then SIRT(20) seeded with that output must equal SIRT(40) from
/// scratch — SIRT is deterministic on the CPU (no atomics), so the two are the
/// same arithmetic sequence and agree to the f32 floor. Locks that `init`
/// genuinely seeds the iterate rather than being ignored.
#[test]
fn warmstart_split_sirt_equals_continuous() {
    let (n, nang) = (64usize, 90usize);
    let cpu = CpuBackend::new();
    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let p = |num_iter, init: Option<Volume<f32>>| ReconParams {
        num_gridx: Some(n),
        num_iter,
        init,
        ..Default::default()
    };
    let full = recon::recon(&sino, &geom, Algorithm::Sirt, &p(40, None), &cpu).unwrap();
    let half = recon::recon(&sino, &geom, Algorithm::Sirt, &p(20, None), &cpu).unwrap();
    let chained = recon::recon(&sino, &geom, Algorithm::Sirt, &p(20, Some(half)), &cpu).unwrap();

    let (mut se, mut sref) = (0.0f64, 0.0f64);
    for (a, b) in chained.array.iter().zip(full.array.iter()) {
        se += (*a as f64 - *b as f64).powi(2);
        sref += (*b as f64).powi(2);
    }
    let nrmse = (se / full.array.len() as f64).sqrt() / (sref / full.array.len() as f64).sqrt();
    eprintln!("split-vs-continuous SIRT NRMSE = {nrmse:.2e}");
    assert!(
        nrmse < 1e-5,
        "warm-started split diverged from continuous: {nrmse:.2e}"
    );
}

/// Every iterative method honours `init` except the row-action pair (ART/BART);
/// supplying `init` there must error rather than silently drop the caller's seed.
#[test]
fn warmstart_rejected_for_unsupported() {
    let (n, nang) = (32usize, 60usize);
    let cpu = CpuBackend::new();
    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let vol = Volume::new(phantom.insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        num_iter: 3,
        reg_par: vec![1e-3],
        init: Some(vol.clone()),
        ..Default::default()
    };
    let err = recon::recon(&sino, &geom, Algorithm::Art, &params, &cpu);
    assert!(
        err.is_err(),
        "ART silently accepted an unsupported warm-start init"
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

/// Largest absolute element-wise difference between two volumes.
fn max_abs_diff(a: &Volume<f32>, b: &Volume<f32>) -> f32 {
    a.array
        .iter()
        .zip(b.array.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Contrast-normalized roughness over a centered disk: mean squared
/// cardinal-neighbor difference divided by the variance, so it is comparable
/// across reconstructions with different amplitude scales. Lower = smoother.
fn roughness_disk(a: &Array2<f32>, n: usize, radius_frac: f32) -> f32 {
    let c = (n as f32 - 1.0) / 2.0;
    let r2 = (radius_frac * n as f32 / 2.0).powi(2);
    let inside = |iy: usize, ix: usize| {
        let (dy, dx) = (iy as f32 - c, ix as f32 - c);
        dx * dx + dy * dy <= r2
    };
    let (mut vals, mut diff2, mut npair) = (Vec::new(), 0.0f32, 0.0f32);
    for iy in 0..n {
        for ix in 0..n {
            if !inside(iy, ix) {
                continue;
            }
            vals.push(a[[iy, ix]]);
            if iy + 1 < n && inside(iy + 1, ix) {
                diff2 += (a[[iy + 1, ix]] - a[[iy, ix]]).powi(2);
                npair += 1.0;
            }
            if ix + 1 < n && inside(iy, ix + 1) {
                diff2 += (a[[iy, ix + 1]] - a[[iy, ix]]).powi(2);
                npair += 1.0;
            }
        }
    }
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    (diff2 / npair) / var
}

#[test]
fn pml_quad_with_zero_reg_equals_mlem() {
    // pml_quad is ospml_quad with num_block=1; at reg=0 its quadratic update
    // degenerates to the linear MLEM step, so the two must be identical.
    let n = 64;
    let nang = 90;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let vol = Volume::new(phantom.insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        num_iter: 15,
        ..Default::default() // reg_par empty ⇒ 0
    };
    let mlem = recon::recon(&sino, &geom, Algorithm::Mlem, &params, &cpu).unwrap();
    let pml = recon::recon(&sino, &geom, Algorithm::PmlQuad, &params, &cpu).unwrap();

    let d = max_abs_diff(&mlem, &pml);
    eprintln!("max |MLEM − pml_quad(reg=0)| = {d:e}");
    assert_eq!(d, 0.0, "pml_quad(reg=0) is not identical to MLEM: {d:e}");
}

#[test]
fn ospml_quad_with_zero_reg_equals_osem() {
    // The ordered-subset penalized method at reg=0 must reproduce OSEM exactly.
    let n = 64;
    let nang = 90;
    let num_block = 6;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let vol = Volume::new(phantom.insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        num_iter: 12,
        num_block,
        ind_block: interleaved_ind_block(nang, num_block),
        ..Default::default() // reg_par empty ⇒ 0
    };
    let osem = recon::recon(&sino, &geom, Algorithm::Osem, &params, &cpu).unwrap();
    let ospml = recon::recon(&sino, &geom, Algorithm::OspmlQuad, &params, &cpu).unwrap();

    let d = max_abs_diff(&osem, &ospml);
    eprintln!("max |OSEM − ospml_quad(reg=0)| = {d:e}");
    assert_eq!(d, 0.0, "ospml_quad(reg=0) is not identical to OSEM: {d:e}");
}

#[test]
fn ospml_quad_regularization_reconstructs_and_smooths() {
    let n = 96;
    let nang = 150;
    let num_block = 10;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let common = ReconParams {
        num_gridx: Some(n),
        num_iter: 18,
        num_block,
        ind_block: interleaved_ind_block(nang, num_block),
        ..Default::default()
    };
    let unreg = recon::recon(&sino, &geom, Algorithm::OspmlQuad, &common, &cpu).unwrap();
    let reg = recon::recon(
        &sino,
        &geom,
        Algorithm::OspmlQuad,
        &ReconParams {
            reg_par: vec![0.1],
            ..common
        },
        &cpu,
    )
    .unwrap();

    let reg_slice = reg.array.index_axis(Axis(0), 0).to_owned();
    let unreg_slice = unreg.array.index_axis(Axis(0), 0).to_owned();

    assert!(
        reg_slice.iter().all(|&v| v >= -1e-6),
        "ospml_quad produced negatives"
    );
    let corr = pearson_disk(&reg_slice, &phantom, n, 0.85);
    let rough_reg = roughness_disk(&reg_slice, n, 0.85);
    let rough_unreg = roughness_disk(&unreg_slice, n, 0.85);
    eprintln!(
        "ospml_quad reg=0.1: r = {corr:.4}, roughness {rough_reg:.4} vs unreg {rough_unreg:.4}"
    );
    assert!(
        corr > 0.9,
        "ospml_quad (reg) correlates poorly with phantom: r = {corr:.4}"
    );
    assert!(
        rough_reg < rough_unreg,
        "quadratic penalty did not smooth: roughness {rough_reg:.4} >= unreg {rough_unreg:.4}"
    );
}

#[test]
fn pml_hybrid_with_zero_reg_equals_mlem() {
    // The hybrid prior also vanishes at reg=0, so pml_hybrid(reg=0) ≡ MLEM.
    let n = 64;
    let nang = 90;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let vol = Volume::new(phantom.insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        num_iter: 15,
        ..Default::default() // reg_par empty ⇒ 0
    };
    let mlem = recon::recon(&sino, &geom, Algorithm::Mlem, &params, &cpu).unwrap();
    let pml = recon::recon(&sino, &geom, Algorithm::PmlHybrid, &params, &cpu).unwrap();

    let d = max_abs_diff(&mlem, &pml);
    eprintln!("max |MLEM − pml_hybrid(reg=0)| = {d:e}");
    assert_eq!(d, 0.0, "pml_hybrid(reg=0) is not identical to MLEM: {d:e}");
}

#[test]
fn ospml_hybrid_without_delta_equals_ospml_quad() {
    // With no edge threshold (reg_par has only the strength), the hybrid edge
    // factor γ = 1, so the hybrid prior is exactly the quadratic prior.
    let n = 64;
    let nang = 90;
    let num_block = 6;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let vol = Volume::new(phantom.insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        num_iter: 12,
        num_block,
        ind_block: interleaved_ind_block(nang, num_block),
        reg_par: vec![0.3], // strength only, no delta
        ..Default::default()
    };
    let quad = recon::recon(&sino, &geom, Algorithm::OspmlQuad, &params, &cpu).unwrap();
    let hybrid = recon::recon(&sino, &geom, Algorithm::OspmlHybrid, &params, &cpu).unwrap();

    let d = max_abs_diff(&quad, &hybrid);
    eprintln!("max |ospml_quad − ospml_hybrid(no δ)| = {d:e}");
    assert_eq!(
        d, 0.0,
        "ospml_hybrid without delta differs from ospml_quad: {d:e}"
    );
}

#[test]
fn ospml_hybrid_preserves_edges_better_than_quad() {
    // At matched penalty strength, the edge-preserving hybrid prior should track
    // the sharp phantom better than the plain quadratic prior, which blurs edges.
    let n = 96;
    let nang = 150;
    let num_block = 10;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let common = ReconParams {
        num_gridx: Some(n),
        num_iter: 18,
        num_block,
        ind_block: interleaved_ind_block(nang, num_block),
        ..Default::default()
    };
    // Strong, blurring quadratic penalty (strength only).
    let quad = recon::recon(
        &sino,
        &geom,
        Algorithm::OspmlQuad,
        &ReconParams {
            reg_par: vec![0.3],
            ..common.clone()
        },
        &cpu,
    )
    .unwrap();
    // Same strength, but with a small edge threshold to preserve jumps.
    let hybrid = recon::recon(
        &sino,
        &geom,
        Algorithm::OspmlHybrid,
        &ReconParams {
            reg_par: vec![0.3, 0.05],
            ..common
        },
        &cpu,
    )
    .unwrap();

    let quad_s = quad.array.index_axis(Axis(0), 0).to_owned();
    let hybrid_s = hybrid.array.index_axis(Axis(0), 0).to_owned();
    assert!(
        hybrid_s.iter().all(|&v| v >= -1e-6),
        "ospml_hybrid produced negatives"
    );
    let corr_quad = pearson_disk(&quad_s, &phantom, n, 0.85);
    let corr_hybrid = pearson_disk(&hybrid_s, &phantom, n, 0.85);
    eprintln!("edge preservation: quad r = {corr_quad:.4}, hybrid r = {corr_hybrid:.4}");
    assert!(
        corr_hybrid > corr_quad,
        "hybrid did not preserve edges better than quad: {corr_hybrid:.4} <= {corr_quad:.4}"
    );
}

#[test]
fn grad_reconstructs_phantom_with_bb_step() {
    // Least-squares gradient descent with the Barzilai–Borwein self-tuning step
    // (reg_par[0] < 0). Unlike the EM family it imposes no positivity, so the
    // full phantom (with negative ellipses) is the right target. The data
    // residual must shrink as iterations grow, and the result correlate with the
    // phantom.
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
        reg_par: vec![-1.0], // Barzilai–Borwein adaptive step
        ..Default::default()
    };

    let r10 = residual_norm(
        &recon::recon(&sino, &geom, Algorithm::Grad, &p(10), &cpu).unwrap(),
        &sino,
        &geom,
        &cpu,
    );
    let rec = recon::recon(&sino, &geom, Algorithm::Grad, &p(200), &cpu).unwrap();
    let r200 = residual_norm(&rec, &sino, &geom, &cpu);
    eprintln!("grad (BB) residual: 10 iters = {r10:.3}, 200 iters = {r200:.3}");
    assert!(
        r200 < r10,
        "grad residual did not decrease: {r10} -> {r200}"
    );

    let slice = rec.array.index_axis(Axis(0), 0).to_owned();
    let corr = pearson_disk(&slice, &phantom, n, 0.85);
    eprintln!("grad (BB, 200 iters) Pearson correlation = {corr:.4}");
    assert!(
        corr > 0.9,
        "grad correlates poorly with phantom: r = {corr:.4}"
    );
}

#[test]
fn grad_fixed_step_decreases_residual() {
    // The fixed-step path (reg_par[0] ≥ 0) takes a constant step λ = reg_par[0].
    // A stable λ is projector-dependent: tomopy's r = 1/√(ncols·nang/2) puts the
    // step-1 iteration right at the stability boundary for its Siddon projector,
    // but this linear-interp adjoint pair has a larger operator norm, so the
    // tomopy-default unit step diverges and a smaller step is needed. With a
    // stably-small step the residual decreases monotonically with iterations.
    let n = 64;
    let nang = 90;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let p = |iters| ReconParams {
        num_gridx: Some(n),
        num_iter: iters,
        reg_par: vec![0.05], // stable fixed step for this projector
        ..Default::default()
    };

    let r5 = residual_norm(
        &recon::recon(&sino, &geom, Algorithm::Grad, &p(5), &cpu).unwrap(),
        &sino,
        &geom,
        &cpu,
    );
    let r100 = residual_norm(
        &recon::recon(&sino, &geom, Algorithm::Grad, &p(100), &cpu).unwrap(),
        &sino,
        &geom,
        &cpu,
    );
    eprintln!("grad (fixed step 0.05) residual: 5 iters = {r5:.3}, 100 iters = {r100:.3}");
    assert!(
        r100.is_finite() && r100 < r5,
        "grad fixed-step residual did not decrease: {r5} -> {r100}"
    );
}

/// Sum of squares over a centered disk (reconstruction energy).
fn disk_sumsq(a: &Array2<f32>, n: usize, radius_frac: f32) -> f32 {
    let c = (n as f32 - 1.0) / 2.0;
    let r2 = (radius_frac * n as f32 / 2.0).powi(2);
    let mut s = 0.0f32;
    for iy in 0..n {
        for ix in 0..n {
            let (dy, dx) = (iy as f32 - c, ix as f32 - c);
            if dx * dx + dy * dy <= r2 {
                s += a[[iy, ix]].powi(2);
            }
        }
    }
    s
}

#[test]
fn tikh_without_reg_equals_grad() {
    // tikh adds 2·reg_par[1]·(x − reg_data) to grad's gradient. With no reg_par[1]
    // (and the default zero prior) that term is identically zero, so tikh must be
    // bit-identical to grad at the same step.
    let n = 64;
    let nang = 90;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        num_iter: 30,
        reg_par: vec![-1.0],  // BB step, no Tikhonov weight ⇒ term vanishes
        ..Default::default()  // reg_data empty ⇒ zero prior
    };
    let g = recon::recon(&sino, &geom, Algorithm::Grad, &params, &cpu).unwrap();
    let t = recon::recon(&sino, &geom, Algorithm::Tikh, &params, &cpu).unwrap();

    let d = max_abs_diff(&g, &t);
    eprintln!("max |grad − tikh(no reg)| = {d:e}");
    assert_eq!(d, 0.0, "tikh without reg is not identical to grad: {d:e}");
}

#[test]
fn tikh_zero_prior_shrinks_energy() {
    // The Tikhonov term with a zero prior is a ridge penalty ‖x‖², so a positive
    // weight shrinks the reconstruction's energy relative to plain grad while
    // still tracking the phantom (damping, not destruction).
    let n = 96;
    let nang = 150;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let common = ReconParams {
        num_gridx: Some(n),
        num_iter: 120,
        reg_par: vec![-1.0], // BB step
        ..Default::default()
    };
    let g = recon::recon(&sino, &geom, Algorithm::Grad, &common, &cpu).unwrap();
    let t = recon::recon(
        &sino,
        &geom,
        Algorithm::Tikh,
        &ReconParams {
            reg_par: vec![-1.0, 0.5], // BB step + Tikhonov weight toward zero prior
            ..common
        },
        &cpu,
    )
    .unwrap();

    let g_s = g.array.index_axis(Axis(0), 0).to_owned();
    let t_s = t.array.index_axis(Axis(0), 0).to_owned();
    let e_g = disk_sumsq(&g_s, n, 0.85);
    let e_t = disk_sumsq(&t_s, n, 0.85);
    let corr = pearson_disk(&t_s, &phantom, n, 0.85);
    eprintln!("tikh ridge: energy grad = {e_g:.1}, tikh = {e_t:.1}, tikh r = {corr:.4}");
    assert!(
        e_t < e_g,
        "Tikhonov ridge did not shrink energy: tikh {e_t:.1} >= grad {e_g:.1}"
    );
    assert!(
        corr > 0.5,
        "tikh ridge destroyed the reconstruction: r = {corr:.4}"
    );
}

#[test]
fn tv_reconstructs_phantom() {
    // Chambolle–Pock TV reconstruction of the piecewise-constant phantom. TV
    // imposes no positivity, so the full signed phantom is the target.
    let n = 96;
    let nang = 150;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        num_iter: 200,
        reg_par: vec![0.01], // TV strength
        ..Default::default()
    };
    let rec = recon::recon(&sino, &geom, Algorithm::Tv, &params, &cpu).unwrap();
    let s = rec.array.index_axis(Axis(0), 0).to_owned();

    assert!(
        s.iter().all(|v| v.is_finite()),
        "TV produced non-finite values"
    );
    let corr = pearson_disk(&s, &phantom, n, 0.85);
    eprintln!("TV (λ=0.01, 200 iters) Pearson correlation = {corr:.4}");
    assert!(
        corr > 0.9,
        "TV correlates poorly with phantom: r = {corr:.4}"
    );
}

#[test]
fn tv_stronger_lambda_smooths() {
    // At matched iterations a larger TV strength yields a smoother reconstruction
    // (lower roughness over the disk) while still reconstructing the phantom.
    let n = 96;
    let nang = 150;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let p = |lambda: f32| ReconParams {
        num_gridx: Some(n),
        num_iter: 200,
        reg_par: vec![lambda],
        ..Default::default()
    };
    let weak = recon::recon(&sino, &geom, Algorithm::Tv, &p(0.01), &cpu).unwrap();
    let strong = recon::recon(&sino, &geom, Algorithm::Tv, &p(0.5), &cpu).unwrap();

    let weak_s = weak.array.index_axis(Axis(0), 0).to_owned();
    let strong_s = strong.array.index_axis(Axis(0), 0).to_owned();
    let rough_weak = roughness_disk(&weak_s, n, 0.85);
    let rough_strong = roughness_disk(&strong_s, n, 0.85);
    let corr_strong = pearson_disk(&strong_s, &phantom, n, 0.85);
    eprintln!(
        "TV smoothing: roughness weak(λ=0.01) = {rough_weak:.4}, strong(λ=0.5) = {rough_strong:.4}, strong r = {corr_strong:.4}"
    );
    assert!(
        rough_strong < rough_weak,
        "stronger TV did not smooth: {rough_strong:.4} >= {rough_weak:.4}"
    );
    assert!(
        corr_strong > 0.8,
        "strong-TV reconstruction degraded too far: r = {corr_strong:.4}"
    );
}

#[test]
fn art_reconstructs_and_converges() {
    // ART is row-action Kaczmarz with immediate per-ray updates and no positivity
    // constraint, so the full signed phantom is the target. The data residual must
    // shrink with iterations and the result correlate strongly with the phantom.
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
    let r5 = residual_norm(
        &recon::recon(&sino, &geom, Algorithm::Art, &p(5), &cpu).unwrap(),
        &sino,
        &geom,
        &cpu,
    );
    let rec = recon::recon(&sino, &geom, Algorithm::Art, &p(20), &cpu).unwrap();
    let r20 = residual_norm(&rec, &sino, &geom, &cpu);
    eprintln!("ART residual: 5 iters = {r5:.1}, 20 iters = {r20:.1}");
    assert!(r20 < r5, "ART residual did not decrease: {r5} -> {r20}");

    let slice = rec.array.index_axis(Axis(0), 0).to_owned();
    assert!(
        slice.iter().all(|v| v.is_finite()),
        "ART produced non-finite"
    );
    let corr = pearson_disk(&slice, &phantom, n, 0.85);
    eprintln!("ART (20 iters) Pearson correlation = {corr:.4}");
    assert!(
        corr > 0.95,
        "ART correlates poorly with phantom: r = {corr:.4}"
    );
}

#[test]
fn bart_ordered_subsets_reconstruct_and_accelerate() {
    // BART is ordered-subset SART. It reconstructs the phantom, and more subsets
    // accelerate convergence (lower residual at matched iterations) the same way
    // OSEM accelerates MLEM.
    let n = 96;
    let nang = 150;
    let cpu = CpuBackend::new();

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(phantom.clone().insert_axis(Axis(0)));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 1, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let p = |iters, nb| ReconParams {
        num_gridx: Some(n),
        num_iter: iters,
        num_block: nb,
        ind_block: interleaved_ind_block(nang, nb),
        ..Default::default()
    };

    let rec = recon::recon(&sino, &geom, Algorithm::Bart, &p(15, 15), &cpu).unwrap();
    let slice = rec.array.index_axis(Axis(0), 0).to_owned();
    assert!(
        slice.iter().all(|v| v.is_finite()),
        "BART produced non-finite"
    );
    let corr = pearson_disk(&slice, &phantom, n, 0.85);
    eprintln!("BART (15 iters × 15 blocks) Pearson correlation = {corr:.4}");
    assert!(
        corr > 0.95,
        "BART correlates poorly with phantom: r = {corr:.4}"
    );

    // Ordered-subset acceleration: 15 blocks reach a lower residual than 1 block
    // at the same iteration count.
    let res_many = residual_norm(
        &recon::recon(&sino, &geom, Algorithm::Bart, &p(8, 15), &cpu).unwrap(),
        &sino,
        &geom,
        &cpu,
    );
    let res_one = residual_norm(
        &recon::recon(&sino, &geom, Algorithm::Bart, &p(8, 1), &cpu).unwrap(),
        &sino,
        &geom,
        &cpu,
    );
    eprintln!("BART residual at 8 iters: 15 blocks = {res_many:.1}, 1 block = {res_one:.1}");
    assert!(
        res_many < res_one,
        "ordered subsets did not accelerate: 15-block {res_many:.1} >= 1-block {res_one:.1}"
    );
}
