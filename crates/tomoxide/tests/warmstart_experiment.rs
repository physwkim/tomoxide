//! Measure-first experiment for reconstruction-algorithm *chaining* (warm-start):
//! does feeding one method's output as the initial iterate of another improve
//! quality or convergence speed, the way ptychography chains DM/RAAR → ML?
//!
//! Runs on the CPU backend so the analytic (FBP) and iterative solvers share one
//! orientation/scale convention. (The CUDA analytic path is no longer the odd one
//! out — the Phase 1/2 unification gave it the CPU handedness and the `π/nproj`
//! dθ weight, see `docs/ARCHITECTURE.md` §4.1 — but keeping the seed and the
//! solver on one backend keeps the comparison clear of the ~1.6 % cross-backend
//! ramp-shape residual, which would otherwise ride on the warm-start seed.) A
//! Shepp–Logan phantom is the ground truth; it is forward-projected at a
//! sparse-ish angle count with additive Gaussian measurement noise, then
//! reconstructed. Quality is NRMSE / Pearson vs the ground truth over a disk.
//!
//! Ignored by default:
//!   cargo test -p tomoxide --release --test warmstart_experiment -- --ignored --nocapture

use ndarray::{Array2, Axis};
use std::f32::consts::PI;
use tomoxide::{recon, sim, Algorithm, Angles, CpuBackend, Geometry, ReconParams, Volume};

/// Deterministic standard-normal stream (LCG + Box–Muller), so the experiment is
/// reproducible without an RNG dependency.
struct Gauss {
    s: u64,
    spare: Option<f32>,
}
impl Gauss {
    fn new(seed: u64) -> Self {
        Gauss {
            s: seed,
            spare: None,
        }
    }
    fn u01(&mut self) -> f32 {
        self.s = self
            .s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.s >> 33) as f32) / ((1u64 << 31) as f32)
    }
    fn next(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let (mut u1, u2) = (self.u01(), self.u01());
        if u1 < 1e-12 {
            u1 = 1e-12;
        }
        let mag = (-2.0 * u1.ln()).sqrt();
        self.spare = Some(mag * (2.0 * PI * u2).sin());
        mag * (2.0 * PI * u2).cos()
    }
}

/// NRMSE and Pearson between reconstruction `a` and ground truth `b`, over a
/// centred disk of radius `radius_frac · n/2` (ignores the untouched corners).
fn metrics(a: &Array2<f32>, b: &Array2<f32>, n: usize, radius_frac: f32) -> (f64, f64) {
    let c = (n as f32 - 1.0) / 2.0;
    let r2 = (radius_frac * n as f32 / 2.0).powi(2);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for iy in 0..n {
        for ix in 0..n {
            let (dy, dx) = (iy as f32 - c, ix as f32 - c);
            if dx * dx + dy * dy <= r2 {
                xs.push(a[[iy, ix]] as f64);
                ys.push(b[[iy, ix]] as f64);
            }
        }
    }
    let nn = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / nn, ys.iter().sum::<f64>() / nn);
    let (mut sxy, mut sxx, mut syy, mut se, mut sb) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let (dx, dy) = (x - mx, y - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
        se += (x - y) * (x - y);
        sb += y * y;
    }
    let nrmse = (se / nn).sqrt() / (sb / nn).sqrt();
    let pearson = sxy / (sxx.sqrt() * syy.sqrt());
    (nrmse, pearson)
}

fn slice0(v: &Volume<f32>) -> Array2<f32> {
    v.array.index_axis(Axis(0), 0).to_owned()
}

fn base_params(n: usize, num_iter: usize) -> ReconParams {
    ReconParams {
        num_gridx: Some(n),
        num_iter,
        ..Default::default()
    }
}

#[test]
#[ignore]
fn warmstart_chaining_experiment() {
    let cpu = CpuBackend::new();
    let (n, nang) = (256usize, 64usize); // sparse-ish: FBP will streak
    let gt2d = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
    let gt3d = gt2d.clone().insert_axis(Axis(0));
    let vol = Volume::new(gt3d);
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, PI), n, 1, 1.0);

    // Clean sinogram, then additive Gaussian noise at σ = 3% of the mean |value|.
    let mut sino = sim::project(&vol, &geom, &cpu).unwrap();
    let mean_abs = sino.array.iter().map(|v| v.abs()).sum::<f32>() / sino.array.len() as f32;
    let sigma = 0.03 * mean_abs;
    let mut g = Gauss::new(0x00C0_FFEE_1234_5678);
    sino.array.mapv_inplace(|v| (v + sigma * g.next()).max(0.0)); // clamp ≥0 for EM

    let tv_lambda = 2e-3f32;
    let budget = 40usize; // fixed total iteration budget so chains are fair

    // ---- from-scratch baselines ----
    let fbp = recon::recon(&sino, &geom, Algorithm::Fbp, &base_params(n, 1), &cpu).unwrap();
    let sirt = recon::recon(&sino, &geom, Algorithm::Sirt, &base_params(n, budget), &cpu).unwrap();
    let mlem = recon::recon(&sino, &geom, Algorithm::Mlem, &base_params(n, budget), &cpu).unwrap();
    let osem = recon::recon(
        &sino,
        &geom,
        Algorithm::Osem,
        &ReconParams {
            num_block: 8,
            ..base_params(n, budget)
        },
        &cpu,
    )
    .unwrap();
    let tv = recon::recon(
        &sino,
        &geom,
        Algorithm::Tv,
        &ReconParams {
            reg_par: vec![tv_lambda],
            ..base_params(n, budget)
        },
        &cpu,
    )
    .unwrap();

    // ---- chains (warm-start), same total budget ----
    // FBP → SIRT(budget)
    let fbp_sirt = recon::recon(
        &sino,
        &geom,
        Algorithm::Sirt,
        &ReconParams {
            init: Some(fbp.clone()),
            ..base_params(n, budget)
        },
        &cpu,
    )
    .unwrap();
    // FBP → TV(budget)
    let fbp_tv = recon::recon(
        &sino,
        &geom,
        Algorithm::Tv,
        &ReconParams {
            init: Some(fbp.clone()),
            reg_par: vec![tv_lambda],
            ..base_params(n, budget)
        },
        &cpu,
    )
    .unwrap();
    // OSEM(10, nb8) → MLEM(30)
    let osem_pre = recon::recon(
        &sino,
        &geom,
        Algorithm::Osem,
        &ReconParams {
            num_block: 8,
            ..base_params(n, 10)
        },
        &cpu,
    )
    .unwrap();
    let osem_mlem = recon::recon(
        &sino,
        &geom,
        Algorithm::Mlem,
        &ReconParams {
            init: Some(osem_pre),
            ..base_params(n, 30)
        },
        &cpu,
    )
    .unwrap();
    // SIRT(20) → TV(20)
    let sirt_pre = recon::recon(&sino, &geom, Algorithm::Sirt, &base_params(n, 20), &cpu).unwrap();
    let sirt_tv = recon::recon(
        &sino,
        &geom,
        Algorithm::Tv,
        &ReconParams {
            init: Some(sirt_pre),
            reg_par: vec![tv_lambda],
            ..base_params(n, 20)
        },
        &cpu,
    )
    .unwrap();

    let rows: [(&str, &Volume<f32>); 9] = [
        ("FBP (analytic)", &fbp),
        ("SIRT(40) scratch", &sirt),
        ("MLEM(40) scratch", &mlem),
        ("OSEM(40,nb8) scratch", &osem),
        ("TV(40) scratch", &tv),
        ("FBP -> SIRT(40)", &fbp_sirt),
        ("FBP -> TV(40)", &fbp_tv),
        ("OSEM(10) -> MLEM(30)", &osem_mlem),
        ("SIRT(20) -> TV(20)", &sirt_tv),
    ];
    println!(
        "\n== Quality vs ground truth (n={n}, nang={nang}, noise σ=3%, budget={budget} iters) =="
    );
    println!("  {:<24} {:>10} {:>10}", "config", "NRMSE", "Pearson");
    for (name, v) in rows {
        let (nrmse, r) = metrics(&slice0(v), &gt2d, n, 0.9);
        println!("  {name:<24} {nrmse:>10.4} {r:>10.4}");
    }

    // ---- convergence speed: SIRT from zero vs from FBP, NRMSE at each budget ----
    println!("\n== SIRT convergence: NRMSE at N iterations, scratch vs FBP-warm-start ==");
    println!("  {:>6} {:>12} {:>12}", "iters", "scratch", "FBP-init");
    for &iters in &[2usize, 5, 10, 20, 40] {
        let s0 = recon::recon(&sino, &geom, Algorithm::Sirt, &base_params(n, iters), &cpu).unwrap();
        let s1 = recon::recon(
            &sino,
            &geom,
            Algorithm::Sirt,
            &ReconParams {
                init: Some(fbp.clone()),
                ..base_params(n, iters)
            },
            &cpu,
        )
        .unwrap();
        let (e0, _) = metrics(&slice0(&s0), &gt2d, n, 0.9);
        let (e1, _) = metrics(&slice0(&s1), &gt2d, n, 0.9);
        println!("  {iters:>6} {e0:>12.4} {e1:>12.4}");
    }
}
