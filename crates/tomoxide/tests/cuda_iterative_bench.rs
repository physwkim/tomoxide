//! Wall-clock benchmark for the CUDA iterative suite vs CPU. Ignored by default;
//! run with:
//!   cargo test -p tomoxide --features cuda --release --test cuda_iterative_bench -- --ignored --nocapture
#![cfg(feature = "cuda")]

use std::time::Instant;
use tomoxide::{
    recon, sim, Algorithm, Angles, Backend, CpuBackend, CudaBackend, DeviceKind, Dtype,
    FilteredBackproject, ForwardProject, Geometry, IterativeReconstruct, ReconParams, Volume,
};

/// A `CudaBackend` view with the device-resident iterative path hidden
/// (`iterative_reconstruct()` → `None`), so `recon` uses the generic host solver
/// composed from the CUDA projector/backprojector — the pre-device-resident path
/// (full H2D + kernel + D2H per project and per backproject, every iteration,
/// plus host `ndarray` elementwise). Same CUDA kernels as the device-resident
/// path, so timing the two isolates exactly what device-residency removes.
struct PerIterCuda<'a>(&'a CudaBackend);

impl Backend for PerIterCuda<'_> {
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn device(&self) -> DeviceKind {
        self.0.device()
    }
    fn supports(&self, dt: Dtype) -> bool {
        self.0.supports(dt)
    }
    fn projector(&self) -> Option<&dyn ForwardProject> {
        self.0.projector()
    }
    fn backprojector(&self) -> Option<&dyn FilteredBackproject> {
        self.0.backprojector()
    }
    fn iterative_reconstruct(&self) -> Option<&dyn IterativeReconstruct> {
        None
    }
}

fn stack(slice2d: &ndarray::Array2<f32>, nz: usize) -> ndarray::Array3<f32> {
    let (h, w) = slice2d.dim();
    let mut v = ndarray::Array3::<f32>::zeros((nz, h, w));
    for z in 0..nz {
        v.index_axis_mut(ndarray::Axis(0), z).assign(slice2d);
    }
    v
}

fn time_recon(
    label: &str,
    sino: &tomoxide::Tomo<f32>,
    geom: &Geometry,
    algo: Algorithm,
    params: &ReconParams,
    backend: &dyn Backend,
) -> f64 {
    // one warm-up (CUDA init / first-touch allocations excluded)
    let _ = recon::recon(sino, geom, algo, params, backend).unwrap();
    let t = Instant::now();
    let _ = recon::recon(sino, geom, algo, params, backend).unwrap();
    let dt = t.elapsed().as_secs_f64();
    println!("  {label:<24} {dt:8.3} s");
    dt
}

#[test]
#[ignore]
fn bench_iterative_cpu_vs_cuda() {
    let cuda = match CudaBackend::new() {
        Ok(c) => c,
        Err(_) => {
            println!("no CUDA device; skipping");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let periter = PerIterCuda(&cuda);

    // Full device-resident iterative suite: device-resident vs per-iteration
    // CUDA vs CPU. The non-negative phantom is used throughout (EM/penalized-ML
    // require it; the others are unaffected). Each entry carries its own params
    // (reg_par / num_block) so the penalized and regularized methods are set up
    // exactly as `recon` expects.
    // run_cpu: the CPU baseline is only measured at 512² (the cpu/cuda ratio);
    // at 1024² a full CPU pass over all 8 methods would take hours and the
    // device-residency story is the per-iteration-vs-device-resident CUDA gain.
    for &(n, nz, nproj, iters, run_cpu) in &[
        (512usize, 8usize, 720usize, 30usize, true),
        (1024, 4, 720, 30, false),
    ] {
        let geom = Geometry::parallel(
            Angles::uniform(nproj, 0.0, std::f32::consts::PI),
            n,
            nz,
            1.0,
        );
        let phantom = sim::shepp2d(n).unwrap().mapv(|v| v.max(0.0));
        let vol = Volume::new(stack(&phantom, nz));
        let sino = sim::project(&vol, &geom, &cpu).unwrap();
        let base = ReconParams {
            num_iter: iters,
            num_gridx: Some(n),
            ..Default::default()
        };
        let methods: [(Algorithm, ReconParams); 8] = [
            (Algorithm::Sirt, base.clone()),
            (Algorithm::Mlem, base.clone()),
            (
                Algorithm::Osem,
                ReconParams {
                    num_block: 8,
                    ..base.clone()
                },
            ),
            (
                Algorithm::OspmlQuad,
                ReconParams {
                    num_block: 8,
                    reg_par: vec![0.1],
                    ..base.clone()
                },
            ),
            (
                Algorithm::PmlHybrid,
                ReconParams {
                    reg_par: vec![0.1, 0.01],
                    ..base.clone()
                },
            ),
            (
                Algorithm::Grad,
                ReconParams {
                    reg_par: vec![1e-3],
                    ..base.clone()
                },
            ),
            (
                Algorithm::Tikh,
                ReconParams {
                    reg_par: vec![-1.0, 0.1],
                    ..base.clone()
                },
            ),
            (
                Algorithm::Tv,
                ReconParams {
                    reg_par: vec![1e-3],
                    ..base.clone()
                },
            ),
        ];
        for (algo, params) in &methods {
            println!(
                "\n{algo:?}  n={n} nz={nz} nproj={nproj} iters={iters} num_block={}",
                params.num_block
            );
            let g_pi = time_recon("cuda per-iteration", &sino, &geom, *algo, params, &periter);
            let g = time_recon("cuda device-resident", &sino, &geom, *algo, params, &cuda);
            if run_cpu {
                let c = time_recon("cpu", &sino, &geom, *algo, params, &cpu);
                println!("  speedup (cpu/cuda)       {:8.2}x", c / g);
            }
            println!("  device-resident gain     {:8.2}x", g_pi / g);
        }
    }

    // Row-action methods (pure host; CUDA == CPU by construction).
    for &(n, nz, nproj, iters) in &[(256usize, 2usize, 180usize, 5usize)] {
        let geom = Geometry::parallel(
            Angles::uniform(nproj, 0.0, std::f32::consts::PI),
            n,
            nz,
            1.0,
        );
        let phantom = sim::shepp2d(n).unwrap();
        let vol = Volume::new(stack(&phantom, nz));
        let sino = sim::project(&vol, &geom, &cpu).unwrap();
        for algo in [Algorithm::Art, Algorithm::Bart] {
            let params = ReconParams {
                num_iter: iters,
                num_gridx: Some(n),
                num_block: 3,
                ..Default::default()
            };
            println!("\n{algo:?}  n={n} nz={nz} nproj={nproj} iters={iters}");
            let c = time_recon("cpu", &sino, &geom, algo, &params, &cpu);
            let g = time_recon("cuda", &sino, &geom, algo, &params, &cuda);
            println!("  ratio (cuda/cpu)         {:8.2}x", g / c);
        }
    }
}
