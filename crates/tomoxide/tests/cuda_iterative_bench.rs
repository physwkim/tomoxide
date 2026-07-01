//! Wall-clock benchmark for the CUDA iterative suite vs CPU. Ignored by default;
//! run with:
//!   cargo test -p tomoxide --features cuda --release --test cuda_iterative_bench -- --ignored --nocapture
#![cfg(feature = "cuda")]

use ndarray::Array3;
use std::time::Instant;
use tomoxide::{
    recon, sim, Algorithm, Angles, Backend, CpuBackend, CudaBackend, Geometry, Layout, ReconParams,
    Tomo, Volume,
};

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
    println!("  {label:<22} {dt:8.3} s");
    dt
}

/// The pre-device-resident CUDA SIRT: the generic host solver composed from
/// `projector()`/`backprojector()`, which round-trips the whole volume/sinogram
/// host↔device *every* iteration (full H2D + kernel + D2H per project and per
/// backproject) plus host `ndarray` elementwise. Replicated verbatim here so the
/// bench isolates exactly what the device-resident path removes. Times a warmed
/// second run.
fn time_periter_cuda_sirt(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    cuda: &CudaBackend,
) -> f64 {
    let proj = cuda.projector().expect("cuda projector");
    let bp = cuda.backprojector().expect("cuda backprojector");
    let run = || {
        let n = params.num_gridx.unwrap();
        let nz = sino.n_rows();
        let b = sino.as_layout(Layout::Sinogram);
        let (nang, ncols) = (b.n_angles(), b.n_cols());
        let ones_img = Volume::new(Array3::from_elem((nz, n, n), 1.0));
        let mut ray = Tomo::new(Array3::zeros((nz, nang, ncols)), Layout::Sinogram);
        proj.project(&ones_img, geom, &mut ray).unwrap();
        let rw = ray
            .array
            .mapv(|v| if v.abs() > 1e-6 { 1.0 / v } else { 0.0 });
        let ones_sino = Tomo::new(Array3::from_elem((nz, nang, ncols), 1.0), Layout::Sinogram);
        let mut sens = Volume::new(Array3::zeros((nz, n, n)));
        bp.backproject(&ones_sino, geom, &mut sens).unwrap();
        let cw = sens
            .array
            .mapv(|v| if v.abs() > 1e-6 { 1.0 / v } else { 0.0 });
        let mut vol = Volume::new(Array3::zeros((nz, n, n)));
        let mut ax = Tomo::new(Array3::zeros((nz, nang, ncols)), Layout::Sinogram);
        let mut corr = Volume::new(Array3::zeros((nz, n, n)));
        for _ in 0..params.num_iter.max(1) {
            proj.project(&vol, geom, &mut ax).unwrap();
            let mut resid = &b.array - &ax.array;
            resid *= &rw;
            bp.backproject(&Tomo::new(resid, Layout::Sinogram), geom, &mut corr)
                .unwrap();
            vol.array += &(&cw * &corr.array);
        }
    };
    run(); // warm-up
    let t = Instant::now();
    run();
    t.elapsed().as_secs_f64()
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

    // Simultaneous methods (GPU forward+backproject per iteration).
    for &(n, nz, nproj, iters) in &[(512usize, 8usize, 720usize, 50usize), (1024, 4, 720, 30)] {
        let geom = Geometry::parallel(
            Angles::uniform(nproj, 0.0, std::f32::consts::PI),
            n,
            nz,
            1.0,
        );
        let phantom = sim::shepp2d(n).unwrap();
        let vol = Volume::new(stack(&phantom, nz));
        let sino = sim::project(&vol, &geom, &cpu).unwrap();
        let params = ReconParams {
            num_iter: iters,
            num_gridx: Some(n),
            ..Default::default()
        };
        println!("\nSIRT  n={n} nz={nz} nproj={nproj} iters={iters}");
        let c = time_recon("cpu", &sino, &geom, Algorithm::Sirt, &params, &cpu);
        let g_pi = time_periter_cuda_sirt(&sino, &geom, &params, &cuda);
        println!("  {:<22} {g_pi:8.3} s", "cuda per-iteration");
        let g = time_recon(
            "cuda device-resident",
            &sino,
            &geom,
            Algorithm::Sirt,
            &params,
            &cuda,
        );
        println!("  speedup (cpu/cuda)     {:8.2}x", c / g);
        println!("  device-resident gain   {:8.2}x", g_pi / g);
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
            println!("  ratio (cuda/cpu)       {:8.2}x", g / c);
        }
    }
}
