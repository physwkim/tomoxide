//! FP32 vs FP16 GPU reconstruction throughput.
//!
//! Times end-to-end `recon::recon` (host f32→f16 upload, half/float cuFFT
//! filter, back-projection, download) for the **same** synthetic sinogram at
//! `--dtype float32` and `--dtype float16`, on CUDA. The half path is only
//! implemented for the fused `fbp`/`linerec` and `fourierrec` algorithms, so
//! those are the choices here.
//!
//! The sinogram content is irrelevant to the timing — FBP/Fourierrec do a fixed
//! amount of work per element regardless of value — so it is filled with a
//! cheap deterministic pattern (no projection needed) and the build is instant.
//! A correlation check on the reconstructed mid-slice confirms the f16 run
//! actually reconstructed (rather than silently erroring to zeros).
//!
//! Usage: bench_dtype [algorithm] [n] [nproj] [nz] [iters]
//!   defaults: fbp 512 720 64 10   (nz must be even for fourierrec)

use std::time::Instant;

use ndarray::{Array3, Axis};
use tomoxide::{recon, Algorithm, Angles, CudaBackend, Dtype, Geometry, Layout, ReconParams, Tomo};

fn time_recon(
    sino: &Tomo<f32>,
    geom: &Geometry,
    algo: Algorithm,
    params: &ReconParams,
    cuda: &CudaBackend,
    iters: usize,
) -> (f64, f64, ndarray::Array2<f32>) {
    // Warm up: first call pays CUDA context init + cuFFT plan creation.
    let mut last = recon::recon(sino, geom, algo, params, cuda).expect("recon");
    let _ = recon::recon(sino, geom, algo, params, cuda).expect("recon");
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        last = recon::recon(sino, geom, algo, params, cuda).expect("recon");
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let median = times[times.len() / 2];
    let mid = last.dims().0 / 2;
    (min, median, last.array.index_axis(Axis(0), mid).to_owned())
}

fn pearson(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.sum() / n, b.sum() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x - ma, y - mb);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let algo_s = a.get(1).map(|s| s.as_str()).unwrap_or("fbp");
    let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let nproj: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(720);
    let nz: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let iters: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(10);

    let algo: Algorithm = match algo_s {
        "fbp" => Algorithm::Fbp,
        "linerec" => Algorithm::Linerec,
        "fourierrec" => Algorithm::Fourierrec,
        other => {
            eprintln!(
                "bench_dtype: algorithm '{other}' has no f16 path (use fbp|linerec|fourierrec)"
            );
            return;
        }
    };

    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bench_dtype: CUDA unavailable ({e}) — f16 is a GPU-only path, skipping");
            return;
        }
    };

    // Synthetic sinogram in Sinogram layout [nz, nproj, n] (recon normalizes to
    // this anyway); a smooth deterministic fill, value-independent timing.
    let sino_arr = Array3::<f32>::from_shape_fn((nz, nproj, n), |(z, p, x)| {
        let fx = x as f32 / n as f32 - 0.5;
        let fp = p as f32 / nproj as f32;
        0.5 + 0.4 * ((fx * 6.0 + z as f32 * 0.01).cos() * (fp * std::f32::consts::TAU).sin())
    });
    let sino = Tomo::new(sino_arr, Layout::Sinogram);
    let geom = Geometry::parallel(
        Angles::uniform(nproj, 0.0, std::f32::consts::PI),
        n,
        nz,
        1.0,
    );

    let p32 = ReconParams {
        num_gridx: Some(n),
        dtype: Dtype::F32,
        ..Default::default()
    };
    let p16 = ReconParams {
        num_gridx: Some(n),
        dtype: Dtype::F16,
        ..Default::default()
    };

    let gvox = (nz * n * n) as f64;
    println!(
        "bench_dtype: algo={algo_s} n={n} nproj={nproj} nz={nz} iters={iters} \
         ({:.1} Mvoxel volume, {:.1} Msample sinogram)",
        gvox / 1e6,
        (nz * nproj * n) as f64 / 1e6
    );

    let (min32, med32, s32) = time_recon(&sino, &geom, algo, &p32, &cuda, iters);
    let (min16, med16, s16) = time_recon(&sino, &geom, algo, &p16, &cuda, iters);

    println!("  f32: min {min32:7.2} ms   median {med32:7.2} ms");
    println!("  f16: min {min16:7.2} ms   median {med16:7.2} ms");
    println!(
        "  speedup (f32/f16): min {:.2}x   median {:.2}x",
        min32 / min16,
        med32 / med16
    );
    println!("  f16↔f32 mid-slice Pearson = {:.5}", pearson(&s16, &s32));
}
