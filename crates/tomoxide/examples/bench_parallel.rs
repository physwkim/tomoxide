//! Throughput benchmark for the CPU loop-parallelization (Layer 1 + Layer 2).
//!
//! Recon time depends only on the input *dimensions*, not the sample values, so
//! a synthetic sinogram of the right shape isolates the kernel cost cleanly.
//!
//!   cargo run --release --example bench_parallel -- cpu  512 512 192
//!   cargo run --release --features cuda --example bench_parallel -- cuda 512 512 192
//!
//! Args: <backend cpu|cuda> [nd] [nang] [nz]. Compare CPU serial vs parallel by
//! pinning the rayon pool:  RAYON_NUM_THREADS=1 cargo run --release ... cpu ...

use std::time::Instant;

use ndarray::Array3;
use tomoxide::recon::recon;
use tomoxide::{
    Algorithm, Angles, BackendKind, Engine, Geometry, Layout, PhaseMethod, ReconParams, Tomo,
};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Order-stable FNV-1a hash over each element's raw f32 bits. Bit-identical
/// volumes hash equal; any single-bit difference changes the hash. Used to
/// confirm single-GPU vs multi-GPU output parity.
fn checksum(arr: &ndarray::Array3<f32>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in arr.iter() {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn time_recon(
    sino: &Tomo<f32>,
    geom: &Geometry,
    algo: Algorithm,
    params: &ReconParams,
    backend: &dyn tomoxide::Backend,
    nz: usize,
    reps: usize,
) {
    // Warmup (plan caching, page faults) + parity checksum.
    let warm = recon(sino, geom, algo, params, backend).unwrap();
    let csum = checksum(&warm.array);
    if let Ok(dir) = std::env::var("TOMOXIDE_BENCH_DUMP") {
        let bytes: Vec<u8> = warm.array.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}_{algo:?}.f32"), bytes).unwrap();
    }
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let _ = recon(sino, geom, algo, params, backend).unwrap();
        ts.push(t.elapsed().as_secs_f64());
    }
    let med = median(ts);
    println!(
        "  {:<11} {:8.3} s   {:8.1} slices/s   csum={csum:016x}",
        format!("{algo:?}"),
        med,
        nz as f64 / med
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let backend_kind = args.get(1).map(String::as_str).unwrap_or("cpu");
    let nd: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let nang: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
    let nz: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(192);
    let reps: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);
    // Optional 6th arg: run only one algorithm (fbp|gridrec|fourierrec|lprec|
    // paganin) so a matrix driver can isolate each cell — one OOM/timeout then
    // does not lose the other algorithms' numbers. Default "all".
    let only = args
        .get(6)
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| "all".into());
    let want = |name: &str| only == "all" || only == name;

    let kind = match backend_kind {
        "cuda" => BackendKind::Cuda,
        "wgpu" => BackendKind::Wgpu,
        _ => BackendKind::Cpu,
    };
    let engine = Engine::new(kind).unwrap();
    let backend = engine.backend();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into());

    println!(
        "backend={}  nd={nd} nang={nang} nz={nz}  reps={reps}  RAYON_NUM_THREADS={threads}",
        backend.name()
    );

    // Equally spaced angles over [0, pi) — required for lprec's log-polar grid.
    let theta: Vec<f32> = (0..nang)
        .map(|i| i as f32 * std::f32::consts::PI / nang as f32)
        .collect();
    let geom = Geometry::parallel(Angles(theta.clone()), nd, nz, 1.0);
    let params = ReconParams {
        num_gridx: Some(nd),
        ..Default::default()
    };

    // Synthetic sinogram [nz, nang, nd] (sinogram layout). Deterministic non-zero
    // fill so no value-dependent fast path is hit.
    let sino_arr = Array3::<f32>::from_shape_fn((nz, nang, nd), |(z, a, d)| {
        (((z * 31 + a * 7 + d * 3) % 101) as f32) / 101.0
    });
    let sino = Tomo::new(sino_arr, Layout::Sinogram);

    println!("-- reconstruction (per-slice loop) --");
    for algo in [
        Algorithm::Fbp,
        Algorithm::Gridrec,
        Algorithm::Fourierrec,
        Algorithm::Lprec,
    ] {
        if !want(&format!("{algo:?}").to_lowercase()) {
            continue;
        }
        time_recon(&sino, &geom, algo, &params, backend, nz, reps);
    }

    if !want("paganin") {
        return;
    }
    // Phase retrieval [nproj=nz, dy=nd, dz=nd] (projection layout, per-projection
    // 2-D FFT loop).
    println!("-- phase retrieval (per-projection FFT loop) --");
    let proj_arr = Array3::<f32>::from_shape_fn((nz, nd, nd), |(p, i, j)| {
        (((p * 17 + i * 5 + j * 2) % 97) as f32) / 97.0 + 0.5
    });
    let method = PhaseMethod::Paganin {
        pixel_size: 1e-4,
        dist: 50.0,
        energy: 30.0,
        alpha: 1e-3,
    };
    let pcsum = {
        let mut warm = Tomo::new(proj_arr.clone(), Layout::Projection);
        tomoxide::prep::retrieve_phase(&mut warm, method, backend).unwrap();
        checksum(&warm.array)
    };
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut p = Tomo::new(proj_arr.clone(), Layout::Projection);
        let t = Instant::now();
        tomoxide::prep::retrieve_phase(&mut p, method, backend).unwrap();
        ts.push(t.elapsed().as_secs_f64());
    }
    let med = median(ts);
    println!(
        "  {:<11} {:8.3} s   {:8.1} proj/s   csum={pcsum:016x}",
        "Paganin",
        med,
        nz as f64 / med
    );
}
