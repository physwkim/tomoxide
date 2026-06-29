//! Device-only profiling harness for the on-device stripe kernels.
//!
//! Runs `reconstruct_chunk_raw` (which applies the GPU stripe pass before the
//! reconstruction) for each ported method, on a realistic chunk, with a few
//! warmup iterations and then `iters` timed iterations. The host route is *not*
//! run, so under `nsys` the `vo_*` / `fw_* `/ `ti_*` kernels stand alone and
//! their per-kernel time can be summed with `nsys stats`.
//!
//!   nsys profile -o /tmp/prof_stripe --force-overwrite true \
//!     ./target/release/examples/prof_stripe [nproj] [nz] [ncols] [iters]
//!   nsys stats --report cuda_gpu_kern_sum /tmp/prof_stripe.nsys-rep

use ndarray::Array3;
use std::time::Instant;
use tomoxide::params::StripeMethod;
use tomoxide::{
    Algorithm, Angles, BackendKind, Dtype, Engine, Geometry, Layout, ReconParams, Tomo,
};

fn make_proj(nproj: usize, nz: usize, ncols: usize) -> Array3<f32> {
    Array3::from_shape_fn((nproj, nz, ncols), |(p, z, x)| {
        let base =
            0.5 + 0.4 * ((p as f32 * 0.017 + x as f32 * 0.013 + z as f32 * 0.011).sin() * 0.5);
        let stripe = 1.0 + 0.06 * (((x * 7 + 13) % 17) as f32 / 17.0 - 0.5);
        (base * stripe).clamp(0.1, 0.9)
    })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let nproj: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1500);
    let nz: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let ncols: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let iters: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);

    let engine = Engine::new(BackendKind::Cuda).expect("cuda engine");
    let backend = engine.backend();
    let theta: Vec<f32> = (0..nproj)
        .map(|p| p as f32 * std::f32::consts::PI / nproj as f32)
        .collect();
    let geom = Geometry::parallel(Angles(theta.clone()), ncols, nz, 1.0);
    let params = ReconParams {
        num_gridx: Some(ncols),
        dtype: Dtype::F32,
        ..Default::default()
    };
    let ar = backend
        .analytic_reconstruct()
        .expect("analytic_reconstruct");
    let mut recon = ar
        .streaming(Algorithm::Fbp, &params, &geom, ncols, nz)
        .expect("streaming")
        .expect("cuda streaming reconstructor");

    let raw = Tomo::new(make_proj(nproj, nz, ncols), Layout::Projection);

    let methods: &[(&str, StripeMethod)] = &[
        (
            "ti",
            StripeMethod::Ti {
                nblock: 0,
                beta: 1.5,
            },
        ),
        (
            "fw",
            StripeMethod::Fw {
                sigma: 2.0,
                level: None,
            },
        ),
        (
            "voall",
            StripeMethod::VoAll {
                snr: 3.0,
                la_size: 61,
                sm_size: 21,
            },
        ),
    ];

    println!("prof_stripe: nproj={nproj} nz={nz} ncols={ncols} iters={iters}");
    for &(name, stripe) in methods {
        // Warmup (plan/handle caches, first-touch allocations).
        for _ in 0..2 {
            recon
                .reconstruct_chunk_raw(&raw, None, None, &geom, stripe)
                .expect("raw ok")
                .expect("device handled");
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            recon
                .reconstruct_chunk_raw(&raw, None, None, &geom, stripe)
                .expect("raw ok")
                .expect("device handled");
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("  {name:<6} wall {ms:8.2} ms/chunk (stripe + recon + up/download)");
    }
}
