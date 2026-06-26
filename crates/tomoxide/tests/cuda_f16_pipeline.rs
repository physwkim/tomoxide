//! Exercises the out-of-core f16 Fbp/Linerec path: `analytic_fbp_stream_f16`'s
//! tiled async pipeline (`analytic_fbp_pipeline_f16`). On a large GPU the stack
//! always fits one chunk, so we force the tiler with the
//! `TOMOXIDE_CUDA_MAX_FREE_BYTES` debug hook and check the tiled-pipeline volume
//! matches the single-chunk f16 volume.
//!
//! This test sets a process-global env var, so it lives in its own test binary
//! (one test per file) to avoid racing other CUDA tests' device state.

use ndarray::{Array2, Axis};
use tomoxide::{
    recon, sim, Algorithm, Angles, CpuBackend, CudaBackend, Dtype, Geometry, ReconParams,
};

fn pearson(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
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

#[test]
fn cuda_fbp_f16_tiled_pipeline_matches_single_chunk() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (n, nang, nz) = (128usize, 180usize, 8usize);
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = ndarray::Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let vol = tomoxide::Volume::new(stack);
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, nz, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();
    let params = ReconParams {
        num_gridx: Some(n),
        dtype: Dtype::F16,
        ..Default::default()
    };

    // Single chunk (the whole stack fits the real free-memory budget).
    let single = recon::recon(&sino, &geom, Algorithm::Fbp, &params, &cuda).unwrap();

    // Force the tiler: a tiny free-memory budget splits nz=8 into ≥2 chunks and
    // routes through analytic_fbp_pipeline_f16.
    std::env::set_var("TOMOXIDE_CUDA_MAX_FREE_BYTES", "3000000");
    let tiled = recon::recon(&sino, &geom, Algorithm::Fbp, &params, &cuda);
    std::env::remove_var("TOMOXIDE_CUDA_MAX_FREE_BYTES");
    let tiled = tiled.unwrap();

    assert_eq!(tiled.array.dim(), single.array.dim());
    // Identical stacked slices ⇒ no cross-z interpolation difference at tile
    // edges, so the tiled pipeline should reproduce the single-chunk volume to
    // the f16 floor.
    for z in 0..nz {
        let a = single.array.index_axis(Axis(0), z).to_owned();
        let b = tiled.array.index_axis(Axis(0), z).to_owned();
        let r = pearson(&a, &b);
        assert!(
            r > 0.999,
            "tiled f16 pipeline slice {z} disagrees with single-chunk f16: r = {r:.6}"
        );
    }
    eprintln!("f16 tiled pipeline matches single-chunk across all {nz} slices");
}
