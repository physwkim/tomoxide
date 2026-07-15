//! Exercises the out-of-core CUDA laminography path (`analytic_lamino_stream`):
//! projection-angle chunking (accumulated back-projection) × output rh-tiling.
//! On a large GPU the whole stack fits one chunk, so we force the chunker with
//! the `TOMOXIDE_CUDA_MAX_FREE_BYTES` debug hook and check the chunked volume
//! matches the single-shot volume (`analytic_lamino_chunk`) to the f32 floor.
//!
//! This test sets a process-global env var, so it lives in its own test binary
//! (one test per file) to avoid racing other CUDA tests' device state.

use ndarray::{Array2, Axis};
use tomoxide::{
    recon, sim, Algorithm, Angles, Beam, Center, CpuBackend, CudaBackend, Detector, Geometry,
    ReconParams, Volume,
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

fn nrmse(test: &[f32], reference: &[f32]) -> f32 {
    let mut se = 0.0f64;
    let mut sr = 0.0f64;
    for (&t, &r) in test.iter().zip(reference.iter()) {
        se += (t as f64 - r as f64).powi(2);
        sr += (r as f64).powi(2);
    }
    if sr <= 0.0 {
        return 0.0;
    }
    (se / sr).sqrt() as f32
}

#[test]
fn cuda_lamino_chunked_matches_single_shot() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();

    // Small stack: nz=32 detector rows, nproj=90 angles, n=64. Physical fidelity
    // is not under test here — only that the chunked path reproduces the single-
    // shot path — so any valid [nz, nproj, n] sinogram works. Build one by
    // parallel-projecting a stacked Shepp phantom, then reconstruct it as
    // laminography with both budgets.
    let (n, nproj, nz) = (64usize, 90usize, 32usize);
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = ndarray::Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let vol = Volume::new(stack);
    let angles = Angles::uniform(nproj, 0.0, std::f32::consts::PI);
    let geom_p = Geometry::parallel(angles.clone(), n, nz, 1.0);
    let sino = sim::project(&vol, &geom_p, &cpu).unwrap();

    // Laminography geometry: 20° pitch → phi = π/2 + 20°.
    let lamino_angle_deg = 20.0f32;
    let phi = std::f32::consts::FRAC_PI_2 + lamino_angle_deg * std::f32::consts::PI / 180.0;
    let geom_lam = Geometry {
        angles,
        center: Center::Scalar(n as f32 / 2.0),
        beam: Beam::Laminography { phi },
        detector: Detector {
            width: n,
            height: nz,
            pixel_size: 1.0,
        },
    };
    let rh = 48usize;
    let params = ReconParams {
        lamino_rh: Some(rh),
        ..Default::default()
    };

    // Single shot: the whole stack fits the real free-memory budget → fast path
    // = analytic_lamino_chunk, byte-identical to the un-streamed reconstruction.
    let single = recon::recon(&sino, &geom_lam, Algorithm::Linerec, &params, &cuda).unwrap();

    // Force the out-of-core chunker: a tiny budget drives ncproj < nproj (angle
    // chunking) and ncz < rh (rh-tiling), so the reconstruction runs the nested
    // filter-to-host + accumulate-per-angle-chunk path.
    std::env::set_var("TOMOXIDE_CUDA_MAX_FREE_BYTES", "500000");
    let chunked = recon::recon(&sino, &geom_lam, Algorithm::Linerec, &params, &cuda);
    std::env::remove_var("TOMOXIDE_CUDA_MAX_FREE_BYTES");
    let chunked = chunked.unwrap();

    assert_eq!(chunked.array.dim(), single.array.dim());
    assert_eq!(chunked.array.dim(), (rh, n, n));

    // Per-rh-slice correlation: the angle-chunked accumulation and nz-sub-chunked
    // filter differ from the single shot only in float summation order and the
    // cuFFT batch-algorithm choice, so every slice agrees to ~f32 rounding.
    for z in 0..rh {
        let a = single.array.index_axis(Axis(0), z).to_owned();
        let b = chunked.array.index_axis(Axis(0), z).to_owned();
        let r = pearson(&a, &b);
        assert!(
            r > 0.999,
            "chunked lamino slice {z} disagrees with single-shot: r = {r:.6}"
        );
    }
    let err = nrmse(
        chunked.array.as_slice().unwrap(),
        single.array.as_slice().unwrap(),
    );
    eprintln!("chunked vs single-shot lamino: global NRMSE = {err:.3e}");
    assert!(err < 1e-3, "chunked lamino NRMSE too large: {err:.3e}");
}

/// Multi-GPU laminography shards the output rh axis across devices: the whole
/// stack is filtered once (device 0) into a host-resident stack shared read-only
/// by per-device back-projection shards over disjoint rh ranges. Pin the device
/// set with `TOMOXIDE_CUDA_DEVICES` (each test runs in its own nextest process,
/// so setting the env is race-free) and check the multi-GPU volume matches the
/// single-GPU (whole-stack, fast-path) volume to the cuFFT floor. Skips unless
/// ≥2 CUDA devices are visible.
#[test]
fn cuda_lamino_multi_gpu_matches_single_gpu() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let all = tomoxide::cuda::selected_devices();
    if all.len() < 2 {
        eprintln!(
            "skipping multi-GPU lamino test: only {} device(s)",
            all.len()
        );
        return;
    }
    let cpu = CpuBackend::new();

    let (n, nproj, nz) = (64usize, 90usize, 32usize);
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = ndarray::Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let vol = Volume::new(stack);
    let angles = Angles::uniform(nproj, 0.0, std::f32::consts::PI);
    let geom_p = Geometry::parallel(angles.clone(), n, nz, 1.0);
    let sino = sim::project(&vol, &geom_p, &cpu).unwrap();

    let lamino_angle_deg = 20.0f32;
    let phi = std::f32::consts::FRAC_PI_2 + lamino_angle_deg * std::f32::consts::PI / 180.0;
    let geom_lam = Geometry {
        angles,
        center: Center::Scalar(n as f32 / 2.0),
        beam: Beam::Laminography { phi },
        detector: Detector {
            width: n,
            height: nz,
            pixel_size: 1.0,
        },
    };
    let rh = 48usize;
    let params = ReconParams {
        lamino_rh: Some(rh),
        ..Default::default()
    };

    // Single GPU (device 0): the whole stack fits → fast path = single-shot.
    std::env::set_var("TOMOXIDE_CUDA_DEVICES", "0");
    let single = recon::recon(&sino, &geom_lam, Algorithm::Linerec, &params, &cuda);
    // All visible GPUs: rh sharded across devices.
    let devs = all
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    std::env::set_var("TOMOXIDE_CUDA_DEVICES", &devs);
    let multi = recon::recon(&sino, &geom_lam, Algorithm::Linerec, &params, &cuda);
    std::env::remove_var("TOMOXIDE_CUDA_DEVICES");
    let single = single.unwrap();
    let multi = multi.unwrap();

    assert_eq!(multi.array.dim(), single.array.dim());
    assert_eq!(multi.array.dim(), (rh, n, n));
    for z in 0..rh {
        let a = single.array.index_axis(Axis(0), z).to_owned();
        let b = multi.array.index_axis(Axis(0), z).to_owned();
        let r = pearson(&a, &b);
        assert!(
            r > 0.999,
            "multi-GPU lamino slice {z} disagrees with single-GPU: r = {r:.6}"
        );
    }
    let err = nrmse(
        multi.array.as_slice().unwrap(),
        single.array.as_slice().unwrap(),
    );
    eprintln!(
        "multi-GPU ({} devices) vs single-GPU lamino: global NRMSE = {err:.3e}",
        all.len()
    );
    assert!(err < 1e-3, "multi-GPU lamino NRMSE too large: {err:.3e}");
}
