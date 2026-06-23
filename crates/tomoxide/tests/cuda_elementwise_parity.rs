//! CUDA elementwise preprocessing parity (M4): dark/flat correction + minus-log
//! on the GPU vs the CPU backend.
#![cfg(feature = "cuda")]

use ndarray::Array3;
use tomoxide::backend::{Backend, Elementwise};
use tomoxide::{CpuBackend, CudaBackend, Frames, Layout, Tomo};

fn max_abs(a: &Tomo<f32>, b: &Tomo<f32>) -> f32 {
    a.array
        .iter()
        .zip(b.array.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn cuda_darkflat_and_minus_log_match_cpu() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (nproj, nz, nx) = (8usize, 3, 16);

    // Deterministic raw intensities in a sensible transmission range, with
    // flat > dark so the denominator is well-conditioned.
    let data = Array3::from_shape_fn((nproj, nz, nx), |(p, z, x)| {
        200.0 + 600.0 * ((p * 7 + z * 3 + x) % 11) as f32 / 11.0
    });
    let flat = Array3::from_shape_fn((2, nz, nx), |(_f, z, x)| 950.0 + (z * nx + x) as f32 * 0.1);
    let dark = Array3::from_shape_fn((2, nz, nx), |(_f, z, x)| 10.0 + (z + x) as f32 * 0.05);
    let frames_flat = Frames::new(flat);
    let frames_dark = Frames::new(dark);

    // darkflat parity.
    let mut t_cpu = Tomo::new(data.clone(), Layout::Projection);
    let mut t_cuda = Tomo::new(data.clone(), Layout::Projection);
    cpu.elementwise()
        .unwrap()
        .darkflat(&mut t_cpu, &frames_flat, &frames_dark)
        .unwrap();
    cuda.elementwise()
        .unwrap()
        .darkflat(&mut t_cuda, &frames_flat, &frames_dark)
        .unwrap();
    let d_df = max_abs(&t_cpu, &t_cuda);
    eprintln!("darkflat max|Δ| = {d_df:e}");
    assert!(d_df < 1e-5, "darkflat GPU≠CPU: {d_df}");

    // minus_log parity (run on the corrected data).
    cpu.elementwise().unwrap().minus_log(&mut t_cpu).unwrap();
    cuda.elementwise().unwrap().minus_log(&mut t_cuda).unwrap();
    let d_ml = max_abs(&t_cpu, &t_cuda);
    eprintln!("minus_log max|Δ| = {d_ml:e}");
    // CUDA logf vs libm ln differ by ~1 ULP; loose absolute floor.
    assert!(d_ml < 1e-5, "minus_log GPU≠CPU: {d_ml}");
}
