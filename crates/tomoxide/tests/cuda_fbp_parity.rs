//! CUDA FBP back-projection parity (M4).
//!
//! Reconstructs a forward-projected Shepp-Logan phantom with `recon(Fbp)` on the
//! CUDA backend (FBP filter on the CPU via the shared definition, back-projection
//! on the GPU via tomocupy's `cfunc_linerec`) and checks it against both the
//! phantom (round-trip) and the pure-CPU reconstruction. The GPU kernel uses
//! tomocupy's convention — a y-flip and a `4/nproj` scale vs the CPU
//! back-projector's `π/nproj` — so the comparison is scale-invariant Pearson
//! correlation, with the y-flip applied to line the two grids up.
//!
//! Runs only with the `cuda` feature (needs an NVIDIA device + nvcc at build).
#![cfg(feature = "cuda")]

use ndarray::{Array2, Axis};
use tomoxide::{recon, sim, Algorithm, Angles, CpuBackend, CudaBackend, Geometry, ReconParams};

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
    for (&x, &y) in xs.iter().zip(&ys) {
        let (dx, dy) = (x - mx, y - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

fn flipud(a: &Array2<f32>) -> Array2<f32> {
    let (nr, nc) = a.dim();
    Array2::from_shape_fn((nr, nc), |(i, j)| a[[nr - 1 - i, j]])
}

#[test]
fn cuda_fbp_matches_cpu_and_phantom() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (n, nang) = (128usize, 180usize);
    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };

    // cfunc_linerec interpolates vertically across slices, so it needs ≥2 rows
    // (a single slice yields zeros). Stack the phantom into a few identical
    // slices and compare an interior one.
    let nz = 4usize;
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = ndarray::Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let vol = tomoxide::Volume::new(stack);
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, nz, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let cpu_rec = recon::recon(&sino, &geom, Algorithm::Fbp, &params, &cpu).unwrap();
    let cuda_rec = recon::recon(&sino, &geom, Algorithm::Fbp, &params, &cuda).unwrap();
    let mid = nz / 2;
    let cpu_slice = cpu_rec.array.index_axis(Axis(0), mid).to_owned();
    let cuda_slice = cuda_rec.array.index_axis(Axis(0), mid).to_owned();
    assert_eq!(cuda_slice.dim(), (n, n));

    // CUDA vs CPU back-projector: identical up to the kernel's y-flip + scale.
    let r_cpu = pearson_disk(&cuda_slice, &flipud(&cpu_slice), n, 0.85);
    eprintln!("cuda↔cpu (y-flipped) Pearson = {r_cpu:.5}");
    assert!(r_cpu > 0.99, "CUDA FBP disagrees with CPU FBP: r = {r_cpu:.5}");

    // Round-trip: CUDA reconstruction recovers the phantom (best of both flips).
    let r_phantom = pearson_disk(&cuda_slice, &phantom, n, 0.85)
        .max(pearson_disk(&flipud(&cuda_slice), &phantom, n, 0.85));
    eprintln!("cuda↔phantom Pearson = {r_phantom:.5}");
    assert!(r_phantom > 0.85, "CUDA FBP recovers phantom poorly: r = {r_phantom:.5}");
}

#[test]
fn cuda_fourierrec_matches_cpu_and_phantom() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (n, nang) = (128usize, 180usize);
    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };
    // cfunc_fourierrec pairs slices into complex, so it needs an even count.
    let nz = 4usize;
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = ndarray::Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let vol = tomoxide::Volume::new(stack);
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, nz, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let cpu_rec = recon::recon(&sino, &geom, Algorithm::Fourierrec, &params, &cpu).unwrap();
    let cuda_rec = recon::recon(&sino, &geom, Algorithm::Fourierrec, &params, &cuda).unwrap();
    let mid = nz / 2;
    let cpu_slice = cpu_rec.array.index_axis(Axis(0), mid).to_owned();
    let cuda_slice = cuda_rec.array.index_axis(Axis(0), mid).to_owned();
    assert_eq!(cuda_slice.dim(), (n, n));

    // GPU cfunc_fourierrec vs CPU fourierrec: same central-slice-theorem method,
    // possibly differing by the grid handedness (scale/flip-invariant compare).
    let r_cpu = pearson_disk(&cuda_slice, &cpu_slice, n, 0.85)
        .max(pearson_disk(&flipud(&cuda_slice), &cpu_slice, n, 0.85));
    eprintln!("cuda↔cpu fourierrec Pearson = {r_cpu:.5}");
    assert!(r_cpu > 0.97, "CUDA fourierrec disagrees with CPU: r = {r_cpu:.5}");

    let r_phantom = pearson_disk(&cuda_slice, &phantom, n, 0.85)
        .max(pearson_disk(&flipud(&cuda_slice), &phantom, n, 0.85));
    eprintln!("cuda↔phantom fourierrec Pearson = {r_phantom:.5}");
    assert!(r_phantom > 0.9, "CUDA fourierrec recovers phantom poorly: r = {r_phantom:.5}");
}
