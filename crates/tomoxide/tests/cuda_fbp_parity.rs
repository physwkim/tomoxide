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
use tomoxide::{
    recon, sim, Algorithm, Angles, CpuBackend, CudaBackend, Dtype, FilterName, Geometry,
    ReconParams,
};

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
        // Pin ramp: these analytic parity/phantom checks are calibrated for the
        // sharp filter (the default is parzen).
        filter_name: FilterName::Ramp,
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

    // CUDA vs CPU back-projector: same orientation now, identical up to scale.
    let r_cpu = pearson_disk(&cuda_slice, &cpu_slice, n, 0.85);
    eprintln!("cuda↔cpu Pearson = {r_cpu:.5}");
    assert!(
        r_cpu > 0.99,
        "CUDA FBP disagrees with CPU FBP: r = {r_cpu:.5}"
    );

    // Round-trip: CUDA reconstruction recovers the phantom (best of both flips).
    let r_phantom = pearson_disk(&cuda_slice, &phantom, n, 0.85).max(pearson_disk(
        &flipud(&cuda_slice),
        &phantom,
        n,
        0.85,
    ));
    eprintln!("cuda↔phantom Pearson = {r_phantom:.5}");
    assert!(
        r_phantom > 0.85,
        "CUDA FBP recovers phantom poorly: r = {r_phantom:.5}"
    );
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
        // Pin ramp: these analytic parity/phantom checks are calibrated for the
        // sharp filter (the default is parzen).
        filter_name: FilterName::Ramp,
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
    let r_cpu = pearson_disk(&cuda_slice, &cpu_slice, n, 0.85).max(pearson_disk(
        &flipud(&cuda_slice),
        &cpu_slice,
        n,
        0.85,
    ));
    eprintln!("cuda↔cpu fourierrec Pearson = {r_cpu:.5}");
    assert!(
        r_cpu > 0.97,
        "CUDA fourierrec disagrees with CPU: r = {r_cpu:.5}"
    );

    let r_phantom = pearson_disk(&cuda_slice, &phantom, n, 0.85).max(pearson_disk(
        &flipud(&cuda_slice),
        &phantom,
        n,
        0.85,
    ));
    eprintln!("cuda↔phantom fourierrec Pearson = {r_phantom:.5}");
    assert!(
        r_phantom > 0.9,
        "CUDA fourierrec recovers phantom poorly: r = {r_phantom:.5}"
    );
}

#[test]
fn cuda_fbp_filter_matches_cpu() {
    use tomoxide::backend::Backend;
    use tomoxide::{Layout, Tomo};
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (n, nang, nz) = (128usize, 90usize, 3usize);
    // A textured sinogram (off-centre rotation axis to exercise the phase).
    let sino = ndarray::Array3::from_shape_fn((nz, nang, n), |(z, a, x)| {
        (1.0 + 0.5 * ((a * 5 + x * 3 + z) as f32 * 0.013).sin()) + 0.1 * z as f32
    });
    let mut geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, nz, 1.0);
    geom.center = tomoxide::Center::Scalar(n as f32 / 2.0 - 3.5); // off-centre

    let kernel = cpu
        .fbp_filter()
        .unwrap()
        .make_filter(FilterName::Ramp, n)
        .unwrap();

    let mut t_cpu = Tomo::new(sino.clone(), Layout::Sinogram);
    let mut t_cuda = Tomo::new(sino.clone(), Layout::Sinogram);
    cpu.fbp_filter()
        .unwrap()
        .apply(&mut t_cpu, &kernel, &geom)
        .unwrap();
    cuda.fbp_filter()
        .unwrap()
        .apply(&mut t_cuda, &kernel, &geom)
        .unwrap();

    // Since the convention unification (Phase 2) the CUDA analytic filter carries
    // the same net gain as tomopy (the CPU path) — the tomocupy ½ was removed from
    // `build_filter_w`. So `cuda == cpu`; compare the filter *shape* directly to the
    // cuFFT/rustfft round-off floor. See `cuda/mod.rs::build_filter_w`.
    let scale = t_cpu.array.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_abs = t_cpu
        .array
        .iter()
        .zip(t_cuda.array.iter())
        .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
    eprintln!(
        "fbp filter max|Δ| (cuda vs cpu) = {max_abs:e} (rel {:e})",
        max_abs / scale
    );
    // cuFFT vs rustfft: f32 FFT round-off floor.
    assert!(
        max_abs / scale < 1e-4,
        "GPU filter ≠ CPU filter: rel {}",
        max_abs / scale
    );
}

#[test]
fn cuda_fused_equals_per_stage() {
    // The fused on-device analytic path (recon → analytic_reconstruct) reconstructs
    // the same volume as composing the per-capability stages (filter then
    // back-project) — same kernels, same data, only the intermediate stays on the
    // device. The match is to the single-precision FFT floor, not bit-exact: on a
    // multi-GPU host the fused path splits the z-stack across devices, so its cuFFT
    // filter batch (nz_chunk·nproj) is smaller than the per-stage whole-stack batch
    // and cuFFT picks a different algorithm per batch size (documented in the cuda
    // module). The tolerance sits ~5 orders below the output scale, so the all-zero
    // regression that a <2-slice linerec chunk produces still trips it.
    use tomoxide::backend::Backend;
    use tomoxide::{Layout, Tomo, Volume};
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let (n, nang, nz) = (96usize, 72usize, 4usize);
    let params = ReconParams {
        num_gridx: Some(n),
        // Pin ramp: these analytic parity/phantom checks are calibrated for the
        // sharp filter (the default is parzen).
        filter_name: FilterName::Ramp,
        ..Default::default()
    };
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = ndarray::Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, nz, 1.0);
    let sino = sim::project(&Volume::new(stack), &geom, &CpuBackend::new()).unwrap();

    // Fused (device-resident) path.
    let fused = recon::recon(&sino, &geom, Algorithm::Fbp, &params, &cuda).unwrap();

    // Per-stage path through the same GPU capabilities.
    let kernel = cuda
        .fbp_filter()
        .unwrap()
        .make_filter(FilterName::Ramp, n)
        .unwrap();
    let mut filtered = Tomo::new(sino.array.clone(), Layout::Sinogram);
    cuda.fbp_filter()
        .unwrap()
        .apply(&mut filtered, &kernel, &geom)
        .unwrap();
    let mut vol = Volume::new(ndarray::Array3::zeros((nz, n, n)));
    cuda.backprojector()
        .unwrap()
        .backproject(&filtered, &geom, &mut vol)
        .unwrap();

    let maxabs = vol.array.iter().fold(0.0f32, |m, &b| m.max(b.abs()));
    let max_d = fused
        .array
        .iter()
        .zip(vol.array.iter())
        .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
    eprintln!("fused vs per-stage max|Δ| = {max_d:e}  (max|val| = {maxabs:e})");
    // Per-stage must itself be a real reconstruction (guards against both paths
    // degenerating to zeros, which would make the diff trivially pass).
    assert!(
        maxabs > 1e-3,
        "per-stage reconstruction is degenerate: max|val| = {maxabs:e}"
    );
    assert!(
        max_d <= 1e-4 * maxabs,
        "fused device path differs from per-stage by {max_d:e} (> 1e-4·{maxabs:e}) — \
         an all-zero or wrong-path regression, not the FFT floor"
    );
}

/// Build the stacked-phantom sinogram shared by the f16 parity tests.
fn f16_phantom_sino(
    cpu: &CpuBackend,
    n: usize,
    nz: usize,
    nang: usize,
) -> (Array2<f32>, Geometry, tomoxide::Tomo<f32>) {
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = ndarray::Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let vol = tomoxide::Volume::new(stack);
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, nz, 1.0);
    let sino = sim::project(&vol, &geom, cpu).unwrap();
    (phantom, geom, sino)
}

#[test]
fn cuda_fbp_f16_matches_f32_and_phantom() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    // pad = (4·n).next_power_of_two() ⇒ the half cuFFT's power-of-two width holds.
    let (n, nang, nz) = (128usize, 180usize, 4usize);
    let (phantom, geom, sino) = f16_phantom_sino(&cpu, n, nz, nang);

    let p32 = ReconParams {
        num_gridx: Some(n),
        // Pin ramp: these analytic parity/phantom checks are calibrated for the
        // sharp filter (the default is parzen).
        filter_name: FilterName::Ramp,
        ..Default::default()
    };
    let p16 = ReconParams {
        num_gridx: Some(n),
        dtype: Dtype::F16,
        filter_name: FilterName::Ramp,
        ..Default::default()
    };
    let rec32 = recon::recon(&sino, &geom, Algorithm::Fbp, &p32, &cuda).unwrap();
    let rec16 = recon::recon(&sino, &geom, Algorithm::Fbp, &p16, &cuda).unwrap();
    let mid = nz / 2;
    let s32 = rec32.array.index_axis(Axis(0), mid).to_owned();
    let s16 = rec16.array.index_axis(Axis(0), mid).to_owned();
    assert_eq!(s16.dim(), (n, n));

    // f16 keeps the same geometry/scale as f32 — straight Pearson, no flip.
    let r = pearson_disk(&s16, &s32, n, 0.85);
    eprintln!("cuda FBP f16↔f32 Pearson = {r:.5}");
    assert!(r > 0.99, "f16 FBP disagrees with f32 FBP: r = {r:.5}");

    // And it still recovers the phantom (half precision is approximate but the
    // structure must survive).
    let r_ph =
        pearson_disk(&s16, &phantom, n, 0.85).max(pearson_disk(&flipud(&s16), &phantom, n, 0.85));
    eprintln!("cuda FBP f16↔phantom Pearson = {r_ph:.5}");
    assert!(
        r_ph > 0.85,
        "f16 FBP recovers phantom poorly: r = {r_ph:.5}"
    );
}

#[test]
fn cuda_fourierrec_f16_matches_f32_and_phantom() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (n, nang, nz) = (128usize, 180usize, 4usize);
    let (phantom, geom, sino) = f16_phantom_sino(&cpu, n, nz, nang);

    let p32 = ReconParams {
        num_gridx: Some(n),
        // Pin ramp: these analytic parity/phantom checks are calibrated for the
        // sharp filter (the default is parzen).
        filter_name: FilterName::Ramp,
        ..Default::default()
    };
    let p16 = ReconParams {
        num_gridx: Some(n),
        dtype: Dtype::F16,
        filter_name: FilterName::Ramp,
        ..Default::default()
    };
    let rec32 = recon::recon(&sino, &geom, Algorithm::Fourierrec, &p32, &cuda).unwrap();
    let rec16 = recon::recon(&sino, &geom, Algorithm::Fourierrec, &p16, &cuda).unwrap();
    let mid = nz / 2;
    let s32 = rec32.array.index_axis(Axis(0), mid).to_owned();
    let s16 = rec16.array.index_axis(Axis(0), mid).to_owned();
    assert_eq!(s16.dim(), (n, n));

    let r = pearson_disk(&s16, &s32, n, 0.85).max(pearson_disk(&flipud(&s16), &s32, n, 0.85));
    eprintln!("cuda fourierrec f16↔f32 Pearson = {r:.5}");
    assert!(r > 0.97, "f16 fourierrec disagrees with f32: r = {r:.5}");

    let r_ph =
        pearson_disk(&s16, &phantom, n, 0.85).max(pearson_disk(&flipud(&s16), &phantom, n, 0.85));
    eprintln!("cuda fourierrec f16↔phantom Pearson = {r_ph:.5}");
    assert!(
        r_ph > 0.9,
        "f16 fourierrec recovers phantom poorly: r = {r_ph:.5}"
    );
}
