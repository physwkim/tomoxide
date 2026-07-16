//! CUDA `Fft` capability + the methods that compose from it (gridrec, lprec,
//! Paganin phase) running on the GPU vs the CPU backend.
//!
//! Implementing one cuFFT-backed `Fft` capability makes every Fft-composing
//! method work on CUDA through the backend-agnostic code — exactly as they
//! compose onto wgpu. cuFFT vs rustfft differ only by f32 FFT round-off, so the
//! GPU results match the CPU ones to a tight floor.
#![cfg(feature = "cuda")]

use ndarray::{Array2, Array3, Axis};
use tomoxide::backend::Fft;
use tomoxide::{
    prep, recon, sim, Algorithm, Angles, CpuBackend, CudaBackend, Geometry, Layout, PhaseMethod,
    ReconParams, Tomo, Volume,
};

fn cuda_or_skip() -> Option<CudaBackend> {
    match CudaBackend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            None
        }
    }
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = a.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-12);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / scale
}

#[test]
fn cuda_fft_roundtrips() {
    let Some(cuda) = cuda_or_skip() else { return };
    // 1-D: ifft(fft(x)) == x.
    let orig: Vec<tomoxide::Complex32> = (0..16)
        .map(|k| tomoxide::Complex32::new(k as f32, -(k as f32) * 0.3))
        .collect();
    let mut buf = orig.clone();
    Fft::fft_1d(&cuda, &mut buf, 8, 2, false).unwrap();
    Fft::fft_1d(&cuda, &mut buf, 8, 2, true).unwrap();
    for (a, b) in buf.iter().zip(&orig) {
        assert!(
            (a.re - b.re).abs() < 1e-3 && (a.im - b.im).abs() < 1e-3,
            "fft_1d rt"
        );
    }
    // 2-D: ifft2(fft2(x)) == x.
    let orig2: Vec<tomoxide::Complex32> = (0..12)
        .map(|k| tomoxide::Complex32::new(k as f32, 1.0))
        .collect();
    let mut b2 = orig2.clone();
    cuda.fft_2d(&mut b2, 3, 4, 1, false).unwrap();
    cuda.fft_2d(&mut b2, 3, 4, 1, true).unwrap();
    for (a, b) in b2.iter().zip(&orig2) {
        assert!(
            (a.re - b.re).abs() < 1e-3 && (a.im - b.im).abs() < 1e-3,
            "fft_2d rt"
        );
    }
}

/// A batch too large for one plan's scratch is run in sub-batches, and doing so
/// changes nothing about the result.
///
/// cuFFT needs no scratch for a smooth length but falls back to Bluestein for
/// any other, where the scratch scales with `batch` — a laminography `2*rh` of
/// 5988 (= 2²·3·499) asked for 8.8 GiB and `cufftMakePlanMany` failed outright.
/// The transforms are contiguous, so splitting the batch is exact; this pins
/// that. `TOMOXIDE_CUFFT_MAX_WORK_MB` forces the split at a size that fits a
/// test — without it, reaching the real 512 MiB cap would take gigabytes, and
/// the test would pass while never touching the path it exists to cover.
#[test]
fn cuda_fft_split_batch_matches_unsplit() {
    let Some(cuda) = cuda_or_skip() else { return };
    let len = 4099usize; // prime and large enough that cuFFT goes Bluestein here
    let batch = 512usize;

    let split_of = |cap_mb: Option<&str>| -> i32 {
        match cap_mb {
            Some(v) => std::env::set_var("TOMOXIDE_CUFFT_MAX_WORK_MB", v),
            None => std::env::remove_var("TOMOXIDE_CUFFT_MAX_WORK_MB"),
        }
        unsafe { tomoxide::cuda::ffi::tomoxide_fft_plan_split(1, len as i32, 0, batch as i32, 0) }
    };

    // The cap is what decides: uncapped this shape runs in one plan, and at
    // 1 MiB it cannot. Assert both, so a cuFFT whose scratch no longer behaves
    // this way fails here rather than silently skipping the split below.
    assert_eq!(split_of(None), batch as i32, "unsplit baseline");
    let sub = split_of(Some("1"));
    assert!(
        sub < batch as i32 && sub >= 1,
        "cap of 1 MiB must split: {sub}"
    );

    let orig: Vec<tomoxide::Complex32> = (0..len * batch)
        .map(|k| {
            let x = k as f32 * 0.017;
            tomoxide::Complex32::new(x.sin(), (x * 0.6).cos())
        })
        .collect();

    let mut got = orig.clone();
    Fft::fft_1d(&cuda, &mut got, len, batch, false).unwrap(); // split (cap still 1 MiB)
    std::env::remove_var("TOMOXIDE_CUFFT_MAX_WORK_MB");
    let mut want = orig.clone();
    Fft::fft_1d(&cuda, &mut want, len, batch, false).unwrap(); // one plan

    let d = max_rel(
        &want.iter().flat_map(|c| [c.re, c.im]).collect::<Vec<_>>(),
        &got.iter().flat_map(|c| [c.re, c.im]).collect::<Vec<_>>(),
    );
    eprintln!("fft_1d split(sub={sub}) ↔ unsplit(batch={batch}) max rel = {d:e}");
    assert!(d < 1e-6, "split batch changed the transform: rel {d}");
}

fn sino(n: usize, nang: usize, nz: usize, cpu: &CpuBackend) -> (Tomo<f32>, Geometry, Array2<f32>) {
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, nz, 1.0);
    let s = sim::project(&Volume::new(stack), &geom, cpu).unwrap();
    (s, geom, phantom)
}

#[test]
fn cuda_gridrec_matches_cpu() {
    let Some(cuda) = cuda_or_skip() else { return };
    let cpu = CpuBackend::new();
    let (n, nang) = (128usize, 180usize);
    let (s, geom, _) = sino(n, nang, 1, &cpu);
    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };
    let rc = recon::recon(&s, &geom, Algorithm::Gridrec, &params, &cpu).unwrap();
    let rg = recon::recon(&s, &geom, Algorithm::Gridrec, &params, &cuda).unwrap();
    let d = max_rel(rc.array.as_slice().unwrap(), rg.array.as_slice().unwrap());
    eprintln!("gridrec cuda↔cpu max rel = {d:e}");
    assert!(d < 2e-3, "gridrec GPU≠CPU: rel {d}");
}

#[test]
fn cuda_lprec_matches_cpu() {
    let Some(cuda) = cuda_or_skip() else { return };
    let cpu = CpuBackend::new();
    let (n, nang) = (128usize, 180usize);
    let (s, geom, _) = sino(n, nang, 1, &cpu);
    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };
    let rc = recon::recon(&s, &geom, Algorithm::Lprec, &params, &cpu).unwrap();
    let rg = recon::recon(&s, &geom, Algorithm::Lprec, &params, &cuda).unwrap();
    // Since the convention unification lprec matches CPU directly (same scale, same
    // orientation — lprec never flipped). The only residual is the ramp *shape*: the
    // CPU backend ports tomopy (plain linear ramp) and the CUDA backend ports
    // tomocupy (the degree-12 `_wint` quadrature ramp); they diverge ~0.6% near
    // DC/Nyquist. This is the deliberate per-backend split (see `backend::RampShape`,
    // `cuda/mod.rs::build_filter_w`, `docs/ARCHITECTURE.md` §4.1), not a bug, so the
    // bar accommodates it. It still sits far below the gross-bug signatures this test
    // exists to catch (the pre-fix theta-order bug gave rel ≈ 1.0, the vertical-flip
    // bug ≈ 0.58); the legitimate ramp-shape residual is a deterministic 5.6e-3.
    let d = max_rel(rc.array.as_slice().unwrap(), rg.array.as_slice().unwrap());
    eprintln!("lprec cuda ↔ cpu max rel = {d:e}");
    assert!(
        d < 1.5e-2,
        "lprec GPU≠CPU (beyond the wint/linear ramp gap): rel {d}"
    );
}

#[test]
fn cuda_paganin_matches_cpu() {
    let Some(cuda) = cuda_or_skip() else { return };
    let cpu = CpuBackend::new();
    let (n, nang) = (128usize, 16usize);
    let (s, _geom, _) = sino(n, nang, 1, &cpu);
    let data = s.to_layout(Layout::Projection);
    let phase = PhaseMethod::Paganin {
        pixel_size: 1e-4,
        dist: 50.0,
        energy: 30.0,
        alpha: 1e-3,
    };

    let mut d_cpu = data.clone();
    let mut d_cuda = data.clone();
    prep::retrieve_phase(&mut d_cpu, phase, &cpu).unwrap();
    prep::retrieve_phase(&mut d_cuda, phase, &cuda).unwrap();
    let d = max_rel(
        d_cpu.array.as_slice().unwrap(),
        d_cuda.array.as_slice().unwrap(),
    );
    eprintln!("paganin cuda↔cpu max rel = {d:e}");
    assert!(d < 2e-3, "paganin GPU≠CPU: rel {d}");
}
