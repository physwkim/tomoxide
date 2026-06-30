//! Regression test for the wgpu 1-D dispatch overflow.
//!
//! WebGPU caps each `dispatch_workgroups` dimension at 65535, so a flat 1-D
//! launch blows the validation limit once a kernel needs more than
//! `65535 · WORKGROUP` (= 16,776,960) threads — reached by any whole-volume
//! reconstruction (e.g. a 512² back-projection over ≥64 slices). The fix folds
//! the workgroup count into a 2-D `(wx, wy)` grid and every kernel recovers its
//! flat index as `gid.y · num_workgroups.x · WG + gid.x`. These tests drive two
//! representative kernels — a pure elementwise pass and the multi-index
//! back-projector — past the cap and assert the GPU result still matches the CPU
//! reference; before the fix they aborted with a dispatch-size validation error.
//!
//! Only built under `gpu-wgpu`; needs a real GPU adapter (skipped by the default
//! workspace run). Run: `cargo test -p tomoxide --features gpu-wgpu`.
#![cfg(feature = "gpu-wgpu")]

use ndarray::Array3;
use tomoxide::wgpu::WgpuBackend;
use tomoxide::{Angles, Backend, CpuBackend, Geometry, Layout, Tomo, Volume};

#[test]
fn minus_log_above_dispatch_cap_matches_cpu() {
    let cpu = CpuBackend::new();
    let gpu = WgpuBackend::new().expect("wgpu device init");

    // Shape is irrelevant to minus_log (flat elementwise); only the element
    // count matters. 4040 · 4150 = 16,766,000 … bump rows to clear the cap.
    let (nz, nang, ncols) = (200usize, 4040usize, 22usize);
    let total = nz * nang * ncols;
    assert!(
        total > 16_776_960,
        "test must exceed the 1-D dispatch cap, got {total}"
    );

    // Mixed positive/tiny/non-finite-producing inputs so clamping and the
    // finite-guard both run, exactly as on the CPU path.
    let data: Vec<f32> = (0..total)
        .map(|i| match i % 5 {
            0 => 0.0,  // → clamped to 1e-6 → -log(1e-6)
            1 => 1e-9, // → clamped
            2 => 1.0,  // → 0.0
            3 => 2.5,  // → -ln(2.5)
            _ => 0.37,
        })
        .collect();
    let arr = Array3::from_shape_vec((nz, nang, ncols), data).unwrap();

    let mut tc = Tomo::new(arr.clone(), Layout::Projection);
    let mut tg = Tomo::new(arr, Layout::Projection);
    cpu.elementwise().unwrap().minus_log(&mut tc).unwrap();
    gpu.elementwise().unwrap().minus_log(&mut tg).unwrap();

    let (sc, sg) = (tc.array.as_slice().unwrap(), tg.array.as_slice().unwrap());
    let mut maxabs = 0.0f32;
    for (&c, &g) in sc.iter().zip(sg) {
        maxabs = maxabs.max((c - g).abs());
    }
    eprintln!("minus_log({total} elems) GPU vs CPU max|Δ| = {maxabs:.3e}");
    // GPU `log` differs from libm by a few ULP; values are O(1)–O(14), so 1e-5
    // is generous yet ~1000× tighter than any index-recovery bug (which would
    // scramble whole rows) would produce.
    assert!(
        maxabs < 1e-5,
        "minus_log past dispatch cap diverged: max|Δ| = {maxabs:.3e}"
    );
}

#[test]
fn backproject_above_dispatch_cap_matches_cpu() {
    let cpu = CpuBackend::new();
    let gpu = WgpuBackend::new().expect("wgpu device init");

    // total back-projection threads = nz · ny · nx. With ny = nx = 128 a single
    // angle keeps per-thread work trivial, and nz = 1100 → 18,022,400 threads,
    // just past the 16,776,960 cap. The volume side is the index decomposition
    // (flat → iz,iy,ix) that the 2-D fold must keep correct.
    let (nz, n, nang) = (1100usize, 128usize, 1usize);
    let total = nz * n * n;
    assert!(
        total > 16_776_960,
        "test must exceed the 1-D dispatch cap, got {total}"
    );

    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, nz, 1.0);

    // A non-trivial but cheap sinogram: a ramp across columns, varying per row so
    // the per-row index decomposition is actually exercised (a constant sino
    // would mask an iz mix-up).
    let mut sino = Array3::<f32>::zeros((nz, nang, n));
    for iz in 0..nz {
        for ic in 0..n {
            sino[[iz, 0, ic]] = (ic as f32 - n as f32 / 2.0) + (iz % 7) as f32;
        }
    }
    let sino = Tomo::new(sino, Layout::Sinogram);

    let mut vc = Volume::new(Array3::zeros((nz, n, n)));
    let mut vg = Volume::new(Array3::zeros((nz, n, n)));
    cpu.backprojector()
        .unwrap()
        .backproject(&sino, &geom, &mut vc)
        .unwrap();
    gpu.backprojector()
        .unwrap()
        .backproject(&sino, &geom, &mut vg)
        .unwrap();

    let mut maxabs = 0.0f32;
    let mut sumref = 0.0f64;
    for (c, g) in vc.array.iter().zip(vg.array.iter()) {
        maxabs = maxabs.max((c - g).abs());
        sumref += (*c as f64).abs();
    }
    let meanref = sumref / total as f64;
    eprintln!(
        "backproject({total} threads) GPU vs CPU max|Δ| = {maxabs:.3e}, mean|ref| = {meanref:.3e}"
    );
    // Only multiply-accumulate rounding differs; with a single angle and O(64)
    // magnitude values, 1e-3 absolute leaves huge headroom yet a scrambled index
    // (rows summed into the wrong slice) would diverge by O(meanref) ≫ 1e-3.
    assert!(
        maxabs < 1e-3,
        "backproject past dispatch cap diverged: max|Δ| = {maxabs:.3e}"
    );
    // Sanity: the reference is non-trivial, so the bar above is meaningful.
    assert!(
        meanref > 1.0,
        "reference back-projection unexpectedly empty"
    );
}
