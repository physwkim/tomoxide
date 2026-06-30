//! # tomoxide-wgpu
//!
//! A portable GPU backend built on `wgpu`, so the GPU reconstruction path
//! runs on hardware CUDA can't target — notably **Metal** on Apple Silicon, and
//! Vulkan/DX12 elsewhere. Kernels are WGSL ports of the CUDA kernels (see
//! `shaders`).
//!
//! Gated behind the **`gpu-wgpu`** feature because `wgpu` is a heavy
//! dependency; the default workspace build skips it and
//! [`WgpuBackend::new`] reports the backend as unavailable.
#![cfg_attr(not(feature = "gpu-wgpu"), allow(dead_code))]

#[cfg(feature = "gpu-wgpu")]
pub mod shaders;

#[cfg(feature = "gpu-wgpu")]
mod compute;
#[cfg(feature = "gpu-wgpu")]
mod kernels;

use crate::backend::{Backend, DeviceKind};
use crate::dtype::Dtype;
use crate::error::{Error, Result};

/// Handle to the portable GPU backend.
#[cfg(feature = "gpu-wgpu")]
pub struct WgpuBackend {
    /// The logical device.
    pub device: wgpu::Device,
    /// The command queue.
    pub queue: wgpu::Queue,
}

/// Handle to the portable GPU backend (stub: compiled without `gpu-wgpu`).
#[cfg(not(feature = "gpu-wgpu"))]
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuBackend;

impl WgpuBackend {
    /// Initialise a wgpu device, picking a high-performance adapter.
    ///
    /// Without the `gpu-wgpu` feature this returns [`Error::BackendUnavailable`].
    #[cfg(not(feature = "gpu-wgpu"))]
    pub fn new() -> Result<Self> {
        Err(Error::BackendUnavailable(
            "compiled without the `gpu-wgpu` feature".into(),
        ))
    }

    /// Initialise a wgpu device, picking a high-performance adapter (Metal on
    /// Apple Silicon, Vulkan/DX12 elsewhere). Blocks on the async wgpu requests
    /// via `pollster`; returns [`Error::BackendUnavailable`] if no adapter or
    /// device can be acquired.
    #[cfg(feature = "gpu-wgpu")]
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    #[cfg(feature = "gpu-wgpu")]
    async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::BackendUnavailable("no wgpu adapter available".into()))?;
        // `Limits::default()` is the conservative downlevel/WebGL profile
        // (256 MiB max buffer, 128 MiB max storage binding). A whole-volume
        // reconstruction's filter/FFT buffers blow past that (e.g. a 512²
        // nz=128 FBP needs a ~1 GiB spectrum buffer), so request the adapter's
        // reported maxima — the real hardware capability — instead.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("tomoxide-wgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| Error::BackendUnavailable(format!("wgpu request_device: {e}")))?;
        Ok(Self { device, queue })
    }
}

impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }
    fn device(&self) -> DeviceKind {
        DeviceKind::Wgpu
    }
    fn supports(&self, dt: Dtype) -> bool {
        // f16 needs the `shader-f16` device feature; advertise f32 for now.
        dt == Dtype::F32
    }

    /// Elementwise preprocessing (dark/flat correction, `−ln`) runs on the GPU.
    #[cfg(feature = "gpu-wgpu")]
    fn elementwise(&self) -> Option<&dyn crate::backend::Elementwise> {
        Some(self)
    }

    /// Parallel-beam filtered back-projection runs on the GPU.
    #[cfg(feature = "gpu-wgpu")]
    fn backprojector(&self) -> Option<&dyn crate::backend::FilteredBackproject> {
        Some(self)
    }

    /// Parallel-beam forward projection (Radon transform) runs on the GPU.
    #[cfg(feature = "gpu-wgpu")]
    fn projector(&self) -> Option<&dyn crate::backend::ForwardProject> {
        Some(self)
    }

    /// 3-D median / dezinger rank filters run on the GPU (bit-exact with CPU).
    #[cfg(feature = "gpu-wgpu")]
    fn rank_filter(&self) -> Option<&dyn crate::backend::RankFilter> {
        Some(self)
    }

    /// Batched radix-2 FFT (power-of-two lengths) runs on the GPU.
    #[cfg(feature = "gpu-wgpu")]
    fn fft(&self) -> Option<&dyn crate::backend::Fft> {
        Some(self)
    }

    /// FBP apodization-filter application (pad → FFT → ×filter → IFFT → crop)
    /// runs on the GPU, closing the full GPU filtered-back-projection path.
    #[cfg(feature = "gpu-wgpu")]
    fn fbp_filter(&self) -> Option<&dyn crate::backend::FbpFilter> {
        Some(self)
    }
    // Remaining capability accessors stay `None` until their WGSL kernels land.
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "gpu-wgpu"))]
    use super::*;

    #[test]
    #[cfg(not(feature = "gpu-wgpu"))]
    fn unavailable_without_feature() {
        assert!(matches!(
            WgpuBackend::new(),
            Err(Error::BackendUnavailable(_))
        ));
    }
}

/// GPU smoke test — requires a real adapter, so it only builds under
/// `gpu-wgpu` and is skipped by the default workspace test run. Run with:
/// `cargo test -p tomoxide-wgpu --features gpu-wgpu`.
#[cfg(all(test, feature = "gpu-wgpu"))]
mod gpu_tests {
    use super::*;
    use wgpu::util::DeviceExt;

    /// Trivial in-place doubling kernel — exercises the full compute path
    /// (device → shader module → storage buffer → dispatch → readback).
    const DOUBLE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&data)) { return; }
    data[i] = data[i] * 2.0;
}
"#;

    #[test]
    fn device_init_and_compute_roundtrip() {
        let be = WgpuBackend::new().expect("wgpu device init");
        let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = host.len();
        let bytes = (n * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

        let storage = be
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("data"),
                contents: bytemuck::cast_slice(&host),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let staging = be.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = be
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("double"),
                source: wgpu::ShaderSource::Wgsl(DOUBLE_WGSL.into()),
            });
        let pipeline = be
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("double"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bind_group = be.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage.as_entire_binding(),
            }],
        });
        let mut enc = be
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&storage, 0, &staging, 0, bytes);
        be.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        be.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();

        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    // --- Elementwise capability: parity vs the CPU backend ------------------
    // GPU f32 transcendentals/divisions differ from libm by a few ULP, so the
    // bar is a small relative+absolute tolerance, not bit-for-bit.
    use crate::backend::Backend;
    use crate::data::{Frames, Layout, Tomo, Volume};
    use crate::dtype::Complex32;
    use crate::geometry::{Angles, Center, Geometry};
    use ndarray::Array3;

    /// Assert two flat f32 sequences agree within a relative+absolute tolerance.
    fn assert_close(gpu: &[f32], cpu: &[f32], rtol: f32, atol: f32) {
        assert_eq!(gpu.len(), cpu.len(), "length mismatch");
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let tol = atol + rtol * c.abs();
            assert!(
                (g - c).abs() <= tol,
                "index {i}: gpu={g} cpu={c} |Δ|={} > tol={tol}",
                (g - c).abs()
            );
        }
    }

    fn ramp_tomo(np: usize, nr: usize, nc: usize, layout: Layout) -> Tomo<f32> {
        let n = np * nr * nc;
        let data: Vec<f32> = (0..n).map(|k| 0.5 + k as f32 * 0.01).collect();
        Tomo::new(Array3::from_shape_vec((np, nr, nc), data).unwrap(), layout)
    }

    #[test]
    fn minus_log_matches_cpu() {
        let be = WgpuBackend::new().expect("wgpu device init");
        let cpu = crate::cpu::CpuBackend;

        let base = ramp_tomo(3, 4, 5, Layout::Projection);
        let mut g = base.clone();
        let mut c = base.clone();
        be.elementwise().unwrap().minus_log(&mut g).unwrap();
        cpu.elementwise().unwrap().minus_log(&mut c).unwrap();

        assert_close(
            g.array.as_slice().unwrap(),
            c.array.as_slice().unwrap(),
            1e-5,
            1e-6,
        );
    }

    #[test]
    fn minus_log_scrubs_nonfinite_like_cpu() {
        let be = WgpuBackend::new().expect("wgpu device init");
        let cpu = crate::cpu::CpuBackend;

        // Values at/below the 1e-6 clamp drive -ln(...) large but finite; the
        // clamp floor itself (1e-6) gives the max magnitude. Mix in zeros and
        // negatives to exercise the clamp path on both backends identically.
        let data = vec![0.0f32, -3.0, 1e-9, 1.0, 2.5, 1e-7];
        let base = Tomo::new(
            Array3::from_shape_vec((1, 2, 3), data).unwrap(),
            Layout::Projection,
        );
        let mut g = base.clone();
        let mut c = base.clone();
        be.elementwise().unwrap().minus_log(&mut g).unwrap();
        cpu.elementwise().unwrap().minus_log(&mut c).unwrap();

        assert_close(
            g.array.as_slice().unwrap(),
            c.array.as_slice().unwrap(),
            1e-5,
            1e-5,
        );
    }

    #[test]
    fn darkflat_matches_cpu_projection_and_sinogram() {
        let be = WgpuBackend::new().expect("wgpu device init");
        let cpu = crate::cpu::CpuBackend;

        // 2 dark + 3 flat frames over a 4×5 plane; data is 3 projections.
        let dark = Frames::new(
            Array3::from_shape_vec(
                (2, 4, 5),
                (0..2 * 4 * 5).map(|k| 0.1 + k as f32 * 0.002).collect(),
            )
            .unwrap(),
        );
        let flat = Frames::new(
            Array3::from_shape_vec(
                (3, 4, 5),
                (0..3 * 4 * 5).map(|k| 1.0 + k as f32 * 0.003).collect(),
            )
            .unwrap(),
        );

        // Build a projection-layout base (3 projections over the 4×5 detector
        // plane) and derive the sinogram case by converting it, so the detector
        // plane stays consistent with the dark/flat planes in both layouts.
        let proj_base = ramp_tomo(3, 4, 5, Layout::Projection);
        for layout in [Layout::Projection, Layout::Sinogram] {
            let base = proj_base.to_layout(layout);
            let mut g = base.clone();
            let mut c = base.clone();
            be.elementwise()
                .unwrap()
                .darkflat(&mut g, &flat, &dark)
                .unwrap();
            cpu.elementwise()
                .unwrap()
                .darkflat(&mut c, &flat, &dark)
                .unwrap();

            assert_eq!(g.layout, c.layout, "layout preserved");
            assert_close(
                g.array.as_standard_layout().as_slice().unwrap(),
                c.array.as_standard_layout().as_slice().unwrap(),
                1e-5,
                1e-6,
            );
        }
    }

    // --- FilteredBackproject capability: parity vs the CPU backend ----------
    #[test]
    fn backproject_matches_cpu() {
        let be = WgpuBackend::new().expect("wgpu device init");
        let cpu = crate::cpu::CpuBackend;

        // [row, angle, col] sinogram with structure: 2 rows, 6 angles, 8 cols.
        let (nz, nang, ncols) = (2usize, 6usize, 8usize);
        let sarr = Array3::from_shape_fn((nz, nang, ncols), |(z, a, col)| {
            ((z * 3 + a) as f32 * 0.3 + col as f32 * 0.17).sin() + 0.05 * col as f32
        });
        let s = Tomo::new(sarr, Layout::Sinogram);
        let mut geom = Geometry::parallel(
            Angles::uniform(nang, 0.0, std::f32::consts::PI),
            ncols,
            nz,
            1.0,
        );

        // Reconstruct an interior 4×4 ROI inside the 8-wide detector (the
        // physically sensible field-of-view): every ray maps to t ∈ ~[0.7, 6.8],
        // safely away from the detector edges 0 and ncols−1. That keeps GPU/CPU
        // divergence to pure multiply-accumulate rounding. A full-detector-width
        // grid lets corner rays graze the hard inclusion cutoff (t≈0 or
        // t≈ncols−1), where a sub-ULP t difference flips a whole edge sample —
        // a legitimate discontinuity, not a rounding error.
        let recon = 4usize;
        // Exercise both center buffer paths: scalar (default 4.0) and per-row.
        for center in [geom.center.clone(), Center::PerRow(vec![3.5, 4.0])] {
            geom.center = center;
            let mut g = Volume::new(Array3::<f32>::zeros((nz, recon, recon)));
            let mut c = Volume::new(Array3::<f32>::zeros((nz, recon, recon)));
            be.backprojector()
                .unwrap()
                .backproject(&s, &geom, &mut g)
                .unwrap();
            cpu.backprojector()
                .unwrap()
                .backproject(&s, &geom, &mut c)
                .unwrap();

            assert_eq!(g.array.dim(), c.array.dim());
            assert_close(
                g.array.as_slice().unwrap(),
                c.array.as_slice().unwrap(),
                1e-4,
                1e-5,
            );
        }
    }

    // --- ForwardProject capability: parity vs the CPU backend --------------
    #[test]
    fn forward_project_matches_cpu() {
        let be = WgpuBackend::new().expect("wgpu device init");
        let cpu = crate::cpu::CpuBackend;

        // Interior 4×4×2 volume projected onto an 8-wide / 6-angle detector. The
        // small grid keeps every pixel's t inside (0, ncols−1), away from the
        // hard splat-inclusion cutoff (same edge-grazing hazard as backproject).
        let (nz, ny, nx) = (2usize, 4usize, 4usize);
        let (nang, ncols) = (6usize, 8usize);
        let varr = Array3::from_shape_fn((nz, ny, nx), |(z, y, x)| {
            0.3 + z as f32 * 0.5 + y as f32 * 0.11 + x as f32 * 0.07
        });
        let v = Volume::new(varr);
        let mut geom = Geometry::parallel(
            Angles::uniform(nang, 0.0, std::f32::consts::PI),
            ncols,
            nz,
            1.0,
        );

        // Exercise both center buffer paths: scalar (default 4.0) and per-row.
        for center in [geom.center.clone(), Center::PerRow(vec![3.5, 4.0])] {
            geom.center = center;
            let mut g = Tomo::new(Array3::<f32>::zeros((nz, nang, ncols)), Layout::Sinogram);
            let mut c = Tomo::new(Array3::<f32>::zeros((nz, nang, ncols)), Layout::Sinogram);
            be.projector().unwrap().project(&v, &geom, &mut g).unwrap();
            cpu.projector().unwrap().project(&v, &geom, &mut c).unwrap();

            assert_eq!(g.layout, Layout::Sinogram);
            assert_eq!(g.array.dim(), c.array.dim());
            assert_close(
                g.array.as_slice().unwrap(),
                c.array.as_slice().unwrap(),
                1e-4,
                1e-5,
            );
        }
    }

    // --- RankFilter capability: BIT-EXACT parity vs the CPU backend --------
    // Pure gather + order statistic + one subtraction → no rounding divergence.
    #[test]
    fn median3d_and_remove_outlier3d_match_cpu_bit_exact() {
        let be = WgpuBackend::new().expect("wgpu device init");
        let cpu = crate::cpu::CpuBackend;

        // Volume with a planted zinger spike so the dezinger threshold path hits.
        let (dz, dy, dx) = (4usize, 5usize, 6usize);
        let mut varr = Array3::from_shape_fn((dz, dy, dx), |(z, y, x)| {
            ((z * 7 + y * 3 + x) as f32 * 0.13).sin() + 0.2 * x as f32
        });
        varr[[2, 2, 3]] = 99.0; // outlier

        // median3d (threshold 0): plain local median everywhere.
        let mut g = Volume::new(varr.clone());
        let mut c = Volume::new(varr.clone());
        be.rank_filter().unwrap().median3d(&mut g, 3).unwrap();
        cpu.rank_filter().unwrap().median3d(&mut c, 3).unwrap();
        assert_eq!(g.array, c.array, "median3d not bit-exact");

        // remove_outlier3d (threshold 1.0): only the spike is replaced.
        let mut g = Tomo::new(varr.clone(), Layout::Projection);
        let mut c = Tomo::new(varr.clone(), Layout::Projection);
        be.rank_filter()
            .unwrap()
            .remove_outlier3d(&mut g, 1.0, 3)
            .unwrap();
        cpu.rank_filter()
            .unwrap()
            .remove_outlier3d(&mut c, 1.0, 3)
            .unwrap();
        assert_eq!(g.array, c.array, "remove_outlier3d not bit-exact");
    }

    #[test]
    fn median3d_rejects_window_over_cap() {
        let be = WgpuBackend::new().expect("wgpu device init");
        // size 9 → diameter 9 → 729 voxels > the 343 GPU window cap.
        let mut vol = Volume::new(Array3::<f32>::zeros((9, 9, 9)));
        assert!(matches!(
            be.rank_filter().unwrap().median3d(&mut vol, 9),
            Err(crate::error::Error::InvalidParam(_))
        ));
    }

    // --- Fft capability: tolerance parity vs CPU (rustfft) + roundtrip -----
    fn assert_complex_close(g: &[Complex32], c: &[Complex32], tol: f32, what: &str) {
        assert_eq!(g.len(), c.len());
        for (i, (a, b)) in g.iter().zip(c.iter()).enumerate() {
            assert!(
                (a.re - b.re).abs() <= tol && (a.im - b.im).abs() <= tol,
                "{what} index {i}: gpu={a:?} cpu={b:?}"
            );
        }
    }

    #[test]
    fn fft_1d_roundtrips_and_matches_cpu() {
        let be = WgpuBackend::new().expect("wgpu device init");
        let cpu = crate::cpu::CpuBackend;
        let (len, batch) = (8usize, 3usize);
        let base: Vec<Complex32> = (0..len * batch)
            .map(|k| Complex32::new((k as f32 * 0.3).sin(), (k as f32 * 0.17).cos()))
            .collect();

        // Forward parity vs rustfft.
        let mut g = base.clone();
        let mut c = base.clone();
        be.fft().unwrap().fft_1d(&mut g, len, batch, false).unwrap();
        cpu.fft()
            .unwrap()
            .fft_1d(&mut c, len, batch, false)
            .unwrap();
        assert_complex_close(&g, &c, 1e-3, "fft_1d forward");

        // Roundtrip: ifft(fft(x)) == x.
        be.fft().unwrap().fft_1d(&mut g, len, batch, true).unwrap();
        assert_complex_close(&g, &base, 1e-3, "fft_1d roundtrip");
    }

    #[test]
    fn fft_2d_roundtrips_and_matches_cpu() {
        let be = WgpuBackend::new().expect("wgpu device init");
        let cpu = crate::cpu::CpuBackend;
        let (rows, cols, batch) = (4usize, 8usize, 2usize);
        let base: Vec<Complex32> = (0..rows * cols * batch)
            .map(|k| Complex32::new((k as f32 * 0.21).sin(), (k as f32 * 0.09).cos()))
            .collect();

        let mut g = base.clone();
        let mut c = base.clone();
        be.fft()
            .unwrap()
            .fft_2d(&mut g, rows, cols, batch, false)
            .unwrap();
        cpu.fft()
            .unwrap()
            .fft_2d(&mut c, rows, cols, batch, false)
            .unwrap();
        assert_complex_close(&g, &c, 2e-3, "fft_2d forward");

        be.fft()
            .unwrap()
            .fft_2d(&mut g, rows, cols, batch, true)
            .unwrap();
        assert_complex_close(&g, &base, 2e-3, "fft_2d roundtrip");
    }

    #[test]
    fn fft_2d_non_power_of_two_matches_cpu() {
        // Non-power-of-two 2-D shapes take the separable fallback: a 1-D pass
        // along cols (radix-2 or Bluestein), a host transpose, a 1-D pass along
        // rows, and a transpose back. Each axis may itself be Bluestein, so this
        // exercises pow2×non-pow2 (4×6), non-pow2×pow2 (6×4), and non-pow2×
        // non-pow2 (3×5, 6×10). Must match rustfft's 2-D transform and round-trip.
        // Observed on Metal: forward rel ≈ 1e-6, roundtrip ≈ 1e-5.
        let be = WgpuBackend::new().expect("wgpu device init");
        let cpu = crate::cpu::CpuBackend;
        for &(rows, cols, batch) in &[(4usize, 6usize, 2usize), (6, 4, 1), (3, 5, 2), (6, 10, 1)] {
            let base: Vec<Complex32> = (0..rows * cols * batch)
                .map(|k| Complex32::new((k as f32 * 0.21).sin(), (k as f32 * 0.09).cos()))
                .collect();
            let mut g = base.clone();
            let mut c = base.clone();
            be.fft()
                .unwrap()
                .fft_2d(&mut g, rows, cols, batch, false)
                .unwrap();
            cpu.fft()
                .unwrap()
                .fft_2d(&mut c, rows, cols, batch, false)
                .unwrap();
            let peak = c.iter().map(|z| z.norm()).fold(0.0f32, f32::max).max(1.0);
            let err = g
                .iter()
                .zip(&c)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0f32, f32::max);
            be.fft()
                .unwrap()
                .fft_2d(&mut g, rows, cols, batch, true)
                .unwrap();
            let rt = g
                .iter()
                .zip(&base)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0f32, f32::max);
            eprintln!(
                "fft_2d {rows}×{cols}×{batch}: peak={peak:.3}, fwd rel={:.3e}, roundtrip max|Δ|={rt:.3e}",
                err / peak
            );
            assert!(
                err <= 1e-5 * peak,
                "fft_2d forward {rows}×{cols}: max|Δ|={err:.3e} > {:.3e}",
                1e-5 * peak
            );
            assert!(
                rt <= 1e-4,
                "fft_2d roundtrip {rows}×{cols}: max|Δ|={rt:.3e}"
            );
        }
    }

    #[test]
    fn fft_1d_bluestein_matches_cpu_for_non_power_of_two() {
        // Non-power-of-two lengths run the Bluestein chirp-z path on the GPU
        // (3 radix-2 FFTs of length m = next_pow2(2n−1) + chirp multiplies), so
        // they must match rustfft (which also uses Bluestein internally) and
        // round-trip cleanly. Forward error is taken relative to the spectrum's
        // peak magnitude. Observed on Metal: rel ≈ 1e-7, roundtrip ≈ 1e-6; the
        // 1e-5·peak / 1e-4 bars leave ~50× headroom yet are far tighter than a
        // chirp/index bug (wrong sign, broken kernel symmetry) would produce.
        let be = WgpuBackend::new().expect("wgpu device init");
        let cpu = crate::cpu::CpuBackend;
        for &(len, batch) in &[(3usize, 2usize), (5, 1), (6, 3), (12, 2), (100, 1)] {
            let base: Vec<Complex32> = (0..len * batch)
                .map(|k| Complex32::new((k as f32 * 0.3).sin(), (k as f32 * 0.17).cos()))
                .collect();
            let mut g = base.clone();
            let mut c = base.clone();
            be.fft().unwrap().fft_1d(&mut g, len, batch, false).unwrap();
            cpu.fft()
                .unwrap()
                .fft_1d(&mut c, len, batch, false)
                .unwrap();
            let peak = c.iter().map(|z| z.norm()).fold(0.0f32, f32::max).max(1.0);
            let err = g
                .iter()
                .zip(&c)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0f32, f32::max);
            // Round-trip: ifft(fft(x)) == x.
            be.fft().unwrap().fft_1d(&mut g, len, batch, true).unwrap();
            let rt = g
                .iter()
                .zip(&base)
                .map(|(a, b)| (a - b).norm())
                .fold(0.0f32, f32::max);
            eprintln!(
                "bluestein len={len}: peak={peak:.3}, fwd rel={:.3e}, roundtrip max|Δ|={rt:.3e}",
                err / peak
            );
            assert!(
                err <= 1e-5 * peak,
                "bluestein forward len={len}: max|Δ|={err:.3e} > {:.3e}",
                1e-5 * peak
            );
            assert!(rt <= 1e-4, "bluestein roundtrip len={len}: max|Δ|={rt:.3e}");
        }
    }

    // --- FBP filter apply: parity vs the CPU backend ------------------------
    // The whole pad → FFT → ×filter → IFFT → crop pipeline runs on the GPU; the
    // bar is a tolerance (f32 twiddles + accumulation order), not bit-exact.

    #[test]
    fn fbp_filter_apply_matches_cpu() {
        use crate::params::FilterName;
        let be = WgpuBackend::new().expect("wgpu device init");
        let cpu = crate::cpu::CpuBackend;

        // 6 detector lanes (3 angles × 2 rows) of width 16; the ramp filter
        // zero-pads each lane to 32 before transforming.
        let base = ramp_tomo(3, 2, 16, Layout::Sinogram);
        let geom = Geometry::parallel(Angles::uniform(3, 0.0, std::f32::consts::PI), 16, 2, 1.0);

        // CPU and GPU build the identical filter (shared make_fbp_filter).
        let filter = cpu
            .fbp_filter()
            .unwrap()
            .make_filter(FilterName::Ramp, 16)
            .unwrap();
        let gfilter = be
            .fbp_filter()
            .unwrap()
            .make_filter(FilterName::Ramp, 16)
            .unwrap();
        assert_eq!(filter, gfilter, "make_filter differs between backends");

        let mut g = base.clone();
        let mut c = base.clone();
        be.fbp_filter()
            .unwrap()
            .apply(&mut g, &filter, &geom)
            .unwrap();
        cpu.fbp_filter()
            .unwrap()
            .apply(&mut c, &filter, &geom)
            .unwrap();

        assert_close(
            g.array.as_slice().unwrap(),
            c.array.as_slice().unwrap(),
            2e-3,
            2e-3,
        );
    }

    #[test]
    fn fbp_filter_apply_rejects_non_power_of_two() {
        let be = WgpuBackend::new().expect("wgpu device init");
        // pad = 24 is ≥ ncols (8) but not a power of two: the radix-2 GPU FFT
        // cannot transform it, so apply must error rather than corrupt data.
        let mut sino = ramp_tomo(1, 1, 8, Layout::Sinogram);
        let filter = vec![1.0f32; 24];
        let geom = Geometry::parallel(Angles::uniform(1, 0.0, std::f32::consts::PI), 8, 1, 1.0);
        assert!(matches!(
            be.fbp_filter().unwrap().apply(&mut sino, &filter, &geom),
            Err(crate::error::Error::InvalidParam(_))
        ));
    }
}
