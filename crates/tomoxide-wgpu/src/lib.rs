//! # tomoxide-wgpu
//!
//! A portable GPU backend built on [`wgpu`], so the GPU reconstruction path
//! runs on hardware CUDA can't target — notably **Metal** on Apple Silicon, and
//! Vulkan/DX12 elsewhere. Kernels are WGSL ports of the CUDA kernels (see
//! [`shaders`]).
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

use tomoxide_core::backend::{Backend, DeviceKind};
use tomoxide_core::dtype::Dtype;
use tomoxide_core::error::{Error, Result};

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
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("tomoxide-wgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
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
    fn elementwise(&self) -> Option<&dyn tomoxide_core::backend::Elementwise> {
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
    use ndarray::Array3;
    use tomoxide_core::backend::Backend;
    use tomoxide_core::data::{Frames, Layout, Tomo};

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
        let cpu = tomoxide_cpu::CpuBackend;

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
        let cpu = tomoxide_cpu::CpuBackend;

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
        let cpu = tomoxide_cpu::CpuBackend;

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
}
