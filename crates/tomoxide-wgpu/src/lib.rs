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
    // Capability accessors stay `None` until the WGSL kernels land in M6.
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
}
