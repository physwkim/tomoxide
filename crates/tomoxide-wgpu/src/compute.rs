//! Small reusable wgpu compute primitives shared by the WGSL kernel ports:
//! buffer upload, single-pass 1-D dispatch (auto bind-group layout, workgroup
//! size 256), and host readback. Only compiled under `gpu-wgpu`.

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use crate::WgpuBackend;

/// Threads per workgroup for the 1-D dispatch helper (must match the
/// `@workgroup_size(256)` in every kernel dispatched through [`WgpuBackend::dispatch1d`]).
pub(crate) const WORKGROUP: u32 = 256;

impl WgpuBackend {
    /// A read/write storage buffer initialised from `data`, also usable as a
    /// `copy` source so its contents can be read back with [`Self::download_f32`].
    pub(crate) fn storage_rw(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// A read-only storage buffer initialised from `data`.
    pub(crate) fn storage_ro(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// A uniform buffer holding a single `Pod` value (pad it to a multiple of 16
    /// bytes so it satisfies the WGSL uniform layout rules).
    pub(crate) fn uniform<T: Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(value),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Compile `wgsl`, then dispatch its `entry` compute function over
    /// `n_threads` invocations, binding `buffers` at `@group(0)` in slice order.
    /// Uses the pipeline's auto-deduced bind-group layout, so each buffer's
    /// address space (`storage`/`uniform`, read vs read_write) is taken from the
    /// shader declaration.
    ///
    /// The dispatch launches `ceil(n_threads / WORKGROUP)` workgroups, so the
    /// kernel's `@workgroup_size` MUST equal [`WORKGROUP`] or threads at the tail
    /// go unlaunched. To make that impossible to get wrong, a `const WG` is
    /// injected into the shader source from [`WORKGROUP`]; every 1-D kernel
    /// declares `@workgroup_size(WG)` rather than a literal, so the size has a
    /// single source of truth here.
    pub(crate) fn dispatch1d(
        &self,
        wgsl: &str,
        entry: &str,
        buffers: &[&wgpu::Buffer],
        n_threads: u32,
    ) {
        let src = format!("const WG : u32 = {WORKGROUP}u;\n{wgsl}");
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entry),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(entry) });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(entry),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_threads.div_ceil(WORKGROUP), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Copy `len` f32s back from a `COPY_SRC` storage buffer to the host, blocking
    /// until the GPU work completes.
    pub(crate) fn download_f32(&self, src: &wgpu::Buffer, len: usize) -> Vec<f32> {
        let bytes = (len * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download"),
            });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let out = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        out
    }
}
