//! Small reusable wgpu compute primitives shared by the WGSL kernel ports:
//! buffer upload, single-pass 1-D dispatch (auto bind-group layout, workgroup
//! size 256), and host readback. Only compiled under `gpu-wgpu`.

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use crate::wgpu::WgpuBackend;

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

    /// A read-only storage buffer initialised from a `u32` index array (lprec
    /// gather/scatter targets bind as `array<u32>`).
    pub(crate) fn storage_ro_u32(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// An uninitialised read/write storage buffer of `len` f32s, usable as a
    /// `copy` source (readback) and destination (clear / upload). The caller must
    /// fully write it before reading, or [`Self::zero_buffer`] it first for
    /// accumulation targets (the atomic gather/wrap grid).
    pub(crate) fn storage_empty(&self, label: &str, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Zero a storage buffer on-device — cheaper than uploading a host zero
    /// vector for the MB-scale accumulation grid.
    pub(crate) fn zero_buffer(&self, buf: &wgpu::Buffer) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zero_buffer"),
            });
        enc.clear_buffer(buf, 0, None);
        self.queue.submit(Some(enc.finish()));
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

    /// WGSL header (binding declaration + accumulate/load helpers) for an
    /// atomically-accumulated f32 storage buffer `name` at `@group(0)
    /// @binding(binding)`. The scatter kernels (forward projection, fourierrec
    /// gather/wrap, lprec gather) declare their accumulation target through this
    /// so the *same kernel source* runs on both device classes:
    ///
    /// - with [`wgpu::Features::SHADER_FLOAT32_ATOMIC`] (`self.f32_atomics`):
    ///   `array<atomic<f32>>` + native `atomicAdd` — one atomic per update;
    /// - without: the portable emulation on `array<atomic<u32>>` — a
    ///   compare-exchange loop on the bit-cast lane, which costs a
    ///   read-modify-retry round per update (2–3× the memory traffic).
    ///
    /// Both variants leave the buffer's raw bytes as IEEE f32, so zeroing,
    /// readback ([`Self::download_f32`]) and non-atomic re-binding as
    /// `array<f32>` in later kernels are identical. The generated helpers are
    /// `atom_add_{name}(idx, v)` and `atom_load_{name}(idx) -> f32`.
    pub(crate) fn atomic_f32_decl(&self, name: &str, binding: u32) -> String {
        if self.f32_atomics {
            format!(
                "@group(0) @binding({binding}) var<storage, read_write> {name} : array<atomic<f32>>;\n\
                 fn atom_add_{name}(idx : u32, v : f32) {{ atomicAdd(&{name}[idx], v); }}\n\
                 fn atom_load_{name}(idx : u32) -> f32 {{ return atomicLoad(&{name}[idx]); }}\n"
            )
        } else {
            format!(
                "@group(0) @binding({binding}) var<storage, read_write> {name} : array<atomic<u32>>;\n\
                 fn atom_add_{name}(idx : u32, v : f32) {{\n\
                     var old = atomicLoad(&{name}[idx]);\n\
                     loop {{\n\
                         let r = atomicCompareExchangeWeak(&{name}[idx], old, bitcast<u32>(bitcast<f32>(old) + v));\n\
                         if (r.exchanged) {{ break; }}\n\
                         old = r.old_value;\n\
                     }}\n\
                 }}\n\
                 fn atom_load_{name}(idx : u32) -> f32 {{ return bitcast<f32>(atomicLoad(&{name}[idx])); }}\n"
            )
        }
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
    ///
    /// WebGPU caps each dispatch dimension at 65535 workgroups, which a 1-D
    /// dispatch blows past once `n_threads` exceeds `65535 * WORKGROUP` (≈16.8 M —
    /// reached by a 512² back-projection's `nz·n·n` threads). So the workgroup
    /// count is folded into a 2-D `(wx, wy)` grid with `wx ≤ 65535`, and every
    /// kernel recovers its flat thread index as
    /// `gid.y * num_workgroups.x * WG + gid.x` — a formula that also yields plain
    /// `gid.x` in the common `wy == 1` case, so the same kernel source works for
    /// both. Tail workgroups (the grid rounds up) are handled by each kernel's
    /// existing `idx >= len` bounds check.
    /// Fetch (or compile once and cache) the compute pipeline for `src`/`entry`.
    ///
    /// Keyed by a hash of the full source (including any injected `const`s) plus
    /// the entry name, so distinct kernels and size-specialized variants each get
    /// their own cached pipeline while a repeated dispatch reuses the compiled
    /// one. This turns the per-dispatch naga compile + driver pipeline build into
    /// a one-time cost per unique kernel — the dominant wgpu overhead vs CUDA's
    /// precompiled kernels (see the `pipelines` field doc on [`WgpuBackend`]).
    pub(crate) fn cached_pipeline(
        &self,
        src: &str,
        entry: &str,
    ) -> std::sync::Arc<wgpu::ComputePipeline> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        src.hash(&mut h);
        entry.hash(&mut h);
        let key = h.finish();
        if let Some(p) = self.pipelines.lock().unwrap().get(&key) {
            return p.clone();
        }
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
        let arc = std::sync::Arc::new(pipeline);
        self.pipelines.lock().unwrap().insert(key, arc.clone());
        arc
    }

    pub(crate) fn dispatch1d(
        &self,
        wgsl: &str,
        entry: &str,
        buffers: &[&wgpu::Buffer],
        n_threads: u32,
    ) {
        let src = format!("const WG : u32 = {WORKGROUP}u;\n{wgsl}");
        let pipeline = self.cached_pipeline(&src, entry);
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
            // (pipeline is an Arc; &pipeline derefs to &ComputePipeline here.)
            // Fold the workgroup count into a 2-D grid so neither dimension
            // exceeds WebGPU's 65535 per-dimension cap (see method doc).
            let wg = n_threads.div_ceil(WORKGROUP);
            const MAX_DIM: u32 = 65535;
            let (wx, wy) = if wg <= MAX_DIM {
                (wg, 1)
            } else {
                (MAX_DIM, wg.div_ceil(MAX_DIM))
            };
            pass.dispatch_workgroups(wx, wy, 1);
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
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wgpu device poll");
        rx.recv().unwrap().unwrap();
        let out = {
            let view = slice
                .get_mapped_range()
                .expect("mapped range after successful map_async");
            bytemuck::cast_slice(&view).to_vec()
        };
        staging.unmap();
        out
    }
}
