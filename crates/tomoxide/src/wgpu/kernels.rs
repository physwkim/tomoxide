//! WGSL kernel ports backing the [`Backend`](crate::backend::Backend)
//! capability traits. Only compiled under `gpu-wgpu`.
//!
//! GPU f32 is not bit-exact with the CPU/numpy reference (parallel reduction
//! order and approximate transcendentals), so the parity bar here is a
//! tolerance, not Δ=0 — see the `gpu_tests` in [`crate`].

use crate::backend::{
    make_fbp_filter, Elementwise, FbpFilter, Fft, FilteredBackproject, ForwardProject,
    FourierReconstruct, LpRecReconstruct, RampShape, RankFilter,
};
use crate::data::{Frames, Layout, Tomo, Volume};
use crate::dtype::Complex32;
use crate::error::{Error, Result};
use crate::geometry::{Beam, Geometry};
use crate::params::FilterName;
use bytemuck::{Pod, Zeroable};
use ndarray::{Array3, Axis};

use crate::wgpu::shaders::{
    BACKPROJECT_WGSL, BLUESTEIN_WGSL, ELEMENTWISE_WGSL, FBP_FILTER_WGSL, FFT_SHARED_WGSL,
    FFT_TRANSPOSE_WGSL, FFT_WGSL, FOURIERREC_WGSL, LPREC_WGSL, MEDFILT3D_WGSL, PROJECT_WGSL,
};
use crate::wgpu::WgpuBackend;

/// Max window the GPU median kernel can hold (must match `MAX_WIN` in
/// `medfilt3d.wgsl`): diameter 7, i.e. `size ≤ 7`.
const MEDFILT_MAX_WIN: usize = 343;

/// Largest transform length the shared-memory FFT kernel handles. At `vec2<f32>`
/// (8 bytes) per element this is 32 KiB of workgroup memory, within the common
/// desktop 48 KiB budget; longer transforms fall back to the global multi-pass
/// kernel. The per-adapter `max_compute_workgroup_storage_size` is still checked
/// at dispatch so downlevel adapters (16 KiB) drop to the global path sooner.
const SHARED_FFT_MAX: usize = 4096;

/// Threads per workgroup for the shared-memory FFT of length `n`: one thread per
/// two elements, capped at 256 (a good occupancy point; each thread strides over
/// the remaining butterflies). Always ≥ 1 and a power of two for power-of-two `n`.
fn shared_fft_workgroup(n: usize) -> u32 {
    ((n / 2).clamp(1, 256)) as u32
}

/// Uniform block for the `darkflat` kernel. Padded to 16 bytes to satisfy the
/// WGSL uniform-buffer layout rules.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DfParams {
    n_elems: u32,
    plane_size: u32,
    _pad0: u32,
    _pad1: u32,
}

impl Elementwise for WgpuBackend {
    /// `(data − dark) / (flat − dark)`, averaging the flat/dark frame stacks.
    ///
    /// Mirrors [`CpuBackend::darkflat`](../../tomoxide_cpu): the frame averaging
    /// and the divide-by-zero guard run host-side (a single mean per plane,
    /// cheap and order-sensitive), then the per-element broadcast subtraction
    /// and division run on the GPU.
    fn darkflat(&self, data: &mut Tomo<f32>, flat: &Frames<f32>, dark: &Frames<f32>) -> Result<()> {
        let dark2d = dark
            .array
            .mean_axis(Axis(0))
            .ok_or_else(|| Error::InvalidParam("empty dark stack".into()))?;
        let flat2d = flat
            .array
            .mean_axis(Axis(0))
            .ok_or_else(|| Error::InvalidParam("empty flat stack".into()))?;
        let mut denom = &flat2d - &dark2d;
        // Guard against divide-by-zero where flat == dark.
        denom.mapv_inplace(|v| if v.abs() < 1e-6 { 1.0 } else { v });

        // Work in projection layout so element i = proj*plane + (row*cols+col)
        // and the dark/denom plane is indexed by `i % plane`.
        let restore = data.layout == Layout::Sinogram;
        if restore {
            *data = data.to_layout(Layout::Projection);
        }
        let (np, nr, nc) = data.array.dim();
        if dark2d.dim() != (nr, nc) {
            return Err(Error::ShapeMismatch {
                expected: format!("{:?}", (nr, nc)),
                found: format!("{:?}", dark2d.dim()),
            });
        }
        let plane = nr * nc;

        let host: Vec<f32> = data.array.iter().copied().collect();
        let dark_std = dark2d.as_standard_layout();
        let denom_std = denom.as_standard_layout();
        let data_buf = self.storage_rw("df_data", &host);
        let dark_buf = self.storage_ro("df_dark", dark_std.as_slice().expect("standard layout"));
        let denom_buf = self.storage_ro("df_denom", denom_std.as_slice().expect("standard layout"));
        let params = DfParams {
            n_elems: host.len() as u32,
            plane_size: plane as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let param_buf = self.uniform("df_params", &params);
        self.dispatch1d(
            ELEMENTWISE_WGSL,
            "darkflat",
            &[&data_buf, &dark_buf, &denom_buf, &param_buf],
            host.len() as u32,
        );
        let out = self.download_f32(&data_buf, host.len());
        data.array = Array3::from_shape_vec((np, nr, nc), out).expect("len matches dims");

        if restore {
            *data = data.to_layout(Layout::Sinogram);
        }
        Ok(())
    }

    /// In-place `−ln(max(x, 1e-6))` with non-finite results scrubbed to `0`.
    ///
    /// Order-independent, so it runs on whatever layout `data` is in. GPU `log`
    /// differs from libm `ln` by a few ULP — callers compare with a tolerance.
    fn minus_log(&self, data: &mut Tomo<f32>) -> Result<()> {
        let dims = data.array.dim();
        let host: Vec<f32> = data.array.iter().copied().collect();
        let buf = self.storage_rw("ml_data", &host);
        self.dispatch1d(ELEMENTWISE_WGSL, "minus_log", &[&buf], host.len() as u32);
        let out = self.download_f32(&buf, host.len());
        data.array = Array3::from_shape_vec(dims, out).expect("len matches dims");
        Ok(())
    }
}

/// Uniform block for the `backproject` kernel. Padded to 32 bytes (a 16-byte
/// multiple) to satisfy the WGSL uniform-buffer layout rules.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BpParams {
    nproj: u32,
    ncols: u32,
    ny: u32,
    nx: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl FilteredBackproject for WgpuBackend {
    /// Parallel-beam voxel-driven back-projection on the GPU.
    ///
    /// Mirrors [`CpuBackend::backproject`](../../tomoxide_cpu): one GPU thread
    /// per output voxel sums the (already filtered) sinogram along all angles by
    /// linear interpolation and scales by `π / n_angles`. The per-angle
    /// `(cosθ, sinθ)` and the per-row `center` are computed host-side with the
    /// same `sin_cos` as the CPU path, so the only GPU/CPU divergence is the
    /// multiply-accumulate rounding — callers compare with a tolerance, not Δ=0.
    fn backproject(&self, sino: &Tomo<f32>, geom: &Geometry, out: &mut Volume<f32>) -> Result<()> {
        if geom.beam != Beam::Parallel {
            return Err(Error::InvalidParam(
                "wgpu back-projection currently supports parallel beam only".into(),
            ));
        }
        let s = sino.as_layout(Layout::Sinogram); // [row, angle, col]
        let nz = s.n_rows();
        let nang = s.n_angles();
        let ncols = s.n_cols();
        let (oz, ny, nx) = out.dims();
        if oz != nz {
            return Err(Error::ShapeMismatch {
                expected: format!("{nz} sinogram rows"),
                found: oz.to_string(),
            });
        }
        let angles = &geom.angles.0;
        if angles.len() != nang {
            return Err(Error::ShapeMismatch {
                expected: format!("{nang} angles"),
                found: angles.len().to_string(),
            });
        }

        let sino_std = s.array.as_standard_layout();
        let sino_buf = self.storage_ro("bp_sino", sino_std.as_slice().expect("standard layout"));
        out.array = self.backproject_from_dev(&sino_buf, nz, nang, ncols, geom, (ny, nx))?;
        Ok(())
    }
}

impl WgpuBackend {
    /// Device-resident filtered back-projection from an already-uploaded
    /// (filtered) sinogram buffer (`[nz·nang, ncols]`, sinogram C-order),
    /// returning the reconstructed volume as `[nz, ny, nx]`. Shared by
    /// [`FilteredBackproject::backproject`] (which uploads the host sinogram
    /// first) and the fused [`AnalyticReconstruct`] path for fbp/linerec (which
    /// passes the filter's on-device output straight in). `geom` carries the
    /// per-row centre and angles used for the sampling.
    pub(crate) fn backproject_from_dev(
        &self,
        sino_buf: &wgpu::Buffer,
        nz: usize,
        nang: usize,
        ncols: usize,
        geom: &Geometry,
        out_dims: (usize, usize),
    ) -> Result<Array3<f32>> {
        let (ny, nx) = out_dims;
        let (cossin, center) = cossin_center(geom, nz);
        let scale = std::f32::consts::PI / nang as f32;

        let cossin_buf = self.storage_ro("bp_cossin", &cossin);
        let center_buf = self.storage_ro("bp_center", &center);
        let total = nz * ny * nx;
        let vol_buf = self.storage_rw("bp_vol", &vec![0.0f32; total]);
        let params = BpParams {
            nproj: nang as u32,
            ncols: ncols as u32,
            ny: ny as u32,
            nx: nx as u32,
            scale,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let param_buf = self.uniform("bp_params", &params);
        self.dispatch1d(
            BACKPROJECT_WGSL,
            "backproject",
            &[sino_buf, &cossin_buf, &center_buf, &vol_buf, &param_buf],
            total as u32,
        );
        let result = self.download_f32(&vol_buf, total);
        Ok(Array3::from_shape_vec((nz, ny, nx), result).expect("len matches dims"))
    }
}

/// Per-angle `(cosθ, sinθ)` interleaved plus the per-row `center`, both computed
/// host-side with the same `sin_cos` as the CPU reference so the GPU sampling
/// (and its inclusion-boundary decision) is bit-identical to the CPU path. The
/// center is expanded to one value per row so the `Scalar` and `PerRow` cases
/// share a single buffer path. Shared by the back- and forward-projectors.
fn cossin_center(geom: &Geometry, nz: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cossin = Vec::with_capacity(geom.angles.0.len() * 2);
    for &a in &geom.angles.0 {
        let (sn, c) = a.sin_cos();
        cossin.push(c);
        cossin.push(sn);
    }
    let center = (0..nz).map(|r| geom.center.at(r)).collect();
    (cossin, center)
}

/// Transpose every `d0 × d1` image in a contiguous `[batch][d0][d1]` complex
/// buffer to `[batch][d1][d0]`, in place via a per-image scratch. Used by the
/// non-power-of-two [`fft_2d`](WgpuBackend::fft_2d) fallback to make the column
/// axis contiguous for a 1-D pass and to restore row-major order afterwards.
fn transpose_images(buf: &mut [Complex32], d0: usize, d1: usize, batch: usize) {
    let img = d0 * d1;
    let mut tmp = vec![Complex32::default(); img];
    for b in 0..batch {
        let base = b * img;
        for r in 0..d0 {
            for c in 0..d1 {
                tmp[c * d0 + r] = buf[base + r * d1 + c];
            }
        }
        buf[base..base + img].copy_from_slice(&tmp);
    }
}

impl ForwardProject for WgpuBackend {
    /// Parallel-beam pixel-driven forward projection (the Radon transform).
    ///
    /// Mirrors [`CpuBackend::project`](../../tomoxide_cpu) — the exact linear-
    /// interp adjoint of [`Self::backproject`]. Forward projection is a scatter
    /// (each pixel splats onto two detector columns), so to stay race-free the
    /// GPU maps one thread per `(row, angle)`: each owns a disjoint detector-
    /// column span and visits pixels in the CPU's `(iy, ix)` order, so the
    /// per-column accumulation order matches and only the multiply-accumulate
    /// rounding diverges (tolerance parity, not Δ=0). `out` is overwritten with
    /// a fresh `[row, angle, col]` sinogram.
    fn project(&self, vol: &Volume<f32>, geom: &Geometry, out: &mut Tomo<f32>) -> Result<()> {
        if geom.beam != Beam::Parallel {
            return Err(Error::InvalidParam(
                "wgpu forward projection currently supports parallel beam only".into(),
            ));
        }
        let (nz, ny, nx) = vol.dims();
        let nang = geom.angles.0.len();
        let ncols = geom.detector.width;
        if nang == 0 || ncols == 0 {
            return Err(Error::InvalidParam(
                "geometry has no angles or zero detector width".into(),
            ));
        }

        let (cossin, center) = cossin_center(geom, nz);
        let vol_std = vol.array.as_standard_layout();
        let vol_buf = self.storage_ro("fp_vol", vol_std.as_slice().expect("standard layout"));
        let cossin_buf = self.storage_ro("fp_cossin", &cossin);
        let center_buf = self.storage_ro("fp_center", &center);
        let total = nz * nang * ncols;
        let sino_buf = self.storage_rw("fp_sino", &vec![0.0f32; total]);
        let params = FpParams {
            nproj: nang as u32,
            ncols: ncols as u32,
            ny: ny as u32,
            nx: nx as u32,
            // π/nproj — the adjoint gain matching the back-projector, so {A, Aᵀ}
            // is a matched pair (the iterative solvers rely on this) and the
            // forward output matches the CPU/CUDA convention.
            scale: std::f32::consts::PI / nang as f32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let param_buf = self.uniform("fp_params", &params);
        // One thread per object voxel; voxels atomic-splat onto the sinogram
        // (the exact transpose of the voxel-driven back-projector). sino_buf was
        // zero-initialised above, which the accumulating kernel requires.
        self.dispatch1d(
            PROJECT_WGSL,
            "project",
            &[&vol_buf, &cossin_buf, &center_buf, &sino_buf, &param_buf],
            (nz * ny * nx) as u32,
        );
        let result = self.download_f32(&sino_buf, total);
        let array = Array3::from_shape_vec((nz, nang, ncols), result).expect("len matches dims");
        *out = Tomo::new(array, Layout::Sinogram);
        Ok(())
    }
}

/// Uniform block for the `project` kernel. Padded to 32 bytes (a 16-byte
/// multiple) to satisfy the WGSL uniform-buffer layout rules.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FpParams {
    nproj: u32,
    ncols: u32,
    ny: u32,
    nx: u32,
    scale: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Uniform block for the `medfilt3d` kernel. Padded to 32 bytes (a 16-byte
/// multiple) to satisfy the WGSL uniform-buffer layout rules.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MfParams {
    dz: u32,
    dy: u32,
    dx: u32,
    radius: u32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl WgpuBackend {
    /// 3-D median/dezinger core shared by [`RankFilter::median3d`] and
    /// [`RankFilter::remove_outlier3d`] — one GPU thread per voxel, bit-exact
    /// with `medfilt3d_core` on the CPU (pure gather + order statistic).
    fn medfilt3d_gpu(
        &self,
        arr: &Array3<f32>,
        radius: usize,
        threshold: f32,
    ) -> Result<Array3<f32>> {
        let diameter = 2 * radius + 1;
        let total = diameter * diameter * diameter;
        if total > MEDFILT_MAX_WIN {
            return Err(Error::InvalidParam(format!(
                "wgpu median window {diameter}³={total} exceeds the GPU cap of \
                 {MEDFILT_MAX_WIN} (size ≤ 7); use the CPU backend for larger windows"
            )));
        }
        let (dz, dy, dx) = arr.dim();
        let host: Vec<f32> = arr.iter().copied().collect();
        let n = host.len();
        let inp = self.storage_ro("mf_inp", &host);
        let outp = self.storage_rw("mf_outp", &vec![0.0f32; n]);
        let params = MfParams {
            dz: dz as u32,
            dy: dy as u32,
            dx: dx as u32,
            radius: radius as u32,
            threshold,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let param_buf = self.uniform("mf_params", &params);
        self.dispatch1d(
            MEDFILT3D_WGSL,
            "medfilt3d",
            &[&inp, &outp, &param_buf],
            n as u32,
        );
        let out = self.download_f32(&outp, n);
        Ok(Array3::from_shape_vec((dz, dy, dx), out).expect("len matches dims"))
    }
}

impl RankFilter for WgpuBackend {
    /// 3-D median filter (tomopy `median_filter3d`). Bit-exact with the CPU.
    fn median3d(&self, vol: &mut Volume<f32>, size: usize) -> Result<()> {
        let radius = (size.max(3) - 1) / 2;
        vol.array = self.medfilt3d_gpu(&vol.array, radius, 0.0)?;
        Ok(())
    }

    /// Outlier (zinger) removal (tomopy `remove_outlier3d`): the same kernel as
    /// [`Self::median3d`] but replacing a voxel by the local median only where
    /// it deviates from it by at least `diff`. Bit-exact with the CPU.
    fn remove_outlier3d(&self, data: &mut Tomo<f32>, diff: f32, size: usize) -> Result<()> {
        let radius = (size.max(3) - 1) / 2;
        data.array = self.medfilt3d_gpu(&data.array, radius, diff)?;
        Ok(())
    }
}

/// Uniform block for the FBP filter-multiply kernel. Padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FfltParams {
    pad: u32,
    ncols: u32,
    pad_side: u32,
    _pad0: u32,
    scale: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

/// Uniform block for the radix-2 FFT kernel (16 bytes — already a multiple of 16,
/// so no padding is needed).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FftParams {
    n: u32,
    logn: u32,
    m: u32,
    sign: f32,
}

/// Uniform block for the Bluestein convolution-multiply kernel. Padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BzParams {
    m: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Uniform block for the FFT transpose kernel. Padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

impl WgpuBackend {
    /// In-place radix-2 FFT of `data` as `lanes` contiguous transforms of length
    /// `n`. Does not normalize; the inverse `1/n` is applied by the caller.
    ///
    /// Dispatches the single-pass shared-memory kernel when the transform fits
    /// workgroup memory (the common FBP/gridrec/fourierrec lengths), and falls
    /// back to the multi-submission global kernel for larger `n` — see
    /// [`Self::fft_shared_pass`] and [`Self::fft_global_passes`].
    fn fft_passes(&self, data: &wgpu::Buffer, n: usize, lanes: usize, inverse: bool) {
        let lim = self.device.limits();
        let wg = shared_fft_workgroup(n);
        let fits_shared = n <= SHARED_FFT_MAX
            && (n * std::mem::size_of::<[f32; 2]>()) as u32
                <= lim.max_compute_workgroup_storage_size
            && wg <= lim.max_compute_invocations_per_workgroup;
        if fits_shared {
            self.fft_shared_pass(data, n, lanes, inverse, wg);
        } else {
            self.fft_global_passes(data, n, lanes, inverse);
        }
    }

    /// Single-pass shared-memory FFT: one workgroup per transform, all stages in
    /// workgroup memory. `wg` threads per workgroup (must equal
    /// [`shared_fft_workgroup`]); the caller guarantees `n` fits shared memory.
    /// The butterfly math matches [`Self::fft_global_passes`] bit-for-bit — only
    /// the backing memory differs.
    fn fft_shared_pass(&self, data: &wgpu::Buffer, n: usize, lanes: usize, inverse: bool, wg: u32) {
        let logn = n.trailing_zeros();
        let sign = if inverse { 1.0f32 } else { -1.0f32 };
        let params = self.uniform(
            "fft_sh_p",
            &FftParams {
                n: n as u32,
                logn,
                m: 0,
                sign,
            },
        );
        // Inject the workgroup size, transform length, and log2 as `const`, so the
        // shared array is sized exactly and `@workgroup_size` has one source of
        // truth matching the dispatched workgroup count.
        let src = format!(
            "const WG : u32 = {wg}u;\nconst NN : u32 = {n}u;\nconst LOGN : u32 = {logn}u;\n{FFT_SHARED_WGSL}"
        );
        let pipeline = self.cached_pipeline(&src, "fft_shared");
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fft_shared"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fft_shared"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft_shared"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per transform, folded into a 2-D grid so neither
            // dimension exceeds WebGPU's 65535 cap (the kernel recovers the lane
            // as `wg.y * num_workgroups.x + wg.x`).
            const MAX_DIM: u32 = 65535;
            let lanes = lanes as u32;
            let (wx, wy) = if lanes <= MAX_DIM {
                (lanes, 1)
            } else {
                (MAX_DIM, lanes.div_ceil(MAX_DIM))
            };
            pass.dispatch_workgroups(wx, wy, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    /// Multi-submission global-memory radix-2 passes (bit-reversal + `log2(n)`
    /// butterfly stages). Used for transforms too large for shared memory.
    /// Submissions serialize on the queue, so each stage observes the previous
    /// one's writes; transient uniform buffers stay alive through the pending
    /// submissions even after this returns.
    fn fft_global_passes(&self, data: &wgpu::Buffer, n: usize, lanes: usize, inverse: bool) {
        let logn = n.trailing_zeros();
        let sign = if inverse { 1.0f32 } else { -1.0f32 };
        let p0 = self.uniform(
            "fft_p",
            &FftParams {
                n: n as u32,
                logn,
                m: 0,
                sign,
            },
        );
        self.dispatch1d(FFT_WGSL, "bitrev", &[data, &p0], (n * lanes) as u32);
        let mut m = 2u32;
        for _ in 0..logn {
            let p = self.uniform(
                "fft_p",
                &FftParams {
                    n: n as u32,
                    logn,
                    m,
                    sign,
                },
            );
            self.dispatch1d(FFT_WGSL, "butterfly", &[data, &p], (n * lanes / 2) as u32);
            m <<= 1;
        }
    }

    /// Transpose each `rows × cols` image of `src` into `cols × rows` in `dst`.
    fn fft_transpose(
        &self,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        batch: usize,
    ) {
        let params = TParams {
            rows: rows as u32,
            cols: cols as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let p = self.uniform("fft_t", &params);
        self.dispatch1d(
            FFT_TRANSPOSE_WGSL,
            "transpose",
            &[src, dst, &p],
            (rows * cols * batch) as u32,
        );
    }

    /// Upload an interleaved complex buffer to a `read_write` storage buffer
    /// (each `Complex32` becomes a `vec2<f32>`).
    fn upload_complex(&self, label: &str, buf: &[Complex32]) -> wgpu::Buffer {
        let host: Vec<f32> = buf.iter().flat_map(|c| [c.re, c.im]).collect();
        self.storage_rw(label, &host)
    }

    /// Read an interleaved complex buffer back into `buf`, scaling by `norm`.
    fn download_complex(&self, data: &wgpu::Buffer, buf: &mut [Complex32], norm: f32) {
        let out = self.download_f32(data, buf.len() * 2);
        for (c, chunk) in buf.iter_mut().zip(out.chunks_exact(2)) {
            c.re = chunk[0] * norm;
            c.im = chunk[1] * norm;
        }
    }

    /// Bluestein (chirp-z) batched 1-D DFT for an **arbitrary** length `n`, used
    /// by [`Fft::fft_1d`] when `n` is not a power of two (the radix-2 kernel only
    /// handles power-of-two lengths). `buf` holds `lanes` contiguous transforms
    /// of length `n`; the result overwrites it in place. `inverse` selects the
    /// IDFT and applies the matching `1/n` normalization (like the radix-2 path).
    ///
    /// A length-`n` DFT `X[k] = Σ x[j]·exp(s·2πi·jk/n)` (s = −1 forward, +1
    /// inverse) is rewritten via `jk = (j² + k² − (k−j)²)/2` as
    /// `X[k] = p[k]·Σ (x[j]·p[j])·h[k−j]` with chirps `p[j] = exp(s·πi·j²/n)` and
    /// `h[m] = conj(p[|m|])` — a linear convolution evaluated by a power-of-two
    /// circular convolution of length `m = next_power_of_two(2n−1)` (FFT both,
    /// multiply spectra, inverse FFT). The chirps, the input premultiply, and the
    /// output postmultiply/crop are done host-side so the `j² mod 2n` argument
    /// reduction matches the CPU reference's precision; the three FFTs and the
    /// spectral multiply run on the GPU in one serialized submission chain.
    fn fft_bluestein(&self, buf: &mut [Complex32], n: usize, lanes: usize, inverse: bool) {
        let m = (2 * n - 1).next_power_of_two();
        let s = if inverse { 1.0f32 } else { -1.0f32 };
        let pi = std::f32::consts::PI;

        // Chirp p[j] = exp(s·πi·j²/n); reduce j² mod 2n first so the angle stays
        // small and precise even for large j (πj²/n grows quadratically).
        let two_n = 2 * n as u64;
        let p: Vec<Complex32> = (0..n)
            .map(|j| {
                let r = ((j as u64 * j as u64) % two_n) as f32;
                let ang = s * pi * r / n as f32;
                Complex32::new(ang.cos(), ang.sin())
            })
            .collect();

        // Per-lane premultiplied, zero-padded input a[l·m + j] = x[l·n + j]·p[j].
        let mut a_host = vec![Complex32::new(0.0, 0.0); lanes * m];
        for l in 0..lanes {
            for j in 0..n {
                a_host[l * m + j] = buf[l * n + j] * p[j];
            }
        }

        // Symmetric kernel h on the length-m ring: h[0], h[±j] = conj(p[j]).
        let mut h_host = vec![Complex32::new(0.0, 0.0); m];
        h_host[0] = p[0].conj();
        for j in 1..n {
            let hj = p[j].conj();
            h_host[j] = hj;
            h_host[m - j] = hj;
        }

        // FFT both, multiply spectra (h broadcast across lanes), inverse FFT.
        let a_buf = self.upload_complex("bz_a", &a_host);
        let h_buf = self.upload_complex("bz_h", &h_host);
        self.fft_passes(&h_buf, m, 1, false);
        self.fft_passes(&a_buf, m, lanes, false);
        let p_u = self.uniform(
            "bz_p",
            &BzParams {
                m: m as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );
        self.dispatch1d(
            BLUESTEIN_WGSL,
            "cmul",
            &[&a_buf, &h_buf, &p_u],
            (lanes * m) as u32,
        );
        self.fft_passes(&a_buf, m, lanes, true);

        // Download the (unnormalized inverse → ×m) convolution, then postmultiply
        // by p[k], crop to n, and apply 1/m (convolution) plus 1/n (IDFT).
        let out = self.download_f32(&a_buf, lanes * m * 2);
        let conv_norm = 1.0 / m as f32;
        let inv_norm = if inverse { 1.0 / n as f32 } else { 1.0 };
        let scale = conv_norm * inv_norm;
        for l in 0..lanes {
            for k in 0..n {
                let off = l * m + k;
                let c = Complex32::new(out[2 * off], out[2 * off + 1]) * p[k];
                buf[l * n + k] = Complex32::new(c.re * scale, c.im * scale);
            }
        }
    }
}

impl Fft for WgpuBackend {
    /// Batched 1-D FFT. Power-of-two `len` runs the direct radix-2 kernel; any
    /// other `len` runs the Bluestein chirp-z transform (also radix-2 under the
    /// hood), so the GPU handles arbitrary lengths like the CPU backend rather
    /// than erroring out. `inverse` divides by `len`, matching the CPU backend.
    fn fft_1d(&self, buf: &mut [Complex32], len: usize, batch: usize, inverse: bool) -> Result<()> {
        if len == 0 || batch == 0 {
            return Ok(());
        }
        if buf.len() != len * batch {
            return Err(Error::ShapeMismatch {
                expected: (len * batch).to_string(),
                found: buf.len().to_string(),
            });
        }
        if len.is_power_of_two() {
            let data = self.upload_complex("fft_1d", buf);
            self.fft_passes(&data, len, batch, inverse);
            let norm = if inverse { 1.0 / len as f32 } else { 1.0 };
            self.download_complex(&data, buf, norm);
        } else {
            self.fft_bluestein(buf, len, batch, inverse);
        }
        Ok(())
    }

    /// Batched 2-D FFT. Power-of-two `rows` and `cols` run the fast on-device
    /// path — a row pass + transpose + row pass + transpose, so both axes run as
    /// contiguous radix-2 transforms. Any other dims fall back to a separable
    /// pair of [`fft_1d`] passes (each radix-2 or Bluestein per length) with a
    /// host transpose between them, so the GPU handles arbitrary 2-D shapes like
    /// the CPU backend rather than erroring. `inverse` divides by `rows·cols`.
    fn fft_2d(
        &self,
        buf: &mut [Complex32],
        rows: usize,
        cols: usize,
        batch: usize,
        inverse: bool,
    ) -> Result<()> {
        if rows == 0 || cols == 0 || batch == 0 {
            return Ok(());
        }
        if buf.len() != rows * cols * batch {
            return Err(Error::ShapeMismatch {
                expected: (rows * cols * batch).to_string(),
                found: buf.len().to_string(),
            });
        }
        if !rows.is_power_of_two() || !cols.is_power_of_two() {
            // Separable fallback: FFT along the contiguous `cols` axis, transpose
            // each image so `rows` become contiguous, FFT along `rows`, transpose
            // back. Each 1-D pass selects radix-2 or Bluestein per length, and the
            // per-axis inverse divisors (1/cols then 1/rows) compose to the 2-D
            // 1/(rows·cols) — so the result matches the on-device pow2 path and
            // the CPU backend exactly.
            self.fft_1d(buf, cols, rows * batch, inverse)?;
            transpose_images(buf, rows, cols, batch);
            self.fft_1d(buf, rows, cols * batch, inverse)?;
            transpose_images(buf, cols, rows, batch);
            return Ok(());
        }
        let data = self.upload_complex("fft_2d", buf);
        // Row pass: `batch·rows` contiguous transforms of length `cols`.
        self.fft_passes(&data, cols, rows * batch, inverse);
        // Transpose to make columns contiguous, transform length `rows`, back.
        let scratch = self.storage_rw("fft_2d_t", &vec![0.0f32; rows * cols * batch * 2]);
        self.fft_transpose(&data, &scratch, rows, cols, batch);
        self.fft_passes(&scratch, rows, cols * batch, inverse);
        self.fft_transpose(&scratch, &data, cols, rows, batch);
        let norm = if inverse {
            1.0 / (rows * cols) as f32
        } else {
            1.0
        };
        self.download_complex(&data, buf, norm);
        Ok(())
    }
}

impl FbpFilter for WgpuBackend {
    /// Build the FBP apodized ramp filter on the host via the shared
    /// [`make_fbp_filter`] with [`RampShape::Linear`] — wgpu is a portable
    /// fallback that mirrors the CPU (tomopy) ramp, not the CUDA (tomocupy)
    /// `_wint` ramp.
    fn make_filter(&self, name: FilterName, n: usize) -> Result<Vec<f32>> {
        make_fbp_filter(name, n, RampShape::Linear)
    }

    /// Apply `filter` to every projection of `sino` on the GPU. Each detector
    /// lane (axis 2 in both layouts) is centred in a `pad = filter.len()`-wide
    /// buffer and edge-replicate-padded on both borders, forward-transformed,
    /// multiplied by the real filter, inverse-transformed, scaled by `1/pad`,
    /// and the centred `n_cols`-wide window cropped back out.
    /// Forward FFT, frequency-domain multiply, and inverse FFT all run on the
    /// GPU in one serialized submission chain — no host round-trip between the
    /// transforms. Mirrors `CpuBackend::apply`, including the per-row
    /// rotation-centre phase (tomocupy `fbp_filter_center`): the per-lane shift
    /// `ncols/2 − center` is uploaded and folded into the GPU filter-multiply,
    /// so after this pass the back-projectors assume centre = `ncols/2`. At the
    /// default centre the shift is zero and only the ramp applies. Requires a
    /// power-of-two `pad` (the GPU FFT is radix-2 only); other lengths error so
    /// the caller can fall back to CPU.
    fn apply(&self, sino: &mut Tomo<f32>, filter: &[f32], geom: &Geometry) -> Result<()> {
        let ncols = sino.n_cols();
        let batch = sino.array.len() / ncols;
        // Filter on-device, then download the compact real result and scatter it
        // back into `sino` (lane order matches `filter_to_device`'s C-order upload).
        let out = self.filter_to_device(sino, filter, geom)?;
        let host_out = self.download_f32(&out, batch * ncols);
        for (l, mut lane) in sino.array.lanes_mut(Axis(2)).into_iter().enumerate() {
            let base = l * ncols;
            for (i, slot) in lane.iter_mut().enumerate() {
                *slot = host_out[base + i];
            }
        }
        Ok(())
    }
}

impl WgpuBackend {
    /// Device-resident FBP filter: upload only the raw REAL sinogram
    /// (`batch·ncols` f32), then pack / pad / FFT / ×filter / IFFT / crop entirely
    /// on-GPU, returning the filtered real sinogram **as a device buffer**
    /// (`[batch·ncols]`, sinogram C-order — lane `l` at `[l·ncols, (l+1)·ncols)`).
    ///
    /// This is the shared core of [`FbpFilter::apply`] (which downloads + scatters
    /// the result to host) and the fused [`AnalyticReconstruct`] path (which feeds
    /// the buffer straight into the gridding / back-projection kernels, skipping
    /// the filtered-sinogram download + re-upload round-trip). The old host path
    /// built the full complex padded batch on the host and round-tripped 2× the
    /// data; only the compact real sinogram crosses the bus now.
    pub(crate) fn filter_to_device(
        &self,
        sino: &Tomo<f32>,
        filter: &[f32],
        geom: &Geometry,
    ) -> Result<wgpu::Buffer> {
        let pad = filter.len();
        if pad == 0 {
            return Err(Error::InvalidParam("empty filter".into()));
        }
        let ncols = sino.n_cols();
        if pad < ncols {
            return Err(Error::ShapeMismatch {
                expected: format!(">= {ncols} (n_cols)"),
                found: pad.to_string(),
            });
        }
        if !pad.is_power_of_two() {
            return Err(Error::InvalidParam(format!(
                "wgpu FBP filter requires a power-of-two length (got {pad}); use the CPU backend"
            )));
        }
        let batch = sino.array.len() / ncols;
        let pad_side = pad / 2 - ncols / 2;
        // Per-lane centre shift δ = ncols/2 − center(row). Lane `l` (C-order over
        // the two non-detector axes) maps to slice row `l/d1` (Sinogram) or
        // `l%d1` (Projection), matching CpuBackend::apply. δ=0 at the default
        // centre keeps the GPU goldens identical.
        let half = ncols as f32 / 2.0;
        let d1 = sino.array.shape()[1];
        let deltas: Vec<f32> = (0..batch)
            .map(|l| {
                let row = match sino.layout {
                    Layout::Sinogram => l / d1,
                    Layout::Projection => l % d1,
                };
                half - geom.center.at(row)
            })
            .collect();
        // Upload the sinogram in C order; lane `l` then occupies
        // `[l·ncols, (l+1)·ncols)`, matching `pack`/`unpack` and the `lanes`
        // enumeration order used for `deltas`.
        let std = sino.array.as_standard_layout();
        let sino_host = std
            .as_slice()
            .expect("as_standard_layout yields a contiguous slice");
        let sino_buf = self.storage_ro("fbp_sino", sino_host);
        let data = self.storage_empty("fbp_spectrum", batch * pad * 2);
        let w = self.storage_ro("fbp_w", filter);
        let deltas_buf = self.storage_ro("fbp_deltas", &deltas);
        let p = self.uniform(
            "fbp_p",
            &FfltParams {
                pad: pad as u32,
                ncols: ncols as u32,
                pad_side: pad_side as u32,
                _pad0: 0,
                scale: 1.0 / pad as f32,
                _pad1: 0.0,
                _pad2: 0.0,
                _pad3: 0.0,
            },
        );
        self.dispatch1d(
            FBP_FILTER_WGSL,
            "pack",
            &[&sino_buf, &data, &p],
            (batch * pad) as u32,
        );
        self.fft_passes(&data, pad, batch, false);
        self.dispatch1d(
            FBP_FILTER_WGSL,
            "apply_filter",
            &[&data, &w, &p, &deltas_buf],
            (batch * pad) as u32,
        );
        self.fft_passes(&data, pad, batch, true);
        // `unpack` crops the real central window and folds the 1/pad inverse-FFT
        // normalisation (fft_passes leaves the inverse unscaled) into `scale`.
        let out = self.storage_empty("fbp_out", batch * ncols);
        self.dispatch1d(
            FBP_FILTER_WGSL,
            "unpack",
            &[&data, &out, &p],
            (batch * ncols) as u32,
        );
        Ok(out)
    }
}

/// Uniform block for the device-resident fourierrec kernels. 64 bytes (all
/// scalars, so 16-byte-aligned as WGSL uniforms require). Mirrors `FrParams` in
/// `fourierrec.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrParams {
    nz: u32,
    nang: u32,
    nd: u32,
    n: u32,
    ng: u32,
    nf: u32,
    crop: u32,
    _p0: u32,
    mu: f32,
    coeff0: f32,
    coeff1: f32,
    gscale: f32,
    phi_sign: f32,
    inv_nf2: f32,
    _p1: f32,
    _p2: f32,
}

impl FourierReconstruct for WgpuBackend {
    /// Device-resident Gaussian-USFFT (tomocupy `cfunc_fourierrec`) — the whole
    /// gather / wrap / FFT / deapodize chain runs on the GPU, mirroring
    /// [`crate::recon::fourierrec`] step-for-step. Only the filtered sinogram is
    /// uploaded and the volume downloaded, replacing the host-gridding + per-call
    /// FFT-round-trip path that dominated wgpu fourierrec wall time.
    fn reconstruct(&self, filtered: &Tomo<f32>, geom: &Geometry, n: usize) -> Result<Volume<f32>> {
        let b = filtered.as_layout(Layout::Sinogram);
        let nz = b.n_rows();
        let nang = b.n_angles();
        let nd = b.n_cols();

        // The device path drives the radix-2 `fft_passes` directly, so both the
        // length-`nd` 1-D FFT and the `2*nd` 2-D FFT must be power-of-two. For a
        // non-power-of-two detector width, delegate to the generic host gridding
        // (which selects Bluestein per length through the `Fft` capability).
        if !nd.is_power_of_two() {
            return Ok(Volume::new(crate::recon::fourierrec::fourierrec(
                filtered, geom, n, self,
            )?));
        }

        let bdata = b
            .array
            .as_slice()
            .expect("contiguous sinogram (as_layout yields a standard-layout copy)");
        // Upload the filtered sinogram, then run the device-resident core.
        let sino_buf = self.storage_ro("fr_sino", bdata);
        self.fourierrec_from_dev(&sino_buf, nz, nang, nd, geom, n)
    }
}

impl WgpuBackend {
    /// Device-resident Fourier-grid (regridding) reconstruction from an
    /// already-uploaded filtered sinogram buffer (`[nz·nang, nd]`, sinogram
    /// C-order). Shared by [`Fourierrec::reconstruct`] (which uploads the host
    /// sinogram first) and the fused [`AnalyticReconstruct`] path (which passes
    /// the filter's on-device output straight in, skipping the round-trip). All
    /// Gaussian-kernel arithmetic mirrors the CPU port (recon/fourierrec.rs) so
    /// the grids coincide. Caller guarantees `nd.is_power_of_two()`.
    pub(crate) fn fourierrec_from_dev(
        &self,
        sino_buf: &wgpu::Buffer,
        nz: usize,
        nang: usize,
        nd: usize,
        geom: &Geometry,
        n: usize,
    ) -> Result<Volume<f32>> {
        // Gaussian-kernel parameters — identical arithmetic to the CPU port so
        // the grids coincide (recon/fourierrec.rs).
        const EPS: f64 = 1e-3;
        let ndf = nd as f64;
        let neg_log_eps = -EPS.ln();
        let mu = neg_log_eps / (2.0 * ndf * ndf);
        let inside = mu * neg_log_eps + (mu * ndf) * (mu * ndf) / 4.0;
        let m = (2.0 * ndf / std::f64::consts::PI * inside.sqrt()).ceil() as usize;

        let ng = 2 * nd + 2 * m;
        let nf = 2 * nd;
        let crop = (nd - n.min(nd)) / 2;
        let mu32 = mu as f32;
        let ndf32 = nd as f32;
        let params = FrParams {
            nz: nz as u32,
            nang: nang as u32,
            nd: nd as u32,
            n: n as u32,
            ng: ng as u32,
            nf: nf as u32,
            crop: crop as u32,
            _p0: 0,
            mu: mu32,
            coeff0: std::f32::consts::PI / (mu32 * 4.0 * ndf32 * ndf32),
            coeff1: -std::f32::consts::PI * std::f32::consts::PI / mu32,
            gscale: 4.0 / ndf32,
            phi_sign: 1.0 - (nd % 4) as f32,
            inv_nf2: 1.0 / (nf as f32 * nf as f32),
            _p1: 0.0,
            _p2: 0.0,
        };

        // (cos θ, sin θ) per angle — the +sin gather convention of the CPU port.
        let trig: Vec<f32> = geom
            .angles
            .0
            .iter()
            .flat_map(|&a| {
                let (s, c) = a.sin_cos();
                [c, s]
            })
            .collect();

        // Inject the Gaussian half-width `m` as a `const` so `array<f32, 2m+1>`
        // is sized exactly (dispatch1d injects `WG` on top of this).
        let src = format!("const M : u32 = {m}u;\n{FOURIERREC_WGSL}");

        let trig_buf = self.storage_ro("fr_trig", &trig);
        let radial = self.storage_empty("fr_radial", nz * nang * nd * 2);
        let grid = self.storage_empty("fr_grid", nz * ng * ng * 2);
        let inner = self.storage_empty("fr_inner", nz * nf * nf * 2);
        let scratch = self.storage_empty("fr_scratch", nz * nf * nf * 2);
        let out = self.storage_empty("fr_out", nz * n * n);
        let p = self.uniform("fr_p", &params);

        // Zero the accumulation grid before the atomic gather/wrap.
        self.zero_buffer(&grid);

        let radial_threads = (nz * nang * nd) as u32;
        // 1. complex radial buffer with pre-FFT shift modulation.
        self.dispatch1d(
            &src,
            "build_radial",
            &[sino_buf, &radial, &p],
            radial_threads,
        );
        // 2. centred 1-D FFT of every projection (length nd), then post-modulate.
        self.fft_passes(&radial, nd, nz * nang, false);
        self.dispatch1d(&src, "postmod", &[&radial, &p], radial_threads);
        // 3. Gaussian gather onto the oversampled grid; fold borders.
        self.dispatch1d(
            &src,
            "gather",
            &[&radial, &grid, &trig_buf, &p],
            radial_threads,
        );
        self.dispatch1d(&src, "wrap", &[&grid, &p], (nz * ng * ng) as u32);
        // 4. extract the 2*nd interior with the pre-inverse-FFT shift.
        self.dispatch1d(&src, "extract", &[&grid, &inner, &p], (nz * nf * nf) as u32);
        // 5. centred inverse 2-D FFT (row pass, transpose, row pass, transpose);
        //    the 1/nf^2 normalisation is folded into deapodize.
        self.fft_passes(&inner, nf, nf * nz, true);
        self.fft_transpose(&inner, &scratch, nf, nf, nz);
        self.fft_passes(&scratch, nf, nf * nz, true);
        self.fft_transpose(&scratch, &inner, nf, nf, nz);
        self.dispatch1d(&src, "fftshift", &[&inner, &p], (nz * nf * nf) as u32);
        // 6. deapodize, central crop, unit-disk mask → real output.
        self.dispatch1d(&src, "deapodize", &[&inner, &out, &p], (nz * n * n) as u32);

        let host = self.download_f32(&out, nz * n * n);
        let arr = Array3::from_shape_vec((nz, n, n), host)
            .expect("out buffer length nz*n*n matches the volume shape");
        Ok(Volume::new(arr))
    }
}

/// Uniform block for the device-resident lprec kernels. 48 bytes (16-aligned).
/// Mirrors `LpParams` in `lprec.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LpParams {
    nz: u32,
    nang: u32,
    n: u32,
    ntheta: u32,
    nrho: u32,
    npts: u32,
    _p0: u32,
    _p1: u32,
    scale: f32,
    _p2: f32,
    _p3: f32,
    _p4: f32,
}

impl LpRecReconstruct for WgpuBackend {
    /// Device-resident log-polar reconstruction (tomocupy `lprec`) — the cubic
    /// B-spline prefilter, the polar↔log-polar↔Cartesian gather/scatter, and the
    /// 2-D FFT convolution all run on the GPU, mirroring [`crate::recon::lprec`]
    /// step-for-step. The geometry grids are precomputed once on the host
    /// (`build_grids`) and uploaded; this replaces the host-interpolation +
    /// per-call FFT-round-trip fallback that dominated wgpu lprec wall time.
    fn reconstruct(&self, filtered: &Tomo<f32>, geom: &Geometry, n: usize) -> Result<Volume<f32>> {
        let b = filtered.as_layout(Layout::Sinogram);
        let nz = b.n_rows();
        let nang = b.n_angles();
        let nd = b.n_cols();

        // Same geometry guards as the CPU port (recon/lprec.rs): square geometry
        // with equally spaced angles spanning [0, π).
        if nang < 2 || nd != n {
            return Err(Error::InvalidParam(format!(
                "lprec requires square geometry with detector width == grid size (got n={n}, ncols={nd})"
            )));
        }
        let angles = &geom.angles.0;
        let dth = (angles[1] - angles[0]).abs();
        let nproj_test = (std::f32::consts::PI / dth).round() as usize;
        if nproj_test != nang {
            return Err(Error::InvalidParam(
                "lprec requires equally spaced angles spanning [0, π)".into(),
            ));
        }

        // Precompute the log-polar grids on the host (angle-independent), reusing
        // the CPU port's builder with this backend's Fft for the precompute FFTs.
        let grids = crate::recon::lprec::build_grids(n, nang, &crate::cpu::CpuBackend::new())?;
        let ntheta = grids.ntheta;
        let nrho = grids.nrho;
        let scale = 2.0 / (nrho as f32 * ntheta as f32);

        let bdata = b
            .array
            .as_slice()
            .expect("contiguous sinogram (as_layout yields a standard-layout copy)");

        let g = self.storage_rw("lp_g", bdata);
        let fl = self.storage_empty("lp_fl", nz * nrho * ntheta);
        let flc = self.storage_empty("lp_flc", nz * nrho * ntheta * 2);
        let scratch = self.storage_empty("lp_scratch", nz * nrho * ntheta * 2);
        let flcre = self.storage_empty("lp_flcre", nz * nrho * ntheta);
        let f = self.storage_empty("lp_f", nz * n * n);

        let kfull_host: Vec<f32> = grids.kfull.iter().flat_map(|c| [c.re, c.im]).collect();
        let kfull_buf = self.storage_ro("lp_kfull", &kfull_host);
        let lpids: Vec<u32> = grids.lpids.iter().map(|&x| x as u32).collect();
        let wids: Vec<u32> = grids.wids.iter().map(|&x| x as u32).collect();
        let cids: Vec<u32> = grids.cids.iter().map(|&x| x as u32).collect();
        let lpids_buf = self.storage_ro_u32("lp_lpids", &lpids);
        let wids_buf = self.storage_ro_u32("lp_wids", &wids);
        let cids_buf = self.storage_ro_u32("lp_cids", &cids);

        let mk = |npts: usize| {
            self.uniform(
                "lp_p",
                &LpParams {
                    nz: nz as u32,
                    nang: nang as u32,
                    n: n as u32,
                    ntheta: ntheta as u32,
                    nrho: nrho as u32,
                    npts: npts as u32,
                    _p0: 0,
                    _p1: 0,
                    scale,
                    _p2: 0.0,
                    _p3: 0.0,
                    _p4: 0.0,
                },
            )
        };
        // npts=0 uniform shared by the whole-grid kernels (prefilter, r2c, cmul,
        // take_re); they read only nz/nang/n/ntheta/nrho/scale.
        let p0 = mk(0);

        // Cubic-B-spline prefilter: detector axis then angle axis.
        self.dispatch1d(LPREC_WGSL, "prefilter_rows", &[&g, &p0], (nz * nang) as u32);
        self.dispatch1d(LPREC_WGSL, "prefilter_cols", &[&g, &p0], (nz * n) as u32);

        self.zero_buffer(&f);
        for k in 0..grids.lp2p1.len() {
            self.zero_buffer(&fl);

            // 1. gather polar → log-polar (main set, then wrapping set).
            let p_main = mk(lpids.len());
            let xs_main = self.storage_ro("lp_xs", &grids.lp2p2[k]);
            let ys_main = self.storage_ro("lp_ys", &grids.lp2p1[k]);
            self.dispatch1d(
                LPREC_WGSL,
                "gather",
                &[&g, &fl, &lpids_buf, &xs_main, &ys_main, &p_main],
                (nz * lpids.len()) as u32,
            );
            if !wids.is_empty() {
                let p_w = mk(wids.len());
                let xs_w = self.storage_ro("lp_xsw", &grids.lp2p2w[k]);
                let ys_w = self.storage_ro("lp_ysw", &grids.lp2p1w[k]);
                self.dispatch1d(
                    LPREC_WGSL,
                    "gather",
                    &[&g, &fl, &wids_buf, &xs_w, &ys_w, &p_w],
                    (nz * wids.len()) as u32,
                );
            }

            // 2. 2-D FFT convolution: fl → complex, forward, ×kfull, inverse,
            //    take real × scale.
            let grid_threads = (nz * nrho * ntheta) as u32;
            self.dispatch1d(
                LPREC_WGSL,
                "real_to_complex",
                &[&fl, &flc, &p0],
                grid_threads,
            );
            self.fft_passes(&flc, ntheta, nrho * nz, false);
            self.fft_transpose(&flc, &scratch, nrho, ntheta, nz);
            self.fft_passes(&scratch, nrho, ntheta * nz, false);
            self.fft_transpose(&scratch, &flc, ntheta, nrho, nz);
            self.dispatch1d(LPREC_WGSL, "cmul", &[&flc, &kfull_buf, &p0], grid_threads);
            self.fft_passes(&flc, ntheta, nrho * nz, true);
            self.fft_transpose(&flc, &scratch, nrho, ntheta, nz);
            self.fft_passes(&scratch, nrho, ntheta * nz, true);
            self.fft_transpose(&scratch, &flc, ntheta, nrho, nz);
            self.dispatch1d(LPREC_WGSL, "take_re", &[&flc, &flcre, &p0], grid_threads);

            // 3. scatter log-polar → Cartesian disk (accumulates across spans).
            let p_c = mk(cids.len());
            let xs_c = self.storage_ro("lp_xsc", &grids.c2lp1[k]);
            let ys_c = self.storage_ro("lp_ysc", &grids.c2lp2[k]);
            self.dispatch1d(
                LPREC_WGSL,
                "scatter",
                &[&flcre, &f, &cids_buf, &xs_c, &ys_c, &p_c],
                (nz * cids.len()) as u32,
            );
        }

        let host = self.download_f32(&f, nz * n * n);
        let arr = Array3::from_shape_vec((nz, n, n), host)
            .expect("out buffer length nz*n*n matches the volume shape");
        Ok(Volume::new(arr))
    }
}

impl crate::backend::AnalyticReconstruct for WgpuBackend {
    /// Fused fbp/linerec/fourierrec: FBP-filter the sinogram on-device and feed
    /// the filtered buffer straight into the back-projection / Fourier-gridding
    /// kernels, keeping it device-resident. The composed capability path
    /// (recon::recon's fallback) downloads the filtered sinogram after
    /// `FbpFilter::apply` and re-uploads it inside the recon; this fuses the two
    /// so the sinogram crosses the bus once (up) and only the volume comes back.
    fn reconstruct(
        &self,
        sino: &Tomo<f32>,
        geom: &Geometry,
        algorithm: crate::params::Algorithm,
        params: &crate::params::ReconParams,
    ) -> Result<Volume<f32>> {
        use crate::params::Algorithm;
        let ncols = sino.n_cols();
        let n = params.num_gridx.unwrap_or(ncols);
        let nz = sino.n_rows();
        let nang = sino.n_angles();
        let kernel = self.make_filter(params.filter_name, ncols)?;

        // fourierrec's device core drives the radix-2 FFT directly, so it needs a
        // power-of-two detector width. For other widths, compose the host path
        // (filter on-device to host, then the generic Bluestein-capable gridding)
        // exactly as recon::recon's fallback does — the fused path is a pure
        // optimisation and must not change which inputs are accepted.
        if algorithm == Algorithm::Fourierrec && !ncols.is_power_of_two() {
            let mut filtered = sino.clone();
            <Self as FbpFilter>::apply(self, &mut filtered, &kernel, geom)?;
            return Ok(Volume::new(crate::recon::fourierrec::fourierrec(
                &filtered, geom, n, self,
            )?));
        }

        // Filter in sinogram layout so the on-device buffer is [nz·nang, ncols] in
        // the C-order the recon cores expect; the per-row centre shift is folded
        // in by the filter (after it, the back-projector assumes centre = ncols/2).
        let s = sino.as_layout(Layout::Sinogram);
        let filt_dev = self.filter_to_device(&s, &kernel, geom)?;

        match algorithm {
            // fourierrec grids against the original angles; the centre shift is
            // already baked into the filtered data.
            Algorithm::Fourierrec => self.fourierrec_from_dev(&filt_dev, nz, nang, ncols, geom, n),
            // fbp / linerec back-project against a centre = ncols/2 geometry, the
            // axis the filter shifted the projections onto.
            Algorithm::Fbp | Algorithm::Linerec => {
                let centered = Geometry {
                    center: crate::geometry::Center::Scalar(ncols as f32 / 2.0),
                    ..geom.clone()
                };
                let arr =
                    self.backproject_from_dev(&filt_dev, nz, nang, ncols, &centered, (n, n))?;
                Ok(Volume::new(arr))
            }
            // recon::recon only routes these three algorithms here.
            other => Err(Error::InvalidParam(format!(
                "wgpu analytic_reconstruct does not handle {other:?}"
            ))),
        }
    }
}

impl WgpuBackend {
    /// Forward-project `vol_buf` into `sino_buf` (both device-resident), zeroing
    /// `sino_buf` first (the per-voxel atomic-splat kernel accumulates). `cossin_buf` /
    /// `center_buf` are the geometry buffers uploaded once by the iterative solver
    /// so the per-iteration projection reuses them. Matched adjoint gain π/nproj.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_into_dev(
        &self,
        vol_buf: &wgpu::Buffer,
        cossin_buf: &wgpu::Buffer,
        center_buf: &wgpu::Buffer,
        sino_buf: &wgpu::Buffer,
        dims: (usize, usize, usize), // (nz, ny, nx)
        nang: usize,
        ncols: usize,
    ) {
        let (nz, ny, nx) = dims;
        self.zero_buffer(sino_buf);
        let params = FpParams {
            nproj: nang as u32,
            ncols: ncols as u32,
            ny: ny as u32,
            nx: nx as u32,
            scale: std::f32::consts::PI / nang as f32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let pbuf = self.uniform("fp_params", &params);
        // One thread per object voxel (atomic-splat); sino zeroed above.
        self.dispatch1d(
            PROJECT_WGSL,
            "project",
            &[vol_buf, cossin_buf, center_buf, sino_buf, &pbuf],
            (nz * ny * nx) as u32,
        );
    }

    /// Back-project `sino_buf` into `vol_buf` (both device-resident; the voxel
    /// kernel overwrites, so no pre-zero). Matched adjoint gain π/nproj. Shares
    /// the geometry buffers with [`Self::forward_into_dev`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn backproject_into_dev(
        &self,
        sino_buf: &wgpu::Buffer,
        cossin_buf: &wgpu::Buffer,
        center_buf: &wgpu::Buffer,
        vol_buf: &wgpu::Buffer,
        dims: (usize, usize, usize), // (nz, ny, nx)
        nang: usize,
        ncols: usize,
    ) {
        let (nz, ny, nx) = dims;
        let params = BpParams {
            nproj: nang as u32,
            ncols: ncols as u32,
            ny: ny as u32,
            nx: nx as u32,
            scale: std::f32::consts::PI / nang as f32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let pbuf = self.uniform("bp_params", &params);
        self.dispatch1d(
            BACKPROJECT_WGSL,
            "backproject",
            &[sino_buf, cossin_buf, center_buf, vol_buf, &pbuf],
            (nz * ny * nx) as u32,
        );
    }

    /// Device-resident SIRT (tomopy `sirt`): keep the volume, measured sinogram,
    /// and the R/C weights resident on the GPU across every iteration — one
    /// upload, one download — instead of the generic host solver's per-iteration
    /// forward/back-projection round-trips. Mirrors the CUDA `sirt_device`
    /// arithmetic (matched π/nproj projector pair, R = 1/A(1), C = 1/Aᵀ(1),
    /// x += C∘Aᵀ(R∘(b−Ax))), so the result matches the host SIRT within GPU ULPs.
    fn sirt_device(
        &self,
        sino: &Tomo<f32>,
        geom: &Geometry,
        params: &crate::params::ReconParams,
    ) -> Result<Volume<f32>> {
        let s = sino.as_layout(Layout::Sinogram);
        let nz = s.n_rows();
        let nang = s.n_angles();
        let ncols = s.n_cols();
        let n = params.num_gridx.unwrap_or(ncols);
        let nvol = nz * n * n;
        let nsino = nz * nang * ncols;

        // Geometry buffers uploaded once, reused every iteration.
        let (cossin, center) = cossin_center(geom, nz);
        let cossin_buf = self.storage_ro("sirt_cossin", &cossin);
        let center_buf = self.storage_ro("sirt_center", &center);

        // Measured sinogram b (device-resident for the whole solve).
        let sino_std = s.array.as_standard_layout();
        let b_buf = self.storage_ro(
            "sirt_b",
            sino_std
                .as_slice()
                .expect("as_standard_layout yields a contiguous slice"),
        );

        // Seed x: warm-start `init` (validated [nz, n, n]) or zeros.
        let vol_buf = match &params.init {
            Some(v) => {
                if v.dims() != (nz, n, n) {
                    return Err(Error::ShapeMismatch {
                        expected: format!("init volume [{nz}, {n}, {n}]"),
                        found: format!("{:?}", v.dims()),
                    });
                }
                let vstd = v.array.as_standard_layout();
                self.storage_rw(
                    "sirt_vol",
                    vstd.as_slice().expect("standard layout init volume"),
                )
            }
            None => self.storage_rw("sirt_vol", &vec![0.0f32; nvol]),
        };

        let ax_buf = self.storage_empty("sirt_ax", nsino); // A x, reused as residual
        let corr_buf = self.storage_empty("sirt_corr", nvol); // Aᵀ(…)
        let rw_buf = self.storage_empty("sirt_rw", nsino); // R = 1/A(1)
        let cw_buf = self.storage_empty("sirt_cw", nvol); // C = 1/Aᵀ(1)
        let ones_v = self.storage_ro("sirt_ones_v", &vec![1.0f32; nvol]);
        let ones_s = self.storage_ro("sirt_ones_s", &vec![1.0f32; nsino]);

        let dims = (nz, n, n);
        // R = 1 / A(1)
        self.forward_into_dev(
            &ones_v,
            &cossin_buf,
            &center_buf,
            &ax_buf,
            dims,
            nang,
            ncols,
        );
        self.dispatch1d(
            ELEMENTWISE_WGSL,
            "iter_recip",
            &[&ax_buf, &rw_buf],
            nsino as u32,
        );
        // C = 1 / Aᵀ(1)
        self.backproject_into_dev(
            &ones_s,
            &cossin_buf,
            &center_buf,
            &corr_buf,
            dims,
            nang,
            ncols,
        );
        self.dispatch1d(
            ELEMENTWISE_WGSL,
            "iter_recip",
            &[&corr_buf, &cw_buf],
            nvol as u32,
        );

        for _ in 0..params.num_iter.max(1) {
            // ax = A x
            self.forward_into_dev(
                &vol_buf,
                &cossin_buf,
                &center_buf,
                &ax_buf,
                dims,
                nang,
                ncols,
            );
            // ax = (b − ax) ∘ R
            self.dispatch1d(
                ELEMENTWISE_WGSL,
                "iter_residual",
                &[&ax_buf, &b_buf, &rw_buf],
                nsino as u32,
            );
            // corr = Aᵀ(ax)
            self.backproject_into_dev(
                &ax_buf,
                &cossin_buf,
                &center_buf,
                &corr_buf,
                dims,
                nang,
                ncols,
            );
            // x += C ∘ corr
            self.dispatch1d(
                ELEMENTWISE_WGSL,
                "iter_update",
                &[&vol_buf, &cw_buf, &corr_buf],
                nvol as u32,
            );
        }

        let host = self.download_f32(&vol_buf, nvol);
        let array = Array3::from_shape_vec((nz, n, n), host)
            .expect("out buffer length nz*n*n matches the volume shape");
        Ok(Volume::new(array))
    }
}

impl crate::backend::IterativeReconstruct for WgpuBackend {
    /// SIRT runs device-resident; every other iterative algorithm returns `None`
    /// so recon::recon falls back to the generic host solver.
    fn solve(
        &self,
        sino: &Tomo<f32>,
        geom: &Geometry,
        algorithm: crate::params::Algorithm,
        params: &crate::params::ReconParams,
    ) -> Result<Option<Volume<f32>>> {
        // The device forward/back-projectors are parallel-beam only; anything else
        // falls back to the host solver.
        if geom.beam != Beam::Parallel {
            return Ok(None);
        }
        match algorithm {
            crate::params::Algorithm::Sirt => self.sirt_device(sino, geom, params).map(Some),
            _ => Ok(None),
        }
    }
}
