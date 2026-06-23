//! # tomoxide-cuda
//!
//! The CUDA backend. It re-uses tomocupy's battle-tested `.cu` kernels through
//! a thin C-ABI shim (see `ffi` and `cuda/shim.cpp`) rather than rewriting
//! them. Compiled only when the **`cuda` feature** is enabled and an NVIDIA
//! toolkit is present; otherwise [`CudaBackend::new`] reports the backend as
//! unavailable so the rest of the workspace still builds and runs (on CPU).
//!
//! ## M4 scope
//! GPU capabilities wired through the shim: the FBP **filter** (`cfunc_filter`,
//! cuFFT), parallel-beam **back-projection** (`cfunc_linerec`), **Fourier**
//! reconstruction (`cfunc_fourierrec`, via the `FourierReconstruct` capability),
//! and **elementwise** dark/flat + minus-log. The `AnalyticReconstruct`
//! capability fuses the analytic chain into a **device-resident** path:
//! `recon(Fbp/Linerec/Fourierrec, &CudaBackend)` uploads the sinogram once,
//! runs pad → filter → crop → back-projection (or pack → fourierrec → unpack)
//! all on the device, and downloads the volume once — no per-stage host copies.
//! Still on the CPU/wgpu side only: `lprec` and GPU stripe/phase.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

#[cfg(feature = "cuda")]
pub mod ffi;

use tomoxide_core::backend::{Backend, DeviceKind};
use tomoxide_core::dtype::Dtype;
#[cfg(not(feature = "cuda"))]
use tomoxide_core::error::Error;
use tomoxide_core::error::Result;

/// Handle to the CUDA backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaBackend;

impl CudaBackend {
    /// Initialise the CUDA backend.
    ///
    /// Without the `cuda` feature this always returns
    /// [`Error::BackendUnavailable`]. With the feature it probes for a device
    /// via the shim and fails if none is present.
    pub fn new() -> Result<Self> {
        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::BackendUnavailable(
                "compiled without the `cuda` feature".into(),
            ))
        }
        #[cfg(feature = "cuda")]
        {
            let count = unsafe { ffi::tomoxide_cuda_device_count() };
            if count <= 0 {
                return Err(tomoxide_core::error::Error::BackendUnavailable(
                    "no CUDA device found".into(),
                ));
            }
            Ok(CudaBackend)
        }
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }
    fn device(&self) -> DeviceKind {
        DeviceKind::Cuda
    }
    fn supports(&self, dt: Dtype) -> bool {
        // tomocupy compiles f32 and f16 (`*fp16`) kernel variants.
        matches!(dt, Dtype::F32 | Dtype::F16)
    }

    /// FBP filter on the GPU (`cfunc_filter`, cuFFT), applying the shared ramp
    /// definition so the filtered sinogram matches the CPU path.
    #[cfg(feature = "cuda")]
    fn fbp_filter(&self) -> Option<&dyn tomoxide_core::backend::FbpFilter> {
        Some(self)
    }

    /// Parallel-beam back-projection on the GPU (`cfunc_linerec`).
    #[cfg(feature = "cuda")]
    fn backprojector(&self) -> Option<&dyn tomoxide_core::backend::FilteredBackproject> {
        Some(self)
    }

    /// Fourier-gridding reconstruction on the GPU (`cfunc_fourierrec`).
    #[cfg(feature = "cuda")]
    fn fourier_reconstruct(&self) -> Option<&dyn tomoxide_core::backend::FourierReconstruct> {
        Some(self)
    }

    /// Fused on-device analytic reconstruction (filter → back-projection /
    /// fourierrec without per-stage host copies).
    #[cfg(feature = "cuda")]
    fn analytic_reconstruct(&self) -> Option<&dyn tomoxide_core::backend::AnalyticReconstruct> {
        Some(self)
    }

    /// Dark/flat correction + minus-log on the GPU.
    #[cfg(feature = "cuda")]
    fn elementwise(&self) -> Option<&dyn tomoxide_core::backend::Elementwise> {
        Some(self)
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::{ffi, CudaBackend};
    use ndarray::{Array3, Axis};
    use std::os::raw::c_void;
    use tomoxide_core::backend::{
        make_fbp_filter, Elementwise, FbpFilter, FilteredBackproject, FourierReconstruct,
    };
    use tomoxide_core::data::{Frames, Layout, Tomo, Volume};
    use tomoxide_core::error::{Error, Result};
    use tomoxide_core::geometry::{Beam, Geometry};
    use tomoxide_core::params::FilterName;

    /// RAII wrapper over a `cudaMalloc` allocation (freed on drop).
    struct DevBuf {
        ptr: *mut c_void,
        bytes: usize,
    }

    impl DevBuf {
        fn new(bytes: usize) -> Result<Self> {
            let ptr = unsafe { ffi::tomoxide_cuda_malloc(bytes) };
            if ptr.is_null() {
                return Err(Error::Backend(format!("cudaMalloc({bytes}) failed")));
            }
            Ok(DevBuf { ptr, bytes })
        }

        fn from_host_f32(data: &[f32]) -> Result<Self> {
            let bytes = std::mem::size_of_val(data);
            let buf = DevBuf::new(bytes)?;
            let rc = unsafe {
                ffi::tomoxide_cuda_memcpy_h2d(buf.ptr, data.as_ptr() as *const c_void, bytes)
            };
            if rc != 0 {
                return Err(Error::Backend(format!("cudaMemcpy H2D failed ({rc})")));
            }
            Ok(buf)
        }

        fn zeroed(bytes: usize) -> Result<Self> {
            let buf = DevBuf::new(bytes)?;
            let rc = unsafe { ffi::tomoxide_cuda_memset(buf.ptr, 0, bytes) };
            if rc != 0 {
                return Err(Error::Backend(format!("cudaMemset failed ({rc})")));
            }
            Ok(buf)
        }

        fn to_host_f32(&self, out: &mut [f32]) -> Result<()> {
            let bytes = std::mem::size_of_val(out);
            debug_assert!(bytes <= self.bytes);
            let rc = unsafe {
                ffi::tomoxide_cuda_memcpy_d2h(out.as_mut_ptr() as *mut c_void, self.ptr, bytes)
            };
            if rc != 0 {
                return Err(Error::Backend(format!("cudaMemcpy D2H failed ({rc})")));
            }
            Ok(())
        }
    }

    impl Drop for DevBuf {
        fn drop(&mut self) {
            unsafe { ffi::tomoxide_cuda_free(self.ptr) };
        }
    }

    impl FilteredBackproject for CudaBackend {
        /// Parallel-beam voxel-driven back-projection via tomocupy's
        /// `cfunc_linerec` (phi = π/2). The sinogram must already be filtered and
        /// centred on the detector midpoint (`recon` does this through the shared
        /// FBP filter), so the kernel assumes centre `n/2`. Output is
        /// `[nz, n, n]` with the kernel's y-flip and `4/nproj` scaling (tomocupy
        /// convention — a fixed handedness/scale vs the CPU back-projector, which
        /// scale-invariant correlation tests account for).
        fn backproject(
            &self,
            sino: &Tomo<f32>,
            geom: &Geometry,
            out: &mut Volume<f32>,
        ) -> Result<()> {
            if geom.beam != Beam::Parallel {
                return Err(Error::InvalidParam(
                    "cuda back-projection supports parallel beam only".into(),
                ));
            }
            let s = sino.to_layout(Layout::Sinogram); // [nz, nproj, ncols]
            let nz = s.n_rows();
            let nproj = s.n_angles();
            let ncols = s.n_cols();
            let (oz, ny, nx) = out.dims();
            if oz != nz {
                return Err(Error::ShapeMismatch {
                    expected: format!("{nz} sinogram rows"),
                    found: oz.to_string(),
                });
            }
            if ny != ncols || nx != ncols {
                return Err(Error::InvalidParam(format!(
                    "cuda back-projection needs a square grid = detector width {ncols}; got {ny}x{nx}"
                )));
            }
            let theta = &geom.angles.0;
            if theta.len() != nproj {
                return Err(Error::ShapeMismatch {
                    expected: format!("{nproj} angles"),
                    found: theta.len().to_string(),
                });
            }
            let sino_slice = s
                .array
                .as_slice()
                .ok_or_else(|| Error::InvalidParam("non-contiguous sinogram".into()))?;

            // Device buffers: filtered sinogram, theta, output volume.
            let g = DevBuf::from_host_f32(sino_slice)?;
            let theta_d = DevBuf::from_host_f32(theta)?;
            let f = DevBuf::zeroed(nz * ncols * ncols * std::mem::size_of::<f32>())?;

            // cfunc_linerec(nproj, nz, n, ncproj=nproj, ncz=nz): whole stack at once.
            let handle = unsafe { ffi::tomoxide_linerec_new(nproj, nz, ncols, nproj, nz) };
            if handle.is_null() {
                return Err(Error::Backend("cfunc_linerec allocation failed".into()));
            }
            let phi = std::f32::consts::FRAC_PI_2; // parallel beam
            unsafe {
                ffi::tomoxide_linerec_backproject(
                    handle,
                    f.ptr,
                    g.ptr,
                    theta_d.ptr as *const f32,
                    phi,
                    0,
                    std::ptr::null_mut(),
                );
            }
            let rc = unsafe { ffi::tomoxide_cuda_sync() };
            unsafe { ffi::tomoxide_linerec_free(handle) };
            if rc != 0 {
                return Err(Error::Backend(format!("cuda kernel sync failed ({rc})")));
            }

            let mut host = vec![0.0f32; nz * ncols * ncols];
            f.to_host_f32(&mut host)?;
            out.array = Array3::from_shape_vec((nz, ncols, ncols), host)
                .map_err(|e| Error::InvalidParam(format!("cuda volume shape: {e}")))?;
            Ok(())
        }
    }

    /// Complex FBP weight `w[z, k] = ramp[k]·exp(-2πi·k·δ_z/pad)/pad`, for
    /// `k in 0..ne/2+1`, `δ_z = ncols/2 − center(z)` (half spectrum ⇒ `f_k = k ≥
    /// 0`), interleaved re/im — folds the ramp, signed-frequency centre-shift
    /// phase, and `1/ne` cuFFT-inverse normalization (matches the CPU filter).
    fn build_filter_w(
        filter: &[f32],
        geom: &Geometry,
        nz: usize,
        ncols: usize,
        pad: usize,
    ) -> Vec<f32> {
        let nfreq = pad / 2 + 1;
        let half = ncols as f32 / 2.0;
        let inv_pad = 1.0f32 / pad as f32;
        let mut w = vec![0.0f32; nz * nfreq * 2];
        for z in 0..nz {
            let delta = half - geom.center.at(z);
            for (k, &fk) in filter[..nfreq].iter().enumerate() {
                let ang = -std::f32::consts::TAU * k as f32 * delta / pad as f32;
                let idx = z * nfreq + k;
                w[2 * idx] = fk * ang.cos() * inv_pad;
                w[2 * idx + 1] = fk * ang.sin() * inv_pad;
            }
        }
        w
    }

    fn ck(rc: i32, what: &str) -> Result<()> {
        if rc != 0 {
            return Err(Error::Backend(format!("cuda {what} failed ({rc})")));
        }
        Ok(())
    }

    impl tomoxide_core::backend::AnalyticReconstruct for CudaBackend {
        /// Fused, device-resident analytic reconstruction: upload the raw
        /// sinogram once, then pad → cuFFT filter → crop → back-projection
        /// (`Fbp`/`Linerec` via `cfunc_linerec`) or pack → `cfunc_fourierrec` →
        /// unpack (`Fourierrec`) — all on the device — and download the volume
        /// once. No per-stage host round-trips. Square grid (`n == ncols`).
        fn reconstruct(
            &self,
            sino: &Tomo<f32>,
            geom: &Geometry,
            algorithm: tomoxide_core::params::Algorithm,
            params: &tomoxide_core::params::ReconParams,
        ) -> Result<Volume<f32>> {
            use tomoxide_core::backend::make_fbp_filter;
            use tomoxide_core::params::Algorithm;

            if geom.beam != Beam::Parallel {
                return Err(Error::InvalidParam(
                    "cuda analytic reconstruct supports parallel beam only".into(),
                ));
            }
            let s = sino.to_layout(Layout::Sinogram); // [nz, nproj, ncols]
            let (nz, nproj, ncols) = s.array.dim();
            let n = params.num_gridx.unwrap_or(ncols);
            if n != ncols {
                return Err(Error::InvalidParam(format!(
                    "cuda analytic reconstruct needs a square grid = detector width {ncols}; got {n}"
                )));
            }
            let theta = &geom.angles.0;
            if theta.len() != nproj {
                return Err(Error::ShapeMismatch {
                    expected: format!("{nproj} angles"),
                    found: theta.len().to_string(),
                });
            }
            let raw = s
                .array
                .as_slice()
                .ok_or_else(|| Error::InvalidParam("non-contiguous sinogram".into()))?;

            let filter = make_fbp_filter(params.filter_name, ncols)?;
            let pad = filter.len();
            let pad_side = pad / 2 - ncols / 2;
            let w = build_filter_w(&filter, geom, nz, ncols, pad);
            let fsz = std::mem::size_of::<f32>();
            let null = std::ptr::null_mut::<c_void>();

            // Upload the raw sinogram + filter weights + angles ONCE.
            let sino_dev = DevBuf::from_host_f32(raw)?;
            let w_dev = DevBuf::from_host_f32(&w)?;
            let theta_dev = DevBuf::from_host_f32(theta)?;

            // pad → filter → crop, all device-resident.
            let gpad = DevBuf::zeroed(nz * nproj * pad * fsz)?;
            ck(
                unsafe {
                    ffi::tomoxide_pad(sino_dev.ptr, gpad.ptr, nz, nproj, ncols, pad, pad_side, null)
                },
                "pad",
            )?;
            let fh = unsafe { ffi::tomoxide_filter_new(nproj, nz, pad) };
            if fh.is_null() {
                return Err(Error::Backend("cfunc_filter allocation failed".into()));
            }
            unsafe { ffi::tomoxide_filter_apply(fh, gpad.ptr, w_dev.ptr, null) };
            unsafe { ffi::tomoxide_filter_free(fh) };
            let gf = DevBuf::zeroed(nz * nproj * ncols * fsz)?;
            ck(
                unsafe {
                    ffi::tomoxide_crop(gpad.ptr, gf.ptr, nz, nproj, ncols, pad, pad_side, null)
                },
                "crop",
            )?;

            let vol = match algorithm {
                Algorithm::Fbp | Algorithm::Linerec => {
                    let f = DevBuf::zeroed(nz * n * n * fsz)?;
                    let h = unsafe { ffi::tomoxide_linerec_new(nproj, nz, n, nproj, nz) };
                    if h.is_null() {
                        return Err(Error::Backend("cfunc_linerec allocation failed".into()));
                    }
                    unsafe {
                        ffi::tomoxide_linerec_backproject(
                            h,
                            f.ptr,
                            gf.ptr,
                            theta_dev.ptr as *const f32,
                            std::f32::consts::FRAC_PI_2,
                            0,
                            null,
                        );
                    }
                    unsafe { ffi::tomoxide_linerec_free(h) };
                    ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync")?;
                    let mut host = vec![0.0f32; nz * n * n];
                    f.to_host_f32(&mut host)?;
                    host
                }
                Algorithm::Fourierrec => {
                    if nz % 2 != 0 {
                        return Err(Error::InvalidParam(format!(
                            "cuda fourierrec needs an even slice count; got nz={nz}"
                        )));
                    }
                    let gc = DevBuf::zeroed(nz * nproj * ncols * fsz)?; // complex [nz/2,nproj,ncols]
                    ck(
                        unsafe { ffi::tomoxide_pack_pairs(gf.ptr, gc.ptr, nz, nproj, ncols, null) },
                        "pack",
                    )?;
                    let fc = DevBuf::zeroed(nz * n * n * fsz)?; // complex [nz/2,n,n]
                    let h = unsafe {
                        ffi::tomoxide_fourierrec_new(nproj, nz / 2, n, theta_dev.ptr as *const f32)
                    };
                    if h.is_null() {
                        return Err(Error::Backend("cfunc_fourierrec allocation failed".into()));
                    }
                    unsafe { ffi::tomoxide_fourierrec_backproject(h, fc.ptr, gc.ptr, null) };
                    unsafe { ffi::tomoxide_fourierrec_free(h) };
                    let vol_dev = DevBuf::zeroed(nz * n * n * fsz)?;
                    ck(
                        unsafe { ffi::tomoxide_unpack_pairs(fc.ptr, vol_dev.ptr, nz, n, null) },
                        "unpack",
                    )?;
                    ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync")?;
                    let mut host = vec![0.0f32; nz * n * n];
                    vol_dev.to_host_f32(&mut host)?;
                    host
                }
                other => {
                    return Err(Error::InvalidParam(format!(
                        "cuda analytic reconstruct: unsupported algorithm {other:?}"
                    )))
                }
            };

            Ok(Volume::new(
                Array3::from_shape_vec((nz, n, n), vol)
                    .map_err(|e| Error::InvalidParam(format!("cuda volume shape: {e}")))?,
            ))
        }
    }

    impl FbpFilter for CudaBackend {
        /// Shared FBP filter definition (same ramp the CPU/wgpu backends build),
        /// so the GPU filter applies an identical kernel.
        fn make_filter(&self, name: FilterName, n: usize) -> Result<Vec<f32>> {
            make_fbp_filter(name, n)
        }

        /// FBP filtering on the GPU via tomocupy's `cfunc_filter` (cuFFT R2C →
        /// ×w → C2R). The complex weight `w` folds the same ramp, signed-
        /// frequency centre-shift phase, and `1/ne` normalization the CPU
        /// `FbpFilter` uses, so the result matches the CPU filter (to the FFT
        /// f32 floor). Edge-replicate padding to `ne = filter.len()`.
        fn apply(&self, sino: &mut Tomo<f32>, filter: &[f32], geom: &Geometry) -> Result<()> {
            let pad = filter.len();
            if pad == 0 {
                return Err(Error::InvalidParam("empty filter".into()));
            }
            let orig = sino.layout;
            let s = sino.to_layout(Layout::Sinogram); // [nz, nproj, ncols]
            let (nz, nproj, ncols) = s.array.dim();
            if pad < ncols {
                return Err(Error::ShapeMismatch {
                    expected: format!(">= {ncols} (n_cols)"),
                    found: pad.to_string(),
                });
            }
            let pad_side = pad / 2 - ncols / 2;
            let src = s
                .array
                .as_slice()
                .ok_or_else(|| Error::InvalidParam("non-contiguous sinogram".into()))?;

            let w = build_filter_w(filter, geom, nz, ncols, pad);

            // Padded real sinogram [nz, nproj, pad], edge-replicated borders.
            let mut g = vec![0.0f32; nz * nproj * pad];
            for z in 0..nz {
                for p in 0..nproj {
                    let row = (z * nproj + p) * ncols;
                    let first = src[row];
                    let last = src[row + ncols - 1];
                    let dst = (z * nproj + p) * pad;
                    for x in 0..pad_side {
                        g[dst + x] = first;
                    }
                    for x in 0..ncols {
                        g[dst + pad_side + x] = src[row + x];
                    }
                    for x in (pad_side + ncols)..pad {
                        g[dst + x] = last;
                    }
                }
            }

            let g_dev = DevBuf::from_host_f32(&g)?;
            let w_dev = DevBuf::from_host_f32(&w)?;
            let handle = unsafe { ffi::tomoxide_filter_new(nproj, nz, pad) };
            if handle.is_null() {
                return Err(Error::Backend("cfunc_filter allocation failed".into()));
            }
            unsafe {
                ffi::tomoxide_filter_apply(handle, g_dev.ptr, w_dev.ptr, std::ptr::null_mut());
            }
            let rc = unsafe { ffi::tomoxide_cuda_sync() };
            unsafe { ffi::tomoxide_filter_free(handle) };
            if rc != 0 {
                return Err(Error::Backend(format!("cuda filter sync failed ({rc})")));
            }
            g_dev.to_host_f32(&mut g)?;

            // Crop the centred [pad_side, pad_side+ncols) window back to ncols.
            let mut out = vec![0.0f32; nz * nproj * ncols];
            for z in 0..nz {
                for p in 0..nproj {
                    let dst = (z * nproj + p) * ncols;
                    let srcp = (z * nproj + p) * pad + pad_side;
                    out[dst..dst + ncols].copy_from_slice(&g[srcp..srcp + ncols]);
                }
            }
            let arr = Array3::from_shape_vec((nz, nproj, ncols), out)
                .map_err(|e| Error::InvalidParam(format!("cuda filter shape: {e}")))?;
            *sino = Tomo::new(arr, Layout::Sinogram).to_layout(orig);
            Ok(())
        }
    }

    impl Elementwise for CudaBackend {
        /// `(data − mean(dark)) / max(mean(flat) − mean(dark), 1e-6)` on the GPU
        /// (tomocupy `darkflat_correction`). Frame averages and the clamped
        /// denominator are computed host-side; the per-projection broadcast runs
        /// on the device.
        fn darkflat(
            &self,
            data: &mut Tomo<f32>,
            flat: &Frames<f32>,
            dark: &Frames<f32>,
        ) -> Result<()> {
            let dark2d = dark
                .array
                .mean_axis(Axis(0))
                .ok_or_else(|| Error::InvalidParam("empty dark stack".into()))?;
            let flat2d = flat
                .array
                .mean_axis(Axis(0))
                .ok_or_else(|| Error::InvalidParam("empty flat stack".into()))?;
            let mut denom = &flat2d - &dark2d;
            denom.mapv_inplace(|v| if v.abs() < 1e-6 { 1.0 } else { v });

            let restore = data.layout == Layout::Sinogram;
            if restore {
                *data = data.to_layout(Layout::Projection);
            }
            let (nproj, nz, nx) = data.array.dim();
            {
                let host = data
                    .array
                    .as_slice()
                    .ok_or_else(|| Error::InvalidParam("non-contiguous data".into()))?;
                let d_data = DevBuf::from_host_f32(host)?;
                let d_dark = DevBuf::from_host_f32(
                    dark2d.as_slice().expect("contiguous dark2d"),
                )?;
                let d_denom = DevBuf::from_host_f32(
                    denom.as_slice().expect("contiguous denom"),
                )?;
                let rc = unsafe {
                    ffi::tomoxide_darkflat(
                        d_data.ptr,
                        d_dark.ptr,
                        d_denom.ptr,
                        nproj,
                        nz,
                        nx,
                        std::ptr::null_mut(),
                    )
                };
                if rc != 0 {
                    return Err(Error::Backend(format!("cuda darkflat failed ({rc})")));
                }
                let sync = unsafe { ffi::tomoxide_cuda_sync() };
                if sync != 0 {
                    return Err(Error::Backend(format!("cuda darkflat sync failed ({sync})")));
                }
                let out = data
                    .array
                    .as_slice_mut()
                    .ok_or_else(|| Error::InvalidParam("non-contiguous data".into()))?;
                d_data.to_host_f32(out)?;
            }
            if restore {
                *data = data.to_layout(Layout::Sinogram);
            }
            Ok(())
        }

        /// In-place `−ln(max(x, 1e-6))` (non-finite → 0) on the GPU.
        fn minus_log(&self, data: &mut Tomo<f32>) -> Result<()> {
            let n = data.array.len();
            let host = data
                .array
                .as_slice()
                .ok_or_else(|| Error::InvalidParam("non-contiguous data".into()))?;
            let d_data = DevBuf::from_host_f32(host)?;
            let rc = unsafe { ffi::tomoxide_minuslog(d_data.ptr, n, std::ptr::null_mut()) };
            if rc != 0 {
                return Err(Error::Backend(format!("cuda minus_log failed ({rc})")));
            }
            let sync = unsafe { ffi::tomoxide_cuda_sync() };
            if sync != 0 {
                return Err(Error::Backend(format!("cuda minus_log sync failed ({sync})")));
            }
            let out = data
                .array
                .as_slice_mut()
                .ok_or_else(|| Error::InvalidParam("non-contiguous data".into()))?;
            d_data.to_host_f32(out)?;
            Ok(())
        }
    }

    impl FourierReconstruct for CudaBackend {
        /// Fourier-gridding reconstruction via tomocupy's `cfunc_fourierrec`.
        ///
        /// The kernel processes `nz/2` **complex** slice-pairs: slice `s` and
        /// `s + nz/2` of the filtered sinogram are packed into the real/imag of
        /// one complex slice (tomocupy `FourierRec.backprojection`), and the
        /// output volume is de-interleaved the same way. Requires an even slice
        /// count and a square grid `n == ncols`. The FBP filter (incl. the
        /// rotation-centre shift) is applied by the caller, so the kernel is
        /// centre-agnostic. Output carries tomocupy's grid convention (verified
        /// by scale/flip-invariant correlation vs the CPU `fourierrec`).
        fn reconstruct(
            &self,
            filtered: &Tomo<f32>,
            geom: &Geometry,
            n: usize,
        ) -> Result<Volume<f32>> {
            let s = filtered.to_layout(Layout::Sinogram); // [nz, nproj, ncols]
            let nz = s.n_rows();
            let nproj = s.n_angles();
            let ncols = s.n_cols();
            if nz % 2 != 0 {
                return Err(Error::InvalidParam(format!(
                    "cuda fourierrec needs an even slice count (complex pairing); got nz={nz}"
                )));
            }
            if n != ncols {
                return Err(Error::InvalidParam(format!(
                    "cuda fourierrec needs a square grid = detector width {ncols}; got {n}"
                )));
            }
            let theta = &geom.angles.0;
            if theta.len() != nproj {
                return Err(Error::ShapeMismatch {
                    expected: format!("{nproj} angles"),
                    found: theta.len().to_string(),
                });
            }
            let src = s
                .array
                .as_slice()
                .ok_or_else(|| Error::InvalidParam("non-contiguous sinogram".into()))?;

            // Pack slice pairs (s, s+nz/2) into interleaved complex: for each
            // complex element [s, p, x], re = filtered[s], im = filtered[s+nz/2].
            let half = nz / 2;
            let mut g = vec![0.0f32; nz * nproj * ncols];
            for sp in 0..half {
                for p in 0..nproj {
                    for x in 0..ncols {
                        let idx = sp * nproj * ncols + p * ncols + x;
                        g[2 * idx] = src[sp * nproj * ncols + p * ncols + x];
                        g[2 * idx + 1] = src[(sp + half) * nproj * ncols + p * ncols + x];
                    }
                }
            }

            let g_dev = DevBuf::from_host_f32(&g)?;
            let theta_dev = DevBuf::from_host_f32(theta)?;
            // Output: complex [nz/2, n, n] = nz*n*n floats.
            let f_dev = DevBuf::zeroed(nz * n * n * std::mem::size_of::<f32>())?;

            let handle = unsafe {
                ffi::tomoxide_fourierrec_new(nproj, half, n, theta_dev.ptr as *const f32)
            };
            if handle.is_null() {
                return Err(Error::Backend("cfunc_fourierrec allocation failed".into()));
            }
            unsafe {
                ffi::tomoxide_fourierrec_backproject(
                    handle,
                    f_dev.ptr,
                    g_dev.ptr,
                    std::ptr::null_mut(),
                );
            }
            let rc = unsafe { ffi::tomoxide_cuda_sync() };
            unsafe { ffi::tomoxide_fourierrec_free(handle) };
            if rc != 0 {
                return Err(Error::Backend(format!("cuda fourierrec sync failed ({rc})")));
            }

            let mut fbuf = vec![0.0f32; nz * n * n];
            f_dev.to_host_f32(&mut fbuf)?;
            // De-interleave: re → slice sp, im → slice sp+nz/2.
            let mut vol = vec![0.0f32; nz * n * n];
            for sp in 0..half {
                for y in 0..n {
                    for x in 0..n {
                        let idx = sp * n * n + y * n + x;
                        vol[sp * n * n + y * n + x] = fbuf[2 * idx];
                        vol[(sp + half) * n * n + y * n + x] = fbuf[2 * idx + 1];
                    }
                }
            }
            Ok(Volume::new(
                Array3::from_shape_vec((nz, n, n), vol)
                    .map_err(|e| Error::InvalidParam(format!("cuda fourierrec shape: {e}")))?,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn unavailable_without_feature() {
        assert!(matches!(
            CudaBackend::new(),
            Err(Error::BackendUnavailable(_))
        ));
    }

    #[test]
    fn advertises_cuda_device() {
        let b = CudaBackend::default();
        assert_eq!(b.name(), "cuda");
        assert_eq!(b.device(), DeviceKind::Cuda);
        assert!(b.supports(Dtype::F16));
    }
}
