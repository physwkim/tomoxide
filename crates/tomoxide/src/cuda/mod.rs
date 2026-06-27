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
//! A cuFFT-backed [`Fft`](crate::backend::Fft) capability additionally
//! composes every Fft-based method onto CUDA through the backend-agnostic code:
//! `gridrec`, `lprec`, and Paganin/GPaganin/Farago phase all run on the GPU.
//! Still CPU-only: stripe removal (`remove_stripe` takes no backend).
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

#[cfg(feature = "cuda")]
pub mod ffi;

use crate::backend::{Backend, DeviceKind};
use crate::dtype::Dtype;
#[cfg(not(feature = "cuda"))]
use crate::error::Error;
use crate::error::Result;

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
                return Err(crate::error::Error::BackendUnavailable(
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
    fn fbp_filter(&self) -> Option<&dyn crate::backend::FbpFilter> {
        Some(self)
    }

    /// Parallel-beam back-projection on the GPU (`cfunc_linerec`).
    #[cfg(feature = "cuda")]
    fn backprojector(&self) -> Option<&dyn crate::backend::FilteredBackproject> {
        Some(self)
    }

    /// Fourier-gridding reconstruction on the GPU (`cfunc_fourierrec`).
    #[cfg(feature = "cuda")]
    fn fourier_reconstruct(&self) -> Option<&dyn crate::backend::FourierReconstruct> {
        Some(self)
    }

    /// Fused on-device analytic reconstruction (filter → back-projection /
    /// fourierrec without per-stage host copies).
    #[cfg(feature = "cuda")]
    fn analytic_reconstruct(&self) -> Option<&dyn crate::backend::AnalyticReconstruct> {
        Some(self)
    }

    /// Dark/flat correction + minus-log on the GPU.
    #[cfg(feature = "cuda")]
    fn elementwise(&self) -> Option<&dyn crate::backend::Elementwise> {
        Some(self)
    }

    /// Batched C2C FFT (cuFFT). Implementing this composes every Fft-based
    /// method (gridrec, lprec, Paganin/GPaganin/Farago phase, Fourier-wavelet
    /// stripe) onto CUDA through the backend-agnostic code.
    #[cfg(feature = "cuda")]
    fn fft(&self) -> Option<&dyn crate::backend::Fft> {
        Some(self)
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::{ffi, CudaBackend};
    use crate::backend::{
        make_fbp_filter, Elementwise, FbpFilter, FilteredBackproject, FourierReconstruct,
        StreamingAnalytic,
    };
    use crate::data::{Frames, Layout, Tomo, Volume};
    use crate::error::{Error, Result};
    use crate::geometry::{Beam, Geometry};
    use crate::params::FilterName;
    use ndarray::{Array3, ArrayViewMut2, Axis};
    use rayon::prelude::*;
    use rayon::{ThreadPool, ThreadPoolBuilder};
    use std::os::raw::c_void;
    use std::sync::{Condvar, Mutex, OnceLock};

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

        /// Upload `data` (host f32) as half-precision: each element is rounded to
        /// `f16` so the device buffer holds 2 bytes/element. Used for the f16
        /// analytic path (sinogram and filter weights).
        fn from_host_f16(data: &[f32]) -> Result<Self> {
            // The f32→f16 round of a full sinogram/volume is tens-to-hundreds of
            // millions of scalars; single-threaded it dominated the f16 path's
            // overhead. par_iter().collect() keeps element order.
            let h: Vec<half::f16> = data.par_iter().map(|&x| half::f16::from_f32(x)).collect();
            let bytes = std::mem::size_of_val(h.as_slice());
            let buf = DevBuf::new(bytes)?;
            let rc = unsafe {
                ffi::tomoxide_cuda_memcpy_h2d(buf.ptr, h.as_ptr() as *const c_void, bytes)
            };
            if rc != 0 {
                return Err(Error::Backend(format!(
                    "cudaMemcpy H2D (f16) failed ({rc})"
                )));
            }
            Ok(buf)
        }

        /// Upload host f32 `data` into this **already-allocated** buffer (no
        /// realloc). Used by the streaming reconstructor, which reuses one buffer
        /// across chunks. `data` must fit (`len*4 ≤ self.bytes`).
        fn copy_from_host_f32(&self, data: &[f32]) -> Result<()> {
            let bytes = std::mem::size_of_val(data);
            debug_assert!(bytes <= self.bytes);
            let rc = unsafe {
                ffi::tomoxide_cuda_memcpy_h2d(self.ptr, data.as_ptr() as *const c_void, bytes)
            };
            if rc != 0 {
                return Err(Error::Backend(format!("cudaMemcpy H2D failed ({rc})")));
            }
            Ok(())
        }

        /// Upload host f32 `data` rounded to `f16` into this already-allocated
        /// buffer (no realloc). f16 streaming counterpart of
        /// [`copy_from_host_f32`]; `len*2 ≤ self.bytes`.
        fn copy_from_host_f16(&self, data: &[f32]) -> Result<()> {
            let h: Vec<half::f16> = data.par_iter().map(|&x| half::f16::from_f32(x)).collect();
            let bytes = std::mem::size_of_val(h.as_slice());
            debug_assert!(bytes <= self.bytes);
            let rc = unsafe {
                ffi::tomoxide_cuda_memcpy_h2d(self.ptr, h.as_ptr() as *const c_void, bytes)
            };
            if rc != 0 {
                return Err(Error::Backend(format!(
                    "cudaMemcpy H2D (f16) failed ({rc})"
                )));
            }
            Ok(())
        }

        /// Download `count` half-precision elements and widen them back to f32.
        fn to_host_f16_as_f32(&self, count: usize) -> Result<Vec<f32>> {
            let mut h = vec![half::f16::from_f32(0.0); count];
            let bytes = std::mem::size_of_val(h.as_slice());
            debug_assert!(bytes <= self.bytes);
            let rc = unsafe {
                ffi::tomoxide_cuda_memcpy_d2h(h.as_mut_ptr() as *mut c_void, self.ptr, bytes)
            };
            if rc != 0 {
                return Err(Error::Backend(format!(
                    "cudaMemcpy D2H (f16) failed ({rc})"
                )));
            }
            // Widen f16→f32 in parallel (see from_host_f16); order preserved.
            Ok(h.par_iter().map(|x| x.to_f32()).collect())
        }
    }

    impl Drop for DevBuf {
        fn drop(&mut self) {
            unsafe { ffi::tomoxide_cuda_free(self.ptr) };
        }
    }

    /// An owned CUDA stream. Work issued on it runs in order but overlaps work on
    /// other streams; the async pipeline uses one per double-buffer slot so a
    /// chunk's compute can run while another slot's H2D/D2H copies are in flight.
    struct Stream {
        ptr: *mut c_void,
    }

    impl Stream {
        fn new() -> Result<Self> {
            let ptr = unsafe { ffi::tomoxide_cuda_stream_create() };
            if ptr.is_null() {
                return Err(Error::Backend("cudaStreamCreate failed".into()));
            }
            Ok(Stream { ptr })
        }

        /// Block the calling thread until every operation on this stream finishes.
        fn sync(&self) -> Result<()> {
            let rc = unsafe { ffi::tomoxide_cuda_stream_sync(self.ptr) };
            if rc != 0 {
                return Err(Error::Backend(format!(
                    "cudaStreamSynchronize failed ({rc})"
                )));
            }
            Ok(())
        }
    }

    impl Drop for Stream {
        fn drop(&mut self) {
            unsafe { ffi::tomoxide_cuda_stream_destroy(self.ptr) };
        }
    }

    /// A page-locked (pinned) host buffer of `f32`. Async H2D/D2H copies only
    /// overlap kernel execution when their host side is pinned — pageable memory
    /// forces the driver into a synchronous staged copy, defeating the pipeline.
    struct PinnedBuf<T = f32> {
        ptr: *mut c_void,
        len: usize, // elements
        _marker: std::marker::PhantomData<T>,
    }

    impl<T: Copy> PinnedBuf<T> {
        fn new(len: usize) -> Result<Self> {
            let ptr = unsafe { ffi::tomoxide_cuda_host_alloc(len * std::mem::size_of::<T>()) };
            if ptr.is_null() {
                return Err(Error::Backend(format!("cudaHostAlloc({len} elems) failed")));
            }
            Ok(PinnedBuf {
                ptr,
                len,
                _marker: std::marker::PhantomData,
            })
        }

        fn as_mut_slice(&mut self) -> &mut [T] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) }
        }

        fn as_slice(&self) -> &[T] {
            unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
        }
    }

    impl<T> Drop for PinnedBuf<T> {
        fn drop(&mut self) {
            unsafe { ffi::tomoxide_cuda_host_free(self.ptr) };
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
            let s = sino.as_layout(Layout::Sinogram); // [nz, nproj, ncols], no copy if already
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

    /// Pad → cuFFT filter → crop → `cfunc_linerec` back-projection for one
    /// z-chunk, entirely on the **current** device. Every device buffer is local
    /// to the calling thread, so this is safe to run on many devices at once (one
    /// thread per GPU, each having called `cudaSetDevice`). Returns the chunk's
    /// volume `[nz, n, n]`. `w` is the filter weight slice for *these* z rows.
    #[allow(clippy::too_many_arguments)]
    fn analytic_fbp_chunk(
        raw: &[f32],
        w: &[f32],
        theta: &[f32],
        nz: usize,
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        pad_side: usize,
    ) -> Result<Vec<f32>> {
        let fsz = std::mem::size_of::<f32>();
        let null = std::ptr::null_mut::<c_void>();
        let sino_dev = DevBuf::from_host_f32(raw)?;
        let w_dev = DevBuf::from_host_f32(w)?;
        let theta_dev = DevBuf::from_host_f32(theta)?;
        let gpad = DevBuf::zeroed(nz * nproj * pad * fsz)?;
        ck(
            unsafe {
                ffi::tomoxide_pad(
                    sino_dev.ptr,
                    gpad.ptr,
                    nz,
                    nproj,
                    ncols,
                    pad,
                    pad_side,
                    null,
                )
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
            unsafe { ffi::tomoxide_crop(gpad.ptr, gf.ptr, nz, nproj, ncols, pad, pad_side, null) },
            "crop",
        )?;
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
        Ok(host)
    }

    /// Half-precision (`Dtype::F16`) FBP/Linerec on the **current** device, whole
    /// stack in one chunk. Mirrors [`analytic_fbp_chunk`] but the sinogram, filter
    /// weights, padded/filtered buffers and volume are `f16` (2 bytes/element) and
    /// the filter runs a half-precision cuFFT — so the padded width `pad` MUST be a
    /// power of two (enforced by the caller). theta stays f32. The result is the
    /// f32-widened volume; because the half cuFFT and the half back-projection
    /// accumulate in 16-bit, it matches the f32 path only by correlation, not
    /// bit-exactly (tomocupy `--dtype float16`).
    #[allow(clippy::too_many_arguments)]
    fn analytic_fbp_chunk_f16(
        raw: &[f32],
        w: &[f32],
        theta: &[f32],
        nz: usize,
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        pad_side: usize,
    ) -> Result<Vec<f32>> {
        let hsz = std::mem::size_of::<half::f16>();
        let null = std::ptr::null_mut::<c_void>();
        let sino_dev = DevBuf::from_host_f16(raw)?;
        let w_dev = DevBuf::from_host_f16(w)?;
        let theta_dev = DevBuf::from_host_f32(theta)?;
        let gpad = DevBuf::zeroed(nz * nproj * pad * hsz)?;
        ck(
            unsafe {
                ffi::tomoxide_pad_fp16(
                    sino_dev.ptr,
                    gpad.ptr,
                    nz,
                    nproj,
                    ncols,
                    pad,
                    pad_side,
                    null,
                )
            },
            "pad_fp16",
        )?;
        let fh = unsafe { ffi::tomoxide_filter_fp16_new(nproj, nz, pad) };
        if fh.is_null() {
            return Err(Error::Backend(
                "cfunc_filter (f16) allocation failed".into(),
            ));
        }
        unsafe { ffi::tomoxide_filter_fp16_apply(fh, gpad.ptr, w_dev.ptr, null) };
        unsafe { ffi::tomoxide_filter_fp16_free(fh) };
        let gf = DevBuf::zeroed(nz * nproj * ncols * hsz)?;
        ck(
            unsafe {
                ffi::tomoxide_crop_fp16(gpad.ptr, gf.ptr, nz, nproj, ncols, pad, pad_side, null)
            },
            "crop_fp16",
        )?;
        let f = DevBuf::zeroed(nz * n * n * hsz)?;
        let h = unsafe { ffi::tomoxide_linerec_fp16_new(nproj, nz, n, nproj, nz) };
        if h.is_null() {
            return Err(Error::Backend(
                "cfunc_linerec (f16) allocation failed".into(),
            ));
        }
        unsafe {
            ffi::tomoxide_linerec_fp16_backproject(
                h,
                f.ptr,
                gf.ptr,
                theta_dev.ptr as *const f32,
                std::f32::consts::FRAC_PI_2,
                0,
                null,
            );
        }
        unsafe { ffi::tomoxide_linerec_fp16_free(h) };
        ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync")?;
        f.to_host_f16_as_f32(nz * n * n)
    }

    /// Handle-reusing fused FBP/Linerec reconstructor for streaming
    /// ([`StreamingAnalytic`]). The cuFFT filter plan (`filt`), the back-projection
    /// handle (`lrec`; for f16 this owns the layered texture array), the device
    /// buffers and the uploaded `theta` live for the whole stream, so an N-chunk
    /// job pays that setup **once** instead of the per-chunk new/free that
    /// [`analytic_fbp_chunk`]/[`analytic_fbp_chunk_f16`] do. Buffers and handles are
    /// sized to `max_nz` (the first/largest chunk); a smaller trailing chunk reuses
    /// them zero-padded to `max_nz` so the cuFFT batch stays fixed. Holds raw device
    /// pointers and is created/driven on a single compute thread — `!Send`/`!Sync`
    /// by construction (matches the per-thread-device contract of the chunk fns).
    struct CudaFbpStream {
        f16: bool,
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        pad_side: usize,
        max_nz: usize,
        filter: Vec<f32>, // ramp kernel (length `pad`), built once
        sino: DevBuf,
        gpad: DevBuf,
        gf: DevBuf,
        f: DevBuf,
        w: DevBuf,
        theta: DevBuf,
        // f16 only: device-side f32 staging so the host uploads/downloads f32 and
        // the f32↔f16 cast runs on the GPU (`f2h_ker`/`h2f_ker`), instead of the
        // host rayon convert. `sino_f32` receives the H2D'd f32 sinogram (cast →
        // `sino`); `f_f32` receives the GPU-cast f32 volume (← `f`) before D2H. None
        // on the f32 path, which needs no cast. Mirrors tomocupy's GPU-side astype.
        sino_f32: Option<DevBuf>,
        f_f32: Option<DevBuf>,
        filt: *mut c_void,
        lrec: *mut c_void,
    }

    impl CudaFbpStream {
        /// Allocate the persistent buffers and `cfunc_filter`/`cfunc_linerec`
        /// handles for a `max_nz`-slice chunk. `filter` is the ramp kernel
        /// (`make_fbp_filter`), `theta` the chunk-invariant angles. The current
        /// device must already be selected (the caller binds it).
        fn new(
            filter: Vec<f32>,
            theta: &[f32],
            ncols: usize,
            n: usize,
            max_nz: usize,
            f16: bool,
        ) -> Result<Self> {
            let nproj = theta.len();
            let pad = filter.len();
            let pad_side = pad / 2 - ncols / 2;
            let esz = if f16 {
                std::mem::size_of::<half::f16>()
            } else {
                std::mem::size_of::<f32>()
            };
            let nfreq2 = (pad / 2 + 1) * 2;
            let sino = DevBuf::zeroed(max_nz * nproj * ncols * esz)?;
            let gpad = DevBuf::zeroed(max_nz * nproj * pad * esz)?;
            let gf = DevBuf::zeroed(max_nz * nproj * ncols * esz)?;
            let f = DevBuf::zeroed(max_nz * n * n * esz)?;
            let w = DevBuf::zeroed(max_nz * nfreq2 * esz)?;
            let theta_dev = DevBuf::from_host_f32(theta)?;
            // f16: device f32 staging for the GPU-side cast (see field docs).
            let fsz = std::mem::size_of::<f32>();
            let (sino_f32, f_f32) = if f16 {
                (
                    Some(DevBuf::zeroed(max_nz * nproj * ncols * fsz)?),
                    Some(DevBuf::zeroed(max_nz * n * n * fsz)?),
                )
            } else {
                (None, None)
            };
            let (filt, lrec) = unsafe {
                if f16 {
                    (
                        ffi::tomoxide_filter_fp16_new(nproj, max_nz, pad),
                        ffi::tomoxide_linerec_fp16_new(nproj, max_nz, n, nproj, max_nz),
                    )
                } else {
                    (
                        ffi::tomoxide_filter_new(nproj, max_nz, pad),
                        ffi::tomoxide_linerec_new(nproj, max_nz, n, nproj, max_nz),
                    )
                }
            };
            if filt.is_null() || lrec.is_null() {
                // Free whichever allocation succeeded so a partial failure leaks
                // nothing (the Drop guard only runs on a fully-built value).
                unsafe {
                    if !filt.is_null() {
                        if f16 {
                            ffi::tomoxide_filter_fp16_free(filt)
                        } else {
                            ffi::tomoxide_filter_free(filt)
                        }
                    }
                    if !lrec.is_null() {
                        if f16 {
                            ffi::tomoxide_linerec_fp16_free(lrec)
                        } else {
                            ffi::tomoxide_linerec_free(lrec)
                        }
                    }
                }
                return Err(Error::Backend(
                    "cuda streaming reconstructor: cfunc handle allocation failed".into(),
                ));
            }
            Ok(Self {
                f16,
                nproj,
                ncols,
                n,
                pad,
                pad_side,
                max_nz,
                filter,
                sino,
                gpad,
                gf,
                f,
                w,
                theta: theta_dev,
                sino_f32,
                f_f32,
                filt,
                lrec,
            })
        }
    }

    impl Drop for CudaFbpStream {
        fn drop(&mut self) {
            unsafe {
                if self.f16 {
                    ffi::tomoxide_filter_fp16_free(self.filt);
                    ffi::tomoxide_linerec_fp16_free(self.lrec);
                } else {
                    ffi::tomoxide_filter_free(self.filt);
                    ffi::tomoxide_linerec_free(self.lrec);
                }
            }
        }
    }

    impl StreamingAnalytic for CudaFbpStream {
        fn reconstruct_chunk(&mut self, sino: &Tomo<f32>, geom: &Geometry) -> Result<Volume<f32>> {
            let s = sino.as_layout(Layout::Sinogram); // [nz, nproj, ncols]
            let (nz, nproj, ncols) = s.array.dim();
            if nproj != self.nproj || ncols != self.ncols {
                return Err(Error::ShapeMismatch {
                    expected: format!("nproj={} ncols={}", self.nproj, self.ncols),
                    found: format!("nproj={nproj} ncols={ncols}"),
                });
            }
            if nz > self.max_nz {
                return Err(Error::InvalidParam(format!(
                    "streaming reconstruct_chunk: nz={nz} exceeds max_nz={}",
                    self.max_nz
                )));
            }
            let std = s.array.as_standard_layout();
            let raw = std
                .as_slice()
                .ok_or_else(|| Error::InvalidParam("non-contiguous sinogram".into()))?;
            let w_host = build_filter_w(&self.filter, geom, nz, ncols, self.pad);
            let null = std::ptr::null_mut::<c_void>();
            let partial = nz < self.max_nz;
            // Zero the unused tail of a partial trailing chunk so the always-`max_nz`
            // kernels and the fixed cuFFT batch see zeros there (→ zero output we
            // drop); full chunks overwrite the whole buffer so they skip the memset.
            if partial {
                ck(
                    unsafe { ffi::tomoxide_cuda_memset(self.sino.ptr, 0, self.sino.bytes) },
                    "memset sino",
                )?;
                ck(
                    unsafe { ffi::tomoxide_cuda_memset(self.w.ptr, 0, self.w.bytes) },
                    "memset w",
                )?;
            }
            if self.f16 {
                // Upload f32 and cast f32→f16 on the GPU (no host rayon convert).
                // For a partial chunk only the valid `nz` rows are uploaded+cast;
                // the memset above already zeroed `sino`'s tail. `w` is tiny
                // (nz·nfreq2) so it keeps the host convert.
                let sino_f32 = self.sino_f32.as_ref().expect("f16 path has sino_f32");
                sino_f32.copy_from_host_f32(raw)?;
                ck(
                    unsafe {
                        ffi::tomoxide_cast_f32_to_f16(
                            sino_f32.ptr,
                            self.sino.ptr,
                            nz * nproj * ncols,
                            null,
                        )
                    },
                    "cast f32->f16 sino",
                )?;
                self.w.copy_from_host_f16(&w_host)?;
            } else {
                self.sino.copy_from_host_f32(raw)?;
                self.w.copy_from_host_f32(&w_host)?;
            }
            // pad → cuFFT filter → crop → back-project, all at the handle's `max_nz`
            // batch. `cfunc_linerec` accumulates into `f`, so zero it each chunk.
            let m = self.max_nz;
            let (pad, ps, n) = (self.pad, self.pad_side, self.n);
            if self.f16 {
                ck(
                    unsafe {
                        ffi::tomoxide_pad_fp16(
                            self.sino.ptr,
                            self.gpad.ptr,
                            m,
                            nproj,
                            ncols,
                            pad,
                            ps,
                            null,
                        )
                    },
                    "pad_fp16",
                )?;
                unsafe {
                    ffi::tomoxide_filter_fp16_apply(self.filt, self.gpad.ptr, self.w.ptr, null)
                };
                ck(
                    unsafe {
                        ffi::tomoxide_crop_fp16(
                            self.gpad.ptr,
                            self.gf.ptr,
                            m,
                            nproj,
                            ncols,
                            pad,
                            ps,
                            null,
                        )
                    },
                    "crop_fp16",
                )?;
                ck(
                    unsafe { ffi::tomoxide_cuda_memset(self.f.ptr, 0, self.f.bytes) },
                    "memset f f16",
                )?;
                unsafe {
                    ffi::tomoxide_linerec_fp16_backproject(
                        self.lrec,
                        self.f.ptr,
                        self.gf.ptr,
                        self.theta.ptr as *const f32,
                        std::f32::consts::FRAC_PI_2,
                        0,
                        null,
                    );
                }
                // Cast the f16 volume to f32 on the GPU, then D2H f32 (no host widen).
                let f_f32 = self.f_f32.as_ref().expect("f16 path has f_f32");
                ck(
                    unsafe {
                        ffi::tomoxide_cast_f16_to_f32(self.f.ptr, f_f32.ptr, nz * n * n, null)
                    },
                    "cast f16->f32 vol",
                )?;
                ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync")?;
                let mut host = vec![0.0f32; nz * n * n];
                f_f32.to_host_f32(&mut host)?;
                Ok(Volume::new(
                    Array3::from_shape_vec((nz, n, n), host)
                        .expect("nz*n*n volume length matches shape"),
                ))
            } else {
                ck(
                    unsafe {
                        ffi::tomoxide_pad(
                            self.sino.ptr,
                            self.gpad.ptr,
                            m,
                            nproj,
                            ncols,
                            pad,
                            ps,
                            null,
                        )
                    },
                    "pad",
                )?;
                unsafe { ffi::tomoxide_filter_apply(self.filt, self.gpad.ptr, self.w.ptr, null) };
                ck(
                    unsafe {
                        ffi::tomoxide_crop(
                            self.gpad.ptr,
                            self.gf.ptr,
                            m,
                            nproj,
                            ncols,
                            pad,
                            ps,
                            null,
                        )
                    },
                    "crop",
                )?;
                ck(
                    unsafe { ffi::tomoxide_cuda_memset(self.f.ptr, 0, self.f.bytes) },
                    "memset f",
                )?;
                unsafe {
                    ffi::tomoxide_linerec_backproject(
                        self.lrec,
                        self.f.ptr,
                        self.gf.ptr,
                        self.theta.ptr as *const f32,
                        std::f32::consts::FRAC_PI_2,
                        0,
                        null,
                    );
                }
                ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync")?;
                let mut host = vec![0.0f32; nz * n * n];
                self.f.to_host_f32(&mut host)?;
                Ok(Volume::new(
                    Array3::from_shape_vec((nz, n, n), host)
                        .expect("nz*n*n volume length matches shape"),
                ))
            }
        }
    }

    /// State for one double-buffer slot of the async FBP pipeline: its own stream,
    /// device buffers sized to the largest chunk, pinned host staging for the
    /// sino-in / filter-weight / volume-out copies, and the per-chunk `cfunc_*`
    /// handles plus the chunk index currently in flight (released on drain).
    struct FbpSlot {
        stream: Stream,
        sino: DevBuf,
        gpad: DevBuf,
        gf: DevBuf,
        f: DevBuf,
        w: DevBuf,
        pin_in: PinnedBuf,
        pin_w: PinnedBuf,
        pin_out: PinnedBuf,
        inflight: Option<usize>,
        filt: *mut c_void,
        lrec: *mut c_void,
    }

    /// Asynchronous, double-buffered Fbp/Linerec back-projection over `chunks`
    /// (each `(z0, len)`, `len ≥ 2`), implementing the tomocupy JSR 2023 (Fig. 1)
    /// overlap: while chunk *k* computes on the GPU, chunk *k+1* is being uploaded
    /// (H2D) and chunk *k−1* downloaded (D2H). Two slots (`k % 2`) ping-pong; each
    /// slot's whole `H2D → pad → filter → crop → linerec → D2H` sequence runs on
    /// its own stream (ordering the data dependency), and the two streams plus
    /// pinned host staging let the copy engines and the SMs run concurrently.
    /// Numerically equivalent to [`analytic_fbp_chunk`] per chunk, so the result
    /// matches the sequential tiled path to the single-precision FFT floor.
    #[allow(clippy::too_many_arguments)]
    fn analytic_fbp_pipeline(
        raw: &[f32],
        w: &[f32],
        theta: &[f32],
        chunks: &[(usize, usize)],
        nz_total: usize,
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        pad_side: usize,
    ) -> Result<Vec<f32>> {
        let fsz = std::mem::size_of::<f32>();
        let nfreq2 = (pad / 2 + 1) * 2;
        let maxlen = chunks.iter().map(|&(_, l)| l).max().unwrap_or(0);
        if maxlen == 0 {
            return Ok(Vec::new());
        }
        // theta is read-only and shared by both streams — upload once.
        let theta_dev = DevBuf::from_host_f32(theta)?;

        let mut slots: Vec<FbpSlot> = Vec::with_capacity(2);
        for _ in 0..2 {
            slots.push(FbpSlot {
                stream: Stream::new()?,
                sino: DevBuf::new(maxlen * nproj * ncols * fsz)?,
                gpad: DevBuf::new(maxlen * nproj * pad * fsz)?,
                gf: DevBuf::new(maxlen * nproj * ncols * fsz)?,
                f: DevBuf::new(maxlen * n * n * fsz)?,
                w: DevBuf::new(maxlen * nfreq2 * fsz)?,
                pin_in: PinnedBuf::new(maxlen * nproj * ncols)?,
                pin_w: PinnedBuf::new(maxlen * nfreq2)?,
                pin_out: PinnedBuf::new(maxlen * n * n)?,
                inflight: None,
                filt: std::ptr::null_mut(),
                lrec: std::ptr::null_mut(),
            });
        }

        let mut out = vec![0.0f32; nz_total * n * n];

        for k in 0..chunks.len() {
            let s = k % 2;
            // Drain the chunk that previously held this slot (k−2): wait for its
            // stream, free its handles, copy its downloaded volume out — then the
            // buffers are free to reuse.
            if let Some(ci) = slots[s].inflight.take() {
                drain_fbp_slot(&mut slots[s], ci, chunks, n, &mut out)?;
            }

            let (z0, len) = chunks[k];
            let st = slots[s].stream.ptr;
            // Stage host inputs into this slot's pinned buffers (host→host).
            slots[s].pin_in.as_mut_slice()[..len * nproj * ncols]
                .copy_from_slice(&raw[z0 * nproj * ncols..(z0 + len) * nproj * ncols]);
            slots[s].pin_w.as_mut_slice()[..len * nfreq2]
                .copy_from_slice(&w[z0 * nfreq2..(z0 + len) * nfreq2]);

            // Async H2D, then the kernel chain, then async D2H — all ordered on the
            // slot's stream (intra-chunk dependency); cross-chunk overlap comes from
            // the other slot running on its own stream.
            ck(
                unsafe {
                    ffi::tomoxide_cuda_memcpy_h2d_async(
                        slots[s].sino.ptr,
                        slots[s].pin_in.ptr,
                        len * nproj * ncols * fsz,
                        st,
                    )
                },
                "h2d sino",
            )?;
            ck(
                unsafe {
                    ffi::tomoxide_cuda_memcpy_h2d_async(
                        slots[s].w.ptr,
                        slots[s].pin_w.ptr,
                        len * nfreq2 * fsz,
                        st,
                    )
                },
                "h2d w",
            )?;
            ck(
                unsafe {
                    ffi::tomoxide_pad(
                        slots[s].sino.ptr,
                        slots[s].gpad.ptr,
                        len,
                        nproj,
                        ncols,
                        pad,
                        pad_side,
                        st,
                    )
                },
                "pad",
            )?;
            let filt = unsafe { ffi::tomoxide_filter_new(nproj, len, pad) };
            if filt.is_null() {
                return Err(Error::Backend("cfunc_filter allocation failed".into()));
            }
            slots[s].filt = filt;
            unsafe { ffi::tomoxide_filter_apply(filt, slots[s].gpad.ptr, slots[s].w.ptr, st) };
            ck(
                unsafe {
                    ffi::tomoxide_crop(
                        slots[s].gpad.ptr,
                        slots[s].gf.ptr,
                        len,
                        nproj,
                        ncols,
                        pad,
                        pad_side,
                        st,
                    )
                },
                "crop",
            )?;
            // linerec accumulates into `f`, so zero it first (on the stream).
            ck(
                unsafe {
                    ffi::tomoxide_cuda_memset_async(slots[s].f.ptr, 0, len * n * n * fsz, st)
                },
                "memset f",
            )?;
            let lrec = unsafe { ffi::tomoxide_linerec_new(nproj, len, n, nproj, len) };
            if lrec.is_null() {
                return Err(Error::Backend("cfunc_linerec allocation failed".into()));
            }
            slots[s].lrec = lrec;
            unsafe {
                ffi::tomoxide_linerec_backproject(
                    lrec,
                    slots[s].f.ptr,
                    slots[s].gf.ptr,
                    theta_dev.ptr as *const f32,
                    std::f32::consts::FRAC_PI_2,
                    0,
                    st,
                );
            }
            ck(
                unsafe {
                    ffi::tomoxide_cuda_memcpy_d2h_async(
                        slots[s].pin_out.ptr,
                        slots[s].f.ptr,
                        len * n * n * fsz,
                        st,
                    )
                },
                "d2h vol",
            )?;
            slots[s].inflight = Some(k);
        }

        // Drain the last (≤2) in-flight chunks.
        for slot in &mut slots {
            if let Some(ci) = slot.inflight.take() {
                drain_fbp_slot(slot, ci, chunks, n, &mut out)?;
            }
        }
        Ok(out)
    }

    /// Finish the chunk in flight on `slot`: wait for its stream, free its
    /// per-chunk `cfunc_*` handles, and copy its downloaded volume from pinned
    /// host memory into `out` at the chunk's z-range.
    fn drain_fbp_slot(
        slot: &mut FbpSlot,
        ci: usize,
        chunks: &[(usize, usize)],
        n: usize,
        out: &mut [f32],
    ) -> Result<()> {
        slot.stream.sync()?;
        unsafe {
            ffi::tomoxide_filter_free(slot.filt);
            ffi::tomoxide_linerec_free(slot.lrec);
        }
        slot.filt = std::ptr::null_mut();
        slot.lrec = std::ptr::null_mut();
        let (z0, len) = chunks[ci];
        out[z0 * n * n..(z0 + len) * n * n]
            .copy_from_slice(&slot.pin_out.as_slice()[..len * n * n]);
        Ok(())
    }

    /// Memory-safe driver for the fused Fbp/Linerec path: split the z-stack into
    /// tiles sized by [`fbp_tile_z`] (free device memory + the 32-bit index
    /// ceiling) and run [`analytic_fbp_chunk`] on each, concatenating the volumes.
    /// When the whole stack already fits in one tile this is a single chunk call
    /// (numerically identical to the un-streamed path). When it tiles, the cuFFT
    /// filter batch becomes the tile size, so — like the multi-GPU split — the
    /// result shifts at the single-precision FFT floor (~1e-7) and, since the
    /// tile size tracks free device memory, is not bit-reproducible across hosts
    /// with different free memory. The current device must already be selected.
    /// Returns the volume `[nz, n, n]`.
    #[allow(clippy::too_many_arguments)]
    fn analytic_fbp_stream(
        raw: &[f32],
        w: &[f32],
        theta: &[f32],
        nz: usize,
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        pad_side: usize,
    ) -> Result<Vec<f32>> {
        // Whole stack fits in one tile at the full memory budget → no tiling and
        // no pipeline; one chunk, byte-identical to the un-streamed path.
        let tile_full = fbp_tile_z(nproj, ncols, n, pad, device_free_bytes(), false).min(nz.max(1));
        if tile_full >= nz {
            return analytic_fbp_chunk(raw, w, theta, nz, nproj, ncols, n, pad, pad_side);
        }
        // Tiling is needed. Size chunks so TWO are resident (half the memory
        // budget) and run them through the async H2D∥compute∥D2H pipeline. Chunks
        // are an even split that stays ≥2 slices (the cfunc_linerec invariant) —
        // a greedy `[tile, …, 1]` tail would silently zero its last slice.
        let tile = fbp_tile_z(nproj, ncols, n, pad, device_free_bytes() / 2, false).min(nz.max(1));
        let k = linerec_chunk_count(nz, nz.div_ceil(tile));
        let chunks = even_z_chunks(nz, k);
        analytic_fbp_pipeline(raw, w, theta, &chunks, nz, nproj, ncols, n, pad, pad_side)
    }

    /// f16 counterpart of [`FbpSlot`]: half device + pinned buffers. `theta` is
    /// shared (uploaded once) and stays f32, so it is not a slot field.
    struct FbpSlotF16 {
        stream: Stream,
        sino: DevBuf,
        gpad: DevBuf,
        gf: DevBuf,
        f: DevBuf,
        w: DevBuf,
        pin_in: PinnedBuf<half::f16>,
        pin_w: PinnedBuf<half::f16>,
        pin_out: PinnedBuf<half::f16>,
        inflight: Option<usize>,
        filt: *mut c_void,
        lrec: *mut c_void,
    }

    /// Half-precision counterpart of [`analytic_fbp_pipeline`]. Same double-
    /// buffered H2D∥compute∥D2H overlap, but device/pinned buffers are `f16` and
    /// the per-chunk host staging *converts* (f32→f16 into the pinned input, and
    /// f16→f32 out of the pinned output on drain). Those conversions — the single
    /// largest host cost of the f16 path — run on CPU worker threads while a
    /// previous chunk's kernels run on the GPU, so together with the halved
    /// transfers they are hidden behind compute. Numerically equal per chunk to
    /// [`analytic_fbp_chunk_f16`] (correlation vs f32, not bit-exact).
    #[allow(clippy::too_many_arguments)]
    fn analytic_fbp_pipeline_f16(
        raw: &[f32],
        w: &[f32],
        theta: &[f32],
        chunks: &[(usize, usize)],
        nz_total: usize,
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        pad_side: usize,
    ) -> Result<Vec<f32>> {
        let hsz = std::mem::size_of::<half::f16>();
        let nfreq2 = (pad / 2 + 1) * 2;
        let maxlen = chunks.iter().map(|&(_, l)| l).max().unwrap_or(0);
        if maxlen == 0 {
            return Ok(Vec::new());
        }
        let theta_dev = DevBuf::from_host_f32(theta)?;

        let mut slots: Vec<FbpSlotF16> = Vec::with_capacity(2);
        for _ in 0..2 {
            slots.push(FbpSlotF16 {
                stream: Stream::new()?,
                sino: DevBuf::new(maxlen * nproj * ncols * hsz)?,
                gpad: DevBuf::new(maxlen * nproj * pad * hsz)?,
                gf: DevBuf::new(maxlen * nproj * ncols * hsz)?,
                f: DevBuf::new(maxlen * n * n * hsz)?,
                w: DevBuf::new(maxlen * nfreq2 * hsz)?,
                pin_in: PinnedBuf::new(maxlen * nproj * ncols)?,
                pin_w: PinnedBuf::new(maxlen * nfreq2)?,
                pin_out: PinnedBuf::new(maxlen * n * n)?,
                inflight: None,
                filt: std::ptr::null_mut(),
                lrec: std::ptr::null_mut(),
            });
        }

        let mut out = vec![0.0f32; nz_total * n * n];

        for k in 0..chunks.len() {
            let s = k % 2;
            if let Some(ci) = slots[s].inflight.take() {
                drain_fbp_slot_f16(&mut slots[s], ci, chunks, n, &mut out)?;
            }

            let (z0, len) = chunks[k];
            let st = slots[s].stream.ptr;
            // Stage host inputs into pinned buffers, converting f32→f16 in
            // parallel. This runs while the other slot's chunk computes on the GPU.
            {
                let src = &raw[z0 * nproj * ncols..(z0 + len) * nproj * ncols];
                slots[s].pin_in.as_mut_slice()[..len * nproj * ncols]
                    .par_iter_mut()
                    .zip(src.par_iter())
                    .for_each(|(d, &x)| *d = half::f16::from_f32(x));
                let srcw = &w[z0 * nfreq2..(z0 + len) * nfreq2];
                slots[s].pin_w.as_mut_slice()[..len * nfreq2]
                    .par_iter_mut()
                    .zip(srcw.par_iter())
                    .for_each(|(d, &x)| *d = half::f16::from_f32(x));
            }

            ck(
                unsafe {
                    ffi::tomoxide_cuda_memcpy_h2d_async(
                        slots[s].sino.ptr,
                        slots[s].pin_in.ptr,
                        len * nproj * ncols * hsz,
                        st,
                    )
                },
                "h2d sino f16",
            )?;
            ck(
                unsafe {
                    ffi::tomoxide_cuda_memcpy_h2d_async(
                        slots[s].w.ptr,
                        slots[s].pin_w.ptr,
                        len * nfreq2 * hsz,
                        st,
                    )
                },
                "h2d w f16",
            )?;
            ck(
                unsafe {
                    ffi::tomoxide_pad_fp16(
                        slots[s].sino.ptr,
                        slots[s].gpad.ptr,
                        len,
                        nproj,
                        ncols,
                        pad,
                        pad_side,
                        st,
                    )
                },
                "pad_fp16",
            )?;
            let filt = unsafe { ffi::tomoxide_filter_fp16_new(nproj, len, pad) };
            if filt.is_null() {
                return Err(Error::Backend(
                    "cfunc_filter (f16) allocation failed".into(),
                ));
            }
            slots[s].filt = filt;
            unsafe { ffi::tomoxide_filter_fp16_apply(filt, slots[s].gpad.ptr, slots[s].w.ptr, st) };
            ck(
                unsafe {
                    ffi::tomoxide_crop_fp16(
                        slots[s].gpad.ptr,
                        slots[s].gf.ptr,
                        len,
                        nproj,
                        ncols,
                        pad,
                        pad_side,
                        st,
                    )
                },
                "crop_fp16",
            )?;
            ck(
                unsafe {
                    ffi::tomoxide_cuda_memset_async(slots[s].f.ptr, 0, len * n * n * hsz, st)
                },
                "memset f f16",
            )?;
            let lrec = unsafe { ffi::tomoxide_linerec_fp16_new(nproj, len, n, nproj, len) };
            if lrec.is_null() {
                return Err(Error::Backend(
                    "cfunc_linerec (f16) allocation failed".into(),
                ));
            }
            slots[s].lrec = lrec;
            unsafe {
                ffi::tomoxide_linerec_fp16_backproject(
                    lrec,
                    slots[s].f.ptr,
                    slots[s].gf.ptr,
                    theta_dev.ptr as *const f32,
                    std::f32::consts::FRAC_PI_2,
                    0,
                    st,
                );
            }
            ck(
                unsafe {
                    ffi::tomoxide_cuda_memcpy_d2h_async(
                        slots[s].pin_out.ptr,
                        slots[s].f.ptr,
                        len * n * n * hsz,
                        st,
                    )
                },
                "d2h vol f16",
            )?;
            slots[s].inflight = Some(k);
        }

        for slot in &mut slots {
            if let Some(ci) = slot.inflight.take() {
                drain_fbp_slot_f16(slot, ci, chunks, n, &mut out)?;
            }
        }
        Ok(out)
    }

    /// f16 counterpart of [`drain_fbp_slot`]: wait for the slot's stream, free its
    /// per-chunk handles, then widen its downloaded f16 volume into `out` (the
    /// f16→f32 round runs in parallel and overlaps later chunks' GPU work).
    fn drain_fbp_slot_f16(
        slot: &mut FbpSlotF16,
        ci: usize,
        chunks: &[(usize, usize)],
        n: usize,
        out: &mut [f32],
    ) -> Result<()> {
        slot.stream.sync()?;
        unsafe {
            ffi::tomoxide_filter_fp16_free(slot.filt);
            ffi::tomoxide_linerec_fp16_free(slot.lrec);
        }
        slot.filt = std::ptr::null_mut();
        slot.lrec = std::ptr::null_mut();
        let (z0, len) = chunks[ci];
        out[z0 * n * n..(z0 + len) * n * n]
            .par_iter_mut()
            .zip(slot.pin_out.as_slice()[..len * n * n].par_iter())
            .for_each(|(d, &h)| *d = h.to_f32());
        Ok(())
    }

    /// f16 driver for the fused Fbp/Linerec path, mirroring [`analytic_fbp_stream`]:
    /// when the whole stack fits one tile, a single [`analytic_fbp_chunk_f16`] —
    /// the fastest path, since the GPU back-projection (memory-bandwidth bound, so
    /// already ~2× faster in half) dominates and forcing tiles would only add
    /// per-tile cuFFT-plan and pinned-allocation overhead. Only when the stack is
    /// genuinely out-of-core does it tile and run the async pipeline, which there
    /// lets f16 process a z-stack 2× larger than f32 would before tiling. The
    /// current device must be selected.
    #[allow(clippy::too_many_arguments)]
    fn analytic_fbp_stream_f16(
        raw: &[f32],
        w: &[f32],
        theta: &[f32],
        nz: usize,
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        pad_side: usize,
    ) -> Result<Vec<f32>> {
        // f16 elements are half the bytes of the f32 sizing `fbp_tile_z` assumes,
        // so double the budget handed to it; its 32-bit index ceiling is
        // element-count based and dtype-independent.
        let budget = device_free_bytes().saturating_mul(2);
        let tile_full = fbp_tile_z(nproj, ncols, n, pad, budget, true).min(nz.max(1));
        if tile_full >= nz {
            return analytic_fbp_chunk_f16(raw, w, theta, nz, nproj, ncols, n, pad, pad_side);
        }
        let tile = fbp_tile_z(nproj, ncols, n, pad, budget / 2, true).min(nz.max(1));
        let k = linerec_chunk_count(nz, nz.div_ceil(tile));
        let chunks = even_z_chunks(nz, k);
        analytic_fbp_pipeline_f16(raw, w, theta, &chunks, nz, nproj, ncols, n, pad, pad_side)
    }

    /// Pad → cuFFT filter → crop → pack pairs → `cfunc_fourierrec` → unpack for
    /// the whole stack on the **current** device. Returns the volume `[nz, n, n]`.
    ///
    /// Single-device only, by design. tomocupy's `gather`/`wrap` kernels scatter
    /// onto the oversampled grid with `atomicAdd` (overlapping Gaussian stencils),
    /// so float accumulation order — hence the low bits of the output — is
    /// **run-to-run nondeterministic** (~1e-7 relative, single-precision floor;
    /// the result is correlation-verified against the CPU `fourierrec`, not
    /// bit-exact). Because the output is already nondeterministic and the complex
    /// slice-pairing `(s, s+nz/2)` spans the whole z-axis, a multi-GPU split would
    /// add complexity without a verifiable benefit, so it is deliberately not
    /// done. Determinizing the gather would need a pull-based rewrite (no inverse
    /// index ⇒ O(n³·nproj) per grid point) that would also break parity with the
    /// vendored kernel; out of scope for this port.
    #[allow(clippy::too_many_arguments)]
    fn analytic_fourierrec(
        raw: &[f32],
        w: &[f32],
        theta: &[f32],
        nz: usize,
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        pad_side: usize,
    ) -> Result<Vec<f32>> {
        let fsz = std::mem::size_of::<f32>();
        let null = std::ptr::null_mut::<c_void>();
        let sino_dev = DevBuf::from_host_f32(raw)?;
        let w_dev = DevBuf::from_host_f32(w)?;
        let theta_dev = DevBuf::from_host_f32(theta)?;
        let gpad = DevBuf::zeroed(nz * nproj * pad * fsz)?;
        ck(
            unsafe {
                ffi::tomoxide_pad(
                    sino_dev.ptr,
                    gpad.ptr,
                    nz,
                    nproj,
                    ncols,
                    pad,
                    pad_side,
                    null,
                )
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
            unsafe { ffi::tomoxide_crop(gpad.ptr, gf.ptr, nz, nproj, ncols, pad, pad_side, null) },
            "crop",
        )?;
        let gc = DevBuf::zeroed(nz * nproj * ncols * fsz)?; // complex [nz/2,nproj,ncols]
        ck(
            unsafe { ffi::tomoxide_pack_pairs(gf.ptr, gc.ptr, nz, nproj, ncols, null) },
            "pack",
        )?;
        let fc = DevBuf::zeroed(nz * n * n * fsz)?; // complex [nz/2,n,n]
        let h =
            unsafe { ffi::tomoxide_fourierrec_new(nproj, nz / 2, n, theta_dev.ptr as *const f32) };
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
        Ok(host)
    }

    /// Half-precision (`Dtype::F16`) Fourierrec on the **current** device, whole
    /// stack in one chunk. Mirrors [`analytic_fourierrec`] with `f16` buffers and a
    /// half-precision cuFFT filter (`pad` must be a power of two; enforced by the
    /// caller). theta stays f32. Like the vendored f32 path the gather scatters with
    /// `atomicAdd`, so the low bits are nondeterministic; in f16 the precision floor
    /// is coarser — correlation-verified against f32, not bit-exact.
    #[allow(clippy::too_many_arguments)]
    fn analytic_fourierrec_f16(
        raw: &[f32],
        w: &[f32],
        theta: &[f32],
        nz: usize,
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        pad_side: usize,
    ) -> Result<Vec<f32>> {
        let hsz = std::mem::size_of::<half::f16>();
        let null = std::ptr::null_mut::<c_void>();
        let sino_dev = DevBuf::from_host_f16(raw)?;
        let w_dev = DevBuf::from_host_f16(w)?;
        let theta_dev = DevBuf::from_host_f32(theta)?;
        let gpad = DevBuf::zeroed(nz * nproj * pad * hsz)?;
        ck(
            unsafe {
                ffi::tomoxide_pad_fp16(
                    sino_dev.ptr,
                    gpad.ptr,
                    nz,
                    nproj,
                    ncols,
                    pad,
                    pad_side,
                    null,
                )
            },
            "pad_fp16",
        )?;
        let fh = unsafe { ffi::tomoxide_filter_fp16_new(nproj, nz, pad) };
        if fh.is_null() {
            return Err(Error::Backend(
                "cfunc_filter (f16) allocation failed".into(),
            ));
        }
        unsafe { ffi::tomoxide_filter_fp16_apply(fh, gpad.ptr, w_dev.ptr, null) };
        unsafe { ffi::tomoxide_filter_fp16_free(fh) };
        let gf = DevBuf::zeroed(nz * nproj * ncols * hsz)?;
        ck(
            unsafe {
                ffi::tomoxide_crop_fp16(gpad.ptr, gf.ptr, nz, nproj, ncols, pad, pad_side, null)
            },
            "crop_fp16",
        )?;
        let gc = DevBuf::zeroed(nz * nproj * ncols * hsz)?; // complex [nz/2,nproj,ncols]
        ck(
            unsafe { ffi::tomoxide_pack_pairs_fp16(gf.ptr, gc.ptr, nz, nproj, ncols, null) },
            "pack_fp16",
        )?;
        let fc = DevBuf::zeroed(nz * n * n * hsz)?; // complex [nz/2,n,n]
        let h = unsafe {
            ffi::tomoxide_fourierrec_fp16_new(nproj, nz / 2, n, theta_dev.ptr as *const f32)
        };
        if h.is_null() {
            return Err(Error::Backend(
                "cfunc_fourierrec (f16) allocation failed".into(),
            ));
        }
        unsafe { ffi::tomoxide_fourierrec_fp16_backproject(h, fc.ptr, gc.ptr, null) };
        unsafe { ffi::tomoxide_fourierrec_fp16_free(h) };
        let vol_dev = DevBuf::zeroed(nz * n * n * hsz)?;
        ck(
            unsafe { ffi::tomoxide_unpack_pairs_fp16(fc.ptr, vol_dev.ptr, nz, n, null) },
            "unpack_fp16",
        )?;
        ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync")?;
        vol_dev.to_host_f16_as_f32(nz * n * n)
    }

    /// Memory-safe driver for the fused Fourierrec path: split the z-stack into
    /// **even** tiles sized by [`fourierrec_tile_z`] and run [`analytic_fourierrec`]
    /// on each. Pairing `(s, s+nz/2)` is just a real-FFT packing trick, so
    /// re-pairing within each contiguous tile reconstructs the same per-slice
    /// volume; tiles concatenate in z-order. `nz` is even (checked by the caller).
    /// The current device must already be selected. Returns volume `[nz, n, n]`.
    #[allow(clippy::too_many_arguments)]
    fn analytic_fourierrec_stream(
        raw: &[f32],
        w: &[f32],
        theta: &[f32],
        nz: usize,
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        pad_side: usize,
    ) -> Result<Vec<f32>> {
        let nfreq2 = (pad / 2 + 1) * 2;
        let tile = fourierrec_tile_z(nproj, n, pad, device_free_bytes()).min(nz.max(2));
        if tile >= nz {
            return analytic_fourierrec(raw, w, theta, nz, nproj, ncols, n, pad, pad_side);
        }
        let mut out = Vec::with_capacity(nz * n * n);
        let mut z0 = 0;
        while z0 < nz {
            let t = tile.min(nz - z0); // even: even tile, even remainder
            let v = analytic_fourierrec(
                &raw[z0 * nproj * ncols..(z0 + t) * nproj * ncols],
                &w[z0 * nfreq2..(z0 + t) * nfreq2],
                theta,
                t,
                nproj,
                ncols,
                n,
                pad,
                pad_side,
            )?;
            out.extend(v);
            z0 += t;
        }
        Ok(out)
    }

    impl crate::backend::AnalyticReconstruct for CudaBackend {
        /// Fused, device-resident analytic reconstruction: upload the raw
        /// sinogram once, then pad → cuFFT filter → crop → back-projection
        /// (`Fbp`/`Linerec` via `cfunc_linerec`) or pack → `cfunc_fourierrec` →
        /// unpack (`Fourierrec`) — all on the device — and download the volume
        /// once. No per-stage host round-trips. Square grid (`n == ncols`).
        fn reconstruct(
            &self,
            sino: &Tomo<f32>,
            geom: &Geometry,
            algorithm: crate::params::Algorithm,
            params: &crate::params::ReconParams,
        ) -> Result<Volume<f32>> {
            use crate::backend::make_fbp_filter;
            use crate::params::Algorithm;

            if geom.beam != Beam::Parallel {
                return Err(Error::InvalidParam(
                    "cuda analytic reconstruct supports parallel beam only".into(),
                ));
            }
            let s = sino.as_layout(Layout::Sinogram); // [nz, nproj, ncols], no copy if already
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
            let nfreq2 = (pad / 2 + 1) * 2; // floats per z row of `w`

            // Half-precision path (tomocupy `--dtype float16`): single GPU, whole
            // stack in one chunk. The half cuFFT filter needs a power-of-two
            // transform width, so `pad` must be a power of two. f16 tiling, the
            // async pipeline, and the multi-GPU split are not implemented here yet
            // (f16 already doubles the per-GPU capacity); a stack too large for one
            // device surfaces as a `cudaMalloc` failure.
            if params.dtype == crate::dtype::Dtype::F16 {
                if !pad.is_power_of_two() {
                    return Err(Error::InvalidParam(format!(
                        "cuda f16 analytic path needs a power-of-two padded width (half cuFFT); \
                         filter pad={pad}. Use a power-of-two detector width, or the default f32 dtype."
                    )));
                }
                let devices = selected_devices();
                unsafe { ffi::tomoxide_cuda_set_device(*devices.first().unwrap_or(&0)) };
                let vol = match algorithm {
                    Algorithm::Fbp | Algorithm::Linerec => {
                        analytic_fbp_stream_f16(raw, &w, theta, nz, nproj, ncols, n, pad, pad_side)?
                    }
                    Algorithm::Fourierrec => {
                        if nz % 2 != 0 {
                            return Err(Error::InvalidParam(format!(
                                "cuda fourierrec needs an even slice count; got nz={nz}"
                            )));
                        }
                        analytic_fourierrec_f16(raw, &w, theta, nz, nproj, ncols, n, pad, pad_side)?
                    }
                    other => {
                        return Err(Error::InvalidParam(format!(
                            "cuda f16 analytic reconstruct: unsupported algorithm {other:?}"
                        )))
                    }
                };
                return Ok(Volume::new(
                    Array3::from_shape_vec((nz, n, n), vol)
                        .map_err(|e| Error::InvalidParam(format!("cuda volume shape: {e}")))?,
                ));
            }

            let vol = match algorithm {
                Algorithm::Fbp | Algorithm::Linerec => {
                    let devices = selected_devices();
                    // cfunc_linerec interpolates across z, so each device's chunk
                    // must hold ≥2 slices or it back-projects to zeros. Cap the
                    // split at nz/2 GPUs: a 4-GPU nz=4 job would otherwise hand
                    // every GPU a single slice and reconstruct an all-zero volume.
                    let k = linerec_chunk_count(nz, devices.len());
                    if k <= 1 {
                        unsafe { ffi::tomoxide_cuda_set_device(*devices.first().unwrap_or(&0)) };
                        analytic_fbp_stream(raw, &w, theta, nz, nproj, ncols, n, pad, pad_side)?
                    } else {
                        // Contiguous z-chunks (sizes differ by ≤1), one GPU each,
                        // run concurrently; each thread owns its device's buffers.
                        //
                        // The back-projection is deterministic and per-slice
                        // independent, but the cuFFT filter batch is nz_chunk·nproj
                        // — cuFFT picks its algorithm by batch size, so chunked vs
                        // whole-stack filtering round differently. Multi-GPU output
                        // therefore differs from single-GPU at the single-precision
                        // FFT floor (~1e-7 relative); each device is bit-identical
                        // to the others and every config is internally
                        // deterministic. Bit-exactness across device counts would
                        // require filtering the full stack on every device (4×
                        // redundant filter+upload) and is intentionally not paid.
                        //
                        // Capture as shared slices so each `move` worker copies a
                        // reference rather than moving the owned buffers.
                        let w: &[f32] = &w;
                        let parts: Vec<Result<Vec<f32>>> = std::thread::scope(|scope| {
                            even_z_chunks(nz, k)
                                .into_iter()
                                .zip(devices.iter().copied())
                                .map(|((a, len), dev)| {
                                    scope.spawn(move || -> Result<Vec<f32>> {
                                        unsafe { ffi::tomoxide_cuda_set_device(dev) };
                                        analytic_fbp_stream(
                                            &raw[a * nproj * ncols..(a + len) * nproj * ncols],
                                            &w[a * nfreq2..(a + len) * nfreq2],
                                            theta,
                                            len,
                                            nproj,
                                            ncols,
                                            n,
                                            pad,
                                            pad_side,
                                        )
                                    })
                                })
                                .collect::<Vec<_>>()
                                .into_iter()
                                .map(|h| {
                                    h.join().unwrap_or_else(|_| {
                                        Err(Error::Backend("cuda analytic worker panicked".into()))
                                    })
                                })
                                .collect()
                        });
                        let mut vol = Vec::with_capacity(nz * n * n);
                        for p in parts {
                            vol.extend(p?);
                        }
                        vol
                    }
                }
                Algorithm::Fourierrec => {
                    if nz % 2 != 0 {
                        return Err(Error::InvalidParam(format!(
                            "cuda fourierrec needs an even slice count; got nz={nz}"
                        )));
                    }
                    let devices = selected_devices();
                    unsafe { ffi::tomoxide_cuda_set_device(*devices.first().unwrap_or(&0)) };
                    analytic_fourierrec_stream(raw, &w, theta, nz, nproj, ncols, n, pad, pad_side)?
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

        /// Reuse one set of `cfunc_filter`/`cfunc_linerec` handles across all
        /// streaming chunks (see [`CudaFbpStream`]). Only the fused FBP/Linerec
        /// back-projection path is handle-reusing here; `Fourierrec` (its packing +
        /// `cfunc_fourierrec`), gridrec and lprec return `None` and the caller falls
        /// back to per-chunk [`reconstruct`]. Binds the first selected device, as the
        /// f16 one-shot path does, since the handles are device-resident.
        fn streaming(
            &self,
            algorithm: crate::params::Algorithm,
            params: &crate::params::ReconParams,
            geom: &Geometry,
            ncols: usize,
            max_nz: usize,
        ) -> Result<Option<Box<dyn StreamingAnalytic>>> {
            use crate::params::Algorithm;
            if geom.beam != Beam::Parallel
                || !matches!(algorithm, Algorithm::Fbp | Algorithm::Linerec)
            {
                return Ok(None);
            }
            let n = params.num_gridx.unwrap_or(ncols);
            if n != ncols {
                return Ok(None); // square-grid only, like `reconstruct`
            }
            let f16 = params.dtype == crate::dtype::Dtype::F16;
            // `make_fbp_filter` pads to `(4·ncols).next_power_of_two()`, always a
            // power of two, so the f16 half-cuFFT width constraint holds by
            // construction (mirrors the assert in `reconstruct`).
            let filter = make_fbp_filter(params.filter_name, ncols)?;
            let devices = selected_devices();
            unsafe { ffi::tomoxide_cuda_set_device(*devices.first().unwrap_or(&0)) };
            let recon = CudaFbpStream::new(filter, &geom.angles.0, ncols, n, max_nz, f16)?;
            Ok(Some(Box::new(recon)))
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
            let s = sino.as_layout(Layout::Sinogram); // [nz, nproj, ncols], no copy if already
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

            // Complex weight `w` [nz, pad/2+1] — small (≈ nz·pad·8 B) and
            // per-z (folds `geom.center.at(z)`). Built whole once, then sliced by
            // z-tile so each tile keeps its own slices' centre-shift phases.
            let nfreq = pad / 2 + 1;
            let w = build_filter_w(filter, geom, nz, ncols, pad);

            // The padded device buffer `tile·nproj·pad` and the filter's internal
            // R2C buffer index in 32-bit `int`, so the z-stack is processed in
            // tiles kept strictly under 2³¹ elements and inside free memory (same
            // overflow class the fused path tiles via `analytic_fbp_stream`).
            // Unfiltered `nz·nproj·pad` faulted lprec at nd=2048, nz=256 (= 2³¹).
            let tile = filter_tile_z(nproj, pad, device_free_bytes()).min(nz.max(1));
            let mut out = vec![0.0f32; nz * nproj * ncols];
            let mut z0 = 0;
            while z0 < nz {
                let tz = tile.min(nz - z0);

                // Padded real sinogram for this z-tile [tz, nproj, pad],
                // edge-replicated borders.
                let mut g = vec![0.0f32; tz * nproj * pad];
                for z in 0..tz {
                    for p in 0..nproj {
                        let row = ((z0 + z) * nproj + p) * ncols;
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
                let w_dev = DevBuf::from_host_f32(&w[z0 * nfreq * 2..(z0 + tz) * nfreq * 2])?;
                let handle = unsafe { ffi::tomoxide_filter_new(nproj, tz, pad) };
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

                // Crop the centred [pad_side, pad_side+ncols) window back to ncols
                // into this tile's z-slices of the output.
                for z in 0..tz {
                    for p in 0..nproj {
                        let dst = ((z0 + z) * nproj + p) * ncols;
                        let srcp = (z * nproj + p) * pad + pad_side;
                        out[dst..dst + ncols].copy_from_slice(&g[srcp..srcp + ncols]);
                    }
                }
                z0 += tz;
            }

            let arr = Array3::from_shape_vec((nz, nproj, ncols), out)
                .map_err(|e| Error::InvalidParam(format!("cuda filter shape: {e}")))?;
            *sino = Tomo::new(arr, Layout::Sinogram).to_layout(orig);
            Ok(())
        }
    }

    /// Devices to spread the per-slice loop over. Default is **all** visible
    /// devices (multi-GPU); `TOMOXIDE_CUDA_DEVICES` overrides with a
    /// comma-separated index list — e.g. `0` pins a single GPU, `0,2` uses two.
    /// Out-of-range / unparsable entries are dropped; an empty result falls back
    /// to device 0.
    fn selected_devices() -> Vec<i32> {
        let count = unsafe { ffi::tomoxide_cuda_device_count() }.max(0);
        if let Ok(s) = std::env::var("TOMOXIDE_CUDA_DEVICES") {
            if !s.trim().is_empty() {
                let v: Vec<i32> = s
                    .split(',')
                    .filter_map(|t| t.trim().parse::<i32>().ok())
                    .filter(|&d| d >= 0 && d < count)
                    .collect();
                return if v.is_empty() { vec![0] } else { v };
            }
        }
        if count <= 0 {
            vec![0]
        } else {
            (0..count).collect()
        }
    }

    /// Hard ceiling on a single CUDA buffer's element count. The vendored
    /// tomocupy kernels compute linear indices in 32-bit `int`, so any buffer
    /// whose element count reaches 2³¹ overflows to a negative index and writes
    /// out of bounds (SIGSEGV). Streaming keeps every tile's largest buffer
    /// strictly under this — independent of how much memory is free.
    const I32_INDEX_LIMIT: usize = i32::MAX as usize; // 2³¹ − 1

    /// Free memory (bytes) on the **current** device. Caller must have already
    /// `cudaSetDevice`'d the device it means to allocate on. Falls back to a
    /// conservative 2 GiB if the query fails, so streaming still makes progress.
    fn device_free_bytes() -> usize {
        // Test/debug hook: cap the reported free memory to force the streaming
        // tiler + async pipeline onto stacks that would otherwise fit one chunk
        // (the only way to exercise the out-of-core path on a large GPU).
        if let Some(v) = std::env::var_os("TOMOXIDE_CUDA_MAX_FREE_BYTES") {
            if let Some(n) = v.to_str().and_then(|s| s.trim().parse::<usize>().ok()) {
                return n;
            }
        }
        let mut free: usize = 0;
        let mut total: usize = 0;
        let rc = unsafe { ffi::tomoxide_cuda_mem_info(&mut free, &mut total) };
        if rc == 0 && free > 0 {
            free
        } else {
            2 * 1024 * 1024 * 1024
        }
    }

    /// Largest z-tile for the fused Fbp/Linerec path on the current device.
    /// Bounded by BOTH (a) the 32-bit index ceiling on the padded buffer
    /// `tile·nproj·pad`, and (b) ~80% of free device memory against the
    /// per-z footprint (sino + padded + cropped + volume + the filter's own
    /// internal R2C buffers, ≈ padded again). When `tex_array` is set (the f16
    /// path), one more cropped-sinogram-sized buffer is added for
    /// cfunc_linerec's hardware-interpolation texture array. Always ≥ 1.
    fn fbp_tile_z(
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        free_bytes: usize,
        tex_array: bool,
    ) -> usize {
        let fsz = std::mem::size_of::<f32>();
        // Per-z device bytes (conservative: cfunc_filter allocates ~one more
        // padded buffer internally, so count `nproj·pad` twice). The trailing
        // `nproj·ncols` (f16 only) is cfunc_linerec's layered texture array,
        // a second copy of the cropped filtered sinogram, so tiling does not
        // OOM on it.
        let tex = if tex_array { nproj * ncols } else { 0 };
        let per_z = (nproj * ncols + 2 * nproj * pad + nproj * ncols + n * n + tex) * fsz;
        let by_mem = (free_bytes / 100 * 80) / per_z.max(1);
        // 88% of 2³¹ over the dominant per-z stride leaves >200M headroom for the
        // in-plane offset terms the kernels add on top of `z·nproj·pad`.
        let by_idx = (I32_INDEX_LIMIT / 100 * 88) / (nproj * pad).max(1);
        by_mem.min(by_idx).max(1)
    }

    /// Single owner of cfunc_linerec's "chunk needs ≥2 z-slices" invariant: the
    /// kernel interpolates the back-projection vertically across slices, so a
    /// chunk holding a single slice reconstructs to **all zeros**. Given a desired
    /// number of contiguous z-chunks `want` (device count for the multi-GPU split,
    /// or the memory/index tile count for single-GPU streaming), returns a count
    /// that keeps an even split at ≥2 slices per chunk — capped at `nz/2`, floored
    /// at 1. `nz < 4` collapses to a single whole-stack chunk (two ≥2 chunks don't
    /// fit); `nz < 2` is the kernel's own single-slice degenerate case, which no
    /// chunking can rescue.
    fn linerec_chunk_count(nz: usize, want: usize) -> usize {
        want.min(nz / 2).max(1)
    }

    /// Split `nz` into `k` contiguous chunks whose lengths differ by at most one
    /// (the first `nz % k` chunks get one extra slice), as `(start, len)` pairs
    /// summing to `nz`. With `k` from [`linerec_chunk_count`] every chunk holds
    /// ≥2 slices, so no chunk back-projects to zeros.
    fn even_z_chunks(nz: usize, k: usize) -> Vec<(usize, usize)> {
        let base = nz / k;
        let rem = nz % k;
        let mut chunks = Vec::with_capacity(k);
        let mut z0 = 0;
        for i in 0..k {
            let len = base + if i < rem { 1 } else { 0 };
            chunks.push((z0, len));
            z0 += len;
        }
        chunks
    }

    /// Largest z-tile for the **composed** FBP filter (`FbpFilter::apply`, used by
    /// the gridrec/lprec path, which filters before its own back-projection).
    /// Same 32-bit ceiling as the fused path: the padded device buffer
    /// `tile·nproj·pad` and the filter's internal R2C complex buffer both index
    /// in `int`, so a single tile of `nz·nproj·pad ≥ 2³¹` overflows (lprec at
    /// nd=2048, nz=256 hits exactly 2³¹ and SIGSEGVs). Bound by 88% of 2³¹ over
    /// `nproj·pad` and ~80% of free memory over the per-z device footprint
    /// (padded buffer + the internal R2C buffer, ≈ padded again).
    fn filter_tile_z(nproj: usize, pad: usize, free_bytes: usize) -> usize {
        let fsz = std::mem::size_of::<f32>();
        let per_z = 2 * nproj * pad * fsz;
        let by_mem = (free_bytes / 100 * 80) / per_z.max(1);
        let by_idx = (I32_INDEX_LIMIT / 100 * 88) / (nproj * pad).max(1);
        by_mem.min(by_idx).max(1)
    }

    /// Largest z-tile (in real slices) for the fused Fourierrec path. The
    /// oversampled grid is `(2n+2m)²` complex per *pair* (= 2 real slices), and
    /// `cfunc_fourierrec` indexes it with `int`, so the pair count
    /// `tile/2 · (2n+2m)²` must stay under the 32-bit ceiling; memory is bounded
    /// by the same grid (`fde`) which dominates. Returned value is **even**
    /// (pairs) and ≥ 2.
    fn fourierrec_tile_z(nproj: usize, n: usize, pad: usize, free_bytes: usize) -> usize {
        let fsz = std::mem::size_of::<f32>();
        // m = ceil(2n·(1/π)·sqrt(-mu·ln eps + (mu·n)²/4)) with eps=1e-3; for the
        // sizes we run this is 4, so 2n+2m = 2n+8.
        let stride = 2 * n + 8;
        let grid_pair = stride * stride * 2 * fsz; // real2 grid per pair (fde)
                                                   // Per-pair device bytes: fde grid + padded/cropped/packed/output buffers
                                                   // (≈ 2·nproj·pad real for the padded stages, plus n·n complex output).
        let per_pair = grid_pair + (2 * nproj * pad + 4 * n * n) * fsz;
        let by_mem = (free_bytes / 100 * 80) / per_pair.max(1); // pairs
                                                                // 88% of 2³¹, in *pairs*, bounded by BOTH index-bearing stages: the
                                                                // padded buffer (`z·nproj·pad`, z = 2·pairs real slices) and the
                                                                // oversampled grid (`pair·stride²`). The grid's gather adds an in-plane
                                                                // offset up to ~6n·stride on top, which the 12% headroom absorbs.
        let margin = I32_INDEX_LIMIT / 100 * 88;
        let by_pad = margin / (2 * nproj * pad).max(1);
        let by_grid = margin / (stride * stride).max(1);
        let pairs = by_mem.min(by_pad).min(by_grid).max(1);
        (pairs * 2).max(2)
    }

    /// One device-pinned rayon pool per selected GPU, built once. Each pool's
    /// worker threads call `cudaSetDevice` at startup, so their `cudaMalloc`,
    /// cuFFT plans (thread-local cache), and per-thread default stream all land
    /// on that GPU. Host cores are split evenly across the pools.
    struct DevicePools {
        devices: Vec<i32>,
        pools: Vec<ThreadPool>,
    }

    fn device_pools() -> &'static DevicePools {
        static POOLS: OnceLock<DevicePools> = OnceLock::new();
        POOLS.get_or_init(|| {
            let devices = selected_devices();
            let total = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let per = (total / devices.len().max(1)).max(1);
            let pools = devices
                .iter()
                .map(|&dev| {
                    ThreadPoolBuilder::new()
                        .num_threads(per)
                        .start_handler(move |_| {
                            unsafe { ffi::tomoxide_cuda_set_device(dev) };
                        })
                        .build()
                        .expect("build cuda device pool")
                })
                .collect();
            DevicePools { devices, pools }
        })
    }

    /// Minimal counting semaphore (std-only) used to bound how many per-slice
    /// reconstructions hold device buffers at once. A worker blocks in
    /// [`acquire`](Semaphore::acquire) until a permit is free; the returned guard
    /// returns the permit on drop.
    struct Semaphore {
        permits: Mutex<usize>,
        cv: Condvar,
    }

    struct SemGuard<'a>(&'a Semaphore);

    impl Semaphore {
        fn new(permits: usize) -> Self {
            Self {
                permits: Mutex::new(permits),
                cv: Condvar::new(),
            }
        }
        fn acquire(&self) -> SemGuard<'_> {
            let mut p = self.permits.lock().unwrap();
            while *p == 0 {
                p = self.cv.wait(p).unwrap();
            }
            *p -= 1;
            SemGuard(self)
        }
    }

    impl Drop for SemGuard<'_> {
        fn drop(&mut self) {
            *self.0.permits.lock().unwrap() += 1;
            self.0.cv.notify_one();
        }
    }

    /// How many per-slice reconstructions may run concurrently on one device.
    /// The composed path allocates an oversampled `(2n)²` complex grid (plus
    /// cuFFT workspace) per in-flight slice; fanning all host cores at large `n`
    /// over-subscribes 32 GB and OOMs. Cap concurrency at ~70% of free device
    /// memory over a conservative per-slice footprint, clamped to the pool size.
    /// `TOMOXIDE_CUDA_MAX_INFLIGHT` overrides the computed value.
    fn max_inflight(n: usize, free_bytes: usize, pool_threads: usize) -> usize {
        if let Ok(s) = std::env::var("TOMOXIDE_CUDA_MAX_INFLIGHT") {
            if let Ok(v) = s.trim().parse::<usize>() {
                if v >= 1 {
                    return v.min(pool_threads);
                }
            }
        }
        // ~3× the (2n)² complex grid covers grid + plan workspace + staging.
        let per_slice = 3 * (2 * n) * (2 * n) * std::mem::size_of::<crate::dtype::Complex32>();
        let by_mem = (free_bytes / 100 * 70) / per_slice.max(1);
        by_mem.clamp(1, pool_threads.max(1))
    }

    impl crate::backend::Fft for CudaBackend {
        /// Fan the per-slice loop across the selected GPUs (and host cores).
        /// Slices are partitioned into one contiguous chunk per device; each
        /// chunk runs on that device's pinned pool, all devices concurrently.
        ///
        /// Bit-identical regardless of device count: each slice's cuFFT uses a
        /// fixed per-slice batch (independent of how slices are spread), and the
        /// host gather/deapodize is deterministic f32 — so gridrec/lprec/phase
        /// give max|Δ|=0 single-GPU vs multi-GPU. (This is *not* true of the
        /// fused filter path, whose batch scales with the chunk; see
        /// [`analytic_fbp_chunk`]'s caller.) Note gridrec is host-gather bound:
        /// one GPU already saturates the host cores, so extra GPUs help the
        /// GPU-heavier lprec/phase far more than gridrec.
        fn for_each_slice(
            &self,
            out: &mut Array3<f32>,
            f: &(dyn Fn(usize, ArrayViewMut2<f32>) -> Result<()> + Sync),
        ) -> Result<()> {
            let dp = device_pools();
            let d = dp.devices.len();
            let n = out.shape()[1]; // square recon grid width, before `out` is borrowed
            let slabs: Vec<ArrayViewMut2<f32>> = out.axis_iter_mut(Axis(0)).collect();
            let nz = slabs.len();

            // Single device: one pinned pool, rayon across its host cores, with
            // in-flight slices capped to fit device memory (see `max_inflight`).
            if d <= 1 {
                let threads = dp.pools[0].current_num_threads();
                return dp.pools[0].install(move || {
                    let sem = Semaphore::new(max_inflight(n, device_free_bytes(), threads));
                    slabs
                        .into_par_iter()
                        .enumerate()
                        .try_for_each(|(row, slab)| {
                            let _permit = sem.acquire();
                            f(row, slab)
                        })
                });
            }

            // Multi-GPU: contiguous chunks (sizes differ by ≤1), one per device.
            let base = nz / d;
            let rem = nz % d;
            let mut chunks: Vec<(usize, Vec<ArrayViewMut2<f32>>)> = Vec::with_capacity(d);
            let mut remaining = slabs;
            let mut offset = 0;
            for i in 0..d {
                let len = base + if i < rem { 1 } else { 0 };
                let tail = remaining.split_off(len);
                chunks.push((offset, remaining));
                remaining = tail;
                offset += len;
            }

            std::thread::scope(|scope| -> Result<()> {
                let handles: Vec<_> = chunks
                    .into_iter()
                    .zip(dp.pools.iter())
                    .map(|((off, chunk), pool)| {
                        scope.spawn(move || -> Result<()> {
                            let threads = pool.current_num_threads();
                            pool.install(|| {
                                let sem =
                                    Semaphore::new(max_inflight(n, device_free_bytes(), threads));
                                chunk.into_par_iter().enumerate().try_for_each(|(i, slab)| {
                                    let _permit = sem.acquire();
                                    f(off + i, slab)
                                })
                            })
                        })
                    })
                    .collect();
                for h in handles {
                    h.join()
                        .map_err(|_| Error::Backend("cuda slice worker panicked".into()))??;
                }
                Ok(())
            })
        }

        fn fft_1d(
            &self,
            buf: &mut [crate::dtype::Complex32],
            len: usize,
            batch: usize,
            inverse: bool,
        ) -> Result<()> {
            if len == 0 || batch == 0 {
                return Ok(());
            }
            if buf.len() != len * batch {
                return Err(Error::ShapeMismatch {
                    expected: (len * batch).to_string(),
                    found: buf.len().to_string(),
                });
            }
            let mut flat = complex_to_flat(buf);
            let d = DevBuf::from_host_f32(&flat)?;
            ck(
                unsafe { ffi::tomoxide_fft_1d(d.ptr, len, batch, inverse as i32) },
                "fft_1d",
            )?;
            d.to_host_f32(&mut flat)?;
            flat_to_complex(&flat, buf);
            Ok(())
        }

        fn fft_2d(
            &self,
            buf: &mut [crate::dtype::Complex32],
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
            let mut flat = complex_to_flat(buf);
            let d = DevBuf::from_host_f32(&flat)?;
            ck(
                unsafe { ffi::tomoxide_fft_2d(d.ptr, rows, cols, batch, inverse as i32) },
                "fft_2d",
            )?;
            d.to_host_f32(&mut flat)?;
            flat_to_complex(&flat, buf);
            Ok(())
        }
    }

    /// Interleave `[re, im, …]` from a complex slice for upload.
    fn complex_to_flat(buf: &[crate::dtype::Complex32]) -> Vec<f32> {
        let mut flat = vec![0.0f32; buf.len() * 2];
        for (i, c) in buf.iter().enumerate() {
            flat[2 * i] = c.re;
            flat[2 * i + 1] = c.im;
        }
        flat
    }

    /// Write an interleaved `[re, im, …]` buffer back into a complex slice.
    fn flat_to_complex(flat: &[f32], buf: &mut [crate::dtype::Complex32]) {
        for (i, c) in buf.iter_mut().enumerate() {
            c.re = flat[2 * i];
            c.im = flat[2 * i + 1];
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
                let d_dark = DevBuf::from_host_f32(dark2d.as_slice().expect("contiguous dark2d"))?;
                let d_denom = DevBuf::from_host_f32(denom.as_slice().expect("contiguous denom"))?;
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
                    return Err(Error::Backend(format!(
                        "cuda darkflat sync failed ({sync})"
                    )));
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
                return Err(Error::Backend(format!(
                    "cuda minus_log sync failed ({sync})"
                )));
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
            let s = filtered.as_layout(Layout::Sinogram); // [nz, nproj, ncols], no copy if already
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
                return Err(Error::Backend(format!(
                    "cuda fourierrec sync failed ({rc})"
                )));
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
        let b = CudaBackend;
        assert_eq!(b.name(), "cuda");
        assert_eq!(b.device(), DeviceKind::Cuda);
        assert!(b.supports(Dtype::F16));
    }
}
