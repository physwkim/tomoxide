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

/// Name of the active CUDA device (e.g. `"NVIDIA RTX 5000 Ada Generation"`),
/// or `None` without the `cuda` feature or when the query fails. Used to key the
/// chunk-tuning cache so a chunk tuned on one GPU is not reused on another model.
/// CUDA device indices a CUDA reconstruction will use, in order. Mirrors the
/// internal selection: `TOMOXIDE_CUDA_DEVICES` (comma-separated indices) when set
/// and non-empty, else all visible devices; an empty/invalid override falls back
/// to `[0]`. Returns `[]` without the `cuda` feature. The multi-GPU `recon`
/// orchestrator fans one z-shard process per returned index.
pub fn selected_devices() -> Vec<i32> {
    #[cfg(not(feature = "cuda"))]
    {
        Vec::new()
    }
    #[cfg(feature = "cuda")]
    {
        cuda_impl::selected_devices()
    }
}

pub fn device_name() -> Option<String> {
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
    #[cfg(feature = "cuda")]
    {
        let mut buf = [0u8; 256];
        let rc = unsafe {
            ffi::tomoxide_cuda_device_name(buf.as_mut_ptr() as *mut std::os::raw::c_char, buf.len())
        };
        if rc != 0 {
            return None;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let name = String::from_utf8_lossy(&buf[..end]).trim().to_string();
        if name.is_empty() {
            None
        } else {
            Some(name)
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

    /// Parallel-beam forward projection on the GPU — the discrete adjoint of
    /// `backprojector` (`cfunc_linerec`), which unlocks the iterative suite
    /// (SIRT/MLEM/OSEM/OSPML/PML/GRAD/TIKH/TV) on CUDA via the generic solvers.
    #[cfg(feature = "cuda")]
    fn projector(&self) -> Option<&dyn crate::backend::ForwardProject> {
        Some(self)
    }

    /// Row-action (ART/BART) projection rows. These solvers are sequential
    /// Kaczmarz updates with no GPU kernel in this design, and the rows are
    /// geometry-only, so CUDA reuses the shared host geometry — `recon(Art|Bart,
    /// …, cuda)` runs the same computation, and yields the same result, as the
    /// CPU backend.
    #[cfg(feature = "cuda")]
    fn ray_projector(&self) -> Option<&dyn crate::backend::RayProject> {
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

    /// Device-resident log-polar reconstruction (`cuda/lprec.cu`): the
    /// gather/scatter + spline prefilter run on the GPU instead of the host.
    #[cfg(feature = "cuda")]
    fn lprec_reconstruct(&self) -> Option<&dyn crate::backend::LpRecReconstruct> {
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

    /// Pinned (`cudaHostAlloc`) staging buffer so the streaming reader can read a
    /// chunk's projections straight into page-locked memory and the H2D is a
    /// direct DMA — no driver staging copy competing with the reader for host
    /// bandwidth. Falls back to a `Vec` if pinning fails.
    #[cfg(feature = "cuda")]
    fn alloc_host_buffer(&self, len: usize) -> Box<dyn crate::backend::HostBuffer> {
        match cuda_impl::PinnedHostBuffer::new(len) {
            Ok(b) => Box::new(b),
            Err(_) => Box::new(crate::backend::VecHostBuffer::new(len)),
        }
    }

    /// CUDA has the device-resident raw path
    /// ([`reconstruct_chunk_raw`](crate::backend::StreamingAnalytic::reconstruct_chunk_raw)),
    /// so the pipeline reads chunks straight into the pinned staging buffer above.
    #[cfg(feature = "cuda")]
    fn wants_raw_chunks(&self) -> bool {
        true
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::{ffi, CudaBackend};
    use crate::backend::{
        make_fbp_filter, parallel_ray_rows, Elementwise, FbpFilter, FilteredBackproject,
        ForwardProject, FourierReconstruct, LpRecReconstruct, RampShape, RayProject, RayRow,
        StreamingAnalytic,
    };
    use crate::data::{Frames, Layout, Tomo, Volume};
    use crate::error::{Error, Result};
    use crate::geometry::{Beam, Geometry};
    use crate::params::{FilterName, StripeMethod};
    use ndarray::{Array3, ArrayViewMut2, Axis};
    use rayon::prelude::*;
    use rayon::{ThreadPool, ThreadPoolBuilder};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::os::raw::c_void;
    use std::sync::{Mutex, OnceLock};

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

        /// Upload host f64 `data` into a fresh device buffer (FW wavelet path is
        /// f64; also used for the per-level damping vectors).
        fn from_host_f64(data: &[f64]) -> Result<Self> {
            let bytes = std::mem::size_of_val(data);
            let buf = DevBuf::new(bytes)?;
            let rc = unsafe {
                ffi::tomoxide_cuda_memcpy_h2d(buf.ptr, data.as_ptr() as *const c_void, bytes)
            };
            if rc != 0 {
                return Err(Error::Backend(format!(
                    "cudaMemcpy H2D (f64) failed ({rc})"
                )));
            }
            Ok(buf)
        }

        /// Upload host i32 `data` into a fresh device buffer (lprec target index
        /// sets: `lpids`/`wids`/`cids`).
        fn from_host_i32(data: &[i32]) -> Result<Self> {
            let bytes = std::mem::size_of_val(data);
            let buf = DevBuf::new(bytes)?;
            let rc = unsafe {
                ffi::tomoxide_cuda_memcpy_h2d(buf.ptr, data.as_ptr() as *const c_void, bytes)
            };
            if rc != 0 {
                return Err(Error::Backend(format!(
                    "cudaMemcpy H2D (i32) failed ({rc})"
                )));
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

        /// Download `count` f64 elements into a fresh Vec (FW test helper).
        #[cfg(test)]
        fn to_host_f64(&self, count: usize) -> Result<Vec<f64>> {
            let mut out = vec![0.0f64; count];
            let bytes = std::mem::size_of_val(out.as_slice());
            debug_assert!(bytes <= self.bytes);
            let rc = unsafe {
                ffi::tomoxide_cuda_memcpy_d2h(out.as_mut_ptr() as *mut c_void, self.ptr, bytes)
            };
            if rc != 0 {
                return Err(Error::Backend(format!(
                    "cudaMemcpy D2H (f64) failed ({rc})"
                )));
            }
            Ok(out)
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

    thread_local! {
        /// Per-thread reusable device scratch for the composed FFT path, keyed by
        /// exact byte size. `Fft::for_each_slice` runs the per-slice loop on
        /// device-pinned worker threads, and each thread issues many same-shaped
        /// `fft_1d`/`fft_2d` calls, so a `cudaMalloc`/`cudaFree` per call serialized
        /// on the driver and dominated the per-slice cost. Thread-local ⇒ no
        /// locking and each buffer lives on the device its worker is pinned to;
        /// keyed by size ⇒ a constant-shape loop reuses one allocation for all
        /// slices. Buffers are freed when the worker thread exits (process end).
        static FFT_SCRATCH: RefCell<HashMap<usize, DevBuf>> = RefCell::new(HashMap::new());
    }

    /// Run `f` with a thread-local device scratch buffer of exactly `bytes`,
    /// allocating it once per (thread, size) and reusing it thereafter. Replaces
    /// the per-call `DevBuf::from_host_f32` allocate-free in the FFT wrappers.
    fn with_fft_scratch<R>(bytes: usize, f: impl FnOnce(&DevBuf) -> Result<R>) -> Result<R> {
        use std::collections::hash_map::Entry;
        FFT_SCRATCH.with(|cell| {
            let mut map = cell.borrow_mut();
            let buf = match map.entry(bytes) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => e.insert(DevBuf::new(bytes)?),
            };
            f(buf)
        })
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

    /// A pinned (`cudaHostAlloc`) [`HostBuffer`] of `f32` for the streaming
    /// reader to fill via `read_slice_into`, so a chunk's H2D is a direct DMA. The
    /// page-locked bytes are plain host memory — safe to fill on the reader thread
    /// and upload/free on the compute thread — so it is `Send` (the raw pointer in
    /// `PinnedBuf` only blocks the auto-derive).
    pub(super) struct PinnedHostBuffer(PinnedBuf<f32>);

    impl PinnedHostBuffer {
        /// Allocate `len` pinned `f32`. Errors propagate so the caller can fall
        /// back to a plain `Vec` buffer.
        pub(super) fn new(len: usize) -> Result<Self> {
            Ok(PinnedHostBuffer(PinnedBuf::<f32>::new(len)?))
        }
    }

    // SAFETY: the wrapped pointer addresses page-locked host memory with no
    // thread affinity; `cudaHostAlloc`/`cudaFreeHost` and plain loads/stores are
    // valid from any thread, and the buffer is moved (not shared) across the
    // reader→compute channel.
    unsafe impl Send for PinnedHostBuffer {}

    impl crate::backend::HostBuffer for PinnedHostBuffer {
        fn as_mut_slice(&mut self) -> &mut [f32] {
            self.0.as_mut_slice()
        }
        fn as_slice(&self) -> &[f32] {
            self.0.as_slice()
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
            // Materialize a C-contiguous host buffer for the flat D2H upload:
            // the analytic path hands a contiguous `sino.clone()` (borrowed, no
            // copy), but the iterative solvers hand strided sub-sinograms
            // (`select(Axis(1), …)` over an angle subset, e.g. OSEM) — copy those.
            let sino_std = s.array.as_standard_layout();
            let sino_slice = sino_std
                .as_slice()
                .expect("as_standard_layout is C-contiguous");

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

    impl ForwardProject for CudaBackend {
        /// Parallel-beam forward projection (`forwardprojection_ker`), the exact
        /// discrete transpose of [`FilteredBackproject::backproject`]
        /// (`cfunc_linerec`, phi = π/2): same y-flip, centre `n/2`, and `4/nproj`
        /// scale. The two therefore form an `{A, Aᵀ}` pair, which is what the
        /// generic iterative solvers (SIRT/MLEM/OSEM/OSPML/PML/GRAD/TIKH/TV)
        /// require — the recon carries the same y-flip/scale handedness as the
        /// CUDA analytic path. Like the back-projector, the kernel hard-wires
        /// centre `n/2` (it ignores `geom.center`) and assumes the detector width
        /// equals the grid `n`. Output is `[nz, nproj, n]` in `Sinogram` layout.
        fn project(&self, vol: &Volume<f32>, geom: &Geometry, out: &mut Tomo<f32>) -> Result<()> {
            if geom.beam != Beam::Parallel {
                return Err(Error::InvalidParam(
                    "cuda forward projection supports parallel beam only".into(),
                ));
            }
            let (nz, ny, nx) = vol.dims();
            if ny != nx {
                return Err(Error::InvalidParam(format!(
                    "cuda forward projection needs a square grid; got {ny}x{nx}"
                )));
            }
            let n = nx;
            let nproj = geom.angles.0.len();
            let ncols = geom.detector.width;
            if ncols != n {
                return Err(Error::InvalidParam(format!(
                    "cuda forward projection needs detector width = grid {n}; got {ncols}"
                )));
            }
            let theta = &geom.angles.0;
            if theta.len() != nproj {
                return Err(Error::ShapeMismatch {
                    expected: format!("{nproj} angles"),
                    found: theta.len().to_string(),
                });
            }
            // Materialize a C-contiguous host buffer for the flat D2H upload
            // (borrowed when already contiguous, copied otherwise), so the
            // iterative solvers may hand any array layout — symmetric with the
            // back-projector's sinogram handling.
            let vol_std = vol.array.as_standard_layout();
            let vol_slice = vol_std
                .as_slice()
                .expect("as_standard_layout is C-contiguous");

            // Device buffers: input volume, theta, zeroed output sinogram (the
            // kernel only atomic-adds into it).
            let f = DevBuf::from_host_f32(vol_slice)?;
            let theta_d = DevBuf::from_host_f32(theta)?;
            let g = DevBuf::zeroed(nz * nproj * n * std::mem::size_of::<f32>())?;

            let phi = std::f32::consts::FRAC_PI_2; // parallel beam
            unsafe {
                ffi::tomoxide_forwardproject(
                    g.ptr,
                    f.ptr,
                    theta_d.ptr as *const f32,
                    phi,
                    nz as i32,
                    n as i32,
                    nproj as i32,
                    std::ptr::null_mut(),
                );
            }
            let rc = unsafe { ffi::tomoxide_cuda_sync() };
            if rc != 0 {
                return Err(Error::Backend(format!("cuda kernel sync failed ({rc})")));
            }

            let mut host = vec![0.0f32; nz * nproj * n];
            g.to_host_f32(&mut host)?;
            let array = Array3::from_shape_vec((nz, nproj, n), host)
                .map_err(|e| Error::InvalidParam(format!("cuda sinogram shape: {e}")))?;
            *out = Tomo::new(array, Layout::Sinogram);
            Ok(())
        }
    }

    impl RayProject for CudaBackend {
        /// Row-action (ART/BART) rows. The row-action solvers are sequential
        /// Kaczmarz updates (no GPU kernel in this design) and the rows are
        /// geometry-only, so CUDA reuses the shared host geometry
        /// [`parallel_ray_rows`] — byte-identical to the CPU backend's rows.
        fn ray_rows(&self, geom: &Geometry, n: usize) -> Result<Vec<Vec<RayRow>>> {
            parallel_ray_rows(geom, n)
        }
    }

    /// Complex FBP weight `w[z, k] = ½·ramp[k]·exp(-2πi·k·δ_z/pad)/pad`, for
    /// `k in 0..ne/2+1`, `δ_z = ncols/2 − center(z)` (half spectrum ⇒ `f_k = k ≥
    /// 0`), interleaved re/im — folds the ramp, signed-frequency centre-shift
    /// phase, and the `½/ne` cuFFT-inverse normalization.
    ///
    /// The `½` matches tomocupy's net FBP filter gain: with the back-projection
    /// `4/nproj` byte-identical to tomocupy, tomoxide's CUDA analytic output was
    /// measured at exactly 2.000× tomocupy across all slices (scale-invariant
    /// Pearson parity hid it). The factor lives only here, so this single site
    /// corrects every CUDA analytic method (fbp/linerec/fourierrec/lamino, f32 +
    /// f16, one-shot + streaming). The residual sub-0.1% is the deferred
    /// `make_fbp_filter` `_wint` quadrature shape, not scale.
    fn build_filter_w(
        filter: &[f32],
        geom: &Geometry,
        nz: usize,
        ncols: usize,
        pad: usize,
    ) -> Vec<f32> {
        let nfreq = pad / 2 + 1;
        let half = ncols as f32 / 2.0;
        let inv_pad = 0.5f32 / pad as f32;
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

    /// Laminography back-projection (`cfunc_linerec` with a tilted rotation axis)
    /// for one output-z chunk, entirely on the **current** device. Mirrors
    /// [`analytic_fbp_chunk`]'s pad → cuFFT filter → crop, but the back-projection
    /// (a) feeds the scalar tilt `phi = π/2 + lamino_angle` instead of `π/2`,
    /// (b) reconstructs `rh` output slices (not the `nz` detector rows — every
    /// output voxel samples a detector row that depends on `(x,y,z)` once the axis
    /// is tilted, so a parallel-beam per-slice mapping no longer holds), and
    /// (c) passes the chunk's global z-start `sz` so the kernel's
    /// `z = (tz + sz) − nz/2` lands on the right detector plane. The full
    /// projection stack (`nz` rows) is filtered once and back-projected into every
    /// output slice. Returns the chunk volume `[rh, n, n]` (with the kernel's
    /// `(n−1−ty)` y-flip and `4/nproj` scale already applied, matching tomocupy).
    #[allow(clippy::too_many_arguments)]
    fn analytic_lamino_chunk(
        raw: &[f32],
        w: &[f32],
        theta: &[f32],
        nz: usize,
        nproj: usize,
        ncols: usize,
        n: usize,
        pad: usize,
        pad_side: usize,
        phi: f32,
        rh: usize,
        sz: i32,
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
        // Output is `rh` slices, not `nz`; ncz = rh, ncproj = nproj (whole stack).
        let f = DevBuf::zeroed(rh * n * n * fsz)?;
        let h = unsafe { ffi::tomoxide_linerec_new(nproj, nz, n, nproj, rh) };
        if h.is_null() {
            return Err(Error::Backend("cfunc_linerec allocation failed".into()));
        }
        unsafe {
            ffi::tomoxide_linerec_backproject(
                h,
                f.ptr,
                gf.ptr,
                theta_dev.ptr as *const f32,
                phi,
                sz,
                null,
            );
        }
        unsafe { ffi::tomoxide_linerec_free(h) };
        ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync")?;
        let mut host = vec![0.0f32; rh * n * n];
        f.to_host_f32(&mut host)?;
        Ok(host)
    }

    /// Auto reconstruction height for laminography (tomocupy `reader.py`
    /// `rh0 = ceil(nz / cos(lamino_angle) / 2) * 2`), with `nz` the detector-row
    /// count after any cropping. Even by construction.
    fn lamino_recon_height(nz: usize, lamino_angle_deg: f32) -> usize {
        let c = (lamino_angle_deg * std::f32::consts::PI / 180.0).cos();
        (((nz as f32 / c) / 2.0).ceil() as usize) * 2
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
        // Device-resident raw path (`reconstruct_chunk_raw`): the raw projection
        // chunk lands here as f32 `[nproj, nz, ncols]`, is dark/flat-corrected and
        // minus-logged in place, then transposed into `sino`. Always f32 (the
        // darkflat/minuslog kernels are f32); the f16 cast happens at the
        // transpose step. `dark2d`/`denom` hold the host-averaged `[nz, ncols]`
        // correction frames (matching `CudaBackend::darkflat`).
        proj: DevBuf,
        dark2d: DevBuf,
        denom: DevBuf,
        // f16 only: device-side f32 staging so the host uploads/downloads f32 and
        // the f32↔f16 cast runs on the GPU (`f2h_ker`/`h2f_ker`), instead of the
        // host rayon convert. `sino_f32` receives the H2D'd f32 sinogram (cast →
        // `sino`); `f_f32` receives the GPU-cast f32 volume (← `f`) before D2H. None
        // on the f32 path, which needs no cast. Mirrors tomocupy's GPU-side astype.
        sino_f32: Option<DevBuf>,
        f_f32: Option<DevBuf>,
        // On-device stripe scratch (`tomoxide_stripe_ti`): `max_nz * 7 * ncols`
        // f64, allocated lazily on the first device-stripe chunk. None until a
        // chunk actually requests a GPU-ported stripe method.
        ti_scratch: Option<DevBuf>,
        // Vo all-stripe scratch (`vo_on_device`): all the per-chunk working
        // buffers, sized to `max_nz` and reused across chunks. Allocated lazily
        // on the first VoAll chunk (a stream not using VoAll never pays the
        // ~5.5 GB it holds at large dims). See [`VoScratch`] for the buffer
        // roles and the lifetime-disjoint aliasing that keeps the set small.
        vo_scratch: Option<VoScratch>,
        filt: *mut c_void,
        lrec: *mut c_void,
        // Fourierrec device-resident path (f32 and f16). When `fourier` is set the
        // back-projection tail of `finish_recon` is pack → `cfunc_fourierrec` →
        // unpack instead of `cfunc_linerec`, reusing the same raw-path normalize/
        // transpose/stripe machinery and the shared `filt` handle. `gc` holds the
        // packed complex sinogram `[max_nz/2, nproj, ncols]` and `fc` the complex
        // reconstruction `[max_nz/2, n, n]` (f16- or f32-wide per `f16`); `frec` is
        // the `cfunc_fourierrec` handle (built for `max_nz/2` pairs, f16 or f32
        // variant). All three are unused (None/null) on the FBP/Linerec path, which
        // keeps `lrec`.
        fourier: bool,
        gc: Option<DevBuf>,
        fc: Option<DevBuf>,
        frec: *mut c_void,
        // Log-polar device-resident path (f32 only). When `lprec` is set the
        // back-projection tail of `finish_recon` runs the [`LpRecDev`] runtime
        // (spline prefilter → per-span gather/FFT/scatter) on the filtered `gf`
        // instead of `cfunc_linerec`/`cfunc_fourierrec`, reusing the same raw-path
        // normalize/transpose/stripe machinery and the shared `filt` handle. The
        // grids are built once and held here (reused across chunks); `flc` is the
        // `[max_nz, nrho, ntheta]` complex work buffer. Both unused (None) on the
        // FBP/Linerec/Fourierrec paths. lprec/fourier/linerec are mutually exclusive.
        lprec: Option<LpRecDev>,
        flc: Option<DevBuf>,
        // Recycled host volume buffers handed back by the writer thread (see
        // `give_reuse_buffer` / `download_volume`). The pinned `out_pinned` is
        // reused across chunks, so the D2H'd volume must be copied into an owned
        // `Send` buffer for the writer. Allocating that buffer fresh each chunk
        // pays ~190 ms of page-faults on a 536 MB `[8, n, n]` chunk; reusing a
        // warm buffer from the writer drops the copy to ~34 ms. Symmetric to the
        // reader's pinned input pool. Empty until the writer returns the first
        // spent volume (the first few chunks still allocate fresh).
        reuse_pool: Vec<Vec<f32>>,
        // Pinned `[max_nz, n, n]` f32 host staging for the volume download. A D2H
        // into pageable memory is driver-staged at ~⅓ the PCIe rate and is
        // synchronous; downloading into this page-locked buffer (then moving it
        // into the writer's owned `Volume`) runs the copy at full bandwidth and
        // avoids the per-chunk `vec![0.0; …]` zero-then-overwrite. Reused across
        // chunks. See [`download_volume`].
        out_pinned: PinnedBuf,
    }

    /// Persistent device scratch for the Vo all-stripe pass, sized to the
    /// handle's fixed `max_nz` batch and reused across chunks (a smaller chunk
    /// uses the `nz`-prefix). Allocating once and reusing removes ~24 per-chunk
    /// `cudaMalloc`/`cudaFree` pairs — the large frees synchronise the device, so
    /// this is wall-clock, not just bookkeeping.
    ///
    /// The large `[max_nz,nproj,ncols]` buffers dominate the footprint, so roles
    /// with disjoint lifetimes share one buffer. The sharing is intra-function
    /// only (each function writes solely its own buffers); the one cross-function
    /// reuse is `big_a`, which `vo_rs_large` reads as its read-only input `s` and
    /// never writes, so the caller can reuse it afterwards. Large f32 roles:
    ///   big_a:     rs_dead `smooth` → `work` (input to rs_large) → rs_sort `sortedv`
    ///   big_b:     rs_sort `smoothed`
    ///   rl_out:    rs_large normalised `out` (returned as `dead_out`)
    ///   rl_sort:   rs_large `sinosort` → `sortdummy`
    ///   rl_smooth: rs_large `sinosmooth`
    /// Large i32: `perm` (rs_sort) and `perm2` (rs_large) kept separate. The
    /// `[max_nz,ncols]`/`[max_nz]` buffers are ~0.5 MB each, so each small role
    /// gets its own buffer (no aliasing); `detsort`/`detraw` are shared by the
    /// two `vo_detect_mask` calls, which never overlap.
    struct VoScratch {
        big_a: DevBuf,
        big_b: DevBuf,
        rl_out: DevBuf,
        rl_sort: DevBuf,
        rl_smooth: DevBuf,
        perm: DevBuf,
        perm2: DevBuf,
        listdiff: DevBuf,
        listdiffbck: DevBuf,
        listfact: DevBuf,
        mask_dead: DevBuf,
        lf64: DevBuf,
        lf32: DevBuf,
        mask_large: DevBuf,
        detsort: DevBuf,
        detraw: DevBuf,
        goodx: DevBuf,
        goodcount: DevBuf,
    }

    impl VoScratch {
        fn new(max_nz: usize, nproj: usize, ncols: usize) -> Result<Self> {
            let (f32sz, f64sz, i32sz) = (
                std::mem::size_of::<f32>(),
                std::mem::size_of::<f64>(),
                std::mem::size_of::<i32>(),
            );
            let vol = max_nz * nproj * ncols;
            let cols = max_nz * ncols;
            Ok(VoScratch {
                big_a: DevBuf::new(vol * f32sz)?,
                big_b: DevBuf::new(vol * f32sz)?,
                rl_out: DevBuf::new(vol * f32sz)?,
                rl_sort: DevBuf::new(vol * f32sz)?,
                rl_smooth: DevBuf::new(vol * f32sz)?,
                perm: DevBuf::new(vol * i32sz)?,
                perm2: DevBuf::new(vol * i32sz)?,
                listdiff: DevBuf::new(cols * f32sz)?,
                listdiffbck: DevBuf::new(cols * f32sz)?,
                listfact: DevBuf::new(cols * f32sz)?,
                mask_dead: DevBuf::new(cols * f32sz)?,
                lf64: DevBuf::new(cols * f64sz)?,
                lf32: DevBuf::new(cols * f32sz)?,
                mask_large: DevBuf::new(cols * f32sz)?,
                detsort: DevBuf::new(cols * f32sz)?,
                detraw: DevBuf::new(cols * f32sz)?,
                goodx: DevBuf::new(cols * i32sz)?,
                goodcount: DevBuf::new(max_nz * i32sz)?,
            })
        }
    }

    impl CudaFbpStream {
        /// Allocate the persistent buffers and `cfunc_filter`/`cfunc_linerec`
        /// handles for a `max_nz`-slice chunk. `filter` is the ramp kernel
        /// (`make_fbp_filter`), `theta` the chunk-invariant angles. The current
        /// device must already be selected (the caller binds it).
        #[allow(clippy::too_many_arguments)]
        fn new(
            filter: Vec<f32>,
            theta: &[f32],
            ncols: usize,
            n: usize,
            max_nz: usize,
            f16: bool,
            fourier: bool,
            lprec: Option<LpRecDev>,
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
            // Raw-path staging (always f32): the projection chunk plus the small
            // per-chunk dark/denominator frames.
            let fsz_proj = std::mem::size_of::<f32>();
            let proj = DevBuf::zeroed(max_nz * nproj * ncols * fsz_proj)?;
            let dark2d = DevBuf::zeroed(max_nz * ncols * fsz_proj)?;
            let denom = DevBuf::zeroed(max_nz * ncols * fsz_proj)?;
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
            // The FBP filter handle is shared by both back-projection tails. The
            // tail handle is `cfunc_linerec` (FBP/Linerec) or `cfunc_fourierrec`
            // (Fourierrec, f16 or f32 — built for `max_nz/2` packed pairs). The
            // Fourierrec packed/complex scratch (`gc`/`fc`) is allocated only on
            // that path.
            let filt = unsafe {
                if f16 {
                    ffi::tomoxide_filter_fp16_new(nproj, max_nz, pad)
                } else {
                    ffi::tomoxide_filter_new(nproj, max_nz, pad)
                }
            };
            // lprec uses only the shared `filt` handle plus the uploaded grids; it
            // builds no back-projection tail handle. Its `[max_nz, nrho, ntheta]`
            // complex work buffer is allocated here and reused across chunks.
            let flc = match &lprec {
                Some(lp) => Some(DevBuf::zeroed(lp.flc_bytes(max_nz))?),
                None => None,
            };
            // Pinned host staging for the per-chunk volume D2H (see field docs).
            let out_pinned = PinnedBuf::<f32>::new(max_nz * n * n)?;
            let (lrec, frec, gc, fc) = if lprec.is_some() {
                (std::ptr::null_mut(), std::ptr::null_mut(), None, None)
            } else if fourier {
                // Packed/complex scratch sized to the element width (f16 or f32) and
                // the matching `max_nz/2`-pair handle. The f16 tail mirrors
                // `analytic_fourierrec_f16`.
                let gc = DevBuf::zeroed(max_nz * nproj * ncols * esz)?;
                let fc = DevBuf::zeroed(max_nz * n * n * esz)?;
                let frec = unsafe {
                    if f16 {
                        ffi::tomoxide_fourierrec_fp16_new(
                            nproj,
                            max_nz / 2,
                            n,
                            theta_dev.ptr as *const f32,
                        )
                    } else {
                        ffi::tomoxide_fourierrec_new(
                            nproj,
                            max_nz / 2,
                            n,
                            theta_dev.ptr as *const f32,
                        )
                    }
                };
                (std::ptr::null_mut(), frec, Some(gc), Some(fc))
            } else {
                let lrec = unsafe {
                    if f16 {
                        ffi::tomoxide_linerec_fp16_new(nproj, max_nz, n, nproj, max_nz)
                    } else {
                        ffi::tomoxide_linerec_new(nproj, max_nz, n, nproj, max_nz)
                    }
                };
                (lrec, std::ptr::null_mut(), None, None)
            };
            // lprec has no tail handle, so only `filt` must be non-null there.
            let tail = if fourier { frec } else { lrec };
            if filt.is_null() || (lprec.is_none() && tail.is_null()) {
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
                    if !frec.is_null() {
                        if f16 {
                            ffi::tomoxide_fourierrec_fp16_free(frec)
                        } else {
                            ffi::tomoxide_fourierrec_free(frec)
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
                proj,
                dark2d,
                denom,
                sino_f32,
                f_f32,
                ti_scratch: None,
                vo_scratch: None,
                filt,
                lrec,
                fourier,
                gc,
                fc,
                frec,
                lprec,
                flc,
                out_pinned,
                reuse_pool: Vec::new(),
            })
        }
    }

    impl Drop for CudaFbpStream {
        fn drop(&mut self) {
            unsafe {
                if self.f16 {
                    ffi::tomoxide_filter_fp16_free(self.filt);
                } else {
                    ffi::tomoxide_filter_free(self.filt);
                }
                if self.lprec.is_some() {
                    // lprec builds no tail handle (only the shared `filt`); the
                    // grids/flc are freed by their own DevBuf/LpRecDev drops.
                } else if self.fourier {
                    if self.f16 {
                        ffi::tomoxide_fourierrec_fp16_free(self.frec);
                    } else {
                        ffi::tomoxide_fourierrec_free(self.frec);
                    }
                } else if self.f16 {
                    ffi::tomoxide_linerec_fp16_free(self.lrec);
                } else {
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
            let null = std::ptr::null_mut::<c_void>();
            let partial = nz < self.max_nz;
            // Zero the unused tail of a partial trailing chunk so the always-`max_nz`
            // kernels see zeros there (→ zero output we drop); full chunks overwrite
            // the whole `sino` so they skip the memset. (`w`'s tail is zeroed in
            // `finish_recon`.)
            if partial {
                ck(
                    unsafe { ffi::tomoxide_cuda_memset(self.sino.ptr, 0, self.sino.bytes) },
                    "memset sino",
                )?;
            }
            if self.f16 {
                // Upload f32 and cast f32→f16 on the GPU (no host rayon convert).
                // For a partial chunk only the valid `nz` rows are uploaded+cast;
                // the memset above already zeroed `sino`'s tail.
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
            } else {
                self.sino.copy_from_host_f32(raw)?;
            }
            self.finish_recon(nz, geom)
        }

        fn reconstruct_chunk_raw(
            &mut self,
            data: &[f32],
            dims: (usize, usize, usize),
            flat: Option<&Frames<f32>>,
            dark: Option<&Frames<f32>>,
            geom: &Geometry,
            stripe: StripeMethod,
        ) -> Result<Option<Volume<f32>>> {
            // `data` is the caller's contiguous, C-order projection-layout chunk
            // `[nproj, nz, ncols]` (read straight into a pinned staging buffer);
            // no layout check is needed — the streaming pipeline only ever feeds
            // projection-order chunks here.
            //
            // Defer the whole chunk to the host when the requested stripe method
            // has no on-device port — checked before any GPU work so we don't
            // normalize on the device only to have the host redo it.
            if !gpu_supports_stripe(stripe) {
                return Ok(None);
            }
            let (nproj, nz, ncols) = dims;
            if nproj != self.nproj || ncols != self.ncols {
                return Err(Error::ShapeMismatch {
                    expected: format!("nproj={} ncols={}", self.nproj, self.ncols),
                    found: format!("nproj={nproj} ncols={ncols}"),
                });
            }
            if nz > self.max_nz {
                return Err(Error::InvalidParam(format!(
                    "streaming reconstruct_chunk_raw: nz={nz} exceeds max_nz={}",
                    self.max_nz
                )));
            }
            if data.len() != nproj * nz * ncols {
                return Err(Error::ShapeMismatch {
                    expected: format!("{} elems ([{nproj}, {nz}, {ncols}])", nproj * nz * ncols),
                    found: format!("{} elems", data.len()),
                });
            }
            let null = std::ptr::null_mut::<c_void>();
            // One H2D: the raw projection chunk as the contiguous [nproj, nz, ncols]
            // prefix of the max-sized f32 `proj` buffer. When `data` is pinned
            // (the streaming reader's staging buffer) this is a direct DMA with no
            // driver staging copy.
            self.proj.copy_from_host_f32(data)?;
            // Dark/flat correction on the device, mirroring `CudaBackend::darkflat`
            // (host-averaged `dark2d`, clamped `denom`, device broadcast) so the
            // output is bit-identical to the host normalize path. Skipped when
            // flat/dark are absent (already-normalized input).
            if let (Some(flat), Some(dark)) = (flat, dark) {
                let dark2d = dark
                    .array
                    .mean_axis(Axis(0))
                    .ok_or_else(|| Error::InvalidParam("empty dark stack".into()))?;
                let flat2d = flat
                    .array
                    .mean_axis(Axis(0))
                    .ok_or_else(|| Error::InvalidParam("empty flat stack".into()))?;
                if dark2d.dim() != (nz, ncols) {
                    return Err(Error::ShapeMismatch {
                        expected: format!("flat/dark frame [{nz}, {ncols}]"),
                        found: format!("{:?}", dark2d.dim()),
                    });
                }
                let mut denom = &flat2d - &dark2d;
                denom.mapv_inplace(|v| if v.abs() < 1e-6 { 1.0 } else { v });
                self.dark2d
                    .copy_from_host_f32(dark2d.as_slice().expect("contiguous dark2d"))?;
                self.denom
                    .copy_from_host_f32(denom.as_slice().expect("contiguous denom"))?;
                ck(
                    unsafe {
                        ffi::tomoxide_darkflat(
                            self.proj.ptr,
                            self.dark2d.ptr,
                            self.denom.ptr,
                            nproj,
                            nz,
                            ncols,
                            null,
                        )
                    },
                    "darkflat",
                )?;
            }
            // minus-log over the valid [nproj, nz, ncols] prefix.
            ck(
                unsafe { ffi::tomoxide_minuslog(self.proj.ptr, nproj * nz * ncols, null) },
                "minuslog",
            )?;
            // Transpose projection → sinogram on the device. The transpose writes
            // only the `nz` valid rows of the sinogram, so zero the tail first on a
            // partial chunk (as `reconstruct_chunk` does). Both paths transpose
            // into an f32 buffer (`sino` for f32, the `sino_f32` staging buffer for
            // f16) so on-device stripe removal runs on f32; the f16 cast follows.
            let partial = nz < self.max_nz;
            if partial {
                ck(
                    unsafe { ffi::tomoxide_cuda_memset(self.sino.ptr, 0, self.sino.bytes) },
                    "memset sino",
                )?;
            }
            let f32_target = if self.f16 {
                self.sino_f32.as_ref().expect("f16 path has sino_f32").ptr
            } else {
                self.sino.ptr
            };
            ck(
                unsafe {
                    ffi::tomoxide_transpose(self.proj.ptr, f32_target, nproj, nz, ncols, null)
                },
                "transpose",
            )?;
            // On-device stripe removal on the f32 sinogram (no-op for
            // `StripeMethod::None`). `gpu_supports_stripe` was checked above, so a
            // `false` here is a logic error rather than a silent host fallback.
            let handled = self.stripe_on_device(f32_target, nz, stripe)?;
            debug_assert!(handled, "gpu_supports_stripe accepted an unhandled method");
            if self.f16 {
                ck(
                    unsafe {
                        ffi::tomoxide_cast_f32_to_f16(
                            f32_target,
                            self.sino.ptr,
                            nz * nproj * ncols,
                            null,
                        )
                    },
                    "cast f32->f16 sino",
                )?;
            }
            Ok(Some(self.finish_recon(nz, geom)?))
        }

        fn give_reuse_buffer(&mut self, buf: Vec<f32>) {
            // The number of volume buffers in circulation is bounded by the
            // pipeline's channel depths, so the pool stays small; cap it anyway so
            // a buffer can never accumulate without limit.
            if self.reuse_pool.len() < 4 {
                self.reuse_pool.push(buf);
            }
        }
    }

    /// Whether the device-resident path can run `stripe` on the GPU. `None` is a
    /// no-op (always handled). Methods without a GPU port make the caller defer
    /// the whole chunk to the host. Kept in sync with [`CudaFbpStream::stripe_on_device`].
    fn gpu_supports_stripe(stripe: StripeMethod) -> bool {
        match stripe {
            StripeMethod::None | StripeMethod::Ti { nblock: 0, .. } | StripeMethod::Fw { .. } => {
                true
            }
            // Vo all-stripe: the on-device median filters keep their window in a
            // fixed 256-element thread-local buffer, so larger windows fall back
            // to the host route.
            StripeMethod::VoAll {
                la_size, sm_size, ..
            } => la_size <= 256 && sm_size <= 256,
            _ => false,
        }
    }

    /// Münch damping vector `D = ifftshift(damp)` (matches `stripe::damp_vector`),
    /// computed in f32 like tomopy and returned as f64 for the device multiply.
    fn fw_damp_vector(my: usize, sigma: f32) -> Vec<f64> {
        let two_sig2 = 2.0f32 * sigma * sigma;
        let damp: Vec<f32> = (0..my)
            .map(|k| {
                let y_hat = ((-(my as i64) + 2 * k as i64) as f32 + 1.0) / 2.0;
                1.0f32 - (-(y_hat * y_hat) / two_sig2).exp()
            })
            .collect();
        let half = my / 2;
        (0..my).map(|i| damp[(i + half) % my] as f64).collect()
    }

    impl CudaFbpStream {
        /// Apply on-device stripe removal to the f32 sinogram at `ptr` (the valid
        /// `[nz, nproj, ncols]` prefix), in place. Returns `Ok(true)` when the
        /// method was handled on the device, `Ok(false)` when it has no GPU port
        /// (kept consistent with [`gpu_supports_stripe`]).
        fn stripe_on_device(
            &mut self,
            ptr: *mut c_void,
            nz: usize,
            stripe: StripeMethod,
        ) -> Result<bool> {
            let null = std::ptr::null_mut::<c_void>();
            match stripe {
                StripeMethod::None => Ok(true),
                // Titarenko, whole-sinogram (nblock=0). nblock>0 uses `_ringb`,
                // which is not ported — leave it to the host.
                StripeMethod::Ti { nblock: 0, beta } => {
                    if self.ti_scratch.is_none() {
                        let bytes = self.max_nz * 7 * self.ncols * std::mem::size_of::<f64>();
                        self.ti_scratch = Some(DevBuf::zeroed(bytes)?);
                    }
                    let scratch = self.ti_scratch.as_ref().expect("ti_scratch allocated");
                    ck(
                        unsafe {
                            ffi::tomoxide_stripe_ti(
                                ptr,
                                nz,
                                self.nproj,
                                self.ncols,
                                beta,
                                scratch.ptr,
                                null,
                            )
                        },
                        "stripe_ti",
                    )?;
                    Ok(true)
                }
                // Fourier-Wavelet. Multi-level db5 DWT/IDWT with per-column FFT
                // damping of the vertical band; orchestrated below.
                StripeMethod::Fw { sigma, level } => {
                    self.fw_on_device(ptr, nz, sigma, level)?;
                    Ok(true)
                }
                // Vo all-stripe (rs_dead → rs_sort). Window sizes are gated to
                // <= 256 by `gpu_supports_stripe`.
                StripeMethod::VoAll {
                    snr,
                    la_size,
                    sm_size,
                } => {
                    if self.vo_scratch.is_none() {
                        self.vo_scratch =
                            Some(VoScratch::new(self.max_nz, self.nproj, self.ncols)?);
                    }
                    let sc = self.vo_scratch.as_ref().expect("vo_scratch allocated");
                    self.vo_on_device(ptr, nz, snr, la_size, sm_size, sc)?;
                    Ok(true)
                }
                _ => Ok(false),
            }
        }

        /// Multi-level Fourier-Wavelet stripe removal on the device f32 sinogram
        /// `ptr [nz, nproj, ncol]`, mirroring `stripe::fw_slice` /
        /// `remove_stripe_fw` for all `nz` slices at once: db5 DWT/IDWT in f64
        /// (each band rounded to f32 like tomopy's `pywt` float32 pass) with the
        /// vertical-detail band damped column-wise via the f32 cuFFT shim. Held
        /// to correlation parity with the CPU golden, not bit-exactness.
        ///
        /// Per-call device buffers are allocated fresh (the band shapes depend on
        /// the chunk's `nz` through `level`); reuse/caching is a later
        /// optimization, not a correctness concern.
        fn fw_on_device(
            &self,
            ptr: *mut c_void,
            nz: usize,
            sigma: f32,
            level: Option<usize>,
        ) -> Result<()> {
            let null = std::ptr::null_mut::<c_void>();
            let (nproj, ncol) = (self.nproj, self.ncols);
            if nproj == 0 || nz == 0 || ncol == 0 {
                return Ok(());
            }
            // Auto-level matches the CPU golden: ceil(log2(max(nproj, nz, ncol))),
            // where the chunk's `nz` plays tomopy's `nrows`.
            let level = level.unwrap_or_else(|| {
                let size = nproj.max(nz).max(ncol);
                (size as f64).log2().ceil() as usize
            });
            if level == 0 {
                return Ok(());
            }
            let nx = nproj + nproj / 8; // pad=True
            let xshift = (nx - nproj) / 2;
            let f64sz = std::mem::size_of::<f64>();

            // approx = pad(sino) → f64 [nz, nx, ncol].
            let mut approx = DevBuf::new(nz * nx * ncol * f64sz)?;
            ck(
                unsafe { ffi::tomoxide_fw_pad(ptr, approx.ptr, nz, nproj, ncol, nx, xshift, null) },
                "fw_pad",
            )?;

            // Per-level detail bands (kept for the inverse pass) and their shapes.
            let mut chs: Vec<DevBuf> = Vec::with_capacity(level);
            let mut cvs: Vec<DevBuf> = Vec::with_capacity(level);
            let mut cds: Vec<DevBuf> = Vec::with_capacity(level);
            let mut dims: Vec<(usize, usize)> = Vec::with_capacity(level);

            // Reusable interleaved-complex scratch for the damping FFT, sized for
            // the largest band (level 0); smaller levels use a contiguous prefix.
            let or0 = (nx + 9) / 2;
            let oc0 = (ncol + 9) / 2;
            let cplx = DevBuf::new(nz * or0 * oc0 * 2 * std::mem::size_of::<f32>())?;

            // Forward: `level`-deep db5 decomposition (rows pass then cols pass),
            // each band rounded to f32, vertical band damped.
            let (mut r, mut c) = (nx, ncol);
            for _ in 0..level {
                let or = (r + 9) / 2;
                let oc = (c + 9) / 2;
                // Rows pass (last axis): approx[nz,r,c] → cols_a, cols_d [nz,r,oc].
                let cols_a = DevBuf::new(nz * r * oc * f64sz)?;
                let cols_d = DevBuf::new(nz * r * oc * f64sz)?;
                ck(
                    unsafe {
                        ffi::tomoxide_fw_dwt_rows(
                            approx.ptr, cols_a.ptr, cols_d.ptr, nz, r, c, null,
                        )
                    },
                    "fw_dwt_rows",
                )?;
                // Cols pass (middle axis): cols_a → ca,ch; cols_d → cv,cd [nz,or,oc].
                let ca = DevBuf::new(nz * or * oc * f64sz)?;
                let ch = DevBuf::new(nz * or * oc * f64sz)?;
                let cv = DevBuf::new(nz * or * oc * f64sz)?;
                let cd = DevBuf::new(nz * or * oc * f64sz)?;
                ck(
                    unsafe {
                        ffi::tomoxide_fw_dwt_cols(cols_a.ptr, ca.ptr, ch.ptr, nz, r, oc, null)
                    },
                    "fw_dwt_cols(approx)",
                )?;
                ck(
                    unsafe {
                        ffi::tomoxide_fw_dwt_cols(cols_d.ptr, cv.ptr, cd.ptr, nz, r, oc, null)
                    },
                    "fw_dwt_cols(detail)",
                )?;
                // Round every band to f32 (tomopy band quantization).
                let n_band = nz * or * oc;
                ck(
                    unsafe { ffi::tomoxide_fw_round(ca.ptr, n_band, null) },
                    "fw_round(ca)",
                )?;
                ck(
                    unsafe { ffi::tomoxide_fw_round(ch.ptr, n_band, null) },
                    "fw_round(ch)",
                )?;
                ck(
                    unsafe { ffi::tomoxide_fw_round(cv.ptr, n_band, null) },
                    "fw_round(cv)",
                )?;
                ck(
                    unsafe { ffi::tomoxide_fw_round(cd.ptr, n_band, null) },
                    "fw_round(cd)",
                )?;
                // Damp the vertical-detail band cv along axis 0 (my=or, mx=oc):
                // real(ifft(fft(col) · D)), D = ifftshift(damp).
                let d = DevBuf::from_host_f64(&fw_damp_vector(or, sigma))?;
                ck(
                    unsafe { ffi::tomoxide_fw_damp_gather(cv.ptr, cplx.ptr, nz, or, oc, null) },
                    "fw_damp_gather",
                )?;
                ck(
                    unsafe { ffi::tomoxide_fft_1d(cplx.ptr, or, nz * oc, 0) },
                    "fw_damp_fft_fwd",
                )?;
                ck(
                    unsafe { ffi::tomoxide_fw_damp_apply(cplx.ptr, d.ptr, nz, or, oc, null) },
                    "fw_damp_apply",
                )?;
                ck(
                    unsafe { ffi::tomoxide_fft_1d(cplx.ptr, or, nz * oc, 1) },
                    "fw_damp_fft_inv",
                )?;
                ck(
                    unsafe { ffi::tomoxide_fw_damp_scatter(cplx.ptr, cv.ptr, nz, or, oc, null) },
                    "fw_damp_scatter",
                )?;

                chs.push(ch);
                cvs.push(cv);
                cds.push(cd);
                dims.push((or, oc));
                approx = ca; // running approximation for the next level
                r = or;
                c = oc;
            }

            // Inverse: crop the running approximation to each level's band shape,
            // then idwt2 (cols pass then rows pass) with the damped details.
            let mut sli = approx;
            let (mut sr, mut sc) = (r, c);
            for n in (0..level).rev() {
                let (or, oc) = dims[n];
                let cropped = DevBuf::new(nz * or * oc * f64sz)?;
                ck(
                    unsafe {
                        ffi::tomoxide_fw_crop(sli.ptr, cropped.ptr, nz, sr, sc, or, oc, null)
                    },
                    "fw_crop",
                )?;
                // idwt2 cols pass (middle axis): combine (ca,ch) and (cv,cd).
                let rr = 2 * or + 2 - 10;
                let cols_a = DevBuf::new(nz * rr * oc * f64sz)?;
                let cols_d = DevBuf::new(nz * rr * oc * f64sz)?;
                ck(
                    unsafe {
                        ffi::tomoxide_fw_idwt_cols(
                            cropped.ptr,
                            chs[n].ptr,
                            cols_a.ptr,
                            nz,
                            or,
                            oc,
                            null,
                        )
                    },
                    "fw_idwt_cols(approx)",
                )?;
                ck(
                    unsafe {
                        ffi::tomoxide_fw_idwt_cols(
                            cvs[n].ptr, cds[n].ptr, cols_d.ptr, nz, or, oc, null,
                        )
                    },
                    "fw_idwt_cols(detail)",
                )?;
                // idwt2 rows pass (last axis): combine the two column results.
                let rc_ = 2 * oc + 2 - 10;
                let out = DevBuf::new(nz * rr * rc_ * f64sz)?;
                ck(
                    unsafe {
                        ffi::tomoxide_fw_idwt_rows(
                            cols_a.ptr, cols_d.ptr, out.ptr, nz, rr, oc, null,
                        )
                    },
                    "fw_idwt_rows",
                )?;
                sli = out;
                sr = rr;
                sc = rc_;
            }

            // Crop back to the sinogram region → write f32 into `ptr` in place.
            ck(
                unsafe {
                    ffi::tomoxide_fw_final(sli.ptr, ptr, nz, nproj, ncol, sr, sc, xshift, null)
                },
                "fw_final",
            )?;
            Ok(())
        }

        /// `_detect_stripe` + `binary_dilation` for a per-column `listfact`
        /// `[nz, nc]`, writing the dilated mask. `border_zero` protects the two
        /// outer columns each side (the `_rs_dead` rule); `_rs_large` passes
        /// `false`.
        fn vo_detect_mask(
            &self,
            listfact: &DevBuf,
            mask: &DevBuf,
            nz: usize,
            snr: f32,
            border_zero: bool,
            sc: &VoScratch,
        ) -> Result<()> {
            let null = std::ptr::null_mut::<c_void>();
            let nc = self.ncols;
            let (sorted, rawmask) = (&sc.detsort, &sc.detraw);
            ck(
                unsafe { ffi::tomoxide_vo_slicesort(listfact.ptr, sorted.ptr, nz, nc, 0, null) },
                "vo_slicesort",
            )?;
            ck(
                unsafe {
                    ffi::tomoxide_vo_detect_rawmask(
                        listfact.ptr,
                        sorted.ptr,
                        rawmask.ptr,
                        nz,
                        nc,
                        snr,
                        null,
                    )
                },
                "vo_detect_rawmask",
            )?;
            ck(
                unsafe {
                    ffi::tomoxide_vo_dilate(
                        rawmask.ptr,
                        mask.ptr,
                        nz,
                        nc,
                        if border_zero { 1 } else { 0 },
                        null,
                    )
                },
                "vo_dilate",
            )?;
            Ok(())
        }

        /// `_rs_large` (Vo algorithm 5) on `s [nz,nrow,nc]` with `norm=true`:
        /// replace detected large-stripe columns with the rank-smoothed profile,
        /// normalising by the per-column intensity factor first. Writes the result
        /// into `sc.rl_out` (the caller's `dead_out`). `s` may alias `sc.big_a`;
        /// it is only read (never written) here, so the caller may reuse it after.
        fn vo_rs_large(
            &self,
            s: &DevBuf,
            nz: usize,
            snr: f32,
            size: usize,
            drop_ratio: f32,
            sc: &VoScratch,
        ) -> Result<()> {
            let null = std::ptr::null_mut::<c_void>();
            let (nrow, nc) = (self.nproj, self.ncols);
            let dr = drop_ratio.clamp(0.0, 0.8) as f64;
            let ndrop = (0.5 * dr * nrow as f64) as usize;

            // sinosort = sort each column ascending (perm unused here).
            let sinosort = &sc.rl_sort;
            ck(
                unsafe {
                    ffi::tomoxide_vo_colsort(s.ptr, sinosort.ptr, null, nz, nrow, nc, 1, null)
                },
                "vo_colsort(sinosort)",
            )?;
            // sinosmooth = per-row median along columns.
            let sinosmooth = &sc.rl_smooth;
            ck(
                unsafe {
                    ffi::tomoxide_vo_median_axis1(
                        sinosort.ptr,
                        sinosmooth.ptr,
                        nz,
                        nrow,
                        nc,
                        size,
                        null,
                    )
                },
                "vo_median(sinosmooth)",
            )?;
            // Per-column intensity factor (f64 for normalise, f32 for detect).
            ck(
                unsafe {
                    ffi::tomoxide_vo_rs_large_listfact(
                        sinosort.ptr,
                        sinosmooth.ptr,
                        sc.lf64.ptr,
                        sc.lf32.ptr,
                        nz,
                        nrow,
                        nc,
                        ndrop,
                        null,
                    )
                },
                "vo_rs_large_listfact",
            )?;
            // Mask (no border protection in _rs_large). `sinosort`/`rl_sort` is
            // now free; it is reused as `sortdummy` below.
            self.vo_detect_mask(&sc.lf32, &sc.mask_large, nz, snr, false, sc)?;
            // Normalised result, written straight into `rl_out`. colsort reads it
            // (without modifying it) for the permutation, then scatter_masked
            // overwrites only the masked columns in place — no seeding copy.
            let out = &sc.rl_out;
            ck(
                unsafe {
                    ffi::tomoxide_vo_normalize(s.ptr, sc.lf64.ptr, out.ptr, nz, nrow, nc, null)
                },
                "vo_normalize",
            )?;
            // Sort the normalised copy for its permutation (sorted values unused,
            // dumped into `rl_sort` which `sinosort` has finished with).
            ck(
                unsafe {
                    ffi::tomoxide_vo_colsort(
                        out.ptr,
                        sc.rl_sort.ptr,
                        sc.perm2.ptr,
                        nz,
                        nrow,
                        nc,
                        1,
                        null,
                    )
                },
                "vo_colsort(out)",
            )?;
            // Overwrite masked columns of `out` with the smoothed profile.
            ck(
                unsafe {
                    ffi::tomoxide_vo_scatter_masked(
                        sc.perm2.ptr,
                        sinosmooth.ptr,
                        sc.mask_large.ptr,
                        out.ptr,
                        nz,
                        nrow,
                        nc,
                        null,
                    )
                },
                "vo_scatter_masked",
            )?;
            Ok(())
        }

        /// Vo all-stripe removal on the device f32 sinogram `ptr [nz,nproj,ncol]`,
        /// in place: `_rs_dead` (uniform-smooth → per-column L1 diff → median →
        /// detect+dilate → border-protect → bilinear dead-column fill →
        /// `_rs_large`) followed by `_rs_sort` (column sort → cross-column median
        /// → unsort). Mirrors `stripe::remove_all_stripe` for all `nz` at once;
        /// correlation parity with the CPU golden.
        fn vo_on_device(
            &self,
            ptr: *mut c_void,
            nz: usize,
            snr: f32,
            la_size: usize,
            sm_size: usize,
            sc: &VoScratch,
        ) -> Result<()> {
            let null = std::ptr::null_mut::<c_void>();
            let (nrow, nc) = (self.nproj, self.ncols);
            // Matches the CPU guard in `remove_all_stripe`.
            if nrow < 2 || nz == 0 || nc < 4 || la_size == 0 || sm_size == 0 {
                return Ok(());
            }

            // ---- _rs_dead ----
            // sinosmooth = uniform_filter1d along the projection axis (size 10).
            let smooth = &sc.big_a;
            ck(
                unsafe { ffi::tomoxide_vo_uniform_axis0(ptr, smooth.ptr, nz, nrow, nc, 10, null) },
                "vo_uniform_axis0",
            )?;
            // listdiff[z,c] = sum_r |sino - smooth|.
            ck(
                unsafe {
                    ffi::tomoxide_vo_absdiff_colsum(
                        ptr,
                        smooth.ptr,
                        sc.listdiff.ptr,
                        nz,
                        nrow,
                        nc,
                        null,
                    )
                },
                "vo_absdiff_colsum",
            )?;
            // `smooth`/`big_a` is now free; it is reused as `work` then `sortedv`.
            // listdiffbck = 1-D median of listdiff over columns (window la_size).
            ck(
                unsafe {
                    ffi::tomoxide_vo_median_axis1(
                        sc.listdiff.ptr,
                        sc.listdiffbck.ptr,
                        nz,
                        1,
                        nc,
                        la_size,
                        null,
                    )
                },
                "vo_median(listdiffbck)",
            )?;
            // listfact = listdiff / listdiffbck.
            let cols = nz * nc;
            ck(
                unsafe {
                    ffi::tomoxide_vo_ratio(
                        sc.listdiff.ptr,
                        sc.listdiffbck.ptr,
                        sc.listfact.ptr,
                        cols,
                        null,
                    )
                },
                "vo_ratio(listfact)",
            )?;
            // Dead-column mask with border protection.
            self.vo_detect_mask(&sc.listfact, &sc.mask_dead, nz, snr, true, sc)?;
            // Good-column lists, then bilinear fill of the dead columns.
            ck(
                unsafe {
                    ffi::tomoxide_vo_build_goodx(
                        sc.mask_dead.ptr,
                        sc.goodx.ptr,
                        sc.goodcount.ptr,
                        nz,
                        nc,
                        null,
                    )
                },
                "vo_build_goodx",
            )?;
            // `work` (reusing `big_a`) is written in full by interp_fill (good
            // columns copied from `ptr`, dead columns interpolated) — no seed copy.
            let work = &sc.big_a;
            ck(
                unsafe {
                    ffi::tomoxide_vo_interp_fill(
                        ptr,
                        work.ptr,
                        sc.mask_dead.ptr,
                        sc.goodx.ptr,
                        sc.goodcount.ptr,
                        nz,
                        nrow,
                        nc,
                        null,
                    )
                },
                "vo_interp_fill",
            )?;
            // Residual large-stripe pass (VoAll always runs it, norm=True). Reads
            // `work` only; result lands in `sc.rl_out`.
            self.vo_rs_large(work, nz, snr, la_size, 0.1, sc)?;
            let dead_out = &sc.rl_out;

            // ---- _rs_sort (dim=1) ----
            // Sort each column ascending, keeping the permutation. `work`/`big_a`
            // is free now (rs_large only read it), so reuse it for `sortedv`.
            let sortedv = &sc.big_a;
            ck(
                unsafe {
                    ffi::tomoxide_vo_colsort(
                        dead_out.ptr,
                        sortedv.ptr,
                        sc.perm.ptr,
                        nz,
                        nrow,
                        nc,
                        1,
                        null,
                    )
                },
                "vo_colsort(rs_sort)",
            )?;
            // Smooth the sorted profiles across columns (window sm_size).
            let smoothed = &sc.big_b;
            ck(
                unsafe {
                    ffi::tomoxide_vo_median_axis1(
                        sortedv.ptr,
                        smoothed.ptr,
                        nz,
                        nrow,
                        nc,
                        sm_size,
                        null,
                    )
                },
                "vo_median(rs_sort)",
            )?;
            // Unsort back into the original projection order, in place into `ptr`.
            ck(
                unsafe {
                    ffi::tomoxide_vo_unsort_scatter(
                        sc.perm.ptr,
                        smoothed.ptr,
                        ptr,
                        nz,
                        nrow,
                        nc,
                        null,
                    )
                },
                "vo_unsort_scatter",
            )?;
            Ok(())
        }
    }

    /// Download an `[nz, n, n]` f32 volume from `src` through the handle's pinned
    /// staging buffer and move it into an owned [`Volume`] for the writer thread.
    ///
    /// The destination being page-locked is what matters: a D2H into pageable
    /// host memory is driver-staged at ~⅓ the PCIe rate (and is synchronous, so
    /// it cannot overlap kernels), whereas the pinned copy runs at full
    /// bandwidth. The volume must then move into an owned `Send` buffer so the
    /// pinned staging can be reused by the next chunk. A fresh allocation for
    /// that buffer pays ~190 ms of first-touch page-faults on a 536 MB chunk, so
    /// `pool` supplies a warm buffer recycled from the writer (see
    /// `give_reuse_buffer`); the copy into it is ~34 ms. The pool is empty for
    /// the first few chunks (nothing returned yet), which fall back to a fresh
    /// allocation.
    fn download_volume(
        src: &DevBuf,
        pin: &mut PinnedBuf<f32>,
        pool: &mut Vec<Vec<f32>>,
        nz: usize,
        n: usize,
    ) -> Result<Volume<f32>> {
        let count = nz * n * n;
        src.to_host_f32(&mut pin.as_mut_slice()[..count])?;
        // Reuse a warm buffer from the writer if one is available, else allocate.
        let mut host = pool.pop().unwrap_or_default();
        host.resize(count, 0.0);
        host.copy_from_slice(&pin.as_slice()[..count]);
        Ok(Volume::new(
            Array3::from_shape_vec((nz, n, n), host).expect("nz*n*n volume length matches shape"),
        ))
    }

    impl CudaFbpStream {
        /// Shared back half of both streaming entry points. `self.sino` already
        /// holds the chunk's sinogram for the valid `nz` rows (f16-cast when
        /// `self.f16`; the tail zeroed by the caller when `nz < max_nz`). Builds
        /// and uploads the per-chunk filter weights, then runs pad → cuFFT filter
        /// → crop → back-project at the handle's fixed `max_nz` batch and downloads
        /// the `[nz, n, n]` volume.
        fn finish_recon(&mut self, nz: usize, geom: &Geometry) -> Result<Volume<f32>> {
            let (nproj, ncols) = (self.nproj, self.ncols);
            let w_host = build_filter_w(&self.filter, geom, nz, ncols, self.pad);
            let null = std::ptr::null_mut::<c_void>();
            let partial = nz < self.max_nz;
            // `w` is the only buffer `finish_recon` fills partially; zero its tail on
            // a partial chunk so the fixed-batch cuFFT sees zeros there.
            if partial {
                ck(
                    unsafe { ffi::tomoxide_cuda_memset(self.w.ptr, 0, self.w.bytes) },
                    "memset w",
                )?;
            }
            if self.f16 {
                self.w.copy_from_host_f16(&w_host)?;
            } else {
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
                if self.fourier {
                    // f16 Fourierrec tail: pack pairs → `cfunc_fourierrec` (f16) →
                    // unpack, mirroring the f32 fourier branch and
                    // `analytic_fourierrec_f16` at the handle's `max_nz/2`-pair batch.
                    let gc = self.gc.as_ref().expect("fourier path has gc");
                    let fc = self.fc.as_ref().expect("fourier path has fc");
                    ck(
                        unsafe {
                            ffi::tomoxide_pack_pairs_fp16(
                                self.gf.ptr,
                                gc.ptr,
                                m,
                                nproj,
                                ncols,
                                null,
                            )
                        },
                        "pack_fp16",
                    )?;
                    unsafe {
                        ffi::tomoxide_fourierrec_fp16_backproject(self.frec, fc.ptr, gc.ptr, null)
                    };
                    ck(
                        unsafe { ffi::tomoxide_unpack_pairs_fp16(fc.ptr, self.f.ptr, m, n, null) },
                        "unpack_fp16",
                    )?;
                } else {
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
                download_volume(f_f32, &mut self.out_pinned, &mut self.reuse_pool, nz, n)
            } else if self.lprec.is_some() {
                // Device-resident log-polar (f32): same pad → filter → crop as the
                // FBP tail produces the filtered sinogram in `gf`; the held
                // `LpRecDev` runtime (grids built once, reused across chunks) then
                // does spline prefilter → per-span gather/FFT/scatter into `f`.
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
                // The scatter accumulates over spans, so zero the output first.
                ck(
                    unsafe { ffi::tomoxide_cuda_memset(self.f.ptr, 0, self.f.bytes) },
                    "memset f",
                )?;
                let lp = self.lprec.as_ref().expect("lprec path has grids");
                let flc = self.flc.as_ref().expect("lprec path has flc");
                lp.run(self.gf.ptr, flc.ptr, self.f.ptr, nz, nproj, n)?;
                ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync")?;
                download_volume(&self.f, &mut self.out_pinned, &mut self.reuse_pool, nz, n)
            } else if self.fourier {
                // Device-resident Fourierrec (f32): same pad → filter → crop as the
                // FBP tail, then pack pairs → `cfunc_fourierrec` → unpack, mirroring
                // `analytic_fourierrec` but at the handle's fixed `max_nz` batch
                // (the `max_nz/2`-pair handle reuses the cuFFT plans across chunks).
                let gc = self.gc.as_ref().expect("fourier path has gc");
                let fc = self.fc.as_ref().expect("fourier path has fc");
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
                    unsafe { ffi::tomoxide_pack_pairs(self.gf.ptr, gc.ptr, m, nproj, ncols, null) },
                    "pack",
                )?;
                unsafe { ffi::tomoxide_fourierrec_backproject(self.frec, fc.ptr, gc.ptr, null) };
                ck(
                    unsafe { ffi::tomoxide_unpack_pairs(fc.ptr, self.f.ptr, m, n, null) },
                    "unpack",
                )?;
                ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync")?;
                download_volume(&self.f, &mut self.out_pinned, &mut self.reuse_pool, nz, n)
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
                download_volume(&self.f, &mut self.out_pinned, &mut self.reuse_pool, nz, n)
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

            // Parallel beam and laminography (tilted axis) are both handled here;
            // cone beam is not. Laminography routes to a dedicated whole-stack
            // single-GPU path below (its output z-extent and detector-row coupling
            // break the per-slice chunking the parallel path uses).
            if !matches!(geom.beam, Beam::Parallel | Beam::Laminography { .. }) {
                return Err(Error::InvalidParam(
                    "cuda analytic reconstruct supports parallel beam and laminography only".into(),
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

            let filter = make_fbp_filter(params.filter_name, ncols, RampShape::Wint)?;
            let pad = filter.len();
            let pad_side = pad / 2 - ncols / 2;
            let w = build_filter_w(&filter, geom, nz, ncols, pad);
            let nfreq2 = (pad / 2 + 1) * 2; // floats per z row of `w`

            // Laminography (tilted rotation axis): whole projection stack, single
            // GPU, single output-z chunk. The tilt couples every detector row into
            // every output voxel, so the parallel-beam per-slice/multi-GPU split
            // does not apply; `rh` (recon height) differs from the detector-row
            // count `nz`. Only Fbp/Linerec (the direct back-projector) and f32 are
            // supported for now — fourierrec-lamino and f16 are not wired.
            if let Beam::Laminography { phi } = geom.beam {
                if !matches!(algorithm, Algorithm::Fbp | Algorithm::Linerec) {
                    return Err(Error::InvalidParam(format!(
                        "cuda laminography supports Fbp/Linerec only; got {algorithm:?}"
                    )));
                }
                if params.dtype == crate::dtype::Dtype::F16 {
                    return Err(Error::InvalidParam(
                        "cuda laminography f16 path is not implemented; use the default f32 dtype"
                            .into(),
                    ));
                }
                let lamino_angle_deg =
                    (phi - std::f32::consts::FRAC_PI_2) * 180.0 / std::f32::consts::PI;
                let rh = params
                    .lamino_rh
                    .unwrap_or_else(|| lamino_recon_height(nz, lamino_angle_deg));
                let devices = selected_devices();
                unsafe { ffi::tomoxide_cuda_set_device(*devices.first().unwrap_or(&0)) };
                let vol = analytic_lamino_chunk(
                    raw, &w, theta, nz, nproj, ncols, n, pad, pad_side, phi, rh, 0,
                )?;
                return Ok(Volume::new(
                    Array3::from_shape_vec((rh, n, n), vol)
                        .map_err(|e| Error::InvalidParam(format!("cuda volume shape: {e}")))?,
                ));
            }

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

        /// Reuse one set of device handles + uploaded grids across all streaming
        /// chunks (see [`CudaFbpStream`]). Handle-reusing for FBP/Linerec
        /// (`cfunc_filter`/`cfunc_linerec`), Fourierrec (`cfunc_fourierrec`), and
        /// lprec (the [`LpRecDev`] grids — built once here, not per chunk). Gridrec
        /// returns `None` and the caller falls back to per-chunk [`reconstruct`].
        /// Binds the first selected device, as the f16 one-shot path does, since
        /// the handles are device-resident.
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
                || !matches!(
                    algorithm,
                    Algorithm::Fbp | Algorithm::Linerec | Algorithm::Fourierrec | Algorithm::Lprec
                )
            {
                return Ok(None);
            }
            let n = params.num_gridx.unwrap_or(ncols);
            if n != ncols {
                return Ok(None); // square-grid only, like `reconstruct`
            }
            let f16 = params.dtype == crate::dtype::Dtype::F16;
            let fourier = matches!(algorithm, Algorithm::Fourierrec);
            let is_lprec = matches!(algorithm, Algorithm::Lprec);
            // Fourierrec packs slice pairs (s, s+max_nz/2) for the real-FFT path, so
            // it needs an even chunk (f32 and f16 both have a device-resident tail;
            // see `finish_recon`'s fourier branch). An odd chunk → per-chunk fallback.
            if fourier && max_nz % 2 != 0 {
                return Ok(None);
            }
            // lprec's gather/scatter + spline runtime is f32-only (no f16 port);
            // f16 lprec falls back to the per-chunk host-interp path.
            if is_lprec && f16 {
                return Ok(None);
            }
            // `make_fbp_filter` pads to `(4·ncols).next_power_of_two()`, always a
            // power of two, so the f16 half-cuFFT width constraint holds by
            // construction (mirrors the assert in `reconstruct`).
            let filter = make_fbp_filter(params.filter_name, ncols, RampShape::Wint)?;
            let devices = selected_devices();
            unsafe { ffi::tomoxide_cuda_set_device(*devices.first().unwrap_or(&0)) };
            // Build + upload the lprec grids once for the whole stream (the chunk
            // loop reuses them); other methods carry no grids.
            let lprec = if is_lprec {
                Some(LpRecDev::new(n, geom.angles.0.len(), self)?)
            } else {
                None
            };
            let recon = CudaFbpStream::new(
                filter,
                &geom.angles.0,
                ncols,
                n,
                max_nz,
                f16,
                fourier,
                lprec,
            )?;
            Ok(Some(Box::new(recon)))
        }
    }

    impl FbpFilter for CudaBackend {
        /// FBP filter via the shared [`make_fbp_filter`] with
        /// [`RampShape::Wint`] — the CUDA backend ports tomocupy, so it builds
        /// tomocupy's `_wint` quadrature ramp (the CPU/wgpu backends build
        /// tomopy's linear ramp). Apodization/padding/layout are shared.
        fn make_filter(&self, name: FilterName, n: usize) -> Result<Vec<f32>> {
            make_fbp_filter(name, n, RampShape::Wint)
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
    pub(super) fn selected_devices() -> Vec<i32> {
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

    /// Total memory (bytes) on the **current** device. Caller must have already
    /// `cudaSetDevice`'d the device it means to query. Falls back to a
    /// conservative 8 GiB if the query fails.
    ///
    /// Used to size the per-slice worker pool: unlike free memory, total is a
    /// stable property of the GPU and does not shrink as thread-local cuFFT plan
    /// caches fill up, so the in-flight cap it yields is the same on every
    /// reconstruction (free memory would collapse the cap once plans are cached).
    fn device_total_bytes() -> usize {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let rc = unsafe { ffi::tomoxide_cuda_mem_info(&mut free, &mut total) };
        if rc == 0 && total > 0 {
            total
        } else {
            8 * 1024 * 1024 * 1024
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

    /// A device-pinned rayon pool sized to exactly `nthreads`, built once per
    /// `(device, nthreads)` and cached for the process lifetime.
    ///
    /// The per-slice loop runs here instead of on the full host-core pool so that
    /// the number of *distinct* worker threads equals the in-flight cap. That
    /// matters because each worker lazily creates a **thread-local cuFFT plan**
    /// (see `cuda/fft.cu`) that is never destroyed: if the loop fanned across all
    /// 96 host cores, up to 96 oversampled `(2n)²` plan workspaces would
    /// accumulate and exhaust VRAM at large `n` (one GPU OOMs, or the next
    /// reconstruction sees no free memory and collapses to a serial cap). Pinning
    /// the loop to `nthreads = max_inflight` makes plan-count == concurrency ==
    /// cap by construction — one number governs both. Pools are leaked (process
    /// lifetime, like `device_pools`) so plans persist and are reused; only a
    /// handful of distinct `(device, nthreads)` keys arise per run.
    fn slice_pool(device: i32, nthreads: usize) -> &'static ThreadPool {
        static REG: OnceLock<Mutex<HashMap<(i32, usize), &'static ThreadPool>>> = OnceLock::new();
        let reg = REG.get_or_init(|| Mutex::new(HashMap::new()));
        let mut m = reg.lock().unwrap();
        if let Some(p) = m.get(&(device, nthreads)) {
            return p;
        }
        let pool: &'static ThreadPool = Box::leak(Box::new(
            ThreadPoolBuilder::new()
                .num_threads(nthreads.max(1))
                .start_handler(move |_| {
                    unsafe { ffi::tomoxide_cuda_set_device(device) };
                })
                .build()
                .expect("build cuda slice pool"),
        ));
        m.insert((device, nthreads), pool);
        pool
    }

    /// How many per-slice reconstructions may run concurrently on one device —
    /// which is also the worker-thread count handed to [`slice_pool`], so it
    /// bounds VRAM by construction (one thread ⇒ one in-flight slice ⇒ one
    /// persistent cuFFT plan).
    ///
    /// Each worker holds, for the lifetime of the process, an oversampled `(2n)²`
    /// cuFFT plan workspace, plus — while a slice is in flight — its `(2n)²`
    /// complex grid and padded staging buffers. The persistent plan dominates at
    /// large `n`, so the budget is taken against **total** device memory (a stable
    /// figure) rather than free memory: free memory shrinks as plans are cached,
    /// which would make the cap collapse on the second reconstruction. Cap at
    /// ~70% of total over a per-slice footprint of 6× the `(2n)²` grid (grid +
    /// plan workspace + staging), clamped to the host-core budget. Smaller `n`
    /// clamps to the pool size (plans are tiny, full host parallelism is kept).
    /// `TOMOXIDE_CUDA_MAX_INFLIGHT` overrides the computed value.
    fn max_inflight(n: usize, total_bytes: usize, pool_threads: usize) -> usize {
        if let Ok(s) = std::env::var("TOMOXIDE_CUDA_MAX_INFLIGHT") {
            if let Ok(v) = s.trim().parse::<usize>() {
                if v >= 1 {
                    return v.min(pool_threads);
                }
            }
        }
        // 6× the (2n)² complex grid covers the persistent plan workspace + the
        // in-flight grid + padded staging; calibrated so large-n fits VRAM with
        // headroom (e.g. n=2048 → ~27 of 32 GB) while small-n stays pool-bound.
        let per_slice = 6 * (2 * n) * (2 * n) * std::mem::size_of::<crate::dtype::Complex32>();
        let by_mem = (total_bytes / 100 * 70) / per_slice.max(1);
        by_mem.clamp(1, pool_threads.max(1))
    }

    impl crate::backend::Fft for CudaBackend {
        /// Fan the per-slice loop across the selected GPUs (and host cores).
        /// Slices are partitioned into one contiguous chunk per device; each
        /// chunk runs on that device's [`slice_pool`] — a pinned pool sized to the
        /// device's in-flight cap — all devices concurrently.
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

            // Single device: run on a pinned pool sized to `max_inflight`, so the
            // in-flight slice count == worker count == persistent cuFFT plan count
            // (see `slice_pool` / `max_inflight`). Total VRAM (queried on a pinned
            // worker) gives a stable cap that does not collapse once plans cache.
            if d <= 1 {
                let threads = dp.pools[0].current_num_threads();
                let total = dp.pools[0].install(device_total_bytes);
                let inflight = max_inflight(n, total, threads);
                let pool = slice_pool(dp.devices[0], inflight);
                return pool.install(move || {
                    slabs
                        .into_par_iter()
                        .enumerate()
                        .try_for_each(|(row, slab)| f(row, slab))
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
                    .zip(dp.devices.iter())
                    .map(|(((off, chunk), pool), &dev)| {
                        scope.spawn(move || -> Result<()> {
                            // Size this device's per-slice pool to its in-flight
                            // cap (total VRAM queried on a worker pinned to `dev`),
                            // matching the single-GPU path so plan-count == cap.
                            let threads = pool.current_num_threads();
                            let total = pool.install(device_total_bytes);
                            let inflight = max_inflight(n, total, threads);
                            let spool = slice_pool(dev, inflight);
                            spool.install(|| {
                                chunk
                                    .into_par_iter()
                                    .enumerate()
                                    .try_for_each(|(i, slab)| f(off + i, slab))
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
            let flat = complex_as_f32_mut(buf);
            with_fft_scratch(std::mem::size_of_val(flat), |d| {
                d.copy_from_host_f32(flat)?;
                ck(
                    unsafe { ffi::tomoxide_fft_1d(d.ptr, len, batch, inverse as i32) },
                    "fft_1d",
                )?;
                d.to_host_f32(flat)
            })
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
            let flat = complex_as_f32_mut(buf);
            with_fft_scratch(std::mem::size_of_val(flat), |d| {
                d.copy_from_host_f32(flat)?;
                ck(
                    unsafe { ffi::tomoxide_fft_2d(d.ptr, rows, cols, batch, inverse as i32) },
                    "fft_2d",
                )?;
                d.to_host_f32(flat)
            })
        }
    }

    /// Reinterpret a `Complex32` slice as the equivalent interleaved
    /// `[re, im, …]` f32 slice — zero-copy, both for upload and for receiving the
    /// transform back in place.
    ///
    /// Sound because `Complex32 = num_complex::Complex<f32>` is `#[repr(C)]` with
    /// fields `{ re: f32, im: f32 }`, so `N` complex values occupy exactly the
    /// same bytes as `2N` contiguous f32 (matching cuFFT's `cufftComplex`). This
    /// replaces the per-call interleave/deinterleave `Vec<f32>` allocate-and-copy
    /// in `fft_1d`/`fft_2d` — the dominant host overhead of the composed path.
    fn complex_as_f32_mut(buf: &mut [crate::dtype::Complex32]) -> &mut [f32] {
        let len = buf.len() * 2;
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, len) }
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

    /// Device-resident log-polar geometry grids plus the per-chunk runtime,
    /// shared by the whole-volume [`LpRecReconstruct`] path and the streaming
    /// reconstructor ([`CudaFbpStream`]). Built once — host `build_grids` then a
    /// single upload — and reused across every tile/chunk, so a streamed job pays
    /// the (host) precompute and the grid upload exactly once.
    struct LpRecDev {
        ntheta: usize,
        nrho: usize,
        n_lp: usize,
        n_w: usize,
        n_c: usize,
        /// Full Hermitian convolution kernel `[nrho, ntheta]` (folds the
        /// deapodization + the tomocupy constant).
        kfull: DevBuf,
        /// Span-independent target index sets (i32).
        lpids: DevBuf,
        wids: DevBuf,
        cids: DevBuf,
        /// Per-span coordinate arrays, NSPAN spans concatenated contiguously
        /// (span `k` lives at offset `k * npts`).
        lp2p1: DevBuf,
        lp2p2: DevBuf,
        lp2p1w: DevBuf,
        lp2p2w: DevBuf,
        c2lp1: DevBuf,
        c2lp2: DevBuf,
    }

    impl LpRecDev {
        /// Precompute the log-polar grids on the host (`fft` backs the small
        /// precompute transforms — the CUDA backend's own Fft is fine) and upload
        /// them.
        fn new(n: usize, nproj: usize, fft: &dyn crate::backend::Fft) -> Result<Self> {
            use crate::recon::lprec::{build_grids, LP_NSPAN};
            let grids = build_grids(n, nproj, fft)?;
            let kfull_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    grids.kfull.as_ptr() as *const f32,
                    grids.kfull.len() * 2,
                )
            };
            // The log-polar function is real, so the per-span convolution runs as
            // an in-place R2C → cmul → C2R. The spectrum is the half-complex
            // `[nrho, ntheta_c]` (ntheta_c = ntheta/2+1, the conjugate-symmetric
            // half), so `kfull` — the FFT of a real kernel, hence Hermitian — is
            // cropped to its first `ntheta_c` columns per row to match.
            let (nrho, ntheta) = (grids.nrho, grids.ntheta);
            let ntheta_c = ntheta / 2 + 1;
            let ntheta_pad = 2 * ntheta_c;
            let mut kfull_half = Vec::with_capacity(nrho * ntheta_c * 2);
            for row in 0..nrho {
                let base = row * ntheta * 2;
                for col in 0..ntheta_c {
                    kfull_half.push(kfull_f32[base + col * 2]);
                    kfull_half.push(kfull_f32[base + col * 2 + 1]);
                }
            }
            let to_i32 = |v: &[usize]| -> Vec<i32> { v.iter().map(|&x| x as i32).collect() };
            // Gather targets index the `[nrho, ntheta]` grid row-major; remap them
            // to the padded real layout (row stride ntheta_pad) used in place by
            // the R2C buffer. cids (Cartesian-output indices) are unaffected.
            let to_padded_i32 = |v: &[usize]| -> Vec<i32> {
                v.iter()
                    .map(|&t| ((t / ntheta) * ntheta_pad + (t % ntheta)) as i32)
                    .collect()
            };
            let concat = |spans: &[Vec<f32>; LP_NSPAN]| -> Vec<f32> {
                let mut v = Vec::with_capacity(spans.iter().map(Vec::len).sum());
                for sp in spans {
                    v.extend_from_slice(sp);
                }
                v
            };
            Ok(LpRecDev {
                ntheta: grids.ntheta,
                nrho: grids.nrho,
                n_lp: grids.lpids.len(),
                n_w: grids.wids.len(),
                n_c: grids.cids.len(),
                kfull: DevBuf::from_host_f32(&kfull_half)?,
                lpids: DevBuf::from_host_i32(&to_padded_i32(&grids.lpids))?,
                wids: DevBuf::from_host_i32(&to_padded_i32(&grids.wids))?,
                cids: DevBuf::from_host_i32(&to_i32(&grids.cids))?,
                lp2p1: DevBuf::from_host_f32(&concat(&grids.lp2p1))?,
                lp2p2: DevBuf::from_host_f32(&concat(&grids.lp2p2))?,
                lp2p1w: DevBuf::from_host_f32(&concat(&grids.lp2p1w))?,
                lp2p2w: DevBuf::from_host_f32(&concat(&grids.lp2p2w))?,
                c2lp1: DevBuf::from_host_f32(&concat(&grids.c2lp1))?,
                c2lp2: DevBuf::from_host_f32(&concat(&grids.c2lp2))?,
            })
        }

        /// Complex half-width of the R2C spectrum (`ntheta/2 + 1`).
        fn ntheta_c(&self) -> usize {
            self.ntheta / 2 + 1
        }

        /// Padded real row width of the in-place R2C buffer (`2*(ntheta/2+1)`).
        fn ntheta_pad(&self) -> usize {
            2 * self.ntheta_c()
        }

        /// Bytes of the in-place R2C work buffer for `cz` slices: the padded real
        /// `[cz, nrho, ntheta_pad]` overlays the half-complex `[cz, nrho,
        /// ntheta_c]`, i.e. `cz*nrho*ntheta_c` complex — roughly half the old
        /// `[cz, nrho, ntheta]` full-complex buffer.
        fn flc_bytes(&self, cz: usize) -> usize {
            cz * self.nrho * self.ntheta_c() * 2 * std::mem::size_of::<f32>()
        }

        /// Per-chunk runtime (port of `recon/lprec.rs::process_row`): `g` is the
        /// **filtered** sinogram `[cz, nproj, n]`, consumed in place as the
        /// spline-coefficient buffer; `flc` a `[cz, nrho, ntheta]` complex scratch
        /// (clobbered); `f` the `[cz, n, n]` output, which the caller must zero
        /// first (the scatter accumulates over the NSPAN spans).
        fn run(
            &self,
            g: *mut c_void,
            flc: *mut c_void,
            f: *mut c_void,
            cz: usize,
            nproj: usize,
            n: usize,
        ) -> Result<()> {
            use crate::recon::lprec::LP_NSPAN;
            let null = std::ptr::null_mut::<c_void>();
            let (nrho, ntheta) = (self.nrho, self.ntheta);
            let (ntheta_c, ntheta_pad) = (self.ntheta_c(), self.ntheta_pad());
            let off = |buf: &DevBuf, elems: usize| -> *const c_void {
                unsafe { (buf.ptr as *const f32).add(elems) as *const c_void }
            };
            ck(
                unsafe {
                    ffi::tomoxide_lprec_prefilter_rows(g, cz as i32, nproj as i32, n as i32, null)
                },
                "lprec prefilter rows",
            )?;
            ck(
                unsafe {
                    ffi::tomoxide_lprec_prefilter_cols(g, cz as i32, nproj as i32, n as i32, null)
                },
                "lprec prefilter cols",
            )?;
            let flc_bytes = self.flc_bytes(cz);
            for k in 0..LP_NSPAN {
                ck(
                    unsafe { ffi::tomoxide_cuda_memset(flc, 0, flc_bytes) },
                    "memset flc",
                )?;
                ck(
                    unsafe {
                        ffi::tomoxide_lprec_gather(
                            g,
                            flc,
                            self.lpids.ptr,
                            off(&self.lp2p2, k * self.n_lp),
                            off(&self.lp2p1, k * self.n_lp),
                            self.n_lp as i32,
                            cz as i32,
                            nproj as i32,
                            n as i32,
                            nrho as i32,
                            ntheta_pad as i32,
                            null,
                        )
                    },
                    "lprec gather main",
                )?;
                ck(
                    unsafe {
                        ffi::tomoxide_lprec_gather(
                            g,
                            flc,
                            self.wids.ptr,
                            off(&self.lp2p2w, k * self.n_w),
                            off(&self.lp2p1w, k * self.n_w),
                            self.n_w as i32,
                            cz as i32,
                            nproj as i32,
                            n as i32,
                            nrho as i32,
                            ntheta_pad as i32,
                            null,
                        )
                    },
                    "lprec gather wrap",
                )?;
                ck(
                    unsafe { ffi::tomoxide_fft_2d_r2c(flc, nrho, ntheta, cz) },
                    "lprec fft fwd (r2c)",
                )?;
                ck(
                    unsafe {
                        ffi::tomoxide_lprec_cmul(
                            flc,
                            self.kfull.ptr,
                            cz as i32,
                            nrho as i32,
                            ntheta_c as i32,
                            null,
                        )
                    },
                    "lprec cmul",
                )?;
                ck(
                    unsafe { ffi::tomoxide_fft_2d_c2r(flc, nrho, ntheta, cz) },
                    "lprec fft inv (c2r)",
                )?;
                ck(
                    unsafe {
                        ffi::tomoxide_lprec_scatter(
                            flc,
                            f,
                            self.cids.ptr,
                            off(&self.c2lp1, k * self.n_c),
                            off(&self.c2lp2, k * self.n_c),
                            self.n_c as i32,
                            cz as i32,
                            n as i32,
                            nrho as i32,
                            ntheta as i32,
                            ntheta_pad as i32,
                            null,
                        )
                    },
                    "lprec scatter",
                )?;
            }
            Ok(())
        }
    }

    impl LpRecReconstruct for CudaBackend {
        /// Device-resident log-polar reconstruction — the GPU port of
        /// [`crate::recon::lprec`]'s per-slice runtime. The geometry grids are
        /// precomputed on the host by `build_grids` (the same precompute the CPU
        /// path uses) and uploaded once; the cubic-B-spline prefilter, the
        /// polar↔log-polar gather/scatter, and the per-span FFT convolution all
        /// run on the device (`cuda/lprec.cu`), replacing the host interpolation
        /// that dominated the composed `Fft`-only path. The slice batch is
        /// z-tiled to bound the large `[tile, nrho, ntheta]` complex work buffer.
        fn reconstruct(
            &self,
            filtered: &Tomo<f32>,
            geom: &Geometry,
            n: usize,
        ) -> Result<Volume<f32>> {
            if geom.beam != Beam::Parallel {
                return Err(Error::InvalidParam(
                    "cuda lprec supports parallel beam only".into(),
                ));
            }
            let s = filtered.as_layout(Layout::Sinogram); // [nz, nproj, ncols]
            let (nz, nproj, ncols) = s.array.dim();
            if ncols != n {
                return Err(Error::InvalidParam(format!(
                    "cuda lprec needs a square grid = detector width {ncols}; got {n}"
                )));
            }
            if nproj < 2 {
                return Err(Error::InvalidParam("cuda lprec needs >= 2 angles".into()));
            }
            let angles = &geom.angles.0;
            if angles.len() != nproj {
                return Err(Error::ShapeMismatch {
                    expected: format!("{nproj} angles"),
                    found: angles.len().to_string(),
                });
            }
            // Equally-spaced [0, π) guard (matches the CPU lprec precondition the
            // log-polar span tiling assumes).
            let dth = (angles[1] - angles[0]).abs();
            let nproj_test = (std::f32::consts::PI / dth).round() as usize;
            if nproj_test != nproj {
                return Err(Error::InvalidParam(
                    "cuda lprec requires equally spaced angles spanning [0, π)".into(),
                ));
            }
            let raw = s
                .array
                .as_slice()
                .ok_or_else(|| Error::InvalidParam("non-contiguous sinogram".into()))?;

            let devices = selected_devices();
            unsafe { ffi::tomoxide_cuda_set_device(*devices.first().unwrap_or(&0)) };

            // Precompute + upload the geometry grids once (shared device runtime).
            let fft: &dyn crate::backend::Fft = self;
            let lp = LpRecDev::new(n, nproj, fft)?;
            let (nrho, ntheta) = (lp.nrho, lp.ntheta);

            // z-tile so the [tile, nrho, ntheta] complex work buffer (plus the g
            // and f tiles) fits. A third of free memory leaves headroom for the
            // cuFFT plan and the resident grid uploads.
            let fsz = std::mem::size_of::<f32>();
            let per_slice = nrho * ntheta * 2 * fsz + nproj * ncols * fsz + n * n * fsz;
            let tile = (device_free_bytes() / 3 / per_slice.max(1)).clamp(1, nz);

            let mut vol = vec![0.0f32; nz * n * n];
            let mut z = 0;
            while z < nz {
                let cz = tile.min(nz - z);
                let g = DevBuf::from_host_f32(&raw[z * nproj * ncols..(z + cz) * nproj * ncols])?;
                let flc = DevBuf::zeroed(lp.flc_bytes(cz))?;
                let f = DevBuf::zeroed(cz * n * n * fsz)?; // scatter accumulates → zeroed
                lp.run(g.ptr, flc.ptr, f.ptr, cz, nproj, n)?;
                ck(unsafe { ffi::tomoxide_cuda_sync() }, "lprec sync")?;
                f.to_host_f32(&mut vol[z * n * n..(z + cz) * n * n])?;
                z += cz;
            }

            Ok(Volume::new(
                Array3::from_shape_vec((nz, n, n), vol)
                    .map_err(|e| Error::InvalidParam(format!("cuda lprec shape: {e}")))?,
            ))
        }
    }

    #[cfg(all(test, feature = "cuda"))]
    mod stripe_gpu_tests {
        use super::*;
        use crate::data::{Layout, Tomo};
        use crate::params::StripeMethod;
        use ndarray::Array3;

        fn pearson(a: &[f32], b: &[f32]) -> f64 {
            let n = a.len() as f64;
            let (mut sa, mut sb) = (0.0, 0.0);
            for i in 0..a.len() {
                sa += a[i] as f64;
                sb += b[i] as f64;
            }
            let (ma, mb) = (sa / n, sb / n);
            let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
            for i in 0..a.len() {
                let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
                cov += da * db;
                va += da * da;
                vb += db * db;
            }
            cov / (va.sqrt() * vb.sqrt())
        }

        // Synthetic sinogram [nz, nproj, ncol]: a smooth angle/column structure
        // plus a per-column constant offset (the classic stripe → ring), so the
        // Titarenko correction has something to remove.
        fn synthetic(nz: usize, nproj: usize, ncol: usize) -> Vec<f32> {
            let mut v = vec![0.0f32; nz * nproj * ncol];
            for z in 0..nz {
                for p in 0..nproj {
                    for c in 0..ncol {
                        let base = (p as f32 * 0.05).sin() + (c as f32 * 0.03).cos();
                        let stripe = (((c * 7 + 13) % 17) as f32 / 17.0 - 0.5) * 0.4;
                        v[(z * nproj + p) * ncol + c] = base + 1.0 + stripe + z as f32 * 0.01;
                    }
                }
            }
            v
        }

        // The GPU Titarenko kernel solves the same (HᵀH + αI)q = f systems as the
        // CPU golden via CG; parallel f64 reductions differ in summation order, so
        // it is held to correlation parity, not bit-exactness.
        #[test]
        fn stripe_ti_matches_cpu_golden() {
            let (nz, nproj, ncol) = (4usize, 180usize, 192usize);
            let host = synthetic(nz, nproj, ncol);

            // CPU golden.
            let mut cpu = Tomo::new(
                Array3::from_shape_vec((nz, nproj, ncol), host.clone()).unwrap(),
                Layout::Sinogram,
            );
            crate::prep::remove_stripe(
                &mut cpu,
                StripeMethod::Ti {
                    nblock: 0,
                    beta: 1.5,
                },
            )
            .unwrap();
            let cpu_v: Vec<f32> = cpu.array.iter().copied().collect();

            // GPU: run the kernel on a device copy of the same sinogram.
            let _b = CudaBackend::new().expect("cuda backend");
            ck(unsafe { ffi::tomoxide_cuda_set_device(0) }, "set_device").unwrap();
            let dev = DevBuf::from_host_f32(&host).unwrap();
            let scratch = DevBuf::zeroed(nz * 7 * ncol * std::mem::size_of::<f64>()).unwrap();
            let null = std::ptr::null_mut::<c_void>();
            ck(
                unsafe {
                    ffi::tomoxide_stripe_ti(dev.ptr, nz, nproj, ncol, 1.5, scratch.ptr, null)
                },
                "stripe_ti",
            )
            .unwrap();
            ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync").unwrap();
            let mut gpu_v = vec![0.0f32; nz * nproj * ncol];
            dev.to_host_f32(&mut gpu_v).unwrap();

            let r = pearson(&cpu_v, &gpu_v);
            let max_abs: f32 = cpu_v
                .iter()
                .zip(&gpu_v)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let nan = gpu_v.iter().filter(|v| !v.is_finite()).count();
            println!("TI GPU vs CPU: pearson={r:.8} max_abs_diff={max_abs:.3e} nan={nan}");
            assert_eq!(nan, 0, "GPU TI produced non-finite values");
            // Same CG systems, same combine: matches the CPU golden to the f32
            // reduction-order floor (parallel dot products reassociate the sums).
            assert!(r > 0.99999, "TI GPU vs CPU correlation too low: {r}");
        }

        // Validate the db5 DWT/IDWT kernels against the pywt oracle (the same
        // [5,4] case as the CPU `wavelet::dwt2_idwt2_match_pywt` test).
        #[test]
        fn fw_wavelet_matches_pywt() {
            let _b = CudaBackend::new().expect("cuda backend");
            ck(unsafe { ffi::tomoxide_cuda_set_device(0) }, "set_device").unwrap();
            let null = std::ptr::null_mut::<c_void>();
            let f8 = std::mem::size_of::<f64>();
            let (nz, r, c) = (1usize, 5usize, 4usize);
            let (oc, or) = ((c + 9) / 2, (r + 9) / 2); // 6, 7
            let a: Vec<f64> = (0..r * c).map(|i| (i + 1) as f64).collect();
            let dev_a = DevBuf::from_host_f64(&a).unwrap();

            // Forward dwt2: rows then cols.
            let cols_a = DevBuf::zeroed(nz * r * oc * f8).unwrap();
            let cols_d = DevBuf::zeroed(nz * r * oc * f8).unwrap();
            ck(
                unsafe {
                    ffi::tomoxide_fw_dwt_rows(dev_a.ptr, cols_a.ptr, cols_d.ptr, nz, r, c, null)
                },
                "dwt_rows",
            )
            .unwrap();
            let (ca, ch, cv, cd) = (
                DevBuf::zeroed(nz * or * oc * f8).unwrap(),
                DevBuf::zeroed(nz * or * oc * f8).unwrap(),
                DevBuf::zeroed(nz * or * oc * f8).unwrap(),
                DevBuf::zeroed(nz * or * oc * f8).unwrap(),
            );
            ck(
                unsafe { ffi::tomoxide_fw_dwt_cols(cols_a.ptr, ca.ptr, ch.ptr, nz, r, oc, null) },
                "dwt_cols a",
            )
            .unwrap();
            ck(
                unsafe { ffi::tomoxide_fw_dwt_cols(cols_d.ptr, cv.ptr, cd.ptr, nz, r, oc, null) },
                "dwt_cols d",
            )
            .unwrap();
            ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync").unwrap();

            let ca_h = ca.to_host_f64(nz * or * oc).unwrap();
            let ch_h = ch.to_host_f64(nz * or * oc).unwrap();
            let cv_h = cv.to_host_f64(nz * or * oc).unwrap();
            let cd_h = cd.to_host_f64(nz * or * oc).unwrap();
            assert!(
                (ca_h[0] - 32.046313561434275).abs() < 1e-9,
                "ca {}",
                ca_h[0]
            );
            assert!((ch_h[0] - 0.426202580246038).abs() < 1e-9, "ch {}", ch_h[0]);
            assert!(
                (cv_h[0] - 0.1888883078383144).abs() < 1e-9,
                "cv {}",
                cv_h[0]
            );
            assert!(cd_h[0].abs() < 1e-9, "cd {}", cd_h[0]);

            // Inverse idwt2: cols then rows. rR = 2*or+2-10 = 6, rC = 2*oc+2-10 = 4.
            let rr = 2 * or + 2 - 10;
            let colsa2 = DevBuf::zeroed(nz * rr * oc * f8).unwrap();
            let colsd2 = DevBuf::zeroed(nz * rr * oc * f8).unwrap();
            ck(
                unsafe { ffi::tomoxide_fw_idwt_cols(ca.ptr, ch.ptr, colsa2.ptr, nz, or, oc, null) },
                "idwt_cols a",
            )
            .unwrap();
            ck(
                unsafe { ffi::tomoxide_fw_idwt_cols(cv.ptr, cd.ptr, colsd2.ptr, nz, or, oc, null) },
                "idwt_cols d",
            )
            .unwrap();
            let rc2 = 2 * oc + 2 - 10;
            let out = DevBuf::zeroed(nz * rr * rc2 * f8).unwrap();
            ck(
                unsafe {
                    ffi::tomoxide_fw_idwt_rows(colsa2.ptr, colsd2.ptr, out.ptr, nz, rr, oc, null)
                },
                "idwt_rows",
            )
            .unwrap();
            ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync").unwrap();
            let out_h = out.to_host_f64(nz * rr * rc2).unwrap();
            assert_eq!((rr, rc2), (6, 4));
            for row in 0..5 {
                for col in 0..4 {
                    let want = a[row * 4 + col];
                    let got = out_h[row * rc2 + col];
                    assert!(
                        (got - want).abs() < 1e-8,
                        "recon[{row},{col}] {got} vs {want}"
                    );
                }
            }
            assert!(
                (out_h[5 * rc2 + 3] - 20.0).abs() < 1e-8,
                "recon[5,3] {}",
                out_h[5 * rc2 + 3]
            );
        }

        // The Vo per-column bitonic sort must reproduce Rust's stable ascending
        // `sort_by` exactly — including tie-breaking by original row — so the
        // unsort scatter lands every value back on the right projection. Uses a
        // column with deliberate ties (clamped values) to exercise the composite
        // (value, row) key.
        #[test]
        fn vo_colsort_matches_stable_sort() {
            let _b = CudaBackend::new().expect("cuda backend");
            ck(unsafe { ffi::tomoxide_cuda_set_device(0) }, "set_device").unwrap();
            let null = std::ptr::null_mut::<c_void>();
            let (nz, nrow, nc) = (2usize, 90usize, 7usize);
            let mut host = vec![0.0f32; nz * nrow * nc];
            for z in 0..nz {
                for p in 0..nrow {
                    for c in 0..nc {
                        // Quantised to create many ties; pow2 padding (128) exercised.
                        let raw = (p as f32 * 0.3 + c as f32 * 1.7 + z as f32).sin();
                        host[(z * nrow + p) * nc + c] = (raw * 4.0).round() / 4.0;
                    }
                }
            }
            let dev = DevBuf::from_host_f32(&host).unwrap();
            let f4 = std::mem::size_of::<f32>();
            let i4 = std::mem::size_of::<i32>();
            let sorted = DevBuf::zeroed(nz * nrow * nc * f4).unwrap();
            let perm = DevBuf::zeroed(nz * nrow * nc * i4).unwrap();
            ck(
                unsafe {
                    ffi::tomoxide_vo_colsort(dev.ptr, sorted.ptr, perm.ptr, nz, nrow, nc, 1, null)
                },
                "vo_colsort",
            )
            .unwrap();
            ck(unsafe { ffi::tomoxide_cuda_sync() }, "sync").unwrap();
            let mut sorted_h = vec![0.0f32; nz * nrow * nc];
            sorted.to_host_f32(&mut sorted_h).unwrap();
            let mut perm_h = vec![0i32; nz * nrow * nc];
            {
                let bytes = std::mem::size_of_val(perm_h.as_slice());
                ck(
                    unsafe {
                        ffi::tomoxide_cuda_memcpy_d2h(
                            perm_h.as_mut_ptr() as *mut c_void,
                            perm.ptr,
                            bytes,
                        )
                    },
                    "perm d2h",
                )
                .unwrap();
            }
            for z in 0..nz {
                for c in 0..nc {
                    let mut idx: Vec<usize> = (0..nrow).collect();
                    idx.sort_by(|&a, &b| {
                        host[(z * nrow + a) * nc + c].total_cmp(&host[(z * nrow + b) * nc + c])
                    });
                    for rank in 0..nrow {
                        let o = (z * nrow + rank) * nc + c;
                        assert_eq!(
                            perm_h[o] as usize, idx[rank],
                            "perm mismatch z={z} c={c} rank={rank}"
                        );
                        let want = host[(z * nrow + idx[rank]) * nc + c];
                        assert_eq!(sorted_h[o], want, "value mismatch z={z} c={c} rank={rank}");
                    }
                }
            }
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
