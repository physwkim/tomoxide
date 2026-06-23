//! # tomoxide-cuda
//!
//! The CUDA backend. It re-uses tomocupy's battle-tested `.cu` kernels through
//! a thin C-ABI shim (see `ffi` and `cuda/shim.cpp`) rather than rewriting
//! them. Compiled only when the **`cuda` feature** is enabled and an NVIDIA
//! toolkit is present; otherwise [`CudaBackend::new`] reports the backend as
//! unavailable so the rest of the workspace still builds and runs (on CPU).
//!
//! ## M4 scope
//! The GPU path is the FBP **back-projection** (`cfunc_linerec`, parallel
//! beam): the heavy O(N³) reduction runs on the device, while the FBP **filter**
//! reuses the shared CPU definition ([`tomoxide_cpu::CpuBackend`]'s
//! `FbpFilter`), so `recon(Fbp, &CudaBackend)` filters on the host and
//! back-projects on the GPU. The other `cfunc_*` classes (fourierrec/lprec, and
//! the cufft filter) are scaffolded in the shim history but not wired here.
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
pub struct CudaBackend {
    /// CPU backend used for the (host-side) FBP filter — the shared filter
    /// definition, so the filtered sinogram the GPU back-projects is identical
    /// to the pure-CPU path.
    #[cfg(feature = "cuda")]
    cpu: tomoxide_cpu::CpuBackend,
}

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
            Ok(CudaBackend {
                cpu: tomoxide_cpu::CpuBackend,
            })
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

    /// FBP filter: the shared CPU definition (host-side), so the filtered
    /// sinogram is bit-identical to the pure-CPU path before GPU back-projection.
    #[cfg(feature = "cuda")]
    fn fbp_filter(&self) -> Option<&dyn tomoxide_core::backend::FbpFilter> {
        self.cpu.fbp_filter()
    }

    /// Parallel-beam back-projection on the GPU (`cfunc_linerec`).
    #[cfg(feature = "cuda")]
    fn backprojector(&self) -> Option<&dyn tomoxide_core::backend::FilteredBackproject> {
        Some(self)
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::{ffi, CudaBackend};
    use ndarray::Array3;
    use std::os::raw::c_void;
    use tomoxide_core::backend::FilteredBackproject;
    use tomoxide_core::data::{Layout, Tomo, Volume};
    use tomoxide_core::error::{Error, Result};
    use tomoxide_core::geometry::{Beam, Geometry};

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
