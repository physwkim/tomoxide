//! # tomoxide-cuda
//!
//! The CUDA backend. It re-uses tomocupy's battle-tested `.cu` kernels through
//! a thin C-ABI shim (see `ffi` and `cuda/shim.cpp`) rather than rewriting
//! them. Compiled only when the **`cuda` feature** is enabled and an NVIDIA
//! toolkit is present; otherwise [`CudaBackend::new`] reports the backend as
//! unavailable so the rest of the workspace still builds and runs (on CPU).
//!
//! In this scaffold the backend advertises the device but exposes no
//! capabilities yet — the FFI signatures are committed in `ffi` and the
//! streaming wiring lands in milestone M4 (see `docs/ROADMAP.md`).
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
    /// [`Error::BackendUnavailable`]; the binary was compiled without CUDA
    /// support. With the feature, a real implementation will probe for a
    /// device here (milestone M4).
    pub fn new() -> Result<Self> {
        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::BackendUnavailable(
                "compiled without the `cuda` feature".into(),
            ))
        }
        #[cfg(feature = "cuda")]
        {
            // TODO(M4): cudaGetDeviceCount > 0 check via the shim.
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
    // Capability accessors stay at their `None` defaults until M4 wires the
    // streaming buffers/streams to the FFI in `ffi.rs`.
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
