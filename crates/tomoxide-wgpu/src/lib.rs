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

    /// Initialise a wgpu device, picking a high-performance adapter.
    ///
    /// Wiring the adapter/device request (`Instance::request_adapter` →
    /// `Adapter::request_device`) and the WGSL pipelines lands in milestone M6;
    /// the exact wgpu call shape is pinned to the resolved `wgpu` version then.
    #[cfg(feature = "gpu-wgpu")]
    pub fn new() -> Result<Self> {
        Err(Error::BackendUnavailable(
            "wgpu device init not yet wired (milestone M6)".into(),
        ))
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
