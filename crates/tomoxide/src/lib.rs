//! # tomoxide
//!
//! Umbrella crate for the tomoxide tomographic reconstruction toolkit. It
//! re-exports the data model, algorithms, and preprocessing, and owns the
//! [`Engine`] (backend selection) and the high-level [`pipeline`].
//!
//! ```no_run
//! use tomoxide::{Engine, BackendKind};
//! let engine = Engine::new(BackendKind::Auto)?;       // CPU on a GPU-less box
//! println!("backend: {}", engine.name());
//! # Ok::<(), tomoxide::Error>(())
//! ```
//!
//! See `docs/ARCHITECTURE.md` for the design and `docs/ROADMAP.md` for status.
#![forbid(unsafe_code)]

pub mod engine;
pub mod pipeline;

pub use engine::Engine;
pub use pipeline::{reconstruct, PrepOptions, ReconSteps};

// Re-export the building blocks so downstream code needs one dependency.
pub use tomoxide_core::{
    Algorithm, Angles, Backend, BackendKind, Beam, Center, Complex32, Dataset, Detector, Dtype,
    Error, FilterName, Frames, Geometry, Layout, PhaseMethod, ReconParams, Result, StripeMethod,
    Tomo, Volume,
};
pub use tomoxide_core::backend;
pub use tomoxide_cpu::CpuBackend;
pub use tomoxide_cuda::CudaBackend;
pub use tomoxide_io as io;
pub use tomoxide_prep as prep;
pub use tomoxide_recon as recon;
pub use tomoxide_sim as sim;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_engine_falls_back_to_cpu() {
        let e = Engine::new(BackendKind::Auto).unwrap();
        // No CUDA/wgpu compiled in by default → CPU.
        assert_eq!(e.name(), "cpu");
    }

    #[test]
    fn explicit_cpu_engine() {
        let e = Engine::new(BackendKind::Cpu).unwrap();
        assert_eq!(e.name(), "cpu");
        assert!(e.backend().elementwise().is_some());
    }

    #[test]
    fn cuda_engine_unavailable_without_feature() {
        assert!(Engine::new(BackendKind::Cuda).is_err());
    }
}
