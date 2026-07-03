//! # tomoxide
//!
//! A Rust tomographic reconstruction toolkit — the algorithmic breadth of
//! tomopy (CPU `libtomo`) and the GPU streaming reconstruction of tomocupy
//! (CUDA), behind one tri-backend abstraction (CPU / CUDA / wgpu).
//!
//! This is a single crate; its modules are the foundational data model and
//! backend traits ([`backend`], [`data`], [`dtype`], [`error`], [`geometry`],
//! [`params`]), the backends ([`cpu`], [`cuda`], [`wgpu`]), reconstruction
//! ([`recon`]), preprocessing ([`prep`]), I/O ([`io`]), simulation ([`sim`]),
//! and the high-level [`engine`] + [`pipeline`].
//!
//! ```no_run
//! use tomoxide::{Engine, BackendKind};
//! let engine = Engine::new(BackendKind::Auto)?;       // CPU on a GPU-less box
//! println!("backend: {}", engine.name());
//! # Ok::<(), tomoxide::Error>(())
//! ```
//!
//! See `docs/ARCHITECTURE.md` for the design and `CHANGELOG.md` for status.

// Foundational layer (was tomoxide-core).
pub mod backend;
pub mod data;
pub mod dtype;
pub mod error;
pub mod geometry;
pub mod params;

// Backends.
pub mod cpu;
pub mod cuda;
pub mod wgpu;

// Algorithms, preprocessing, I/O, simulation.
pub mod io;
pub mod prep;
pub mod recon;
pub mod sim;

// High-level orchestration.
pub mod engine;
pub mod pipeline;

pub use engine::Engine;
pub use pipeline::{reconstruct, CancelToken, PrepOptions, ReconSteps};

// Flat re-exports of the common building blocks (was tomoxide-core's root).
pub use backend::{
    Backend, DeviceBuffer, DeviceKind, Elementwise, FbpFilter, Fft, FilteredBackproject,
    ForwardProject, IterativeReconstruct, RankFilter,
};
pub use cpu::CpuBackend;
pub use cuda::CudaBackend;
pub use data::{Dataset, Frames, Layout, Slice2D, Tomo, Volume};
pub use dtype::{Complex32, Dtype, Element};
pub use error::{Error, Result};
pub use geometry::{Angles, Beam, Center, Detector, Geometry};
pub use params::{Algorithm, BackendKind, FilterName, PhaseMethod, ReconParams, StripeMethod};

#[cfg(test)]
mod tests {
    use super::*;

    // Asserts the default (no GPU backend compiled in) Auto→CPU fallback, so it
    // only holds when neither GPU feature is enabled; under `--features cuda`
    // Auto resolves to cuda.
    #[cfg(not(any(feature = "cuda", feature = "gpu-wgpu")))]
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

    // Only meaningful when the cuda feature is absent; with `--features cuda`
    // on a CUDA host the engine is available, which is the correct behaviour.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_engine_unavailable_without_feature() {
        assert!(Engine::new(BackendKind::Cuda).is_err());
    }

    // --- foundational sanity checks (were tomoxide-core's unit tests) ---
    #[test]
    fn angles_uniform_spans_half_turn() {
        let a = Angles::uniform(4, 0.0, std::f32::consts::PI);
        assert_eq!(a.len(), 4);
        assert!((a.0[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn tomo_layout_roundtrip_swaps_axes() {
        let arr = ndarray::Array3::<f32>::zeros((3, 5, 7));
        let t = Tomo::new(arr, Layout::Projection);
        let s = t.to_layout(Layout::Sinogram);
        assert_eq!(s.array.shape(), &[5, 3, 7]);
        assert_eq!(s.n_angles(), 3);
    }

    #[test]
    fn algorithm_parses_and_classifies() {
        assert_eq!("fbp".parse::<Algorithm>().unwrap(), Algorithm::Fbp);
        assert!(Algorithm::Fbp.is_analytic());
        assert!(!Algorithm::Sirt.is_analytic());
    }
}
