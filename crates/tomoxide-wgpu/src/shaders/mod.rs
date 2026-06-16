//! WGSL compute shaders (ports of the CUDA kernels), embedded at compile time.
//!
//! Only compiled under the `gpu-wgpu` feature. Bodies are scaffolds to be
//! filled in during milestone M6 (see `docs/ROADMAP.md`).

/// `(data − dark) / (flat − dark)` and `−log`, elementwise.
pub const ELEMENTWISE_WGSL: &str = include_str!("elementwise.wgsl");
/// FBP apodization filter application in the frequency domain.
pub const FBP_FILTER_WGSL: &str = include_str!("fbp_filter.wgsl");
/// Parallel-beam back-projection (sinogram → slice).
pub const BACKPROJECT_WGSL: &str = include_str!("backproject.wgsl");
/// Parallel-beam forward projection (slice → sinogram), the Radon transform.
pub const PROJECT_WGSL: &str = include_str!("project.wgsl");
