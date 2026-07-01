//! WGSL compute shaders (ports of the CUDA kernels), embedded at compile time.
//!
//! Only compiled under the `gpu-wgpu` feature.

/// `(data − dark) / (flat − dark)` and `−log`, elementwise.
pub const ELEMENTWISE_WGSL: &str = include_str!("elementwise.wgsl");
/// FBP apodization filter application in the frequency domain.
pub const FBP_FILTER_WGSL: &str = include_str!("fbp_filter.wgsl");
/// Parallel-beam back-projection (sinogram → slice).
pub const BACKPROJECT_WGSL: &str = include_str!("backproject.wgsl");
/// Parallel-beam forward projection (slice → sinogram), the Radon transform.
pub const PROJECT_WGSL: &str = include_str!("project.wgsl");
/// 3-D median / dezinger rank filter (clamp-to-center windowed order statistic).
pub const MEDFILT3D_WGSL: &str = include_str!("medfilt3d.wgsl");
/// Batched radix-2 FFT (bit-reversal permute + per-stage butterflies).
pub const FFT_WGSL: &str = include_str!("fft.wgsl");
/// Per-image complex transpose (column pass of the 2-D FFT as a row pass).
pub const FFT_TRANSPOSE_WGSL: &str = include_str!("fft_transpose.wgsl");
/// Bluestein chirp-z spectral multiply (arbitrary-length FFT via radix-2).
pub const BLUESTEIN_WGSL: &str = include_str!("bluestein.wgsl");
