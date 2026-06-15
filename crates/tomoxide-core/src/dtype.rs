//! Element scalar types and the runtime [`Dtype`] tag.
//!
//! `f32` is the default everywhere. `f16` (`half::f16`) mirrors tomocupy's
//! `--dtype float16` and is only meaningful on the CUDA/wgpu backends; the CPU
//! backend operates in `f32`.

use num_complex::Complex;

/// Single-precision complex, used by the FFT capability.
pub type Complex32 = Complex<f32>;

/// Runtime tag identifying a scalar element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// 32-bit IEEE float.
    F32,
    /// 16-bit IEEE float (`half::f16`).
    F16,
}

impl Dtype {
    /// Size of one element in bytes.
    pub const fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
        }
    }

    /// Canonical lowercase name (matches tomocupy's `--dtype` values).
    pub const fn as_str(self) -> &'static str {
        match self {
            Dtype::F32 => "float32",
            Dtype::F16 => "float16",
        }
    }
}

/// A scalar element that can live in a device buffer.
///
/// Sealed in spirit: only `f32` and `half::f16` implement it.
pub trait Element: Copy + Send + Sync + 'static {
    /// The runtime tag for this element type.
    const DTYPE: Dtype;
    /// The additive identity.
    fn zero() -> Self;
}

impl Element for f32 {
    const DTYPE: Dtype = Dtype::F32;
    fn zero() -> Self {
        0.0
    }
}

impl Element for half::f16 {
    const DTYPE: Dtype = Dtype::F16;
    fn zero() -> Self {
        half::f16::from_f32_const(0.0)
    }
}
