//! The tri-backend abstraction.
//!
//! Algorithms in `tomoxide-recon`/`tomoxide-prep` are written against the
//! capability traits below and dispatch through `&dyn Backend`, so the same
//! code runs on CPU, CUDA, or wgpu. A backend implements the subset of
//! capabilities it supports and exposes them through the accessor methods on
//! [`Backend`]; missing ones default to `None`. See `docs/ARCHITECTURE.md` §2.

use crate::data::{Frames, Tomo, Volume};
use crate::dtype::{Complex32, Dtype, Element};
use crate::error::Result;
use crate::geometry::Geometry;
use crate::params::FilterName;

/// Which physical device a backend runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    /// Host CPU.
    Cpu,
    /// NVIDIA CUDA device.
    Cuda,
    /// Portable GPU via wgpu (Metal/Vulkan/DX12).
    Wgpu,
}

/// A buffer of `T` living on a backend's device.
///
/// Host↔device transfers are explicit, mirroring tomocupy's pinned-memory
/// staging. On the CPU backend this wraps a `Vec`/`ndarray`.
pub trait DeviceBuffer<T: Element> {
    /// Number of elements.
    fn len(&self) -> usize;
    /// Whether the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Copy `src` (host) into this buffer.
    fn copy_from_host(&mut self, src: &[T]) -> Result<()>;
    /// Copy this buffer out to `dst` (host).
    fn copy_to_host(&self, dst: &mut [T]) -> Result<()>;
}

/// A backend: a device plus the set of numerical capabilities it provides.
///
/// Capability accessors return `Some(self-as-trait)` when supported. Callers
/// resolve a capability with e.g.
/// `backend.backprojector().ok_or(Error::MissingCapability { .. })?`.
pub trait Backend: Send + Sync {
    /// Short backend name, e.g. `"cpu"`, `"cuda"`, `"wgpu"`.
    fn name(&self) -> &'static str;
    /// The device this backend runs on.
    fn device(&self) -> DeviceKind;
    /// Whether this backend can operate in element type `dt`.
    fn supports(&self, dt: Dtype) -> bool;

    /// FFT capability (cuFFT / rustfft / wgpu).
    fn fft(&self) -> Option<&dyn Fft> {
        None
    }
    /// FBP filter construction + application.
    fn fbp_filter(&self) -> Option<&dyn FbpFilter> {
        None
    }
    /// Filtered/plain back-projection.
    fn backprojector(&self) -> Option<&dyn FilteredBackproject> {
        None
    }
    /// Forward projection (Radon).
    fn projector(&self) -> Option<&dyn ForwardProject> {
        None
    }
    /// Row-action (single-ray) projection for ART/BART.
    fn ray_projector(&self) -> Option<&dyn RayProject> {
        None
    }
    /// Elementwise preprocessing (dark/flat, minus-log, …).
    fn elementwise(&self) -> Option<&dyn Elementwise> {
        None
    }
    /// Rank filters (median, outlier removal).
    fn rank_filter(&self) -> Option<&dyn RankFilter> {
        None
    }
}

/// Batched fast Fourier transforms.
pub trait Fft {
    /// In-place batched 1-D FFT along the last axis.
    ///
    /// `len` is the transform length and `batch` the number of independent
    /// transforms (`buf.len() == len * batch`). `inverse` selects IFFT.
    fn fft_1d(&self, buf: &mut [Complex32], len: usize, batch: usize, inverse: bool) -> Result<()>;

    /// In-place batched 2-D FFT. `buf.len() == rows * cols * batch`.
    fn fft_2d(
        &self,
        buf: &mut [Complex32],
        rows: usize,
        cols: usize,
        batch: usize,
        inverse: bool,
    ) -> Result<()>;
}

/// FBP apodization filter.
pub trait FbpFilter {
    /// Build the frequency-domain filter kernel of length `n`.
    fn make_filter(&self, name: FilterName, n: usize) -> Result<Vec<f32>>;

    /// Apply `filter` to a sinogram in place: the apodized ramp **and** the
    /// rotation-centre correction, folded into one frequency-domain pass
    /// (tomocupy `fbp_filter_center`).
    ///
    /// For each detector lane the kernel is the ramp times a per-row Fourier
    /// phase `exp(-2πi·f_k·(n/2 − center)/pad)` with the *signed* frequency
    /// `f_k` (so the inverse transform stays real) — a band-limited sub-pixel
    /// shift that lands the rotation axis on the detector midpoint `n/2`. After
    /// this pass **every analytic back-projector and Fourier grid reconstructs
    /// against a centre = `n/2` geometry**, so the rotation centre is owned in
    /// this one place: fbp/linerec back-project at `n/2`, and fourierrec/lprec
    /// are centre-agnostic. At the default centre `n/2` the shift is zero and
    /// this is the pure ramp, so the centre-aligned goldens are unchanged.
    /// `geom` supplies the per-row centre.
    ///
    /// Each lane is centred in the `filter.len()`-wide buffer and
    /// **edge-replicate**-padded on both borders before the transform (then the
    /// centred window is cropped back out), matching tomocupy's `ne = 4·n`
    /// padding so the long-tailed ramp does not ring against a hard zero step.
    ///
    /// Out of scope here: gridrec runs its own integrated filter+recenter (it
    /// never calls this method), and the iterative back-projectors keep
    /// `geom.center` in the projector/adjoint pair (their data is not filtered).
    fn apply(&self, sino: &mut Tomo<f32>, filter: &[f32], geom: &Geometry) -> Result<()>;
}

/// Build the full frequency-domain apodized ramp filter for a projection of
/// width `n`.
///
/// Backend-agnostic: the kernel is pure host arithmetic identical on every
/// device, so all [`FbpFilter`] implementations build it through this one
/// function — only [`FbpFilter::apply`] differs by backend. Keeping a single
/// definition here means CPU and GPU cannot drift to different filter shapes.
///
/// The returned kernel has length `pad = (4·n).next_power_of_two()` — the
/// projection is edge-replicate-padded to `pad` and centred before
/// transforming (see [`FbpFilter::apply`]) so the ramp convolution neither
/// wraps around nor rings against a hard zero step — and is laid out in
/// `rustfft` (fftfreq) order, symmetric about the Nyquist bin. The `4·n`
/// width matches tomocupy's `ne` (`fbp_filter_center`: `ne = 4·n` for the
/// float32 path; the `next_power_of_two` rounding here is exact for every
/// power-of-two width — the whole golden set — and matches tomocupy's own
/// float16 pow2-rounding for the rest, keeping the wgpu radix-2 FFT usable at
/// any width). The ramp magnitude `r` runs `0` at DC to `1` at Nyquist;
/// `name` apodizes it. The window set matches tomopy/tomocupy; exact `_wint`
/// quadrature weighting is reconciled when tomopy golden data is available.
pub fn make_fbp_filter(name: FilterName, n: usize) -> Result<Vec<f32>> {
    if n == 0 {
        return Err(crate::error::Error::InvalidParam(
            "filter length must be > 0".into(),
        ));
    }
    let pad = (4 * n).next_power_of_two();
    let pi = std::f32::consts::PI;
    let mut f = vec![0.0f32; pad];
    for (k, slot) in f.iter_mut().enumerate() {
        // |fftfreq| bin index, then r = normalized frequency in [0, 1].
        let fk = if k <= pad / 2 { k } else { pad - k };
        let r = 2.0 * fk as f32 / pad as f32;
        let ramp = r;
        *slot = match name {
            FilterName::None => 1.0, // identity: no apodization, no ramp
            FilterName::Ramp => ramp,
            FilterName::Shepp => {
                let x = pi * r / 2.0;
                if x == 0.0 {
                    ramp
                } else {
                    ramp * (x.sin() / x)
                }
            }
            FilterName::Cosine => ramp * (pi * r / 2.0).cos(),
            FilterName::Cosine2 => {
                let c = (pi * r / 2.0).cos();
                ramp * c * c
            }
            FilterName::Hamming => ramp * (0.54 + 0.46 * (pi * r).cos()),
            FilterName::Hann => ramp * 0.5 * (1.0 + (pi * r).cos()),
            FilterName::Parzen => ramp * (1.0 - r).powi(3),
        };
    }
    Ok(f)
}

/// Back-projection: sinogram → volume.
pub trait FilteredBackproject {
    /// Back-project `sino` into `out` using `geom`. The caller is responsible
    /// for any prior filtering (analytic) or this being one step of an
    /// iterative solver (plain back-projection of a residual).
    fn backproject(&self, sino: &Tomo<f32>, geom: &Geometry, out: &mut Volume<f32>) -> Result<()>;
}

/// Forward projection: volume → sinogram (the Radon transform).
pub trait ForwardProject {
    /// Project `vol` into `out` using `geom`.
    fn project(&self, vol: &Volume<f32>, geom: &Geometry, out: &mut Tomo<f32>) -> Result<()>;
}

/// One row of the forward operator: the pixels a single ray touches and the
/// weight each contributes. This is the sparse `d`-th row of [`ForwardProject`]
/// for one (angle, detector) pair.
#[derive(Clone, Debug, Default)]
pub struct RayRow {
    /// Linear pixel indices into an `n × n` slice (`iy·n + ix`).
    pub pixels: Vec<u32>,
    /// Projection weights, one per entry of `pixels`.
    pub weights: Vec<f32>,
}

/// Row-action (Kaczmarz) projection: the sparse rows of the forward operator.
///
/// The row-action reconstructors (ART, BART) read and update the reconstruction
/// one ray at a time, so they cannot compose the whole-sinogram
/// [`ForwardProject`]/[`FilteredBackproject`]. They instead consume the explicit
/// per-ray rows produced here.
pub trait RayProject {
    /// Build the per-ray rows for an `n × n` grid: `rows[p][d]` lists the pixels
    /// (and weights) that project onto detector column `d` at angle index `p`.
    ///
    /// The rows are geometry-only (reconstruction-independent), so one call is
    /// reused across every iteration. A single rotation center is used (the
    /// `geom` center at row 0), matching tomopy's row-action `art` (which takes
    /// `center[0]` for all slices); per-slice center variation is not modeled.
    fn ray_rows(&self, geom: &Geometry, n: usize) -> Result<Vec<Vec<RayRow>>>;
}

/// Elementwise preprocessing kernels.
pub trait Elementwise {
    /// Dark/flat-field correction: `(data − dark) / (flat − dark)`.
    fn darkflat(&self, data: &mut Tomo<f32>, flat: &Frames<f32>, dark: &Frames<f32>) -> Result<()>;
    /// In-place `−log` with clipping/NaN handling.
    fn minus_log(&self, data: &mut Tomo<f32>) -> Result<()>;
}

/// Rank/order-statistic filters.
pub trait RankFilter {
    /// 3-D median filter with cubic window `size`.
    fn median3d(&self, vol: &mut Volume<f32>, size: usize) -> Result<()>;
    /// Replace outliers exceeding `diff` from the local 3-D-cube median
    /// (dezinger; tomopy `remove_outlier3d`).
    fn remove_outlier3d(&self, data: &mut Tomo<f32>, diff: f32, size: usize) -> Result<()>;
}
