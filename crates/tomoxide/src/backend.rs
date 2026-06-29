//! The tri-backend abstraction.
//!
//! Algorithms in `tomoxide-recon`/`tomoxide-prep` are written against the
//! capability traits below and dispatch through `&dyn Backend`, so the same
//! code runs on CPU, CUDA, or wgpu. A backend implements the subset of
//! capabilities it supports and exposes them through the accessor methods on
//! [`Backend`]; missing ones default to `None`. See `docs/ARCHITECTURE.md` §2.

use ndarray::{Array3, ArrayViewMut2, Axis};

use crate::data::{Frames, Tomo, Volume};
use crate::dtype::{Complex32, Dtype, Element};
use crate::error::Result;
use crate::geometry::Geometry;
use crate::params::{FilterName, StripeMethod};

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

/// A page-locked-capable host buffer of `f32` that the out-of-core reader fills
/// (directly, via `rust-hdf5`'s `read_slice_into`) and a backend uploads.
///
/// The CUDA backend returns one backed by `cudaHostAlloc` (pinned) memory so the
/// subsequent H2D is a pure DMA with no driver staging copy — eliminating the
/// per-chunk host-bandwidth cost that otherwise competes with the reader. Other
/// backends fall back to a plain `Vec`. `Send` so the filled buffer can cross the
/// pipeline's reader→compute channel; the bytes are plain host memory, safe to
/// read and free from any thread.
pub trait HostBuffer: Send {
    /// The whole buffer, for the reader to fill.
    fn as_mut_slice(&mut self) -> &mut [f32];
    /// The filled buffer, for the H2D upload (or a host-path view).
    fn as_slice(&self) -> &[f32];
}

/// Plain heap-`Vec` [`HostBuffer`] — the default for backends without pinned
/// memory (CPU/wgpu) and the CUDA fallback when `cudaHostAlloc` fails.
pub(crate) struct VecHostBuffer(Vec<f32>);

impl VecHostBuffer {
    /// A zeroed buffer of `len` `f32`.
    pub(crate) fn new(len: usize) -> Self {
        VecHostBuffer(vec![0.0f32; len])
    }
}

impl HostBuffer for VecHostBuffer {
    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.0
    }
    fn as_slice(&self) -> &[f32] {
        &self.0
    }
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
    /// Monolithic Fourier-gridding reconstruction (e.g. CUDA `cfunc_fourierrec`)
    /// for backends that don't expose the generic [`Fft`] capability the
    /// CPU/wgpu `fourierrec` composes from.
    fn fourier_reconstruct(&self) -> Option<&dyn FourierReconstruct> {
        None
    }
    /// Fused analytic reconstruction kept resident on the device: filter →
    /// back-projection (or Fourier gridding) without intermediate host copies
    /// (CUDA). When present, [`crate`]'s analytic dispatch routes the whole
    /// `Fbp`/`Linerec`/`Fourierrec` chain here (one upload, one download)
    /// instead of composing the host-roundtripping per-capability stages.
    fn analytic_reconstruct(&self) -> Option<&dyn AnalyticReconstruct> {
        None
    }
    /// Monolithic log-polar (lprec) reconstruction with the gather/scatter +
    /// spline prefilter resident on the device (CUDA `cuda/lprec.cu`), for
    /// backends that would otherwise run lprec's cubic interpolation on the host
    /// through the generic [`Fft`] capability. When absent, lprec composes from
    /// [`Fft`] (CPU/wgpu).
    fn lprec_reconstruct(&self) -> Option<&dyn LpRecReconstruct> {
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

    /// Allocate a host staging buffer of `len` `f32` for an out-of-core read
    /// chunk's projections. The CUDA backend overrides this to return page-locked
    /// (pinned) memory so the chunk's H2D is a direct DMA; the default is a plain
    /// `Vec`. Infallible — a backend that cannot pin (allocation failure) falls
    /// back to a `Vec` so reading always proceeds.
    fn alloc_host_buffer(&self, len: usize) -> Box<dyn HostBuffer> {
        Box::new(VecHostBuffer::new(len))
    }

    /// Whether the streaming pipeline should hand this backend **raw** projection
    /// chunks read straight into an [`alloc_host_buffer`](Backend::alloc_host_buffer)
    /// staging buffer (for [`reconstruct_chunk_raw`](StreamingAnalytic::reconstruct_chunk_raw)'s
    /// device-resident upload), instead of a host-assembled [`Tomo`]. True only
    /// for backends with a device-resident raw path that benefits from pinned
    /// reads (CUDA); the default `false` keeps CPU/wgpu on the owned-array path
    /// (no per-chunk copy into a staging buffer).
    fn wants_raw_chunks(&self) -> bool {
        false
    }
}

/// Batched fast Fourier transforms.
///
/// `Send + Sync` so a shared `&dyn Fft` can be handed to concurrent host
/// workers (rayon, device-pinned pools); see [`for_each_slice`](Fft::for_each_slice)
/// for how a backend chooses to drive the per-slice reconstruction loop.
pub trait Fft: Send + Sync {
    /// Drive a reconstructor's per-slice loop, writing each output slice
    /// `out[[row, .., ..]]` via `f(row, slab)`. The slices are independent (no
    /// shared mutable state, no float reassociation), so the *backend* owns the
    /// execution strategy and every strategy yields a bit-identical volume:
    ///
    /// - default: serial — correct for device FFTs that must be driven from one
    ///   host thread (e.g. wgpu, single-stream),
    /// - [`CpuBackend`](crate::cpu::CpuBackend): rayon across host cores,
    /// - [`CudaBackend`](crate::cuda::CudaBackend): device-pinned rayon pools
    ///   that fan slices across the selected GPUs *and* host cores.
    ///
    /// Reconstructors (gridrec / fourierrec / lprec / phase) call this instead
    /// of looping themselves, so multi-core and multi-GPU scheduling lives in
    /// one place per backend rather than being special-cased at each call site.
    fn for_each_slice(
        &self,
        out: &mut Array3<f32>,
        f: &(dyn Fn(usize, ArrayViewMut2<f32>) -> Result<()> + Sync),
    ) -> Result<()> {
        for (row, slab) in out.axis_iter_mut(Axis(0)).enumerate() {
            f(row, slab)?;
        }
        Ok(())
    }

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

/// Fused, device-resident analytic reconstruction (raw sinogram → volume):
/// the backend applies the FBP filter and the back-projection / Fourier
/// gridding itself, keeping all intermediates on the device. Lets the analytic
/// dispatcher avoid the per-capability host round-trips when a backend (CUDA)
/// can fuse the chain. Must support at least `Fbp`/`Linerec` and `Fourierrec`.
pub trait AnalyticReconstruct {
    /// Reconstruct from the **unfiltered** sinogram; the backend computes the
    /// FBP filter (`params.filter_name`) and applies it internally.
    fn reconstruct(
        &self,
        sino: &Tomo<f32>,
        geom: &Geometry,
        algorithm: crate::params::Algorithm,
        params: &crate::params::ReconParams,
    ) -> Result<Volume<f32>>;

    /// Build a [`StreamingAnalytic`] bound to fixed `(algorithm, params, ncols,
    /// max_nz)` so a multi-chunk streaming job creates the FBP-filter / back-
    /// projection handles (cuFFT plans, f16 textures) **once** and reuses them
    /// across z-chunks, instead of the per-chunk new/free [`reconstruct`] does.
    ///
    /// `geom` supplies the (chunk-invariant) projection angles; per-chunk centre
    /// shifts are taken from the geometry handed to
    /// [`StreamingAnalytic::reconstruct_chunk`]. Returns `Ok(None)` when this
    /// backend cannot reuse handles for `algorithm` (e.g. the CPU backend, or
    /// gridrec/lprec) — the caller falls back to per-chunk [`reconstruct`].
    fn streaming(
        &self,
        _algorithm: crate::params::Algorithm,
        _params: &crate::params::ReconParams,
        _geom: &Geometry,
        _ncols: usize,
        _max_nz: usize,
    ) -> Result<Option<Box<dyn StreamingAnalytic>>> {
        Ok(None)
    }
}

/// A reusable analytic reconstructor bound to fixed dims (see
/// [`AnalyticReconstruct::streaming`]). Holds the device-resident FBP-filter and
/// back-projection handles for its whole lifetime so the streaming driver pays
/// the cuFFT-plan / f16-texture-array setup once per run rather than per chunk —
/// matching tomocupy's create-once `BackprojFunctions`. Single-threaded by
/// construction: it is created and driven on the streaming compute thread.
pub trait StreamingAnalytic {
    /// Reconstruct one z-chunk's volume `[nz, n, n]` from the **unfiltered**
    /// sinogram `[nz, nproj, ncols]`. `nz` may be ≤ the `max_nz` the
    /// reconstructor was built with (a smaller trailing chunk reuses the same
    /// handles, zero-padded to `max_nz`); `nz > max_nz` is an error.
    fn reconstruct_chunk(&mut self, sino: &Tomo<f32>, geom: &Geometry) -> Result<Volume<f32>>;

    /// Device-resident fast path: reconstruct one z-chunk directly from the
    /// **raw, un-normalized** projection chunk `[nproj, nz, ncols]` (plus
    /// optional `flat`/`dark` frames), performing dark/flat correction,
    /// minus-log, and the projection→sinogram transpose **on the device** so the
    /// chunk crosses PCIe exactly once up and the volume once down — no host
    /// normalize round-trip and no host transpose copy.
    ///
    /// `stripe` requests on-device stripe removal applied to the transposed f32
    /// sinogram (before any f16 cast). The reconstructor returns `Ok(None)` —
    /// deferring the *whole* chunk to the host path — both when it has no
    /// device-resident path at all and when it has one but cannot run the
    /// requested `stripe` method on the device. `StripeMethod::None` is always
    /// handled (no-op). This keeps the device/host gating decision inside the
    /// backend: the caller simply falls back to host normalize + transpose +
    /// host `remove_stripe` + [`reconstruct_chunk`] whenever this returns `None`.
    ///
    /// `data` is the contiguous, C-order raw projection chunk in
    /// [`Layout::Projection`](crate::data::Layout) — `[nproj, nz, ncols]` with
    /// `dims == (nproj, nz, ncols)`. Taking a slice (not a [`Tomo`]) lets the
    /// caller hand the pinned read buffer straight through with no intervening
    /// owned `ndarray` allocation (`ndarray` cannot own page-locked memory).
    fn reconstruct_chunk_raw(
        &mut self,
        _data: &[f32],
        _dims: (usize, usize, usize),
        _flat: Option<&Frames<f32>>,
        _dark: Option<&Frames<f32>>,
        _geom: &Geometry,
        _stripe: StripeMethod,
    ) -> Result<Option<Volume<f32>>> {
        Ok(None)
    }

    /// Hand a spent host volume buffer back to the reconstructor for reuse on a
    /// later chunk. The streaming pipeline calls this with the `Vec` backing each
    /// volume after the writer is done with it, so a reconstructor that copies
    /// device output into an owned `Send` buffer (the CUDA path) can recycle a
    /// warm allocation instead of paying ~190 ms of first-touch page-faults per
    /// 536 MB chunk. The default ignores the buffer (backends that build the
    /// volume on the host have nothing to recycle).
    fn give_reuse_buffer(&mut self, _buf: Vec<f32>) {}
}

/// Direct Fourier-gridding reconstruction (sinogram → volume) for a backend
/// with a monolithic Fourier method that doesn't decompose into the [`Fft`]
/// capability (CUDA `cfunc_fourierrec`). The caller applies the FBP filter
/// first, so the input is the **filtered** sinogram.
pub trait FourierReconstruct {
    /// Reconstruct an `[nz, n, n]` volume from a filtered sinogram
    /// `[nz, nproj, ncols]`.
    fn reconstruct(&self, filtered: &Tomo<f32>, geom: &Geometry, n: usize) -> Result<Volume<f32>>;
}

/// Monolithic log-polar (lprec) reconstruction from a **filtered** sinogram.
pub trait LpRecReconstruct {
    /// Reconstruct an `[nz, n, n]` volume from a filtered sinogram
    /// `[nz, nproj, ncols]` via the log-polar method.
    fn reconstruct(&self, filtered: &Tomo<f32>, geom: &Geometry, n: usize) -> Result<Volume<f32>>;
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
