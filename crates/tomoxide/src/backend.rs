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
use crate::error::{Error, Result};
use crate::geometry::{Beam, Geometry};
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
    /// Device-resident iterative reconstruction (bypasses the per-iteration
    /// host↔device round-trips of the generic solvers). `None` ⇒ use the generic
    /// solver composed from [`projector`](Backend::projector) /
    /// [`backprojector`](Backend::backprojector).
    fn iterative_reconstruct(&self) -> Option<&dyn IterativeReconstruct> {
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

/// Invert a square `n×n` f64 matrix by Gauss–Jordan elimination with partial
/// pivoting. Used only to build the small (12×12) inverse-Vandermonde matrix
/// inside [`wint_ramp`]; not a general-purpose linear-algebra routine. The
/// caller guarantees the matrix is invertible (a Vandermonde of distinct
/// nodes), so the pivot is always nonzero.
fn invert_matrix(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    // Augmented [A | I].
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            row
        })
        .collect();
    for col in 0..n {
        // Partial pivot: largest magnitude at/below the diagonal in this column.
        let mut piv = col;
        for r in (col + 1)..n {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        m.swap(col, piv);
        let inv_d = 1.0 / m[col][col];
        for v in m[col].iter_mut() {
            *v *= inv_d;
        }
        let pivot_row = m[col].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r != col {
                let factor = row[col];
                if factor != 0.0 {
                    for (slot, &pv) in row.iter_mut().zip(pivot_row.iter()) {
                        *slot -= factor * pv;
                    }
                }
            }
        }
    }
    m.into_iter().map(|row| row[n..].to_vec()).collect()
}

/// Quadrature ramp weights — a port of tomocupy
/// `fbp_filter.FBPFilter._wint` (`order = 12`). Returns one weight per
/// frequency sample `t[k]` (`t[k] = k/pad`, `k = 0..pad/2`).
///
/// The weight approximates the ideal ramp `t/pad` but with the degree-`order`
/// Newton–Cotes integration shape: a high-order interpolatory quadrature on the
/// nodes `s = linspace(1e-40, 1, order)` (built via the inverse Vandermonde
/// `iv` and the per-monomial interval integrals `u`), accumulated over
/// overlapping `order`-point windows of `t` with the overlap-compensation
/// weights `p`, plus tomocupy's 40-sample linear endpoint correction. Versus a
/// plain linear ramp this deviates ≈1% near DC/Nyquist; matching it is what
/// closes the residual scale between tomoxide's analytic CUDA output and
/// tomocupy beyond the leading filter normalization. `iv`, `u`, `W1`, `W2`, `p`
/// depend only on `order`; only the window loop reads `t`.
fn wint_ramp(order: usize, t: &[f64]) -> Vec<f64> {
    let n = order;
    let big_n = t.len();
    debug_assert!(big_n >= n, "wint_ramp needs at least `order` samples");
    // Nodes and `V[i][j] = s[j]^i = exp(i·ln s[j])` (matches tomocupy's
    // `exp(outer(arange(n), log(s)))`, including the `s[0]=1e-40` underflow).
    let s: Vec<f64> = (0..n)
        .map(|i| 1e-40 + (1.0 - 1e-40) * i as f64 / (n as f64 - 1.0))
        .collect();
    let logs: Vec<f64> = s.iter().map(|&x| x.ln()).collect();
    let v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| (i as f64 * logs[j]).exp()).collect())
        .collect();
    let iv = invert_matrix(&v);
    // `u[i][j] = ∫_{s[j]}^{s[j+1]} x^i dx`, i.e. `s^{i+1}/(i+1)` differenced over
    // `j`, for `i = 0..=n` (n+1 rows) and `j = 0..n-1` (n-1 cols).
    let anti = |i: usize, j: usize| ((i as f64 + 1.0) * logs[j]).exp() / (i as f64 + 1.0);
    let u: Vec<Vec<f64>> = (0..=n)
        .map(|i| (0..n - 1).map(|j| anti(i, j + 1) - anti(i, j)).collect())
        .collect();
    // `W1 = iv · u[1..=n]` (the `x·pₙ(x)` term), `W2 = iv · u[0..n]` (the
    // `const·pₙ(x)` term), both `n × (n-1)`.
    let mut w1 = vec![vec![0.0f64; n - 1]; n];
    let mut w2 = vec![vec![0.0f64; n - 1]; n];
    for r in 0..n {
        for c in 0..n - 1 {
            let (mut a1, mut a2) = (0.0, 0.0);
            for k in 0..n {
                a1 += iv[r][k] * u[k + 1][c];
                a2 += iv[r][k] * u[k][c];
            }
            w1[r][c] = a1;
            w2[r][c] = a2;
        }
    }
    // Overlap compensation `p` (length `big_n-1`): `1/1 .. 1/(n-1)` rising, a
    // flat `1/(n-1)` middle, then `1/(n-1) .. 1/1` falling.
    let mut p = Vec::with_capacity(big_n - 1);
    for i in 1..n {
        p.push(1.0 / i as f64);
    }
    let mid = big_n as isize - 2 * (n as isize - 1) - 1;
    for _ in 0..mid.max(0) {
        p.push(1.0 / (n as f64 - 1.0));
    }
    for i in (1..n).rev() {
        p.push(1.0 / i as f64);
    }
    // Windowed quadrature accumulation. Window `j` maps `[t[j], t[j+n-1]]` onto
    // the node domain: `∫ (t[j] + Δ·x)·pₙ = Δ²·W1 + Δ·t[j]·W2`, `Δ = t[j+n-1]-t[j]`.
    let mut w = vec![0.0f64; big_n];
    for j in 0..=(big_n - n) {
        let dt = t[j + n - 1] - t[j];
        for k in 0..n - 1 {
            let pjk = p[j + k];
            for row in 0..n {
                w[j + row] += pjk * (dt * dt * w1[row][k] + dt * t[j] * w2[row][k]);
            }
        }
    }
    // tomocupy endpoint fix: replace the last 40 samples with a line through the
    // anchor `w[big_n-40]` (`wn[-40:] = w[-40]/(N-40)·arange(N-40, N)`).
    if big_n >= 40 {
        let anchor = big_n - 40;
        let base = w[anchor] / anchor as f64;
        for (kk, slot) in w.iter_mut().enumerate().skip(anchor) {
            *slot = base * kk as f64;
        }
    }
    w
}

/// Which base ramp shape [`make_fbp_filter`] builds — the one knob on which the
/// CPU/wgpu and CUDA backends deliberately diverge, so each matches its own
/// reference library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RampShape {
    /// tomopy's classic straight-line ramp `r = 2·fk/pad`. The CPU and wgpu
    /// backends port tomopy, so they build this shape.
    Linear,
    /// tomocupy's degree-12 `_wint` quadrature ramp ([`wint_ramp`]). The CUDA
    /// backend ports tomocupy, so it builds this shape; combined with the
    /// `0.5/pad` gain in `build_filter_w` it reproduces tomocupy's net FBP
    /// filter bit-for-bit.
    Wint,
}

/// Build the full frequency-domain apodized ramp filter for a projection of
/// width `n`.
///
/// Backend-agnostic apodization and layout: the windowing, padding, clamp and
/// symmetric FFT layout are pure host arithmetic identical on every device, so
/// all [`FbpFilter`] implementations build the filter through this one function
/// — only [`FbpFilter::apply`] differs by backend. The single exception is the
/// base ramp *shape* ([`RampShape`]): the CPU/wgpu backends pass
/// [`RampShape::Linear`] (tomopy), the CUDA backend [`RampShape::Wint`]
/// (tomocupy), because the two reference libraries genuinely differ on the ramp
/// and each tomoxide backend ports a different one. Keeping the apodization,
/// padding and layout in a single definition means the backends cannot drift on
/// anything *except* that one deliberate, documented ramp-shape axis.
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
/// any width).
///
/// The base ramp is the physical inversion filter `|ω|` in cycles/pixel, so it
/// runs `0` at DC to `0.5` at Nyquist and `name` apodizes it.
/// [`RampShape::Linear`] is the plain `k/pad` line (tomopy's ramp shape);
/// [`RampShape::Wint`] is `pad·wint(t)` ([`wint_ramp`], tomocupy's degree-12
/// quadrature ramp). Both tomopy and tomocupy carry an extra factor `2` here
/// (their ramp peaks at `1` at Nyquist), which puts every analytic
/// reconstruction at `2×` the physical attenuation μ; tomoxide drops it so the
/// analytic output is μ per pixel-unit — the same scale the iterative solvers
/// and `recon::lamino`'s own `|f|/ne` ramp converge to. Either way tomocupy's
/// post-processing is mirrored: the windowed ramp is clamped to `≥0` and the DC
/// bin is doubled (a no-op here since the ramp is `0` at DC). A grid too short
/// for the order-12 rule falls back to the linear ramp even when `Wint` is
/// requested (tomocupy's `_wint` is itself undefined there). The window set
/// matches tomopy/tomocupy.
pub fn make_fbp_filter(name: FilterName, n: usize, shape: RampShape) -> Result<Vec<f32>> {
    if n == 0 {
        return Err(crate::error::Error::InvalidParam(
            "filter length must be > 0".into(),
        ));
    }
    let pad = (4 * n).next_power_of_two();
    let nhalf = pad / 2 + 1;
    const ORDER: usize = 12;
    // The order-12 overlap quadrature needs at least `2·(ORDER−1)+1` samples for
    // a non-degenerate window structure (its flat middle block has length
    // `nhalf − 2·(ORDER−1) − 1`, which must be ≥ 0); this is exactly the regime
    // where tomocupy's `_wint` is defined (it raises on a negative dimension
    // below it). Real recons (`nhalf ≫ 23`) always take this path; only tiny
    // test/edge grids fall back to the plain linear ramp.
    const WINT_MIN: usize = 2 * ORDER - 1;
    // Half-spectrum base ramp: the physical `|ω|` inversion filter in
    // cycles/pixel (0 at DC, 0.5 at Nyquist). tomocupy's degree-12 `_wint`
    // quadrature ramp (`pad·wint(t) ≈ t`) for `Wint`, else the plain linear ramp
    // `k/pad` — also the fallback when the grid is too short for the order-12
    // rule. Both upstreams double this (peak 1 at Nyquist), putting analytic
    // output at 2×μ; we drop the factor 2 to land on the physical μ. `None`
    // carries no ramp, so skip.
    let ramp_half: Vec<f32> = if name == FilterName::None {
        Vec::new()
    } else if shape == RampShape::Wint && nhalf >= WINT_MIN {
        let t: Vec<f64> = (0..nhalf).map(|k| k as f64 / pad as f64).collect();
        wint_ramp(ORDER, &t)
            .iter()
            .map(|&wk| (pad as f64 * wk) as f32)
            .collect()
    } else {
        (0..nhalf).map(|k| k as f32 / pad as f32).collect()
    };
    let mut f = vec![0.0f32; pad];
    for (k, slot) in f.iter_mut().enumerate() {
        // |fftfreq| bin index, then r = normalized frequency in [0, 1].
        let fk = if k <= pad / 2 { k } else { pad - k };
        let r = 2.0 * fk as f32 / pad as f32;
        if name == FilterName::None {
            *slot = 1.0; // identity: no apodization, no ramp
            continue;
        }
        let ramp = ramp_half[fk];
        // tomocupy doubles the DC bin (the ≥0 clamp lives in `apodize`).
        let mut v = apodize(name, ramp, r);
        if k == 0 {
            v *= 2.0;
        }
        *slot = v;
    }
    Ok(f)
}

/// Apply the `name` apodisation window to a base ramp value at normalised
/// frequency `r ∈ [0, 1]` (`1` = Nyquist), clamped to ≥ 0. This is the window
/// *shape* only — the caller supplies the base `ramp` and handles the DC bin and
/// the `None` = identity case. Shared by [`make_fbp_filter`] and the
/// laminography ramp filter ([`lam_ramp_weights`]) so the window definitions
/// have a single source of truth.
pub(crate) fn apodize(name: FilterName, ramp: f32, r: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let v = match name {
        // No window: plain ramp. (`make_fbp_filter` treats `None` as full
        // identity — no ramp either — before it ever calls here.)
        FilterName::None | FilterName::Ramp => ramp,
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
    if v < 0.0 {
        0.0
    } else {
        v
    }
}

/// Per-frequency real multiplier for the laminography projection ramp filter,
/// length `ne` (the edge-padded FFT length `2·detw`). Bin `k` maps to the
/// centred frequency `f = min(k, ne−k)`; the base is the plain linear ramp
/// `f/ne` (tomoxide's lamino ramp convention — no wint quadrature), apodised by
/// `name`. `Ramp`/`None` leave the plain ramp unchanged (the prior behaviour);
/// `Parzen` (the default) and the others match tomocupy's `LamFourierRec`, which
/// filters projections with the selected FBP filter. Applied on the FFT'd,
/// edge-padded projection lines in place of the old hard-coded `|f|/ne` ramp.
pub(crate) fn lam_ramp_weights(ne: usize, name: FilterName) -> Vec<f32> {
    (0..ne)
        .map(|k| {
            let fk = if k <= ne / 2 { k } else { ne - k };
            let ramp = fk as f32 / ne as f32;
            let r = 2.0 * fk as f32 / ne as f32;
            apodize(name, ramp, r)
        })
        .collect()
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

/// Parallel-beam row-action rows for an `n × n` grid — the shared,
/// backend-independent geometry behind every [`RayProject`] implementation.
///
/// Transposes the pixel-driven splat of the forward projector into per-detector
/// rows: each pixel `(iy, ix)` projects to
/// `t = (ix − n/2)·cosθ + (iy − n/2)·sinθ + center` and splits linearly between
/// detector columns `⌊t⌋` and `⌊t⌋+1`, so `rows[p][⌊t⌋]` gains weight `1 − frac`
/// and `rows[p][⌊t⌋+1]` gains `frac`. These are exactly the rows of the same
/// operator `R` the simultaneous methods use. A single `center` (row 0) is used
/// for all slices, matching tomopy `art`.
///
/// The computation is pure host geometry (no device kernel — the row-action
/// solvers ART/BART are sequential Kaczmarz updates), so CPU and CUDA share this
/// one definition and produce byte-identical rows.
pub(crate) fn parallel_ray_rows(geom: &Geometry, n: usize) -> Result<Vec<Vec<RayRow>>> {
    if geom.beam != Beam::Parallel {
        return Err(Error::InvalidParam(
            "row-action projection currently supports parallel beam only".into(),
        ));
    }
    let nang = geom.angles.0.len();
    let ncols = geom.detector.width;
    if nang == 0 || ncols == 0 {
        return Err(Error::InvalidParam(
            "geometry has no angles or zero detector width".into(),
        ));
    }
    let center = geom.center.at(0);
    let c0 = n as f32 / 2.0; // cx == cy == n/2
    let mut rows: Vec<Vec<RayRow>> = (0..nang).map(|_| vec![RayRow::default(); ncols]).collect();
    for (ia, &a) in geom.angles.0.iter().enumerate() {
        let (sn, cs) = a.sin_cos();
        let arows = &mut rows[ia];
        for iy in 0..n {
            let gy = iy as f32 - c0;
            for ix in 0..n {
                let gx = ix as f32 - c0;
                let t = gx * cs + gy * sn + center;
                let t0 = t.floor();
                let i0 = t0 as isize;
                if i0 >= 0 && (i0 as usize) + 1 < ncols {
                    let frac = t - t0;
                    let pix = (iy * n + ix) as u32;
                    let d0 = i0 as usize;
                    arows[d0].pixels.push(pix);
                    arows[d0].weights.push(1.0 - frac);
                    arows[d0 + 1].pixels.push(pix);
                    arows[d0 + 1].weights.push(frac);
                }
            }
        }
    }
    Ok(rows)
}

/// Device-resident iterative reconstruction.
///
/// The generic host solvers in [`crate::recon`] compose [`ForwardProject`] and
/// [`FilteredBackproject`], each of which round-trips the whole volume/sinogram
/// host↔device **every iteration**. A backend that can keep both resident on the
/// device across all iterations — uploading once, iterating on-device, and
/// downloading once — implements this to bypass that per-iteration transfer.
pub trait IterativeReconstruct {
    /// Run `algorithm` device-resident and return the reconstructed volume, or
    /// `Ok(None)` if this backend has no device-resident path for `algorithm`
    /// (the caller then falls back to the generic host solver). This mirrors
    /// [`AnalyticReconstruct::streaming`]'s opt-in-or-fall-back contract.
    fn solve(
        &self,
        sino: &Tomo<f32>,
        geom: &Geometry,
        algorithm: crate::params::Algorithm,
        params: &crate::params::ReconParams,
    ) -> Result<Option<Volume<f32>>>;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `wint_ramp(12, ·)` reproduces tomocupy `FBPFilter._wint(12, ·)`.
    ///
    /// Golden values dumped from tomocupy (cupy float64) for `ne = 512`,
    /// `t = arange(0, ne/2+1)/ne`. Indices 0..=13 probe the sensitive low-`k`
    /// quadrature start, 64/128/200 the linear bulk, 216 the last quadrature
    /// sample, and 217/256 the 40-sample linear endpoint correction.
    #[test]
    fn wint_ramp_matches_tomocupy_golden() {
        let ne = 512usize;
        let t: Vec<f64> = (0..ne / 2 + 1).map(|k| k as f64 / ne as f64).collect();
        let w = wint_ramp(12, &t);
        let golden: &[(usize, f64)] = &[
            (0, 2.453569604067e-07),
            (1, 4.042956064990e-06),
            (2, 7.660118337826e-06),
            (3, 9.761576171718e-06),
            (4, 2.031827452663e-05),
            (5, 1.049037500880e-05),
            (6, 3.259454732421e-05),
            (7, 1.901878617086e-05),
            (8, 3.484518329315e-05),
            (9, 3.256638900053e-05),
            (10, 3.873577972859e-05),
            (11, 4.168480391830e-05),
            (12, 4.608636383489e-05),
            (13, 4.912083786810e-05),
            (64, 2.441406251635e-04),
            (128, 4.882812503246e-04),
            (200, 7.629394536308e-04),
            (216, 8.239746099211e-04),
            (217, 8.277893071893e-04),
            (256, 9.765625006472e-04),
        ];
        for &(i, g) in golden {
            let rel = (w[i] - g).abs() / g.abs();
            assert!(
                rel < 1e-5,
                "wint[{i}] = {:.12e}, golden {:.12e}, rel {:.2e}",
                w[i],
                g,
                rel
            );
        }
    }
}
