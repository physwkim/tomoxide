//! # tomoxide-cpu
//!
//! The CPU backend — pure Rust (`ndarray` + `rayon`). It is the **parity
//! target** for tomopy's `libtomo`: where a kernel is ported, the CPU result
//! is the reference the GPU backends are diffed against.
//!
//! In this scaffold a few elementwise/preprocessing ops are real; the heavy
//! kernels (FFT, back-projection, forward projection) return
//! [`Error::NotImplemented`] with the upstream `file:line` to port from.
#![forbid(unsafe_code)]

use crate::backend::{
    Backend, DeviceBuffer, DeviceKind, Elementwise, FbpFilter, Fft, FilteredBackproject,
    ForwardProject, RampShape, RankFilter, RayProject, RayRow,
};
use crate::data::{Frames, Layout, Tomo, Volume};
use crate::dtype::{Complex32, Dtype, Element};
use crate::error::{Error, Result};
use crate::geometry::{Beam, Center, Geometry};
use crate::params::FilterName;
use ndarray::{Array3, Axis};
use rayon::prelude::*;
use rustfft::FftPlanner;

/// A host-resident buffer. On the CPU backend "device memory" is just a `Vec`.
#[derive(Clone, Debug, Default)]
pub struct CpuBuffer<T> {
    /// The backing storage.
    pub data: Vec<T>,
}

impl<T: Element> CpuBuffer<T> {
    /// Allocate `len` zeroed elements.
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![T::zero(); len],
        }
    }
}

impl<T: Element> DeviceBuffer<T> for CpuBuffer<T> {
    fn len(&self) -> usize {
        self.data.len()
    }
    fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.data.len() {
            return Err(Error::ShapeMismatch {
                expected: self.data.len().to_string(),
                found: src.len().to_string(),
            });
        }
        self.data.copy_from_slice(src);
        Ok(())
    }
    fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.data.len() {
            return Err(Error::ShapeMismatch {
                expected: self.data.len().to_string(),
                found: dst.len().to_string(),
            });
        }
        dst.copy_from_slice(&self.data);
        Ok(())
    }
}

/// The CPU backend handle.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Create the CPU backend (always available).
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }
    fn device(&self) -> DeviceKind {
        DeviceKind::Cpu
    }
    fn supports(&self, dt: Dtype) -> bool {
        // The CPU path is f32-only; f16 is a GPU concern (tomocupy `*fp16`).
        dt == Dtype::F32
    }
    fn fft(&self) -> Option<&dyn Fft> {
        Some(self)
    }
    fn fbp_filter(&self) -> Option<&dyn FbpFilter> {
        Some(self)
    }
    fn backprojector(&self) -> Option<&dyn FilteredBackproject> {
        Some(self)
    }
    fn projector(&self) -> Option<&dyn ForwardProject> {
        Some(self)
    }
    fn ray_projector(&self) -> Option<&dyn RayProject> {
        Some(self)
    }
    fn elementwise(&self) -> Option<&dyn Elementwise> {
        Some(self)
    }
    fn rank_filter(&self) -> Option<&dyn RankFilter> {
        Some(self)
    }
}

// ----------------------------------------------------------------------------
// Elementwise — real implementations.
// ----------------------------------------------------------------------------

impl Elementwise for CpuBackend {
    /// `(data − dark) / (flat − dark)`, averaging the flat/dark frame stacks.
    ///
    /// Ports tomocupy `proc_functions.darkflat_correction` (proc_functions.py:55).
    fn darkflat(&self, data: &mut Tomo<f32>, flat: &Frames<f32>, dark: &Frames<f32>) -> Result<()> {
        let dark2d = dark
            .array
            .mean_axis(Axis(0))
            .ok_or_else(|| Error::InvalidParam("empty dark stack".into()))?;
        let flat2d = flat
            .array
            .mean_axis(Axis(0))
            .ok_or_else(|| Error::InvalidParam("empty flat stack".into()))?;
        let mut denom = &flat2d - &dark2d;
        // Guard against divide-by-zero where flat == dark.
        denom.mapv_inplace(|v| if v.abs() < 1e-6 { 1.0 } else { v });

        // Process in projection layout so each slab is [row, col].
        let restore = data.layout == Layout::Sinogram;
        if restore {
            *data = data.to_layout(Layout::Projection);
        }
        for mut slab in data.array.axis_iter_mut(Axis(0)) {
            slab -= &dark2d;
            slab /= &denom;
        }
        if restore {
            *data = data.to_layout(Layout::Sinogram);
        }
        Ok(())
    }

    /// In-place `−ln(x)` with clamping and non-finite scrubbing.
    ///
    /// Ports tomopy `prep/normalize.py::minus_log` (normalize.py:72) and
    /// tomocupy `proc_functions.minus_log`.
    fn minus_log(&self, data: &mut Tomo<f32>) -> Result<()> {
        // Elementwise and order-independent, so parallelise across the whole
        // projection volume (tens of millions of samples) via rayon — the rest
        // of this backend's prep is already `par_chunks_mut`; the old
        // `mapv_inplace` here was the lone single-threaded stage. Bit-identical:
        // same per-element expression, no cross-element dependence.
        let map = |v: &mut f32| {
            let clamped = v.max(1e-6);
            let out = -clamped.ln();
            *v = if out.is_finite() { out } else { 0.0 };
        };
        match data.array.as_slice_memory_order_mut() {
            Some(s) => s.par_iter_mut().for_each(map),
            // Non-contiguous fallback keeps correctness for strided views.
            None => data.array.iter_mut().for_each(map),
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// FBP filter — apodized ramp construction + rfft/irfft application with the
// folded rotation-centre phase and edge-replicate padding (fbp_filter_center).
// ----------------------------------------------------------------------------

impl FbpFilter for CpuBackend {
    /// Build the full frequency-domain apodized ramp filter for a projection of
    /// width `n`.
    ///
    /// Delegates to [`crate::backend::make_fbp_filter`] with
    /// [`RampShape::Linear`] — the CPU backend ports tomopy, whose ramp is the
    /// plain straight line (the CUDA backend ports tomocupy and uses the `_wint`
    /// quadrature ramp instead). Apodization/padding/layout are shared.
    fn make_filter(&self, name: FilterName, n: usize) -> Result<Vec<f32>> {
        crate::backend::make_fbp_filter(name, n, RampShape::Linear)
    }

    /// Apply `filter` to every projection of `sino` in place, folding in the
    /// rotation-center shift (tomocupy `fbp_filter_center`).
    ///
    /// Each `(row, angle)` detector lane is centred in a `filter.len()`-wide
    /// buffer and edge-replicate-padded on both borders, forward-transformed,
    /// multiplied by the ramp filter **and** a per-row Fourier-shift phase
    /// `exp(-2πi·f_k·(ncols/2 − center)/pad)` (signed frequency `f_k`, see the
    /// body comment), inverse-transformed, and the centred `n_cols`-wide window
    /// cropped back out. The phase is the band-limited sub-pixel shift that
    /// moves the rotation axis from detector column `center` to the midpoint
    /// `ncols/2`, so after this pass **every analytic back-projector
    /// reconstructs against a centre = `ncols/2` geometry**: the rotation centre
    /// is owned in this one place and the back-projectors / Fourier grids are
    /// centre-agnostic (matching tomocupy
    /// `backproj_functions.py::fbp_filter_center`, including its `ne = 4·n`
    /// edge-replicated padding). At the default centre `ncols/2` the shift is
    /// zero and the phase is unity, so the centre-aligned goldens are
    /// unaffected.
    fn apply(&self, sino: &mut Tomo<f32>, filter: &[f32], geom: &Geometry) -> Result<()> {
        let pad = filter.len();
        if pad == 0 {
            return Err(Error::InvalidParam("empty filter".into()));
        }
        let ncols = sino.n_cols();
        if pad < ncols {
            return Err(Error::ShapeMismatch {
                expected: format!(">= {ncols} (n_cols)"),
                found: pad.to_string(),
            });
        }
        // Combined per-row kernel: ramp × centre-shift phase. `δ = ncols/2 −
        // center` is the shift (in detector pixels) that lands the axis on the
        // midpoint; `cos(0)=1, sin(0)=0` makes δ=0 the identity ramp exactly.
        //
        // The phase MUST use the SIGNED frequency `f_k = k (k≤pad/2) else k−pad`,
        // not the raw index: only the signed form is Hermitian-symmetric, so the
        // inverse transform of (real ramp × phase) stays real. A raw index
        // negates the negative-frequency half at a half-integer δ and collapses
        // the slice — the same sub-pixel-centre trap documented in `gridrec`.
        let half = ncols as f32 / 2.0;
        let two_pi = std::f32::consts::TAU;
        let make_w = |delta: f32| -> Vec<Complex32> {
            (0..pad)
                .map(|k| {
                    let fk = if k <= pad / 2 {
                        k as f32
                    } else {
                        k as f32 - pad as f32
                    };
                    let ang = -two_pi * fk * delta / pad as f32;
                    Complex32::new(filter[k] * ang.cos(), filter[k] * ang.sin())
                })
                .collect()
        };
        // Most geometries share one centre (Scalar); build a single kernel then,
        // and one per slice only for a PerRow centre.
        let n_rows = sino.n_rows();
        let wrows: Vec<Vec<Complex32>> = match &geom.center {
            Center::Scalar(c) => vec![make_w(half - c)],
            Center::PerRow(_) => (0..n_rows)
                .map(|r| make_w(half - geom.center.at(r)))
                .collect(),
        };
        let scalar = wrows.len() == 1;

        let layout = sino.layout;
        let d1 = sino.array.shape()[1];
        let mut planner = FftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(pad);
        let inv = planner.plan_fft_inverse(pad);
        let norm = 1.0 / pad as f32;
        let mut buf = vec![Complex32::new(0.0, 0.0); pad];
        // Centre the width-`ncols` lane in the `pad`-wide buffer and edge-
        // replicate the borders (tomocupy `fbp_filter_center`: `pad_side =
        // ne//2 − n//2`, `tmp[:pad_side] = data[:1]`, `tmp[pad_side+n:] =
        // data[-1:]`). Edge-replication, not zero-fill, keeps the long-tailed
        // ramp from ringing against a hard step at the projection borders; the
        // centred window is cropped back out at `[pad_side, pad_side+ncols)`.
        let pad_side = pad / 2 - ncols / 2;
        // Axis 2 is the detector column in both layouts; `lanes_mut` iterates the
        // other two axes in C order, so `flat = i0·d1 + i1` recovers the slice row
        // (`i0` in Sinogram order, `i1` in Projection order) for the per-row phase.
        for (flat, mut lane) in sino.array.lanes_mut(Axis(2)).into_iter().enumerate() {
            let w = if scalar {
                &wrows[0]
            } else {
                let row = match layout {
                    Layout::Sinogram => flat / d1,
                    Layout::Projection => flat % d1,
                };
                &wrows[row]
            };
            let first = lane[0];
            let last = lane[ncols - 1];
            for slot in buf[..pad_side].iter_mut() {
                *slot = Complex32::new(first, 0.0);
            }
            for (i, &v) in lane.iter().enumerate() {
                buf[pad_side + i] = Complex32::new(v, 0.0);
            }
            for slot in buf[pad_side + ncols..].iter_mut() {
                *slot = Complex32::new(last, 0.0);
            }
            fwd.process(&mut buf);
            for (c, wk) in buf.iter_mut().zip(w.iter()) {
                *c *= *wk;
            }
            inv.process(&mut buf);
            for (i, slot) in lane.iter_mut().enumerate() {
                *slot = buf[pad_side + i].re * norm;
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Heavy numeric kernels — stubs that name the port source.
// ----------------------------------------------------------------------------

impl Fft for CpuBackend {
    /// `rustfft` is re-entrant (immutable plan, per-call scratch), so fan the
    /// per-slice recon loop across host cores. Bit-identical to the serial
    /// default — each slice is independent.
    fn for_each_slice(
        &self,
        out: &mut Array3<f32>,
        f: &(dyn Fn(usize, ndarray::ArrayViewMut2<f32>) -> Result<()> + Sync),
    ) -> Result<()> {
        let slabs: Vec<_> = out.axis_iter_mut(Axis(0)).collect();
        slabs
            .into_par_iter()
            .enumerate()
            .try_for_each(|(row, slab)| f(row, slab))
    }

    /// In-place batched 1-D FFT via `rustfft`. `inverse` divides by `len` so
    /// `ifft(fft(x)) == x` (rustfft itself applies no normalization).
    fn fft_1d(&self, buf: &mut [Complex32], len: usize, batch: usize, inverse: bool) -> Result<()> {
        if len == 0 || batch == 0 {
            return Ok(());
        }
        if buf.len() != len * batch {
            return Err(Error::ShapeMismatch {
                expected: (len * batch).to_string(),
                found: buf.len().to_string(),
            });
        }
        let mut planner = FftPlanner::<f32>::new();
        let fft = if inverse {
            planner.plan_fft_inverse(len)
        } else {
            planner.plan_fft_forward(len)
        };
        // The `batch` independent transforms are disjoint length-`len` chunks, so
        // they run in parallel. The plan (`Arc<dyn rustfft::Fft>`) is Send+Sync and
        // shared read-only; each worker keeps its own scratch (the same buffer
        // `process` allocates internally), so the result is identical to serial.
        let scratch_len = fft.get_inplace_scratch_len();
        buf.par_chunks_mut(len).for_each_init(
            || vec![Complex32::new(0.0, 0.0); scratch_len],
            |scratch, chunk| fft.process_with_scratch(chunk, scratch),
        );
        if inverse {
            let norm = 1.0 / len as f32;
            buf.par_iter_mut().for_each(|c| *c *= norm);
        }
        Ok(())
    }
    /// In-place batched 2-D FFT via separable row–column 1-D transforms.
    /// `inverse` divides each image by `rows·cols`.
    fn fft_2d(
        &self,
        buf: &mut [Complex32],
        rows: usize,
        cols: usize,
        batch: usize,
        inverse: bool,
    ) -> Result<()> {
        if rows == 0 || cols == 0 || batch == 0 {
            return Ok(());
        }
        if buf.len() != rows * cols * batch {
            return Err(Error::ShapeMismatch {
                expected: (rows * cols * batch).to_string(),
                found: buf.len().to_string(),
            });
        }
        let mut planner = FftPlanner::<f32>::new();
        let row_fft = if inverse {
            planner.plan_fft_inverse(cols)
        } else {
            planner.plan_fft_forward(cols)
        };
        let col_fft = if inverse {
            planner.plan_fft_inverse(rows)
        } else {
            planner.plan_fft_forward(rows)
        };
        // The `batch` images are disjoint `rows*cols` chunks → parallel. Plans are
        // Send+Sync (shared read-only); each worker owns its column-gather buffer
        // and per-plan scratch, so every image's separable row-then-column pass is
        // byte-identical to the serial path.
        let row_scratch = row_fft.get_inplace_scratch_len();
        let col_scratch = col_fft.get_inplace_scratch_len();
        buf.par_chunks_mut(rows * cols).for_each_init(
            || {
                (
                    vec![Complex32::new(0.0, 0.0); rows],
                    vec![Complex32::new(0.0, 0.0); row_scratch],
                    vec![Complex32::new(0.0, 0.0); col_scratch],
                )
            },
            |(col, rscr, cscr), img| {
                // Transform each contiguous row (length `cols`).
                for row in img.chunks_mut(cols) {
                    row_fft.process_with_scratch(row, rscr);
                }
                // Transform each column (length `rows`, stride `cols`).
                for c in 0..cols {
                    for (r, slot) in col.iter_mut().enumerate() {
                        *slot = img[r * cols + c];
                    }
                    col_fft.process_with_scratch(col, cscr);
                    for (r, &v) in col.iter().enumerate() {
                        img[r * cols + c] = v;
                    }
                }
            },
        );
        if inverse {
            let norm = 1.0 / (rows * cols) as f32;
            buf.par_iter_mut().for_each(|c| *c *= norm);
        }
        Ok(())
    }
}

impl FilteredBackproject for CpuBackend {
    /// Parallel-beam voxel-driven back-projection — the pure adjoint `Wᵀ`.
    ///
    /// For each output pixel `(iy, ix)` and angle θ the detector coordinate is
    /// `t = (ix − cx)·cosθ + (iy − cy)·sinθ + center`; the (already filtered, for
    /// FBP) sinogram is sampled there by linear interpolation and summed over
    /// angles — no gain. The FBP angular-quadrature weight `π / n_angles` (the
    /// dθ of the back-projection integral) is applied by the analytic dispatcher
    /// (`recon::analytic`), NOT here, so the iterative solvers get the exact
    /// unweighted adjoint of [`ForwardProject::project`]. Slices (`z` rows) are
    /// independent and run in parallel via rayon; `center` is taken per row. The
    /// mapping matches the forward projector so phantom → project → FBP
    /// round-trips.
    ///
    /// Ports the parallel-beam back-projection of tomopy `libtomo/recon/fbp.c`.
    fn backproject(&self, sino: &Tomo<f32>, geom: &Geometry, out: &mut Volume<f32>) -> Result<()> {
        if geom.beam != Beam::Parallel {
            return Err(Error::InvalidParam(
                "cpu back-projection currently supports parallel beam only".into(),
            ));
        }
        let s = sino.as_layout(Layout::Sinogram); // [row, angle, col], contiguous
        let nz = s.n_rows();
        let nang = s.n_angles();
        let ncols = s.n_cols();
        let (oz, ny, nx) = out.dims();
        if oz != nz {
            return Err(Error::ShapeMismatch {
                expected: format!("{nz} sinogram rows"),
                found: oz.to_string(),
            });
        }
        let angles = &geom.angles.0;
        if angles.len() != nang {
            return Err(Error::ShapeMismatch {
                expected: format!("{nang} angles"),
                found: angles.len().to_string(),
            });
        }
        // (cos θ, sin θ) per angle.
        let trig: Vec<(f32, f32)> = angles
            .iter()
            .map(|&a| {
                let (sn, c) = a.sin_cos();
                (c, sn)
            })
            .collect();
        let cx = nx as f32 / 2.0;
        let cy = ny as f32 / 2.0;

        let sino_slice = s
            .array
            .as_slice()
            .ok_or_else(|| Error::InvalidParam("non-contiguous sinogram".into()))?;
        let out_slice = out
            .array
            .as_slice_mut()
            .ok_or_else(|| Error::InvalidParam("non-contiguous volume".into()))?;

        out_slice
            .par_chunks_mut(ny * nx)
            .enumerate()
            .for_each(|(row, slab)| {
                let center = geom.center.at(row);
                let base = row * nang * ncols;
                for iy in 0..ny {
                    let gy = iy as f32 - cy;
                    for ix in 0..nx {
                        let gx = ix as f32 - cx;
                        let mut acc = 0.0f32;
                        for (ia, &(c, sn)) in trig.iter().enumerate() {
                            let t = gx * c + gy * sn + center;
                            let t0 = t.floor();
                            let i0 = t0 as isize;
                            if i0 >= 0 && (i0 as usize) + 1 < ncols {
                                let frac = t - t0;
                                let off = base + ia * ncols + i0 as usize;
                                acc += sino_slice[off] * (1.0 - frac) + sino_slice[off + 1] * frac;
                            }
                        }
                        slab[iy * nx + ix] = acc;
                    }
                }
            });
        Ok(())
    }
}

impl ForwardProject for CpuBackend {
    /// Parallel-beam pixel-driven forward projection — the plain line-integral
    /// Radon transform `W` (unit pixel spacing, no gain), the exact unweighted
    /// adjoint of [`FilteredBackproject::backproject`]'s `Wᵀ`.
    ///
    /// Each object pixel `(iy, ix)` with value `f` splats onto detector column
    /// `t = (ix − cx)·cosθ + (iy − cy)·sinθ + center` for every angle, splitting
    /// `f` linearly between the two nearest columns — the exact adjoint of the
    /// back-projector's linear interpolation (same boundary rule), so the two
    /// round-trip and any consistent solve of `W x = p` converges to the
    /// physical μ (matching ART/BART and tomopy `project.c`, whose output is
    /// likewise the true line integral). `out` is overwritten with a fresh
    /// `[row, angle, col]` sinogram; slices run in parallel with a per-row
    /// `center`.
    ///
    /// Ports tomopy `libtomo/recon/project.c`.
    fn project(&self, vol: &Volume<f32>, geom: &Geometry, out: &mut Tomo<f32>) -> Result<()> {
        if geom.beam != Beam::Parallel {
            return Err(Error::InvalidParam(
                "cpu forward projection currently supports parallel beam only".into(),
            ));
        }
        let (nz, ny, nx) = vol.dims();
        let nang = geom.angles.0.len();
        let ncols = geom.detector.width;
        if nang == 0 || ncols == 0 {
            return Err(Error::InvalidParam(
                "geometry has no angles or zero detector width".into(),
            ));
        }
        let trig: Vec<(f32, f32)> = geom
            .angles
            .0
            .iter()
            .map(|&a| {
                let (sn, c) = a.sin_cos();
                (c, sn)
            })
            .collect();
        let cx = nx as f32 / 2.0;
        let cy = ny as f32 / 2.0;

        let vol_slice = vol
            .array
            .as_slice()
            .ok_or_else(|| Error::InvalidParam("non-contiguous volume".into()))?;
        let mut data = vec![0.0f32; nz * nang * ncols];
        data.par_chunks_mut(nang * ncols)
            .enumerate()
            .for_each(|(row, slab)| {
                let center = geom.center.at(row);
                let vbase = row * ny * nx;
                for iy in 0..ny {
                    let gy = iy as f32 - cy;
                    for ix in 0..nx {
                        let f = vol_slice[vbase + iy * nx + ix];
                        if f == 0.0 {
                            continue;
                        }
                        let gx = ix as f32 - cx;
                        for (ia, &(c, sn)) in trig.iter().enumerate() {
                            let t = gx * c + gy * sn + center;
                            let t0 = t.floor();
                            let i0 = t0 as isize;
                            if i0 >= 0 && (i0 as usize) + 1 < ncols {
                                let frac = t - t0;
                                let off = ia * ncols + i0 as usize;
                                slab[off] += f * (1.0 - frac);
                                slab[off + 1] += f * frac;
                            }
                        }
                    }
                }
            });
        let array = ndarray::Array3::from_shape_vec((nz, nang, ncols), data)
            .map_err(|e| Error::InvalidParam(format!("forward-projection shape: {e}")))?;
        *out = Tomo::new(array, Layout::Sinogram);
        Ok(())
    }
}

impl RayProject for CpuBackend {
    /// Sparse forward-operator rows for the row-action methods (ART/BART).
    ///
    /// Delegates to the shared, backend-independent geometry
    /// [`crate::backend::parallel_ray_rows`] (the rows are geometry-only, so CPU
    /// and CUDA produce identical rows).
    fn ray_rows(&self, geom: &Geometry, n: usize) -> Result<Vec<Vec<RayRow>>> {
        crate::backend::parallel_ray_rows(geom, n)
    }
}

/// Core 3-D median / dezinger kernel — a direct port of tomopy
/// `libtomo/misc/median_filt3d.c::medfilt3D_float` (the `dimZ > 0` branch the
/// Python wrappers always hit). For every voxel of `input` `[z, y, x]` it
/// gathers the `(2·radius+1)³` neighbourhood with **clamp-to-center** boundary
/// handling (an out-of-range index on any axis reverts to that axis's *center*
/// index, not the edge), sorts it, and takes the value at `total/2`.
///
/// `mu_threshold == 0` → plain median at every voxel. Otherwise (dezinger) the
/// median replaces the original only where `|input − median| ≥ mu_threshold`;
/// elsewhere the original passes through unchanged (matching the C, which
/// pre-copies `Output = Input` and writes only on the threshold hit).
fn medfilt3d_core(input: &Array3<f32>, radius: usize, mu_threshold: f32) -> Array3<f32> {
    let (dz, dy, dx) = input.dim();
    let r = radius as isize;
    let diameter = 2 * radius + 1;
    let total = diameter * diameter * diameter;
    let midval = total / 2; // tomopy: int division → lower-middle of the sorted window
    let (dzi, dyi, dxi) = (dz as isize, dy as isize, dx as isize);

    let out: Vec<f32> = (0..dz * dy * dx)
        .into_par_iter()
        .map(|flat| {
            // Flat index → (z, y, x) for a C-contiguous `[z, y, x]` array.
            let z = (flat / (dy * dx)) as isize;
            let rem = flat % (dy * dx);
            let y = (rem / dx) as isize;
            let x = (rem % dx) as isize;
            let center = input[[z as usize, y as usize, x as usize]];

            let mut window = Vec::with_capacity(total);
            // Axis mapping mirrors the C call (dimX = x, dimY = y, dimZ = z);
            // each axis clamps to its own center independently.
            for di in -r..=r {
                let xi = {
                    let v = x + di;
                    if v < 0 || v >= dxi {
                        x
                    } else {
                        v
                    }
                };
                for dj in -r..=r {
                    let yj = {
                        let v = y + dj;
                        if v < 0 || v >= dyi {
                            y
                        } else {
                            v
                        }
                    };
                    for dk in -r..=r {
                        let zk = {
                            let v = z + dk;
                            if v < 0 || v >= dzi {
                                z
                            } else {
                                v
                            }
                        };
                        window.push(input[[zk as usize, yj as usize, xi as usize]]);
                    }
                }
            }
            // `total_cmp` orders finite floats identically to C's `floatcomp`
            // (`<`) and is panic-free on the off-chance of a NaN.
            window.sort_by(|a, b| a.total_cmp(b));
            let median = window[midval];

            // One uniform rule covers both C branches: with `mu_threshold == 0`
            // the median always wins (`|Δ| ≥ 0` is always true → plain median),
            // and with a positive threshold only deviations ≥ it are replaced.
            if (center - median).abs() >= mu_threshold {
                median
            } else {
                center
            }
        })
        .collect();

    Array3::from_shape_vec((dz, dy, dx), out)
        .expect("medfilt3d_core: output length matches input dims")
}

impl RankFilter for CpuBackend {
    /// 3-D median filter (tomopy `misc/corr.py::median_filter3d` →
    /// `median_filt3d.c`). `size` is the cubic window width: the radius is
    /// `(max(size, 3) − 1) / 2`, so the minimum kernel is 3³ (radius 1).
    fn median3d(&self, vol: &mut Volume<f32>, size: usize) -> Result<()> {
        let radius = (size.max(3) - 1) / 2;
        vol.array = medfilt3d_core(&vol.array, radius, 0.0);
        Ok(())
    }

    /// Outlier (zinger) removal (tomopy `misc/corr.py::remove_outlier3d`).
    /// Same kernel as [`RankFilter::median3d`] but with the dezinger threshold `diff`: a
    /// voxel is replaced by the local median only when it deviates from it by
    /// at least `diff`; all others pass through unchanged.
    fn remove_outlier3d(&self, data: &mut Tomo<f32>, diff: f32, size: usize) -> Result<()> {
        let radius = (size.max(3) - 1) / 2;
        data.array = medfilt3d_core(&data.array, radius, diff);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Angles;
    use ndarray::Array3;
    use std::f32::consts::PI;

    #[test]
    fn minus_log_matches_definition() {
        let arr = Array3::from_shape_fn((1, 1, 3), |(_, _, k)| (k as f32 + 1.0) * 0.25);
        let mut t = Tomo::new(arr, Layout::Projection);
        CpuBackend.minus_log(&mut t).unwrap();
        for (k, v) in t.array.iter().enumerate() {
            let expected = -(((k as f32 + 1.0) * 0.25f32).ln());
            assert!((v - expected).abs() < 1e-6, "k={k} got {v} want {expected}");
        }
    }

    #[test]
    fn darkflat_normalizes_to_unity_on_flat_input() {
        // data == flat, dark == 0  ⇒  (flat-0)/(flat-0) == 1
        let data = Array3::from_elem((2, 2, 2), 100.0f32);
        let flat = Array3::from_elem((1, 2, 2), 100.0f32);
        let dark = Array3::from_elem((1, 2, 2), 0.0f32);
        let mut t = Tomo::new(data, Layout::Projection);
        CpuBackend
            .darkflat(&mut t, &Frames::new(flat), &Frames::new(dark))
            .unwrap();
        for v in t.array.iter() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn fft_1d_roundtrips() {
        // Two independent length-4 transforms; inverse(forward(x)) == x.
        let orig: Vec<Complex32> = (0..8).map(|k| Complex32::new(k as f32, 0.0)).collect();
        let mut buf = orig.clone();
        CpuBackend.fft_1d(&mut buf, 4, 2, false).unwrap();
        CpuBackend.fft_1d(&mut buf, 4, 2, true).unwrap();
        for (a, b) in buf.iter().zip(orig.iter()) {
            assert!((a.re - b.re).abs() < 1e-4, "re {} vs {}", a.re, b.re);
            assert!((a.im - b.im).abs() < 1e-4, "im {} vs {}", a.im, b.im);
        }
    }

    #[test]
    fn fft_2d_roundtrips() {
        // 3x4 image; inverse(forward(x)) == x.
        let orig: Vec<Complex32> = (0..12)
            .map(|k| Complex32::new(k as f32, -(k as f32)))
            .collect();
        let mut buf = orig.clone();
        CpuBackend.fft_2d(&mut buf, 3, 4, 1, false).unwrap();
        CpuBackend.fft_2d(&mut buf, 3, 4, 1, true).unwrap();
        for (a, b) in buf.iter().zip(orig.iter()) {
            assert!((a.re - b.re).abs() < 1e-4 && (a.im - b.im).abs() < 1e-4);
        }
    }

    #[test]
    fn fft_1d_rejects_wrong_buffer_len() {
        let mut buf = vec![Complex32::new(0.0, 0.0); 5];
        assert!(matches!(
            CpuBackend.fft_1d(&mut buf, 4, 2, false),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn ramp_filter_is_padded_and_symmetric() {
        let f = CpuBackend.make_filter(FilterName::Ramp, 8).unwrap();
        assert_eq!(f.len(), 32); // (4·8) is already a power of two (tomocupy ne = 4·n)
        assert_eq!(f[0], 0.0); // DC zeroed by the ramp
                               // Rises monotonically to the Nyquist bin (k = 16), then mirrors down.
        assert!(f[16] > f[8] && f[8] > f[4] && f[4] > f[2] && f[2] > f[1]);
        // Physical |ω| ramp in cycles/pixel: 0.5 at Nyquist (k = 16 = pad/2),
        // i.e. k/pad. tomopy/tomocupy's doubled ramp would read 1.0 here and put
        // analytic output at 2×μ.
        assert!((f[16] - 0.5).abs() < 1e-6);
        for k in 1..16 {
            assert!((f[k] - f[32 - k]).abs() < 1e-6, "asymmetry at {k}");
        }
    }

    #[test]
    fn fbp_none_filter_is_identity() {
        // The `None` filter is all ones, so apply() must reproduce the input
        // exactly (this validates the centre / edge-pad / FFT / crop machinery
        // itself; the ramp's correctness is proven by the FBP round-trip test).
        let arr = ndarray::Array3::from_shape_fn((1, 1, 16), |(_, _, k)| (k as f32 * 0.4).sin());
        let orig = arr.clone();
        let mut s = Tomo::new(arr, Layout::Sinogram);
        let kernel = CpuBackend.make_filter(FilterName::None, 16).unwrap();
        let geom = Geometry::parallel(
            crate::geometry::Angles::uniform(1, 0.0, std::f32::consts::PI),
            16,
            1,
            1.0,
        );
        CpuBackend.apply(&mut s, &kernel, &geom).unwrap();
        for (a, b) in s.array.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn backproject_single_angle_smears_along_ray() {
        // θ = 0, center = width/2 ⇒ t = ix; column 1 of the sinogram smears
        // across every output row at output column 1. The back-projector is the
        // pure adjoint Wᵀ (no angular-quadrature gain — the analytic dispatcher
        // applies π/nang itself), so the unit sample lands as 1.0.
        let mut sarr = Array3::<f32>::zeros((1, 1, 4)); // [row, angle, col]
        sarr[[0, 0, 1]] = 1.0;
        let s = Tomo::new(sarr, Layout::Sinogram);
        let geom = Geometry::parallel(Angles::uniform(1, 0.0, PI), 4, 1, 1.0);
        let mut out = Volume::new(Array3::<f32>::zeros((1, 4, 4)));
        CpuBackend.backproject(&s, &geom, &mut out).unwrap();
        for iy in 0..4 {
            assert!((out.array[[0, iy, 1]] - 1.0).abs() < 1e-4, "iy={iy}");
            assert!(out.array[[0, iy, 0]].abs() < 1e-6);
            assert!(out.array[[0, iy, 2]].abs() < 1e-6);
        }
    }

    #[test]
    fn forward_project_center_pixel_hits_center_column() {
        // A single pixel at the grid center (cx = cy = 2) projects to t = center
        // = width/2 = 2 for every angle, so column 2 holds the value everywhere.
        // The projector is the plain line-integral Radon transform (no gain —
        // see `project`), so the unit pixel lands as exactly 1.0 per angle.
        let mut varr = Array3::<f32>::zeros((1, 4, 4));
        varr[[0, 2, 2]] = 1.0;
        let v = Volume::new(varr);
        let nang = 4;
        let geom = Geometry::parallel(Angles::uniform(nang, 0.0, PI), 4, 1, 1.0);
        let mut s = Tomo::new(Array3::<f32>::zeros((1, 4, 4)), Layout::Sinogram);
        CpuBackend.project(&v, &geom, &mut s).unwrap();
        assert_eq!(s.layout, Layout::Sinogram);
        assert_eq!(s.array.dim(), (1, 4, 4));
        for ia in 0..4 {
            assert!((s.array[[0, ia, 2]] - 1.0).abs() < 1e-4, "ia={ia}");
            assert!(s.array[[0, ia, 0]].abs() < 1e-6);
        }
    }
}
