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

use ndarray::Axis;
use rayon::prelude::*;
use rustfft::FftPlanner;
use tomoxide_core::backend::{
    Backend, DeviceBuffer, DeviceKind, Elementwise, FbpFilter, Fft, FilteredBackproject,
    ForwardProject, RankFilter,
};
use tomoxide_core::data::{Frames, Layout, Tomo, Volume};
use tomoxide_core::dtype::{Complex32, Dtype, Element};
use tomoxide_core::error::{Error, Result};
use tomoxide_core::geometry::{Beam, Geometry};
use tomoxide_core::params::FilterName;

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
        data.array.mapv_inplace(|v| {
            let clamped = v.max(1e-6);
            let out = -clamped.ln();
            if out.is_finite() {
                out
            } else {
                0.0
            }
        });
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// FBP filter — kernel construction is real; application needs the FFT (TODO).
// ----------------------------------------------------------------------------

impl FbpFilter for CpuBackend {
    /// Build the full frequency-domain apodized ramp filter for a projection of
    /// width `n`.
    ///
    /// The returned kernel has length `pad = (2·n).next_power_of_two()` — the
    /// projection is zero-padded to `pad` before transforming so the ramp
    /// convolution does not wrap around — and is laid out in `rustfft` (fftfreq)
    /// order, symmetric about the Nyquist bin. The ramp magnitude `r` runs `0`
    /// at DC to `1` at Nyquist; `name` apodizes it. The window set matches
    /// tomopy/tomocupy; exact `_wint` quadrature weighting is reconciled when
    /// tomopy golden data is available.
    fn make_filter(&self, name: FilterName, n: usize) -> Result<Vec<f32>> {
        if n == 0 {
            return Err(Error::InvalidParam("filter length must be > 0".into()));
        }
        let pad = (2 * n).next_power_of_two();
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

    /// Apply `filter` to every projection of `sino` in place.
    ///
    /// Each `(row, angle)` line along the detector axis is zero-padded to
    /// `filter.len()`, forward-transformed, multiplied by the (real) filter,
    /// inverse-transformed, and the leading `n_cols` real samples are written
    /// back. Ramp filtering is a shift-invariant 1-D convolution, so the
    /// rotation center is handled entirely by the back-projector, not here —
    /// `geom` is unused.
    fn apply(&self, sino: &mut Tomo<f32>, filter: &[f32], _geom: &Geometry) -> Result<()> {
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
        let mut planner = FftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(pad);
        let inv = planner.plan_fft_inverse(pad);
        let norm = 1.0 / pad as f32;
        let mut buf = vec![Complex32::new(0.0, 0.0); pad];
        // Axis 2 is the detector column in both layouts, so each lane is one
        // projection regardless of row/angle ordering.
        for mut lane in sino.array.lanes_mut(Axis(2)) {
            for slot in buf.iter_mut() {
                *slot = Complex32::new(0.0, 0.0);
            }
            for (i, &v) in lane.iter().enumerate() {
                buf[i] = Complex32::new(v, 0.0);
            }
            fwd.process(&mut buf);
            for (c, &w) in buf.iter_mut().zip(filter.iter()) {
                *c *= w;
            }
            inv.process(&mut buf);
            for (i, slot) in lane.iter_mut().enumerate() {
                *slot = buf[i].re * norm;
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Heavy numeric kernels — stubs that name the port source.
// ----------------------------------------------------------------------------

impl Fft for CpuBackend {
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
        for chunk in buf.chunks_mut(len) {
            fft.process(chunk);
        }
        if inverse {
            let norm = 1.0 / len as f32;
            for c in buf.iter_mut() {
                *c *= norm;
            }
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
        let mut col = vec![Complex32::new(0.0, 0.0); rows];
        for img in buf.chunks_mut(rows * cols) {
            // Transform each contiguous row (length `cols`).
            for row in img.chunks_mut(cols) {
                row_fft.process(row);
            }
            // Transform each column (length `rows`, stride `cols`).
            for c in 0..cols {
                for (r, slot) in col.iter_mut().enumerate() {
                    *slot = img[r * cols + c];
                }
                col_fft.process(&mut col);
                for (r, &v) in col.iter().enumerate() {
                    img[r * cols + c] = v;
                }
            }
        }
        if inverse {
            let norm = 1.0 / (rows * cols) as f32;
            for c in buf.iter_mut() {
                *c *= norm;
            }
        }
        Ok(())
    }
}

impl FilteredBackproject for CpuBackend {
    /// Parallel-beam voxel-driven back-projection.
    ///
    /// For each output pixel `(iy, ix)` and angle θ the detector coordinate is
    /// `t = (ix − cx)·cosθ + (iy − cy)·sinθ + center`; the (already filtered, for
    /// FBP) sinogram is sampled there by linear interpolation and summed, then
    /// scaled by `π / n_angles`. Slices (`z` rows) are independent and run in
    /// parallel via rayon; `center` is taken per row. The mapping matches the
    /// forward projector so phantom → project → FBP round-trips.
    ///
    /// Ports the parallel-beam back-projection of tomopy `libtomo/recon/fbp.c`.
    fn backproject(&self, sino: &Tomo<f32>, geom: &Geometry, out: &mut Volume<f32>) -> Result<()> {
        if geom.beam != Beam::Parallel {
            return Err(Error::InvalidParam(
                "cpu back-projection currently supports parallel beam only".into(),
            ));
        }
        let s = sino.to_layout(Layout::Sinogram); // [row, angle, col], contiguous
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
        let scale = std::f32::consts::PI / nang as f32;

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
                        slab[iy * nx + ix] = acc * scale;
                    }
                }
            });
        Ok(())
    }
}

impl ForwardProject for CpuBackend {
    /// Parallel-beam pixel-driven forward projection (the Radon transform).
    ///
    /// Each object pixel `(iy, ix)` with value `f` splats onto detector column
    /// `t = (ix − cx)·cosθ + (iy − cy)·sinθ + center` for every angle, splitting
    /// `f` linearly between the two nearest columns. This is the exact adjoint
    /// of the back-projector's linear interpolation (same boundary rule), so the
    /// two round-trip. `out` is overwritten with a fresh `[row, angle, col]`
    /// sinogram; slices run in parallel with a per-row `center`.
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

impl RankFilter for CpuBackend {
    fn median3d(&self, _vol: &mut Volume<f32>, _size: usize) -> Result<()> {
        Err(Error::todo(
            "cpu RankFilter::median3d",
            "tomopy libtomo/misc/median_filt3d.c",
        ))
    }
    fn remove_outlier(&self, _data: &mut Tomo<f32>, _diff: f32, _size: usize) -> Result<()> {
        Err(Error::todo(
            "cpu RankFilter::remove_outlier",
            "tomopy misc/corr.py:413 remove_outlier3d",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use std::f32::consts::PI;
    use tomoxide_core::geometry::Angles;

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
        assert_eq!(f.len(), 16); // (2·8) is already a power of two
        assert_eq!(f[0], 0.0); // DC zeroed by the ramp
                               // Rises monotonically to the Nyquist bin (k = 8), then mirrors down.
        assert!(f[8] > f[4] && f[4] > f[2] && f[2] > f[1]);
        assert!((f[8] - 1.0).abs() < 1e-6); // pure ramp == 1 at Nyquist
        for k in 1..8 {
            assert!((f[k] - f[16 - k]).abs() < 1e-6, "asymmetry at {k}");
        }
    }

    #[test]
    fn fbp_none_filter_is_identity() {
        // The `None` filter is all ones, so apply() must reproduce the input
        // exactly (this validates the zero-pad / FFT / crop machinery itself;
        // the ramp's correctness is proven by the FBP round-trip test).
        let arr = ndarray::Array3::from_shape_fn((1, 1, 16), |(_, _, k)| (k as f32 * 0.4).sin());
        let orig = arr.clone();
        let mut s = Tomo::new(arr, Layout::Sinogram);
        let kernel = CpuBackend.make_filter(FilterName::None, 16).unwrap();
        let geom = Geometry::parallel(
            tomoxide_core::geometry::Angles::uniform(1, 0.0, std::f32::consts::PI),
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
        // across every output row at output column 1.
        let mut sarr = Array3::<f32>::zeros((1, 1, 4)); // [row, angle, col]
        sarr[[0, 0, 1]] = 1.0;
        let s = Tomo::new(sarr, Layout::Sinogram);
        let geom = Geometry::parallel(Angles::uniform(1, 0.0, PI), 4, 1, 1.0);
        let mut out = Volume::new(Array3::<f32>::zeros((1, 4, 4)));
        CpuBackend.backproject(&s, &geom, &mut out).unwrap();
        for iy in 0..4 {
            assert!((out.array[[0, iy, 1]] - PI).abs() < 1e-4, "iy={iy}");
            assert!(out.array[[0, iy, 0]].abs() < 1e-6);
            assert!(out.array[[0, iy, 2]].abs() < 1e-6);
        }
    }

    #[test]
    fn forward_project_center_pixel_hits_center_column() {
        // A single pixel at the grid center (cx = cy = 2) projects to t = center
        // = width/2 = 2 for every angle, so column 2 holds the value everywhere.
        let mut varr = Array3::<f32>::zeros((1, 4, 4));
        varr[[0, 2, 2]] = 1.0;
        let v = Volume::new(varr);
        let geom = Geometry::parallel(Angles::uniform(4, 0.0, PI), 4, 1, 1.0);
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
