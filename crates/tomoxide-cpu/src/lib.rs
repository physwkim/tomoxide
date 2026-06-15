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
use rustfft::FftPlanner;
use tomoxide_core::backend::{
    Backend, DeviceBuffer, DeviceKind, Elementwise, FbpFilter, Fft, FilteredBackproject,
    ForwardProject, RankFilter,
};
use tomoxide_core::data::{Frames, Layout, Tomo, Volume};
use tomoxide_core::dtype::{Complex32, Dtype, Element};
use tomoxide_core::error::{Error, Result};
use tomoxide_core::geometry::Geometry;
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
    fn fft_2d(
        &self,
        _b: &mut [Complex32],
        _rows: usize,
        _cols: usize,
        _batch: usize,
        _inv: bool,
    ) -> Result<()> {
        Err(Error::todo("cpu Fft::fft_2d", "use rustfft (milestone M1)"))
    }
}

impl FilteredBackproject for CpuBackend {
    fn backproject(
        &self,
        _sino: &Tomo<f32>,
        _geom: &Geometry,
        _out: &mut Volume<f32>,
    ) -> Result<()> {
        Err(Error::todo(
            "cpu FilteredBackproject::backproject",
            "tomopy libtomo/recon/fbp.c, gridrec/gridrec.c:195",
        ))
    }
}

impl ForwardProject for CpuBackend {
    fn project(&self, _vol: &Volume<f32>, _geom: &Geometry, _out: &mut Tomo<f32>) -> Result<()> {
        Err(Error::todo(
            "cpu ForwardProject::project",
            "tomopy libtomo/recon/project.c",
        ))
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
    fn stub_kernels_report_not_implemented() {
        let v = Volume::new(Array3::<f32>::zeros((1, 4, 4)));
        let s = Tomo::new(Array3::<f32>::zeros((4, 1, 4)), Layout::Projection);
        let geom = Geometry::parallel(
            tomoxide_core::geometry::Angles::uniform(4, 0.0, std::f32::consts::PI),
            4,
            1,
            1.0,
        );
        let mut out = v.clone();
        let mut s2 = s.clone();
        assert!(matches!(
            CpuBackend.backproject(&s, &geom, &mut out),
            Err(Error::NotImplemented { .. })
        ));
        assert!(matches!(
            CpuBackend.project(&v, &geom, &mut s2),
            Err(Error::NotImplemented { .. })
        ));
    }
}
