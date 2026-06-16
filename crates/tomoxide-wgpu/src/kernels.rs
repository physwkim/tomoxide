//! WGSL kernel ports backing the [`Backend`](tomoxide_core::backend::Backend)
//! capability traits. Only compiled under `gpu-wgpu`.
//!
//! GPU f32 is not bit-exact with the CPU/numpy reference (parallel reduction
//! order and approximate transcendentals), so the parity bar here is a
//! tolerance, not Δ=0 — see the `gpu_tests` in [`crate`].

use bytemuck::{Pod, Zeroable};
use ndarray::{Array3, Axis};
use tomoxide_core::backend::{
    make_fbp_filter, Elementwise, FbpFilter, Fft, FilteredBackproject, ForwardProject, RankFilter,
};
use tomoxide_core::data::{Frames, Layout, Tomo, Volume};
use tomoxide_core::dtype::Complex32;
use tomoxide_core::error::{Error, Result};
use tomoxide_core::geometry::{Beam, Geometry};
use tomoxide_core::params::FilterName;

use crate::shaders::{
    BACKPROJECT_WGSL, BLUESTEIN_WGSL, ELEMENTWISE_WGSL, FBP_FILTER_WGSL, FFT_TRANSPOSE_WGSL,
    FFT_WGSL, MEDFILT3D_WGSL, PROJECT_WGSL,
};
use crate::WgpuBackend;

/// Max window the GPU median kernel can hold (must match `MAX_WIN` in
/// `medfilt3d.wgsl`): diameter 7, i.e. `size ≤ 7`.
const MEDFILT_MAX_WIN: usize = 343;

/// Uniform block for the `darkflat` kernel. Padded to 16 bytes to satisfy the
/// WGSL uniform-buffer layout rules.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DfParams {
    n_elems: u32,
    plane_size: u32,
    _pad0: u32,
    _pad1: u32,
}

impl Elementwise for WgpuBackend {
    /// `(data − dark) / (flat − dark)`, averaging the flat/dark frame stacks.
    ///
    /// Mirrors [`CpuBackend::darkflat`](../../tomoxide_cpu): the frame averaging
    /// and the divide-by-zero guard run host-side (a single mean per plane,
    /// cheap and order-sensitive), then the per-element broadcast subtraction
    /// and division run on the GPU.
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

        // Work in projection layout so element i = proj*plane + (row*cols+col)
        // and the dark/denom plane is indexed by `i % plane`.
        let restore = data.layout == Layout::Sinogram;
        if restore {
            *data = data.to_layout(Layout::Projection);
        }
        let (np, nr, nc) = data.array.dim();
        if dark2d.dim() != (nr, nc) {
            return Err(Error::ShapeMismatch {
                expected: format!("{:?}", (nr, nc)),
                found: format!("{:?}", dark2d.dim()),
            });
        }
        let plane = nr * nc;

        let host: Vec<f32> = data.array.iter().copied().collect();
        let dark_std = dark2d.as_standard_layout();
        let denom_std = denom.as_standard_layout();
        let data_buf = self.storage_rw("df_data", &host);
        let dark_buf = self.storage_ro("df_dark", dark_std.as_slice().expect("standard layout"));
        let denom_buf = self.storage_ro("df_denom", denom_std.as_slice().expect("standard layout"));
        let params = DfParams {
            n_elems: host.len() as u32,
            plane_size: plane as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let param_buf = self.uniform("df_params", &params);
        self.dispatch1d(
            ELEMENTWISE_WGSL,
            "darkflat",
            &[&data_buf, &dark_buf, &denom_buf, &param_buf],
            host.len() as u32,
        );
        let out = self.download_f32(&data_buf, host.len());
        data.array = Array3::from_shape_vec((np, nr, nc), out).expect("len matches dims");

        if restore {
            *data = data.to_layout(Layout::Sinogram);
        }
        Ok(())
    }

    /// In-place `−ln(max(x, 1e-6))` with non-finite results scrubbed to `0`.
    ///
    /// Order-independent, so it runs on whatever layout `data` is in. GPU `log`
    /// differs from libm `ln` by a few ULP — callers compare with a tolerance.
    fn minus_log(&self, data: &mut Tomo<f32>) -> Result<()> {
        let dims = data.array.dim();
        let host: Vec<f32> = data.array.iter().copied().collect();
        let buf = self.storage_rw("ml_data", &host);
        self.dispatch1d(ELEMENTWISE_WGSL, "minus_log", &[&buf], host.len() as u32);
        let out = self.download_f32(&buf, host.len());
        data.array = Array3::from_shape_vec(dims, out).expect("len matches dims");
        Ok(())
    }
}

/// Uniform block for the `backproject` kernel. Padded to 32 bytes (a 16-byte
/// multiple) to satisfy the WGSL uniform-buffer layout rules.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BpParams {
    nproj: u32,
    ncols: u32,
    ny: u32,
    nx: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl FilteredBackproject for WgpuBackend {
    /// Parallel-beam voxel-driven back-projection on the GPU.
    ///
    /// Mirrors [`CpuBackend::backproject`](../../tomoxide_cpu): one GPU thread
    /// per output voxel sums the (already filtered) sinogram along all angles by
    /// linear interpolation and scales by `π / n_angles`. The per-angle
    /// `(cosθ, sinθ)` and the per-row `center` are computed host-side with the
    /// same `sin_cos` as the CPU path, so the only GPU/CPU divergence is the
    /// multiply-accumulate rounding — callers compare with a tolerance, not Δ=0.
    fn backproject(&self, sino: &Tomo<f32>, geom: &Geometry, out: &mut Volume<f32>) -> Result<()> {
        if geom.beam != Beam::Parallel {
            return Err(Error::InvalidParam(
                "wgpu back-projection currently supports parallel beam only".into(),
            ));
        }
        let s = sino.to_layout(Layout::Sinogram); // [row, angle, col]
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

        let (cossin, center) = cossin_center(geom, nz);
        let scale = std::f32::consts::PI / nang as f32;

        let sino_std = s.array.as_standard_layout();
        let sino_buf = self.storage_ro("bp_sino", sino_std.as_slice().expect("standard layout"));
        let cossin_buf = self.storage_ro("bp_cossin", &cossin);
        let center_buf = self.storage_ro("bp_center", &center);
        let total = nz * ny * nx;
        let vol_buf = self.storage_rw("bp_vol", &vec![0.0f32; total]);
        let params = BpParams {
            nproj: nang as u32,
            ncols: ncols as u32,
            ny: ny as u32,
            nx: nx as u32,
            scale,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let param_buf = self.uniform("bp_params", &params);
        self.dispatch1d(
            BACKPROJECT_WGSL,
            "backproject",
            &[&sino_buf, &cossin_buf, &center_buf, &vol_buf, &param_buf],
            total as u32,
        );
        let result = self.download_f32(&vol_buf, total);
        out.array = Array3::from_shape_vec((nz, ny, nx), result).expect("len matches dims");
        Ok(())
    }
}

/// Per-angle `(cosθ, sinθ)` interleaved plus the per-row `center`, both computed
/// host-side with the same `sin_cos` as the CPU reference so the GPU sampling
/// (and its inclusion-boundary decision) is bit-identical to the CPU path. The
/// center is expanded to one value per row so the `Scalar` and `PerRow` cases
/// share a single buffer path. Shared by the back- and forward-projectors.
fn cossin_center(geom: &Geometry, nz: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cossin = Vec::with_capacity(geom.angles.0.len() * 2);
    for &a in &geom.angles.0 {
        let (sn, c) = a.sin_cos();
        cossin.push(c);
        cossin.push(sn);
    }
    let center = (0..nz).map(|r| geom.center.at(r)).collect();
    (cossin, center)
}

impl ForwardProject for WgpuBackend {
    /// Parallel-beam pixel-driven forward projection (the Radon transform).
    ///
    /// Mirrors [`CpuBackend::project`](../../tomoxide_cpu) — the exact linear-
    /// interp adjoint of [`Self::backproject`]. Forward projection is a scatter
    /// (each pixel splats onto two detector columns), so to stay race-free the
    /// GPU maps one thread per `(row, angle)`: each owns a disjoint detector-
    /// column span and visits pixels in the CPU's `(iy, ix)` order, so the
    /// per-column accumulation order matches and only the multiply-accumulate
    /// rounding diverges (tolerance parity, not Δ=0). `out` is overwritten with
    /// a fresh `[row, angle, col]` sinogram.
    fn project(&self, vol: &Volume<f32>, geom: &Geometry, out: &mut Tomo<f32>) -> Result<()> {
        if geom.beam != Beam::Parallel {
            return Err(Error::InvalidParam(
                "wgpu forward projection currently supports parallel beam only".into(),
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

        let (cossin, center) = cossin_center(geom, nz);
        let vol_std = vol.array.as_standard_layout();
        let vol_buf = self.storage_ro("fp_vol", vol_std.as_slice().expect("standard layout"));
        let cossin_buf = self.storage_ro("fp_cossin", &cossin);
        let center_buf = self.storage_ro("fp_center", &center);
        let total = nz * nang * ncols;
        let sino_buf = self.storage_rw("fp_sino", &vec![0.0f32; total]);
        let params = FpParams {
            nproj: nang as u32,
            ncols: ncols as u32,
            ny: ny as u32,
            nx: nx as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let param_buf = self.uniform("fp_params", &params);
        // One thread per (row, angle); each owns a disjoint sinogram column span.
        self.dispatch1d(
            PROJECT_WGSL,
            "project",
            &[&vol_buf, &cossin_buf, &center_buf, &sino_buf, &param_buf],
            (nz * nang) as u32,
        );
        let result = self.download_f32(&sino_buf, total);
        let array = Array3::from_shape_vec((nz, nang, ncols), result).expect("len matches dims");
        *out = Tomo::new(array, Layout::Sinogram);
        Ok(())
    }
}

/// Uniform block for the `project` kernel. Padded to 32 bytes (a 16-byte
/// multiple) to satisfy the WGSL uniform-buffer layout rules.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FpParams {
    nproj: u32,
    ncols: u32,
    ny: u32,
    nx: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Uniform block for the `medfilt3d` kernel. Padded to 32 bytes (a 16-byte
/// multiple) to satisfy the WGSL uniform-buffer layout rules.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MfParams {
    dz: u32,
    dy: u32,
    dx: u32,
    radius: u32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl WgpuBackend {
    /// 3-D median/dezinger core shared by [`RankFilter::median3d`] and
    /// [`RankFilter::remove_outlier3d`] — one GPU thread per voxel, bit-exact
    /// with `medfilt3d_core` on the CPU (pure gather + order statistic).
    fn medfilt3d_gpu(
        &self,
        arr: &Array3<f32>,
        radius: usize,
        threshold: f32,
    ) -> Result<Array3<f32>> {
        let diameter = 2 * radius + 1;
        let total = diameter * diameter * diameter;
        if total > MEDFILT_MAX_WIN {
            return Err(Error::InvalidParam(format!(
                "wgpu median window {diameter}³={total} exceeds the GPU cap of \
                 {MEDFILT_MAX_WIN} (size ≤ 7); use the CPU backend for larger windows"
            )));
        }
        let (dz, dy, dx) = arr.dim();
        let host: Vec<f32> = arr.iter().copied().collect();
        let n = host.len();
        let inp = self.storage_ro("mf_inp", &host);
        let outp = self.storage_rw("mf_outp", &vec![0.0f32; n]);
        let params = MfParams {
            dz: dz as u32,
            dy: dy as u32,
            dx: dx as u32,
            radius: radius as u32,
            threshold,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let param_buf = self.uniform("mf_params", &params);
        self.dispatch1d(
            MEDFILT3D_WGSL,
            "medfilt3d",
            &[&inp, &outp, &param_buf],
            n as u32,
        );
        let out = self.download_f32(&outp, n);
        Ok(Array3::from_shape_vec((dz, dy, dx), out).expect("len matches dims"))
    }
}

impl RankFilter for WgpuBackend {
    /// 3-D median filter (tomopy `median_filter3d`). Bit-exact with the CPU.
    fn median3d(&self, vol: &mut Volume<f32>, size: usize) -> Result<()> {
        let radius = (size.max(3) - 1) / 2;
        vol.array = self.medfilt3d_gpu(&vol.array, radius, 0.0)?;
        Ok(())
    }

    /// Outlier (zinger) removal (tomopy `remove_outlier3d`): the same kernel as
    /// [`Self::median3d`] but replacing a voxel by the local median only where
    /// it deviates from it by at least `diff`. Bit-exact with the CPU.
    fn remove_outlier3d(&self, data: &mut Tomo<f32>, diff: f32, size: usize) -> Result<()> {
        let radius = (size.max(3) - 1) / 2;
        data.array = self.medfilt3d_gpu(&data.array, radius, diff)?;
        Ok(())
    }
}

/// Uniform block for the FBP filter-multiply kernel. Padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FfltParams {
    pad: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Uniform block for the radix-2 FFT kernel (16 bytes — already a multiple of 16,
/// so no padding is needed).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FftParams {
    n: u32,
    logn: u32,
    m: u32,
    sign: f32,
}

/// Uniform block for the Bluestein convolution-multiply kernel. Padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BzParams {
    m: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Uniform block for the FFT transpose kernel. Padded to 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

impl WgpuBackend {
    /// Issue the in-place radix-2 passes (bit-reversal + `log2(n)` butterfly
    /// stages) over `data`, treating it as `lanes` contiguous transforms of
    /// length `n`. Does not normalize; the inverse `1/n` is applied by the
    /// caller. Submissions serialize on the queue, so each stage observes the
    /// previous one's writes; transient uniform buffers stay alive through the
    /// pending submissions even after this returns.
    fn fft_passes(&self, data: &wgpu::Buffer, n: usize, lanes: usize, inverse: bool) {
        let logn = n.trailing_zeros();
        let sign = if inverse { 1.0f32 } else { -1.0f32 };
        let p0 = self.uniform(
            "fft_p",
            &FftParams {
                n: n as u32,
                logn,
                m: 0,
                sign,
            },
        );
        self.dispatch1d(FFT_WGSL, "bitrev", &[data, &p0], (n * lanes) as u32);
        let mut m = 2u32;
        for _ in 0..logn {
            let p = self.uniform(
                "fft_p",
                &FftParams {
                    n: n as u32,
                    logn,
                    m,
                    sign,
                },
            );
            self.dispatch1d(FFT_WGSL, "butterfly", &[data, &p], (n * lanes / 2) as u32);
            m <<= 1;
        }
    }

    /// Transpose each `rows × cols` image of `src` into `cols × rows` in `dst`.
    fn fft_transpose(
        &self,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        batch: usize,
    ) {
        let params = TParams {
            rows: rows as u32,
            cols: cols as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let p = self.uniform("fft_t", &params);
        self.dispatch1d(
            FFT_TRANSPOSE_WGSL,
            "transpose",
            &[src, dst, &p],
            (rows * cols * batch) as u32,
        );
    }

    /// Upload an interleaved complex buffer to a `read_write` storage buffer
    /// (each `Complex32` becomes a `vec2<f32>`).
    fn upload_complex(&self, label: &str, buf: &[Complex32]) -> wgpu::Buffer {
        let host: Vec<f32> = buf.iter().flat_map(|c| [c.re, c.im]).collect();
        self.storage_rw(label, &host)
    }

    /// Read an interleaved complex buffer back into `buf`, scaling by `norm`.
    fn download_complex(&self, data: &wgpu::Buffer, buf: &mut [Complex32], norm: f32) {
        let out = self.download_f32(data, buf.len() * 2);
        for (c, chunk) in buf.iter_mut().zip(out.chunks_exact(2)) {
            c.re = chunk[0] * norm;
            c.im = chunk[1] * norm;
        }
    }

    /// Bluestein (chirp-z) batched 1-D DFT for an **arbitrary** length `n`, used
    /// by [`Fft::fft_1d`] when `n` is not a power of two (the radix-2 kernel only
    /// handles power-of-two lengths). `buf` holds `lanes` contiguous transforms
    /// of length `n`; the result overwrites it in place. `inverse` selects the
    /// IDFT and applies the matching `1/n` normalization (like the radix-2 path).
    ///
    /// A length-`n` DFT `X[k] = Σ x[j]·exp(s·2πi·jk/n)` (s = −1 forward, +1
    /// inverse) is rewritten via `jk = (j² + k² − (k−j)²)/2` as
    /// `X[k] = p[k]·Σ (x[j]·p[j])·h[k−j]` with chirps `p[j] = exp(s·πi·j²/n)` and
    /// `h[m] = conj(p[|m|])` — a linear convolution evaluated by a power-of-two
    /// circular convolution of length `m = next_power_of_two(2n−1)` (FFT both,
    /// multiply spectra, inverse FFT). The chirps, the input premultiply, and the
    /// output postmultiply/crop are done host-side so the `j² mod 2n` argument
    /// reduction matches the CPU reference's precision; the three FFTs and the
    /// spectral multiply run on the GPU in one serialized submission chain.
    fn fft_bluestein(&self, buf: &mut [Complex32], n: usize, lanes: usize, inverse: bool) {
        let m = (2 * n - 1).next_power_of_two();
        let s = if inverse { 1.0f32 } else { -1.0f32 };
        let pi = std::f32::consts::PI;

        // Chirp p[j] = exp(s·πi·j²/n); reduce j² mod 2n first so the angle stays
        // small and precise even for large j (πj²/n grows quadratically).
        let two_n = 2 * n as u64;
        let p: Vec<Complex32> = (0..n)
            .map(|j| {
                let r = ((j as u64 * j as u64) % two_n) as f32;
                let ang = s * pi * r / n as f32;
                Complex32::new(ang.cos(), ang.sin())
            })
            .collect();

        // Per-lane premultiplied, zero-padded input a[l·m + j] = x[l·n + j]·p[j].
        let mut a_host = vec![Complex32::new(0.0, 0.0); lanes * m];
        for l in 0..lanes {
            for j in 0..n {
                a_host[l * m + j] = buf[l * n + j] * p[j];
            }
        }

        // Symmetric kernel h on the length-m ring: h[0], h[±j] = conj(p[j]).
        let mut h_host = vec![Complex32::new(0.0, 0.0); m];
        h_host[0] = p[0].conj();
        for j in 1..n {
            let hj = p[j].conj();
            h_host[j] = hj;
            h_host[m - j] = hj;
        }

        // FFT both, multiply spectra (h broadcast across lanes), inverse FFT.
        let a_buf = self.upload_complex("bz_a", &a_host);
        let h_buf = self.upload_complex("bz_h", &h_host);
        self.fft_passes(&h_buf, m, 1, false);
        self.fft_passes(&a_buf, m, lanes, false);
        let p_u = self.uniform(
            "bz_p",
            &BzParams {
                m: m as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );
        self.dispatch1d(
            BLUESTEIN_WGSL,
            "cmul",
            &[&a_buf, &h_buf, &p_u],
            (lanes * m) as u32,
        );
        self.fft_passes(&a_buf, m, lanes, true);

        // Download the (unnormalized inverse → ×m) convolution, then postmultiply
        // by p[k], crop to n, and apply 1/m (convolution) plus 1/n (IDFT).
        let out = self.download_f32(&a_buf, lanes * m * 2);
        let conv_norm = 1.0 / m as f32;
        let inv_norm = if inverse { 1.0 / n as f32 } else { 1.0 };
        let scale = conv_norm * inv_norm;
        for l in 0..lanes {
            for k in 0..n {
                let off = l * m + k;
                let c = Complex32::new(out[2 * off], out[2 * off + 1]) * p[k];
                buf[l * n + k] = Complex32::new(c.re * scale, c.im * scale);
            }
        }
    }
}

impl Fft for WgpuBackend {
    /// Batched 1-D FFT. Power-of-two `len` runs the direct radix-2 kernel; any
    /// other `len` runs the Bluestein chirp-z transform (also radix-2 under the
    /// hood), so the GPU handles arbitrary lengths like the CPU backend rather
    /// than erroring out. `inverse` divides by `len`, matching the CPU backend.
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
        if len.is_power_of_two() {
            let data = self.upload_complex("fft_1d", buf);
            self.fft_passes(&data, len, batch, inverse);
            let norm = if inverse { 1.0 / len as f32 } else { 1.0 };
            self.download_complex(&data, buf, norm);
        } else {
            self.fft_bluestein(buf, len, batch, inverse);
        }
        Ok(())
    }

    /// Batched 2-D radix-2 FFT, as a row pass + transpose + row pass + transpose
    /// (so both axes run as contiguous radix-2 transforms). Requires power-of-two
    /// `rows` and `cols`. `inverse` divides by `rows·cols`.
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
        if !rows.is_power_of_two() || !cols.is_power_of_two() {
            return Err(Error::InvalidParam(format!(
                "wgpu 2-D FFT requires power-of-two dims (got {rows}×{cols}); use the CPU backend"
            )));
        }
        let data = self.upload_complex("fft_2d", buf);
        // Row pass: `batch·rows` contiguous transforms of length `cols`.
        self.fft_passes(&data, cols, rows * batch, inverse);
        // Transpose to make columns contiguous, transform length `rows`, back.
        let scratch = self.storage_rw("fft_2d_t", &vec![0.0f32; rows * cols * batch * 2]);
        self.fft_transpose(&data, &scratch, rows, cols, batch);
        self.fft_passes(&scratch, rows, cols * batch, inverse);
        self.fft_transpose(&scratch, &data, cols, rows, batch);
        let norm = if inverse {
            1.0 / (rows * cols) as f32
        } else {
            1.0
        };
        self.download_complex(&data, buf, norm);
        Ok(())
    }
}

impl FbpFilter for WgpuBackend {
    /// Build the FBP apodized ramp filter on the host. The kernel is
    /// device-independent, so this delegates to the shared
    /// [`make_fbp_filter`] — CPU and GPU build the identical filter.
    fn make_filter(&self, name: FilterName, n: usize) -> Result<Vec<f32>> {
        make_fbp_filter(name, n)
    }

    /// Apply `filter` to every projection of `sino` on the GPU. Each detector
    /// lane (axis 2 in both layouts) is zero-padded to `pad = filter.len()`,
    /// forward-transformed, multiplied by the real filter, inverse-transformed,
    /// scaled by `1/pad`, and its leading `n_cols` real samples written back.
    /// Forward FFT, frequency-domain multiply, and inverse FFT all run on the
    /// GPU in one serialized submission chain — no host round-trip between the
    /// transforms. Mirrors `CpuBackend::apply`; `geom` is unused because ramp
    /// filtering is shift-invariant (the rotation center is the
    /// back-projector's job). Requires a power-of-two `pad` (the GPU FFT is
    /// radix-2 only); other lengths error so the caller can fall back to CPU.
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
        if !pad.is_power_of_two() {
            return Err(Error::InvalidParam(format!(
                "wgpu FBP filter requires a power-of-two length (got {pad}); use the CPU backend"
            )));
        }
        // Gather every detector lane into a zero-padded batch of complex
        // transforms; lane `L` occupies `[L·pad, L·pad+ncols)`, the rest stays
        // zero. `lanes`/`lanes_mut` iterate in the same order, so `L` maps
        // consistently between gather and scatter regardless of memory layout.
        let batch = sino.array.len() / ncols;
        let mut host = vec![Complex32::new(0.0, 0.0); batch * pad];
        for (l, lane) in sino.array.lanes(Axis(2)).into_iter().enumerate() {
            let base = l * pad;
            for (i, &v) in lane.iter().enumerate() {
                host[base + i] = Complex32::new(v, 0.0);
            }
        }
        let data = self.upload_complex("fbp_filter", &host);
        self.fft_passes(&data, pad, batch, false);
        let w = self.storage_ro("fbp_w", filter);
        let p = self.uniform(
            "fbp_p",
            &FfltParams {
                pad: pad as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );
        self.dispatch1d(
            FBP_FILTER_WGSL,
            "apply_filter",
            &[&data, &w, &p],
            (batch * pad) as u32,
        );
        self.fft_passes(&data, pad, batch, true);
        // fft_passes leaves the inverse unnormalized; fold the 1/pad in here.
        self.download_complex(&data, &mut host, 1.0 / pad as f32);
        for (l, mut lane) in sino.array.lanes_mut(Axis(2)).into_iter().enumerate() {
            let base = l * pad;
            for (i, slot) in lane.iter_mut().enumerate() {
                *slot = host[base + i].re;
            }
        }
        Ok(())
    }
}
