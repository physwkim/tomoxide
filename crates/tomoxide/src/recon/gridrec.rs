//! Fourier-grid (gridrec / direct Fourier inversion) reconstruction.
//!
//! This is the central-slice-theorem method: 1-D Fourier transform each
//! projection, place the resulting radial samples onto a 2-D Cartesian Fourier
//! grid with a gridding (convolution-interpolation) kernel, then inverse-2-D-FFT
//! and divide out the kernel's spatial roll-off (deapodization).
//!
//! tomopy's `libtomo/gridrec/gridrec.c` uses a prolate-spheroidal (PSWF)
//! gridding kernel built from Legendre tables. tomoxide uses a **Kaiser–Bessel**
//! kernel — the modern gridding/NUFFT standard with equivalent accuracy and a
//! closed-form deapodization — so this is a gridrec-*family* reconstruction, not
//! a bit-for-bit port of `gridrec.c`. It is backend-agnostic: the 1-D and 2-D
//! transforms go through the [`Fft`] capability; the gridding is plain array
//! math. Verified by self-round-trip (see `crates/tomoxide/tests/`); absolute
//! tomopy numeric parity is gated on a tomopy install (offline-unavailable).
//!
//! The pure ramp `|ρ|` weight (the polar→Cartesian density compensation,
//! mandatory for DFI) is always applied; additional apodization windows
//! (`shepp`, `hann`, …) are a follow-up — `params.filter_name` is not yet read
//! here.

use ndarray::{Array3, ArrayViewMut2, Axis};
use rayon::prelude::*;

use crate::backend::Fft;
use crate::data::Tomo;
use crate::dtype::Complex32;
use crate::error::Result;
use crate::geometry::Geometry;

/// Kaiser–Bessel kernel half-width in grid cells.
const KW: f32 = 2.0;
/// Kaiser–Bessel shape parameter β (Beatty et al. for ~2× oversampling, W=4).
const BETA: f32 = 9.0;

/// Modified Bessel function of the first kind, order 0 (series form).
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half = x as f64 / 2.0;
    for m in 1..40 {
        term *= (half / m as f64) * (half / m as f64);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum as f32
}

/// Kaiser–Bessel weight at distance `d` (grid cells) from a sample.
fn kb(d: f32, i0_beta: f32) -> f32 {
    let t = 1.0 - (d / KW) * (d / KW);
    if t <= 0.0 {
        0.0
    } else {
        bessel_i0(BETA * t.sqrt()) / i0_beta
    }
}

/// Spatial deapodization weight for image coordinate index `i` of size `m`.
/// `apod(x) ∝ sinc(√((πWx)² − β²))` — strictly positive across the FOV for
/// these parameters, so dividing by it never blows up.
fn apod(i: usize, m: usize) -> f32 {
    let x = (i as f32 - m as f32 / 2.0) / m as f32; // ∈ [−0.5, 0.5)
    let w = 2.0 * KW;
    let a = (std::f32::consts::PI * w * x).powi(2) - BETA * BETA;
    if a > 1e-6 {
        let s = a.sqrt();
        s.sin() / s
    } else if a < -1e-6 {
        let s = (-a).sqrt();
        s.sinh() / s
    } else {
        1.0
    }
}

/// In-place 2-D fft/ifft-shift (quadrant swap) of an `m × m` row-major buffer.
/// `m` is even, so a shift by `m/2` in both axes is its own inverse.
fn quadrant_swap(g: &mut [Complex32], m: usize) {
    let h = m / 2;
    for y in 0..h {
        for x in 0..m {
            let a = y * m + x;
            let b = ((y + h) % m) * m + ((x + h) % m);
            g.swap(a, b);
        }
    }
}

/// Fourier-grid reconstruction of every slice in `sino` (sinogram layout).
pub fn gridrec(sino: &Tomo<f32>, geom: &Geometry, n: usize, fft: &dyn Fft) -> Result<Array3<f32>> {
    let b = sino.to_layout(crate::data::Layout::Sinogram);
    let nz = b.n_rows();
    let nang = b.n_angles();
    let ncols = b.n_cols();
    let pad = (2 * ncols).next_power_of_two();
    let m = pad; // 2-D grid size == radial FFT length
    let two_pi = 2.0 * std::f32::consts::PI;
    let i0_beta = bessel_i0(BETA);

    // (cos θ, sin θ) per angle.
    let trig: Vec<(f32, f32)> = geom
        .angles
        .0
        .iter()
        .map(|&a| a.sin_cos())
        .map(|(s, c)| (c, s))
        .collect();
    // Signed radial frequency ρ for each FFT bin.
    let rho: Vec<f32> = (0..pad)
        .map(|k| {
            if k <= pad / 2 {
                k as f32
            } else {
                k as f32 - pad as f32
            }
        })
        .collect();
    // Precompute the deapodization profile (separable).
    let deapod: Vec<f32> = (0..m).map(|i| apod(i, m)).collect();

    let bdata = b
        .array
        .as_slice()
        .expect("contiguous sinogram (to_layout yields a standard-layout copy)");
    let mut out = Array3::<f32>::zeros((nz, n, n));
    let off = (m - n) / 2;

    // One slice's reconstruction, writing into its own `[n, n]` output view. Reads
    // only shared immutable state (`bdata`, `trig`, `rho`, `deapod`, `geom`, `fft`),
    // so slices are independent and produce bit-identical output whether run
    // serially or fanned across host threads.
    let process_row = |row: usize, mut slab: ArrayViewMut2<f32>| -> Result<()> {
        // 1. Radial 1-D FFTs of all projections (zero-padded from index 0).
        let mut radial = vec![Complex32::new(0.0, 0.0); nang * pad];
        for ia in 0..nang {
            let src = row * nang * ncols + ia * ncols;
            for j in 0..ncols {
                radial[ia * pad + j] = Complex32::new(bdata[src + j], 0.0);
            }
        }
        fft.fft_1d(&mut radial, pad, nang, false)?;

        // 2. Recenter (rotation axis at `center`) and apply the ramp |ρ|, then
        //    grid onto the centered 2-D Fourier plane with the KB kernel.
        let center = geom.center.at(row);
        let mut grid = vec![Complex32::new(0.0, 0.0); m * m];
        let half = m as f32 / 2.0;
        for ia in 0..nang {
            let (c, s) = trig[ia];
            for k in 0..pad {
                let r = rho[k];
                let ramp = r.abs();
                if ramp == 0.0 {
                    continue;
                }
                // exp(+2πi·ρ·center/pad) shifts the projection origin to
                // `center`. The phase must use the SIGNED frequency ρ = rho[k],
                // not the raw bin index k: for k > pad/2 they differ by pad, an
                // extra exp(2πi·center) factor that is 1 only for integer
                // centers — a raw index negates the negative-frequency half at a
                // half-integer center (collapsing the slice) and corrupts every
                // sub-pixel center. Integer centers are unchanged.
                let phase = two_pi * r * center / pad as f32;
                let shift = Complex32::new(phase.cos(), phase.sin());
                let val = radial[ia * pad + k] * shift * ramp;

                let gx = half + r * c;
                let gy = half + r * s;
                let ix0 = (gx - KW).ceil() as isize;
                let ix1 = (gx + KW).floor() as isize;
                let iy0 = (gy - KW).ceil() as isize;
                let iy1 = (gy + KW).floor() as isize;
                for iy in iy0..=iy1 {
                    if iy < 0 || iy as usize >= m {
                        continue;
                    }
                    let wy = kb((iy as f32 - gy).abs(), i0_beta);
                    if wy == 0.0 {
                        continue;
                    }
                    let base = iy as usize * m;
                    for ix in ix0..=ix1 {
                        if ix < 0 || ix as usize >= m {
                            continue;
                        }
                        let w = wy * kb((ix as f32 - gx).abs(), i0_beta);
                        if w != 0.0 {
                            grid[base + ix as usize] += val * w;
                        }
                    }
                }
            }
        }

        // 3. Inverse 2-D FFT (ifftshift → ifft → fftshift).
        quadrant_swap(&mut grid, m);
        fft.fft_2d(&mut grid, m, m, 1, true)?;
        quadrant_swap(&mut grid, m);

        // 4. Deapodize and crop the central n×n.
        for iy in 0..n {
            let gy = off + iy;
            for ix in 0..n {
                let gx = off + ix;
                let de = deapod[gy] * deapod[gx];
                let v = grid[gy * m + gx].re;
                slab[[iy, ix]] = if de.abs() > 1e-6 { v / de } else { v };
            }
        }
        Ok(())
    };

    // Host FFT (rustfft) → fan slices across threads; device FFT → keep serial
    // (its parallelism is on-device, and concurrent host calls would race the
    // single stream). Either path produces the identical volume.
    if fft.host_concurrent() {
        let slabs: Vec<_> = out.axis_iter_mut(Axis(0)).collect();
        slabs
            .into_par_iter()
            .enumerate()
            .try_for_each(|(row, slab)| process_row(row, slab))?;
    } else {
        for (row, slab) in out.axis_iter_mut(Axis(0)).enumerate() {
            process_row(row, slab)?;
        }
    }
    Ok(out)
}
