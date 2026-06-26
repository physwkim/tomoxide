//! USFFT (Gaussian-gridding) Fourier reconstruction — tomocupy `fourierrec`.
//!
//! This is the central-slice-theorem method, like [`crate::gridrec`], but a
//! faithful port of tomocupy's `cfunc_fourierrec.cu` / `kernels_fourierrec.cuh`
//! rather than the Kaiser–Bessel gridrec family: it uses tomocupy's **Gaussian**
//! gridding kernel (half-width `m`, shape `mu` derived from the accuracy target
//! `eps = 1e-3`), an oversampled `(2n+2m)²` Fourier grid with periodic border
//! `wrap`, a centred `2n×2n` inverse FFT, and a closed-form Gaussian
//! deapodization (`divphi`).
//!
//! tomocupy's pipeline is `fbp_filter_center(data)` → `FourierRec.backprojection`
//! (see `reconstruction/backproj_parallel.py`), so the gridding kernel here has
//! **no internal ramp**: it expects an already ramp-filtered sinogram, exactly
//! what [`crate::analytic`] feeds it via [`FbpFilter::apply`]. (gridrec, by
//! contrast, applies the ramp itself.)
//!
//! It is backend-agnostic: the centred 1-D radial transforms and the 2-D inverse
//! transform go through the [`Fft`] capability (so it composes onto any backend,
//! including wgpu, for free), and the gather / wrap / deapodize steps are plain
//! array math. The centred transforms are replicated exactly from tomocupy's
//! `ifftshiftc`/`fftshiftc` sign-modulation (multiply by `(−1)^i` before and
//! after the FFT), so the gather coordinate convention and the `divphi` crop
//! offsets line up with the reference.
//!
//! Verified by analytic-phantom round-trip (Pearson correlation with the
//! phantom) and cross-method agreement with the already-verified gridrec; a
//! bit-for-bit tomocupy numeric parity needs a CUDA golden run, which is
//! offline-unavailable.

use ndarray::{Array3, ArrayViewMut2};

use crate::backend::Fft;
use crate::data::{Layout, Tomo};
use crate::dtype::Complex32;
use crate::error::Result;
use crate::geometry::Geometry;

/// Gridding accuracy target (tomocupy hardcodes `eps = 1e-3`).
const EPS: f64 = 1e-3;

/// Centred-DFT modulation sign — tomocupy's `ifftshiftc`/`fftshiftc` factor
/// `1 − 2·((i+1) % 2)`: `−1` on even indices, `+1` on odd. Multiplying a signal
/// by this before and after an FFT centres the transform (DC at index `N/2`).
#[inline]
fn shift_sign(i: usize) -> f32 {
    if i % 2 == 0 {
        -1.0
    } else {
        1.0
    }
}

/// Fourier-grid reconstruction of every slice in `sino` (sinogram layout) using
/// tomocupy's Gaussian-USFFT gridding. `sino` must be **ramp-filtered already**
/// (the analytic dispatcher applies the FBP filter first); `n` is the output
/// slice size and `fft` provides the centred 1-D and 2-D transforms.
pub fn fourierrec(
    sino: &Tomo<f32>,
    geom: &Geometry,
    n: usize,
    fft: &dyn Fft,
) -> Result<Array3<f32>> {
    let b = sino.as_layout(Layout::Sinogram);
    let nz = b.n_rows();
    let nang = b.n_angles();
    let nd = b.n_cols(); // detector width == tomocupy's `n`

    // Gaussian-kernel parameters (tomocupy `cfunc_fourierrec` constructor).
    let ndf = nd as f64;
    let neg_log_eps = -EPS.ln(); // −ln(eps) > 0
    let mu = neg_log_eps / (2.0 * ndf * ndf);
    // m = ceil(2n·(1/π)·sqrt(−mu·log(eps) + (mu·n)²/4)); == 4 for eps=1e-3.
    let inside = mu * neg_log_eps + (mu * ndf) * (mu * ndf) / 4.0;
    let m = (2.0 * ndf / std::f64::consts::PI * inside.sqrt()).ceil() as usize;

    let ng = 2 * nd + 2 * m; // oversampled grid (with m-wide borders)
    let nf = 2 * nd; // centred inverse-FFT size
    let mu = mu as f32;
    let coeff0 = (std::f32::consts::PI) / (mu * 4.0 * (nd as f32) * (nd as f32)); // gather amplitude
    let coeff1 = -std::f32::consts::PI * std::f32::consts::PI / mu; // Gaussian exponent scale
    let gscale = 4.0 / nd as f32; // tomocupy `scale` in gather
                                  // tomocupy `divphi` global sign `1 − n%4` (== +1 for n divisible by 4, the
                                  // supported case; replicated verbatim so the overall sign matches).
    let phi_sign = 1.0 - (nd % 4) as f32;

    // (cos θ, sin θ) per angle.
    let trig: Vec<(f32, f32)> = geom
        .angles
        .0
        .iter()
        .map(|&a| a.sin_cos())
        .map(|(s, c)| (c, s))
        .collect();

    let bdata = b
        .array
        .as_slice()
        .expect("contiguous sinogram (to_layout yields a standard-layout copy)");
    let mut out = Array3::<f32>::zeros((nz, n, n));
    // Output covers the central n×n of the nd-pixel field of view.
    let crop = (nd - n.min(nd)) / 2;

    // One slice's reconstruction, writing into its own `[n, n]` output view. Reads
    // only shared immutable state, so slices are independent and the output is
    // bit-identical whether run serially or fanned across host threads.
    let process_row = |row: usize, mut slab: ArrayViewMut2<f32>| -> Result<()> {
        // Separable Gaussian weight buffers (length 2m+1), per slice/thread.
        let mut kern0 = vec![0.0f32; 2 * m + 1];
        let mut kern1 = vec![0.0f32; 2 * m + 1];
        // 1. Centred 1-D FFT of every projection (length nd): premodulate by the
        //    shift sign, forward FFT, postmodulate — tomocupy's
        //    ifftshiftc → fft1d → ifftshiftc.
        let mut radial = vec![Complex32::new(0.0, 0.0); nang * nd];
        for ia in 0..nang {
            let src = row * nang * nd + ia * nd;
            for j in 0..nd {
                radial[ia * nd + j] = Complex32::new(bdata[src + j] * shift_sign(j), 0.0);
            }
        }
        fft.fft_1d(&mut radial, nd, nang, false)?;
        for ia in 0..nang {
            for k in 0..nd {
                radial[ia * nd + k] *= shift_sign(k);
            }
        }

        // The rotation-centre shift is folded into the shared FBP filter
        // upstream (FbpFilter::apply moves the axis onto the detector midpoint
        // nd/2), so the radial spectrum is already centred — no per-grid
        // recenter here, and this method is centre-agnostic.

        // 2. Gather each radial Fourier sample onto the (2n+2m)² Cartesian grid
        //    with the separable Gaussian kernel (tomocupy `gather`).
        let mut grid = vec![Complex32::new(0.0, 0.0); ng * ng];
        for ia in 0..nang {
            let (c, s) = trig[ia];
            for td in 0..nd {
                let mut x0 = (td as f32 - nd as f32 / 2.0) / nd as f32 * c;
                // tomocupy uses `-sin θ` here (its projection geometry); tomoxide's
                // forward projector and gridrec place the grid y-coordinate at
                // `+sin θ`, so use the same sign to avoid a vertical flip.
                let mut y0 = (td as f32 - nd as f32 / 2.0) / nd as f32 * s;
                if x0 >= 0.5 {
                    x0 = 0.5 - 1e-5;
                }
                if y0 >= 0.5 {
                    y0 = 0.5 - 1e-5;
                }
                let g0 = radial[ia * nd + td] * gscale;

                let base0 = (2.0 * nd as f32 * x0).floor() as isize - m as isize;
                let base1 = (2.0 * nd as f32 * y0).floor() as isize - m as isize;
                for (i0, kv) in kern0.iter_mut().enumerate() {
                    let w0 = (base0 + i0 as isize) as f32 / (2.0 * nd as f32) - x0;
                    *kv = (coeff1 * w0 * w0).exp();
                }
                for (i1, kv) in kern1.iter_mut().enumerate() {
                    let w1 = (base1 + i1 as isize) as f32 / (2.0 * nd as f32) - y0;
                    *kv = (coeff1 * w1 * w1).exp();
                }

                let col0 = nd as isize + m as isize + base0;
                let row0 = nd as isize + m as isize + base1;
                for (i1, &wy) in kern1.iter().enumerate() {
                    let rbase = (row0 + i1 as isize) as usize * ng;
                    for (i0, &wx) in kern0.iter().enumerate() {
                        let w = coeff0 * wx * wy;
                        let gc = (col0 + i0 as isize) as usize;
                        grid[rbase + gc] += g0 * w;
                    }
                }
            }
        }

        // 3. Fold the m-wide borders back into the interior so the Cartesian grid
        //    convolution is periodic over 2n (tomocupy `wrap`).
        for ty in 0..ng {
            for tx in 0..ng {
                if tx < m || tx >= 2 * nd + m || ty < m || ty >= 2 * nd + m {
                    let tx0 = (tx + 2 * nd - m) % (2 * nd);
                    let ty0 = (ty + 2 * nd - m) % (2 * nd);
                    let v = grid[ty * ng + tx];
                    grid[(ty0 + m) * ng + (tx0 + m)] += v;
                }
            }
        }

        // 4. Extract the interior 2n×2n block and inverse-2-D-FFT it, centred via
        //    the same shift-sign modulation before and after (tomocupy
        //    fftshiftc → ifft2d → fftshiftc; m is even so the border offset
        //    leaves the modulation phase unchanged).
        let mut inner = vec![Complex32::new(0.0, 0.0); nf * nf];
        for ry in 0..nf {
            for rx in 0..nf {
                inner[ry * nf + rx] = grid[(ry + m) * ng + (rx + m)];
            }
        }
        for ry in 0..nf {
            for rx in 0..nf {
                inner[ry * nf + rx] *= shift_sign(rx) * shift_sign(ry);
            }
        }
        fft.fft_2d(&mut inner, nf, nf, 1, true)?;
        for ry in 0..nf {
            for rx in 0..nf {
                inner[ry * nf + rx] *= shift_sign(rx) * shift_sign(ry);
            }
        }

        // 5. Gaussian deapodize (`divphi`), crop the central n×n, and apply the
        //    unit-disk mask (`circ`). The interior read offset is the symmetric
        //    central crop (col +nd/2, row +nd/2): tomocupy's `divphi` adds a `+1`
        //    row bias for its `-sin` projection geometry, but tomoxide's `+sin`
        //    convention (the y-sign above) makes the symmetric offset correct —
        //    verified against the phantom (a `+1` reintroduces a 1-px shift).
        for oy in 0..n {
            let ty = oy + crop; // pixel in the nd field of view
            let dy = ty as f32 / nd as f32 - 0.5;
            let inner_row = ty + nd / 2;
            let my = (ty as f32 - nd as f32 / 2.0) / nd as f32;
            for ox in 0..n {
                let tx = ox + crop;
                let dx = tx as f32 / nd as f32 - 0.5;
                let phi = (mu * (nd as f32) * (nd as f32) * (dx * dx + dy * dy)).exp()
                    / nang as f32
                    * phi_sign;
                let inner_col = tx + nd / 2;
                let v = inner[inner_row * nf + inner_col].re * phi;
                let mx = (tx as f32 - nd as f32 / 2.0) / nd as f32;
                let masked = if 4.0 * mx * mx + 4.0 * my * my < 1.0 {
                    v
                } else {
                    0.0
                };
                slab[[oy, ox]] = masked;
            }
        }
        Ok(())
    };

    // The backend owns the per-slice execution strategy (serial / rayon /
    // multi-GPU); every strategy yields the identical volume.
    fft.for_each_slice(&mut out, &process_row)?;
    Ok(out)
}
