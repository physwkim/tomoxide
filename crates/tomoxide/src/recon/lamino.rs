//! Fourier-based laminography reconstruction — tomocupy `LamFourierRec`.
//!
//! Faithful port of tomocupy's laminographic back-projection
//! (`reconstruction/lamfourierrec.py` + `backproj_lamfourier_parallel.py`, CUDA
//! `cfunc_usfft1d.cu`/`cfunc_usfft2d.cu`/`cfunc_fft2d.cu` and their
//! `kernels_*.cu`). Unlike the 2-D-slice-stacked analytic methods, laminography
//! is intrinsically 3-D: every tilted projection contributes to every output
//! voxel, so this has its own 3-D entry point ([`lamino`]) rather than going
//! through the per-slice `recon::recon` dispatch.
//!
//! The method (Nikitin et al., <https://arxiv.org/abs/2401.11101>) realizes the
//! adjoint Radon transform for a tilted rotation axis as three chained
//! unequally-spaced-FFT (USFFT) operators:
//!   1. [`fft2d_fwd`] — centered 2-D FFT of each (ramp-filtered) projection,
//!      i.e. its slice of the object's 3-D Fourier transform.
//!   2. [`usfft2d_adj`] — scatter those slices into the in-plane `(x,y)`
//!      frequency volume along the laminographic mapping `(θ, φ)` (Gaussian
//!      gridding).
//!   3. [`usfft1d_adj`] — scatter/transform along the tilt (`z`) axis to the
//!      real reconstructed volume.
//!
//! Each USFFT is a Gaussian-gridding operator (accuracy target `eps = 1e-3`,
//! half-width `m`, shape `mu`) identical in spirit to `fourierrec`'s
//! gather, with a `fwd`/`adj` pair that are exact transposes — verified by the
//! adjoint dot-product test `⟨A·f, g⟩ = ⟨f, A*·g⟩`.
//!
//! **Deviations from the CUDA source (both pure optimizations, not algorithm
//! changes).** tomocupy (a) streams the work in `deth`/`ntheta`/`n1` chunks with
//! double-buffering, and (b) exploits the Hermitian symmetry of a real
//! projection's spectrum to halve both detector axes (R2C: `detw → detw/2+1` at
//! `fft2d`, `deth → deth/2+1` at the `usfft2d → usfft1d` interface, with
//! flip-block bookkeeping inside the gather kernels). This CPU port drops both:
//! it processes whole arrays and carries the **full complex** spectra. The
//! gridding geometry (`take_x`, Gaussian weights, `wrap`, deapodization) and the
//! per-operator FFT normalization (cuFFT-style **unnormalized** inverse, applied
//! here by multiplying back the `1/N` that [`Fft`] divides out) are identical, so
//! the result is numerically equivalent for real input while letting every
//! operator be a clean, testable transpose. All FFTs go through the [`Fft`]
//! capability so the method composes onto any backend.
//!
//! No CUDA golden offline; verified by the per-operator adjoint tests, an
//! `fft2d_inv(fft2d_fwd) == id` round-trip, and a self-consistent
//! forward-project → reconstruct round-trip of a 3-D phantom (the matched
//! [`lamino_project`] forward model and [`lamino`] reconstruction recover it).

use crate::backend::Fft;
use crate::dtype::Complex32;
use crate::error::Result;
use std::f32::consts::PI;

/// Gridding accuracy target (tomocupy hardcodes `EPS = 1e-3`).
const EPS: f64 = 1e-3;

/// Centered-DFT modulation sign `1 − 2·((i+1) % 2)`: `−1` on even indices, `+1`
/// on odd (tomocupy's `fftshiftc`/`rfftshiftc` factor).
#[inline]
fn sign(i: usize) -> f32 {
    if i % 2 == 0 {
        -1.0
    } else {
        1.0
    }
}

/// Gaussian-USFFT half-width `m` and shape `mu` for a transform of size `n`
/// (tomocupy: `mu = -ln(eps)/(2n²)`, `m = ceil(2n/π·√(-mu·ln(eps)+(mu·n)²/4))`).
fn usfft_params(n: usize) -> (usize, f32) {
    let nf = n as f64;
    let neg_log_eps = -EPS.ln();
    let mu = neg_log_eps / (2.0 * nf * nf);
    let inside = mu * neg_log_eps + (mu * nf) * (mu * nf) / 4.0;
    let m = (2.0 * nf / std::f64::consts::PI * inside.sqrt()).ceil() as usize;
    (m, mu as f32)
}

// ===========================================================================
// Centered-FFT helpers (modulate → FFT → modulate) on the oversampled grid.
//
// The cuFFT inverse is UNNORMALIZED; tomoxide's `Fft::fft_*(inverse=true)`
// divides by N. To match tomocupy's numerics AND make the inverse the exact
// conjugate-transpose `W^H` of the unnormalized forward `W`, these helpers
// multiply the `1/N` back in on the inverse path.
// ===========================================================================

/// Centered 1-D FFT along axis-0 of a `[2n+2m, batch]` array, over the interior
/// window `[m, m+2n)`. `batch` lines are transformed independently. Mirrors
/// tomocupy's `fftshiftc1d → cufftExec(window) → fftshiftc1d`; the sign
/// modulation uses the *global* grid index `m+i`.
fn centered_fft_axis0(
    fdee: &mut [Complex32],
    n: usize,
    m: usize,
    batch: usize,
    inverse: bool,
    fft: &dyn Fft,
) -> Result<()> {
    let len = 2 * n;
    let mut lines = vec![Complex32::new(0.0, 0.0); batch * len];
    for i in 0..len {
        let s = sign(m + i);
        let src = (m + i) * batch;
        for b in 0..batch {
            lines[b * len + i] = fdee[src + b] * s;
        }
    }
    fft.fft_1d(&mut lines, len, batch, inverse)?;
    // cuFFT-style unnormalized inverse: undo tomoxide's 1/len.
    let renorm = if inverse { len as f32 } else { 1.0 };
    for i in 0..len {
        let s = sign(m + i) * renorm;
        let dst = (m + i) * batch;
        for b in 0..batch {
            fdee[dst + b] = lines[b * len + i] * s;
        }
    }
    Ok(())
}

/// Centered 2-D FFT over the interior window `[m1, m1+2n1) × [m0, m0+2n0)` of an
/// oversampled grid laid out `[batch, gy, gx]` with `gx = 2n0+2m0`,
/// `gy = 2n1+2m1`. Mirrors tomocupy's `fftshiftc2d → cufftExec2d(window) →
/// fftshiftc2d`; sign modulation `sign(m0+ix)·sign(m1+iy)` uses global indices.
#[allow(clippy::too_many_arguments)]
fn centered_fft2d(
    fdee: &mut [Complex32],
    n0: usize,
    n1: usize,
    m0: usize,
    m1: usize,
    batch: usize,
    inverse: bool,
    fft: &dyn Fft,
) -> Result<()> {
    let gx = 2 * n0 + 2 * m0;
    let gy = 2 * n1 + 2 * m1;
    let (wx, wy) = (2 * n0, 2 * n1);
    let mut win = vec![Complex32::new(0.0, 0.0); batch * wy * wx];
    for b in 0..batch {
        for iy in 0..wy {
            let sy = sign(m1 + iy);
            for ix in 0..wx {
                let s = sign(m0 + ix) * sy;
                let src = (m0 + ix) + (m1 + iy) * gx + b * gx * gy;
                win[(b * wy + iy) * wx + ix] = fdee[src] * s;
            }
        }
    }
    fft.fft_2d(&mut win, wy, wx, batch, inverse)?;
    let renorm = if inverse { (wx * wy) as f32 } else { 1.0 };
    for b in 0..batch {
        for iy in 0..wy {
            let sy = sign(m1 + iy);
            for ix in 0..wx {
                let s = sign(m0 + ix) * sy * renorm;
                let dst = (m0 + ix) + (m1 + iy) * gx + b * gx * gy;
                fdee[dst] = win[(b * wy + iy) * wx + ix] * s;
            }
        }
    }
    Ok(())
}

// ===========================================================================
// fft2d — centered 2-D FFT of each projection (tomocupy cfunc_fft2d), full
// complex (the R2C half is an optimization we drop; see module docs).
// ===========================================================================

/// Forward: centered 2-D FFT of each projection. `proj` is `[ntheta, deth,
/// detw]` real, row-major; returns the full `[ntheta, deth, detw]` complex
/// spectrum scaled by `1/(deth·detw)` (tomocupy `fft2d_fwd`: rfftshiftc → FFT →
/// irfftshiftc → mulc).
pub fn fft2d_fwd(
    proj: &[f32],
    ntheta: usize,
    deth: usize,
    detw: usize,
    fft: &dyn Fft,
) -> Result<Vec<Complex32>> {
    // rfftshiftc2d: modulate real input by sign(tx)·sign(ty), embed as complex.
    let mut buf = vec![Complex32::new(0.0, 0.0); ntheta * deth * detw];
    for tz in 0..ntheta {
        for ty in 0..deth {
            let sy = sign(ty);
            for tx in 0..detw {
                let s = sign(tx) * sy;
                buf[(tz * deth + ty) * detw + tx] =
                    Complex32::new(proj[(tz * deth + ty) * detw + tx] * s, 0.0);
            }
        }
    }
    fft.fft_2d(&mut buf, deth, detw, ntheta, false)?;
    // irfftshiftc2d (negated sign) + mulc by 1/(deth·detw).
    let scale = 1.0 / (deth * detw) as f32;
    for tz in 0..ntheta {
        for ty in 0..deth {
            let sy = sign(ty);
            for tx in 0..detw {
                let s = -sign(tx) * sy * scale;
                buf[(tz * deth + ty) * detw + tx] *= s;
            }
        }
    }
    Ok(buf)
}

/// Centered 2-D inverse FFT that **inverts** [`fft2d_fwd`]: `[ntheta, deth,
/// detw]` complex → `[ntheta, deth, detw]` real, with `fft2d_inv(fft2d_fwd(p))
/// == p`. Undoes the post-FFT centering, inverse 2-D FFT, undoes the pre-FFT
/// centering, applies the `deth·detw` renorm, takes the real part. Used as the
/// per-projection inverse FFT in the forward laminography model
/// ([`lamino_project`]).
pub fn fft2d_inv(
    g: &[Complex32],
    ntheta: usize,
    deth: usize,
    detw: usize,
    fft: &dyn Fft,
) -> Result<Vec<f32>> {
    // Undo irfftshiftc (×−sign·sign).
    let mut buf = vec![Complex32::new(0.0, 0.0); ntheta * deth * detw];
    for tz in 0..ntheta {
        for ty in 0..deth {
            let sy = sign(ty);
            for tx in 0..detw {
                let s = -sign(tx) * sy;
                buf[(tz * deth + ty) * detw + tx] = g[(tz * deth + ty) * detw + tx] * s;
            }
        }
    }
    fft.fft_2d(&mut buf, deth, detw, ntheta, true)?;
    // Undo rfftshiftc (×sign·sign) and renormalize (tomocupy's R2C carried a
    // 1/(deth·detw); the inverse multiplies it back). Take the real part.
    let renorm = (deth * detw) as f32;
    let mut out = vec![0.0f32; ntheta * deth * detw];
    for tz in 0..ntheta {
        for ty in 0..deth {
            let sy = sign(ty);
            for tx in 0..detw {
                let s = sign(tx) * sy * renorm;
                out[(tz * deth + ty) * detw + tx] = buf[(tz * deth + ty) * detw + tx].re * s;
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// usfft1d — Gaussian USFFT along the tilt (z) axis (tomocupy cfunc_usfft1d),
// full complex (no R2C half). Dimensions follow tomocupy's python→cuda reorder
// for `cfunc_usfft1d(rh, n1, n, nz)`: n0=n(detw/x), n1=n1(in-plane y), n2=rh
// (reconstructed depth), deth=nz. The oversampled depth grid spans 2·n2+2·m2.
// ===========================================================================

/// Sample positions along the depth-frequency axis: `z[t] = (t−deth/2)/deth·
/// sin(phi)` (tomocupy `take_x`).
fn usfft1d_z(deth: usize, phi: f32) -> Vec<f32> {
    let sp = phi.sin();
    (0..deth)
        .map(|t| (t as f32 - deth as f32 / 2.0) / deth as f32 * sp)
        .collect()
}

/// Adjoint USFFT-1D: scatter the depth-frequency data `g` (`[n1, deth, n0]`
/// complex) onto the reconstructed depth axis and transform to the real volume
/// `f` (`[n1, n2, n0]`). `n0 = detw`, `n2 = rh`.
pub fn usfft1d_adj(
    g: &[Complex32],
    n0: usize,
    n1: usize,
    n2: usize,
    deth: usize,
    phi: f32,
    fft: &dyn Fft,
) -> Result<Vec<f32>> {
    let (m2, mu2) = usfft_params(n2);
    let z = usfft1d_z(deth, phi);
    let ng = 2 * n2 + 2 * m2;
    let mut fdee = vec![Complex32::new(0.0, 0.0); ng * n1 * n0];

    // gather1d (adj): scatter each g sample into the oversampled grid.
    let wscale = (PI / (mu2 * n0 as f32)).sqrt();
    for tz in 0..deth {
        let z0 = z[tz];
        let base = (2.0 * n2 as f32 * z0).floor() as isize - m2 as isize;
        for ty in 0..n1 {
            for tx in 0..n0 {
                let g0 = g[tx + tz * n0 + ty * n0 * deth];
                for i2 in 0..2 * m2 + 1 {
                    let ell2 = base + i2 as isize;
                    let w2 = ell2 as f32 / (2.0 * n2 as f32) - z0;
                    let w = wscale * (-PI * PI / mu2 * w2 * w2).exp();
                    let a = (n2 as isize + m2 as isize + ell2) as usize;
                    fdee[tx + ty * n0 + a * n0 * n1] += g0 * w;
                }
            }
        }
    }

    // wrap1d (adj): fold the m2-wide borders back into the interior.
    for a in 0..ng {
        if a < m2 || a >= 2 * n2 + m2 {
            let a0 = (a + 2 * n2 - m2) % (2 * n2);
            for ty in 0..n1 {
                for tx in 0..n0 {
                    let v = fdee[tx + ty * n0 + a * n0 * n1];
                    fdee[tx + ty * n0 + (a0 + m2) * n0 * n1] += v;
                }
            }
        }
    }

    // Centered inverse FFT along the depth-grid axis over the window [m2, m2+2n2).
    centered_fft_axis0(&mut fdee, n2, m2, n0 * n1, true, fft)?;

    // divker1d (adj): deapodize, take real part → f[n1, n2, n0].
    let mut f = vec![0.0f32; n1 * n2 * n0];
    for tz in 0..n2 {
        let ker = (-mu2 * (tz as f32 - n2 as f32 / 2.0).powi(2)).exp();
        let a = tz + n2 / 2 + m2;
        for ty in 0..n1 {
            for tx in 0..n0 {
                let g_ind = tx + ty * n0 + a * n0 * n1;
                f[tx + tz * n0 + ty * n0 * n2] = fdee[g_ind].re / ker / (2.0 * n2 as f32);
            }
        }
    }
    Ok(f)
}

/// Forward USFFT-1D (exact transpose of [`usfft1d_adj`]): real volume `f`
/// (`[n1, n2, n0]`) → depth-frequency data `g` (`[n1, deth, n0]`). Used by the
/// forward laminography model ([`lamino_project`]).
pub fn usfft1d_fwd(
    f: &[f32],
    n0: usize,
    n1: usize,
    n2: usize,
    deth: usize,
    phi: f32,
    fft: &dyn Fft,
) -> Result<Vec<Complex32>> {
    let (m2, mu2) = usfft_params(n2);
    let z = usfft1d_z(deth, phi);
    let ng = 2 * n2 + 2 * m2;
    let mut fdee = vec![Complex32::new(0.0, 0.0); ng * n1 * n0];

    // divker1d (fwd): f → oversampled grid interior.
    for tz in 0..n2 {
        let ker = (-mu2 * (tz as f32 - n2 as f32 / 2.0).powi(2)).exp();
        let a = tz + n2 / 2 + m2;
        for ty in 0..n1 {
            for tx in 0..n0 {
                let v = f[tx + tz * n0 + ty * n0 * n2] / ker / (2.0 * n2 as f32);
                fdee[tx + ty * n0 + a * n0 * n1] = Complex32::new(v, 0.0);
            }
        }
    }

    // Centered forward FFT along the depth-grid axis.
    centered_fft_axis0(&mut fdee, n2, m2, n0 * n1, false, fft)?;

    // wrap1d (fwd): copy interior into the borders.
    for a in 0..ng {
        if a < m2 || a >= 2 * n2 + m2 {
            let a0 = (a + 2 * n2 - m2) % (2 * n2);
            for ty in 0..n1 {
                for tx in 0..n0 {
                    let v = fdee[tx + ty * n0 + (a0 + m2) * n0 * n1];
                    fdee[tx + ty * n0 + a * n0 * n1] = v;
                }
            }
        }
    }

    // gather1d (fwd): interpolate the oversampled grid at the sample positions.
    let wscale = (PI / (mu2 * n0 as f32)).sqrt();
    let mut g = vec![Complex32::new(0.0, 0.0); n1 * deth * n0];
    for tz in 0..deth {
        let z0 = z[tz];
        let base = (2.0 * n2 as f32 * z0).floor() as isize - m2 as isize;
        for ty in 0..n1 {
            for tx in 0..n0 {
                let mut acc = Complex32::new(0.0, 0.0);
                for i2 in 0..2 * m2 + 1 {
                    let ell2 = base + i2 as isize;
                    let w2 = ell2 as f32 / (2.0 * n2 as f32) - z0;
                    let w = wscale * (-PI * PI / mu2 * w2 * w2).exp();
                    let a = (n2 as isize + m2 as isize + ell2) as usize;
                    acc += fdee[tx + ty * n0 + a * n0 * n1] * w;
                }
                g[tx + tz * n0 + ty * n0 * deth] = acc;
            }
        }
    }
    Ok(g)
}

// ===========================================================================
// usfft2d — Gaussian USFFT scattering each (θ, kx) frequency line into the
// in-plane (x, y) frequency volume (tomocupy cfunc_usfft2d), full complex.
// Dimensions: n0=detw(x-freq out), n1=in-plane y-freq out, deth=detw-freq count
// passed as the FFT batch over ky (deth-frequency) slices. For `cfunc_usfft2d(
// deth, n, n, ntheta, detw, deth)` the kernel uses n0=n(=detw=n2), n1=n, the
// per-ky batch is `nky`, detw=detw, ntheta=ntheta.
// ===========================================================================

/// In-plane frequency sample positions for the laminographic mapping
/// (tomocupy `take_x`): for projection `θ` and detector frequency `(kx, ky)`,
/// `x = ku·cosθ + kv·sinθ·cosφ`, `y = ku·sinθ − kv·cosθ·cosφ`, clamped to
/// `[−0.5+1e-5, 0.5−1e-5]`, with `ku = (kx−detw/2)/detw`, `kv = (ky−nky/2)/nky`.
fn usfft2d_xy(
    theta: &[f32],
    phi: f32,
    ntheta: usize,
    nky: usize,
    detw: usize,
) -> (Vec<f32>, Vec<f32>) {
    let cph = phi.cos();
    let lim = 0.5 - 1e-5;
    let mut xs = vec![0.0f32; ntheta * nky * detw];
    let mut ys = vec![0.0f32; ntheta * nky * detw];
    for (tz, &th) in theta.iter().enumerate().take(ntheta) {
        let (st, ct) = (th.sin(), th.cos());
        for ky in 0..nky {
            let kv = (ky as f32 - nky as f32 / 2.0) / nky as f32;
            for kx in 0..detw {
                let ku = (kx as f32 - detw as f32 / 2.0) / detw as f32;
                let x = (ku * ct + kv * st * cph).clamp(-lim, lim);
                let y = (ku * st - kv * ct * cph).clamp(-lim, lim);
                let ind = kx + ky * detw + tz * detw * nky;
                xs[ind] = x;
                ys[ind] = y;
            }
        }
    }
    (xs, ys)
}

/// Adjoint USFFT-2D: scatter the 2-D frequency data `g` (`[ntheta, nky, detw]`
/// complex) into the in-plane frequency volume `f` (`[n1, nky, n0]` complex,
/// summed over θ). `n0 = detw` (x-frequency), `n1` (y-frequency), `nky` is the
/// number of depth-frequency (`ky`) slices carried as the batch.
#[allow(clippy::too_many_arguments)]
pub fn usfft2d_adj(
    g: &[Complex32],
    n0: usize,
    n1: usize,
    nky: usize,
    ntheta: usize,
    detw: usize,
    theta: &[f32],
    phi: f32,
    fft: &dyn Fft,
) -> Result<Vec<Complex32>> {
    let (m0, mu0) = usfft_params(n0);
    let (m1, mu1) = usfft_params(n1);
    let gx = 2 * n0 + 2 * m0;
    let gy = 2 * n1 + 2 * m1;
    let (xs, ys) = usfft2d_xy(theta, phi, ntheta, nky, detw);
    let mut fdee = vec![Complex32::new(0.0, 0.0); nky * gy * gx];

    // gather2d (adj): scatter each (θ, ky, kx) sample into the (x,y) grid.
    let wpre = PI / (mu0 * mu1 * ntheta as f32).sqrt();
    for tz in 0..ntheta {
        for ky in 0..nky {
            for kx in 0..detw {
                let ind = kx + ky * detw + tz * detw * nky;
                let (x0, y0) = (xs[ind], ys[ind]);
                let g0 = g[ind];
                let base0 = (2.0 * n0 as f32 * x0).floor() as isize - m0 as isize;
                let base1 = (2.0 * n1 as f32 * y0).floor() as isize - m1 as isize;
                for i1 in 0..2 * m1 + 1 {
                    let ell1 = base1 + i1 as isize;
                    let w1 = ell1 as f32 / (2.0 * n1 as f32) - y0;
                    let ew1 = (-PI * PI / mu1 * w1 * w1).exp();
                    let yg = (n1 as isize + m1 as isize + ell1) as usize;
                    for i0 in 0..2 * m0 + 1 {
                        let ell0 = base0 + i0 as isize;
                        let w0 = ell0 as f32 / (2.0 * n0 as f32) - x0;
                        let w = wpre * (-PI * PI / mu0 * w0 * w0).exp() * ew1;
                        let xg = (n0 as isize + m0 as isize + ell0) as usize;
                        fdee[xg + yg * gx + ky * gx * gy] += g0 * w;
                    }
                }
            }
        }
    }

    // wrap2d (adj): fold the borders back into the interior in x and y.
    wrap2d(&mut fdee, n0, n1, nky, m0, m1, true);

    // Centered inverse 2-D FFT over the (x,y) window, per ky slice.
    centered_fft2d(&mut fdee, n0, n1, m0, m1, nky, true, fft)?;

    // divker2d (adj): deapodize → f[n1, nky, n0] (note the n1−ty−1 y-flip).
    let mut f = vec![Complex32::new(0.0, 0.0); n1 * nky * n0];
    for ky in 0..nky {
        for ty in 0..n1 {
            let yg = (n1 - ty - 1) + n1 / 2 + m1;
            for tx in 0..n0 {
                let ker = (-mu0 * (tx as f32 - n0 as f32 / 2.0).powi(2)
                    - mu1 * (ty as f32 - n1 as f32 / 2.0).powi(2))
                .exp();
                let xg = tx + n0 / 2 + m0;
                let g_ind = xg + yg * gx + ky * gx * gy;
                f[tx + ky * n0 + ty * n0 * nky] = fdee[g_ind] / ker / (n0 * n1) as f32;
            }
        }
    }
    Ok(f)
}

/// Forward USFFT-2D (exact transpose of [`usfft2d_adj`]): in-plane frequency
/// volume `f` (`[n1, nky, n0]` complex) → 2-D frequency data `g`
/// (`[ntheta, nky, detw]` complex). Used by the forward laminography model
/// ([`lamino_project`]).
#[allow(clippy::too_many_arguments)]
pub fn usfft2d_fwd(
    f: &[Complex32],
    n0: usize,
    n1: usize,
    nky: usize,
    ntheta: usize,
    detw: usize,
    theta: &[f32],
    phi: f32,
    fft: &dyn Fft,
) -> Result<Vec<Complex32>> {
    let (m0, mu0) = usfft_params(n0);
    let (m1, mu1) = usfft_params(n1);
    let gx = 2 * n0 + 2 * m0;
    let gy = 2 * n1 + 2 * m1;
    let (xs, ys) = usfft2d_xy(theta, phi, ntheta, nky, detw);
    let mut fdee = vec![Complex32::new(0.0, 0.0); nky * gy * gx];

    // divker2d (fwd): f → oversampled grid interior (n1−ty−1 y-flip).
    for ky in 0..nky {
        for ty in 0..n1 {
            let yg = (n1 - ty - 1) + n1 / 2 + m1;
            for tx in 0..n0 {
                let ker = (-mu0 * (tx as f32 - n0 as f32 / 2.0).powi(2)
                    - mu1 * (ty as f32 - n1 as f32 / 2.0).powi(2))
                .exp();
                let xg = tx + n0 / 2 + m0;
                let v = f[tx + ky * n0 + ty * n0 * nky] / ker / (n0 * n1) as f32;
                fdee[xg + yg * gx + ky * gx * gy] = v;
            }
        }
    }

    // Centered forward 2-D FFT over the (x,y) window, per ky slice.
    centered_fft2d(&mut fdee, n0, n1, m0, m1, nky, false, fft)?;

    // wrap2d (fwd): copy interior into the borders.
    wrap2d(&mut fdee, n0, n1, nky, m0, m1, false);

    // gather2d (fwd): interpolate the grid at the sample positions, summing
    // each (θ, ky, kx) from its (x,y) neighborhood.
    let wpre = PI / (mu0 * mu1 * ntheta as f32).sqrt();
    let mut g = vec![Complex32::new(0.0, 0.0); ntheta * nky * detw];
    for tz in 0..ntheta {
        for ky in 0..nky {
            for kx in 0..detw {
                let ind = kx + ky * detw + tz * detw * nky;
                let (x0, y0) = (xs[ind], ys[ind]);
                let base0 = (2.0 * n0 as f32 * x0).floor() as isize - m0 as isize;
                let base1 = (2.0 * n1 as f32 * y0).floor() as isize - m1 as isize;
                let mut acc = Complex32::new(0.0, 0.0);
                for i1 in 0..2 * m1 + 1 {
                    let ell1 = base1 + i1 as isize;
                    let w1 = ell1 as f32 / (2.0 * n1 as f32) - y0;
                    let ew1 = (-PI * PI / mu1 * w1 * w1).exp();
                    let yg = (n1 as isize + m1 as isize + ell1) as usize;
                    for i0 in 0..2 * m0 + 1 {
                        let ell0 = base0 + i0 as isize;
                        let w0 = ell0 as f32 / (2.0 * n0 as f32) - x0;
                        let w = wpre * (-PI * PI / mu0 * w0 * w0).exp() * ew1;
                        let xg = (n0 as isize + m0 as isize + ell0) as usize;
                        acc += fdee[xg + yg * gx + ky * gx * gy] * w;
                    }
                }
                g[ind] = acc;
            }
        }
    }
    Ok(g)
}

/// Shared `wrap2d`: fold (adj) or copy (fwd) the `m`-wide borders of the
/// oversampled `[nky, gy, gx]` grid in both x and y (tomocupy `wrap2d`).
fn wrap2d(
    fdee: &mut [Complex32],
    n0: usize,
    n1: usize,
    nky: usize,
    m0: usize,
    m1: usize,
    adj: bool,
) {
    let gx = 2 * n0 + 2 * m0;
    let gy = 2 * n1 + 2 * m1;
    for ky in 0..nky {
        for ty in 0..gy {
            for tx in 0..gx {
                if tx < m0 || tx >= 2 * n0 + m0 || ty < m1 || ty >= 2 * n1 + m1 {
                    let tx0 = (tx + 2 * n0 - m0) % (2 * n0);
                    let ty0 = (ty + 2 * n1 - m1) % (2 * n1);
                    let id1 = tx + ty * gx + ky * gx * gy;
                    let id2 = (tx0 + m0) + (ty0 + m1) * gx + ky * gx * gy;
                    if adj {
                        let v = fdee[id1];
                        fdee[id2] += v;
                    } else {
                        fdee[id1] = fdee[id2];
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Ramp filtering + 3-D entry points.
// ===========================================================================

/// Ramp-filter each projection along the detector-width (`detw`) axis, with
/// `ne = 2·detw` edge-padding (tomocupy filters projections, not sinograms;
/// this uses a plain `|f|` ramp with `center = detw/2`, no extra apodization
/// window or sub-pixel shift).
fn ramp_filter_detw(
    proj: &mut [f32],
    ntheta: usize,
    deth: usize,
    detw: usize,
    fft: &dyn Fft,
) -> Result<()> {
    let ne = 2 * detw;
    let pad = (ne - detw) / 2;
    let nlines = ntheta * deth;
    // Edge-padded complex lines [nlines, ne].
    let mut buf = vec![Complex32::new(0.0, 0.0); nlines * ne];
    for l in 0..nlines {
        let row = &proj[l * detw..(l + 1) * detw];
        for (k, slot) in buf[l * ne..(l + 1) * ne].iter_mut().enumerate() {
            let v = if k < pad {
                row[0]
            } else if k < pad + detw {
                row[k - pad]
            } else {
                row[detw - 1]
            };
            *slot = Complex32::new(v, 0.0);
        }
    }
    fft.fft_1d(&mut buf, ne, nlines, false)?;
    // |f| ramp on centered frequencies (DC..Nyquist..−1), magnitude in cycles.
    let ramp: Vec<f32> = (0..ne)
        .map(|k| {
            let f = if k <= ne / 2 {
                k as f32
            } else {
                (ne - k) as f32
            };
            f / ne as f32
        })
        .collect();
    for l in 0..nlines {
        for k in 0..ne {
            buf[l * ne + k] *= ramp[k];
        }
    }
    fft.fft_1d(&mut buf, ne, nlines, true)?;
    for l in 0..nlines {
        for k in 0..detw {
            proj[l * detw + k] = buf[l * ne + pad + k].re;
        }
    }
    Ok(())
}

/// `phi = π/2 + lamino_angle·π/180` (tomocupy's tilt convention).
#[inline]
fn lamino_phi(lamino_angle_deg: f32) -> f32 {
    std::f32::consts::FRAC_PI_2 + lamino_angle_deg / 180.0 * PI
}

/// Reconstruct a 3-D laminographic volume from tilted parallel-beam projections
/// — tomocupy `BackprojLamFourierParallel.rec_lam`.
///
/// `proj` is `[nproj, nz, n]` row-major (projection, detector-row, detector-col),
/// with a square `n × n` in-plane field; `theta` are the `nproj` rotation angles
/// (radians); `lamino_angle_deg` is the tilt of the rotation axis from the
/// beam-perpendicular plane; `rh` is the number of reconstructed depth slices.
/// Returns the volume `[rh, n, n]` (depth, y, x). All FFTs use the supplied
/// [`Fft`] backend.
pub fn lamino(
    proj: &[f32],
    theta: &[f32],
    lamino_angle_deg: f32,
    n: usize,
    rh: usize,
    fft: &dyn Fft,
) -> Result<Vec<f32>> {
    let nproj = theta.len();
    let (detw, deth) = (n, proj.len() / (nproj * n));
    assert_eq!(
        proj.len(),
        nproj * deth * detw,
        "proj shape != [nproj, nz, n]"
    );
    let phi = lamino_phi(lamino_angle_deg);

    // FBP ramp filter on projections, then the three chained USFFT operators.
    let mut filtered = proj.to_vec();
    ramp_filter_detw(&mut filtered, nproj, deth, detw, fft)?;
    let p22 = fft2d_fwd(&filtered, nproj, deth, detw, fft)?;
    let p11 = usfft2d_adj(&p22, detw, n, deth, nproj, detw, theta, phi, fft)?;
    let p00 = usfft1d_adj(&p11, detw, n, rh, deth, phi, fft)?;

    // copyTransposed: [n1, rh, n2] → [rh, n1, n2].
    let (n1, n2) = (n, n);
    let mut vol = vec![0.0f32; rh * n1 * n2];
    for ty in 0..n1 {
        for tz in 0..rh {
            for tx in 0..n2 {
                vol[(tz * n1 + ty) * n2 + tx] = p00[(ty * rh + tz) * n2 + tx];
            }
        }
    }
    Ok(vol)
}

/// Forward laminography projector — the matched transpose of [`lamino`] (minus
/// the recon-side ramp filter), used to generate self-consistent test data.
/// `vol` is `[rh, n, n]` (depth, y, x); returns projections `[nproj, nz, n]`.
pub fn lamino_project(
    vol: &[f32],
    theta: &[f32],
    lamino_angle_deg: f32,
    n: usize,
    nz: usize,
    fft: &dyn Fft,
) -> Result<Vec<f32>> {
    let nproj = theta.len();
    let rh = vol.len() / (n * n);
    assert_eq!(vol.len(), rh * n * n, "vol shape != [rh, n, n]");
    let (detw, deth) = (n, nz);
    let phi = lamino_phi(lamino_angle_deg);

    // copyTransposed: [rh, n1, n2] → [n1, rh, n2].
    let (n1, n2) = (n, n);
    let mut p00 = vec![0.0f32; n1 * rh * n2];
    for tz in 0..rh {
        for ty in 0..n1 {
            for tx in 0..n2 {
                p00[(ty * rh + tz) * n2 + tx] = vol[(tz * n1 + ty) * n2 + tx];
            }
        }
    }
    let p11 = usfft1d_fwd(&p00, detw, n, rh, deth, phi, fft)?;
    let p22 = usfft2d_fwd(&p11, detw, n, deth, nproj, detw, theta, phi, fft)?;
    fft2d_inv(&p22, nproj, deth, detw, fft)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;

    // Deterministic pseudo-random fill in [-1, 1] (no rand dep; varies by seed).
    fn fill(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn cfill(seed: u64, n: usize) -> Vec<Complex32> {
        let re = fill(seed, n);
        let im = fill(seed ^ 0xABCDEF, n);
        re.into_iter()
            .zip(im)
            .map(|(r, i)| Complex32::new(r, i))
            .collect()
    }

    fn cdot(a: &[Complex32], b: &[Complex32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x.re * y.re + x.im * y.im) as f64)
            .sum()
    }

    fn rdot(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (*x * *y) as f64).sum()
    }

    #[test]
    fn fft2d_inv_inverts_fwd() {
        let cpu = CpuBackend::new();
        let (ntheta, deth, detw) = (3, 8, 8);
        let p = fill(1, ntheta * deth * detw);
        let g = fft2d_fwd(&p, ntheta, deth, detw, &cpu).unwrap();
        let r = fft2d_inv(&g, ntheta, deth, detw, &cpu).unwrap();
        let err: f32 = p
            .iter()
            .zip(&r)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("fft2d inv round-trip max abs err = {err:.2e}");
        assert!(err < 1e-4, "fft2d_inv does not invert fft2d_fwd: {err:.2e}");
    }

    #[test]
    fn usfft1d_fwd_adj_are_transposes() {
        let cpu = CpuBackend::new();
        let (n0, n1, n2, deth) = (6, 5, 8, 8);
        let f = fill(3, n1 * n2 * n0);
        let g = cfill(4, n1 * deth * n0);
        let phi = std::f32::consts::FRAC_PI_2 + 0.3;
        let af = usfft1d_fwd(&f, n0, n1, n2, deth, phi, &cpu).unwrap();
        let atg = usfft1d_adj(&g, n0, n1, n2, deth, phi, &cpu).unwrap();
        let lhs = cdot(&af, &g);
        let rhs = rdot(&f, &atg);
        let rel = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-9);
        eprintln!("usfft1d adjoint: lhs={lhs:.6} rhs={rhs:.6} rel={rel:.2e}");
        assert!(rel < 1e-4, "usfft1d fwd/adj not transposes: rel={rel:.2e}");
    }

    #[test]
    fn usfft2d_fwd_adj_are_transposes() {
        let cpu = CpuBackend::new();
        let (n0, n1, nky, ntheta, detw) = (6, 5, 4, 7, 6);
        let theta: Vec<f32> = (0..ntheta).map(|i| i as f32 / ntheta as f32 * PI).collect();
        let phi = std::f32::consts::FRAC_PI_2 + 0.35;
        let f = cfill(5, n1 * nky * n0);
        let g = cfill(6, ntheta * nky * detw);
        let af = usfft2d_fwd(&f, n0, n1, nky, ntheta, detw, &theta, phi, &cpu).unwrap();
        let atg = usfft2d_adj(&g, n0, n1, nky, ntheta, detw, &theta, phi, &cpu).unwrap();
        let lhs = cdot(&af, &g);
        let rhs = cdot(&f, &atg);
        let rel = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-9);
        eprintln!("usfft2d adjoint: lhs={lhs:.6} rhs={rhs:.6} rel={rel:.2e}");
        assert!(rel < 1e-4, "usfft2d fwd/adj not transposes: rel={rel:.2e}");
    }

    fn pearson(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().map(|&v| v as f64).sum::<f64>() / n;
        let mb = b.iter().map(|&v| v as f64).sum::<f64>() / n;
        let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
        for (&x, &y) in a.iter().zip(b) {
            let (dx, dy) = (x as f64 - ma, y as f64 - mb);
            cov += dx * dy;
            va += dx * dx;
            vb += dy * dy;
        }
        cov / (va.sqrt() * vb.sqrt()).max(1e-12)
    }

    #[test]
    fn lamino_roundtrip_recovers_phantom() {
        // Forward-project a 3-D phantom through the matched forward model, then
        // reconstruct: the laminographic geometry (take_x) is exercised end to
        // end, so a high correlation confirms the gridding is geometrically
        // correct (the per-operator adjoint tests only confirm the transpose).
        let cpu = CpuBackend::new();
        let (n, rh, nz) = (16usize, 8usize, 16usize);
        let nproj = 64usize;
        let lamino_angle = 20.0f32;
        let theta: Vec<f32> = (0..nproj)
            .map(|i| i as f32 / nproj as f32 * 2.0 * PI)
            .collect();

        // Phantom [rh, n, n]: two offset solid spheres.
        let mut vol = vec![0.0f32; rh * n * n];
        let sphere = |vol: &mut [f32], cz: f32, cy: f32, cx: f32, r: f32, val: f32| {
            for z in 0..rh {
                for y in 0..n {
                    for x in 0..n {
                        let d2 = (z as f32 - cz).powi(2)
                            + (y as f32 - cy).powi(2)
                            + (x as f32 - cx).powi(2);
                        if d2 <= r * r {
                            vol[(z * n + y) * n + x] = val;
                        }
                    }
                }
            }
        };
        sphere(&mut vol, 3.0, 6.0, 6.0, 2.5, 1.0);
        sphere(&mut vol, 5.0, 10.0, 9.0, 1.8, 0.6);

        let proj = lamino_project(&vol, &theta, lamino_angle, n, nz, &cpu).unwrap();
        let rec = lamino(&proj, &theta, lamino_angle, n, rh, &cpu).unwrap();

        let corr = pearson(&rec, &vol);
        eprintln!("lamino round-trip corr = {corr:.4}");
        assert!(corr > 0.7, "lamino round-trip corr too low: {corr:.4}");
    }
}
