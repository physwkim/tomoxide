//! Phase retrieval (ports tomopy `prep/phase.py` + tomocupy
//! `processing/retrieve_phase.py`). Paganin, generalized Paganin (`Gpaganin`),
//! and Farago single-step retrieval are implemented. See `docs/PORTING.md` §D.

use ndarray::ArrayViewMut2;

use crate::backend::{Backend, Fft};
use crate::data::{Layout, Tomo};
use crate::dtype::Complex32;
use crate::error::{Error, Result};
use crate::params::PhaseMethod;

// tomopy's literal constants (tomopy/prep/phase.py:70-73). PHASE_PI is tomopy's
// exact truncated literal, NOT std PI — it feeds `ceil(π·…)` in the pad-width
// calc, where the more precise std value could round to a different integer and
// diverge from tomopy. Required for parity; clippy's approx_constant is a false
// positive here.
#[allow(clippy::approx_constant)]
const PHASE_PI: f64 = 3.14159265359;
const PLANCK_CONSTANT: f64 = 6.58211928e-19; // keV·s
const SPEED_OF_LIGHT: f64 = 299_792_458e2; // cm/s

/// Single-step phase retrieval on a projection stack.
///
/// Paganin params (`pixel_size` cm, `dist` cm, `energy` keV, `alpha`) live in
/// [`PhaseMethod::Paganin`]. The Paganin path is a Fourier low-pass filter on
/// each (zero/edge-padded) radiograph — projector-independent, so it matches
/// tomopy numerically. tomopy's `pad=True` default is always used (its
/// `pad=False` branch is broken upstream and there is no `pad` parameter here).
pub fn retrieve_phase(
    data: &mut Tomo<f32>,
    method: PhaseMethod,
    backend: &dyn Backend,
) -> Result<()> {
    match method {
        PhaseMethod::None => Ok(()),
        PhaseMethod::Paganin {
            pixel_size,
            dist,
            energy,
            alpha,
        } => {
            let fft = require_fft(backend)?;
            // Standard Paganin: `1/(λ·dist·(ix²+iy²)/(4π) + α)` over the squared
            // reciprocal grid (tomopy/tomocupy `_paganin_filter_factor`). The
            // filter is well-conditioned, so the f64 grid matches tomopy within
            // the f32 floor.
            let (ps, dist_f, alpha_f) = (pixel_size as f64, dist as f64, alpha as f64);
            let wl = wavelength(energy as f64);
            run_phase(data, pixel_size, dist, energy, fft, move |nx, ny| {
                pointwise_filter(nx, ny, ps, |ix, iy| {
                    let w2 = ix * ix + iy * iy;
                    (1.0 / (wl * dist_f * w2 / (4.0 * PHASE_PI) + alpha_f)) as f32
                })
            })
        }
        PhaseMethod::GPaganin {
            pixel_size,
            dist,
            energy,
            db,
            w,
        } => {
            let fft = require_fft(backend)?;
            // Generalized Paganin (Paganin et al. 2020): `cos`-based reciprocal
            // grid `kf = cos(ix·2π·ps) + cos(iy·2π·ps)` and filter
            // `1/(1 − (2·aph/W²)·(kf − 2))` with `aph = db·dist·λ/(4π)`
            // (tomocupy `_reciprocal_gridG` + `_paganin_filter_factorG`).
            //
            // The grid/filter are evaluated in f32 to mirror cupy's single-
            // precision arithmetic (`cp.cos` of a float32 grid, weak-scalar
            // promotion). The filter is ill-conditioned — `scale ≈ 1.2e3`
            // amplifies any rounding in `kf` — so matching the reference's actual
            // f32 precision (rather than computing in f64) is what holds parity.
            let (ps, dist_f, db_f, w_f) = (pixel_size as f64, dist as f64, db as f64, w as f64);
            let two_pi_ps = (2.0 * PHASE_PI * ps) as f32;
            let aph = db_f * (dist_f * wavelength(energy as f64)) / (4.0 * PHASE_PI);
            let scale = (2.0 * aph / (w_f * w_f)) as f32;
            run_phase(data, pixel_size, dist, energy, fft, move |nx, ny| {
                pointwise_filter(nx, ny, ps, |ix, iy| {
                    let kf = (ix as f32 * two_pi_ps).cos() + (iy as f32 * two_pi_ps).cos();
                    1.0f32 / (1.0 - scale * (kf - 2.0))
                })
            })
        }
        PhaseMethod::Farago {
            pixel_size,
            dist,
            energy,
            db,
        } => {
            let fft = require_fft(backend)?;
            // Farago (2024): same padded-Fourier machinery as Paganin but with
            // the filter `1/(cos θ + db·sin θ)`, `θ = π·λ·dist·(ix² + iy²)` over
            // the squared reciprocal grid (tomocupy `_reciprocal_grid` +
            // `_farago_filter_factor`).
            //
            // Evaluated in f32 to mirror cupy: `db ≈ 1e3` multiplies `sin θ`, so a
            // 1-ULP error in `θ` is amplified ~1e3×. The grid must be built from
            // the *exact* f32 reciprocal coordinate — numpy/cupy round the
            // `0.5/((n−1)·ps)` scale to f32 *before* the multiply (NEP50 weak
            // scalar), which differs from an f64 grid cast down by up to 1 ULP and
            // diverges ~1e-3 in the normalized filter. See [`reciprocal_coord_f32`].
            let ps = pixel_size as f64;
            let theta_scale = (PHASE_PI * wavelength(energy as f64) * dist as f64) as f32;
            run_phase(data, pixel_size, dist, energy, fft, move |nx, ny| {
                farago_filter_grid(nx, ny, ps, theta_scale, db)
            })
        }
    }
}

/// X-ray wavelength in cm for `energy` keV (tomopy `_wavelength`).
fn wavelength(energy: f64) -> f64 {
    2.0 * PHASE_PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy
}

/// Pad each axis up to a power of two large enough to host the Fresnel kernel
/// (tomopy `_calc_pad_width`): `(2^⌈log2(dim+pad_pix)⌉ − dim)/2`.
fn calc_pad_width(dim: usize, pixel_size: f64, wl: f64, dist: f64) -> usize {
    let pad_pix = (PHASE_PI * wl * dist / (pixel_size * pixel_size)).ceil();
    let dimf = dim as f64;
    ((2.0f64.powf((dimf + pad_pix).log2().ceil()) - dimf) * 0.5) as usize
}

/// Centered reciprocal-space coordinates (tomopy `_reciprocal_coord`):
/// `arange(-(n-1), n, 2) · 0.5/((n-1)·pixel_size)`, length `num_grid`.
fn reciprocal_coord(pixel_size: f64, num_grid: usize) -> Vec<f64> {
    let n = num_grid as f64 - 1.0;
    let scale = 0.5 / (n * pixel_size);
    let mut rc = Vec::with_capacity(num_grid);
    let mut v = -n;
    for _ in 0..num_grid {
        rc.push(v * scale);
        v += 2.0;
    }
    rc
}

/// Centered reciprocal-space coordinates in f32, matching numpy/cupy exactly:
/// `arange(-(n-1), n, 2, float32) · float32(0.5/((n-1)·pixel_size))`. The scale
/// is rounded to f32 *before* the multiply (NEP50 weak-scalar promotion), which
/// differs from `reciprocal_coord(...) as f32` (round once in f64) by up to 1
/// ULP — negligible for Paganin but decisive for the f32-sensitive Farago filter.
fn reciprocal_coord_f32(pixel_size: f64, num_grid: usize) -> Vec<f32> {
    let n = (num_grid - 1) as f64;
    let scale = (0.5 / (n * pixel_size)) as f32;
    let mut rc = Vec::with_capacity(num_grid);
    let mut v = -((num_grid - 1) as f32);
    for _ in 0..num_grid {
        rc.push(v * scale);
        v += 2.0;
    }
    rc
}

/// Build a centered phase filter (row-major) and its max by evaluating `factor`
/// at every f64 reciprocal grid point `(indx[i], indy[j])`. Used by the
/// f64-grid Paganin family; Farago builds its grid directly in f32
/// ([`farago_filter_grid`]).
#[allow(clippy::needless_range_loop)]
fn pointwise_filter(
    nx: usize,
    ny: usize,
    pixel_size: f64,
    factor: impl Fn(f64, f64) -> f32,
) -> (Vec<f32>, f32) {
    let indx = reciprocal_coord(pixel_size, nx);
    let indy = reciprocal_coord(pixel_size, ny);
    let mut filt = vec![0.0f32; nx * ny];
    let mut maxf = f32::NEG_INFINITY;
    for i in 0..nx {
        for j in 0..ny {
            let f = factor(indx[i], indy[j]);
            filt[i * ny + j] = f;
            if f > maxf {
                maxf = f;
            }
        }
    }
    (filt, maxf)
}

/// Centered Farago filter (row-major) and its max, built directly in f32 to
/// mirror cupy. `w2[i,j] = rcx²[i] + rcy²[j]` over the exact f32 reciprocal
/// coordinates (tomocupy `_reciprocal_grid`: squared in f32, summed in f32); the
/// filter is `1/(cos θ + db·sin θ)`, `θ = theta_scale·w2` with
/// `theta_scale = π·λ·dist` (tomocupy `_farago_filter_factor`).
#[allow(clippy::needless_range_loop)]
fn farago_filter_grid(
    nx: usize,
    ny: usize,
    pixel_size: f64,
    theta_scale: f32,
    db: f32,
) -> (Vec<f32>, f32) {
    let rcx = reciprocal_coord_f32(pixel_size, nx);
    let rcy = reciprocal_coord_f32(pixel_size, ny);
    let mut filt = vec![0.0f32; nx * ny];
    let mut maxf = f32::NEG_INFINITY;
    for i in 0..nx {
        let sx = rcx[i] * rcx[i];
        for j in 0..ny {
            let w2 = sx + rcy[j] * rcy[j];
            let theta = theta_scale * w2;
            let f = 1.0f32 / (theta.cos() + db * theta.sin());
            filt[i * ny + j] = f;
            if f > maxf {
                maxf = f;
            }
        }
    }
    (filt, maxf)
}

/// Resolve the backend FFT capability or report it missing.
fn require_fft(backend: &dyn Backend) -> Result<&dyn Fft> {
    backend.fft().ok_or_else(|| Error::MissingCapability {
        backend: backend.name(),
        capability: "Fft",
    })
}

/// Shared single-step phase-retrieval driver (tomopy/tomocupy `_retrieve_phase`):
/// pad each radiograph to a power-of-two host for the Fresnel kernel, multiply by
/// the `fftshift`ed max-normalized filter in Fourier space, and crop back.
/// `build_filter(nx, ny)` returns the centered (un-shifted) filter row-major over
/// the `nx × ny` reciprocal grid and its max (the normalization denominator) —
/// the only part that differs between Paganin, generalized Paganin, and Farago.
#[allow(clippy::needless_range_loop)]
fn run_phase(
    data: &mut Tomo<f32>,
    pixel_size: f32,
    dist: f32,
    energy: f32,
    fft: &dyn Fft,
    build_filter: impl Fn(usize, usize) -> (Vec<f32>, f32),
) -> Result<()> {
    let target = data.layout;
    let proj = data.as_layout(Layout::Projection); // [angle, dy, dz]
    let (nproj, dy, dz) = proj.array.dim();
    if nproj == 0 || dy == 0 || dz == 0 {
        return Ok(());
    }
    let src = &proj.array;

    let (ps, dist_f, energy_f) = (pixel_size as f64, dist as f64, energy as f64);
    let wl = wavelength(energy_f);
    let pad_r = calc_pad_width(dy, ps, wl, dist_f);
    let pad_c = calc_pad_width(dz, ps, wl, dist_f);
    let nx = dy + 2 * pad_r;
    let ny = dz + 2 * pad_c;

    // Pad value: mean of (first col + last col)/2 over the whole stack
    // (tomopy `_calc_pad_val`).
    let mut val_acc = 0.0f64;
    for m in 0..nproj {
        for i in 0..dy {
            val_acc += 0.5 * (src[[m, i, 0]] as f64 + src[[m, i, dz - 1]] as f64);
        }
    }
    let val = (val_acc / (nproj * dy) as f64) as f32;

    // Centered phase filter and its max (the normalization denominator).
    let (filt, maxf) = build_filter(nx, ny);

    let mut out = ndarray::Array3::<f32>::zeros((nproj, dy, dz));

    // One projection's phase retrieval, writing into its own `[dy, dz]` view. Reads
    // only shared immutable state (`src`, `filt`, `fft`), so projections are
    // independent and bit-identical whether serial or fanned across host threads.
    let process_proj = |m: usize, mut slab: ArrayViewMut2<f32>| -> Result<()> {
        let mut buf = vec![Complex32::new(0.0, 0.0); nx * ny];
        let mut prj = vec![0.0f32; nx * ny];
        // Edge-replicate pad (tomopy `_retrieve_phase`: rows first, then cols).
        for k in prj.iter_mut() {
            *k = val;
        }
        for i in 0..dy {
            for j in 0..dz {
                prj[(i + pad_r) * ny + (j + pad_c)] = src[[m, i, j]];
            }
        }
        for i in 0..pad_r {
            for j in 0..ny {
                prj[i * ny + j] = prj[pad_r * ny + j];
            }
        }
        for i in (nx - pad_r)..nx {
            for j in 0..ny {
                prj[i * ny + j] = prj[(nx - pad_r - 1) * ny + j];
            }
        }
        for i in 0..nx {
            let left = prj[i * ny + pad_c];
            for j in 0..pad_c {
                prj[i * ny + j] = left;
            }
            let right = prj[i * ny + (ny - pad_c - 1)];
            for j in (ny - pad_c)..ny {
                prj[i * ny + j] = right;
            }
        }

        for (b, &p) in buf.iter_mut().zip(prj.iter()) {
            *b = Complex32::new(p, 0.0);
        }
        fft.fft_2d(&mut buf, nx, ny, 1, false)?;
        // Multiply by the fftshifted, max-normalized filter (DC at [0,0]).
        for i in 0..nx {
            let si = (i + nx - nx / 2) % nx;
            for j in 0..ny {
                let sj = (j + ny - ny / 2) % ny;
                buf[i * ny + j] *= filt[si * ny + sj] / maxf;
            }
        }
        fft.fft_2d(&mut buf, nx, ny, 1, true)?;
        // Real part, cropped back to the original window.
        for i in 0..dy {
            for j in 0..dz {
                slab[[i, j]] = buf[(i + pad_r) * ny + (j + pad_c)].re;
            }
        }
        Ok(())
    };

    // The backend owns the per-projection execution strategy (serial / rayon /
    // multi-GPU); every strategy yields the identical stack.
    fft.for_each_slice(&mut out, &process_proj)?;

    *data = Tomo::new(out, Layout::Projection).to_layout(target);
    Ok(())
}
