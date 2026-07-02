//! Log-polar (Andersson–Carlsson–Nikitin) reconstruction — tomocupy `lprec`.
//!
//! Faithful port of tomocupy's `reconstruction/lprec.py` (precompute) +
//! `cuda/cfunc_lprec.cu` / `kernels_lprec.cuh` (runtime). The inverse Radon
//! transform is evaluated by mapping the filtered sinogram into log-polar
//! coordinates, where the back-projection becomes a 2-D convolution computed by
//! FFT, then resampling back to the Cartesian grid. Three overlapping angular
//! spans (`Nspan = 3`, each `2β` wide, `β = π/3`) tile the half-circle.
//!
//! Like fourierrec it takes a **pre-filtered** sinogram (tomocupy applies
//! `fbp_filter_center` before `LpRec.backprojection`), so the analytic
//! dispatcher runs [`FbpFilter::apply`] first. It needs only the [`Fft`]
//! capability (precompute 1-D FFTs + runtime 2-D FFTs), so it composes onto any
//! backend.
//!
//! Interpolation uses the cubic-B-spline scheme of the reference: the sinogram
//! is prefiltered into spline coefficients (causal/anticausal recursion, pole
//! `√3−2`), and resampling is a 4×4 cubic-B-spline gather with wrap addressing —
//! the CPU equivalent of the GPU's "cubic via two linear texture fetches" trick
//! (exact, minus the GPU's 9-bit texture-fraction quantization).
//!
//! Requires equally spaced angles in `[0, π)` (the log-polar span tiling assumes
//! it), `ntheta = 2^round(log2(nproj))`, `nrho = 2·2^round(log2(n))`. No CUDA
//! golden offline; verified by phantom round-trip and cross-method agreement
//! with gridrec (the `lprec_parity` test).

use ndarray::{Array3, ArrayViewMut2};
use std::f32::consts::PI;

use rayon::prelude::*;

use crate::backend::Fft;
use crate::data::{Layout, Tomo};
use crate::dtype::Complex32;
use crate::error::{Error, Result};
use crate::geometry::Geometry;

/// Number of overlapping angular spans (tomocupy fixes `Nspan = 3`).
const NSPAN: usize = 3;
/// Cubic-B-spline interpolation pole, `√3 − 2` (reference `defs.cuh` Pole).
const POLE: f32 = -0.267_949_2;

/// Cubic-B-spline basis weights for a sample at fractional offset `f ∈ [0,1)`.
/// Taps land at `floor−1 … floor+2`; matches `bspline_weights` in the reference.
#[inline]
fn bspline_weights(f: f32) -> [f32; 4] {
    let one = 1.0 - f;
    let sq = f * f;
    let one_sq = one * one;
    [
        (1.0 / 6.0) * one_sq * one,
        2.0 / 3.0 - 0.5 * sq * (2.0 - f),
        2.0 / 3.0 - 0.5 * one_sq * (2.0 - one),
        (1.0 / 6.0) * sq * f,
    ]
}

#[inline]
fn wrap(i: isize, n: usize) -> usize {
    let n = n as isize;
    (((i % n) + n) % n) as usize
}

/// 4×4 cubic-B-spline interpolation of `coeffs` (`width × height`, row-major,
/// already prefiltered) at pixel position `(x, y)`, wrap addressing on both axes.
fn cubic_interp2d(coeffs: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let wx = bspline_weights(x - ix);
    let wy = bspline_weights(y - iy);
    let (ix, iy) = (ix as isize, iy as isize);
    let mut sum = 0.0f32;
    for (j, &wyj) in wy.iter().enumerate() {
        let py = wrap(iy - 1 + j as isize, height);
        let row = py * width;
        let mut acc = 0.0f32;
        for (i, &wxi) in wx.iter().enumerate() {
            let px = wrap(ix - 1 + i as isize, width);
            acc += wxi * coeffs[row + px];
        }
        sum += wyj * acc;
    }
    sum
}

/// In-place cubic-B-spline prefilter of one line (samples → spline coefficients),
/// clamped boundaries — `ConvertToInterpolationCoefficients` from the reference.
fn convert_to_coeffs(c: &mut [f32]) {
    let n = c.len();
    if n < 2 {
        return;
    }
    let lambda = (1.0 - POLE) * (1.0 - 1.0 / POLE);
    // Causal initialization (clamped boundary over a 12-sample horizon).
    let horizon = 12.min(n);
    let mut zn = POLE;
    let mut sum = c[0];
    for &ck in c.iter().take(horizon) {
        sum += zn * ck;
        zn *= POLE;
    }
    c[0] = lambda * sum;
    // Causal recursion (each coefficient depends on the previous one).
    let mut prev = c[0];
    for ck in c.iter_mut().skip(1) {
        *ck = lambda * *ck + POLE * prev;
        prev = *ck;
    }
    // Anticausal initialization + recursion.
    c[n - 1] *= POLE / (POLE - 1.0);
    prev = c[n - 1];
    for ck in c.iter_mut().rev().skip(1) {
        *ck = POLE * (prev - *ck);
        prev = *ck;
    }
}

/// In-place ifftshift of a length-`n` complex line (roll by `n/2`).
fn ifftshift(line: &mut [Complex32]) {
    let n = line.len();
    line.rotate_left(n / 2);
}

/// Cubic B3 spline sampled on the grid `x2` (port of `splineB3(x2, 1)`).
fn spline_b3(x2: &[f32]) -> Vec<f32> {
    let sizex = x2.len();
    let mid = (x2[sizex - 1] + x2[0]) / 2.0;
    let xc: Vec<f32> = x2.iter().map(|v| v - mid).collect();
    let stepx = xc[1] - xc[0];
    let ri = 2i32; // ceil(2*r), r=1
    let r = stepx; // r*stepx with r=1
    let center = (((sizex + 1) as f32 / 2.0).ceil() as usize).saturating_sub(1);
    let x2c = xc[center];
    let lo = center as isize - ri as isize;
    let mut out = vec![0.0f32; sizex];
    for ix in -ri..=ri {
        let id = (lo + (ix + ri) as isize) as usize;
        let d = (xc[id] - x2c).abs() / r;
        let v = if d < 1.0 {
            (3.0 * d * d * d - 6.0 * d * d + 4.0) / 6.0
        } else if d < 2.0 {
            (-d * d * d + 6.0 * d * d - 12.0 * d + 8.0) / 6.0
        } else {
            0.0
        };
        out[id] = v;
    }
    out
}

/// `osg` from the reference — the wrapping bound `g`.
fn osg(a_r: f32, theta: f32) -> f32 {
    let mut g = f32::NEG_INFINITY;
    for k in 0..1000 {
        let t = -PI / 2.0 + PI * (k as f32) / 999.0;
        let wre = a_r * t.cos() + (1.0 - a_r);
        let wim = a_r * t.sin();
        let val = (wre * wre + wim * wim).sqrt().ln() + (theta - wim.atan2(wre)).cos().ln();
        if val > g {
            g = val;
        }
    }
    g
}

/// Discretization correction weights for the zeta kernel (fixed constants from
/// `fzeta_loop_weights_adj`).
const CORRECTING: [f64; 11] = [
    -216_254_335.0,
    679_543_284.0,
    -1_412_947_389.0,
    2_415_881_496.0,
    -3_103_579_086.0,
    2_939_942_400.0,
    -2_023_224_114.0,
    984_515_304.0,
    -321_455_811.0,
    63_253_516.0,
    -5_675_265.0,
];

/// Adjoint zeta convolution kernel `fZ[nrho, ntheta]` (port of
/// `fzeta_loop_weights_adj` followed by the `fftshift` in `create_adj`).
fn fzeta_loop_weights_adj(
    ntheta: usize,
    nrho: usize,
    betas: f32,
    rhos: f32,
    osthlarge: usize,
    fft: &dyn Fft,
) -> Result<Vec<Complex32>> {
    let nthetalarge = osthlarge * ntheta;
    // krho = linspace(-Nrho/2, Nrho/2, Nrho, endpoint=False)
    let krho: Vec<f32> = (0..nrho)
        .map(|j| -(nrho as f32) / 2.0 + (nrho as f32) * (j as f32) / (nrho as f32))
        .collect();
    // thsplarge = linspace(-1/2, 1/2, Nthetalarge, endpoint=False) * betas
    let thsplarge: Vec<f32> = (0..nthetalarge)
        .map(|k| (-0.5 + (k as f32) / (nthetalarge as f32)) * betas)
        .collect();

    // h: correction weights, with fftshift sign multiplier folded in.
    let mut correcting = [0.0f64; 11];
    for (i, (c, w)) in correcting.iter_mut().zip(CORRECTING.iter()).enumerate() {
        *c = 1.0 + w / 958_003_200.0;
        if i == 0 {
            *c = 2.0 * (*c - 0.5);
        }
    }
    let mut h = vec![1.0f32; nthetalarge];
    h[0] *= correcting[0] as f32;
    for i in 1..correcting.len() {
        h[i] *= correcting[i] as f32;
        h[nthetalarge - i] *= correcting[i] as f32;
    }
    // s = 1 - 2*((1..=N) % 2): fast fftshift multiplier.
    let s: Vec<f32> = (1..=nthetalarge)
        .map(|i| 1.0 - 2.0 * ((i % 2) as f32))
        .collect();
    for (hv, &sv) in h.iter_mut().zip(s.iter()) {
        *hv *= sv;
    }

    let log_cos: Vec<f32> = thsplarge.iter().map(|&t| t.cos().ln()).collect();
    // fcosa[j,k] = exp(2πi·krho[j]/rhos · log_cos[k]) (a = 0). This nrho×nthetalarge
    // fill (≈8.4M complex exps at 1024²) is the single largest build_grids cost;
    // parallelise it one krho-row per task.
    let mut buf = vec![Complex32::new(0.0, 0.0); nrho * nthetalarge];
    buf.par_chunks_mut(nthetalarge)
        .zip(krho.par_iter())
        .for_each(|(row, &kr)| {
            let scale = 2.0 * PI * kr / rhos;
            for (k, &lc) in log_cos.iter().enumerate() {
                let ph = scale * lc;
                row[k] = Complex32::new(ph.cos(), ph.sin()) * h[k];
            }
        });
    // Batched 1-D FFT over each krho row, length nthetalarge.
    fft.fft_1d(&mut buf, nthetalarge, nrho, false)?;
    // Apply s multiplier, crop columns to ntheta, scale by dthsplarge.
    let dth = thsplarge[1] - thsplarge[0];
    let lo = nthetalarge / 2 - ntheta / 2;
    let mut fz = vec![Complex32::new(0.0, 0.0); nrho * ntheta];
    for j in 0..nrho {
        for t in 0..ntheta {
            let v = buf[j * nthetalarge + (lo + t)] * (s[lo + t] * dth);
            fz[j * ntheta + t] = v;
        }
    }
    // Zero the border rows/cols (fZ[0]=0, fZ[:,0]=0) on the centered array, as in
    // the reference (this happens before `create_adj`'s fftshift).
    fz[..ntheta].fill(Complex32::new(0.0, 0.0));
    for j in 0..nrho {
        fz[j * ntheta] = Complex32::new(0.0, 0.0);
    }
    // Return in centered order; the caller applies the 2-D fftshift that
    // `create_adj` wraps around this call (both rho and theta axes).
    Ok(fz)
}

/// Precomputed log-polar reconstruction grids (geometry only — angle-independent
/// of slice, so built once per `(n, nproj)`).
///
/// `pub(crate)` so the CUDA backend's device-resident lprec path can reuse the
/// same precompute (uploading these grids) instead of duplicating it.
pub(crate) struct LpGrids {
    pub(crate) n: usize,
    pub(crate) nproj: usize,
    pub(crate) ntheta: usize,
    pub(crate) nrho: usize,
    /// Full Hermitian convolution kernel `[nrho, ntheta]` for C2C FFT.
    pub(crate) kfull: Vec<Complex32>,
    /// Per-span polar→log-polar coords + targets (main set).
    pub(crate) lp2p1: [Vec<f32>; NSPAN],
    pub(crate) lp2p2: [Vec<f32>; NSPAN],
    pub(crate) lpids: Vec<usize>,
    /// Per-span polar→log-polar coords + targets (wrapping set).
    pub(crate) lp2p1w: [Vec<f32>; NSPAN],
    pub(crate) lp2p2w: [Vec<f32>; NSPAN],
    pub(crate) wids: Vec<usize>,
    /// Per-span log-polar→Cartesian coords + disk targets.
    pub(crate) c2lp1: [Vec<f32>; NSPAN],
    pub(crate) c2lp2: [Vec<f32>; NSPAN],
    pub(crate) cids: Vec<usize>,
}

/// Number of overlapping angular spans, exposed for the CUDA path's per-span
/// loop. Only the CUDA backend consumes it, so it is dead code without that
/// feature.
#[cfg(feature = "cuda")]
pub(crate) const LP_NSPAN: usize = NSPAN;

pub(crate) fn build_grids(n: usize, nproj: usize, fft: &dyn Fft) -> Result<LpGrids> {
    let ntheta = 1usize << ((nproj as f32).log2().round() as u32);
    let nrho = 2 * (1usize << ((n as f32).log2().round() as u32));
    let beta = PI / NSPAN as f32;
    let hb = beta / 2.0;
    let a_r = hb.sin() / (1.0 + hb.sin());
    let am = (hb.cos() - hb.sin()) / (1.0 + hb.sin());
    let g = osg(a_r, hb);
    let dtheta = 2.0 * beta / ntheta as f32;
    let drho = (g - am.ln()) / nrho as f32;

    // proj = arange(Nproj)*pi/Nproj - beta/2
    let proj: Vec<f32> = (0..nproj)
        .map(|i| i as f32 * PI / nproj as f32 - hb)
        .collect();
    // thsp = arange(-Ntheta/2, Ntheta/2)*dtheta ; rhosp = arange(-Nrho, 0)*drho
    let thsp: Vec<f32> = (0..ntheta)
        .map(|i| (i as f32 - ntheta as f32 / 2.0) * dtheta)
        .collect();
    let rhosp: Vec<f32> = (0..nrho).map(|i| (i as f32 - nrho as f32) * drho).collect();

    // B3com = outer(fft(ifftshift(B3rho)), fft(ifftshift(B3th)))
    let mut b3th: Vec<Complex32> = spline_b3(&thsp)
        .into_iter()
        .map(|v| Complex32::new(v, 0.0))
        .collect();
    ifftshift(&mut b3th);
    fft.fft_1d(&mut b3th, ntheta, 1, false)?;
    let mut b3rho: Vec<Complex32> = spline_b3(&rhosp)
        .into_iter()
        .map(|v| Complex32::new(v, 0.0))
        .collect();
    ifftshift(&mut b3rho);
    fft.fft_1d(&mut b3rho, nrho, 1, false)?;

    // fZ (zeta kernel) / B3com, half spectrum, times const. The centered kernel
    // from `fzeta_loop_weights_adj` is brought into standard FFT order by the 2-D
    // fftshift that `create_adj` wraps around it (both axes), so the R2C-half
    // slice and the B3com division line up with the standard-order spectra.
    let fz_centered = fzeta_loop_weights_adj(ntheta, nrho, 2.0 * beta, g - am.ln(), 4, fft)?;
    let mut fz = vec![Complex32::new(0.0, 0.0); nrho * ntheta];
    for r in 0..nrho {
        let rs = (r + nrho / 2) % nrho;
        for t in 0..ntheta {
            let ts = (t + ntheta / 2) % ntheta;
            fz[r * ntheta + t] = fz_centered[rs * ntheta + ts];
        }
    }
    let nf = n as f32;
    let konst = (nf + 1.0) * (nf - 1.0) / (nf * nf) / 2.0 / 2.0f32.sqrt() * PI / 6.0 * 0.86 * 4.0;
    let halft = ntheta / 2 + 1;
    // fZ_half[r, t] = fZ[r,t] / B3com[r,t] * const, t in [0, ntheta/2].
    let mut fz_half = vec![Complex32::new(0.0, 0.0); nrho * halft];
    fz_half
        .par_chunks_mut(halft)
        .enumerate()
        .for_each(|(r, row)| {
            for (t, slot) in row.iter_mut().enumerate() {
                let b = b3rho[r] * b3th[t];
                let denom = b.norm_sqr();
                let q = if denom > 0.0 {
                    fz[r * ntheta + t] * b.conj() / denom
                } else {
                    Complex32::new(0.0, 0.0)
                };
                *slot = q * konst;
            }
        });
    // Build the full Hermitian kernel for C2C FFT convolution.
    let mut kfull = vec![Complex32::new(0.0, 0.0); nrho * ntheta];
    kfull
        .par_chunks_mut(ntheta)
        .enumerate()
        .for_each(|(r, row)| {
            for (t, slot) in row.iter_mut().enumerate() {
                *slot = if t <= ntheta / 2 {
                    fz_half[r * halft + t]
                } else {
                    let rm = (nrho - r) % nrho;
                    fz_half[rm * halft + (ntheta - t)].conj()
                };
            }
        });

    // ---- C2lp: Cartesian -> log-polar, per span (over the unit disk) ----
    let lin: Vec<f32> = (0..n)
        .map(|i| -1.0 + 2.0 * (i as f32) / (n as f32 - 1.0))
        .collect();
    let mut x1 = vec![0.0f32; n * n];
    let mut x2 = vec![0.0f32; n * n];
    let inv_n = 1.0 / n as f32;
    for i in 0..n {
        for j in 0..n {
            // tomocupy uses `x1 = lin[j] - 1/N` and `x2 = -lin[i] - 1/N` (it
            // negates the row coordinate, "adjust for tomocupy", for its
            // projection geometry). tomoxide's forward projector uses the
            // opposite y-sign (the same divergence fixed in fourierrec), so the
            // faithful row coordinate is the *value-negation* of tomocupy's,
            // `x2 = -(-lin[i] - 1/N) = lin[i] + 1/N` — a true geometric y-flip.
            // Index-mirroring instead (`lin[i] - 1/N`) flips the image too but
            // mis-registers it by 2/N ≈ 1 px; the value-negation form is the one
            // that co-registers with gridrec/the phantom (verified by the shift
            // probe: index-mirror needs a 1-px up shift, value-negation needs 0).
            x1[i * n + j] = lin[j] - inv_n;
            x2[i * n + j] = lin[i] + inv_n;
        }
    }
    let cids: Vec<usize> = (0..n * n)
        .filter(|&k| x1[k] * x1[k] + x2[k] * x2[k] <= 1.0)
        .collect();
    let mut c2lp1: [Vec<f32>; NSPAN] = Default::default();
    let mut c2lp2: [Vec<f32>; NSPAN] = Default::default();
    let th0 = thsp[0];
    let th_span = thsp[ntheta - 1] - thsp[0];
    let rh0 = rhosp[0];
    let rh_span = rhosp[nrho - 1] - rhosp[0];
    for (k, (c1s, c2s)) in c2lp1.iter_mut().zip(c2lp2.iter_mut()).enumerate() {
        let ang = k as f32 * beta + hb;
        let (sa, ca) = ang.sin_cos();
        // Per-point atan2/ln/sqrt over the ~π/4·n² disk points is the dominant
        // build_grids cost; parallelise across points (NSPAN=3 is too small to
        // parallelise the outer loop). `par_iter().unzip()` preserves order.
        let (v1, v2): (Vec<f32>, Vec<f32>) = cids
            .par_iter()
            .map(|&id| {
                let z1 = a_r * (x1[id] * ca + x2[id] * sa) + (1.0 - a_r);
                let z2 = a_r * (-x1[id] * sa + x2[id] * ca);
                let c1 = z2.atan2(z1);
                let c2 = (z1 * z1 + z2 * z2).sqrt().ln();
                (
                    (c1 - th0) / th_span * (ntheta as f32 - 1.0),
                    (c2 - rh0) / rh_span * (nrho as f32 - 1.0),
                )
            })
            .unzip();
        *c1s = v1;
        *c2s = v2;
    }

    // ---- lp2p: log-polar -> polar, per span (main + wrapping sets) ----
    // z1=thsp (tiled over rho), z2=exp(rhosp) (tiled over theta), meshgrid.
    let erho: Vec<f32> = rhosp.iter().map(|&v| v.exp()).collect();
    let mut z1 = vec![0.0f32; nrho * ntheta];
    let mut z2 = vec![0.0f32; nrho * ntheta];
    for r in 0..nrho {
        for t in 0..ntheta {
            z1[r * ntheta + t] = thsp[t];
            z2[r * ntheta + t] = erho[r];
        }
    }
    // pids: indices of proj in each span; proj0/projl and the projp shift.
    let mut pid_first = [0usize; NSPAN];
    let mut pid_len = [0usize; NSPAN];
    let mut proj0 = [0.0f32; NSPAN];
    let mut projl = [0.0f32; NSPAN];
    for k in 0..NSPAN {
        let lo = k as f32 * beta - hb;
        let hi = k as f32 * beta + hb;
        let ids: Vec<usize> = (0..nproj)
            .filter(|&i| proj[i] >= lo && proj[i] < hi)
            .collect();
        pid_first[k] = *ids.first().unwrap_or(&0);
        pid_len[k] = ids.len();
        proj0[k] = proj[pid_first[k]];
        projl[k] = proj[*ids.last().unwrap_or(&0)] - proj[pid_first[k]];
    }
    let projp = (nproj as f32 - 1.0) / (proj0[NSPAN - 1] + projl[NSPAN - 1] - proj0[0]);

    // main set: z2n = (z2 - (1-aR)cos(z1))/aR ; select |z2n|<=1 & z1 in [-b/2,b/2)
    let (lpids, z2n_main): (Vec<usize>, Vec<f32>) = (0..nrho * ntheta)
        .into_par_iter()
        .filter_map(|k| {
            let z2n = (z2[k] - (1.0 - a_r) * z1[k].cos()) / a_r;
            (z1[k] >= -hb && z1[k] < hb && z2n.abs() <= 1.0).then_some((k, z2n))
        })
        .unzip();
    let scale_lp = |k: usize, z1k: f32, z2n: f32| -> (f32, f32) {
        let p1 = (z1k + k as f32 * beta - proj0[k]) / projl[k] * (pid_len[k] as f32 - 1.0)
            + (proj0[k] - proj0[0]) * projp;
        let p2 = (z2n + 1.0) / 2.0 * (n as f32 - 1.0);
        (p1, p2)
    };
    let mut lp2p1: [Vec<f32>; NSPAN] = Default::default();
    let mut lp2p2: [Vec<f32>; NSPAN] = Default::default();
    for k in 0..NSPAN {
        let (v1, v2): (Vec<f32>, Vec<f32>) = lpids
            .par_iter()
            .enumerate()
            .map(|(idx, &id)| scale_lp(k, z1[id], z2n_main[idx]))
            .unzip();
        lp2p1[k] = v1;
        lp2p2[k] = v2;
    }

    // wrapping set: right side (log z2 > g) and left side (log z2 < log(am)-g+drho).
    let am_eg = am * (-g).exp();
    let eg_am = g.exp() / am;
    let drho_step = rhosp[1] - rhosp[0];
    // right side, then left side — concatenated in the same order the two
    // sequential loops produced (each ascending in k).
    let (mut wids, mut z2n_w): (Vec<usize>, Vec<f32>) = (0..nrho * ntheta)
        .into_par_iter()
        .filter_map(|k| {
            if z2[k].ln() > g {
                let z2n = (z2[k] * am_eg - (1.0 - a_r) * z1[k].cos()) / a_r;
                if z1[k] >= -hb && z1[k] < hb && z2n.abs() <= 1.0 {
                    return Some((k, z2n));
                }
            }
            None
        })
        .unzip();
    let am_ln = am.ln();
    let (wids_l, z2n_l): (Vec<usize>, Vec<f32>) = (0..nrho * ntheta)
        .into_par_iter()
        .filter_map(|k| {
            if z2[k].ln() < am_ln - g + drho_step {
                let z2n = (z2[k] * eg_am - (1.0 - a_r) * z1[k].cos()) / a_r;
                if z1[k] >= -hb && z1[k] < hb && z2n.abs() <= 1.0 {
                    return Some((k, z2n));
                }
            }
            None
        })
        .unzip();
    wids.extend(wids_l);
    z2n_w.extend(z2n_l);
    // Angle coordinate of each wrapping point. The reference indexes `z1` by the
    // *position within the wrapping subset* (`z1[lpidsw]`) rather than the grid
    // index — an apparent upstream off-by-population bug. tomoxide uses the
    // consistent grid index `z1[id]`. Either way the effect is negligible: the
    // wrapping set is tiny (≈79 of ≈14.4k log-polar points at n=128) and the
    // cross-method correlation is identical with the reference form, this form,
    // or wrapping disabled entirely (all 0.9676 vs gridrec).
    let mut lp2p1w: [Vec<f32>; NSPAN] = Default::default();
    let mut lp2p2w: [Vec<f32>; NSPAN] = Default::default();
    for k in 0..NSPAN {
        let (v1, v2): (Vec<f32>, Vec<f32>) = wids
            .par_iter()
            .enumerate()
            .map(|(idx, &id)| scale_lp(k, z1[id], z2n_w[idx]))
            .unzip();
        lp2p1w[k] = v1;
        lp2p2w[k] = v2;
    }

    let _ = drho; // drho is captured via drho_step / rhosp construction
    Ok(LpGrids {
        n,
        nproj,
        ntheta,
        nrho,
        kfull,
        lp2p1,
        lp2p2,
        lpids,
        lp2p1w,
        lp2p2w,
        wids,
        c2lp1,
        c2lp2,
        cids,
    })
}

/// Log-polar reconstruction of every slice of a **pre-filtered** sinogram.
pub fn lprec(sino: &Tomo<f32>, geom: &Geometry, n: usize, fft: &dyn Fft) -> Result<Array3<f32>> {
    let b = sino.as_layout(Layout::Sinogram);
    let nz = b.n_rows();
    let nang = b.n_angles();
    let nd = b.n_cols();

    // lprec requires equally spaced angles in [0, π) with nproj == n detector
    // columns (tomocupy's grid sizing). Guard the angle assumption.
    if nang < 2 || nd != n {
        return Err(Error::InvalidParam(format!(
            "lprec requires square geometry with detector width == grid size (got n={n}, ncols={nd})"
        )));
    }
    let angles = &geom.angles.0;
    let dth = (angles[1] - angles[0]).abs();
    let nproj_test = (std::f32::consts::PI / dth).round() as usize;
    if nproj_test != nang {
        return Err(Error::InvalidParam(
            "lprec requires equally spaced angles spanning [0, π)".into(),
        ));
    }

    let grids = build_grids(n, nang, fft)?;
    let (ntheta, nrho) = (grids.ntheta, grids.nrho);
    let bdata = b
        .array
        .as_slice()
        .expect("contiguous sinogram (to_layout yields a standard-layout copy)");
    let mut out = Array3::<f32>::zeros((nz, n, n));

    // One slice's reconstruction, writing into its own `[n, n]` output view. Reads
    // only shared immutable state (`bdata`, `grids`, `fft`), so slices are
    // independent and bit-identical whether serial or fanned across host threads.
    let process_row = |row: usize, mut slab: ArrayViewMut2<f32>| -> Result<()> {
        // Cubic-B-spline prefilter the sinogram slice [nproj, n] into spline
        // coefficients: first along the detector axis, then along the angle axis.
        let mut g = vec![0.0f32; nang * n];
        g.copy_from_slice(&bdata[row * nang * n..row * nang * n + nang * n]);
        for a in 0..nang {
            convert_to_coeffs(&mut g[a * n..a * n + n]);
        }
        let mut col = vec![0.0f32; nang];
        for d in 0..n {
            for a in 0..nang {
                col[a] = g[a * n + d];
            }
            convert_to_coeffs(&mut col);
            for a in 0..nang {
                g[a * n + d] = col[a];
            }
        }

        let mut f = vec![0.0f32; n * n];
        for k in 0..NSPAN {
            // 1. Interp polar -> log-polar grid (main + wrapping points).
            let mut fl = vec![0.0f32; nrho * ntheta];
            for (idx, &target) in grids.lpids.iter().enumerate() {
                // x = lp2p2 (detector coord, width n), y = lp2p1 (angle coord).
                fl[target] += cubic_interp2d(&g, n, nang, grids.lp2p2[k][idx], grids.lp2p1[k][idx]);
            }
            for (idx, &target) in grids.wids.iter().enumerate() {
                fl[target] +=
                    cubic_interp2d(&g, n, nang, grids.lp2p2w[k][idx], grids.lp2p1w[k][idx]);
            }

            // 2. 2-D convolution in log-polar space via FFT × kernel × IFFT.
            let mut flc: Vec<Complex32> = fl.iter().map(|&v| Complex32::new(v, 0.0)).collect();
            fft.fft_2d(&mut flc, nrho, ntheta, 1, false)?;
            for (c, &kf) in flc.iter_mut().zip(grids.kfull.iter()) {
                *c *= kf;
            }
            fft.fft_2d(&mut flc, nrho, ntheta, 1, true)?;
            // tomocupy applies a 2/(nrho·ntheta) scale; the inverse FFT already
            // divides by nrho·ntheta, so the remaining factor is 2.
            let flc_re: Vec<f32> = flc.iter().map(|c| c.re * 2.0).collect();

            // 3. Interp log-polar -> Cartesian disk, accumulate over spans.
            for (idx, &target) in grids.cids.iter().enumerate() {
                // x = C2lp1 (theta coord, width ntheta), y = C2lp2 (rho coord).
                f[target] += cubic_interp2d(
                    &flc_re,
                    ntheta,
                    nrho,
                    grids.c2lp1[k][idx],
                    grids.c2lp2[k][idx],
                );
            }
        }

        for iy in 0..n {
            for ix in 0..n {
                slab[[iy, ix]] = f[iy * n + ix];
            }
        }
        Ok(())
    };

    // The backend owns the per-slice execution strategy (serial / rayon /
    // multi-GPU); every strategy yields the identical volume.
    fft.for_each_slice(&mut out, &process_row)?;
    // Reference the precomputed dims so the struct fields are all exercised even
    // if a future refactor drops one (keeps the port self-documenting).
    debug_assert_eq!(grids.n, n);
    debug_assert_eq!(grids.nproj, nang);
    Ok(out)
}
