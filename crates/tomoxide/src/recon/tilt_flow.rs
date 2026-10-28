//! Laminography tilt from the projections alone, before any reconstruction.
//!
//! Rotating by `dθ` about an axis tilted `τ` out of the detector plane moves a
//! feature at column `x` vertically by `sin τ·(x − x_c)·dθ`: the vertical flow
//! between two neighbouring projections is a plane in `x` whose slope is
//! `sin τ·dθ` and whose zero is the axis column. [`estimate_tilt_flow`] measures
//! that flow block by block and fits the plane, so the tilt scan
//! ([`lamino_tilt_scan`](super::center::lamino_tilt_scan)) can start from a
//! bracket instead of a sweep.

use crate::backend::{Backend, Fft};
use crate::data::{Layout, Tomo};
use crate::dtype::Complex32;
use crate::error::{Error, Result};
use ndarray::{ArrayView2, Axis, Slice};

/// Correlation block edge, px.
const BLOCK: usize = 128;
/// Block stride, px: half-overlapping blocks.
const STRIDE: usize = 64;
/// Horizontal lags the vertical profile takes its maximum over, ±px. Points at
/// different depths along the beam move sideways by different amounts, so a
/// block's peak is smeared in x; the vertical lag is read across the smear.
const DX_WINDOW: usize = 12;
/// A block below this normalised correlation peak is not used.
const MIN_QUALITY: f32 = 0.3;
/// A pair fits its plane from at least this many blocks.
const MIN_BLOCKS: usize = 10;
/// Pairs start at this many evenly spread projections.
const STARTS: usize = 8;

/// The tilt [`estimate_tilt_flow`] read off the projection flow.
#[derive(Debug, Clone, PartialEq)]
pub struct TiltFlow {
    /// Mean `|τ|` over the pairs, degrees. The flow gives the magnitude only.
    pub tilt_deg: f64,
    /// Spread (population σ) of the pairs' tilts, degrees.
    pub tilt_sigma_deg: f64,
    /// Mean in-plane roll of the flow plane, degrees: `atan(∂z/∂y ÷ ∂z/∂x)`.
    /// Near 0 when the axis is upright on the detector.
    pub roll_deg: f64,
    /// Spread of the pairs' rolls, degrees.
    pub roll_sigma_deg: f64,
    /// Mean column where the flow vanishes on the middle row: the axis. A
    /// diagnostic, with a spread several times the rings estimate's; it is
    /// meaningless when the tilt is near 0.
    pub axis_col: f64,
    /// Spread of the pairs' axis columns, px.
    pub axis_col_sigma: f64,
    /// Every pair that fitted, in start order.
    pub pairs: Vec<FlowPair>,
}

/// One projection pair's plane fit.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowPair {
    /// Index of the first projection.
    pub first: usize,
    /// The second is `first + gap`.
    pub gap: usize,
    /// `θ[first + gap] − θ[first]`, degrees.
    pub dtheta_deg: f64,
    /// `asin(|∂z/∂x| ÷ |dθ|)`, degrees.
    pub tilt_deg: f64,
    /// `atan(∂z/∂y ÷ ∂z/∂x)`, degrees.
    pub roll_deg: f64,
    /// Column where the fitted flow is 0 on the middle row.
    pub axis_col: f64,
    /// Blocks correlating above 0.3, the ones the fit started from.
    pub blocks: usize,
    /// RMS residual of the refit, px.
    pub resid_px: f64,
}

/// Estimate the laminography tilt from the vertical flow between projections a
/// few steps apart. `tomo` is the prepped (minus-log) stack and `theta` its
/// angles in radians.
///
/// Pairs start at 8 evenly spread projections. The gaps are the three up to the
/// one whose sideways motion at the frame edge, `(W/2)·k·step`, is nearest the
/// ±12 px the correlation tolerates. Each pair is cut into half-overlapping
/// Hann-windowed 128² blocks, whose plain (not phase-normalised)
/// cross-correlation gives the vertical shift to sub-pixel (parabolic peak). The
/// plane `dz = sx·x + sz·y + c` is fitted by least squares, refitted
/// without blocks beyond 2.5σ, and read as `τ = asin(|sx|/dθ)`,
/// `roll = atan(sz/sx)`, axis `−(c + sz·y_mid)/sx`. The intercept is free, so a
/// vertical stage wobble between the two frames does not bias the tilt.
///
/// Measured on the 261003 step scan at bin 2 (1001 projections over 360°, 20°
/// nominal): τ = 19.40 ± 1.46°, axis 478.8 ± 74.8, roll −1.8 ± 10.3°. Every pair
/// matches the Python original to its printed 0.1°; the axes sit 0.5 px left of
/// it, the block-centre correction.
pub fn estimate_tilt_flow(
    tomo: &Tomo<f32>,
    theta: &[f32],
    backend: &dyn Backend,
) -> Result<TiltFlow> {
    let fft = backend.fft().ok_or_else(|| Error::MissingCapability {
        backend: backend.name(),
        capability: "Fft",
    })?;
    let p = tomo.as_layout(Layout::Projection);
    let (nproj, ny, nx) = p.array.dim();
    if theta.len() != nproj {
        return Err(Error::ShapeMismatch {
            expected: format!("{nproj} angles"),
            found: format!("{}", theta.len()),
        });
    }
    if nproj < 2 || ny < 2 * BLOCK || nx < 2 * BLOCK {
        return Err(Error::InvalidParam(format!(
            "estimate_tilt_flow needs ≥2 projections of ≥{0}×{0} px; got {nproj}×{ny}×{nx}",
            2 * BLOCK
        )));
    }
    let step = (f64::from(theta[nproj - 1]) - f64::from(theta[0])).abs() / (nproj - 1) as f64;
    if !(step > 0.0 && step.is_finite()) {
        return Err(Error::InvalidParam(
            "estimate_tilt_flow: the angles do not advance".into(),
        ));
    }
    let k_max = ((DX_WINDOW as f64 / (nx as f64 / 2.0 * step)).round() as usize).max(1);
    let gaps = k_max.saturating_sub(2).max(1)..=k_max;

    let window = hann(BLOCK);
    let origins: Vec<(usize, usize)> = (0..=ny - BLOCK)
        .step_by(STRIDE)
        .flat_map(|y| (0..=nx - BLOCK).step_by(STRIDE).map(move |x| (y, x)))
        .collect();
    let mut pairs = Vec::new();
    for m in 0..STARTS {
        let first = m * nproj / STARTS;
        for gap in gaps.clone() {
            if first + gap >= nproj {
                continue;
            }
            let shifts = block_shifts(
                p.array.index_axis(Axis(0), first),
                p.array.index_axis(Axis(0), first + gap),
                &origins,
                &window,
                fft,
            )?;
            let Some((blocks, [sx, sz, c], resid_px)) = fit_plane(&shifts) else {
                continue;
            };
            let dtheta = f64::from(theta[first + gap]) - f64::from(theta[first]);
            pairs.push(FlowPair {
                first,
                gap,
                dtheta_deg: dtheta.to_degrees(),
                tilt_deg: (sx.abs() / dtheta.abs()).min(1.0).asin().to_degrees(),
                roll_deg: (sz / sx).atan().to_degrees(),
                axis_col: -(c + sz * (ny - 1) as f64 / 2.0) / sx,
                blocks,
                resid_px,
            });
        }
    }
    if pairs.is_empty() {
        return Err(Error::InvalidParam(format!(
            "estimate_tilt_flow: no pair had {MIN_BLOCKS} blocks correlating above {MIN_QUALITY}"
        )));
    }
    let (tilt_deg, tilt_sigma_deg) = mean_sigma(pairs.iter().map(|p| p.tilt_deg));
    let (roll_deg, roll_sigma_deg) = mean_sigma(pairs.iter().map(|p| p.roll_deg));
    let (axis_col, axis_col_sigma) = mean_sigma(pairs.iter().map(|p| p.axis_col));
    Ok(TiltFlow {
        tilt_deg,
        tilt_sigma_deg,
        roll_deg,
        roll_sigma_deg,
        axis_col,
        axis_col_sigma,
        pairs,
    })
}

/// numpy `hanning(n)`: `0.5 − 0.5·cos(2πi/(n − 1))`.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / (n - 1) as f64).cos() as f32)
        .collect()
}

/// One block's measurement: centre `(x, y)`, vertical shift of `b` against `a`,
/// and the normalised correlation peak.
struct BlockShift {
    x: f64,
    y: f64,
    dz: f64,
    quality: f32,
}

fn block_shifts(
    a: ArrayView2<f32>,
    b: ArrayView2<f32>,
    origins: &[(usize, usize)],
    window: &[f32],
    fft: &dyn Fft,
) -> Result<Vec<BlockShift>> {
    let n = BLOCK * BLOCK;
    let mut fa = vec![Complex32::new(0.0, 0.0); origins.len() * n];
    let mut fb = fa.clone();
    let mut norms = Vec::with_capacity(origins.len());
    for (k, &(y0, x0)) in origins.iter().enumerate() {
        let ea = windowed(a, y0, x0, window, &mut fa[k * n..(k + 1) * n]);
        let eb = windowed(b, y0, x0, window, &mut fb[k * n..(k + 1) * n]);
        norms.push((ea * eb + 1e-12).sqrt());
    }
    fft.fft_2d(&mut fa, BLOCK, BLOCK, origins.len(), false)?;
    fft.fft_2d(&mut fb, BLOCK, BLOCK, origins.len(), false)?;
    for (b, a) in fb.iter_mut().zip(&fa) {
        *b *= a.conj();
    }
    fft.fft_2d(&mut fb, BLOCK, BLOCK, origins.len(), true)?;

    let half = BLOCK / 2;
    let lag = |l: isize| l.rem_euclid(BLOCK as isize) as usize;
    Ok(origins
        .iter()
        .enumerate()
        .map(|(k, &(y0, x0))| {
            let cc = &fb[k * n..(k + 1) * n];
            let peak = cc.iter().map(|c| c.re).fold(f32::NEG_INFINITY, f32::max);
            // prof[p] is the vertical lag p − B/2, maximised over ±DX_WINDOW.
            let prof: Vec<f32> = (0..BLOCK)
                .map(|p| {
                    let row = lag(p as isize - half as isize) * BLOCK;
                    (-(DX_WINDOW as isize)..=DX_WINDOW as isize)
                        .map(|dx| cc[row + lag(dx)].re)
                        .fold(f32::NEG_INFINITY, f32::max)
                })
                .collect();
            let p = (0..BLOCK).fold(0, |m, i| if prof[i] > prof[m] { i } else { m });
            let mut dz = p as f64;
            if p > 0 && p < BLOCK - 1 {
                let (l, c, r) = (prof[p - 1] as f64, prof[p] as f64, prof[p + 1] as f64);
                let den = l - 2.0 * c + r;
                if den != 0.0 {
                    dz += 0.5 * (l - r) / den;
                }
            }
            // The window peaks at (B − 1)/2, so that is where the block's shift
            // is measured; the Python original's x0 + B/2 put the axis 0.5 px right.
            let mid = (BLOCK - 1) as f64 / 2.0;
            BlockShift {
                x: x0 as f64 + mid,
                y: y0 as f64 + mid,
                dz: dz - half as f64,
                quality: (f64::from(peak) / norms[k]) as f32,
            }
        })
        .collect())
}

/// Copy the block at `(y0, x0)` into `out`, mean-removed and windowed; return
/// its energy.
fn windowed(
    img: ArrayView2<f32>,
    y0: usize,
    x0: usize,
    window: &[f32],
    out: &mut [Complex32],
) -> f64 {
    let blk = img
        .slice_axis_move(Axis(0), Slice::from(y0..y0 + BLOCK))
        .slice_axis_move(Axis(1), Slice::from(x0..x0 + BLOCK));
    let mean = blk.iter().map(|&v| f64::from(v)).sum::<f64>() / (BLOCK * BLOCK) as f64;
    let mut energy = 0.0;
    for ((y, x), &v) in blk.indexed_iter() {
        let w = window[y] * window[x];
        let v = ((f64::from(v) - mean) as f32) * w;
        out[y * BLOCK + x] = Complex32::new(v, 0.0);
        energy += f64::from(v) * f64::from(v);
    }
    energy
}

/// Fit `dz = sx·x + sz·y + c` over the blocks above [`MIN_QUALITY`] and refit
/// within 2.5σ: `(blocks above MIN_QUALITY, [sx, sz, c], refit RMS)`. `None`
/// when too few blocks correlate.
fn fit_plane(shifts: &[BlockShift]) -> Option<(usize, [f64; 3], f64)> {
    let good: Vec<&BlockShift> = shifts.iter().filter(|s| s.quality > MIN_QUALITY).collect();
    if good.len() < MIN_BLOCKS {
        return None;
    }
    let sol = plane(good.iter().copied())?;
    let resid = |s: &BlockShift| s.dz - (sol[0] * s.x + sol[1] * s.y + sol[2]);
    let (_, sigma) = mean_sigma(good.iter().map(|s| resid(s)));
    // `≤`, so an exact fit keeps its blocks.
    let kept: Vec<&BlockShift> = good
        .iter()
        .copied()
        .filter(|s| resid(s).abs() <= 2.5 * sigma)
        .collect();
    let [sx, sz, c] = plane(kept.iter().copied())?;
    let rms = (kept
        .iter()
        .map(|s| (s.dz - (sx * s.x + sz * s.y + c)).powi(2))
        .sum::<f64>()
        / kept.len() as f64)
        .sqrt();
    Some((good.len(), [sx, sz, c], rms))
}

/// Least-squares `[sx, sz, c]` through `(x, y, dz)`; `None` when the blocks do
/// not span a plane.
fn plane<'a>(pts: impl Iterator<Item = &'a BlockShift>) -> Option<[f64; 3]> {
    let (mut m, mut v) = ([[0.0f64; 3]; 3], [0.0f64; 3]);
    for s in pts {
        let r = [s.x, s.y, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += r[i] * r[j];
            }
            v[i] += r[i] * s.dz;
        }
    }
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&m);
    if d == 0.0 || !d.is_finite() {
        return None;
    }
    // Cramer's rule: column i replaced by v.
    Some(std::array::from_fn(|i| {
        let mut mi = m;
        for (row, &vi) in mi.iter_mut().zip(&v) {
            row[i] = vi;
        }
        det(&mi) / d
    }))
}

/// Mean and population σ (numpy's default `std`).
fn mean_sigma(xs: impl Iterator<Item = f64> + Clone) -> (f64, f64) {
    let n = xs.clone().count() as f64;
    let mean = xs.clone().sum::<f64>() / n;
    let var = xs.map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}
