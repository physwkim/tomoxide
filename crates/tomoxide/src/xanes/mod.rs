//! Per-voxel XANES peak-energy fitting.
//!
//! Given a stack of reconstructed volumes taken at a series of X-ray energies
//! straddling an absorption edge, this maps each voxel's absorption spectrum to
//! the energy of its edge/whiteline peak — a chemical-state map. The numerics
//! are a direct port of the reference pipeline (`txm_pal_core`, the crate the
//! beamline's Python XANES tool calls); see [`fit`] and [`filter`] for the
//! bit-for-bit provenance.
//!
//! Two layers:
//!
//! - [`fit_peak_energy`] — the per-voxel core: smooth the spectrum, locate the
//!   peak, and fit a [`FitMethod`] curve to a window around it, returning the
//!   fitted peak energy (or NaN if the fit leaves the `[start_e, stop_e]`
//!   window).
//! - [`fit_map`] — a `rayon` driver over a volume view (`(E, z, y, x)`),
//!   parallel across `z`. It reads `f32` views and never materialises the full
//!   `f64` stack, so a caller can stream one z-band at a time (a 40-energy
//!   500³ volume is ~40 GB as `f64`); [`CancelToken`] is polled between slices.
//!
//! Feature-gated behind `xanes` (pulls `levenberg-marquardt`, `nalgebra`,
//! `savgol-rs`, `median`); off by default so the core stays lean.

use ndarray::{Array2, Array3, ArrayView1, ArrayView3, ArrayView4};
use rayon::prelude::*;
use savgol_rs::{savgol_filter, SavGolInput};

use crate::error::{Error, Result};
use crate::pipeline::CancelToken;

mod filter;
mod fit;
mod magnification;
mod reader;
mod result;

use filter::{boxcar, medfilt, multi_3point_average};
use fit::{gaussian_fit_center, quadratic_fit_center};

pub use magnification::{apply_magnification, magnification_corr_factors, MagnificationParams};
pub use reader::{EnergyLayer, MultiEnergyVolume};
pub use result::write_peak_map_h5;

/// Curve model fitted to the windowed peak.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FitMethod {
    /// Downward parabola; peak energy is the vertex `-b / 2a`. Robust and cheap
    /// — the default for whiteline-position mapping.
    #[default]
    Quadratic,
    /// Gaussian `a·exp(-(x-b)²/2c²) + d`; peak energy is the centre `b`.
    Gaussian,
}

/// Optional 1-D spectral smoother applied before peak finding.
///
/// One smoother is applied uniformly regardless of the [`FitMethod`]. This is a
/// deliberate divergence from the reference `TXM-Pal-core` fitter, whose two
/// fit functions key the median branch on different strings (`"median"` in the
/// quadratic path, `"medfilt"` in the gaussian path). Because the app passes a
/// single algorithm name, exactly one fit method there silently skips
/// smoothing — e.g. with `"median"` the gaussian fits run unsmoothed. The
/// enum here removes that string coupling so median smoothing always applies;
/// do not "restore parity" by reintroducing the split keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmoothAlgo {
    /// No smoothing.
    #[default]
    None,
    /// Savitzky–Golay (`smooth_width` window, `smooth_order` polynomial).
    SavGol,
    /// Sliding-window median (`smooth_width`), zero-padded edges.
    Median,
    /// `smooth_width` iterations of a 3-point moving average.
    ThreePoint,
    /// Centred boxcar of width `smooth_width`.
    Boxcar,
}

/// Fit configuration shared by every voxel in a map.
#[derive(Debug, Clone, Copy)]
pub struct FitParams {
    /// Curve model.
    pub method: FitMethod,
    /// Number of energy samples in the fit window centred on the peak.
    pub points: usize,
    /// Lower energy bound: peak search and result validity both clamp here.
    pub start_e: f64,
    /// Upper energy bound.
    pub stop_e: f64,
    /// Pre-fit smoother.
    pub smooth: SmoothAlgo,
    /// Smoother window width (samples / iterations, per algorithm).
    pub smooth_width: usize,
    /// Savitzky–Golay polynomial order (ignored by other smoothers).
    pub smooth_order: usize,
}

impl Default for FitParams {
    fn default() -> Self {
        FitParams {
            method: FitMethod::Quadratic,
            points: 7,
            start_e: f64::NEG_INFINITY,
            stop_e: f64::INFINITY,
            smooth: SmoothAlgo::None,
            smooth_width: 5,
            smooth_order: 2,
        }
    }
}

/// NaN-tolerant "argmax by value" — returns the index of the largest sample.
///
/// The reference uses `partial_cmp().unwrap()`, which panics on a NaN sample
/// (a single bad voxel would abort the whole parallel map). `total_cmp` orders
/// NaN deterministically instead; for all-finite spectra it is identical.
fn argmax(xs: &[f64]) -> Option<usize> {
    xs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
}

/// Same as [`argmax`] but for the minimum sample.
fn argmin(xs: &[f64]) -> Option<usize> {
    xs.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
}

/// Fit one voxel spectrum and return its peak energy, or NaN.
///
/// `energy` and `spectrum` are parallel arrays (same length, energy ascending).
/// Mirrors the reference `quadfit_mc_3d` / `gaussianfit_mc_3d` inner body: the
/// spectrum is optionally smoothed, the peak is located within
/// `[start_e, stop_e]`, and a [`FitMethod`] curve is fitted to a `points`-wide
/// window around it. A fitted centre outside `[start_e, stop_e]` (or a
/// degenerate window) yields NaN so the caller can mask it out.
pub fn fit_peak_energy(energy: &[f64], spectrum: &[f64], p: &FitParams) -> f64 {
    let n = energy.len();
    if n < 2 || spectrum.len() != n {
        return f64::NAN;
    }

    // Energy window: first index at or past each bound (reference semantics).
    let Some(start_idx) = energy.iter().position(|&x| x >= p.start_e) else {
        return f64::NAN;
    };
    let Some(stop_idx) = energy.iter().position(|&x| x >= p.stop_e) else {
        return f64::NAN;
    };
    if start_idx >= stop_idx {
        return f64::NAN;
    }

    // Smoothing.
    let mut slice = spectrum.to_vec();
    match p.smooth {
        SmoothAlgo::None => {}
        SmoothAlgo::SavGol => {
            let input = SavGolInput {
                data: &slice,
                window_length: p.smooth_width,
                poly_order: p.smooth_order,
                derivative: 0,
            };
            // A silent fall-through to the unsmoothed spectrum would fit raw
            // data while the caller believes smoothing is on (plausible but
            // wrong). Surface it as NaN instead; the `fit_map` driver validates
            // the parameters up front so this is unreachable on that path.
            match savgol_filter(&input) {
                Ok(f) => slice = f,
                Err(_) => return f64::NAN,
            }
        }
        SmoothAlgo::Median => slice = medfilt(slice, p.smooth_width, "zeropadding"),
        SmoothAlgo::ThreePoint => slice = multi_3point_average(&slice, p.smooth_width),
        SmoothAlgo::Boxcar => slice = boxcar(&slice, p.smooth_width),
    }

    // Peak within the energy window.
    let sub = &slice[start_idx..stop_idx];
    let Some(rel_max) = argmax(sub) else {
        return f64::NAN;
    };
    let max_idx = rel_max + start_idx;

    // Fit window: `points` samples centred on the peak, clamped to the spectrum.
    let half = p.points / 2;
    let fit_start = max_idx.saturating_sub(half);
    let mut fit_end = max_idx + half + 1;
    if fit_end > slice.len() - 1 {
        fit_end = slice.len() - 1;
    }
    // A degenerate window (peak at an edge) has no data to fit / a zero slope
    // denominator — mask instead of dividing by zero.
    if fit_end <= fit_start {
        return f64::NAN;
    }

    let xdata = energy[fit_start..fit_end].to_vec();
    let ydata = slice[fit_start..fit_end].to_vec();

    let center = match p.method {
        FitMethod::Quadratic => {
            // Initial guess for a·x² + b·x + c.
            let c = slice[fit_start];
            let denom = energy[fit_end] - energy[fit_start];
            if denom == 0.0 {
                return f64::NAN;
            }
            let b = (slice[fit_end] - slice[fit_start]) / denom;
            let mut a = -b / (2.0 * energy[max_idx]);
            if a > 0.0 {
                a = -a;
            }
            quadratic_fit_center(xdata, ydata, vec![a, b, c])
        }
        FitMethod::Gaussian => {
            // Initial guess for a·exp(-(x-b)²/2c²) + d, from the window extrema.
            let (nrjmin, nrjmax) = energy[start_idx..stop_idx]
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
                    (mn.min(v), mx.max(v))
                });
            let Some(rel_min) = argmin(sub) else {
                return f64::NAN;
            };
            let min_idx = rel_min + start_idx;
            let maxy = slice[max_idx];
            let miny = slice[min_idx];
            let cen = energy[max_idx];
            let height = (maxy - miny) * 3.0;
            let sig = (nrjmax - nrjmin) / 6.0;
            let amp = height * sig;
            gaussian_fit_center(xdata, ydata, vec![amp, cen, sig, miny])
        }
    };

    if center >= p.start_e && center <= p.stop_e {
        center
    } else {
        f64::NAN
    }
}

/// Fit a peak-energy map over a volume view, parallel across `z`.
///
/// `volume` is `(E, z, y, x)` and `mask` is `(z, y, x)`; a zero mask voxel is
/// skipped and left NaN. To stream, slice a z-band out of the full stack
/// (`volume.slice(s![.., z0..z1, .., ..])` with the matching mask band) and
/// call this per band — the `f32` views mean the full `f64` stack is never
/// held. `cancel`, if given, is polled once per z-slice; a fired token returns
/// [`Error::Cancelled`].
///
/// Returns a `(z, y, x)` `f64` map of fitted peak energies (NaN where masked,
/// unfittable, or out of the energy window).
///
/// Window contract: unlike the smoother parameters — which are rejected up
/// front with [`Error::InvalidParam`] — an inconsistent search window is *not*
/// an error. If `start_e >= stop_e`, or the window does not overlap `energy`
/// (`start_e` past the last energy), every voxel falls out of the window and
/// the whole map comes back NaN. This is by design (the NaN mask is the signal),
/// but it means an all-NaN result is the caller's cue to check that
/// `start_e < stop_e` and that `[start_e, stop_e]` overlaps the energy grid,
/// not evidence of a fit failure.
pub fn fit_map(
    energy: ArrayView1<f64>,
    volume: ArrayView4<f32>,
    mask: ArrayView3<u8>,
    params: &FitParams,
    cancel: Option<&CancelToken>,
) -> Result<Array3<f64>> {
    let (ne, nz, ny, nx) = volume.dim();
    if energy.len() != ne {
        return Err(Error::ShapeMismatch {
            expected: format!("energy len == volume energy axis ({ne})"),
            found: format!("energy len {}", energy.len()),
        });
    }
    if mask.dim() != (nz, ny, nx) {
        return Err(Error::ShapeMismatch {
            expected: format!("mask {:?}", (nz, ny, nx)),
            found: format!("mask {:?}", mask.dim()),
        });
    }
    // Validate the smoother parameters once, loudly, so no per-voxel smoother
    // call can panic inside the rayon map (aborting the whole reconstruction)
    // or silently NaN/inf the map. Per voxel a SavGol failure still falls back
    // to NaN (see `fit_peak_energy`); the width-driven smoothers are guarded
    // here by construction.
    match params.smooth {
        SmoothAlgo::SavGol => {
            // Validity depends on length, not values; probe a dummy spectrum of
            // the real length.
            let probe = vec![0.0f64; ne];
            let input = SavGolInput {
                data: &probe,
                window_length: params.smooth_width,
                poly_order: params.smooth_order,
                derivative: 0,
            };
            if let Err(e) = savgol_filter(&input) {
                return Err(Error::InvalidParam(format!(
                    "Savitzky–Golay smoothing invalid for {ne} energies \
                     (window {}, order {}): {e}",
                    params.smooth_width, params.smooth_order
                )));
            }
        }
        // Median builds a `median::Filter::new(width)` that panics on a
        // zero-length window; Boxcar divides by `width` (→ inf). Both need a
        // positive width. ThreePoint uses `smooth_width` as an iteration count,
        // where 0 is a harmless no-op, and None does not smooth.
        SmoothAlgo::Median | SmoothAlgo::Boxcar if params.smooth_width == 0 => {
            let algo = if matches!(params.smooth, SmoothAlgo::Median) {
                "median"
            } else {
                "boxcar"
            };
            return Err(Error::InvalidParam(format!(
                "{algo} smoothing needs a window width >= 1 (got 0)"
            )));
        }
        _ => {}
    }
    let energy = energy
        .as_slice()
        .map_or_else(|| energy.to_vec(), <[f64]>::to_vec);

    let cancelled = std::sync::atomic::AtomicBool::new(false);
    let ordering = std::sync::atomic::Ordering::Relaxed;

    let slices: Vec<Array2<f64>> = (0..nz)
        .into_par_iter()
        .map(|iz| {
            let mut out = Array2::from_elem((ny, nx), f64::NAN);
            if cancel.is_some_and(CancelToken::is_cancelled) {
                cancelled.store(true, ordering);
                return out;
            }
            let mut spectrum = vec![0.0f64; ne];
            for iy in 0..ny {
                for ix in 0..nx {
                    if mask[[iz, iy, ix]] == 0 {
                        continue; // left NaN
                    }
                    for (e, s) in spectrum.iter_mut().enumerate() {
                        *s = volume[[e, iz, iy, ix]] as f64;
                    }
                    out[[iy, ix]] = fit_peak_energy(&energy, &spectrum, params);
                }
            }
            out
        })
        .collect();

    if cancelled.load(ordering) || cancel.is_some_and(CancelToken::is_cancelled) {
        return Err(Error::Cancelled);
    }

    let mut vol = Array3::from_elem((nz, ny, nx), f64::NAN);
    for (iz, s) in slices.into_iter().enumerate() {
        vol.slice_mut(ndarray::s![iz, .., ..]).assign(&s);
    }
    Ok(vol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3, Array4};

    /// Sample a Gaussian whiteline peaked at `center` over `energy`.
    fn synth_spectrum(energy: &[f64], center: f64, width: f64) -> Vec<f64> {
        energy
            .iter()
            .map(|&e| (-0.5 * ((e - center) / width).powi(2)).exp())
            .collect()
    }

    fn energy_axis() -> Vec<f64> {
        // 8.30–8.40 keV, 21 points (5 eV step) — a typical Co K-edge scan.
        (0..21).map(|i| 8.300 + i as f64 * 0.005).collect()
    }

    #[test]
    fn quadratic_recovers_known_peak() {
        let energy = energy_axis();
        let truth = 8.352;
        let spectrum = synth_spectrum(&energy, truth, 0.02);
        let p = FitParams {
            method: FitMethod::Quadratic,
            points: 7,
            start_e: 8.30,
            stop_e: 8.40,
            ..Default::default()
        };
        let got = fit_peak_energy(&energy, &spectrum, &p);
        assert!(
            (got - truth).abs() < 0.005,
            "quadratic peak {got} vs truth {truth}"
        );
    }

    #[test]
    fn gaussian_recovers_known_peak() {
        let energy = energy_axis();
        let truth = 8.352;
        let spectrum = synth_spectrum(&energy, truth, 0.02);
        let p = FitParams {
            method: FitMethod::Gaussian,
            points: 9,
            start_e: 8.30,
            stop_e: 8.40,
            ..Default::default()
        };
        let got = fit_peak_energy(&energy, &spectrum, &p);
        assert!(
            (got - truth).abs() < 0.003,
            "gaussian peak {got} vs truth {truth}"
        );
    }

    #[test]
    fn empty_energy_window_is_nan() {
        // A [start_e, stop_e] window past the whole scan has no sample at or
        // above start_e → no valid peak search → NaN, never a fabricated value.
        let energy = energy_axis();
        let spectrum = synth_spectrum(&energy, 8.352, 0.02);
        let p = FitParams {
            method: FitMethod::Quadratic,
            points: 7,
            start_e: 9.00,
            stop_e: 9.10,
            ..Default::default()
        };
        assert!(fit_peak_energy(&energy, &spectrum, &p).is_nan());
    }

    #[test]
    fn fitted_center_out_of_range_is_nan() {
        // A pure linear ramp has no parabola vertex: the best fit drives a→0,
        // pushing the vertex -b/2a to ±∞, which fails the [start_e, stop_e]
        // range check → NaN. Guards against reporting an off-window edge as a
        // peak energy.
        let energy = energy_axis();
        let spectrum: Vec<f64> = (0..energy.len()).map(|i| i as f64).collect();
        let p = FitParams {
            method: FitMethod::Quadratic,
            points: 21,
            start_e: 8.30,
            stop_e: 8.40,
            ..Default::default()
        };
        let got = fit_peak_energy(&energy, &spectrum, &p);
        assert!(got.is_nan(), "expected NaN for rampless vertex, got {got}");
    }

    #[test]
    fn fit_map_masks_and_fits() {
        let energy = energy_axis();
        let ne = energy.len();
        let (nz, ny, nx) = (2, 2, 3);
        let truth = 8.352;
        let spectrum = synth_spectrum(&energy, truth, 0.02);

        let mut volume = Array4::<f32>::zeros((ne, nz, ny, nx));
        for e in 0..ne {
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        volume[[e, iz, iy, ix]] = spectrum[e] as f32;
                    }
                }
            }
        }
        // Mask out one voxel.
        let mut mask = Array3::<u8>::ones((nz, ny, nx));
        mask[[0, 0, 0]] = 0;

        let p = FitParams {
            method: FitMethod::Quadratic,
            points: 7,
            start_e: 8.30,
            stop_e: 8.40,
            ..Default::default()
        };
        let map = fit_map(
            Array1::from(energy).view(),
            volume.view(),
            mask.view(),
            &p,
            None,
        )
        .unwrap();

        assert!(map[[0, 0, 0]].is_nan(), "masked voxel must be NaN");
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    if (iz, iy, ix) == (0, 0, 0) {
                        continue;
                    }
                    let v = map[[iz, iy, ix]];
                    assert!((v - truth).abs() < 0.005, "voxel fit {v} vs {truth}");
                }
            }
        }
    }

    #[test]
    fn fit_map_rejects_invalid_savgol() {
        // An even Savitzky–Golay window is invalid. The driver must reject it
        // loudly up front, not silently NaN every voxel.
        let energy = energy_axis();
        let ne = energy.len();
        let volume = Array4::<f32>::zeros((ne, 1, 1, 2));
        let mask = Array3::<u8>::ones((1, 1, 2));
        let p = FitParams {
            method: FitMethod::Quadratic,
            points: 7,
            start_e: 8.30,
            stop_e: 8.40,
            smooth: SmoothAlgo::SavGol,
            smooth_width: 4, // even window → invalid
            smooth_order: 2,
        };
        let r = fit_map(
            Array1::from(energy).view(),
            volume.view(),
            mask.view(),
            &p,
            None,
        );
        assert!(
            matches!(r, Err(Error::InvalidParam(_))),
            "invalid SavGol window must be a loud error, got {r:?}"
        );
    }

    #[test]
    fn fit_map_rejects_zero_width_median_and_boxcar() {
        // A zero window would panic per voxel (Median → median::Filter::new(0))
        // or emit an all-inf map (Boxcar → 1/0). fit_map must reject it up
        // front, the same way it does an invalid SavGol window.
        let energy = energy_axis();
        let ne = energy.len();
        let volume = Array4::<f32>::zeros((ne, 1, 1, 1));
        let mask = Array3::<u8>::ones((1, 1, 1));
        for smooth in [SmoothAlgo::Median, SmoothAlgo::Boxcar] {
            let p = FitParams {
                smooth,
                smooth_width: 0,
                ..FitParams::default()
            };
            let r = fit_map(
                Array1::from(energy.clone()).view(),
                volume.view(),
                mask.view(),
                &p,
                None,
            );
            assert!(
                matches!(r, Err(Error::InvalidParam(_))),
                "{smooth:?} width=0 must be a loud error, got {r:?}"
            );
        }
    }

    #[test]
    fn fit_map_cancel_returns_cancelled() {
        let energy = energy_axis();
        let ne = energy.len();
        let volume = Array4::<f32>::zeros((ne, 1, 1, 1));
        let mask = Array3::<u8>::ones((1, 1, 1));
        let token = CancelToken::new();
        token.cancel();
        let p = FitParams::default();
        let r = fit_map(
            Array1::from(energy).view(),
            volume.view(),
            mask.view(),
            &p,
            Some(&token),
        );
        assert!(matches!(r, Err(Error::Cancelled)));
    }
}
