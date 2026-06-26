//! Beam-hardening correction.
//!
//! A faithful port of the parts of the `beamhardening` package
//! (aps-7bm/beamhardening) that tomocupy's `processing/external/hardening.py`
//! drives, plus the two correction passes
//! `processing/proc_functions.py::beamhardening` applies after minus-log:
//!
//! 1. **centerline** — remap each minus-log (extinction-length) value to the
//!    monochromatic-equivalent sample pathlength via a LUT built by integrating
//!    the detected (scintillator-absorbed) spectrum over sample thickness
//!    (`np.interp`).
//! 2. **angular** — multiply each detector row by a correction factor for the
//!    bending-magnet spectrum's dependence on vertical fan angle (`np.interp`).
//!
//! ## Cross-section source
//! `beamhardening` uses `xraydb`, which has no Rust port, so this uses
//! [`xraylib`] instead. For pure elements the two are bit-identical; for
//! compounds they differ by ~4e-5 (atomic-weight tables). The algorithm — the
//! Simpson integration, the thickness grid, the `np.interp` passes, and the
//! flat-field angle finding — is reproduced exactly, so against an xraylib-based
//! reference the port matches to the f64 floor (see the parity test), while
//! staying within ~1e-4 of real tomocupy.
//!
//! This module is compiled only with the **`beam-hardening`** feature (it pulls
//! xraylib, whose build runs bindgen / needs libclang).

#[cfg(not(feature = "beam-hardening"))]
mod disabled {
    use crate::data::Tomo;
    use crate::error::{Error, Result};

    /// Beam-hardening correction is a compile-time opt-in. Build with the
    /// `beam-hardening` feature to enable [`BeamCorrector`].
    pub fn beam_correct(_data: &mut Tomo<f32>, _start_row: usize, _end_row: usize) -> Result<()> {
        Err(Error::todo(
            "hardening::beam_correct (build with the `beam-hardening` feature)",
            "tomocupy processing/external/hardening.py:50",
        ))
    }
}
#[cfg(not(feature = "beam-hardening"))]
pub use disabled::beam_correct;

#[cfg(feature = "beam-hardening")]
mod enabled {
    use crate::data::Tomo;
    use crate::error::{Error, Result};
    use ndarray::Array2;
    use std::sync::Once;

    static XRL_INIT: Once = Once::new();
    fn xrl_init() {
        XRL_INIT.call_once(xraylib::init);
    }

    /// A bending-magnet source spectrum at one vertical angle from the ring
    /// plane. `energies_ev`/`power` are paired samples (tomocupy reads these
    /// from `Psi_##urad.dat`).
    #[derive(Clone, Debug)]
    pub struct Spectrum {
        pub angle_urad: f64,
        pub energies_ev: Vec<f64>,
        pub power: Vec<f64>,
    }

    /// A material as an xraylib chemical formula plus density (g/cm³).
    #[derive(Clone, Debug)]
    pub struct Material {
        pub formula: String,
        pub density: f64,
    }

    /// A filter / scintillator: a [`Material`] with an active thickness (µm).
    #[derive(Clone, Debug)]
    pub struct Layer {
        pub material: Material,
        pub thickness_um: f64,
    }

    /// Configuration for the beam-hardening calculation (tomocupy
    /// `beam-hardening-*` config keys + the `beamhardening` `setup.cfg`).
    #[derive(Clone, Debug)]
    pub struct BeamHardeningConfig {
        /// Scintillator material + active thickness.
        pub scintillator: Layer,
        /// Sample material (thickness is swept to build the LUT).
        pub sample: Material,
        /// Beam filters, applied in order.
        pub filters: Vec<Layer>,
        /// Reference transmission for the angular correction (default 0.1).
        pub ref_trans: f64,
        /// Spline-stability threshold on effective transmission (default 1e-5).
        pub threshold_trans: f64,
        /// Source-to-scintillator distance (m).
        pub d_source_m: f64,
        /// Object-space pixel size (µm).
        pub pixel_size_um: f64,
    }

    impl Default for BeamHardeningConfig {
        /// The `beamhardening` `setup.cfg` defaults (APS bending-magnet beam):
        /// LuAG scintillator, no sample/filters set (caller fills them in).
        fn default() -> Self {
            BeamHardeningConfig {
                scintillator: Layer {
                    material: Material {
                        formula: "Lu3Al5O12".into(),
                        density: 6.73,
                    },
                    thickness_um: 100.0,
                },
                sample: Material {
                    formula: "Fe".into(),
                    density: 7.87,
                },
                filters: Vec::new(),
                ref_trans: 0.1,
                threshold_trans: 1e-5,
                d_source_m: 36.0,
                pixel_size_um: 10.0,
            }
        }
    }

    /// Linear attenuation coefficient (1/cm) of a compound over `energies_ev`
    /// (eV). `photo` selects the photoelectric (`CS_Photo_CP`) vs total
    /// (`CS_Total_CP`) cross section. xraydb returns the same product
    /// `ρ · CS`, with xraylib energies in keV.
    fn material_mu(
        formula: &str,
        density: f64,
        energies_ev: &[f64],
        photo: bool,
    ) -> Result<Vec<f64>> {
        xrl_init();
        let mut out = Vec::with_capacity(energies_ev.len());
        for &e in energies_ev {
            let kev = e / 1000.0;
            let cs = if photo {
                xraylib::cs_photo_cp(formula, kev)
            } else {
                xraylib::cs_total_cp(formula, kev)
            }
            .map_err(|e| {
                Error::InvalidParam(format!("xraylib cross section for {formula}: {e}"))
            })?;
            out.push(cs * density);
        }
        Ok(out)
    }

    /// Composite Simpson's rule over non-uniform `x` (scipy
    /// `integrate.simpson`, default `even='simpson'`). The spectra have an odd
    /// sample count (even interval count), so the standard rule applies with no
    /// end correction; the even-`N` branch falls back to a trapezoidal last
    /// interval (not exercised by the bundled spectra).
    fn simpson(y: &[f64], x: &[f64]) -> f64 {
        let n = y.len();
        assert_eq!(n, x.len(), "simpson: y and x length mismatch");
        if n < 2 {
            return 0.0;
        }
        if n == 2 {
            return 0.5 * (x[1] - x[0]) * (y[0] + y[1]);
        }
        // basic_simpson over [start, stop) in steps of 2 (pairs of intervals).
        let basic = |stop: usize| -> f64 {
            let mut acc = 0.0f64;
            let mut i = 0usize;
            while i < stop {
                let h0 = x[i + 1] - x[i];
                let h1 = x[i + 2] - x[i + 1];
                let hsum = h0 + h1;
                let hprod = h0 * h1;
                let h0divh1 = h0 / h1;
                let tmp = hsum / 6.0
                    * (y[i] * (2.0 - 1.0 / h0divh1)
                        + y[i + 1] * (hsum * (hsum / hprod))
                        + y[i + 2] * (2.0 - h0divh1));
                acc += tmp;
                i += 2;
            }
            acc
        };
        if n % 2 == 1 {
            // Odd N (even intervals): pairs (0,1,2)..(N-3,N-2,N-1).
            basic(n - 2)
        } else {
            // Even N (odd intervals): Simpson over the first N-2 intervals, plus
            // a trapezoid on the last (documented fallback; bundled spectra are
            // odd-length so this path is unused).
            let mut r = if n > 3 { basic(n - 3) } else { 0.0 };
            r += 0.5 * (x[n - 1] - x[n - 2]) * (y[n - 1] + y[n - 2]);
            r
        }
    }

    /// numpy `np.interp(xq, xp, fp)` for one query: linear interpolation with
    /// flat extrapolation (clamped to `fp[0]` / `fp[last]`). `xp` ascending.
    fn np_interp_scalar(xq: f64, xp: &[f64], fp: &[f64]) -> f64 {
        let n = xp.len();
        if n == 0 {
            return f64::NAN;
        }
        if xq <= xp[0] {
            return fp[0];
        }
        if xq >= xp[n - 1] {
            return fp[n - 1];
        }
        // binary search for the interval [xp[i], xp[i+1]).
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if xp[mid] <= xq {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
        fp[lo] + slope * (xq - xp[lo])
    }

    /// numpy `np.logspace(start, stop, num)` = `10 ** linspace(start, stop, num)`.
    fn logspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
        (0..num)
            .map(|i| {
                let t = if num == 1 {
                    start
                } else {
                    start + (stop - start) * (i as f64) / ((num - 1) as f64)
                };
                10f64.powf(t)
            })
            .collect()
    }

    /// The fixed sample-thickness grid (µm) from `_find_interp_values_one_angle`:
    /// `sort(concat(-logspace(1,-1,41), [0], logspace(-1,4.5,111)))`.
    fn sample_thickness_grid() -> Vec<f64> {
        let mut v = Vec::with_capacity(41 + 1 + 111);
        for t in logspace(1.0, -1.0, 41) {
            v.push(-t);
        }
        v.push(0.0);
        v.extend(logspace(-1.0, 4.5, 111));
        v.sort_by(|a, b| a.total_cmp(b));
        v
    }

    /// `_find_interp_values_one_angle`: returns `(extinction_length, thickness)`
    /// pairs, ascending in extinction length.
    fn find_interp_values_one_angle(
        cfg: &BeamHardeningConfig,
        energies: &[f64],
        power: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let thicknesses = sample_thickness_grid();
        let sample_ext = material_mu(&cfg.sample.formula, cfg.sample.density, energies, false)?
            .iter()
            .map(|mu| mu * 1.0 * 1e-4) // thickness 1 µm
            .collect::<Vec<_>>();
        let scint_ext = material_mu(
            &cfg.scintillator.material.formula,
            cfg.scintillator.material.density,
            energies,
            true,
        )?
        .iter()
        .map(|mu| mu * cfg.scintillator.thickness_um * 1e-4)
        .collect::<Vec<_>>();
        let scint_abs: Vec<f64> = scint_ext.iter().map(|e| 1.0 - (-e).exp()).collect();

        // absorbed_power = simpson(power * scint_abs).
        let scint_spec: Vec<f64> = power.iter().zip(&scint_abs).map(|(p, a)| p * a).collect();
        let absorbed = simpson(&scint_spec, energies);

        let mut ext = Vec::new();
        let mut thk = Vec::new();
        let mut buf = vec![0.0f64; energies.len()];
        for &t in &thicknesses {
            for k in 0..energies.len() {
                let trans = (-sample_ext[k] * t).exp();
                buf[k] = power[k] * trans * scint_abs[k];
            }
            let detected = simpson(&buf, energies);
            let eff_trans = detected / absorbed;
            if eff_trans > cfg.threshold_trans {
                ext.push(-eff_trans.ln());
                thk.push(t);
            }
        }
        // argsort by extinction length (ascending), stable.
        let mut idx: Vec<usize> = (0..ext.len()).collect();
        idx.sort_by(|&a, &b| ext[a].total_cmp(&ext[b]));
        let ext_s = idx.iter().map(|&i| ext[i]).collect();
        let thk_s = idx.iter().map(|&i| thk[i]).collect();
        Ok((ext_s, thk_s))
    }

    /// The beam-hardening corrector: the two LUTs plus per-row fan angles.
    #[derive(Clone, Debug)]
    pub struct BeamCorrector {
        centerline_ext: Vec<f64>,
        centerline_path: Vec<f64>,
        angular_angles: Vec<f64>,
        angular_corr: Vec<f64>,
        row_angles: Vec<f64>,
        d_source_m: f64,
        pixel_size_um: f64,
    }

    impl BeamCorrector {
        /// Build the LUTs from a configuration and the source spectra (one per
        /// vertical angle). Ports `compute_interp_values`: filter each angle's
        /// spectrum, build its `(extinction, thickness)` calibration, take the
        /// centerline (angle 0) curve, and form the angular correction as the
        /// per-angle pathlength at `ref_trans`, normalised to the centerline.
        pub fn new(cfg: &BeamHardeningConfig, spectra: &[Spectrum]) -> Result<Self> {
            if spectra.is_empty() {
                return Err(Error::InvalidParam(
                    "beam hardening: no source spectra".into(),
                ));
            }
            let mut order: Vec<usize> = (0..spectra.len()).collect();
            order.sort_by(|&a, &b| spectra[a].angle_urad.total_cmp(&spectra[b].angle_urad));

            let mut centerline: Option<(Vec<f64>, Vec<f64>)> = None;
            let mut angles = Vec::with_capacity(spectra.len());
            let mut cal_curve = Vec::with_capacity(spectra.len());
            for &i in &order {
                let s = &spectra[i];
                // apply_filters: multiply by exp(-mu*thickness) for each filter.
                let mut power = s.power.clone();
                for f in &cfg.filters {
                    let att = material_mu(
                        &f.material.formula,
                        f.material.density,
                        &s.energies_ev,
                        false,
                    )?;
                    for k in 0..power.len() {
                        power[k] *= (-(att[k] * f.thickness_um * 1e-4)).exp();
                    }
                }
                let iv = find_interp_values_one_angle(cfg, &s.energies_ev, &power)?;
                if s.angle_urad == 0.0 {
                    centerline = Some(iv.clone());
                }
                cal_curve.push(np_interp_scalar(cfg.ref_trans, &iv.0, &iv.1));
                angles.push(s.angle_urad);
            }
            let (centerline_ext, centerline_path) = centerline.ok_or_else(|| {
                Error::InvalidParam("beam hardening: no angle-0 (centerline) spectrum".into())
            })?;
            let c0 = cal_curve[0];
            let angular_corr: Vec<f64> = cal_curve.iter().map(|c| c / c0).collect();

            Ok(BeamCorrector {
                centerline_ext,
                centerline_path,
                angular_angles: angles,
                angular_corr,
                row_angles: Vec::new(),
                d_source_m: cfg.d_source_m,
                pixel_size_um: cfg.pixel_size_um,
            })
        }

        /// Centerline LUT: `(extinction_length, sample_thickness)`, ascending in
        /// extinction length (the curve `correct` interpolates).
        pub fn centerline_lut(&self) -> (&[f64], &[f64]) {
            (&self.centerline_ext, &self.centerline_path)
        }

        /// Angular LUT: `(vertical_angle_urad, correction_factor)` normalised to
        /// the centerline (factor 1.0 at angle 0).
        pub fn angular_lut(&self) -> (&[f64], &[f64]) {
            (&self.angular_angles, &self.angular_corr)
        }

        /// Per-detector-row vertical fan angles (µrad) from [`find_angles`];
        /// empty until it is called.
        pub fn row_angles(&self) -> &[f64] {
            &self.row_angles
        }

        /// Set the per-detector-row vertical fan angles from a flat field
        /// (`find_angles`): find the brightest row (Gaussian-smoothed vertical
        /// profile), then `angle[row] = |row − center| · pixel_size / d_source`
        /// (µrad, since µm/m = µrad). Must be called before [`correct`].
        pub fn find_angles(&mut self, flat: &Array2<f32>) {
            let (nrows, _ncols) = flat.dim();
            // vertical_slice = sum over columns, in f64.
            let vslice: Vec<f64> = (0..nrows)
                .map(|r| flat.row(r).iter().map(|&v| v as f64).sum())
                .collect();
            // gaussian(200, std=20) window, convolve 'same', argmax.
            let g = gaussian_window(200, 20.0);
            let filtered = convolve_same(&vslice, &g);
            let center_row = argmax(&filtered) as f64;
            self.row_angles = (0..nrows)
                .map(|r| (r as f64 - center_row).abs() * self.pixel_size_um / self.d_source_m)
                .collect();
        }

        /// Apply both correction passes to a projection-layout chunk
        /// (tomocupy `proc_functions.beamhardening`, applied after minus-log).
        /// `data` is `[nproj, nrows, ncols]`; `[start_row, end_row)` selects the
        /// absolute detector rows this chunk spans (indexing the [`find_angles`]
        /// table for the angular pass).
        pub fn correct(
            &self,
            data: &mut Tomo<f32>,
            start_row: usize,
            end_row: usize,
        ) -> Result<()> {
            use crate::data::Layout;
            if data.layout != Layout::Projection {
                return Err(Error::InvalidParam(
                    "beam hardening: data must be in projection layout".into(),
                ));
            }
            let (_nproj, nrows, _ncols) = data.array.dim();
            if end_row.saturating_sub(start_row) != nrows {
                return Err(Error::InvalidParam(format!(
                    "beam hardening: row chunk [{start_row},{end_row}) has {} rows, data has {nrows}",
                    end_row.saturating_sub(start_row)
                )));
            }
            if self.row_angles.is_empty() {
                return Err(Error::InvalidParam(
                    "beam hardening: call find_angles(flat) before correct()".into(),
                ));
            }
            if end_row > self.row_angles.len() {
                return Err(Error::InvalidParam(format!(
                    "beam hardening: end_row {end_row} exceeds detector rows {}",
                    self.row_angles.len()
                )));
            }
            // Per-row angular correction (interp the row's angle on the curve).
            let corr: Vec<f32> = (start_row..end_row)
                .map(|r| {
                    np_interp_scalar(self.row_angles[r], &self.angular_angles, &self.angular_corr)
                        as f32
                })
                .collect();
            // centerline remap of each value, then per-row angular scaling.
            for ((_p, row, _c), v) in data.array.indexed_iter_mut() {
                let path =
                    np_interp_scalar(*v as f64, &self.centerline_ext, &self.centerline_path) as f32;
                *v = path * corr[row];
            }
            Ok(())
        }
    }

    /// scipy `signal.windows.gaussian(M, std)`: `exp(-0.5·((n−(M−1)/2)/std)²)`.
    fn gaussian_window(m: usize, std: f64) -> Vec<f64> {
        let c = (m as f64 - 1.0) / 2.0;
        (0..m)
            .map(|n| {
                let x = n as f64 - c;
                (-0.5 * (x / std) * (x / std)).exp()
            })
            .collect()
    }

    /// Full discrete convolution cropped to the input length (numpy
    /// `convolve(a, v, mode='same')`): the centered window of the full
    /// convolution, length `len(a)`.
    fn convolve_same(a: &[f64], v: &[f64]) -> Vec<f64> {
        let (na, nv) = (a.len(), v.len());
        if na == 0 || nv == 0 {
            return vec![0.0; na];
        }
        let full = na + nv - 1;
        let start = (full - na) / 2; // numpy 'same' offset
        (0..na)
            .map(|i| {
                let k = i + start; // index into the full convolution
                let mut acc = 0.0f64;
                // full[k] = sum_j a[j] * v[k-j], valid j range.
                let jlo = k.saturating_sub(nv - 1);
                let jhi = k.min(na - 1);
                for j in jlo..=jhi {
                    acc += a[j] * v[k - j];
                }
                acc
            })
            .collect()
    }

    fn argmax(v: &[f64]) -> usize {
        let mut best = 0usize;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        best
    }

    /// Parse a two-column `Psi_##urad.dat` spectrum (energy eV, spectral power).
    /// `comments='!'` like numpy `genfromtxt`: text after `!` is ignored.
    fn parse_spectrum(angle_urad: f64, text: &str) -> Spectrum {
        let mut energies_ev = Vec::new();
        let mut power = Vec::new();
        for line in text.lines() {
            let line = line.split('!').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut it = line.split_whitespace();
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                if let (Ok(e), Ok(p)) = (a.parse::<f64>(), b.parse::<f64>()) {
                    energies_ev.push(e);
                    power.push(p);
                }
            }
        }
        Spectrum {
            angle_urad,
            energies_ev,
            power,
        }
    }

    /// The bundled APS bending-magnet source spectra (XOP-generated
    /// `Psi_##urad.dat`, vendored under `crates/tomoxide-prep/data/`), one per
    /// vertical angle 0–40 µrad — the default `read_source_data` set tomocupy
    /// ships with the `beamhardening` package.
    pub fn default_aps_bm_spectra() -> Vec<Spectrum> {
        vec![
            parse_spectrum(0.0, include_str!("../../data/Psi_00urad.dat")),
            parse_spectrum(5.0, include_str!("../../data/Psi_05urad.dat")),
            parse_spectrum(10.0, include_str!("../../data/Psi_10urad.dat")),
            parse_spectrum(15.0, include_str!("../../data/Psi_15urad.dat")),
            parse_spectrum(20.0, include_str!("../../data/Psi_20urad.dat")),
            parse_spectrum(30.0, include_str!("../../data/Psi_30urad.dat")),
            parse_spectrum(40.0, include_str!("../../data/Psi_40urad.dat")),
        ]
    }
}

#[cfg(feature = "beam-hardening")]
pub use enabled::{
    default_aps_bm_spectra, BeamCorrector, BeamHardeningConfig, Layer, Material, Spectrum,
};
