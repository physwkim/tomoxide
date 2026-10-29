//! Reconstruction configuration (a subset of tomocupy's `config.py` groups),
//! serializable to/from TOML. Behind the `config` feature.
//!
//! This is the file format shared by the CLI (`tomoxide init` writes the
//! template; `recon`/`recon_steps` load it via `--config`) and by GUI recipes.
//! The CLI uses it as the default for `backend`/`algorithm`/`rotation_axis`/
//! `filter_name`/`remove_stripe_method`/`retrieve_phase_method`/`num_iter`/
//! `nsino_per_chunk`/`save_format`; any explicit CLI flag overrides its config
//! value. `file_name` is informational — the input file is passed positionally
//! on the command line. Stripe/phase methods are selected by name; their
//! per-method parameters (`fw_*`/`ti_*`/`sf_*`/`vo_*` and the phase physics
//! `pixel_size`/`propagation_distance`/`energy`/`alpha`/`db`/`w`) live here too
//! and are equally overridable by the matching CLI flag. Only the selected
//! method's parameters are consulted. Unknown keys are ignored on load, so a
//! file may carry extra tables (e.g. a GUI's own state) without breaking the
//! CLI.

use crate::error::{Error, Result};
use crate::params::{Algorithm, PhaseMethod, ReconParams, StripeMethod};
use crate::pipeline::PrepOptions;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level reconstruction configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Input DXchange HDF5 file.
    pub file_name: String,
    /// Backend: `auto` | `cpu` | `cuda` | `wgpu`.
    pub backend: String,
    /// Reconstruction algorithm (e.g. `fbp`, `gridrec`, `fourierrec`, `sirt`).
    pub algorithm: String,
    /// FBP/gridrec apodization filter.
    pub filter_name: String,
    /// Rotation-axis column; `None` ⇒ auto-find.
    pub rotation_axis: Option<f32>,
    /// Laminography tilt angle (degrees); `None` ⇒ tomographic reconstruction.
    /// Whole-volume only (the tilt couples all detector rows), so the CLI
    /// honors it in `recon` and rejects it in the streaming `recon_steps`.
    pub lamino_angle: Option<f32>,
    /// Stripe-removal method: `none` | `fw` | `ti` | `sf` | `vo-all` |
    /// `vo-sort` | `vo-filter` | `vo-large` | `vo-dead` | `vo-fit`.
    pub remove_stripe_method: String,
    /// Phase-retrieval method: `none` | `paganin` | `Gpaganin` | `farago`.
    pub retrieve_phase_method: String,
    /// Iterations for iterative algorithms.
    pub num_iter: usize,
    /// Lateral support extension for truncated projections (iterative methods):
    /// edge-replicate extend each projection by `ncols/4` per side, solve on the
    /// wider grid, return the central crop. See `ReconParams::ext_pad`.
    pub ext_pad: bool,
    /// Regularization parameters for iterative methods (`reg_par`).
    pub reg_par: Vec<f32>,
    /// Slices per reconstruction chunk (streaming).
    pub nsino_per_chunk: usize,
    /// Output format: `tiff` | `h5` | `zarr`.
    pub save_format: String,
    /// Reconstruction precision: `float32` | `float16` (CUDA analytic paths only).
    pub dtype: String,
    /// Output base path — each writer adds its own suffix (tiff:
    /// `<base>_NNNNN.tiff` per slice; h5: `<base>.h5`; zarr: `<base>.zarr`);
    /// `None`/empty ⇒ `<input-without-extension>_rec`.
    pub output: Option<String>,

    // --- Stripe-removal parameters (used when the matching method is selected) ---
    /// `fw` damping factor `sigma`.
    pub fw_sigma: f32,
    /// `fw` decomposition level (`0` = auto).
    pub fw_level: usize,
    /// `ti` number of blocks (`0` = whole sinogram at once).
    pub ti_nblock: usize,
    /// `ti` damping factor `beta`.
    pub ti_beta: f32,
    /// `sf` median window size.
    pub sf_size: usize,
    /// `vo-all` signal-to-noise ratio.
    pub vo_snr: f32,
    /// `vo-all` large-stripe window size.
    pub vo_la_size: usize,
    /// `vo-all` small-stripe window size.
    pub vo_sm_size: usize,
    /// `vo-sort` median window size (`0` = tomopy auto: `max(5, 0.01·ncol)`,
    /// `21` for `ncol > 2000`).
    pub vo_sort_size: usize,
    /// `vo-sort` median-window dimensionality (`1` → `(size, 1)`, `2` →
    /// `(size, size)`).
    pub vo_sort_dim: u8,
    /// `vo-filter` Gaussian-window sigma separating the low-/high-pass
    /// components.
    pub vo_filter_sigma: f32,
    /// `vo-filter` inner-sort median window size (`0` = tomopy auto, as
    /// `vo_sort_size`).
    pub vo_filter_size: usize,
    /// `vo-filter` median-window dimensionality (as `vo_sort_dim`).
    pub vo_filter_dim: u8,
    /// `vo-large` signal-to-noise ratio for stripe detection.
    pub vo_large_snr: f32,
    /// `vo-large` median window size.
    pub vo_large_size: usize,
    /// `vo-large` fraction of extreme pixels dropped before estimating the
    /// per-column intensity factor.
    pub vo_large_drop_ratio: f32,
    /// `vo-large` normalize each column by its intensity factor.
    pub vo_large_norm: bool,
    /// `vo-dead` signal-to-noise ratio for stripe detection.
    pub vo_dead_snr: f32,
    /// `vo-dead` median window size.
    pub vo_dead_size: usize,
    /// `vo-dead` run the residual large-stripe pass after filling.
    pub vo_dead_norm: bool,
    /// `vo-fit` Savitzky–Golay polynomial fit order.
    pub vo_fit_order: usize,
    /// `vo-fit` Gaussian smoothing sigma along the detector columns.
    pub vo_fit_sigma_x: f32,
    /// `vo-fit` Gaussian smoothing sigma along the projections.
    pub vo_fit_sigma_y: f32,

    // --- Phase-retrieval physics (used when a phase method is selected) ---
    // Stored as f64 so decimal quantities like `1e-4` serialize cleanly in the
    // template (an f32 field promotes to f64 on write and leaks precision noise,
    // e.g. `0.00009999999747…`); cast to f32 at the reconstruction boundary.
    /// Detector pixel size (cm).
    pub pixel_size: f64,
    /// Sample-to-detector propagation distance (cm).
    pub propagation_distance: f64,
    /// X-ray energy (keV).
    pub energy: f64,
    /// Paganin regularization parameter `alpha`.
    pub alpha: f64,
    /// Gpaganin/farago material `delta/beta` ratio.
    pub db: f64,
    /// Gpaganin characteristic transverse length `W` (cm).
    pub w: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            file_name: String::new(),
            backend: "auto".into(),
            algorithm: "fbp".into(),
            filter_name: "parzen".into(),
            rotation_axis: None,
            lamino_angle: None,
            remove_stripe_method: "none".into(),
            retrieve_phase_method: "none".into(),
            num_iter: 1,
            ext_pad: false,
            reg_par: Vec::new(),
            nsino_per_chunk: 8,
            save_format: "tiff".into(),
            dtype: "float32".into(),
            output: None,

            fw_sigma: 2.0,
            fw_level: 0,
            ti_nblock: 0,
            ti_beta: 1.5,
            sf_size: 5,
            vo_snr: 3.0,
            vo_la_size: 61,
            vo_sm_size: 21,
            // tomopy defaults for the Vo 2018 single-method variants.
            vo_sort_size: 0,
            vo_sort_dim: 1,
            vo_filter_sigma: 3.0,
            vo_filter_size: 0,
            vo_filter_dim: 1,
            vo_large_snr: 3.0,
            vo_large_size: 51,
            vo_large_drop_ratio: 0.1,
            vo_large_norm: true,
            vo_dead_snr: 3.0,
            vo_dead_size: 51,
            vo_dead_norm: true,
            vo_fit_order: 3,
            vo_fit_sigma_x: 5.0,
            vo_fit_sigma_y: 20.0,

            pixel_size: 1e-4,
            propagation_distance: 50.0,
            energy: 30.0,
            alpha: 1e-3,
            db: 1000.0,
            w: 2e-4,
        }
    }
}

impl Config {
    /// Serialize to a TOML string.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self)
            .map_err(|e| Error::InvalidParam(format!("config serialize: {e}")))
    }

    /// Write the config to `path`.
    pub fn write(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_toml()?)
            .map_err(|e| Error::Io(format!("writing config {}: {e}", path.display())))
    }

    /// Load a config from `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Io(format!("reading config {}: {e}", path.display())))?;
        toml::from_str(&text)
            .map_err(|e| Error::InvalidParam(format!("config parse {}: {e}", path.display())))
    }

    /// The [`StripeMethod`] named by `remove_stripe_method` (case-insensitive,
    /// `-`/`_` spellings), filled from this config's parameters for that
    /// method. A `0` level/median size means auto (`None`).
    pub fn stripe_method(&self) -> Result<StripeMethod> {
        let auto = |n: usize| (n != 0).then_some(n);
        Ok(
            match self.remove_stripe_method.to_ascii_lowercase().as_str() {
                "none" => StripeMethod::None,
                "fw" => StripeMethod::Fw {
                    sigma: self.fw_sigma,
                    level: auto(self.fw_level),
                },
                "ti" => StripeMethod::Ti {
                    nblock: self.ti_nblock,
                    beta: self.ti_beta,
                },
                "sf" => StripeMethod::Sf { size: self.sf_size },
                "vo-all" | "vo_all" | "voall" => StripeMethod::VoAll {
                    snr: self.vo_snr,
                    la_size: self.vo_la_size,
                    sm_size: self.vo_sm_size,
                },
                "vo-sort" | "vo_sort" => StripeMethod::VoSort {
                    size: auto(self.vo_sort_size),
                    dim: self.vo_sort_dim,
                },
                "vo-filter" | "vo_filter" => StripeMethod::VoFilter {
                    sigma: self.vo_filter_sigma,
                    size: auto(self.vo_filter_size),
                    dim: self.vo_filter_dim,
                },
                "vo-large" | "vo_large" => StripeMethod::VoLarge {
                    snr: self.vo_large_snr,
                    size: self.vo_large_size,
                    drop_ratio: self.vo_large_drop_ratio,
                    norm: self.vo_large_norm,
                },
                "vo-dead" | "vo_dead" => StripeMethod::VoDead {
                    snr: self.vo_dead_snr,
                    size: self.vo_dead_size,
                    norm: self.vo_dead_norm,
                },
                "vo-fit" | "vo_fit" => StripeMethod::VoFit {
                    order: self.vo_fit_order,
                    sigma: (self.vo_fit_sigma_x, self.vo_fit_sigma_y),
                },
                other => {
                    return Err(Error::InvalidParam(format!(
                        "unknown stripe method '{other}' \
                         (none|fw|ti|sf|vo-all|vo-sort|vo-filter|vo-large|vo-dead|vo-fit)"
                    )))
                }
            },
        )
    }

    /// The [`PhaseMethod`] named by `retrieve_phase_method` (case-insensitive),
    /// filled from this config's physics cast to `f32`.
    pub fn phase_method(&self) -> Result<PhaseMethod> {
        let (pixel_size, dist, energy) = (
            self.pixel_size as f32,
            self.propagation_distance as f32,
            self.energy as f32,
        );
        Ok(
            match self.retrieve_phase_method.to_ascii_lowercase().as_str() {
                "none" => PhaseMethod::None,
                "paganin" => PhaseMethod::Paganin {
                    pixel_size,
                    dist,
                    energy,
                    alpha: self.alpha as f32,
                },
                "gpaganin" => PhaseMethod::GPaganin {
                    pixel_size,
                    dist,
                    energy,
                    db: self.db as f32,
                    w: self.w as f32,
                },
                "farago" => PhaseMethod::Farago {
                    pixel_size,
                    dist,
                    energy,
                    db: self.db as f32,
                },
                other => {
                    return Err(Error::InvalidParam(format!(
                        "unknown phase method '{other}' (none|paganin|Gpaganin|farago)"
                    )))
                }
            },
        )
    }

    /// The one [`Algorithm`] named by `algorithm`. A warm-start chain
    /// (`fbp,sirt`) is the CLI's to run and is refused here.
    pub fn algorithm(&self) -> Result<Algorithm> {
        if self.algorithm.contains(',') {
            return Err(Error::InvalidParam(format!(
                "algorithm '{}' is a chain; name one algorithm",
                self.algorithm
            )));
        }
        self.algorithm.trim().parse()
    }

    /// The [`ReconParams`] this config names, on the detector-width grid.
    pub fn recon_params(&self) -> Result<ReconParams> {
        Ok(ReconParams {
            dtype: self.dtype.parse().map_err(Error::InvalidParam)?,
            filter_name: self.filter_name.parse()?,
            num_iter: self.num_iter,
            reg_par: self.reg_par.clone(),
            ext_pad: self.ext_pad,
            ..Default::default()
        })
    }

    /// The stripe and phase prep this config names.
    pub fn prep(&self) -> Result<PrepOptions> {
        Ok(PrepOptions {
            stripe: self.stripe_method()?,
            phase: self.phase_method()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `remove_stripe_method` spelling maps to its variant with this
    /// config's parameters (the Vo 2018 single-method variants included), and
    /// unknown names list the full set.
    #[test]
    fn stripe_method_parses_every_spelling() {
        let mut c = Config {
            vo_filter_size: 7,
            vo_filter_dim: 2,
            vo_dead_snr: 4.0,
            vo_dead_size: 41,
            vo_dead_norm: false,
            ..Config::default()
        };
        let mut m = |name: &str| {
            c.remove_stripe_method = name.into();
            c.stripe_method()
        };
        assert_eq!(m("none").unwrap(), StripeMethod::None);
        assert_eq!(
            m("FW").unwrap(),
            StripeMethod::Fw {
                sigma: 2.0,
                level: None
            }
        );
        assert_eq!(
            m("vo_all").unwrap(),
            StripeMethod::VoAll {
                snr: 3.0,
                la_size: 61,
                sm_size: 21
            }
        );
        assert_eq!(
            m("vo-sort").unwrap(),
            StripeMethod::VoSort { size: None, dim: 1 }
        );
        assert_eq!(
            m("vo_filter").unwrap(),
            StripeMethod::VoFilter {
                sigma: 3.0,
                size: Some(7),
                dim: 2
            }
        );
        assert_eq!(
            m("vo-large").unwrap(),
            StripeMethod::VoLarge {
                snr: 3.0,
                size: 51,
                drop_ratio: 0.1,
                norm: true
            }
        );
        assert_eq!(
            m("vo-dead").unwrap(),
            StripeMethod::VoDead {
                snr: 4.0,
                size: 41,
                norm: false
            }
        );
        assert_eq!(
            m("vo-fit").unwrap(),
            StripeMethod::VoFit {
                order: 3,
                sigma: (5.0, 20.0)
            }
        );
        let err = m("nope").unwrap_err().to_string();
        assert!(err.contains("vo-fit"), "{err}");
    }

    /// Phase names are case-insensitive and the f64 physics reach the method
    /// as the f32 the CLI flags carry.
    #[test]
    fn phase_method_parses_and_casts() {
        let c = Config {
            retrieve_phase_method: "Gpaganin".into(),
            pixel_size: 0.65e-4,
            ..Config::default()
        };
        assert_eq!(
            c.phase_method().unwrap(),
            PhaseMethod::GPaganin {
                pixel_size: 0.65e-4,
                dist: 50.0,
                energy: 30.0,
                db: 1000.0,
                w: 2e-4
            }
        );
        let c = Config {
            retrieve_phase_method: "nope".into(),
            ..Config::default()
        };
        assert!(c.phase_method().unwrap_err().to_string().contains("farago"));
    }

    /// A recipe's algorithm, params and prep come from its own fields; a
    /// chain or an unknown dtype is refused rather than read as a default.
    #[test]
    fn recipe_reads_algorithm_params_and_prep() {
        let c: Config = toml::from_str(
            r#"
            algorithm = "fourierrec"
            filter_name = "shepp"
            dtype = "float16"
            num_iter = 7
            reg_par = [0.5, 2.0]
            ext_pad = true
            remove_stripe_method = "fw"
            retrieve_phase_method = "paganin"
            alpha = 0.001
            "#,
        )
        .unwrap();
        assert_eq!(c.algorithm().unwrap(), Algorithm::Fourierrec);
        let p = c.recon_params().unwrap();
        assert_eq!(p.filter_name, crate::params::FilterName::Shepp);
        assert_eq!(p.dtype, crate::dtype::Dtype::F16);
        assert_eq!(
            (p.num_iter, p.reg_par, p.ext_pad),
            (7, vec![0.5, 2.0], true)
        );
        assert_eq!(p.num_gridx, None);
        let prep = c.prep().unwrap();
        assert!(matches!(prep.stripe, StripeMethod::Fw { .. }));
        assert!(matches!(prep.phase, PhaseMethod::Paganin { alpha, .. } if alpha == 0.001));

        let chain = Config {
            algorithm: "fbp,sirt".into(),
            ..Config::default()
        };
        assert!(chain.algorithm().unwrap_err().to_string().contains("chain"));
        let bad = Config {
            dtype: "int8".into(),
            ..Config::default()
        };
        assert!(bad.recon_params().unwrap_err().to_string().contains("int8"));
    }
}
