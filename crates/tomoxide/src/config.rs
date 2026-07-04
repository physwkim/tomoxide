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
}
