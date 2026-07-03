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
    /// Stripe-removal method: `none` | `fw` | `ti` | `sf` | `vo-all`.
    pub remove_stripe_method: String,
    /// Phase-retrieval method: `none` | `paganin` | `Gpaganin` | `farago`.
    pub retrieve_phase_method: String,
    /// Iterations for iterative algorithms.
    pub num_iter: usize,
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
