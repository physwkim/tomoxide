//! Reconstruction configuration (a subset of tomocupy's `config.py` groups),
//! serializable to/from TOML for `tomoxide init`.
//!
//! `recon` and `recon_steps` load this via `--config` and use it as the default
//! for `backend`/`algorithm`/`rotation_axis`/`filter_name`/`remove_stripe_method`/
//! `retrieve_phase_method`/`num_iter`/`nsino_per_chunk`/`save_format`; any explicit
//! CLI flag overrides its config value. `file_name` is informational — the input
//! file is passed positionally on the command line. Stripe/phase methods are
//! selected by name here and reconstructed with default parameters; per-method
//! parameters (phase physics, etc.) are CLI flags.

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
    /// Stripe-removal method: `none` | `fw` | `ti` | `sf` | `vo-all`.
    pub remove_stripe_method: String,
    /// Phase-retrieval method: `none` | `paganin` | `Gpaganin` | `farago`.
    pub retrieve_phase_method: String,
    /// Iterations for iterative algorithms.
    pub num_iter: usize,
    /// Slices per reconstruction chunk (streaming).
    pub nsino_per_chunk: usize,
    /// Output format: `tiff` | `h5` | `zarr`.
    pub save_format: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            file_name: String::new(),
            backend: "auto".into(),
            algorithm: "fbp".into(),
            filter_name: "parzen".into(),
            rotation_axis: None,
            remove_stripe_method: "none".into(),
            retrieve_phase_method: "none".into(),
            num_iter: 1,
            nsino_per_chunk: 8,
            save_format: "tiff".into(),
        }
    }
}

impl Config {
    /// Serialize to a TOML string.
    pub fn to_toml(&self) -> anyhow::Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Write the config to `path`.
    pub fn write(&self, path: &Path) -> anyhow::Result<()> {
        std::fs::write(path, self.to_toml()?)?;
        Ok(())
    }

    /// Load a config from `path`.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }
}
