//! Recipe files (docs/GUI.md §5): the GUI recipe IS the CLI config TOML —
//! `tomoxide::config::Config` fields at the top level, consumable directly by
//! `tomoxide-cli --config`, plus one trailing `[gui]` table with GUI-only
//! state (the CLI ignores unknown tables).

use serde::{Deserialize, Serialize};
use std::path::Path;
use tomoxide::config::Config;

/// GUI-only recipe state, stored under `[gui]`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GuiState {
    /// Tune screen's preview slice (detector row).
    pub slice: usize,
}

/// A recipe: the shared CLI config plus the `[gui]` table.
///
/// `gui` is declared last so the flattened `Config` scalars serialize before
/// the nested table (TOML requires values before tables).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Recipe {
    #[serde(flatten)]
    pub config: Config,
    #[serde(default)]
    pub gui: GuiState,
}

impl Recipe {
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let text = toml::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip through a file; the `[gui]` table survives next to the
    /// flattened Config (guards the toml+serde(flatten) combination).
    #[test]
    fn recipe_roundtrips_with_gui_table() {
        let mut recipe = Recipe::default();
        recipe.config.file_name = "/data/scan.h5".into();
        recipe.config.algorithm = "sirt".into();
        recipe.config.rotation_axis = Some(1023.75);
        recipe.config.reg_par = vec![0.5, 0.01];
        recipe.gui.slice = 42;

        let dir = std::env::temp_dir().join(format!("tomoxide-gui-recipe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("recipe.toml");
        recipe.save(&path).unwrap();

        let back = Recipe::load(&path).unwrap();
        assert_eq!(back.config.file_name, "/data/scan.h5");
        assert_eq!(back.config.algorithm, "sirt");
        assert_eq!(back.config.rotation_axis, Some(1023.75));
        assert_eq!(back.config.reg_par, vec![0.5, 0.01]);
        assert_eq!(back.gui.slice, 42);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A recipe written by the GUI parses as a bare CLI Config (the [gui]
    /// table is an ignorable unknown key) — the CLI-shareability contract.
    #[test]
    fn recipe_file_loads_as_cli_config() {
        let mut recipe = Recipe::default();
        recipe.config.algorithm = "lprec".into();
        recipe.gui.slice = 7;

        let dir = std::env::temp_dir().join(format!("tomoxide-gui-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("recipe.toml");
        recipe.save(&path).unwrap();

        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.algorithm, "lprec");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A plain CLI config (no [gui] table) loads as a recipe with defaults.
    #[test]
    fn plain_config_loads_as_recipe() {
        let dir = std::env::temp_dir().join(format!("tomoxide-gui-plain-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let cfg = Config {
            algorithm: "fourierrec".into(),
            ..Default::default()
        };
        cfg.write(&path).unwrap();

        let recipe = Recipe::load(&path).unwrap();
        assert_eq!(recipe.config.algorithm, "fourierrec");
        assert_eq!(recipe.gui.slice, 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
