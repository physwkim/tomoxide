//! `tomoxide::config::Config` (behind the `config` feature): TOML round-trip
//! and the lenient-load properties the CLI and GUI recipes rely on.
#![cfg(feature = "config")]

use std::path::PathBuf;
use tomoxide::config::Config;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tomoxide-config-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn write_load_round_trip_preserves_fields() {
    let dir = temp_dir("roundtrip");
    let path = dir.join("recipe.toml");
    let cfg = Config {
        file_name: "scan.h5".into(),
        algorithm: "sirt".into(),
        rotation_axis: Some(511.5),
        reg_par: vec![0.5, 0.01],
        remove_stripe_method: "fw".into(),
        fw_sigma: 3.0,
        ..Default::default()
    };
    cfg.write(&path).unwrap();

    let back = Config::load(&path).unwrap();
    assert_eq!(back.file_name, "scan.h5");
    assert_eq!(back.algorithm, "sirt");
    assert_eq!(back.rotation_axis, Some(511.5));
    assert_eq!(back.reg_par, vec![0.5, 0.01]);
    assert_eq!(back.remove_stripe_method, "fw");
    assert_eq!(back.fw_sigma, 3.0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn missing_keys_take_defaults_and_unknown_keys_are_ignored() {
    // A partial file (old template or hand-written) plus an extra table a GUI
    // might append — both must load.
    let dir = temp_dir("partial");
    let path = dir.join("partial.toml");
    std::fs::write(
        &path,
        "algorithm = \"gridrec\"\nnot_a_config_key = true\n\n[gui]\nlast_slice = 42\n",
    )
    .unwrap();
    let cfg = Config::load(&path).unwrap();
    assert_eq!(cfg.algorithm, "gridrec");
    assert_eq!(cfg.filter_name, "parzen"); // default
    assert_eq!(cfg.nsino_per_chunk, 8); // default
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn malformed_toml_is_an_invalid_param_error() {
    let dir = temp_dir("malformed");
    let path = dir.join("broken.toml");
    std::fs::write(&path, "algorithm = [unclosed").unwrap();
    let err = Config::load(&path).unwrap_err();
    assert!(
        matches!(err, tomoxide::Error::InvalidParam(_)),
        "got: {err}"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
