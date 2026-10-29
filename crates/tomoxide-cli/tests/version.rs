//! `tomoxide --version` names the commit the binary was built from.

use std::process::Command;

#[test]
fn version_names_the_checkout_commit() {
    let out = Command::new(env!("CARGO_BIN_EXE_tomoxide"))
        .arg("--version")
        .output()
        .unwrap();
    let line = String::from_utf8(out.stdout).unwrap();
    let head = Command::new("git")
        .args([
            "-C",
            env!("CARGO_MANIFEST_DIR"),
            "rev-parse",
            "--short=12",
            "HEAD",
        ])
        .output()
        .unwrap();
    let head = String::from_utf8(head.stdout).unwrap();
    assert!(
        line.starts_with(concat!("tomoxide ", env!("CARGO_PKG_VERSION"), " (")),
        "{line}"
    );
    assert!(line.contains(head.trim()), "{line} names no {head}");
}
