//! Embeds the git commit in `tomoxide --version` as `TOMOXIDE_VERSION`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

fn main() {
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    // Only tomoxide's own checkout: an unpacked or vendored copy can sit inside
    // some other repository, whose commit this is not.
    let rev = match git(&dir, &["rev-parse", "--show-prefix"]) {
        Some(prefix) if prefix == "crates/tomoxide-cli/" => git(
            &dir,
            &["describe", "--tags", "--always", "--dirty", "--abbrev=12"],
        ),
        _ => None,
    };
    let Some(rev) = rev else {
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rustc-env=TOMOXIDE_VERSION={version}");
        return;
    };
    // The reflog moves on every commit, checkout and reset (a soft reset
    // leaves HEAD and the index alone); the sources make `-dirty` follow edits.
    if let Some(git_dir) = git(&dir, &["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/logs/HEAD");
        println!("cargo:rerun-if-changed={git_dir}/index");
    }
    let root = dir.join("../..");
    for path in [
        "crates/tomoxide",
        "crates/tomoxide-cli",
        "Cargo.toml",
        "Cargo.lock",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    println!("cargo:rustc-env=TOMOXIDE_VERSION={version} ({rev})");
}
