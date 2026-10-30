//! `tomoxide --version` names the commit the binary was built from.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .unwrap();
    out.status
        .success()
        .then(|| String::from_utf8(out.stdout).unwrap().trim().to_owned())
}

#[test]
fn version_names_the_checkout_commit() {
    let out = Command::new(env!("CARGO_BIN_EXE_tomoxide"))
        .arg("--version")
        .output()
        .unwrap();
    let line = String::from_utf8(out.stdout).unwrap();
    let rev = line
        .trim()
        .strip_prefix(concat!("tomoxide ", env!("CARGO_PKG_VERSION"), " ("))
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or_else(|| panic!("{line}"));
    let rev = rev.strip_suffix("-dirty").unwrap_or(rev);
    let commit = |name: &str| git(&["rev-parse", "--verify", &format!("{name}^{{commit}}")]);
    assert_eq!(commit(rev), commit("HEAD"), "{line}");
    // Release tags are lightweight; a shallow clone has none.
    if let Some(tag) = git(&["describe", "--tags", "--abbrev=0"]) {
        assert!(
            rev == tag || rev.starts_with(&format!("{tag}-")),
            "{line} names no {tag}"
        );
    }
}
