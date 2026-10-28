//! Phase retrieval couples detector rows (`PrepOptions::row_separable`), so
//! `recon` must take the whole-volume path for a phase plan even where it would
//! otherwise stream, and the streaming-only `recon_steps` must refuse one.

use std::process::{Command, Output};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tomoxide/tests/fixtures/streaming_dxchange.h5"
);

fn paganin_fbp(backend: &str, subcommand: &str, out: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tomoxide"))
        .args(["--backend", backend, subcommand, FIXTURE])
        .args(["--algorithm", "fbp", "--retrieve_phase", "paganin"])
        .args(["--save_format", "h5", "--output"])
        .arg(std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(out))
        .output()
        .unwrap()
}

#[test]
fn recon_steps_refuses_phase_and_points_to_recon() {
    let o = paganin_fbp("cpu", "recon_steps", "phase_recon_steps");
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success());
    assert!(err.contains("supported only by `recon`"), "{err}");
}

/// CUDA streams fbp, which is how a phase plan used to reach the row-chunked
/// pipeline and fail there.
#[cfg(feature = "cuda")]
#[test]
fn cuda_recon_takes_phase_through_the_whole_volume_path() {
    let o = paganin_fbp("cuda", "recon", "phase_recon_cuda");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}
