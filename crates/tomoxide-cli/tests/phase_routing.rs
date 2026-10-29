//! Phase retrieval couples detector rows (`PrepOptions::row_separable`), so
//! `recon` must take the whole-volume path for a phase plan even where it would
//! otherwise stream, and the streaming-only `recon_steps` must refuse one. A
//! `--start_row/--end_row` band must still see the rows around it.

use std::process::{Command, Output};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tomoxide/tests/fixtures/streaming_dxchange.h5"
);

fn out_base(out: &str) -> String {
    format!("{}/{out}", env!("CARGO_TARGET_TMPDIR"))
}

fn paganin_fbp(backend: &str, subcommand: &str, out: &str, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tomoxide"))
        .args(["--backend", backend, subcommand, FIXTURE])
        .args(["--algorithm", "fbp", "--retrieve_phase", "paganin"])
        .args(["--save_format", "h5", "--output", &out_base(out)])
        .args(extra)
        .output()
        .unwrap()
}

/// The default Paganin kernel spans 65 rows, more than the 6-row fixture, so a
/// band's prep reads the whole stack and its rows are the full run's.
#[test]
fn phase_band_reconstructs_the_full_runs_rows() {
    for (out, extra) in [
        ("phase_full", &[][..]),
        ("phase_band", &["--start_row", "2", "--end_row", "4"][..]),
    ] {
        let o = paganin_fbp("cpu", "recon", out, extra);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    let rows = |out| {
        let path = format!("{}.h5", out_base(out));
        tomoxide::io::read_h5_band(&path, "/exchange/data", 2, 4)
            .unwrap()
            .3
    };
    let (full, band) = (rows("phase_full"), rows("phase_band"));
    let scale = full.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let diff = full
        .iter()
        .zip(&band)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        diff <= 1e-5 * scale,
        "band rows differ by {diff} (scale {scale})"
    );
}

#[test]
fn recon_steps_refuses_phase_and_points_to_recon() {
    let o = paganin_fbp("cpu", "recon_steps", "phase_recon_steps", &[]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success());
    assert!(err.contains("supported only by `recon`"), "{err}");
}

/// CUDA streams fbp, which is how a phase plan used to reach the row-chunked
/// pipeline and fail there.
#[cfg(feature = "cuda")]
#[test]
fn cuda_recon_takes_phase_through_the_whole_volume_path() {
    let o = paganin_fbp("cuda", "recon", "phase_recon_cuda", &[]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}
