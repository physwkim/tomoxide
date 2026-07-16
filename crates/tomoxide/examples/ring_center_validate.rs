//! Validate `find_center_rings` against the two pouch scans that
//! `docs/LAMINOGRAPHY_ALIGNMENT.md` is written from.
//!
//! ```text
//! cargo run --release -p tomoxide --example ring_center_validate -- \
//!     /path/to/pouch_bin1.h5 396   /path/to/pouch06_bin1.h5 138
//! ```
//!
//! Expected — the doc's reference implementation, reproduced in Python on the
//! same two files:
//!   aligned    0.55 s: centre 397.50 (truth 396), prominence ≈ 21.0 → trustworthy
//!   misaligned 0.6 s : centre 281.09 (truth 138), prominence ≈  2.4 → NOT trustworthy

use tomoxide::prep::normalize::{minus_log, normalize_dataset};
use tomoxide::recon::center::find_center_rings;
use tomoxide::{CpuBackend, Result};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() % 2 != 0 {
        eprintln!("usage: ring_center_validate <h5> <truth> [<h5> <truth> ...]");
        std::process::exit(2);
    }
    let cpu = CpuBackend::new();
    for pair in args.chunks(2) {
        let (path, truth) = (&pair[0], pair[1].parse::<f32>().unwrap_or(f32::NAN));
        let t0 = std::time::Instant::now();
        let mut reader = tomoxide::io::open_dxchange(path)?;
        let mut ds = reader.read_all()?;
        normalize_dataset(&mut ds, &cpu)?;
        minus_log(&mut ds.data, &cpu)?;
        let load = t0.elapsed();

        let t1 = std::time::Instant::now();
        let r = find_center_rings(&ds.data, &cpu, 10)?;
        println!(
            "{path}\n  centre      = {:.2}   (truth {truth})   Δ = {:+.2}\n  \
             prominence  = {:.2}\n  trustworthy = {}\n  load {:?}, estimate {:?}",
            r.center,
            r.center - truth,
            r.prominence,
            r.trustworthy,
            load,
            t1.elapsed()
        );
    }
    Ok(())
}
