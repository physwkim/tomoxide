//! Validate `estimate_tilt_flow` against the 261003 step scan its Python
//! original was measured on.
//!
//! ```text
//! cargo run --release -p tomoxide --example tilt_flow_validate -- \
//!     /path/to/step1000_bin2.h5
//! ```
//!
//! The Python original on the same file: τ = 19.4 ± 1.5°, roll ≈ 0, axis
//! 480 ± 75 (nominal tilt 20°, rings axis 504).

use std::path::Path;

use tomoxide::align::PreppedStack;
use tomoxide::{CpuBackend, Result};

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: tilt_flow_validate <h5>");
        std::process::exit(2);
    };
    let cpu = CpuBackend::new();
    let t0 = std::time::Instant::now();
    let stack = PreppedStack::load(&cpu, Path::new(&path))?;
    let load = t0.elapsed();
    let t1 = std::time::Instant::now();
    let r = stack.tilt_flow(&cpu)?;
    let est = t1.elapsed();
    for p in &r.pairs {
        println!(
            "proj {:4} k={} dθ {:.2}°  blocks {:3}  τ {:5.2}°  roll {:+.2}°  axis {:6.1}  resid {:.2} px",
            p.first, p.gap, p.dtheta_deg, p.blocks, p.tilt_deg, p.roll_deg, p.axis_col, p.resid_px
        );
    }
    println!(
        "ALL  τ {:.2} ± {:.2}°   axis {:.1} ± {:.1}   roll {:.2} ± {:.2}°   ({} pairs)\n\
         rings axis {:.2}   load {load:?}, estimate {est:?}",
        r.tilt_deg,
        r.tilt_sigma_deg,
        r.axis_col,
        r.axis_col_sigma,
        r.roll_deg,
        r.roll_sigma_deg,
        r.pairs.len(),
        stack.rings(&cpu, 10)?.center,
    );
    Ok(())
}
