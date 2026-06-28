//! Bit-exactness check for the device-resident streaming path.
//!
//! `StreamingAnalytic::reconstruct_chunk_raw` does dark/flat correction,
//! minus-log, and the projection→sinogram transpose **on the GPU** (one upload,
//! one download). It must produce the *same* volume as the host path
//! (`normalize_dataset` + `to_layout(Sinogram)` + `reconstruct_chunk`), because
//! it reuses the very same darkflat/minus-log kernels (host-averaged dark2d /
//! clamped denom) and the transpose is a pure index reorder.
//!
//! For one reconstructor this runs both paths on identical input and reports the
//! max abs diff (expected 0) across: fp32 / fp16, with / without flat-dark, and
//! a full (nz == max_nz) and a partial (nz < max_nz) chunk.
//!
//!   cargo run --release --features cuda --example parity_raw -- [nproj] [maxnz] [ncols]

use ndarray::Array3;
use tomoxide::{
    Algorithm, Angles, BackendKind, Dataset, Dtype, Engine, Frames, Geometry, Layout, ReconParams,
    Tomo, Volume,
};

/// Synthetic raw transmission projection chunk `[nproj, nz, ncols]`, values in
/// (0.1, 0.9] so minus-log is finite and non-trivial.
fn make_proj(nproj: usize, nz: usize, ncols: usize) -> Array3<f32> {
    Array3::from_shape_fn((nproj, nz, ncols), |(p, z, x)| {
        let v = 0.5
            + 0.4 * ((p as f32 * 0.017 + x as f32 * 0.013 + z as f32 * 0.011).sin() * 0.5 + 0.5)
            - 0.2;
        v.clamp(0.1, 0.9)
    })
}

fn max_abs_diff(a: &Volume<f32>, b: &Volume<f32>) -> f64 {
    a.array
        .iter()
        .zip(b.array.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0.0f64, f64::max)
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    engine: &Engine,
    geom_full: &Geometry,
    theta: &[f32],
    nproj: usize,
    max_nz: usize,
    ncols: usize,
    nz: usize,
    dtype: Dtype,
    with_flatdark: bool,
) -> (f64, bool) {
    let backend = engine.backend();
    let params = ReconParams {
        num_gridx: Some(ncols),
        dtype,
        ..Default::default()
    };
    // One reconstructor sized to the full chunk; both paths reuse it.
    let ar = backend
        .analytic_reconstruct()
        .expect("cuda analytic_reconstruct");
    let mut recon = ar
        .streaming(Algorithm::Fbp, &params, geom_full, ncols, max_nz)
        .expect("streaming() ok")
        .expect("cuda provides a streaming reconstructor");

    // Geometry restricted to this chunk's rows (scalar center → unchanged).
    let geom = geom_full;

    let raw = make_proj(nproj, nz, ncols);
    // Non-trivial flat/dark: flat slightly >1, dark small >0, so darkflat is a
    // real correction that keeps the corrected transmission positive.
    let (flat, dark) = if with_flatdark {
        let flat = Frames::new(Array3::from_shape_fn((3, nz, ncols), |(_f, _z, x)| {
            1.0 + 0.02 * ((x as f32) * 0.001).cos()
        }));
        let dark = Frames::new(Array3::from_shape_fn((2, nz, ncols), |(_d, z, _x)| {
            0.01 + 0.005 * (z as f32 * 0.003).sin().abs()
        }));
        (Some(flat), Some(dark))
    } else {
        (None, None)
    };

    // Path A — host reference: normalize on a Dataset, transpose, reconstruct.
    let mut ds = Dataset {
        data: Tomo::new(raw.clone(), Layout::Projection),
        flat: flat.clone(),
        dark: dark.clone(),
        theta: theta.to_vec(),
    };
    tomoxide::prep::normalize_dataset(&mut ds, backend).expect("normalize");
    let mut sino = ds.data.to_layout(Layout::Sinogram);
    sino.array = sino.array.as_standard_layout().to_owned();
    let vol_host = recon
        .reconstruct_chunk(&sino, geom)
        .expect("reconstruct_chunk");

    // Path B — device-resident raw path on the un-normalized projection chunk.
    let raw_tomo = Tomo::new(raw, Layout::Projection);
    let vol_raw = recon
        .reconstruct_chunk_raw(&raw_tomo, flat.as_ref(), dark.as_ref(), geom)
        .expect("reconstruct_chunk_raw ok")
        .expect("cuda returns Some from reconstruct_chunk_raw");

    let d = max_abs_diff(&vol_host, &vol_raw);
    (d, d == 0.0)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let nproj: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let max_nz: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let ncols: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);

    let engine = Engine::new(BackendKind::Cuda).expect("cuda engine");
    let theta: Vec<f32> = (0..nproj)
        .map(|p| p as f32 * std::f32::consts::PI / nproj as f32)
        .collect();
    let geom = Geometry::parallel(Angles(theta.clone()), ncols, max_nz, 1.0);

    println!("parity_raw: nproj={nproj} max_nz={max_nz} ncols={ncols}");
    let mut all_ok = true;
    for &dtype in &[Dtype::F32, Dtype::F16] {
        for &with_fd in &[false, true] {
            for &(label, nz) in &[("full", max_nz), ("partial", max_nz * 5 / 8)] {
                let (d, ok) = run_case(
                    &engine, &geom, &theta, nproj, max_nz, ncols, nz, dtype, with_fd,
                );
                all_ok &= ok;
                println!(
                    "  {:>7} dtype={:<5} flatdark={:<5} nz={:<4} max_abs_diff={:.3e}  {}",
                    label,
                    format!("{dtype:?}"),
                    with_fd,
                    nz,
                    d,
                    if ok { "PASS (bit-exact)" } else { "FAIL" },
                );
            }
        }
    }
    println!(
        "=> {}",
        if all_ok {
            "ALL PASS — device-raw == host, bit-exact"
        } else {
            "FAILURE — device-raw diverges from host"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
}
