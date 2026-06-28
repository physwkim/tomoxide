//! Parity check for on-device stripe removal in the streaming path.
//!
//! `StreamingAnalytic::reconstruct_chunk_raw` applies any GPU-ported stripe
//! method to the transposed f32 sinogram on the device. The result must match
//! the host route (`normalize_dataset` + `to_layout(Sinogram)` +
//! `remove_stripe` + `reconstruct_chunk`). Unlike the no-stripe device path this
//! is *not* bit-exact: the GPU stripe kernels use parallel reductions (the
//! Titarenko CG uses block-wide f64 dot products) that reassociate sums versus
//! the serial CPU golden, so each method is held to a high correlation, not a
//! zero diff.
//!
//! For one reconstructor this runs both routes on identical input and reports
//! the correlation across: fp32 / fp16, with / without flat-dark, and a full
//! (nz == max_nz) and a partial (nz < max_nz) chunk.
//!
//!   cargo run --release --features cuda --example parity_stripe -- [nproj] [maxnz] [ncols]

use ndarray::Array3;
use tomoxide::params::StripeMethod;
use tomoxide::{
    Algorithm, Angles, BackendKind, Dataset, Dtype, Engine, Frames, Geometry, Layout, ReconParams,
    Tomo, Volume,
};

/// Synthetic raw transmission projection chunk `[nproj, nz, ncols]` with a
/// per-column multiplicative stripe baked in, so stripe removal has a real ring
/// artefact to correct. Values stay in (0.1, 0.9] for a finite minus-log.
fn make_proj(nproj: usize, nz: usize, ncols: usize) -> Array3<f32> {
    Array3::from_shape_fn((nproj, nz, ncols), |(p, z, x)| {
        let base =
            0.5 + 0.4 * ((p as f32 * 0.017 + x as f32 * 0.013 + z as f32 * 0.011).sin() * 0.5);
        // Deterministic per-column stripe (a few percent gain variation).
        let stripe = 1.0 + 0.06 * (((x * 7 + 13) % 17) as f32 / 17.0 - 0.5);
        (base * stripe).clamp(0.1, 0.9)
    })
}

fn pearson(a: &Volume<f32>, b: &Volume<f32>) -> f64 {
    let av: Vec<f32> = a.array.iter().copied().collect();
    let bv: Vec<f32> = b.array.iter().copied().collect();
    let n = av.len() as f64;
    let (ma, mb) = (
        av.iter().map(|&v| v as f64).sum::<f64>() / n,
        bv.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (&x, &y) in av.iter().zip(&bv) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    cov / (va.sqrt() * vb.sqrt())
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
    geom: &Geometry,
    theta: &[f32],
    nproj: usize,
    max_nz: usize,
    ncols: usize,
    nz: usize,
    dtype: Dtype,
    with_flatdark: bool,
    stripe: StripeMethod,
) -> (f64, f64, bool) {
    let backend = engine.backend();
    let params = ReconParams {
        num_gridx: Some(ncols),
        dtype,
        ..Default::default()
    };
    let ar = backend
        .analytic_reconstruct()
        .expect("cuda analytic_reconstruct");
    let mut recon = ar
        .streaming(Algorithm::Fbp, &params, geom, ncols, max_nz)
        .expect("streaming() ok")
        .expect("cuda provides a streaming reconstructor");

    let raw = make_proj(nproj, nz, ncols);
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

    // Path A — host reference: normalize, transpose, host stripe, reconstruct.
    let mut ds = Dataset {
        data: Tomo::new(raw.clone(), Layout::Projection),
        flat: flat.clone(),
        dark: dark.clone(),
        theta: theta.to_vec(),
    };
    tomoxide::prep::normalize_dataset(&mut ds, backend).expect("normalize");
    let mut sino = ds.data.to_layout(Layout::Sinogram);
    sino.array = sino.array.as_standard_layout().to_owned();
    tomoxide::prep::remove_stripe(&mut sino, stripe).expect("host remove_stripe");
    let vol_host = recon
        .reconstruct_chunk(&sino, geom)
        .expect("reconstruct_chunk");

    // Path B — device-resident raw path with on-device stripe removal.
    let raw_tomo = Tomo::new(raw, Layout::Projection);
    let vol_dev = recon
        .reconstruct_chunk_raw(&raw_tomo, flat.as_ref(), dark.as_ref(), geom, stripe)
        .expect("reconstruct_chunk_raw ok")
        .expect("cuda returns Some for a GPU-ported stripe method");

    let r = pearson(&vol_host, &vol_dev);
    let d = max_abs_diff(&vol_host, &vol_dev);
    (r, d, r > 0.9999)
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

    let methods: &[(&str, StripeMethod)] = &[(
        "ti",
        StripeMethod::Ti {
            nblock: 0,
            beta: 1.5,
        },
    )];

    println!("parity_stripe: nproj={nproj} max_nz={max_nz} ncols={ncols}");
    let mut all_ok = true;
    for &(mname, stripe) in methods {
        for &dtype in &[Dtype::F32, Dtype::F16] {
            for &with_fd in &[false, true] {
                for &(label, nz) in &[("full", max_nz), ("partial", max_nz * 5 / 8)] {
                    let (r, d, ok) = run_case(
                        &engine, &geom, &theta, nproj, max_nz, ncols, nz, dtype, with_fd, stripe,
                    );
                    all_ok &= ok;
                    println!(
                        "  {mname:<4} {:>7} dtype={:<5} flatdark={:<5} nz={:<4} pearson={:.6} max_abs={:.3e}  {}",
                        label,
                        format!("{dtype:?}"),
                        with_fd,
                        nz,
                        r,
                        d,
                        if ok { "PASS" } else { "FAIL" },
                    );
                }
            }
        }
    }
    println!(
        "=> {}",
        if all_ok {
            "ALL PASS — device stripe == host stripe (correlation parity)"
        } else {
            "FAILURE — device stripe diverges from host"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
}
