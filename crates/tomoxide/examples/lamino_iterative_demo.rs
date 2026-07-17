//! Iterative laminography reconstruction on a real DXchange dataset, for
//! comparing SIRT/TV against the analytic (fourierrec/linerec) path under the
//! high-noise / large-axis-wobble regime where iterative regularization is
//! expected to help.
//!
//! The CLI routes CUDA laminography through `reconstruct_lamino_streaming`,
//! which only implements the analytic methods (fourierrec/fbp/linerec). The
//! ported iterative laminography (SIRT/MLEM/OSEM/PML/TV) lives behind the
//! generic host solvers composed over the tilted CUDA projector pair
//! (`recon::recon` → `iterative`), which the CLI does not expose. This example
//! drives that path directly, replicating `pipeline::reconstruct`'s prep
//! (flat/dark + minus-log, no phase/stripe) before the solve.
//!
//! NOTE on volume height: the generic solvers reconstruct at volume-height =
//! detector rows (`nz`), NOT the tilt-extended analytic recon-height `rh`
//! (`ceil(nz/cos(tilt)/2)*2`). So run the analytic baseline with
//! `--lamino_rh <nz>` to get a matching-height, z-aligned comparison.
//!
//!   cargo run --release --features cuda --example lamino_iterative_demo -- \
//!       <input.h5> <sirt|tv|mlem> <num_iter> <center> <tilt_deg> <reg_par> <out.h5>
//!
//! `reg_par` is the TV strength λ (ignored by SIRT/MLEM; pass 0).

use std::f32::consts::PI;
use std::time::Instant;

use tomoxide::{
    io, prep, recon, Algorithm, Angles, BackendKind, Beam, Center, Engine, Geometry, Layout,
    ReconParams,
};

fn main() -> tomoxide::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 8 {
        eprintln!(
            "usage: {} <input.h5> <sirt|tv|mlem> <num_iter> <center> <tilt_deg> <reg_par> <out.h5>",
            a[0]
        );
        std::process::exit(2);
    }
    let input = &a[1];
    let algorithm = match a[2].as_str() {
        "sirt" => Algorithm::Sirt,
        "tv" => Algorithm::Tv,
        "mlem" => Algorithm::Mlem,
        other => {
            eprintln!("unknown algorithm {other:?} (sirt|tv|mlem)");
            std::process::exit(2);
        }
    };
    let num_iter: usize = a[3].parse().expect("num_iter");
    let center: f32 = a[4].parse().expect("center");
    let tilt_deg: f32 = a[5].parse().expect("tilt_deg");
    let reg_par: f32 = a[6].parse().expect("reg_par");
    let out = &a[7];

    let engine = Engine::new(BackendKind::Cuda)?;
    let backend = engine.backend();

    // Read the whole stack (lamino couples every detector row → no row banding).
    let mut reader = io::open_dxchange(input)?;
    let (nproj, nz, nx, _nflat, _ndark) = reader.read_sizes()?;
    let theta = reader.read_theta()?;
    let mut geom = Geometry::parallel(Angles(theta), nx, nz, 1.0);
    geom.center = Center::Scalar(center);
    geom.beam = Beam::Laminography {
        phi: PI / 2.0 + tilt_deg * PI / 180.0,
    };
    println!(
        "input={input} nproj={nproj} nz={nz} nx={nx} center={center} tilt={tilt_deg}° \
         algo={algorithm:?} num_iter={num_iter} reg_par={reg_par}"
    );
    println!("  volume height = nz = {nz} (generic solver ignores tilt-extended rh)");

    let mut ds = reader.read_all()?;

    // Same projection-domain prep as pipeline::reconstruct: flat/dark + minus-log
    // only (no phase, no stripe), then transpose to sinogram order for the solve.
    let t_prep = Instant::now();
    prep::normalize_dataset(&mut ds, backend)?;
    let sino = ds.data.to_layout(Layout::Sinogram);
    println!(
        "  prep (darkflat + minus-log) {:.1}s",
        t_prep.elapsed().as_secs_f64()
    );

    let params = ReconParams {
        num_iter,
        reg_par: vec![reg_par],
        ..Default::default()
    };

    let t_rec = Instant::now();
    let vol = recon::recon(&sino, &geom, algorithm, &params, backend)?;
    let secs = t_rec.elapsed().as_secs_f64();
    let (vz, vy, vx) = vol.dims();
    println!(
        "  recon {:.1}s ({:.2}s/iter) -> volume [{vz}, {vy}, {vx}]",
        secs,
        secs / num_iter.max(1) as f64
    );

    // Report the volume value range so the downstream TIFF window is grounded.
    let sl = vol.array.as_slice().unwrap();
    let (mut lo, mut hi, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
    for &v in sl {
        lo = lo.min(v);
        hi = hi.max(v);
        sum += v as f64;
    }
    println!(
        "  value range: min={lo:.5} max={hi:.5} mean={:.5}",
        sum / sl.len() as f64
    );

    let mut writer = io::create_writer(out, io::SaveFormat::H5)?;
    writer.reserve(vz)?;
    writer.write_chunk(&vol, 0, vz)?;
    writer.finalize()?;
    println!("  wrote {out}");
    Ok(())
}
