//! `tomoxide` — command-line front-end (ports tomocupy `__main__.py`:
//! `init` / `recon` / `recon_steps` / `status`).

mod config;

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use clap::{Parser, Subcommand};
use tomoxide::io::DatasetReader;
use tomoxide::Algorithm;
use tomoxide::Center;
use tomoxide::{Angles, BackendKind, Dtype, Engine, Geometry, ReconParams};

use crate::config::Config;

/// GPU/CPU tomographic reconstruction (tomopy + tomocupy, in Rust).
#[derive(Parser, Debug)]
#[command(name = "tomoxide", version, about)]
struct Cli {
    /// Backend: auto | cpu | cuda | wgpu.
    #[arg(long, global = true, default_value = "auto")]
    backend: String,
    /// Verbose logging.
    #[arg(long, short, global = true)]
    verbose: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
#[command(rename_all = "snake_case")] // match tomocupy command names (recon_steps)
enum Command {
    /// Write a default configuration file.
    Init {
        /// Output config path.
        #[arg(long, default_value = "tomoxide.toml")]
        config: PathBuf,
    },
    /// Show the resolved backend and configuration.
    Status {
        /// Config file to display (optional).
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Full in-memory reconstruction.
    Recon {
        /// Input DXchange HDF5 file.
        file: PathBuf,
        /// Algorithm (fbp, gridrec, fourierrec, lprec, linerec, sirt, …).
        #[arg(long, default_value = "fbp")]
        algorithm: String,
        /// Rotation-axis column (omit to auto-find).
        #[arg(long)]
        center: Option<f32>,
        /// Reconstruction precision: float32 | float16 (CUDA fbp/linerec/
        /// fourierrec only).
        #[arg(long, default_value = "float32")]
        dtype: String,
        /// Output format: tiff | h5 | zarr (tomocupy `--save-format`).
        #[arg(long, default_value = "tiff")]
        save_format: String,
    },
    /// Chunked / streaming reconstruction (out-of-core).
    ReconSteps {
        /// Input DXchange HDF5 file.
        file: PathBuf,
        /// Algorithm (fbp, gridrec, fourierrec, lprec, linerec, sirt, …).
        #[arg(long, default_value = "fbp")]
        algorithm: String,
        /// Rotation-axis column (omit to use the detector midline).
        #[arg(long)]
        center: Option<f32>,
        /// Reconstruction precision: float32 | float16 (CUDA only).
        #[arg(long, default_value = "float32")]
        dtype: String,
        /// Output format: tiff | h5 | zarr (tomocupy `--save-format`).
        #[arg(long, default_value = "tiff")]
        save_format: String,
        /// Detector rows (slices) reconstructed/written per pipeline chunk
        /// (tomocupy `--nsino-per-chunk`). Smaller = more read/compute/write
        /// overlap; larger = fewer per-chunk launches.
        #[arg(long, default_value_t = DEFAULT_PIPELINE_CHUNK)]
        chunk: usize,
    },
}

/// Default z-rows per streaming pipeline chunk. Small enough that the
/// read‖compute‖write conveyor overlaps well across a typical `nz` (≈128), large
/// enough that per-chunk launch/setup overhead stays amortized. Mirrors
/// tomocupy's `--nsino-per-chunk` default magnitude.
const DEFAULT_PIPELINE_CHUNK: usize = 8;

/// Whether `recon` should reconstruct through the overlapped streaming pipeline
/// instead of the whole-volume path.
///
/// The pipeline overlaps disk read, GPU compute, and disk write and runs
/// normalize/minus-log/transpose **on the device** (one PCIe round-trip), so for
/// the per-slice-independent GPU analytic methods it is a strict win over the
/// serial whole-volume path (which transposes the full projection array on the
/// host before upload). It pays off only when the per-chunk GPU work dominates
/// the per-chunk host setup:
///
/// - **Fbp / Linerec**: device-resident streaming reconstructor (GPU
///   normalize/transpose + reused cuFFT/back-projection handles) — large win.
/// - **Fourierrec**: per-chunk GPU reconstruct (host normalize/transpose, but
///   cuFFT plans are thread-cached across chunks) — moderate win.
/// - **Lprec**: device-resident streaming reconstructor (GPU spline prefilter +
///   gather/FFT/scatter, log-polar grids uploaded once and reused) — the
///   whole-volume path otherwise pays a full-volume host transpose.
///
/// `Gridrec` does a host gather/deapodize per reconstruct call, so chunking
/// multiplies that host round-trip and makes the pipeline *slower* than
/// whole-volume — it stays on the whole-volume path. CPU/wgpu backends have no
/// device-resident path either, so they also stay whole-volume.
fn pipelines_well(engine: &Engine, algo: Algorithm) -> bool {
    engine.name() == "cuda"
        && matches!(
            algo,
            Algorithm::Fbp | Algorithm::Linerec | Algorithm::Fourierrec | Algorithm::Lprec
        )
}

fn parse_backend(s: &str) -> anyhow::Result<BackendKind> {
    Ok(match s {
        "auto" => BackendKind::Auto,
        "cpu" => BackendKind::Cpu,
        "cuda" => BackendKind::Cuda,
        "wgpu" => BackendKind::Wgpu,
        other => return Err(anyhow!("unknown backend '{other}' (auto|cpu|cuda|wgpu)")),
    })
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level)).init();

    let backend_kind = parse_backend(&cli.backend)?;

    match cli.command {
        Command::Init { config } => {
            let cfg = Config::default();
            cfg.write(&config)
                .with_context(|| format!("writing {}", config.display()))?;
            println!("wrote default config to {}", config.display());
        }
        Command::Status { config } => {
            let engine = Engine::new(backend_kind)?;
            println!("tomoxide {}", env!("CARGO_PKG_VERSION"));
            println!("backend (requested {}): {}", cli.backend, engine.name());
            if let Some(path) = config {
                let cfg =
                    Config::load(&path).with_context(|| format!("loading {}", path.display()))?;
                println!("config: {cfg:#?}");
            }
        }
        Command::Recon {
            file,
            algorithm,
            center,
            dtype,
            save_format,
        } => {
            let engine = Engine::new(backend_kind)?;
            let algo: Algorithm = algorithm.parse().map_err(|e| anyhow!("{e}"))?;
            let dtype: Dtype = dtype.parse().map_err(|e| anyhow!("{e}"))?;
            let save_format: tomoxide::io::SaveFormat =
                save_format.parse().map_err(|e| anyhow!("{e}"))?;
            println!(
                "recon: file={} algorithm={:?} center={:?} dtype={} backend={}",
                file.display(),
                algo,
                center,
                dtype.as_str(),
                engine.name()
            );
            let out = recon_out_path(&file);
            if pipelines_well(&engine, algo) {
                // Overlapped streaming path: same output as the whole-volume path
                // (cuFFT-floor identical, Pearson 1.0), lower peak memory, and it
                // hides disk read/write behind GPU compute. See `pipelines_well`.
                run_pipelined(
                    &file,
                    &out,
                    algo,
                    center,
                    dtype,
                    save_format,
                    DEFAULT_PIPELINE_CHUNK,
                    &engine,
                )?;
            } else {
                // Whole-volume path (CPU/wgpu, or chunking-hostile GPU methods).
                let mut reader = tomoxide::io::open_dxchange(&file.to_string_lossy())?;
                let geom = geometry_from_reader(reader.as_mut(), center)?;
                let params = recon_params(&geom, dtype);
                let ds = reader.read_all()?;
                let vol = tomoxide::reconstruct(
                    ds,
                    &geom,
                    algo,
                    &params,
                    &tomoxide::PrepOptions::default(),
                    &engine,
                )?;
                let mut writer = tomoxide::io::create_writer(&out, save_format)?;
                let nz = vol.dims().0;
                writer.reserve(nz)?;
                writer.write_chunk(&vol, 0, nz)?;
            }
            println!("wrote reconstruction to {out}");
        }
        Command::ReconSteps {
            file,
            algorithm,
            center,
            dtype,
            save_format,
            chunk,
        } => {
            let engine = Engine::new(backend_kind)?;
            let algo: Algorithm = algorithm.parse().map_err(|e| anyhow!("{e}"))?;
            let dtype: Dtype = dtype.parse().map_err(|e| anyhow!("{e}"))?;
            let save_format: tomoxide::io::SaveFormat =
                save_format.parse().map_err(|e| anyhow!("{e}"))?;
            println!(
                "recon_steps: file={} algorithm={:?} center={:?} dtype={} chunk={} backend={}",
                file.display(),
                algo,
                center,
                dtype.as_str(),
                chunk,
                engine.name()
            );
            let out = recon_out_path(&file);
            run_pipelined(
                &file,
                &out,
                algo,
                center,
                dtype,
                save_format,
                chunk,
                &engine,
            )?;
            println!("wrote streamed reconstruction to {out}");
        }
    }
    Ok(())
}

/// Run the overlapped read‖compute‖write streaming pipeline for one file.
///
/// Probes geometry (metadata only) on the calling thread, then hands
/// reader/writer **factories** to [`ReconSteps::run_streaming_pipelined`] — the
/// `rust-hdf5` handles are `!Send`, so each I/O object is built on the thread
/// that owns it. Shared by `recon` (auto-pipelined GPU path) and `recon_steps`.
#[allow(clippy::too_many_arguments)]
fn run_pipelined(
    file: &Path,
    out: &str,
    algo: Algorithm,
    center: Option<f32>,
    dtype: Dtype,
    save_format: tomoxide::io::SaveFormat,
    chunk: usize,
    engine: &Engine,
) -> anyhow::Result<()> {
    let path = file.to_string_lossy().into_owned();
    // Probe geometry from a short-lived reader open (metadata only); the pipeline
    // builds its own reader on the reader thread.
    let mut probe = tomoxide::io::open_dxchange(&path)?;
    let geom = geometry_from_reader(probe.as_mut(), center)?;
    drop(probe);
    let params = recon_params(&geom, dtype);
    let read_path = path;
    let write_path = out.to_string();
    tomoxide::ReconSteps::new(chunk).run_streaming_pipelined(
        move || tomoxide::io::open_dxchange(&read_path),
        move || tomoxide::io::create_writer(&write_path, save_format),
        &geom,
        algo,
        &params,
        &tomoxide::PrepOptions::default(),
        engine,
    )?;
    Ok(())
}

/// Build a parallel-beam geometry from the reader's sizes/angles, optionally
/// overriding the rotation center (else the detector midline).
fn geometry_from_reader(
    reader: &mut dyn DatasetReader,
    center: Option<f32>,
) -> anyhow::Result<Geometry> {
    let (_nproj, nz, nx, _nflat, _ndark) = reader.read_sizes()?;
    let theta = reader.read_theta()?;
    let mut geom = Geometry::parallel(Angles(theta), nx, nz, 1.0);
    if let Some(c) = center {
        geom.center = Center::Scalar(c);
    }
    Ok(geom)
}

/// Reconstruction params with the grid sized to the detector width.
fn recon_params(geom: &Geometry, dtype: Dtype) -> ReconParams {
    ReconParams {
        num_gridx: Some(geom.detector.width),
        dtype,
        ..Default::default()
    }
}

/// Output path for a reconstruction: `<input-without-extension>_rec`.
fn recon_out_path(file: &Path) -> String {
    format!("{}_rec", file.with_extension("").display())
}
