//! `tomoxide` — command-line front-end (ports tomocupy `__main__.py`:
//! `init` / `recon` / `recon_steps` / `status`).

mod config;

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};
use clap::{Parser, Subcommand};
use tomoxide::io::DatasetReader;
use tomoxide::{Angles, BackendKind, Engine, Geometry, ReconParams};
use tomoxide::Center;
use tomoxide::Algorithm;

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
    },
    /// Chunked / streaming reconstruction (out-of-core).
    ReconSteps {
        /// Input DXchange HDF5 file.
        file: PathBuf,
    },
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
        } => {
            let engine = Engine::new(backend_kind)?;
            let algo: Algorithm = algorithm.parse().map_err(|e| anyhow!("{e}"))?;
            println!(
                "recon: file={} algorithm={:?} center={:?} backend={}",
                file.display(),
                algo,
                center,
                engine.name()
            );
            let mut reader = tomoxide::io::open_dxchange(&file.to_string_lossy())?;
            let geom = geometry_from_reader(reader.as_mut(), center)?;
            let params = recon_params(&geom);
            let ds = reader.read_all()?;
            let vol = tomoxide::reconstruct(
                ds,
                &geom,
                algo,
                &params,
                &tomoxide::PrepOptions::default(),
                &engine,
            )?;
            let out = recon_out_path(&file);
            let mut writer = tomoxide::io::create_writer(&out, tomoxide::io::SaveFormat::Tiff)?;
            let nz = vol.dims().0;
            writer.write_chunk(&vol, 0, nz)?;
            println!("wrote {nz} reconstructed slices to {out}");
        }
        Command::ReconSteps { file } => {
            let engine = Engine::new(backend_kind)?;
            println!(
                "recon_steps: file={} backend={}",
                file.display(),
                engine.name()
            );
            let path = file.to_string_lossy().into_owned();
            // Probe geometry from a short-lived reader open (metadata only); the
            // pipeline builds its own reader on the reader thread.
            let mut probe = tomoxide::io::open_dxchange(&path)?;
            let geom = geometry_from_reader(probe.as_mut(), None)?;
            drop(probe);
            let params = recon_params(&geom);
            let out = recon_out_path(&file);
            let read_path = path.clone();
            let write_path = out.clone();
            tomoxide::ReconSteps::new(64).run_streaming_pipelined(
                move || tomoxide::io::open_dxchange(&read_path),
                move || tomoxide::io::create_writer(&write_path, tomoxide::io::SaveFormat::Tiff),
                &geom,
                Algorithm::Fbp,
                &params,
                &tomoxide::PrepOptions::default(),
                &engine,
            )?;
            println!("wrote streamed reconstruction to {out}");
        }
    }
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
fn recon_params(geom: &Geometry) -> ReconParams {
    ReconParams {
        num_gridx: Some(geom.detector.width),
        ..Default::default()
    }
}

/// Output path for a reconstruction: `<input-without-extension>_rec`.
fn recon_out_path(file: &Path) -> String {
    format!("{}_rec", file.with_extension("").display())
}
