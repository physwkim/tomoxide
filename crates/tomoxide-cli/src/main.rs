//! `tomoxide` — command-line front-end (ports tomocupy `__main__.py`:
//! `init` / `recon` / `recon_steps` / `status`).

mod config;

use std::path::PathBuf;

use anyhow::{anyhow, Context};
use clap::{Parser, Subcommand};
use tomoxide::{BackendKind, Engine};
use tomoxide_core::params::Algorithm;

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
            // Wiring the DXchange reader → pipeline::reconstruct → writer is
            // milestone M3; surface that honestly rather than pretending.
            match tomoxide::io::open_dxchange(&file.to_string_lossy()) {
                Ok(_) => unreachable!("reader is a stub"),
                Err(e) => println!("not yet runnable: {e}"),
            }
        }
        Command::ReconSteps { file } => {
            let engine = Engine::new(backend_kind)?;
            println!(
                "recon_steps: file={} backend={}",
                file.display(),
                engine.name()
            );
            if let Err(e) = tomoxide::ReconSteps.run() {
                println!("not yet runnable: {e}");
            }
        }
    }
    Ok(())
}
