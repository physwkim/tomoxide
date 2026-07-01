//! `tomoxide` — command-line front-end (ports tomocupy `__main__.py`:
//! `init` / `recon` / `recon_steps` / `status`).

mod chunk_cache;
mod config;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context};
use clap::{Parser, Subcommand};
use tomoxide::io::DatasetReader;
use tomoxide::Algorithm;
use tomoxide::Center;
use tomoxide::{
    Angles, BackendKind, Dtype, Engine, FilterName, Geometry, PhaseMethod, PrepOptions,
    ReconParams, StripeMethod,
};

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
        #[command(flatten)]
        common: CommonRecon,
        /// Detector rows per streaming chunk for the auto-pipelined CUDA path.
        /// Omit to use the `tune_chunk` cache for this file/algorithm/GPU if
        /// present, else the safe default; an explicit value always overrides.
        #[arg(long)]
        chunk: Option<usize>,
        /// First detector row (slice) to reconstruct, inclusive. Together with
        /// `--end-row` restricts output to a contiguous z-shard. Omit for the
        /// whole volume. Set internally by the multi-GPU orchestrator (one
        /// process per GPU), but usable directly to reconstruct a sub-range.
        #[arg(long)]
        start_row: Option<usize>,
        /// One past the last detector row (slice) to reconstruct, exclusive. See
        /// `--start-row`. Omit for the whole volume.
        #[arg(long)]
        end_row: Option<usize>,
        /// Laminography tilt angle in DEGREES (tomocupy `--lamino-angle`). When
        /// set, the rotation axis is tilted by this angle and the whole-stack
        /// laminography back-projection is used (CUDA fbp/linerec, f32 only).
        /// Omit for ordinary parallel-beam tomography.
        #[arg(long)]
        lamino_angle: Option<f32>,
        /// Laminography reconstruction height (output z-slices). Omit for
        /// tomocupy's auto height `ceil(nz / cos(lamino_angle) / 2) * 2`.
        #[arg(long)]
        lamino_rh: Option<usize>,
    },
    /// Chunked / streaming reconstruction (out-of-core).
    ReconSteps {
        #[command(flatten)]
        common: CommonRecon,
        /// Detector rows (slices) reconstructed/written per pipeline chunk
        /// (tomocupy `--nsino-per-chunk`). Omit to use the config's
        /// `nsino_per_chunk`, else the safe default. Smaller = more
        /// read/compute/write overlap; larger = fewer per-chunk launches.
        #[arg(long)]
        chunk: Option<usize>,
    },
    /// Measure the best pipeline `--chunk` for this file/algorithm/GPU and
    /// cache it (so a later `recon` auto-applies it). Times a full streaming
    /// reconstruction per power-of-two candidate and records the fastest.
    TuneChunk {
        /// Input DXchange HDF5 file.
        file: PathBuf,
        /// Algorithm to tune (must be a CUDA pipelined method: fbp, linerec,
        /// fourierrec, lprec).
        #[arg(long, default_value = "fbp")]
        algorithm: String,
        /// Rotation-axis column (omit to use the detector midline).
        #[arg(long)]
        center: Option<f32>,
        /// Reconstruction precision the chunk is tuned for: float32 | float16.
        #[arg(long, default_value = "float32")]
        dtype: String,
    },
}

/// Reconstruction knobs shared by `recon` and `recon_steps`. Every flag is
/// optional; an omitted flag falls back to the `--config` file (if given), then
/// to the built-in default. A given flag always overrides the config
/// (tomocupy-style precedence). See [`resolve`].
#[derive(clap::Args, Debug)]
#[command(rename_all = "snake_case")]
struct CommonRecon {
    /// Input DXchange HDF5 file.
    file: PathBuf,
    /// Optional TOML config (from `tomoxide init`). Supplies defaults for
    /// algorithm/center/filter/stripe/phase/num_iter/save_format/chunk; any flag
    /// below overrides its config value.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Algorithm (fbp, gridrec, fourierrec, lprec, linerec, sirt, …).
    #[arg(long)]
    algorithm: Option<String>,
    /// Rotation-axis column (omit to auto-find / use the detector midline).
    #[arg(long)]
    center: Option<f32>,
    /// Reconstruction precision: float32 | float16 (CUDA analytic paths only).
    #[arg(long, default_value = "float32")]
    dtype: String,
    /// Output format: tiff | h5 | zarr (tomocupy `--save-format`).
    #[arg(long)]
    save_format: Option<String>,
    /// Apodization filter (none|ramp|shepp|cosine|cosine2|hamming|hann|parzen).
    #[arg(long)]
    filter: Option<String>,
    /// Stripe-removal method (none|fw|ti|sf|vo-all), applied before recon with
    /// tomopy/tomocupy default parameters.
    #[arg(long)]
    remove_stripe: Option<String>,
    /// Phase-retrieval method (none|paganin|Gpaganin|farago), applied before
    /// recon using the physics flags below.
    #[arg(long)]
    retrieve_phase: Option<String>,
    /// Iterations for iterative algorithms (sirt/mlem/osem/… ). Ignored by the
    /// analytic methods.
    #[arg(long)]
    num_iter: Option<usize>,
    /// Regularization parameters for iterative methods (`reg_par`), a
    /// comma-separated f32 list (e.g. `--reg_par 0.5,0.01`).
    #[arg(long)]
    reg_par: Option<String>,
    /// Phase retrieval: detector pixel size (cm).
    #[arg(long, default_value_t = 1e-4)]
    pixel_size: f32,
    /// Phase retrieval: sample-to-detector propagation distance (cm).
    #[arg(long, default_value_t = 50.0)]
    propagation_distance: f32,
    /// Phase retrieval: X-ray energy (keV).
    #[arg(long, default_value_t = 30.0)]
    energy: f32,
    /// Phase retrieval (paganin): regularization parameter `alpha`.
    #[arg(long, default_value_t = 1e-3)]
    alpha: f32,
    /// Phase retrieval (Gpaganin/farago): material `delta/beta` ratio.
    #[arg(long, default_value_t = 1000.0)]
    db: f32,
    /// Phase retrieval (Gpaganin): characteristic transverse length `W` (cm).
    #[arg(long, default_value_t = 2e-4)]
    w: f32,
}

/// Fully-resolved reconstruction settings (config merged with CLI flags), plus
/// the string forms needed to forward the choice to multi-GPU shard subprocesses.
struct ReconPlan {
    algorithm: String,
    algo: Algorithm,
    center: Option<f32>,
    dtype: Dtype,
    save_format: tomoxide::io::SaveFormat,
    save_format_str: String,
    filter: FilterName,
    filter_str: String,
    num_iter: usize,
    reg_par: Vec<f32>,
    prep: PrepOptions,
    stripe_str: String,
    phase_str: String,
    pixel_size: f32,
    dist: f32,
    energy: f32,
    alpha: f32,
    db: f32,
    w: f32,
}

/// Map a stripe-removal method name (matching `Config::remove_stripe_method`) to a
/// [`StripeMethod`] with the tomopy/tomocupy default parameters.
fn parse_stripe(name: &str) -> anyhow::Result<StripeMethod> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "none" => StripeMethod::None,
        "fw" => StripeMethod::Fw {
            sigma: 2.0,
            level: None,
        },
        "ti" => StripeMethod::Ti {
            nblock: 0,
            beta: 1.5,
        },
        "sf" => StripeMethod::Sf { size: 5 },
        "vo-all" | "vo_all" | "voall" => StripeMethod::VoAll {
            snr: 3.0,
            la_size: 61,
            sm_size: 21,
        },
        other => {
            return Err(anyhow!(
                "unknown stripe method '{other}' (none|fw|ti|sf|vo-all)"
            ))
        }
    })
}

/// Map a phase-retrieval method name to a [`PhaseMethod`] with the given physics
/// parameters.
fn parse_phase(
    name: &str,
    ps: f32,
    dist: f32,
    energy: f32,
    alpha: f32,
    db: f32,
    w: f32,
) -> anyhow::Result<PhaseMethod> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "none" => PhaseMethod::None,
        "paganin" => PhaseMethod::Paganin {
            pixel_size: ps,
            dist,
            energy,
            alpha,
        },
        "gpaganin" => PhaseMethod::GPaganin {
            pixel_size: ps,
            dist,
            energy,
            db,
            w,
        },
        "farago" => PhaseMethod::Farago {
            pixel_size: ps,
            dist,
            energy,
            db,
        },
        other => {
            return Err(anyhow!(
                "unknown phase method '{other}' (none|paganin|Gpaganin|farago)"
            ))
        }
    })
}

/// Parse a comma-separated `f32` list (empty string ⇒ empty vector).
fn parse_f32_list(s: &str) -> anyhow::Result<Vec<f32>> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f32>()
                .map_err(|e| anyhow!("bad value '{t}': {e}"))
        })
        .collect()
}

/// Resolve the effective reconstruction settings: load the optional config, then
/// override each field with any explicitly-given CLI flag. Returns the plan plus
/// the loaded config (the caller reads `config.backend` for backend fallback).
fn resolve(c: &CommonRecon) -> anyhow::Result<(ReconPlan, Config)> {
    let cfg = match &c.config {
        Some(p) => Config::load(p).with_context(|| format!("loading {}", p.display()))?,
        None => Config::default(),
    };
    let algorithm = c.algorithm.clone().unwrap_or_else(|| cfg.algorithm.clone());
    let algo: Algorithm = algorithm.parse().map_err(|e| anyhow!("{e}"))?;
    let center = c.center.or(cfg.rotation_axis);
    let dtype: Dtype = c.dtype.parse().map_err(|e| anyhow!("{e}"))?;
    let save_format_str = c
        .save_format
        .clone()
        .unwrap_or_else(|| cfg.save_format.clone());
    let save_format: tomoxide::io::SaveFormat =
        save_format_str.parse().map_err(|e| anyhow!("{e}"))?;
    let filter_str = c.filter.clone().unwrap_or_else(|| cfg.filter_name.clone());
    let filter: FilterName = filter_str.parse().map_err(|e| anyhow!("{e}"))?;
    let num_iter = c.num_iter.unwrap_or(cfg.num_iter);
    let reg_par = match &c.reg_par {
        Some(s) => parse_f32_list(s)?,
        None => Vec::new(),
    };
    let stripe_str = c
        .remove_stripe
        .clone()
        .unwrap_or_else(|| cfg.remove_stripe_method.clone());
    let phase_str = c
        .retrieve_phase
        .clone()
        .unwrap_or_else(|| cfg.retrieve_phase_method.clone());
    let prep = PrepOptions {
        stripe: parse_stripe(&stripe_str)?,
        phase: parse_phase(
            &phase_str,
            c.pixel_size,
            c.propagation_distance,
            c.energy,
            c.alpha,
            c.db,
            c.w,
        )?,
    };
    Ok((
        ReconPlan {
            algorithm,
            algo,
            center,
            dtype,
            save_format,
            save_format_str,
            filter,
            filter_str,
            num_iter,
            reg_par,
            prep,
            stripe_str,
            phase_str,
            pixel_size: c.pixel_size,
            dist: c.propagation_distance,
            energy: c.energy,
            alpha: c.alpha,
            db: c.db,
            w: c.w,
        },
        cfg,
    ))
}

/// Default z-rows per streaming pipeline chunk. Small enough that the
/// read‖compute‖write conveyor overlaps well across a typical `nz` (≈128), large
/// enough that per-chunk launch/setup overhead stays amortized. Mirrors
/// tomocupy's `--nsino-per-chunk` default magnitude.
const DEFAULT_PIPELINE_CHUNK: usize = 8;

/// Minimum reconstructed width (`nx`) at which `recon` fans the CUDA TIFF job into
/// one z-shard process per GPU. Below this the per-process CUDA-init + startup
/// overhead of the extra shards outweighs the parallelism, so multi-GPU runs stay
/// on the single-GPU streaming pipeline. Empirical crossover on RTX 5000 Ada
/// (nz≈128): at nx≥2048 the 4-GPU wall drops below single-GPU for all analytic
/// methods; at nx≤1024 sharding regressed by ~0.5 s.
const MULTI_GPU_MIN_NX: usize = 2048;

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
/// - **Fourierrec**: device-resident streaming reconstructor (GPU pack-pairs →
///   `cfunc_fourierrec` → unpack, cuFFT plans reused across chunks) — large win.
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

/// Resolve the backend for a recon subcommand: an explicit top-level `--backend`
/// wins; when it is left at the `auto` default, the `--config` file's `backend`
/// applies (so a config can pin a device without a flag). `auto` either way
/// leaves backend auto-detection to [`Engine`].
fn resolve_backend(flag: &str, cfg: &Config) -> anyhow::Result<BackendKind> {
    let chosen = if flag == "auto" { &cfg.backend } else { flag };
    parse_backend(chosen)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let level = if cli.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level)).init();

    let backend_kind = parse_backend(&cli.backend)?;
    // The recon subcommands may override the backend from a `--config` file when
    // the top-level `--backend` is left at its `auto` default; captured here since
    // the match moves `cli.command`.
    let backend_flag = cli.backend.clone();

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
            common,
            chunk,
            start_row,
            end_row,
            lamino_angle,
            lamino_rh,
        } => {
            let (plan, cfg) = resolve(&common)?;
            let engine = Engine::new(resolve_backend(&backend_flag, &cfg)?)?;
            let file = common.file.clone();
            let dtype = plan.dtype;
            let save_format = plan.save_format;
            println!(
                "recon: file={} algorithm={:?} center={:?} dtype={} filter={} stripe={} phase={} backend={}",
                file.display(),
                plan.algo,
                plan.center,
                dtype.as_str(),
                plan.filter_str,
                plan.stripe_str,
                plan.phase_str,
                engine.name()
            );
            let out = recon_out_path(&file);
            // Laminography is whole-stack only (the tilt couples all detector rows
            // into every output voxel), so it cannot use the per-slice pipelined /
            // multi-GPU shard path — force the whole-volume route below.
            if pipelines_well(&engine, plan.algo) && lamino_angle.is_none() {
                // Resolve the streaming chunk: explicit `--chunk` wins, else the
                // tuned value cached by `tune_chunk` for this file/algorithm/GPU,
                // else the safe default.
                let mut probe = tomoxide::io::open_dxchange(&file.to_string_lossy())?;
                let (nproj, nz, nx, _nflat, _ndark) = probe.read_sizes()?;
                drop(probe);
                let (chunk, source) =
                    resolve_chunk(chunk, &file, &plan.algorithm, dtype, nx, nproj, nz);
                println!("  chunk: {chunk} ({source})");
                // Multi-GPU fan-out: when this is the top-level invocation (no
                // explicit row range), the backend is CUDA, the output is TIFF
                // (per-slice files, so disjoint shards never collide), and more
                // than one device is selected, reconstruct one contiguous z-shard
                // per GPU in its own process (`CUDA_VISIBLE_DEVICES`). Each child
                // reads only its slab and writes at the global slice offset, so the
                // device compute *and* the HDF5 read / TIFF write parallelize —
                // mirroring tomocupy's multi-process shard. Otherwise run a single
                // pipeline over the requested range.
                // Below the crossover, one extra CUDA context + binary startup per
                // shard process (~0.5 s each, paid in parallel but contended) costs
                // more than splitting the small per-slice work saves — sharding a
                // 1024²-or-smaller volume runs *slower* than a single GPU. The
                // dominant per-slice cost grows with nx², so gate on nx: at nx≥2048
                // the compute/I/O the shards parallelize clearly outweighs the
                // startup, and the 4-GPU wall drops well below single-GPU (and below
                // tomocupy). Smaller volumes stay on the single-GPU streaming path.
                let devices = tomoxide::cuda::selected_devices();
                let top_level = start_row.is_none() && end_row.is_none();
                let shardable = engine.name() == "cuda"
                    && matches!(save_format, tomoxide::io::SaveFormat::Tiff)
                    && top_level
                    && devices.len() > 1
                    && nz > devices.len()
                    && nx >= MULTI_GPU_MIN_NX;
                if shardable {
                    run_sharded_subprocesses(&file, &plan, chunk, nz, &devices)?;
                } else {
                    // Overlapped streaming path: same output as the whole-volume
                    // path (cuFFT-floor identical, Pearson 1.0), lower peak memory,
                    // and it hides disk read/write behind GPU compute.
                    run_pipelined(
                        &file,
                        &out,
                        plan.algo,
                        plan.center,
                        dtype,
                        save_format,
                        chunk,
                        start_row,
                        end_row,
                        plan.filter,
                        plan.num_iter,
                        plan.reg_par.clone(),
                        plan.prep,
                        &engine,
                    )?;
                }
            } else {
                // Whole-volume path (CPU/wgpu, chunking-hostile GPU methods, or
                // laminography).
                let mut reader = tomoxide::io::open_dxchange(&file.to_string_lossy())?;
                let mut geom = geometry_from_reader(reader.as_mut(), plan.center)?;
                let mut params = recon_params(
                    &geom,
                    dtype,
                    plan.filter,
                    plan.num_iter,
                    plan.reg_par.clone(),
                );
                if let Some(deg) = lamino_angle {
                    use std::f32::consts::PI;
                    geom.beam = tomoxide::Beam::Laminography {
                        phi: PI / 2.0 + deg * PI / 180.0,
                    };
                    params.lamino_rh = lamino_rh;
                    println!("  laminography: tilt={deg}° rh={lamino_rh:?}");
                }
                let ds = reader.read_all()?;
                let vol =
                    tomoxide::reconstruct(ds, &geom, plan.algo, &params, &plan.prep, &engine)?;
                let mut writer = tomoxide::io::create_writer(&out, save_format)?;
                let nz = vol.dims().0;
                writer.reserve(nz)?;
                writer.write_chunk(&vol, 0, nz)?;
            }
            println!("wrote reconstruction to {out}");
        }
        Command::ReconSteps { common, chunk } => {
            let (plan, cfg) = resolve(&common)?;
            let engine = Engine::new(resolve_backend(&backend_flag, &cfg)?)?;
            let file = common.file.clone();
            let chunk = chunk.unwrap_or(cfg.nsino_per_chunk);
            println!(
                "recon_steps: file={} algorithm={:?} center={:?} dtype={} filter={} stripe={} phase={} chunk={} backend={}",
                file.display(),
                plan.algo,
                plan.center,
                plan.dtype.as_str(),
                plan.filter_str,
                plan.stripe_str,
                plan.phase_str,
                chunk,
                engine.name()
            );
            let out = recon_out_path(&file);
            run_pipelined(
                &file,
                &out,
                plan.algo,
                plan.center,
                plan.dtype,
                plan.save_format,
                chunk,
                None,
                None,
                plan.filter,
                plan.num_iter,
                plan.reg_par,
                plan.prep,
                &engine,
            )?;
            println!("wrote streamed reconstruction to {out}");
        }
        Command::TuneChunk {
            file,
            algorithm,
            center,
            dtype,
        } => {
            let engine = Engine::new(backend_kind)?;
            let algo: Algorithm = algorithm.parse().map_err(|e| anyhow!("{e}"))?;
            let dtype: Dtype = dtype.parse().map_err(|e| anyhow!("{e}"))?;
            if !pipelines_well(&engine, algo) {
                return Err(anyhow!(
                    "tune_chunk applies only to CUDA pipelined algorithms \
                     (fbp, linerec, fourierrec, lprec); {:?} on backend {} uses the \
                     whole-volume path, where --chunk has no effect",
                    algo,
                    engine.name()
                ));
            }
            tune_chunk(&file, &algorithm, algo, center, dtype, &engine)?;
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
    start_row: Option<usize>,
    end_row: Option<usize>,
    filter: FilterName,
    num_iter: usize,
    reg_par: Vec<f32>,
    prep: PrepOptions,
    engine: &Engine,
) -> anyhow::Result<()> {
    let path = file.to_string_lossy().into_owned();
    // Probe geometry from a short-lived reader open (metadata only); the pipeline
    // builds its own reader on the reader thread.
    let mut probe = tomoxide::io::open_dxchange(&path)?;
    let geom = geometry_from_reader(probe.as_mut(), center)?;
    drop(probe);
    let params = recon_params(&geom, dtype, filter, num_iter, reg_par);
    let read_path = path;
    let write_path = out.to_string();
    // Reconstruct only `[start_row, end_row)` (a z-shard); both omitted ⇒ the
    // whole volume (`usize::MAX` is clamped to nz by the reader).
    let z_start = start_row.unwrap_or(0);
    let z_end = end_row.unwrap_or(usize::MAX);
    tomoxide::ReconSteps::new(chunk).run_streaming_pipelined_range(
        z_start,
        z_end,
        move || tomoxide::io::open_dxchange(&read_path),
        move || tomoxide::io::create_writer(&write_path, save_format),
        &geom,
        algo,
        &params,
        &prep,
        engine,
    )?;
    Ok(())
}

/// Fan a CUDA TIFF reconstruction across the selected GPUs by spawning one child
/// `recon` process per device, each pinned to its GPU via `CUDA_VISIBLE_DEVICES`
/// and restricted to a contiguous z-shard with `--start-row`/`--end-row`. The
/// children write per-slice TIFFs at their global slice offset into the shared
/// output directory, so the result is identical to a single-process run — but the
/// HDF5 read, GPU compute, and TIFF write all parallelize across processes
/// (mirroring tomocupy's multi-process shard). Spawned children see exactly one
/// GPU, so they take the single-pipeline branch and do not recurse.
///
/// Each child re-executes this same binary. The parent passes the already-resolved
/// `chunk` (so children skip cache lookups and all use the same value) and the
/// full resolved [`ReconPlan`] as explicit flags (so children need no `--config`
/// and reproduce the parent's filter/stripe/phase/iteration settings exactly).
/// The call fails if any child fails.
fn run_sharded_subprocesses(
    file: &Path,
    plan: &ReconPlan,
    chunk: usize,
    nz: usize,
    devices: &[i32],
) -> anyhow::Result<()> {
    let exe = std::env::current_exe().context("locating current executable")?;
    let n = devices.len();
    // Contiguous shards differing by at most one row, covering [0, nz).
    let base = nz / n;
    let rem = nz % n;
    println!("  multi-GPU: {n} shards across devices {devices:?}");
    let mut children = Vec::with_capacity(n);
    let mut z0 = 0usize;
    for (i, &dev) in devices.iter().enumerate() {
        let rows = base + if i < rem { 1 } else { 0 };
        let z1 = z0 + rows;
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("--backend")
            .arg("cuda")
            .arg("recon")
            .arg(file)
            .arg("--algorithm")
            .arg(&plan.algorithm)
            .arg("--dtype")
            .arg(plan.dtype.as_str())
            .arg("--save_format")
            .arg(&plan.save_format_str)
            .arg("--filter")
            .arg(&plan.filter_str)
            .arg("--remove_stripe")
            .arg(&plan.stripe_str)
            .arg("--retrieve_phase")
            .arg(&plan.phase_str)
            .arg("--num_iter")
            .arg(plan.num_iter.to_string())
            .arg("--chunk")
            .arg(chunk.to_string())
            .arg("--start_row")
            .arg(z0.to_string())
            .arg("--end_row")
            .arg(z1.to_string());
        if !plan.reg_par.is_empty() {
            let csv = plan
                .reg_par
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            cmd.arg("--reg_par").arg(csv);
        }
        // Phase-retrieval physics only matters when a phase method is selected;
        // forward it so the shards match the parent exactly.
        if !plan.phase_str.eq_ignore_ascii_case("none") {
            cmd.arg("--pixel_size")
                .arg(plan.pixel_size.to_string())
                .arg("--propagation_distance")
                .arg(plan.dist.to_string())
                .arg("--energy")
                .arg(plan.energy.to_string())
                .arg("--alpha")
                .arg(plan.alpha.to_string())
                .arg("--db")
                .arg(plan.db.to_string())
                .arg("--w")
                .arg(plan.w.to_string());
        }
        if let Some(c) = plan.center {
            cmd.arg("--center").arg(c.to_string());
        }
        // Pin the child to one physical GPU; clear any inherited multi-device
        // selection so the child's `selected_devices()` is exactly `[0]`.
        cmd.env("CUDA_VISIBLE_DEVICES", dev.to_string())
            .env("TOMOXIDE_CUDA_DEVICES", "0")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());
        let child = cmd
            .spawn()
            .with_context(|| format!("spawning shard {i} on device {dev}"))?;
        children.push((i, dev, z0, z1, child));
        z0 = z1;
    }
    // Wait for all shards; collect every failure rather than bailing on the first.
    let mut failures = Vec::new();
    for (i, dev, s, e, child) in children {
        let output = child
            .wait_with_output()
            .with_context(|| format!("waiting for shard {i} on device {dev}"))?;
        if !output.status.success() {
            let reason = subprocess_failure_reason(&output);
            failures.push(format!(
                "shard {i} (device {dev}, rows [{s}, {e})): {reason}"
            ));
        }
    }
    if !failures.is_empty() {
        return Err(anyhow!(
            "multi-GPU recon failed:\n  {}",
            failures.join("\n  ")
        ));
    }
    Ok(())
}

/// Resolve the streaming chunk size for `recon`'s auto-pipelined path.
///
/// Priority: an explicit `--chunk` always wins; otherwise the value cached by
/// `tune_chunk` for this `(file, algorithm, dtype, gpu)` is used **if** its
/// stored geometry still matches the dataset; otherwise the safe default. The
/// returned `&str` names the source for the log line.
fn resolve_chunk(
    explicit: Option<usize>,
    file: &Path,
    algorithm: &str,
    dtype: Dtype,
    nx: usize,
    nproj: usize,
    nz: usize,
) -> (usize, &'static str) {
    if let Some(c) = explicit {
        return (c.max(1), "--chunk");
    }
    let gpu = tomoxide::cuda::device_name().unwrap_or_else(|| "unknown".into());
    let key = chunk_cache::key(file, algorithm, dtype.as_str(), &gpu);
    if let Some(c) = chunk_cache::ChunkCache::load().get(&key, nx, nproj, nz) {
        return (c, "from cache");
    }
    (DEFAULT_PIPELINE_CHUNK, "default")
}

/// Power-of-two chunk candidates to measure, from [`DEFAULT_PIPELINE_CHUNK`] up
/// to `nz/2` so the pipeline keeps at least two chunks to overlap. A volume too
/// thin to split falls back to a single whole-volume chunk.
fn chunk_candidates(nz: usize) -> Vec<usize> {
    let mut c = Vec::new();
    let mut k = DEFAULT_PIPELINE_CHUNK;
    while k <= nz / 2 {
        c.push(k);
        k *= 2;
    }
    if c.is_empty() {
        c.push(nz.max(1));
    }
    c
}

/// Scratch directory for tuning runs, created next to the input file so a cheap
/// hard link to the dataset lands on the same filesystem. Each measured candidate
/// reconstructs the linked dataset here (never the real `<file>_rec`), and the
/// directory is removed when tuning finishes.
fn tune_scratch_dir(file: &Path) -> PathBuf {
    let parent = file
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    parent.join(format!(".tomoxide_tune_{}", std::process::id()))
}

/// Measure one candidate by re-executing this binary as a fresh `recon`
/// subprocess at `chunk`, timing its wall clock. A separate process is the whole
/// point: the thread-local cuFFT plan cache is never destroyed within a process
/// (see `cuda/fft.cu`), so measuring every candidate in one process leaks plan
/// VRAM across candidates and makes the later/larger ones false-OOM. Each
/// subprocess starts on a clean device and frees all VRAM on exit. The measured
/// wall includes a fixed process + CUDA-init overhead — uniform across candidates,
/// so it does not change which chunk ranks fastest. Returns wall-clock seconds.
#[allow(clippy::too_many_arguments)]
fn measure_chunk_subprocess(
    exe: &Path,
    backend: &str,
    link: &Path,
    algorithm: &str,
    center: Option<f32>,
    dtype: Dtype,
    chunk: usize,
) -> anyhow::Result<f64> {
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--backend")
        .arg(backend)
        .arg("recon")
        .arg(link)
        .arg("--algorithm")
        .arg(algorithm)
        .arg("--dtype")
        .arg(dtype.as_str())
        .arg("--chunk")
        .arg(chunk.to_string());
    if let Some(c) = center {
        cmd.arg("--center").arg(c.to_string());
    }
    // Tuning measures a single-GPU pipeline (the cache is keyed by one device
    // name): pin the candidate to one GPU so it does not fan into the multi-GPU
    // shard path, which would conflate per-shard chunk timing.
    cmd.env("TOMOXIDE_CUDA_DEVICES", "0")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let t = Instant::now();
    let output = cmd
        .output()
        .with_context(|| format!("spawning {} recon", exe.display()))?;
    let secs = t.elapsed().as_secs_f64();
    if !output.status.success() {
        return Err(anyhow!("{}", subprocess_failure_reason(&output)));
    }
    Ok(secs)
}

/// One-line reason a candidate's `recon` subprocess failed, for the tune log.
///
/// A clean OOM returns a non-zero exit with an `Error:` line our extraction
/// surfaces. A *signal* kill leaves no such line: the vendored `cfunc_filter`
/// makes unchecked cuFFT calls, so when a plan's work area cannot be allocated
/// (the chunk does not fit) `cufftXtExec` dereferences a bad work area and the
/// process dies with SIGSEGV rather than a clean error. Without this, the last
/// stderr line is a stray env_logger record, which is useless as a skip reason.
fn subprocess_failure_reason(output: &std::process::Output) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = output.status.signal() {
            return format!("killed by signal {sig} (does not fit in device memory)");
        }
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Prefer anyhow's "Error: ..." line; else the last line that is not an
    // env_logger record (bracketed, e.g. "[2026-.. INFO module] ...").
    if let Some(e) = stderr
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with("Error:"))
    {
        return e
            .trim_start()
            .trim_start_matches("Error:")
            .trim()
            .to_string();
    }
    stderr
        .lines()
        .rev()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('['))
        .unwrap_or("recon subprocess failed")
        .to_string()
}

/// Tune and cache the best pipeline `--chunk` for one file/algorithm/GPU.
///
/// Hybrid: an analytic step builds the candidate set ([`chunk_candidates`] —
/// powers of two keeping ≥2 chunks), then an empirical step times a full
/// streaming reconstruction at each candidate and records the fastest. A
/// candidate that errors (e.g. out of device memory) is skipped, so memory-fit
/// is enforced by measurement rather than an estimated VRAM model. The result is
/// written to the chunk cache so `recon` auto-applies it.
fn tune_chunk(
    file: &Path,
    algorithm: &str,
    algo: Algorithm,
    center: Option<f32>,
    dtype: Dtype,
    engine: &Engine,
) -> anyhow::Result<()> {
    let mut probe = tomoxide::io::open_dxchange(&file.to_string_lossy())?;
    let (nproj, nz, nx, _nflat, _ndark) = probe.read_sizes()?;
    drop(probe);
    let gpu = tomoxide::cuda::device_name().unwrap_or_else(|| "unknown".into());
    let candidates = chunk_candidates(nz);
    println!(
        "tune_chunk: file={} algorithm={:?} dtype={} gpu={} dims=(nx={} nproj={} nz={})",
        file.display(),
        algo,
        dtype.as_str(),
        gpu,
        nx,
        nproj,
        nz
    );
    println!("  candidates (powers of two, ≥2 chunks): {candidates:?}");

    // Each candidate runs as its own `recon` subprocess (see
    // `measure_chunk_subprocess`): the per-process cuFFT plan cache is never
    // freed, so isolating candidates is what keeps the larger ones from
    // false-OOMing on VRAM the earlier ones leaked. They reconstruct a cheap hard
    // link to the dataset inside a scratch dir, so the real `<file>_rec` is never
    // touched.
    let exe = std::env::current_exe().context("locating the tomoxide executable")?;
    let scratch = tune_scratch_dir(file);
    std::fs::create_dir_all(&scratch)
        .with_context(|| format!("creating scratch dir {}", scratch.display()))?;
    let link = scratch.join("in.h5");
    std::fs::hard_link(file, &link)
        .with_context(|| format!("hard-linking {} -> {}", file.display(), link.display()))?;
    let cand_out = recon_out_path(&link);

    let mut results: Vec<(usize, f64)> = Vec::new();
    for &c in &candidates {
        match measure_chunk_subprocess(&exe, engine.name(), &link, algorithm, center, dtype, c) {
            Ok(secs) => {
                println!("  chunk={c:>4}: {secs:.2}s (wall, incl. process+CUDA init)");
                results.push((c, secs));
            }
            Err(e) => println!("  chunk={c:>4}: skipped ({e})"),
        }
        // Drop this candidate's reconstruction before the next (TIFF writes a
        // directory); the hard link stays for the remaining candidates.
        let _ = std::fs::remove_dir_all(&cand_out);
        let _ = std::fs::remove_file(&cand_out);
    }
    // Remove the whole scratch dir (hard link + any leftover output).
    let _ = std::fs::remove_dir_all(&scratch);

    let (best_chunk, best_secs) = results
        .iter()
        .copied()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| anyhow!("all chunk candidates failed to run (see skip reasons above)"))?;

    let key = chunk_cache::key(file, algorithm, dtype.as_str(), &gpu);
    let mut cache = chunk_cache::ChunkCache::load();
    cache.insert(
        key,
        chunk_cache::Entry {
            chunk: best_chunk,
            nx,
            nproj,
            nz,
        },
    );
    cache.save()?;
    println!(
        "  best chunk = {best_chunk} ({best_secs:.2}s) → cached to {}",
        chunk_cache::CACHE_FILE
    );
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

/// Reconstruction params with the grid sized to the detector width, plus the
/// filter (analytic methods) and iteration count / regularization (iterative
/// methods). Fields not relevant to the chosen algorithm are ignored downstream.
fn recon_params(
    geom: &Geometry,
    dtype: Dtype,
    filter_name: FilterName,
    num_iter: usize,
    reg_par: Vec<f32>,
) -> ReconParams {
    ReconParams {
        num_gridx: Some(geom.detector.width),
        dtype,
        filter_name,
        num_iter,
        reg_par,
        ..Default::default()
    }
}

/// Output path for a reconstruction: `<input-without-extension>_rec`.
fn recon_out_path(file: &Path) -> String {
    format!("{}_rec", file.with_extension("").display())
}
