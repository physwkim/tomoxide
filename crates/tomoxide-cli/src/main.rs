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
        /// Detector rows per streaming chunk for the auto-pipelined CUDA path.
        /// Omit to use the `tune_chunk` cache for this file/algorithm/GPU if
        /// present, else the safe default; an explicit value always overrides.
        #[arg(long)]
        chunk: Option<usize>,
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
            chunk,
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
                // Resolve the streaming chunk: explicit `--chunk` wins, else the
                // tuned value cached by `tune_chunk` for this file/algorithm/GPU,
                // else the safe default.
                let mut probe = tomoxide::io::open_dxchange(&file.to_string_lossy())?;
                let (nproj, nz, nx, _nflat, _ndark) = probe.read_sizes()?;
                drop(probe);
                let (chunk, source) = resolve_chunk(chunk, &file, &algorithm, dtype, nx, nproj, nz);
                println!("  chunk: {chunk} ({source})");
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
                    chunk,
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
    cmd.stdout(std::process::Stdio::null())
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
