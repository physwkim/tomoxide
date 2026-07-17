//! The single long-lived worker thread (docs/GUI.md §4).
//!
//! Owns the tomoxide [`Engine`] and performs every reconstruction/IO job off
//! the UI thread. HDF5 handles are `!Send`, so readers are opened *on* this
//! thread (or by the pipelined driver's own factory closures) and never cross
//! it. Jobs arrive on an mpsc channel; results go back as [`Event`]s, each
//! followed by `Context::request_repaint` so the UI wakes promptly.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};

use rsplot::egui;
use tomoxide::io::{DatasetReader, RowBandReader};
use tomoxide::{
    Algorithm, Angles, BackendKind, Center, Engine, FilterName, Geometry, PhaseMethod, ReconParams,
    StripeMethod,
};

/// Metadata of the opened DXchange dataset (read once per open).
pub struct DatasetMeta {
    pub path: PathBuf,
    pub nproj: usize,
    pub nz: usize,
    pub nx: usize,
    pub nflat: usize,
    pub ndark: usize,
    /// Projection angles in radians (generated uniformly if absent).
    pub theta: Vec<f32>,
    /// Finite min/max of projection frame 0 — the display range for the
    /// projection browser (raw-count stacks need it; the default 0..1
    /// colormap saturates them).
    pub data_range: (f32, f32),
}

/// Everything a single-slice preview reconstruction needs, fully resolved to
/// tomoxide types on the UI side (parse errors surface in the panel, not here).
#[derive(Clone)]
pub struct PreviewSpec {
    /// Detector row (slice) to reconstruct.
    pub slice: usize,
    pub algorithm: Algorithm,
    /// Rotation-axis column; `None` ⇒ detector midline.
    pub center: Option<f32>,
    pub filter: FilterName,
    pub num_iter: usize,
    pub reg_par: Vec<f32>,
    /// Truncated-projection support extension (iterative methods; see
    /// `ReconParams::ext_pad`).
    pub ext_pad: bool,
    pub stripe: StripeMethod,
    /// Phase retrieval. A non-`None` method makes the preview read a
    /// [`RowBandReader`] band around the slice (the retrieval couples
    /// detector rows) — see [`run_preview`].
    pub phase: PhaseMethod,
}

/// Rotation-axis auto-detection method (docs/GUI.md §2 Center).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CenterMethod {
    /// Nghia Vo's sinogram-domain Fourier method (`find_center_vo`).
    Vo,
    /// Entropy of trial gridrec reconstructions (`find_center`).
    Entropy,
    /// Phase correlation of the 0°/mirrored-180° projections (`find_center_pc`).
    Pc,
    /// SIFT registration of the 0°/180° pair (needs the `sift-center` feature).
    Sift,
}

impl CenterMethod {
    pub fn label(self) -> &'static str {
        match self {
            CenterMethod::Vo => "vo",
            CenterMethod::Entropy => "entropy",
            CenterMethod::Pc => "pc",
            CenterMethod::Sift => "sift",
        }
    }
}

/// Work requests from the UI thread.
pub enum Job {
    /// Probe a DXchange file: sizes + theta → [`Event::DatasetOpened`].
    OpenDataset(PathBuf),
    /// Read the raw sinogram at detector row `row` → [`Event::Sinogram`].
    ReadSinogram { row: usize },
    /// Reconstruct one slice in memory → [`Event::Preview`]. `generation` is
    /// echoed back so the UI can drop results that were superseded meanwhile.
    Preview { generation: u64, spec: PreviewSpec },
    /// Auto-detect the rotation axis → [`Event::CenterFound`]. `row` picks the
    /// sinogram for Vo/Entropy; `init` seeds the Entropy search.
    FindCenter {
        method: CenterMethod,
        row: usize,
        init: Option<f32>,
    },
    /// Reconstruct the sinogram at `row` once per trial center in
    /// `(start, stop, step)` (`recon::center::write_center`) →
    /// [`Event::CenterSweep`]. The montage input for the Center screen.
    CenterSweep { row: usize, range: (f32, f32, f32) },
    /// Reconstruct the preview slice once per λ in `lambdas` (all other
    /// parameters fixed by `spec`), replacing `reg_par[0]` each time →
    /// [`Event::LambdaSweep`]. The montage + L-curve input for the Tune
    /// screen's regularization-strength tuner.
    LambdaSweep {
        spec: PreviewSpec,
        lambdas: Vec<f32>,
    },
    /// Load + prep the whole projection stack (cached on the worker) and report
    /// the ring estimate **with the mean projection it was computed from** →
    /// [`Event::LaminoRings`]. Step 1 of `docs/LAMINOGRAPHY_ALIGNMENT.md`: the
    /// bullseye is read by eye, and the number is the second opinion.
    LaminoRings { step: usize },
    /// Probe-sweep the rotation axis over `centers` on output slice `slice`
    /// (`None` ⇒ the middle of the volume) at `tilt_deg` → the montage +
    /// focus curve of [`Event::LaminoCenterSweep`]. One launch for the whole
    /// sweep: the centre is an in-plane shift, so it does not move the in-focus
    /// layer and one slice can rank it.
    ///
    /// Ranking is all it does. Over a wide `centers` the focus curve grows
    /// competing lobes and its highest one is measurably not the axis (on the
    /// aligned reference scan: 417 over a known 396, by 0.34 %), so this
    /// **refines a prior** — the axis from step 1 — rather than searching for
    /// one. [`tomoxide::recon::center::judge_sweep`] is what enforces that; see
    /// its verdicts before adopting anything from here.
    LaminoCenterSweep {
        tilt_deg: Option<f32>,
        slice: Option<usize>,
        centers: Vec<f32>,
    },
    /// Score each tilt by a FULL reconstruction, max focus over every slice
    /// (`recon::center::lamino_tilt_scan`) → one [`Event::LaminoTilt`] per
    /// candidate as it lands, then [`Event::LaminoTiltDone`]. Minutes per
    /// candidate; [`Worker::cancel`] stops it at the next boundary.
    LaminoTiltScan { center: f32, tilts: Vec<f32> },
    /// Exit the worker loop.
    Shutdown,
}

/// Results/notifications back to the UI thread.
pub enum Event {
    /// Engine construction finished; payload = backend name (`cpu`/`cuda`/…).
    BackendReady(String),
    /// A dataset was opened and probed.
    DatasetOpened(Arc<DatasetMeta>),
    /// Raw counts sinogram `[nproj, nx]` (row-major) at detector row `row`.
    Sinogram {
        row: usize,
        nproj: usize,
        nx: usize,
        data: Vec<f32>,
    },
    /// A rotation-axis estimate (detector-column units).
    CenterFound {
        method: CenterMethod,
        center: f32,
        millis: u128,
    },
    /// A finished center sweep: one `[ny, nx]` trial reconstruction per
    /// candidate center, concatenated row-major in `frames`.
    CenterSweep {
        centers: Vec<f32>,
        ny: usize,
        nx: usize,
        frames: Vec<f32>,
        millis: u128,
    },
    /// A finished λ sweep: one `[ny, nx]` reconstruction per λ (row-major in
    /// `frames`), plus the L-curve coordinates — data residual `‖Ax − b‖₂`
    /// and roughness (isotropic TV seminorm) of each reconstruction.
    LambdaSweep {
        lambdas: Vec<f32>,
        ny: usize,
        nx: usize,
        frames: Vec<f32>,
        residual: Vec<f64>,
        roughness: Vec<f64>,
        millis: u128,
    },
    /// A finished single-slice preview: `[ny, nx]` row-major reconstruction.
    Preview {
        generation: u64,
        slice: usize,
        ny: usize,
        nx: usize,
        data: Vec<f32>,
        millis: u128,
    },
    /// The ring estimate and the mean projection `[ny, nx]` it came from.
    /// `bytes` is what the cached prepped stack now costs in host RAM.
    LaminoRings {
        center: f32,
        prominence: f32,
        trustworthy: bool,
        ny: usize,
        nx: usize,
        mean: Vec<f32>,
        bytes: usize,
        millis: u128,
    },
    /// A finished laminography centre sweep: one `[ny, nx]` probe reconstruction
    /// per candidate (row-major in `frames`) and its `slice_focus`.
    LaminoCenterSweep {
        centers: Vec<f32>,
        ny: usize,
        nx: usize,
        frames: Vec<f32>,
        focus: Vec<f64>,
        slice: usize,
        millis: u128,
    },
    /// One tilt candidate finished: its score, the in-focus slice `[ny, nx]` the
    /// reconstruction peaked on, and the focus of every slice — the profile that
    /// says whether `z_peak` is a hump on the sample or a spike at a z-edge.
    LaminoTilt {
        tilt_deg: f32,
        focus: f64,
        z_peak: usize,
        depth: usize,
        focus_by_z: Vec<f64>,
        ny: usize,
        nx: usize,
        slice: Vec<f32>,
        done: usize,
        total: usize,
    },
    /// The tilt scan finished (or stopped): `cancelled` says which.
    LaminoTiltDone { cancelled: bool, millis: u128 },
    /// A job failed; `what` names the job for the session log.
    JobFailed { what: String, error: String },
}

/// UI-side handle: send jobs, drain events. Dropping it shuts the thread down.
pub struct Worker {
    pub jobs: Sender<Job>,
    pub events: Receiver<Event>,
    /// Stops the tilt scan at the next candidate boundary. A `Job` cannot do
    /// this: the scan owns the worker for minutes and the job channel is only
    /// drained between jobs, so the request has to reach it out of band.
    pub cancel: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    pub fn spawn(ctx: egui::Context) -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::channel();
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let thread = std::thread::Builder::new()
            .name("tomoxide-worker".into())
            .spawn(move || worker_main(job_rx, event_tx, ctx, worker_cancel))
            .expect("spawning the worker thread");
        Worker {
            jobs: job_tx,
            events: event_rx,
            cancel,
            thread: Some(thread),
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.jobs.send(Job::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn worker_main(
    jobs: Receiver<Job>,
    events: Sender<Event>,
    ctx: egui::Context,
    cancel: Arc<AtomicBool>,
) {
    let send = |event: Event| {
        let _ = events.send(event);
        ctx.request_repaint();
    };

    // Auto picks CUDA when built with it and a device answers, else CPU.
    let engine = match Engine::new(BackendKind::Auto) {
        Ok(engine) => {
            send(Event::BackendReady(engine.name().to_string()));
            engine
        }
        Err(e) => {
            send(Event::JobFailed {
                what: "engine init".into(),
                error: e.to_string(),
            });
            return;
        }
    };

    // Path of the currently opened dataset; per-job readers are opened fresh
    // (H5 handles stay on this thread; opens are cheap next to the reads).
    let mut current: Option<PathBuf> = None;
    // The laminography jobs all need the WHOLE prepped stack — the rings average
    // every projection, and both sweeps back-project all of them — so unlike the
    // row-band jobs above there is nothing to read lazily. It costs
    // nproj·nz·nx·4 bytes (7.5 GB for 1800×1024²), so it is loaded once on
    // demand, kept, and dropped when the dataset changes.
    let mut prepped: Option<PreppedStack> = None;

    while let Ok(job) = jobs.recv() {
        match job {
            Job::Shutdown => break,
            Job::OpenDataset(path) => match probe(&path) {
                Ok(meta) => {
                    // The cache belongs to the old file; a new one invalidates it.
                    prepped = None;
                    current = Some(path);
                    send(Event::DatasetOpened(Arc::new(meta)));
                }
                Err(e) => send(Event::JobFailed {
                    what: format!("open {}", path.display()),
                    error: e.to_string(),
                }),
            },
            Job::ReadSinogram { row } => {
                let Some(path) = &current else {
                    continue;
                };
                match read_sinogram(path, row) {
                    Ok((nproj, nx, data)) => send(Event::Sinogram {
                        row,
                        nproj,
                        nx,
                        data,
                    }),
                    Err(e) => send(Event::JobFailed {
                        what: format!("sinogram row {row}"),
                        error: e.to_string(),
                    }),
                }
            }
            Job::FindCenter { method, row, init } => {
                let Some(path) = &current else {
                    continue;
                };
                let t0 = std::time::Instant::now();
                match run_find_center(&engine, path, method, row, init) {
                    Ok(center) => send(Event::CenterFound {
                        method,
                        center,
                        millis: t0.elapsed().as_millis(),
                    }),
                    Err(e) => send(Event::JobFailed {
                        what: format!("center ({})", method.label()),
                        error: e.to_string(),
                    }),
                }
            }
            Job::CenterSweep { row, range } => {
                let Some(path) = &current else {
                    continue;
                };
                let t0 = std::time::Instant::now();
                match run_center_sweep(&engine, path, row, range) {
                    Ok((centers, ny, nx, frames)) => send(Event::CenterSweep {
                        centers,
                        ny,
                        nx,
                        frames,
                        millis: t0.elapsed().as_millis(),
                    }),
                    Err(e) => send(Event::JobFailed {
                        what: "center sweep".into(),
                        error: e.to_string(),
                    }),
                }
            }
            Job::LambdaSweep { spec, lambdas } => {
                let Some(path) = &current else {
                    continue;
                };
                let t0 = std::time::Instant::now();
                match run_lambda_sweep(&engine, path, &spec, &lambdas) {
                    Ok((ny, nx, frames, residual, roughness)) => send(Event::LambdaSweep {
                        lambdas,
                        ny,
                        nx,
                        frames,
                        residual,
                        roughness,
                        millis: t0.elapsed().as_millis(),
                    }),
                    Err(e) => send(Event::JobFailed {
                        what: "lambda sweep".into(),
                        error: e.to_string(),
                    }),
                }
            }
            Job::LaminoRings { step } => {
                let Some(path) = current.clone() else {
                    continue;
                };
                let t0 = std::time::Instant::now();
                match ensure_prepped(&engine, &mut prepped, &path)
                    .and_then(|st| st.rings(engine.backend(), step))
                {
                    Ok((ring, mean)) => {
                        let (ny, nx) = mean.dim();
                        send(Event::LaminoRings {
                            center: ring.center,
                            prominence: ring.prominence,
                            trustworthy: ring.trustworthy,
                            ny,
                            nx,
                            mean: mean.iter().copied().collect(),
                            bytes: prepped.as_ref().map_or(0, PreppedStack::bytes),
                            millis: t0.elapsed().as_millis(),
                        })
                    }
                    Err(e) => send(Event::JobFailed {
                        what: "lamino rings".into(),
                        error: e.to_string(),
                    }),
                }
            }
            Job::LaminoCenterSweep {
                tilt_deg,
                slice,
                centers,
            } => {
                let Some(path) = current.clone() else {
                    continue;
                };
                let t0 = std::time::Instant::now();
                match ensure_prepped(&engine, &mut prepped, &path)
                    .and_then(|st| st.center_sweep(tilt_deg, slice, &centers))
                {
                    Ok((ny, nx, frames, focus, slice)) => send(Event::LaminoCenterSweep {
                        centers,
                        ny,
                        nx,
                        frames,
                        focus,
                        slice,
                        millis: t0.elapsed().as_millis(),
                    }),
                    Err(e) => send(Event::JobFailed {
                        what: "lamino center sweep".into(),
                        error: e.to_string(),
                    }),
                }
            }
            Job::LaminoTiltScan { center, tilts } => {
                let Some(path) = current.clone() else {
                    continue;
                };
                // A cancel from a previous scan must not kill this one.
                cancel.store(false, Ordering::Relaxed);
                let t0 = std::time::Instant::now();
                let total = tilts.len();
                let mut done = 0usize;
                let r = ensure_prepped(&engine, &mut prepped, &path).and_then(|st| {
                    st.tilt_scan(center, &tilts, &mut |r, img| {
                        done += 1;
                        let (ny, nx) = img.dim();
                        send(Event::LaminoTilt {
                            tilt_deg: r.tilt_deg,
                            focus: r.focus,
                            z_peak: r.z_peak,
                            depth: r.depth,
                            focus_by_z: r.focus_by_z.clone(),
                            ny,
                            nx,
                            slice: img.iter().copied().collect(),
                            done,
                            total,
                        });
                        if cancel.load(Ordering::Relaxed) {
                            return Err(tomoxide::Error::Backend("cancelled".into()));
                        }
                        Ok(())
                    })
                });
                let cancelled = cancel.swap(false, Ordering::Relaxed);
                match r {
                    Ok(_) => send(Event::LaminoTiltDone {
                        cancelled: false,
                        millis: t0.elapsed().as_millis(),
                    }),
                    // A cancel unwinds through the callback as an error; that is
                    // the scan stopping as asked, not a failure to report.
                    Err(_) if cancelled => send(Event::LaminoTiltDone {
                        cancelled: true,
                        millis: t0.elapsed().as_millis(),
                    }),
                    Err(e) => send(Event::JobFailed {
                        what: "lamino tilt scan".into(),
                        error: e.to_string(),
                    }),
                }
            }
            Job::Preview { generation, spec } => {
                let Some(path) = &current else {
                    continue;
                };
                let t0 = std::time::Instant::now();
                match run_preview(&engine, path, &spec) {
                    Ok((ny, nx, data)) => send(Event::Preview {
                        generation,
                        slice: spec.slice,
                        ny,
                        nx,
                        data,
                        millis: t0.elapsed().as_millis(),
                    }),
                    Err(e) => send(Event::JobFailed {
                        what: format!("preview slice {}", spec.slice),
                        error: e.to_string(),
                    }),
                }
            }
        }
    }
}

/// Single-slice preview: banded read → the same prep order as
/// `tomoxide::reconstruct` (normalize+minus-log → phase → sinogram layout →
/// stripe) → reconstruction of the requested row only.
///
/// Phase retrieval couples detector rows, so the read is a [`RowBandReader`]
/// band `[z − m, z + m]` with `m` = the Fresnel kernel's pixel support
/// (`prep::phase::margin_rows`; 0 without phase, i.e. exactly one row read).
/// Prep runs on the whole band; the sinogram is then cropped to the center
/// row *before* stripe removal and reconstruction (every stripe method is
/// per-sinogram independent, so crop-then-stripe equals stripe-then-crop) —
/// the preview stays one-slice cheap however wide the phase kernel is.
fn run_preview(
    engine: &Engine,
    path: &std::path::Path,
    spec: &PreviewSpec,
) -> tomoxide::Result<(usize, usize, Vec<f32>)> {
    let backend = engine.backend();
    let (one, geom) = prep_slice(engine, path, spec)?;
    let params = recon_params(spec, one.array.dim().2, spec.reg_par.clone());
    let vol = tomoxide::recon::recon(&one, &geom, spec.algorithm, &params, backend)?;
    let (_nz1, ny, nxo) = vol.dims();
    Ok((ny, nxo, vol.array.iter().copied().collect()))
}

/// Prep the requested slice down to the one-row sinogram and its geometry —
/// the shared front half of [`run_preview`] and [`run_lambda_sweep`]. Banded
/// read → normalize+minus-log → phase → sinogram layout → crop to the slice
/// row → stripe (see the module comment on crop-then-stripe).
fn prep_slice(
    engine: &Engine,
    path: &std::path::Path,
    spec: &PreviewSpec,
) -> tomoxide::Result<(tomoxide::Tomo, Geometry)> {
    let backend = engine.backend();
    let mut probe = tomoxide::io::open_dxchange(&path.to_string_lossy())?;
    let (_nproj, nz, nx, _nflat, _ndark) = probe.read_sizes()?;
    drop(probe);

    let slice = spec.slice.min(nz.saturating_sub(1));
    let m = tomoxide::prep::phase::margin_rows(&spec.phase);
    let z0 = slice.saturating_sub(m);
    let z1 = (slice + m + 1).min(nz);

    let inner = tomoxide::io::open_dxchange(&path.to_string_lossy())?;
    let mut ds = RowBandReader::new(inner, z0, z1)?.read_all()?;
    tomoxide::prep::normalize_dataset(&mut ds, backend)?;
    tomoxide::prep::retrieve_phase(&mut ds.data, spec.phase, backend)?;
    let sino = ds.data.to_layout(tomoxide::Layout::Sinogram);
    let row = slice - z0;
    let mut one = tomoxide::Tomo::new(
        sino.array
            .slice(ndarray::s![row..row + 1, .., ..])
            .to_owned(),
        tomoxide::Layout::Sinogram,
    );
    tomoxide::prep::remove_stripe(&mut one, spec.stripe)?;

    let mut geom = Geometry::parallel(Angles(ds.theta), nx, 1, 1.0);
    if let Some(c) = spec.center {
        geom.center = Center::Scalar(c);
    }
    Ok((one, geom))
}

/// Reconstruction parameters for a preview, with `reg_par` supplied by the
/// caller (the λ sweep overrides `reg_par[0]` per frame).
fn recon_params(spec: &PreviewSpec, nx: usize, reg_par: Vec<f32>) -> ReconParams {
    ReconParams {
        num_gridx: Some(nx),
        filter_name: spec.filter,
        num_iter: spec.num_iter,
        reg_par,
        ext_pad: spec.ext_pad,
        ..Default::default()
    }
}

/// Reconstruct the preview slice once per λ (overriding `reg_par[0]`, keeping
/// any further entries), and score each result on the L-curve: data residual
/// `‖A x − b‖₂` (the exact fidelity term the regularized solver minimizes —
/// `A` is the same forward projector) and the isotropic TV seminorm (roughness).
/// The corner of `log residual` vs `log roughness` is the principled λ — sharper
/// = smaller λ is *not* automatically better on real data (docs/BENCHMARKS.md
/// §10), so the pick stays the user's.
/// `(ny, nx, frames, residual, roughness)` — the λ montage and its L-curve
/// coordinates (see [`Event::LambdaSweep`]).
type LambdaSweepOut = (usize, usize, Vec<f32>, Vec<f64>, Vec<f64>);

fn run_lambda_sweep(
    engine: &Engine,
    path: &std::path::Path,
    spec: &PreviewSpec,
    lambdas: &[f32],
) -> tomoxide::Result<LambdaSweepOut> {
    let backend = engine.backend();
    let (one, geom) = prep_slice(engine, path, spec)?;
    let nx = one.array.dim().2;

    let mut frames = Vec::new();
    let mut residual = Vec::with_capacity(lambdas.len());
    let mut roughness = Vec::with_capacity(lambdas.len());
    let (mut ny, mut nxo) = (0usize, 0usize);
    for &lam in lambdas {
        let mut reg_par = spec.reg_par.clone();
        match reg_par.first_mut() {
            Some(first) => *first = lam,
            None => reg_par.push(lam),
        }
        let params = recon_params(spec, nx, reg_par);
        let vol = tomoxide::recon::recon(&one, &geom, spec.algorithm, &params, backend)?;
        let (_nz1, y, x) = vol.dims();
        (ny, nxo) = (y, x);
        let sino = tomoxide::sim::project(&vol, &geom, backend)?;
        residual.push(l2_diff(&sino.array, &one.array));
        roughness.push(tv_seminorm(&vol.array));
        frames.extend(vol.array.iter().copied());
    }
    Ok((ny, nxo, frames, residual, roughness))
}

/// Euclidean norm of the elementwise difference of two equally-shaped arrays.
fn l2_diff(a: &ndarray::Array3<f32>, b: &ndarray::Array3<f32>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Isotropic total-variation seminorm `Σ √(∂ₓ² + ∂ᵧ²)` of a one-slice volume
/// `[1, ny, nx]` — the L-curve roughness axis (forward differences, zero at the
/// far edges).
fn tv_seminorm(vol: &ndarray::Array3<f32>) -> f64 {
    let (_nz, ny, nx) = vol.dim();
    let mut sum = 0.0f64;
    for y in 0..ny {
        for x in 0..nx {
            let v = vol[[0, y, x]] as f64;
            let dx = if x + 1 < nx {
                vol[[0, y, x + 1]] as f64 - v
            } else {
                0.0
            };
            let dy = if y + 1 < ny {
                vol[[0, y + 1, x]] as f64 - v
            } else {
                0.0
            };
            sum += (dx * dx + dy * dy).sqrt();
        }
    }
    sum
}

/// A finished centre probe sweep: `[ny, nx]` per candidate concatenated
/// row-major, each candidate's `slice_focus`, and the output slice they were
/// scored on.
type CenterSweepResult = (usize, usize, Vec<f32>, Vec<f64>, usize);

/// The whole flat/dark-corrected, minus-log projection stack, held in host RAM.
///
/// Every laminography alignment step consumes all of it — the rings average the
/// projections, and both sweeps back-project them — so there is no band to read
/// lazily the way the preview jobs do. Loading it is the expensive part
/// (`nproj·nz·nx·4` bytes, and a full pass of prep), and the three steps run one
/// after another on the same data, so it is loaded once and kept.
struct PreppedStack {
    path: PathBuf,
    data: tomoxide::Tomo<f32>,
    theta: Vec<f32>,
    nz: usize,
    nx: usize,
}

impl PreppedStack {
    fn load(engine: &Engine, path: &std::path::Path) -> tomoxide::Result<Self> {
        let backend = engine.backend();
        let mut reader = tomoxide::io::open_dxchange(&path.to_string_lossy())?;
        let theta = reader.read_theta()?;
        let mut ds = reader.read_all()?;
        tomoxide::prep::normalize_dataset(&mut ds, backend)?;
        tomoxide::prep::normalize::minus_log(&mut ds.data, backend)?;
        let (_nproj, nz, nx) = ds
            .data
            .as_layout(tomoxide::data::Layout::Projection)
            .array
            .dim();
        Ok(PreppedStack {
            path: path.to_path_buf(),
            data: ds.data,
            theta,
            nz,
            nx,
        })
    }

    fn bytes(&self) -> usize {
        self.data.array.len() * std::mem::size_of::<f32>()
    }

    /// Geometry at `center`/`tilt_deg`. `tilt_deg = None` is parallel beam.
    fn geometry(&self, center: f32, tilt_deg: Option<f32>) -> Geometry {
        Geometry {
            angles: Angles(self.theta.clone()),
            center: Center::Scalar(center),
            beam: match tilt_deg {
                Some(d) => tomoxide::Beam::Laminography {
                    phi: std::f32::consts::FRAC_PI_2 + d * std::f32::consts::PI / 180.0,
                },
                None => tomoxide::Beam::Parallel,
            },
            detector: tomoxide::Detector {
                width: self.nx,
                height: self.nz,
                pixel_size: 1.0,
            },
        }
    }

    /// The output slice a sweep scores by default: the middle of the volume,
    /// which is `rh/2` under a tilt and `nz/2` without one. Those coincide only
    /// at zero tilt — the tilt stretches the reconstruction deeper than the
    /// detector is tall.
    fn default_slice(&self, tilt_deg: Option<f32>) -> usize {
        match tilt_deg {
            Some(d) => tomoxide::cuda::lamino_recon_height(self.nz, d) / 2,
            None => self.nz / 2,
        }
    }

    fn rings(
        &self,
        backend: &dyn tomoxide::Backend,
        step: usize,
    ) -> tomoxide::Result<(tomoxide::recon::center::RingCenter, ndarray::Array2<f32>)> {
        let ring = tomoxide::recon::center::find_center_rings(&self.data, backend, step)?;
        let mean = tomoxide::recon::center::mean_projection(&self.data, step)?;
        Ok((ring, mean))
    }

    fn center_sweep(
        &self,
        tilt_deg: Option<f32>,
        slice: Option<usize>,
        centers: &[f32],
    ) -> tomoxide::Result<CenterSweepResult> {
        let sz = slice.unwrap_or_else(|| self.default_slice(tilt_deg));
        let seed = centers.first().copied().unwrap_or(self.nx as f32 / 2.0);
        let probe = tomoxide::cuda::center_probe_sweep(
            &self.data,
            &self.geometry(seed, tilt_deg),
            &ReconParams::default(),
            centers,
            sz as i32,
        )?;
        let (_n, ny, nx) = probe.dim();
        let focus = (0..centers.len())
            .map(|i| tomoxide::recon::center::slice_focus(&probe.index_axis(ndarray::Axis(0), i)))
            .collect();
        Ok((ny, nx, probe.iter().copied().collect(), focus, sz))
    }

    fn tilt_scan(
        &self,
        center: f32,
        tilts: &[f32],
        on_tilt: &mut dyn FnMut(
            &tomoxide::recon::center::TiltFocus,
            ndarray::ArrayView2<f32>,
        ) -> tomoxide::Result<()>,
    ) -> tomoxide::Result<Vec<tomoxide::recon::center::TiltFocus>> {
        tomoxide::recon::center::lamino_tilt_scan(
            &self.data,
            &self.geometry(center, Some(0.0)), // beam ignored — `tilts` supplies it
            Algorithm::Fourierrec,
            &ReconParams::default(),
            tilts,
            on_tilt,
        )
    }
}

/// Load the prepped stack unless the cache already holds this file's.
fn ensure_prepped<'a>(
    engine: &Engine,
    slot: &'a mut Option<PreppedStack>,
    path: &std::path::Path,
) -> tomoxide::Result<&'a PreppedStack> {
    if slot.as_ref().is_none_or(|st| st.path != path) {
        *slot = Some(PreppedStack::load(engine, path)?);
    }
    Ok(slot.as_ref().expect("just loaded"))
}

/// Trial reconstructions of one sinogram row over a range of candidate
/// centers (`recon::center::write_center`), for the sweep montage. The FOV
/// disk mask is on (`ratio` 1.0): the corner backprojection smear it removes
/// varies with the trial center and would otherwise dominate the per-frame
/// sharpness metric.
fn run_center_sweep(
    engine: &Engine,
    path: &std::path::Path,
    row: usize,
    range: (f32, f32, f32),
) -> tomoxide::Result<(Vec<f32>, usize, usize, Vec<f32>)> {
    let backend = engine.backend();
    let mut reader = tomoxide::io::open_dxchange(&path.to_string_lossy())?;
    let (_nproj, nz, _nx, _nflat, _ndark) = reader.read_sizes()?;
    let row = row.min(nz.saturating_sub(1));
    let mut ds = reader.read_chunk(row, row + 1)?;
    tomoxide::prep::normalize_dataset(&mut ds, backend)?;
    let (centers, vols) = tomoxide::recon::center::write_center(
        &ds.data,
        &ds.theta,
        backend,
        Some(range),
        None,
        true,
        1.0,
    )?;
    let (_n, ny, nx) = vols.dim();
    Ok((centers, ny, nx, vols.iter().copied().collect()))
}

/// Auto-detect the rotation axis with the chosen method.
///
/// Vo/Entropy read + normalize only the one selected sinogram row; Pc/Sift
/// need whole projections, so they read + normalize the FULL dataset (logged
/// cost: acceptable at tuning time, a row-band reader is the M2 fix).
fn run_find_center(
    engine: &Engine,
    path: &std::path::Path,
    method: CenterMethod,
    row: usize,
    init: Option<f32>,
) -> tomoxide::Result<f32> {
    let backend = engine.backend();
    let mut reader = tomoxide::io::open_dxchange(&path.to_string_lossy())?;
    match method {
        CenterMethod::Vo | CenterMethod::Entropy => {
            let (_nproj, nz, _nx, _nflat, _ndark) = reader.read_sizes()?;
            let row = row.min(nz.saturating_sub(1));
            let mut ds = reader.read_chunk(row, row + 1)?;
            tomoxide::prep::normalize_dataset(&mut ds, backend)?;
            match method {
                // tomopy find_center_vo defaults (smin/smax ±50, srad 6,
                // step 0.25, ratio 0.5, drop 20), as in the parity tests.
                CenterMethod::Vo => tomoxide::recon::center::find_center_vo(
                    &ds.data, backend, None, -50.0, 50.0, 6.0, 0.25, 0.5, 20,
                ),
                _ => tomoxide::recon::center::find_center(
                    &ds.data, &ds.theta, backend, None, init, 0.5,
                ),
            }
        }
        CenterMethod::Pc | CenterMethod::Sift => {
            let mut ds = reader.read_all()?;
            tomoxide::prep::normalize_dataset(&mut ds, backend)?;
            let proj = ds.data.to_layout(tomoxide::Layout::Projection);
            let nproj = proj.array.dim().0;
            if nproj < 2 {
                return Err(tomoxide::Error::InvalidParam(
                    "center pc/sift needs at least two projections".into(),
                ));
            }
            // Partner of the first projection: the angle closest to θ₀ + 180°.
            let theta0 = ds.theta[0];
            let i180 = ds
                .theta
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = ((**a - theta0).abs() - std::f32::consts::PI).abs();
                    let db = ((**b - theta0).abs() - std::f32::consts::PI).abs();
                    da.total_cmp(&db)
                })
                .map(|(i, _)| i)
                .unwrap_or(nproj - 1);
            let proj0 = proj.array.index_axis(ndarray::Axis(0), 0).to_owned();
            let proj180 = proj.array.index_axis(ndarray::Axis(0), i180).to_owned();
            match method {
                CenterMethod::Pc => {
                    tomoxide::recon::center::find_center_pc(&proj0, &proj180, backend, 0.25, init)
                }
                _ => tomoxide::recon::center::find_center_sift(&proj0, &proj180, 0.5),
            }
        }
    }
}

fn probe(path: &std::path::Path) -> tomoxide::Result<DatasetMeta> {
    let mut reader = tomoxide::io::open_dxchange(&path.to_string_lossy())?;
    let (nproj, nz, nx, nflat, ndark) = reader.read_sizes()?;
    let theta = reader.read_theta()?;
    let (_ny, _nx, frame0) =
        tomoxide::io::read_h5_frame(&path.to_string_lossy(), tomoxide::io::dxchange::DATA, 0)?;
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in &frame0 {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        (lo, hi) = (0.0, 1.0);
    }
    Ok(DatasetMeta {
        path: path.to_path_buf(),
        nproj,
        nz,
        nx,
        nflat,
        ndark,
        theta,
        data_range: (lo, hi),
    })
}

/// Raw counts sinogram at one detector row: `[nproj, 1, nx]` chunk flattened
/// to a row-major `[nproj, nx]` image.
fn read_sinogram(path: &std::path::Path, row: usize) -> tomoxide::Result<(usize, usize, Vec<f32>)> {
    let mut reader = tomoxide::io::open_dxchange(&path.to_string_lossy())?;
    let ds = reader.read_chunk(row, row + 1)?;
    let (nproj, _rows, nx) = ds.data.array.dim();
    let flat: Vec<f32> = ds.data.array.iter().copied().collect();
    Ok((nproj, nx, flat))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tomoxide/tests/fixtures/streaming_dxchange.h5"
        ))
    }

    /// The rings step is the one laminography job that needs no GPU: it loads +
    /// preps the whole stack (which the sweeps then reuse) and hands back both
    /// the estimate and the mean projection it came from. The image is not a
    /// nicety — `docs/LAMINOGRAPHY_ALIGNMENT.md` §1 makes reading the bullseye by
    /// eye step one — so what it pins is that the picture reaches the UI at the
    /// projection's own shape, not that a 6-row fixture has rings to find.
    #[test]
    fn lamino_rings_returns_the_mean_projection_it_measured() {
        let engine = Engine::new(BackendKind::Cpu).unwrap();
        let mut slot = None;
        let st = ensure_prepped(&engine, &mut slot, &fixture()).unwrap();
        let (ring, mean) = st.rings(engine.backend(), 1).unwrap();
        assert_eq!(
            mean.dim(),
            (st.nz, st.nx),
            "the mean projection is not a projection-shaped image"
        );
        assert!(
            mean.iter().all(|v| v.is_finite()),
            "the mean projection has non-finite pixels"
        );
        assert!(
            ring.center.is_finite() && ring.prominence >= 0.0,
            "ring estimate is not a number: {ring:?}"
        );
        assert!(st.bytes() > 0);
    }

    /// The cache is keyed on the file, and every laminography step reuses it —
    /// loading 7.5 GB three times over would make the screen unusable. A second
    /// call must not reload, and a different file must not be served the first
    /// file's data.
    #[test]
    fn prepped_stack_is_cached_per_file() {
        let engine = Engine::new(BackendKind::Cpu).unwrap();
        let mut slot = None;
        let first = ensure_prepped(&engine, &mut slot, &fixture()).unwrap() as *const PreppedStack;
        let again = ensure_prepped(&engine, &mut slot, &fixture()).unwrap() as *const PreppedStack;
        assert_eq!(first, again, "the second call reloaded the same file");
        assert_eq!(slot.as_ref().unwrap().path, fixture());
    }

    /// `rh/2` and `nz/2` are the same row only at zero tilt: the tilt stretches
    /// the reconstruction deeper than the detector is tall, and the sweep's
    /// default slice indexes the volume, not the detector.
    #[test]
    fn default_sweep_slice_is_the_middle_of_the_volume_not_the_detector() {
        let engine = Engine::new(BackendKind::Cpu).unwrap();
        let mut slot = None;
        let st = ensure_prepped(&engine, &mut slot, &fixture()).unwrap();
        assert_eq!(st.default_slice(None), st.nz / 2);
        let tilted = st.default_slice(Some(44.0));
        assert_eq!(tilted, tomoxide::cuda::lamino_recon_height(st.nz, 44.0) / 2);
        assert!(
            tilted > st.nz / 2,
            "a 44° tilt must push the middle of the volume past the detector's middle row:              got {tilted} for nz {}",
            st.nz
        );
    }

    /// The whole preview path (probe → geometry → pipelined range recon →
    /// window-shifted in-memory volume) runs headlessly on the CPU backend.
    #[test]
    fn preview_reconstructs_one_slice() {
        let engine = Engine::new(BackendKind::Cpu).unwrap();
        let spec = PreviewSpec {
            slice: 3,
            algorithm: Algorithm::Fbp,
            center: None,
            filter: FilterName::Parzen,
            num_iter: 1,
            reg_par: Vec::new(),
            ext_pad: false,
            stripe: StripeMethod::None,
            phase: PhaseMethod::None,
        };
        let (ny, nx, data) = run_preview(&engine, &fixture(), &spec).unwrap();
        assert_eq!(data.len(), ny * nx);
        assert!(data.iter().any(|&v| v != 0.0), "all-zero reconstruction");
    }

    /// Phase retrieval previews through a row band (`RowBandReader`; the
    /// margin at these physics is 65 rows, far wider than the 6-row fixture,
    /// so the band clamps to the whole file) and still returns one finite,
    /// non-zero slice.
    #[test]
    fn preview_phase_banded_runs() {
        let engine = Engine::new(BackendKind::Cpu).unwrap();
        let spec = PreviewSpec {
            slice: 3,
            algorithm: Algorithm::Fbp,
            center: None,
            filter: FilterName::Parzen,
            num_iter: 1,
            reg_par: Vec::new(),
            ext_pad: false,
            stripe: StripeMethod::None,
            phase: PhaseMethod::Paganin {
                pixel_size: 1e-4,
                dist: 50.0,
                energy: 30.0,
                alpha: 1e-3,
            },
        };
        let (ny, nx, data) = run_preview(&engine, &fixture(), &spec).unwrap();
        assert_eq!(data.len(), ny * nx);
        assert!(data.iter().all(|v| v.is_finite()), "non-finite preview");
        assert!(data.iter().any(|&v| v != 0.0), "all-zero reconstruction");
    }

    /// The user-visible defect behind tomoxide's CUDA batch-domain padding: an
    /// iterative Tune preview (sirt, one slice, chunk = 1) on the CUDA backend
    /// came out garbage because the device solve forward-projected a 1-slice
    /// problem to zero. Skips itself when no CUDA device answers.
    #[cfg(feature = "cuda")]
    #[test]
    fn preview_iterative_cuda_is_finite_and_nonzero() {
        if tomoxide::CudaBackend::new().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let engine = Engine::new(BackendKind::Cuda).unwrap();
        if engine.name() != "cuda" {
            eprintln!("skipping: engine resolved to {}", engine.name());
            return;
        }
        let spec = PreviewSpec {
            slice: 3,
            algorithm: Algorithm::Sirt,
            center: None,
            filter: FilterName::Parzen,
            num_iter: 10,
            reg_par: Vec::new(),
            // The Tune panel default: iterative previews run support-extended.
            ext_pad: true,
            stripe: StripeMethod::None,
            phase: PhaseMethod::None,
        };
        let (ny, nx, data) = run_preview(&engine, &fixture(), &spec).unwrap();
        assert_eq!(data.len(), ny * nx);
        assert!(
            data.iter().all(|v| v.is_finite()),
            "non-finite values in iterative preview"
        );
        assert!(data.iter().any(|&v| v != 0.0), "all-zero reconstruction");
    }

    /// Stripe removal and an out-of-range slice (clamped) still reconstruct.
    #[test]
    fn preview_clamps_slice_and_applies_stripe() {
        let engine = Engine::new(BackendKind::Cpu).unwrap();
        let spec = PreviewSpec {
            slice: usize::MAX,
            algorithm: Algorithm::Fbp,
            center: Some(16.0),
            filter: FilterName::Shepp,
            num_iter: 1,
            reg_par: Vec::new(),
            ext_pad: false,
            stripe: StripeMethod::Sf { size: 3 },
            phase: PhaseMethod::None,
        };
        let (ny, nx, data) = run_preview(&engine, &fixture(), &spec).unwrap();
        assert_eq!(data.len(), ny * nx);
        assert!(data.iter().any(|&v| v != 0.0), "all-zero reconstruction");
    }

    /// SIFT center runs end-to-end — the pin is the OpenCV dynamic-linkage
    /// call chain (now a default feature), not the estimate: the synthetic
    /// fixture may legitimately give SIFT too few keypoint matches, so a
    /// clean `Err` is acceptable; a panic/abort or non-finite `Ok` is not.
    #[cfg(feature = "sift-center")]
    #[test]
    fn find_center_sift_links_and_runs() {
        let engine = Engine::new(BackendKind::Cpu).unwrap();
        let meta = probe(&fixture()).unwrap();
        match run_find_center(&engine, &fixture(), CenterMethod::Sift, meta.nz / 2, None) {
            Ok(c) => assert!(c.is_finite(), "sift center returned non-finite {c}"),
            Err(e) => eprintln!("sift center errored on the synthetic fixture (acceptable): {e}"),
        }
    }

    /// Vo and Entropy run on one normalized sinogram row of the fixture and
    /// land near the detector midline (the fixture is centered).
    #[test]
    fn find_center_vo_and_entropy_run() {
        let engine = Engine::new(BackendKind::Cpu).unwrap();
        let meta = probe(&fixture()).unwrap();
        let mid = meta.nx as f32 / 2.0;
        for method in [CenterMethod::Vo, CenterMethod::Entropy] {
            let c = run_find_center(&engine, &fixture(), method, meta.nz / 2, None).unwrap();
            assert!(
                (c - mid).abs() < meta.nx as f32 / 4.0,
                "{}: center {c} implausibly far from midline {mid}",
                method.label()
            );
        }
    }

    /// The sweep reconstructs one frame per `arange` candidate; the frame
    /// stack is candidate-major with square `ncol × ncol` frames.
    #[test]
    fn center_sweep_shapes_match_candidates() {
        let engine = Engine::new(BackendKind::Cpu).unwrap();
        let meta = probe(&fixture()).unwrap();
        let mid = meta.nx as f32 / 2.0;
        // ± 1 px in 0.5 steps, stop nudged like the view: 5 candidates.
        let range = (mid - 1.0, mid + 1.25, 0.5);
        let (centers, ny, nx, frames) =
            run_center_sweep(&engine, &fixture(), meta.nz / 2, range).unwrap();
        assert_eq!(centers.len(), 5);
        assert_eq!((ny, nx), (meta.nx, meta.nx));
        assert_eq!(frames.len(), centers.len() * ny * nx);
        assert!(centers.windows(2).all(|w| w[1] > w[0]));
        assert!(frames.iter().all(|v| v.is_finite()));
    }

    /// A TV λ sweep reconstructs one frame per λ and scores each on the
    /// L-curve. Frames are λ-major square images; residual/roughness are finite
    /// and per-λ, and stronger regularization is not less rough than weaker
    /// (roughness is monotone non-increasing in λ) — the property the L-curve
    /// display relies on.
    #[test]
    fn lambda_sweep_scores_lcurve() {
        let engine = Engine::new(BackendKind::Cpu).unwrap();
        let meta = probe(&fixture()).unwrap();
        let spec = PreviewSpec {
            slice: meta.nz / 2,
            algorithm: Algorithm::Tv,
            center: None,
            filter: FilterName::Parzen,
            num_iter: 20,
            reg_par: vec![0.001],
            ext_pad: false,
            stripe: StripeMethod::None,
            phase: PhaseMethod::None,
        };
        let lambdas = [0.001_f32, 0.01, 0.1];
        let (ny, nx, frames, residual, roughness) =
            run_lambda_sweep(&engine, &fixture(), &spec, &lambdas).unwrap();
        assert_eq!((ny, nx), (meta.nx, meta.nx));
        assert_eq!(frames.len(), lambdas.len() * ny * nx);
        assert_eq!(residual.len(), lambdas.len());
        assert_eq!(roughness.len(), lambdas.len());
        assert!(residual.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(roughness.iter().all(|v| v.is_finite() && *v >= 0.0));
        // More regularization → smoother (never rougher) reconstruction.
        assert!(
            roughness.windows(2).all(|w| w[1] <= w[0] * 1.001),
            "roughness not monotone in λ: {roughness:?}"
        );
        // The λ frames are not all identical (the sweep actually varied output).
        let size = ny * nx;
        assert!(
            frames[..size] != frames[size..2 * size],
            "λ sweep produced identical frames"
        );
    }

    /// The λ sweep runs end-to-end on CUDA (device-resident 1-slice recon +
    /// forward projection for the residual) and yields finite, varying scores.
    /// Skips itself when no CUDA device answers.
    #[cfg(feature = "cuda")]
    #[test]
    fn lambda_sweep_cuda_is_finite_and_varies() {
        if tomoxide::CudaBackend::new().is_err() {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
        let engine = Engine::new(BackendKind::Cuda).unwrap();
        if engine.name() != "cuda" {
            eprintln!("skipping: engine resolved to {}", engine.name());
            return;
        }
        let meta = probe(&fixture()).unwrap();
        let spec = PreviewSpec {
            slice: meta.nz / 2,
            algorithm: Algorithm::Tv,
            center: None,
            filter: FilterName::Parzen,
            num_iter: 20,
            reg_par: vec![0.001],
            ext_pad: true,
            stripe: StripeMethod::None,
            phase: PhaseMethod::None,
        };
        let lambdas = [0.001_f32, 0.01, 0.1];
        let (_ny, _nx, _frames, residual, roughness) =
            run_lambda_sweep(&engine, &fixture(), &spec, &lambdas).unwrap();
        assert!(residual.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(roughness.iter().all(|v| v.is_finite()));
        // The residual axis must actually move with λ — a flat axis would mean
        // the 1-slice forward projection collapsed (the batch-domain bug).
        let (rmin, rmax) = residual
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        assert!(rmax > rmin, "residual L-curve axis is flat: {residual:?}");
    }

    /// Phase correlation runs on the fixture's 0°/180° projection pair.
    #[test]
    fn find_center_pc_runs() {
        let engine = Engine::new(BackendKind::Cpu).unwrap();
        let meta = probe(&fixture()).unwrap();
        let c = run_find_center(&engine, &fixture(), CenterMethod::Pc, 0, None).unwrap();
        assert!(
            c > 0.0 && c < meta.nx as f32,
            "pc center {c} outside the detector"
        );
    }

    /// Probing the fixture yields its known dimensions.
    #[test]
    fn probe_reads_sizes_and_theta() {
        let meta = probe(&fixture()).unwrap();
        assert!(meta.nproj > 0 && meta.nz > 0 && meta.nx > 0);
        assert_eq!(meta.theta.len(), meta.nproj);
    }
}
