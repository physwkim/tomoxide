//! The single long-lived worker thread (docs/GUI.md §4).
//!
//! Owns the tomoxide [`Engine`] and performs every reconstruction/IO job off
//! the UI thread. HDF5 handles are `!Send`, so readers are opened *on* this
//! thread (or by the pipelined driver's own factory closures) and never cross
//! it. Jobs arrive on an mpsc channel; results go back as [`Event`]s, each
//! followed by `Context::request_repaint` so the UI wakes promptly.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};

use siplot::egui;
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
    /// A finished single-slice preview: `[ny, nx]` row-major reconstruction.
    Preview {
        generation: u64,
        slice: usize,
        ny: usize,
        nx: usize,
        data: Vec<f32>,
        millis: u128,
    },
    /// A job failed; `what` names the job for the session log.
    JobFailed { what: String, error: String },
}

/// UI-side handle: send jobs, drain events. Dropping it shuts the thread down.
pub struct Worker {
    pub jobs: Sender<Job>,
    pub events: Receiver<Event>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    pub fn spawn(ctx: egui::Context) -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::channel();
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("tomoxide-worker".into())
            .spawn(move || worker_main(job_rx, event_tx, ctx))
            .expect("spawning the worker thread");
        Worker {
            jobs: job_tx,
            events: event_rx,
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

fn worker_main(jobs: Receiver<Job>, events: Sender<Event>, ctx: egui::Context) {
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

    while let Ok(job) = jobs.recv() {
        match job {
            Job::Shutdown => break,
            Job::OpenDataset(path) => match probe(&path) {
                Ok(meta) => {
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
    let params = ReconParams {
        num_gridx: Some(nx),
        filter_name: spec.filter,
        num_iter: spec.num_iter,
        reg_par: spec.reg_par.clone(),
        ext_pad: spec.ext_pad,
        ..Default::default()
    };
    let vol = tomoxide::recon::recon(&one, &geom, spec.algorithm, &params, backend)?;
    let (_nz1, ny, nxo) = vol.dims();
    Ok((ny, nxo, vol.array.iter().copied().collect()))
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
