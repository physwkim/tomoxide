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
use tomoxide::{BackendKind, Engine};

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
}

/// Work requests from the UI thread.
pub enum Job {
    /// Probe a DXchange file: sizes + theta → [`Event::DatasetOpened`].
    OpenDataset(PathBuf),
    /// Read the raw sinogram at detector row `row` → [`Event::Sinogram`].
    ReadSinogram { row: usize },
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
    let _engine = match Engine::new(BackendKind::Auto) {
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
        }
    }
}

fn probe(path: &std::path::Path) -> tomoxide::Result<DatasetMeta> {
    let mut reader = tomoxide::io::open_dxchange(&path.to_string_lossy())?;
    let (nproj, nz, nx, nflat, ndark) = reader.read_sizes()?;
    let theta = reader.read_theta()?;
    Ok(DatasetMeta {
        path: path.to_path_buf(),
        nproj,
        nz,
        nx,
        nflat,
        ndark,
        theta,
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
