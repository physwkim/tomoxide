//! Out-of-core streaming reconstruction parity (M5, `ReconSteps::run_streaming`).
//!
//! `run_streaming` reads only each chunk's detector rows from the HDF5 file
//! (`DatasetReader::read_chunk`, a hyperslab) instead of loading the whole
//! dataset like `run`. Because normalize is per-pixel and stripe/reconstruct are
//! per-slice, the out-of-core output is bit-identical to the full read-all path.
//! Driven over a multi-row DXchange fixture
//! (`tools/gen_dxchange_streaming_fixture.py`).

use ndarray::{Array3, Axis};
use tomoxide::io::{self, VolumeWriter};
use tomoxide::{
    Angles, BackendKind, Engine, Geometry, PrepOptions, ReconParams, ReconSteps, Volume,
};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// Writer that assembles written chunks into one volume for comparison.
struct CollectWriter {
    vol: Array3<f32>,
}
impl CollectWriter {
    fn new(nz: usize, n: usize) -> Self {
        CollectWriter {
            vol: Array3::zeros((nz, n, n)),
        }
    }
}
impl VolumeWriter for CollectWriter {
    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, end: usize) -> tomoxide::Result<()> {
        self.vol
            .slice_axis_mut(Axis(0), ndarray::Slice::from(start..end))
            .assign(&vol.array);
        Ok(())
    }
}

/// Writer that assembles chunks into a volume shared across threads, so the
/// pipelined path (which builds its writer on the writer thread) can be compared
/// after the run.
struct SharedCollectWriter {
    vol: std::sync::Arc<std::sync::Mutex<Array3<f32>>>,
}
impl VolumeWriter for SharedCollectWriter {
    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, end: usize) -> tomoxide::Result<()> {
        self.vol
            .lock()
            .unwrap()
            .slice_axis_mut(Axis(0), ndarray::Slice::from(start..end))
            .assign(&vol.array);
        Ok(())
    }
}

#[test]
fn run_streaming_matches_run_all() {
    use tomoxide::Algorithm;
    let path = format!("{FIXTURES}/streaming_dxchange.h5");
    let engine = Engine::new(BackendKind::Cpu).unwrap();

    // Geometry from the file (sizes + angles); rotation center = detector mid.
    let mut probe = io::open_dxchange(&path).unwrap();
    let (_nproj, nz, nx, _nf, _nd) = probe.read_sizes().unwrap();
    let theta = probe.read_theta().unwrap();
    let geom = Geometry::parallel(Angles(theta), nx, nz, 1.0);
    let params = ReconParams {
        num_gridx: Some(nx),
        ..Default::default()
    };
    let prep = PrepOptions::default();

    // Full (read_all) path.
    let mut r_full = io::open_dxchange(&path).unwrap();
    let mut w_full = CollectWriter::new(nz, nx);
    ReconSteps::new(4)
        .run(
            &mut *r_full,
            &mut w_full,
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap();

    // Out-of-core (read_chunk) path, chunk size 4 over nz=6 → chunks [0,4),[4,6).
    let mut r_oc = io::open_dxchange(&path).unwrap();
    let mut w_oc = CollectWriter::new(nz, nx);
    ReconSteps::new(4)
        .run_streaming(
            &mut *r_oc,
            &mut w_oc,
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap();

    assert_eq!(w_full.vol.dim(), w_oc.vol.dim());
    let max_d = w_full
        .vol
        .iter()
        .zip(w_oc.vol.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_d, 0.0,
        "out-of-core differs from read-all: max |Δ| = {max_d}"
    );
}

#[test]
fn run_streaming_pipelined_matches_run_streaming() {
    use std::sync::{Arc, Mutex};
    use tomoxide::Algorithm;
    let path = format!("{FIXTURES}/streaming_dxchange.h5");
    let engine = Engine::new(BackendKind::Cpu).unwrap();

    let mut probe = io::open_dxchange(&path).unwrap();
    let (_nproj, nz, nx, _nf, _nd) = probe.read_sizes().unwrap();
    let theta = probe.read_theta().unwrap();
    drop(probe);
    let geom = Geometry::parallel(Angles(theta), nx, nz, 1.0);
    let params = ReconParams {
        num_gridx: Some(nx),
        ..Default::default()
    };
    let prep = PrepOptions::default();

    // Sequential streaming reference (already proven == read-all run).
    let mut r_seq = io::open_dxchange(&path).unwrap();
    let mut w_seq = CollectWriter::new(nz, nx);
    ReconSteps::new(4)
        .run_streaming(
            &mut *r_seq,
            &mut w_seq,
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap();

    // Pipelined path: reader/writer built on their own threads via factories;
    // the writer collects into a shared volume.
    let shared = Arc::new(Mutex::new(Array3::<f32>::zeros((nz, nx, nx))));
    let read_path = path.clone();
    let shared_w = Arc::clone(&shared);
    ReconSteps::new(4)
        .run_streaming_pipelined(
            move || io::open_dxchange(&read_path),
            move || Ok(Box::new(SharedCollectWriter { vol: shared_w }) as Box<dyn VolumeWriter>),
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap();
    let piped = shared.lock().unwrap();

    assert_eq!(w_seq.vol.dim(), piped.dim());
    let max_d = w_seq
        .vol
        .iter()
        .zip(piped.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_d, 0.0,
        "pipelined differs from sequential streaming: max |Δ| = {max_d}"
    );
}

#[test]
fn run_streaming_rejects_phase() {
    use tomoxide::{Algorithm, PhaseMethod};
    let path = format!("{FIXTURES}/streaming_dxchange.h5");
    let engine = Engine::new(BackendKind::Cpu).unwrap();
    let mut probe = io::open_dxchange(&path).unwrap();
    let (_p, nz, nx, _f, _d) = probe.read_sizes().unwrap();
    let theta = probe.read_theta().unwrap();
    let geom = Geometry::parallel(Angles(theta), nx, nz, 1.0);
    let params = ReconParams {
        num_gridx: Some(nx),
        ..Default::default()
    };
    let prep = PrepOptions {
        phase: PhaseMethod::Paganin {
            pixel_size: 1e-4,
            dist: 50.0,
            energy: 30.0,
            alpha: 1e-3,
        },
        ..Default::default()
    };
    let mut reader = io::open_dxchange(&path).unwrap();
    let mut writer = CollectWriter::new(nz, nx);
    let err = ReconSteps::new(4).run_streaming(
        &mut *reader,
        &mut writer,
        &geom,
        Algorithm::Fbp,
        &params,
        &prep,
        &engine,
    );
    assert!(err.is_err(), "phase must be rejected in out-of-core mode");
}
