//! Cooperative cancellation of the chunked drivers (`CancelToken`).
//!
//! The token is checked at chunk boundaries: a pre-cancelled token stops the
//! run before any chunk is written, and a token fired mid-run truncates it
//! (already-written chunks stay) with `Error::Cancelled`.

use ndarray::{Array3, Axis};
use tomoxide::io::{self, VolumeWriter};
use tomoxide::{
    Algorithm, Angles, BackendKind, CancelToken, Engine, Error, Geometry, PrepOptions, ReconParams,
    ReconSteps, Volume,
};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// Collects chunks and optionally fires a cancel token after the first write.
struct CancellingWriter {
    vol: Array3<f32>,
    chunks_written: usize,
    cancel_after_first: Option<CancelToken>,
}

impl CancellingWriter {
    fn new(nz: usize, n: usize, cancel_after_first: Option<CancelToken>) -> Self {
        CancellingWriter {
            vol: Array3::zeros((nz, n, n)),
            chunks_written: 0,
            cancel_after_first,
        }
    }
}

impl VolumeWriter for CancellingWriter {
    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, end: usize) -> tomoxide::Result<()> {
        self.vol
            .slice_axis_mut(Axis(0), ndarray::Slice::from(start..end))
            .assign(&vol.array);
        self.chunks_written += 1;
        if let Some(t) = &self.cancel_after_first {
            t.cancel();
        }
        Ok(())
    }
}

fn setup() -> (String, Engine, Geometry, ReconParams, usize, usize) {
    let path = format!("{FIXTURES}/streaming_dxchange.h5");
    let engine = Engine::new(BackendKind::Cpu).unwrap();
    let mut probe = io::open_dxchange(&path).unwrap();
    let (_nproj, nz, nx, _nf, _nd) = probe.read_sizes().unwrap();
    let theta = probe.read_theta().unwrap();
    let geom = Geometry::parallel(Angles(theta), nx, nz, 1.0);
    let params = ReconParams {
        num_gridx: Some(nx),
        ..Default::default()
    };
    (path, engine, geom, params, nz, nx)
}

#[test]
fn pre_cancelled_token_stops_before_any_chunk() {
    let (path, engine, geom, params, nz, nx) = setup();
    let prep = PrepOptions::default();
    let token = CancelToken::new();
    token.cancel();

    let mut reader = io::open_dxchange(&path).unwrap();
    let mut writer = CancellingWriter::new(nz, nx, None);
    let err = ReconSteps::new(2)
        .with_cancel(token)
        .run_streaming(
            &mut *reader,
            &mut writer,
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap_err();
    assert!(matches!(err, Error::Cancelled), "got: {err}");
    assert_eq!(writer.chunks_written, 0, "no chunk may be written");
}

#[test]
fn mid_run_cancel_truncates_run_all() {
    let (path, engine, geom, params, nz, nx) = setup();
    let prep = PrepOptions::default();
    let token = CancelToken::new();

    // chunk=2 over nz=6 → 3 chunks; the writer fires the token during the
    // first write, so the sequential driver must stop at the next boundary.
    let mut reader = io::open_dxchange(&path).unwrap();
    let mut writer = CancellingWriter::new(nz, nx, Some(token.clone()));
    let err = ReconSteps::new(2)
        .with_cancel(token)
        .run(
            &mut *reader,
            &mut writer,
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap_err();
    assert!(matches!(err, Error::Cancelled), "got: {err}");
    assert_eq!(
        writer.chunks_written, 1,
        "sequential driver must stop after the chunk that fired the token"
    );
}

#[test]
fn mid_run_cancel_surfaces_from_pipelined() {
    use std::sync::{Arc, Mutex};
    let (path, engine, geom, params, nz, nx) = setup();
    let prep = PrepOptions::default();
    let token = CancelToken::new();

    // The writer thread fires the token on its first chunk. In-flight chunks
    // (PIPELINE_DEPTH) may still land, so only the error — not the exact
    // written count — is deterministic here.
    struct SharedCancellingWriter {
        vol: Arc<Mutex<Array3<f32>>>,
        token: CancelToken,
    }
    impl VolumeWriter for SharedCancellingWriter {
        fn write_chunk(
            &mut self,
            vol: &Volume<f32>,
            start: usize,
            end: usize,
        ) -> tomoxide::Result<()> {
            self.vol
                .lock()
                .unwrap()
                .slice_axis_mut(Axis(0), ndarray::Slice::from(start..end))
                .assign(&vol.array);
            self.token.cancel();
            Ok(())
        }
    }

    let shared = Arc::new(Mutex::new(Array3::<f32>::zeros((nz, nx, nx))));
    let w = Arc::clone(&shared);
    let wt = token.clone();
    // chunk=1 over nz=6 → 6 chunks: more chunks than the pipeline depth, so a
    // post-cancel boundary check is guaranteed to run.
    let err = ReconSteps::new(1)
        .with_cancel(token)
        .run_streaming_pipelined(
            move || io::open_dxchange(&path),
            move || {
                Ok(Box::new(SharedCancellingWriter { vol: w, token: wt }) as Box<dyn VolumeWriter>)
            },
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap_err();
    assert!(matches!(err, Error::Cancelled), "got: {err}");
}

#[test]
fn default_token_never_cancels() {
    let (path, engine, geom, params, nz, nx) = setup();
    let prep = PrepOptions::default();
    let mut reader = io::open_dxchange(&path).unwrap();
    let mut writer = CancellingWriter::new(nz, nx, None);
    ReconSteps::new(2)
        .run_streaming(
            &mut *reader,
            &mut writer,
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap();
    assert_eq!(writer.chunks_written, 3, "all chunks written");
}
