//! `io::InMemoryWriter` — the in-memory VolumeWriter used by GUI previews.
//!
//! Checks the writer contract directly (reserve → chunked writes → shared
//! buffer, shape validation) and end-to-end equality with a plain collecting
//! writer through both the sequential and pipelined drivers.

use ndarray::{Array3, Axis};
use tomoxide::io::{self, InMemoryWriter, VolumeWriter};
use tomoxide::{
    Algorithm, Angles, BackendKind, Engine, Error, Geometry, PrepOptions, ReconParams, ReconSteps,
    Volume,
};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn vol(nz: usize, ny: usize, nx: usize, fill: f32) -> Volume<f32> {
    Volume::new(Array3::from_elem((nz, ny, nx), fill))
}

#[test]
fn chunks_assemble_into_global_range() {
    let mut w = InMemoryWriter::new();
    let buf = w.buffer();
    w.reserve(5).unwrap();
    w.write_chunk(&vol(2, 3, 4, 1.0), 0, 2).unwrap();
    w.write_chunk(&vol(2, 3, 4, 2.0), 2, 4).unwrap();
    w.write_chunk(&vol(1, 3, 4, 3.0), 4, 5).unwrap();
    w.finalize().unwrap();
    drop(w); // the shared handle must outlive the writer

    let guard = buf.lock().unwrap();
    assert_eq!(guard.dims(), Some((5, 3, 4)));
    let v = guard.to_volume().unwrap();
    assert!(v
        .array
        .slice(ndarray::s![0..2, .., ..])
        .iter()
        .all(|&x| x == 1.0));
    assert!(v
        .array
        .slice(ndarray::s![2..4, .., ..])
        .iter()
        .all(|&x| x == 2.0));
    assert!(v
        .array
        .slice(ndarray::s![4..5, .., ..])
        .iter()
        .all(|&x| x == 3.0));
}

#[test]
fn on_chunk_reports_each_global_range() {
    use std::sync::{Arc, Mutex};
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s = Arc::clone(&seen);
    let mut w = InMemoryWriter::new().with_on_chunk(move |a, b| s.lock().unwrap().push((a, b)));
    w.reserve(4).unwrap();
    w.write_chunk(&vol(3, 2, 2, 0.5), 0, 3).unwrap();
    w.write_chunk(&vol(1, 2, 2, 0.5), 3, 4).unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![(0, 3), (3, 4)]);
}

#[test]
fn shape_violations_are_rejected() {
    // Chunk slice count != end - start.
    let mut w = InMemoryWriter::new();
    w.reserve(4).unwrap();
    let err = w.write_chunk(&vol(2, 2, 2, 0.0), 0, 3).unwrap_err();
    assert!(matches!(err, Error::ShapeMismatch { .. }), "got: {err}");

    // Range beyond the reserved slice count.
    let mut w = InMemoryWriter::new();
    w.reserve(2).unwrap();
    let err = w.write_chunk(&vol(2, 2, 2, 0.0), 1, 3).unwrap_err();
    assert!(matches!(err, Error::ShapeMismatch { .. }), "got: {err}");

    // Cross-section changes between chunks.
    let mut w = InMemoryWriter::new();
    w.reserve(4).unwrap();
    w.write_chunk(&vol(2, 2, 2, 0.0), 0, 2).unwrap();
    let err = w.write_chunk(&vol(2, 3, 3, 0.0), 2, 4).unwrap_err();
    assert!(matches!(err, Error::ShapeMismatch { .. }), "got: {err}");
}

/// Reference writer: assigns chunks straight into a preallocated Array3.
struct CollectWriter {
    vol: Array3<f32>,
}

impl VolumeWriter for CollectWriter {
    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, end: usize) -> tomoxide::Result<()> {
        self.vol
            .slice_axis_mut(Axis(0), ndarray::Slice::from(start..end))
            .assign(&vol.array);
        Ok(())
    }
}

#[test]
fn matches_collect_writer_through_run_streaming() {
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
    let prep = PrepOptions::default();
    let steps = ReconSteps::new(4);

    let mut reader = io::open_dxchange(&path).unwrap();
    let mut reference = CollectWriter {
        vol: Array3::zeros((nz, nx, nx)),
    };
    steps
        .run_streaming(
            &mut *reader,
            &mut reference,
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap();

    // Sequential driver.
    let mut reader = io::open_dxchange(&path).unwrap();
    let mut mem = InMemoryWriter::new();
    let buf = mem.buffer();
    steps
        .run_streaming(
            &mut *reader,
            &mut mem,
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap();
    assert_eq!(
        buf.lock().unwrap().to_volume().unwrap().array,
        reference.vol
    );

    // Pipelined driver: writer is built and dropped on the writer thread; the
    // shared handle taken up-front is how the caller gets the result back.
    let mem = InMemoryWriter::new();
    let buf = mem.buffer();
    let mut mem = Some(mem);
    let p = path.clone();
    steps
        .run_streaming_pipelined(
            move || io::open_dxchange(&p),
            move || Ok(Box::new(mem.take().expect("writer built once")) as Box<dyn VolumeWriter>),
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap();
    assert_eq!(
        buf.lock().unwrap().to_volume().unwrap().array,
        reference.vol
    );
}
