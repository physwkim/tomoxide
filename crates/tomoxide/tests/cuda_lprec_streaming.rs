//! Exercises the device-resident Lprec streaming reconstructor: the CUDA
//! `streaming()` handle path (`CudaFbpStream` with `lprec = Some(LpRecDev)`)
//! reuses one log-polar grid set (uploaded once) and runs pad → filter → crop →
//! spline-prefilter → per-span gather/FFT/cmul/iFFT/scatter across chunks. This
//! checks the pipelined output matches the whole-volume `reconstruct` Lprec to
//! the cuFFT floor, including the partial last chunk (nz=6, chunk 4 →
//! [0,4),[4,6)).
//!
//! Sets process-global CUDA device state, so it lives in its own test binary.

use std::sync::{Arc, Mutex};

use ndarray::{Array3, Axis};
use tomoxide::io::{self, VolumeWriter};
use tomoxide::{
    Algorithm, Angles, BackendKind, CudaBackend, Dtype, Engine, Geometry, PrepOptions, ReconParams,
    ReconSteps, Volume,
};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// Writer assembling chunks into a shared volume across the pipeline's threads.
struct SharedCollectWriter {
    vol: Arc<Mutex<Array3<f32>>>,
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

fn pearson(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.sum() / n, b.sum() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x - ma, y - mb);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

#[test]
fn cuda_lprec_pipelined_matches_whole_volume() {
    if CudaBackend::new().is_err() {
        eprintln!("skipping CUDA test: no usable CUDA device");
        return;
    }
    let engine = Engine::new(BackendKind::Cuda).unwrap();
    if engine.name() != "cuda" {
        eprintln!("skipping CUDA test: engine resolved to {}", engine.name());
        return;
    }

    let path = format!("{FIXTURES}/streaming_dxchange.h5");
    let mut probe = io::open_dxchange(&path).unwrap();
    let (nproj, nz, nx, _nf, _nd) = probe.read_sizes().unwrap();
    drop(probe);
    // lprec requires equally spaced angles over the half-open [0, π); the fixture
    // ships endpoint-inclusive [0, π] angles, so build the exclusive grid here.
    // Both recon paths use this same geometry, so the streaming-parity check is
    // unaffected by which angle convention the projections were taken at.
    let theta: Vec<f32> = (0..nproj)
        .map(|i| std::f32::consts::PI * i as f32 / nproj as f32)
        .collect();
    let geom = Geometry::parallel(Angles(theta), nx, nz, 1.0);
    let params = ReconParams {
        num_gridx: Some(nx),
        dtype: Dtype::F32,
        ..Default::default()
    };
    let prep = PrepOptions::default();

    // Whole-volume reference (the benchmark path).
    let mut rd = io::open_dxchange(&path).unwrap();
    let ds = rd.read_all().unwrap();
    let whole =
        tomoxide::reconstruct(ds, &geom, Algorithm::Lprec, &params, &prep, &engine).unwrap();

    // Pipelined device-resident streaming path. chunk 4 over nz=6 → [0,4),[4,6),
    // so the second chunk is partial and the handle's grids/`flc` buffer (sized
    // for max_nz=4) must still produce the correct per-slice result.
    let shared = Arc::new(Mutex::new(Array3::<f32>::zeros((nz, nx, nx))));
    let read_path = path.clone();
    let shared_w = Arc::clone(&shared);
    ReconSteps::new(4)
        .run_streaming_pipelined(
            move || io::open_dxchange(&read_path),
            move || Ok(Box::new(SharedCollectWriter { vol: shared_w }) as Box<dyn VolumeWriter>),
            &geom,
            Algorithm::Lprec,
            &params,
            &prep,
            &engine,
        )
        .unwrap();
    let piped = shared.lock().unwrap();

    assert_eq!(whole.array.dim(), piped.dim());
    for z in 0..nz {
        let a = whole.array.index_axis(Axis(0), z).to_owned();
        let b = piped.index_axis(Axis(0), z).to_owned();
        let r = pearson(&a, &b);
        assert!(
            r > 0.999,
            "pipelined lprec slice {z} disagrees with whole-volume: r = {r:.6}"
        );
    }
    eprintln!("device-resident lprec pipeline matches whole-volume across all {nz} slices");
}
