//! Exercises the device-resident Fourierrec streaming reconstructor: the CUDA
//! `streaming()` handle path (`CudaFbpStream` with `fourier = true`) reuses one
//! `cfunc_filter`/`cfunc_fourierrec` handle set and runs pad → filter → crop →
//! pack-pairs → `cfunc_fourierrec` → unpack-pairs across chunks. This checks the
//! pipelined output matches the whole-volume `reconstruct` Fourierrec to the
//! cuFFT floor, including the partial last chunk (nz=6, chunk 4 → [0,4),[4,6)).
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
fn cuda_fourierrec_pipelined_matches_whole_volume() {
    check_pipelined_matches_whole_volume(Dtype::F32, 0.999);
}

/// f16 Fourierrec also has a device-resident streaming tail (pack → `cfunc_
/// fourierrec` (f16) → unpack); it must match the whole-volume f16 Fourierrec.
/// Half precision + the `atomicAdd` gather floor make the agreement coarser than
/// f32, so the per-slice correlation bar is relaxed.
#[test]
fn cuda_fourierrec_f16_pipelined_matches_whole_volume() {
    check_pipelined_matches_whole_volume(Dtype::F16, 0.99);
}

fn check_pipelined_matches_whole_volume(dtype: Dtype, min_r: f32) {
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
    let (_nproj, nz, nx, _nf, _nd) = probe.read_sizes().unwrap();
    let theta = probe.read_theta().unwrap();
    drop(probe);
    let geom = Geometry::parallel(Angles(theta), nx, nz, 1.0);
    let params = ReconParams {
        num_gridx: Some(nx),
        dtype,
        ..Default::default()
    };
    let prep = PrepOptions::default();

    // Whole-volume reference (the benchmark path).
    let mut rd = io::open_dxchange(&path).unwrap();
    let ds = rd.read_all().unwrap();
    let whole =
        tomoxide::reconstruct(ds, &geom, Algorithm::Fourierrec, &params, &prep, &engine).unwrap();

    // Pipelined device-resident streaming path. chunk 4 over nz=6 → [0,4),[4,6),
    // so the second chunk is partial and the handle's `max_nz=4` pair-packing
    // must still produce the correct per-slice result.
    let shared = Arc::new(Mutex::new(Array3::<f32>::zeros((nz, nx, nx))));
    let read_path = path.clone();
    let shared_w = Arc::clone(&shared);
    ReconSteps::new(4)
        .run_streaming_pipelined(
            move || io::open_dxchange(&read_path),
            move || Ok(Box::new(SharedCollectWriter { vol: shared_w }) as Box<dyn VolumeWriter>),
            &geom,
            Algorithm::Fourierrec,
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
            r > min_r,
            "pipelined fourierrec slice {z} disagrees with whole-volume: r = {r:.6} (min {min_r})"
        );
    }
    eprintln!("device-resident fourierrec pipeline matches whole-volume across all {nz} slices");
}
