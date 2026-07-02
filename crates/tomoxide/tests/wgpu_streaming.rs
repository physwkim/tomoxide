//! wgpu device-resident streaming parity: the pipelined streaming path
//! (`ReconSteps::run_streaming_pipelined` → `WgpuFbpStream::reconstruct_chunk_raw`,
//! CPU normalize + host transpose + fused device recon) must match the
//! whole-volume `reconstruct` for the streaming-eligible analytic methods
//! (fbp / fourierrec), including the partial trailing chunk (nz=6, chunk 4 →
//! [0,4),[4,6)). Exercises dark/flat normalize + minus-log via the fixture's
//! white/dark frames.
#![cfg(feature = "gpu-wgpu")]

use std::sync::{Arc, Mutex};

use ndarray::{Array2, Array3, Axis};
use tomoxide::io::{self, VolumeWriter};
use tomoxide::wgpu::WgpuBackend;
use tomoxide::{
    Algorithm, Angles, BackendKind, Dtype, Engine, Geometry, PrepOptions, ReconParams, ReconSteps,
    Volume,
};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

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

fn pearson(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
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

fn check(algo: Algorithm) {
    if WgpuBackend::new().is_err() {
        eprintln!("skipping wgpu test: no usable adapter");
        return;
    }
    let engine = Engine::new(BackendKind::Wgpu).unwrap();
    assert_eq!(engine.name(), "wgpu");

    let path = format!("{FIXTURES}/streaming_dxchange.h5");
    let mut probe = io::open_dxchange(&path).unwrap();
    let (nproj, nz, nx, _nf, _nd) = probe.read_sizes().unwrap();
    drop(probe);
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

    // Whole-volume reference.
    let mut rd = io::open_dxchange(&path).unwrap();
    let ds = rd.read_all().unwrap();
    let whole = tomoxide::reconstruct(ds, &geom, algo, &params, &prep, &engine).unwrap();

    // Pipelined streaming path; chunk 4 over nz=6 → a partial trailing chunk.
    let shared = Arc::new(Mutex::new(Array3::<f32>::zeros((nz, nx, nx))));
    let read_path = path.clone();
    let shared_w = Arc::clone(&shared);
    ReconSteps::new(4)
        .run_streaming_pipelined(
            move || io::open_dxchange(&read_path),
            move || Ok(Box::new(SharedCollectWriter { vol: shared_w }) as Box<dyn VolumeWriter>),
            &geom,
            algo,
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
            "{algo:?} pipelined slice {z} disagrees with whole-volume: r = {r:.6}"
        );
    }
    eprintln!("wgpu {algo:?} streaming matches whole-volume across all {nz} slices");
}

#[test]
fn wgpu_fbp_pipelined_matches_whole_volume() {
    check(Algorithm::Fbp);
}

#[test]
fn wgpu_fourierrec_pipelined_matches_whole_volume() {
    check(Algorithm::Fourierrec);
}
