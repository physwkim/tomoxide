//! Single-slice regression for the CUDA analytic paths: a GUI preview (and a
//! `recon --start_row R --end_row R+1` CLI run) reconstructs ONE slice with
//! chunk = 1, but the device kernels have batch-domain minimums — the
//! z-bilinear back-projection (kernels_linerec.cuh `vr < nz-1`) back-projects
//! a 1-slice batch to zeros, and `cfunc_fourierrec` packs slice pairs so it
//! needs an even count. `streaming()` and the one-shot `reconstruct` now pad
//! the batch with zero rows to the kernel domain (≥2 slices, even for
//! Fourierrec) and drop the pad rows; these tests pin that a single slice
//! comes out non-zero and equal to the same slice of a multi-slice run.
//!
//! Sets process-global CUDA device state, so it lives in its own test binary.

use std::sync::{Arc, Mutex};

use ndarray::{Array2, Array3, Axis};
use tomoxide::io::{self, VolumeWriter};
use tomoxide::{
    Algorithm, Angles, BackendKind, CudaBackend, Engine, Geometry, PrepOptions, ReconParams,
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

/// `None` when no usable CUDA device answers (test skips itself).
fn cuda_engine() -> Option<Engine> {
    if CudaBackend::new().is_err() {
        eprintln!("skipping CUDA test: no usable CUDA device");
        return None;
    }
    let engine = Engine::new(BackendKind::Cuda).unwrap();
    if engine.name() != "cuda" {
        eprintln!("skipping CUDA test: engine resolved to {}", engine.name());
        return None;
    }
    Some(engine)
}

struct Fixture {
    path: String,
    nz: usize,
    nx: usize,
    geom: Geometry,
    params: ReconParams,
    prep: PrepOptions,
}

fn fixture() -> Fixture {
    let path = format!("{FIXTURES}/streaming_dxchange.h5");
    let mut probe = io::open_dxchange(&path).unwrap();
    let (_nproj, nz, nx, _nf, _nd) = probe.read_sizes().unwrap();
    let theta = probe.read_theta().unwrap();
    drop(probe);
    Fixture {
        path,
        nz,
        nx,
        geom: Geometry::parallel(Angles(theta), nx, nz, 1.0),
        params: ReconParams {
            num_gridx: Some(nx),
            // Ignored by the analytic methods; used by the iterative test.
            num_iter: 10,
            reg_par: vec![0.001],
            ..Default::default()
        },
        prep: PrepOptions::default(),
    }
}

/// Whole-volume streaming reference (chunk = nz, one full chunk).
fn whole_reference(engine: &Engine, fx: &Fixture, algorithm: Algorithm) -> Array3<f32> {
    let shared = Arc::new(Mutex::new(Array3::<f32>::zeros((fx.nz, fx.nx, fx.nx))));
    let read_path = fx.path.clone();
    let w = Arc::clone(&shared);
    ReconSteps::new(fx.nz)
        .run_streaming_pipelined(
            move || io::open_dxchange(&read_path),
            move || Ok(Box::new(SharedCollectWriter { vol: w }) as Box<dyn VolumeWriter>),
            &fx.geom,
            algorithm,
            &fx.params,
            &fx.prep,
            engine,
        )
        .unwrap();
    Arc::try_unwrap(shared).unwrap().into_inner().unwrap()
}

/// The GUI-preview shape: reconstruct exactly one slice with chunk = 1, so the
/// streaming handle is built at max_nz = 1 and must pad to the kernel domain.
fn single_slice(engine: &Engine, fx: &Fixture, algorithm: Algorithm, z: usize) -> Array2<f32> {
    let shared = Arc::new(Mutex::new(Array3::<f32>::zeros((fx.nz, fx.nx, fx.nx))));
    let read_path = fx.path.clone();
    let w = Arc::clone(&shared);
    ReconSteps::new(1)
        .run_streaming_pipelined_range(
            z,
            z + 1,
            move || io::open_dxchange(&read_path),
            move || Ok(Box::new(SharedCollectWriter { vol: w }) as Box<dyn VolumeWriter>),
            &fx.geom,
            algorithm,
            &fx.params,
            &fx.prep,
            engine,
        )
        .unwrap();
    let vol = shared.lock().unwrap();
    vol.index_axis(Axis(0), z).to_owned()
}

fn check_single_slice_matches_whole(algorithm: Algorithm, min_r: f32) {
    let Some(engine) = cuda_engine() else { return };
    let fx = fixture();
    let whole = whole_reference(&engine, &fx, algorithm);
    let z = fx.nz / 2;
    let one = single_slice(&engine, &fx, algorithm, z);
    assert!(
        one.iter().any(|&v| v != 0.0),
        "{algorithm:?}: single-slice streaming reconstructed all zeros"
    );
    let r = pearson(&one, &whole.index_axis(Axis(0), z).to_owned());
    assert!(
        r > min_r,
        "{algorithm:?}: single-slice disagrees with whole-volume slice: r = {r:.6} (min {min_r})"
    );
}

#[test]
fn cuda_fbp_single_slice_matches_whole() {
    // fbp/linerec share cfunc_linerec — the z-bilinear ≥2-slice kernel.
    check_single_slice_matches_whole(Algorithm::Fbp, 0.999);
}

#[test]
fn cuda_fourierrec_single_slice_matches_whole() {
    // Streaming with max_nz = 1 must round the capacity up to an even ≥2
    // (this configuration previously fell back or errored).
    check_single_slice_matches_whole(Algorithm::Fourierrec, 0.999);
}

#[test]
fn cuda_lprec_single_slice_matches_whole() {
    check_single_slice_matches_whole(Algorithm::Lprec, 0.999);
}

/// Iterative single slice: the device-resident solvers share the z-bilinear
/// forward/back-projection kernel pair, so a 1-slice problem forward-projected
/// to zero and never converged (a GUI Tune preview of sirt/tv showed garbage).
/// `IterativeReconstruct::solve` now duplicates the slice into a 2-slice
/// problem (exact: interpolation weights sum to 1 on identical rows) and drops
/// the duplicate. Slices are independent in parallel beam, so the 1-slice
/// solve must match the same slice of a multi-slice solve.
#[test]
fn cuda_iterative_single_slice_matches_multi() {
    let Some(engine) = cuda_engine() else { return };
    let fx = fixture();
    let z = fx.nz / 2;
    for algorithm in [Algorithm::Sirt, Algorithm::Tv] {
        let mut rd = io::open_dxchange(&fx.path).unwrap();
        let ds = rd.read_all().unwrap();
        let whole =
            tomoxide::reconstruct(ds, &fx.geom, algorithm, &fx.params, &fx.prep, &engine).unwrap();

        let mut rd = io::open_dxchange(&fx.path).unwrap();
        let ds1 = rd.read_chunk(z, z + 1).unwrap();
        let geom1 = Geometry::parallel(fx.geom.angles.clone(), fx.nx, 1, 1.0);
        let one =
            tomoxide::reconstruct(ds1, &geom1, algorithm, &fx.params, &fx.prep, &engine).unwrap();
        assert_eq!(one.array.dim(), (1, fx.nx, fx.nx));
        let a = one.array.index_axis(Axis(0), 0).to_owned();
        assert!(
            a.iter().all(|v| v.is_finite()),
            "{algorithm:?}: single-slice iterative produced non-finite values"
        );
        assert!(
            a.iter().any(|&v| v != 0.0),
            "{algorithm:?}: single-slice iterative reconstructed all zeros"
        );
        let r = pearson(&a, &whole.array.index_axis(Axis(0), z).to_owned());
        assert!(
            r > 0.99,
            "{algorithm:?}: single-slice iterative disagrees with multi-slice: r = {r:.6}"
        );

        // The GUI Tune preview shape: the same one slice through the streaming
        // pipeline with chunk = 1 (iterative methods have no streaming handle,
        // so the per-chunk fallback funnels into the same padded solve).
        let streamed = single_slice(&engine, &fx, algorithm, z);
        let r = pearson(&streamed, &a);
        assert!(
            r > 0.999,
            "{algorithm:?}: streamed single-slice disagrees with one-shot: r = {r:.6}"
        );
    }
}

/// The one-shot `reconstruct` path (library API, no streaming): a 1-slice
/// dataset and an odd 3-slice Fourierrec stack must both come out non-zero and
/// match the whole-volume run (the odd Fourierrec case was a hard
/// `InvalidParam` before the batch-domain pad).
#[test]
fn cuda_one_shot_padded_batches_match_whole() {
    let Some(engine) = cuda_engine() else { return };
    let fx = fixture();
    for (algorithm, z0, z1) in [
        (Algorithm::Fbp, 2usize, 3usize),
        (Algorithm::Fourierrec, 1, 4),
    ] {
        let whole = whole_reference(&engine, &fx, algorithm);
        let mut rd = io::open_dxchange(&fx.path).unwrap();
        let ds = rd.read_chunk(z0, z1).unwrap();
        let geom = Geometry::parallel(fx.geom.angles.clone(), fx.nx, z1 - z0, 1.0);
        let vol =
            tomoxide::reconstruct(ds, &geom, algorithm, &fx.params, &fx.prep, &engine).unwrap();
        assert_eq!(vol.array.dim(), (z1 - z0, fx.nx, fx.nx));
        for (i, z) in (z0..z1).enumerate() {
            let a = vol.array.index_axis(Axis(0), i).to_owned();
            assert!(
                a.iter().any(|&v| v != 0.0),
                "{algorithm:?}: one-shot slice {z} reconstructed all zeros"
            );
            let r = pearson(&a, &whole.index_axis(Axis(0), z).to_owned());
            assert!(
                r > 0.999,
                "{algorithm:?}: one-shot slice {z} disagrees with whole-volume: r = {r:.6}"
            );
        }
    }
}
