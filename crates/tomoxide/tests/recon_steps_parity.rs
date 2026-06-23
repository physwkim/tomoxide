//! Streaming/chunked reconstruction parity (M5, `ReconSteps`).
//!
//! `ReconSteps::run` reconstructs and writes by sinogram (z) chunks instead of
//! materializing the whole volume like `reconstruct`. The analytic methods are
//! per-slice independent, so the chunked output must be **bit-identical** to the
//! full in-memory reconstruction. Driven through an in-memory reader/writer so
//! the test runs offline.

use ndarray::{Array3, Axis};
use tomoxide::io::{DatasetReader, VolumeWriter};
use tomoxide::{
    reconstruct, sim, Algorithm, Angles, BackendKind, CpuBackend, Dataset, Engine, Geometry,
    Layout, PrepOptions, ReconParams, ReconSteps, Tomo, Volume,
};

/// Reader that hands back a fixed in-memory dataset (the `recon_steps_all`
/// "read whole dataset to memory" entry point).
struct MemReader {
    ds: Dataset<f32>,
}

impl DatasetReader for MemReader {
    fn read_sizes(&mut self) -> tomoxide::Result<(usize, usize, usize, usize, usize)> {
        let (nproj, nz, nx) = match self.ds.data.layout {
            Layout::Sinogram => {
                let d = self.ds.data.array.dim();
                (d.1, d.0, d.2)
            }
            Layout::Projection => {
                let d = self.ds.data.array.dim();
                (d.0, d.1, d.2)
            }
        };
        Ok((nproj, nz, nx, 0, 0))
    }
    fn read_theta(&mut self) -> tomoxide::Result<Vec<f32>> {
        Ok(self.ds.theta.clone())
    }
    fn read_all(&mut self) -> tomoxide::Result<Dataset<f32>> {
        Ok(self.ds.clone())
    }
}

/// Writer that assembles the chunked volume in memory for comparison.
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

/// Build a synthetic transmission dataset: forward-project a stacked phantom and
/// store `exp(-lineintegral)` so `minus_log` recovers the sinogram (flat/dark
/// absent ⇒ normalize is just minus-log). Returns (dataset, geometry, phantom).
fn synthetic(n: usize, nz: usize, nang: usize) -> (Dataset<f32>, Geometry) {
    let cpu = CpuBackend::new();
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let vol = Volume::new(stack);
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, nz, 1.0);
    let proj = sim::project(&vol, &geom, &cpu).unwrap();
    // transmission = exp(-attenuation integral)
    let trans = proj.array.mapv(|v| (-v).exp());
    let ds = Dataset {
        data: Tomo::new(trans, proj.layout),
        flat: None,
        dark: None,
        theta: geom.angles.0.clone(),
    };
    (ds, geom)
}

#[test]
fn recon_steps_matches_full_reconstruct() {
    let (n, nz, nang) = (64usize, 6usize, 90usize);
    let (ds, geom) = synthetic(n, nz, nang);
    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };
    let prep = PrepOptions::default(); // no stripe, no phase: isolate chunking
    let engine = Engine::new(BackendKind::Cpu).unwrap();

    // Full in-memory reconstruction.
    let full = reconstruct(ds.clone(), &geom, Algorithm::Fbp, &params, &prep, &engine).unwrap();

    // Chunked streaming reconstruction (chunk size that does not divide nz).
    let mut reader = MemReader { ds };
    let mut writer = CollectWriter::new(nz, n);
    ReconSteps::new(4)
        .run(
            &mut reader,
            &mut writer,
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .unwrap();

    assert_eq!(full.array.dim(), writer.vol.dim());
    let max_d = full
        .array
        .iter()
        .zip(writer.vol.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max_d, 0.0, "chunked recon differs from full: max |Δ| = {max_d}");
}

#[test]
fn recon_steps_chunk_size_invariant() {
    // Any chunk size (including ones that don't divide nz, and 1) gives the
    // same volume.
    let (n, nz, nang) = (48usize, 5usize, 72usize);
    let (ds, geom) = synthetic(n, nz, nang);
    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };
    let prep = PrepOptions::default();
    let engine = Engine::new(BackendKind::Cpu).unwrap();

    let run_chunked = |chunk: usize, ds: Dataset<f32>| {
        let mut reader = MemReader { ds };
        let mut writer = CollectWriter::new(nz, n);
        ReconSteps::new(chunk)
            .run(&mut reader, &mut writer, &geom, Algorithm::Fbp, &params, &prep, &engine)
            .unwrap();
        writer.vol
    };

    let a = run_chunked(1, ds.clone());
    let b = run_chunked(2, ds.clone());
    let c = run_chunked(nz, ds.clone());
    assert_eq!(a, b, "chunk 1 vs 2 differ");
    assert_eq!(a, c, "chunk 1 vs whole differ");
}
