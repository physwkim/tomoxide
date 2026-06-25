//! High-level reconstruction pipelines.
//!
//! [`reconstruct`] is the in-memory ("full") path: preprocess → reconstruct.
//! [`ReconSteps`] is the chunked/streaming path (port of tomocupy
//! `rec_steps.py::recon_steps_all`): read → normalize → phase, then reconstruct
//! and write **by sinogram chunks**, so the volume is streamed to the writer a
//! chunk at a time instead of being held whole (see `docs/ARCHITECTURE.md` §5).

use crate::data::{Dataset, Layout, Tomo, Volume};
use crate::error::Result;
use crate::geometry::Geometry;
use crate::params::{Algorithm, PhaseMethod, ReconParams, StripeMethod};

use crate::engine::Engine;

/// Preprocessing options applied before reconstruction.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrepOptions {
    /// Stripe-removal method (default: none).
    pub stripe: StripeMethod,
    /// Phase-retrieval method (default: none).
    pub phase: PhaseMethod,
}

/// Full in-memory reconstruction: normalize → minus-log → (stripe/phase) →
/// reconstruct. Ports tomocupy `rec.py::GPURec` at a high level.
///
/// In this scaffold it runs the real preprocessing wrappers and dispatches to
/// [`crate::recon::recon`]; it surfaces `NotImplemented` from the first
/// unported numeric kernel (the FBP filter / back-projection), which is the
/// expected behaviour until milestone M1.
pub fn reconstruct(
    mut ds: Dataset<f32>,
    geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    prep: &PrepOptions,
    engine: &Engine,
) -> Result<Volume<f32>> {
    let backend = engine.backend();

    // 1. Flat/dark correction + minus-log (real on the CPU backend).
    crate::prep::normalize_dataset(&mut ds, backend)?;

    // 2. Optional projection-domain corrections.
    crate::prep::retrieve_phase(&mut ds.data, prep.phase, backend)?;

    // 3. To sinogram order, then stripe removal, then reconstruct.
    let mut sino = ds.data.to_layout(Layout::Sinogram);
    crate::prep::remove_stripe(&mut sino, prep.stripe)?;
    crate::recon::recon(&sino, geom, algorithm, params, backend)
}

/// Chunked/streaming reconstruction driver (port of tomocupy
/// `rec_steps.py::GPURecSteps::recon_steps_all`).
///
/// Reads the whole dataset into memory, normalizes + phase-retrieves the full
/// projections (these stages couple detector rows, so they run once), then
/// reconstructs and writes **by sinogram (z) chunks** of [`chunk_rows`] slices.
/// This bounds the reconstruction/stripe working set and the output volume to
/// one chunk at a time and streams each chunk to the writer as soon as it is
/// done, instead of materializing the whole volume (which [`reconstruct`] does).
///
/// Because the analytic reconstructors are per-slice independent (FBP / gridrec
/// / fourierrec / lprec have no cross-row coupling), the chunked result is
/// bit-identical to the full in-memory [`reconstruct`].
///
/// [`chunk_rows`]: ReconSteps::chunk_rows
pub struct ReconSteps {
    /// Detector rows (slices) reconstructed and written per chunk.
    pub chunk_rows: usize,
}

impl ReconSteps {
    /// Driver with the given z-chunk size (clamped to ≥1).
    pub fn new(chunk_rows: usize) -> Self {
        ReconSteps {
            chunk_rows: chunk_rows.max(1),
        }
    }

    /// Run the chunked pipeline: read → normalize → phase → (per z-chunk:
    /// stripe → reconstruct → write).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        reader: &mut dyn crate::io::DatasetReader,
        writer: &mut dyn crate::io::VolumeWriter,
        geom: &Geometry,
        algorithm: Algorithm,
        params: &ReconParams,
        prep: &PrepOptions,
        engine: &Engine,
    ) -> Result<()> {
        let backend = engine.backend();

        // Read-all + row-coupling stages once (tomocupy recon_steps_all reads
        // the whole dataset to memory before processing by steps).
        let mut ds = reader.read_all()?;
        crate::prep::normalize_dataset(&mut ds, backend)?;
        crate::prep::retrieve_phase(&mut ds.data, prep.phase, backend)?;
        let sino = ds.data.to_layout(Layout::Sinogram); // [nz, nproj, ncols]
        let nz = sino.n_rows();

        let chunk = self.chunk_rows.max(1);
        let mut r0 = 0;
        while r0 < nz {
            let r1 = (r0 + chunk).min(nz);
            // Per-slice-independent stages on this z-chunk.
            let sub = sino
                .array
                .slice_axis(ndarray::Axis(0), ndarray::Slice::from(r0..r1))
                .to_owned();
            let mut sub = Tomo::new(sub, Layout::Sinogram);
            crate::prep::remove_stripe(&mut sub, prep.stripe)?;
            let chunk_geom = chunk_geometry(geom, r0, r1);
            let vol = crate::recon::recon(&sub, &chunk_geom, algorithm, params, backend)?;
            writer.write_chunk(&vol, r0, r1)?;
            r0 = r1;
        }
        Ok(())
    }
}

impl ReconSteps {
    /// Out-of-core streaming reconstruction: read **only each chunk's detector
    /// rows** from disk ([`read_chunk`](crate::io::DatasetReader::read_chunk)),
    /// normalize and reconstruct that chunk, and write it — so the host never
    /// holds the whole dataset (unlike [`run`](ReconSteps::run), which reads it
    /// all up front). Peak memory is one chunk of projections + one chunk of the
    /// volume.
    ///
    /// Phase retrieval is **not** supported here: Paganin couples detector rows
    /// within a projection, so it cannot run on a row-chunked read — use
    /// [`run`](ReconSteps::run) for a phase pipeline. Normalize (per-pixel) and
    /// stripe/reconstruct (per-slice) chunk cleanly, so the output is identical
    /// to the full path.
    #[allow(clippy::too_many_arguments)]
    pub fn run_streaming(
        &self,
        reader: &mut dyn crate::io::DatasetReader,
        writer: &mut dyn crate::io::VolumeWriter,
        geom: &Geometry,
        algorithm: Algorithm,
        params: &ReconParams,
        prep: &PrepOptions,
        engine: &Engine,
    ) -> Result<()> {
        if prep.phase != PhaseMethod::None {
            return Err(crate::error::Error::InvalidParam(
                "ReconSteps::run_streaming does not support phase retrieval (row-coupled); \
                 use run() for a phase pipeline"
                    .into(),
            ));
        }
        let backend = engine.backend();
        let (_nproj, nz, _nx, _nflat, _ndark) = reader.read_sizes()?;
        let chunk = self.chunk_rows.max(1);
        let mut r0 = 0;
        while r0 < nz {
            let r1 = (r0 + chunk).min(nz);
            let mut ds = reader.read_chunk(r0, r1)?;
            crate::prep::normalize_dataset(&mut ds, backend)?;
            let mut sino = ds.data.to_layout(Layout::Sinogram);
            // to_layout transposes to a non-contiguous view; make it C-contiguous
            // so the (down-stream) back-projector can take a flat slice.
            sino.array = sino.array.as_standard_layout().to_owned();
            crate::prep::remove_stripe(&mut sino, prep.stripe)?;
            let chunk_geom = chunk_geometry(geom, r0, r1);
            let vol = crate::recon::recon(&sino, &chunk_geom, algorithm, params, backend)?;
            writer.write_chunk(&vol, r0, r1)?;
            r0 = r1;
        }
        Ok(())
    }
}

/// A copy of `geom` whose rotation center is restricted to detector rows
/// `[r0, r1)` (so a `PerRow` center lines up with a z-chunk; a scalar center is
/// unchanged).
fn chunk_geometry(geom: &Geometry, r0: usize, r1: usize) -> Geometry {
    use crate::geometry::Center;
    let center = match &geom.center {
        Center::Scalar(c) => Center::Scalar(*c),
        Center::PerRow(v) => Center::PerRow(v[r0..r1].to_vec()),
    };
    Geometry {
        center,
        ..geom.clone()
    }
}
