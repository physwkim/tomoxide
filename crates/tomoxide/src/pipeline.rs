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
        let sino = ds.data.as_layout(Layout::Sinogram); // [nz, nproj, ncols]
        let nz = sino.n_rows();

        // Pre-size single-container writers (H5/Zarr) to the full slice count
        // before the first chunk; TIFF's default no-op ignores it.
        writer.reserve(nz)?;
        let chunk = self.chunk_rows.max(1);
        let mut r0 = 0;
        let mut recon = None; // handle-reusing reconstructor, built on chunk 0
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
            let vol = recon_chunk_reusing(
                &mut recon,
                backend,
                &sub,
                geom,
                &chunk_geom,
                algorithm,
                params,
            )?;
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
        // Pre-size single-container writers (H5/Zarr) to the full slice count
        // before the first chunk; TIFF's default no-op ignores it.
        writer.reserve(nz)?;
        let chunk = self.chunk_rows.max(1);
        let mut r0 = 0;
        let mut recon = None; // handle-reusing reconstructor, built on chunk 0
        while r0 < nz {
            let r1 = (r0 + chunk).min(nz);
            let ds = reader.read_chunk(r0, r1)?;
            let chunk_geom = chunk_geometry(geom, r0, r1);
            let vol = chunk_to_volume(
                &mut recon,
                backend,
                ds,
                geom,
                &chunk_geom,
                algorithm,
                params,
                prep,
            )?;
            writer.write_chunk(&vol, r0, r1)?;
            r0 = r1;
        }
        Ok(())
    }
}

/// In-flight chunk slack per channel for [`ReconSteps::run_streaming_pipelined`].
///
/// `sync_channel(PIPELINE_DEPTH)` lets the reader run up to `PIPELINE_DEPTH`
/// chunks ahead of compute (and compute that many ahead of the writer) before
/// back-pressure stalls it, so peak host memory is bounded to ~`PIPELINE_DEPTH`
/// extra projection chunks + volume chunks rather than the whole dataset. Depth
/// 1 already overlaps all three stages (the classic double-buffer); 2 absorbs
/// per-chunk jitter at the cost of one more buffered chunk each way.
const PIPELINE_DEPTH: usize = 2;

impl ReconSteps {
    /// Pipelined out-of-core reconstruction — same numerics as
    /// [`run_streaming`](ReconSteps::run_streaming) but overlapping disk **read**,
    /// **compute**, and disk **write** across chunks (the tomocupy
    /// `rec.py`/`rec_steps.py` conveyor: read chunk *i+1* while reconstructing
    /// chunk *i* while writing chunk *i-1*).
    ///
    /// Because the HDF5 reader/writer (`rust-hdf5`'s `H5File`) are `!Send`, they
    /// cannot cross a thread boundary. Instead each I/O object is **constructed on
    /// the thread that owns it** via a factory closure: `make_reader` runs on the
    /// reader thread, `make_writer` on the writer thread, and compute (which holds
    /// the backend) stays on the calling thread. Only [`Dataset`]/[`Volume`] (both
    /// `Send`) cross the bounded channels, so no trait change is needed.
    ///
    /// Chunks carry their own row range `[r0, r1)` to the writer, which addresses
    /// each chunk by offset ([`VolumeWriter::write_chunk`](crate::io::VolumeWriter::write_chunk)),
    /// so the pipeline is correct regardless of completion order. The per-chunk
    /// work is identical to `run_streaming`, so the volume is bit-for-bit the same.
    ///
    /// Like `run_streaming`, phase retrieval is rejected (Paganin couples detector
    /// rows within a projection, which a row-chunked read cannot satisfy).
    #[allow(clippy::too_many_arguments)]
    pub fn run_streaming_pipelined<RF, WF>(
        &self,
        make_reader: RF,
        make_writer: WF,
        geom: &Geometry,
        algorithm: Algorithm,
        params: &ReconParams,
        prep: &PrepOptions,
        engine: &Engine,
    ) -> Result<()>
    where
        RF: FnOnce() -> Result<Box<dyn crate::io::DatasetReader>> + Send,
        WF: FnOnce() -> Result<Box<dyn crate::io::VolumeWriter>> + Send,
    {
        if prep.phase != PhaseMethod::None {
            return Err(crate::error::Error::InvalidParam(
                "ReconSteps::run_streaming_pipelined does not support phase retrieval \
                 (row-coupled); use run() for a phase pipeline"
                    .into(),
            ));
        }
        let backend = engine.backend();
        let chunk = self.chunk_rows.max(1);

        // read thread → compute (this thread) → write thread, each bounded.
        let (read_tx, read_rx) =
            std::sync::mpsc::sync_channel::<(usize, usize, Dataset<f32>)>(PIPELINE_DEPTH);
        let (write_tx, write_rx) =
            std::sync::mpsc::sync_channel::<(usize, usize, Volume<f32>)>(PIPELINE_DEPTH);
        // One-shot: the reader thread (which owns `read_sizes`) hands the total
        // slice count to the writer thread so it can `reserve` before chunk 0
        // (H5/Zarr pre-size their container; TIFF's no-op ignores it).
        let (nz_tx, nz_rx) = std::sync::mpsc::sync_channel::<usize>(1);

        std::thread::scope(|s| {
            // Reader thread: build its own reader, stream each chunk's rows.
            let reader_handle = s.spawn(move || -> Result<()> {
                let mut reader = make_reader()?;
                let (_nproj, nz, _nx, _nflat, _ndark) = reader.read_sizes()?;
                // Hand the writer the total slice count for `reserve` (ignore a
                // send error: a dropped receiver means the writer/compute side
                // already aborted, which the join below surfaces).
                let _ = nz_tx.send(nz);
                let mut r0 = 0;
                while r0 < nz {
                    let r1 = (r0 + chunk).min(nz);
                    let ds = reader.read_chunk(r0, r1)?;
                    // A send error means compute hung up (errored/aborted); stop
                    // reading — the compute side already owns the real error.
                    if read_tx.send((r0, r1, ds)).is_err() {
                        break;
                    }
                    r0 = r1;
                }
                Ok(())
                // read_tx drops here → compute's recv loop ends.
            });

            // Writer thread: build its own writer, drain results by row range.
            let writer_handle = s.spawn(move || -> Result<()> {
                let mut writer = make_writer()?;
                // Reserve from the reader's slice count before the first chunk.
                // A recv error means the reader died before sending nz — then no
                // chunks arrive either, so the loop below is a no-op.
                if let Ok(total_nz) = nz_rx.recv() {
                    writer.reserve(total_nz)?;
                }
                while let Ok((r0, r1, vol)) = write_rx.recv() {
                    writer.write_chunk(&vol, r0, r1)?;
                }
                Ok(())
            });

            // Compute on this thread (the backend lives here). Owns write_tx so it
            // drops at the end of the loop, terminating the writer thread.
            let compute = compute_chunks(read_rx, write_tx, backend, geom, algorithm, params, prep);

            // Join I/O threads, then surface the first error in causal order:
            // a reader error truncates the stream (compute ends Ok) so it must be
            // reported; a compute error is the root when reads succeeded.
            let read_res = reader_handle.join().expect("reader thread panicked");
            let write_res = writer_handle.join().expect("writer thread panicked");
            compute.and(read_res).and(write_res)
        })
    }
}

/// Compute stage of [`ReconSteps::run_streaming_pipelined`]: pull each read chunk,
/// run the per-chunk numeric pipeline (identical to `run_streaming`), and push the
/// reconstructed volume to the writer. Takes `write_tx` by value so it is dropped
/// when the stream ends, signalling the writer thread to finish.
#[allow(clippy::too_many_arguments)]
fn compute_chunks(
    read_rx: std::sync::mpsc::Receiver<(usize, usize, Dataset<f32>)>,
    write_tx: std::sync::mpsc::SyncSender<(usize, usize, Volume<f32>)>,
    backend: &dyn crate::backend::Backend,
    geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    prep: &PrepOptions,
) -> Result<()> {
    // Reuse one backend reconstructor across all chunks (see
    // `recon_chunk_reusing`): a many-chunk job pays the cuFFT-plan / f16-texture
    // setup once, not per chunk.
    let mut recon = None;
    while let Ok((r0, r1, ds)) = read_rx.recv() {
        let chunk_geom = chunk_geometry(geom, r0, r1);
        let vol = chunk_to_volume(
            &mut recon,
            backend,
            ds,
            geom,
            &chunk_geom,
            algorithm,
            params,
            prep,
        )?;
        // A send error means the writer hung up (errored); stop — the writer
        // thread owns the real error, surfaced after join.
        if write_tx.send((r0, r1, vol)).is_err() {
            break;
        }
    }
    Ok(())
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

/// Turn one raw read chunk into its reconstructed volume, reusing a single
/// backend reconstructor across chunks. Shared by [`run_streaming`] and the
/// pipelined `compute_chunks` so both run the identical per-chunk numeric
/// pipeline.
///
/// On the first chunk it asks the backend for a handle-reusing
/// [`StreamingAnalytic`] sized to that chunk's `nz` (the largest, since chunks
/// shrink), and reuses it for every later chunk — so a streaming run pays the
/// FBP-filter / back-projection setup (cuFFT plans, f16 texture arrays) **once**
/// instead of per chunk. `slot` carries it across calls: `None` = not yet built,
/// `Some(None)` = the backend declined for this algorithm (CPU backend,
/// gridrec/lprec/fourierrec) so use the stateless [`crate::recon::recon`],
/// `Some(Some(_))` = reuse.
///
/// Two routes, identical output:
/// - **Device-resident** (CUDA, `stripe == None`): the raw projection chunk goes
///   straight to [`StreamingAnalytic::reconstruct_chunk_raw`], which does
///   dark/flat correction, minus-log, and the projection→sinogram transpose on
///   the device — one PCIe upload, one download, no host normalize round-trip or
///   transpose copy. Skipped when stripe removal is requested (it runs on the
///   host sinogram this route never materializes), or when the reconstructor
///   declines (`Ok(None)`).
/// - **Host** (CPU/wgpu, or stripe removal requested): normalize → transpose →
///   stripe removal on the host, then `reconstruct_chunk` / stateless `recon`.
#[allow(clippy::too_many_arguments)]
fn chunk_to_volume(
    slot: &mut Option<Option<Box<dyn crate::backend::StreamingAnalytic>>>,
    backend: &dyn crate::backend::Backend,
    mut ds: Dataset<f32>,
    geom: &Geometry,
    chunk_geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    prep: &PrepOptions,
) -> Result<Volume<f32>> {
    // Build the reusable reconstructor on the first chunk, sized to that chunk's
    // (largest) projection dims.
    if slot.is_none() {
        let built = match backend.analytic_reconstruct() {
            Some(ar) => {
                ar.streaming(algorithm, params, geom, ds.data.n_cols(), ds.data.n_rows())?
            }
            None => None,
        };
        *slot = Some(built);
    }
    // Device-resident fast path: only when no host sinogram stage is needed.
    if prep.stripe == StripeMethod::None {
        if let Some(Some(s)) = slot.as_mut() {
            if let Some(vol) =
                s.reconstruct_chunk_raw(&ds.data, ds.flat.as_ref(), ds.dark.as_ref(), chunk_geom)?
            {
                return Ok(vol);
            }
        }
    }
    // Host path: normalize → transpose (to C-contiguous sinogram so the
    // back-projector can take a flat slice) → stripe removal → reconstruct from
    // the prepared sinogram (`slot` is already built above; `recon_chunk_reusing`
    // just dispatches).
    crate::prep::normalize_dataset(&mut ds, backend)?;
    let mut sino = ds.data.to_layout(Layout::Sinogram);
    sino.array = sino.array.as_standard_layout().to_owned();
    crate::prep::remove_stripe(&mut sino, prep.stripe)?;
    recon_chunk_reusing(slot, backend, &sino, geom, chunk_geom, algorithm, params)
}

/// Reconstruct one **already-prepared** sinogram chunk, reusing a single backend
/// reconstructor across chunks. Used directly by [`ReconSteps::run`] (which
/// normalizes + transposes the whole dataset up front, then slices sinogram
/// chunks) and as the host fallback inside [`chunk_to_volume`].
///
/// Builds the handle-reusing [`StreamingAnalytic`] on the first chunk (sized to
/// that chunk's `nz`, the largest) when `slot` is still `None`, and reuses it
/// afterwards; `Some(None)` means the backend declined for this algorithm so the
/// stateless [`crate::recon::recon`] is used.
fn recon_chunk_reusing(
    slot: &mut Option<Option<Box<dyn crate::backend::StreamingAnalytic>>>,
    backend: &dyn crate::backend::Backend,
    sino: &Tomo<f32>,
    geom: &Geometry,
    chunk_geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
) -> Result<Volume<f32>> {
    if slot.is_none() {
        let built = match backend.analytic_reconstruct() {
            Some(ar) => ar.streaming(algorithm, params, geom, sino.n_cols(), sino.n_rows())?,
            None => None,
        };
        *slot = Some(built);
    }
    match slot.as_mut().expect("slot built above").as_mut() {
        Some(s) => s.reconstruct_chunk(sino, chunk_geom),
        None => crate::recon::recon(sino, chunk_geom, algorithm, params, backend),
    }
}
