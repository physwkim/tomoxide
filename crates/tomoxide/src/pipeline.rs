//! High-level reconstruction pipelines.
//!
//! [`reconstruct`] is the in-memory ("full") path: preprocess → reconstruct.
//! [`ReconSteps`] is the chunked/streaming path (port of tomocupy
//! `rec_steps.py::recon_steps_all`): read → normalize → phase, then reconstruct
//! and write **by sinogram chunks**, so the volume is streamed to the writer a
//! chunk at a time instead of being held whole (see `docs/ARCHITECTURE.md` §5).

use crate::data::{Dataset, Frames, Layout, Tomo, Volume};
use crate::error::Result;
use crate::geometry::Geometry;
use crate::params::{Algorithm, PhaseMethod, ReconParams, StripeMethod};
use ndarray::Array3;

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
        // Full-volume case = the whole z-range; the reader clamps `usize::MAX` to nz.
        self.run_streaming_pipelined_range(
            0,
            usize::MAX,
            make_reader,
            make_writer,
            geom,
            algorithm,
            params,
            prep,
            engine,
        )
    }

    /// Like [`run_streaming_pipelined`](Self::run_streaming_pipelined) but
    /// reconstructs and writes only the **contiguous z-shard** `[z_start, z_end)`
    /// (clamped to the dataset's slice count). Each shard reads only its own rows
    /// and writes them at their *global* slice offset, so several shards — one per
    /// GPU, each in its own process via `CUDA_VISIBLE_DEVICES` — fan a single
    /// reconstruction across devices without coordinating: the per-slice analytic
    /// methods have no cross-row coupling and the TIFF writer keys each slice by
    /// its global index, so disjoint shards never collide.
    ///
    /// `reserve(total_nz)` still receives the *full* slice count (the writer sizes
    /// its container to the whole volume); only the read/reconstruct/write loop is
    /// bounded to the shard. Single-container writers (H5/Zarr) must therefore not
    /// be sharded across processes — the caller restricts sharding to TIFF output.
    #[allow(clippy::too_many_arguments)]
    pub fn run_streaming_pipelined_range<RF, WF>(
        &self,
        z_start: usize,
        z_end: usize,
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
            std::sync::mpsc::sync_channel::<(usize, usize, ChunkMsg)>(PIPELINE_DEPTH);
        let (write_tx, write_rx) =
            std::sync::mpsc::sync_channel::<(usize, usize, Volume<f32>)>(PIPELINE_DEPTH);
        // One-shot: the reader thread (which owns `read_sizes`) hands the total
        // slice count to the writer thread so it can `reserve` before chunk 0
        // (H5/Zarr pre-size their container; TIFF's no-op ignores it).
        let (nz_tx, nz_rx) = std::sync::mpsc::sync_channel::<usize>(1);
        // Pinned-buffer recycle channel (raw mode only): the reader fills a
        // staging buffer pulled from here and the compute thread returns it after
        // upload, so the `POOL` page-locked buffers are allocated ONCE — a fresh
        // `cudaHostAlloc` per chunk would cost more than the staging copy it saves.
        // `POOL = PIPELINE_DEPTH + 1` keeps the reader a full depth ahead while one
        // buffer is in flight on the compute thread.
        const POOL: usize = PIPELINE_DEPTH + 1;
        let (free_tx, free_rx) =
            std::sync::mpsc::sync_channel::<Box<dyn crate::backend::HostBuffer>>(POOL);
        let free_tx_reader = free_tx.clone();
        // Output-volume recycle channel: the writer hands each spent volume's
        // backing `Vec<f32>` back to the compute thread, which feeds it to the
        // reconstructor (`give_reuse_buffer`) so the device→host download copies
        // into a warm allocation instead of a fresh 536 MB one (~190 ms of
        // first-touch page-faults per chunk avoided). Unbounded so the writer
        // never blocks returning a buffer; circulation is bounded by the volume
        // channel depth, so it cannot grow without limit.
        let (out_free_tx, out_free_rx) = std::sync::mpsc::channel::<Vec<f32>>();

        std::thread::scope(|s| {
            // Reader thread: build its own reader, stream each chunk's rows.
            let reader_handle = s.spawn(move || -> Result<()> {
                let mut reader = make_reader()?;
                let (nproj, nz, nx, _nflat, _ndark) = reader.read_sizes()?;
                // This shard's row range, clamped to the dataset (full volume when
                // z_end == usize::MAX). z0 >= z1 means an empty shard: still hand the
                // writer the full count so it can reserve, then read nothing.
                let z0 = z_start.min(nz);
                let z1 = z_end.min(nz);
                // Hand the writer the total slice count for `reserve` (ignore a
                // send error: a dropped receiver means the writer/compute side
                // already aborted, which the join below surfaces). This is the full
                // `nz`, not the shard extent — the writer sizes the whole volume.
                let _ = nz_tx.send(nz);
                // CUDA's device-resident raw path wants chunks read straight into a
                // pinned staging buffer (direct-DMA H2D, no driver staging copy);
                // CPU/wgpu take the owned-`Dataset` path.
                let raw_mode = backend.wants_raw_chunks();
                if raw_mode {
                    // Prime the pool with `POOL` buffers sized to the largest chunk,
                    // allocated once and recycled for the whole run.
                    let max_len = nproj * chunk.min(nz.max(1)) * nx;
                    for _ in 0..POOL {
                        if free_tx_reader
                            .send(backend.alloc_host_buffer(max_len))
                            .is_err()
                        {
                            return Ok(());
                        }
                    }
                    let mut r0 = z0;
                    while r0 < z1 {
                        let r1 = (r0 + chunk).min(z1);
                        let rows = r1 - r0;
                        let need = nproj * rows * nx;
                        // Recycle a pinned buffer (blocks until compute returns one).
                        let mut buf = match free_rx.recv() {
                            Ok(b) => b,
                            Err(_) => break, // compute hung up
                        };
                        let aux =
                            reader.read_chunk_into(r0, r1, &mut buf.as_mut_slice()[..need])?;
                        let msg = ChunkMsg::Raw {
                            data: buf,
                            dims: (nproj, rows, nx),
                            flat: aux.flat,
                            dark: aux.dark,
                            theta: aux.theta,
                        };
                        if read_tx.send((r0, r1, msg)).is_err() {
                            break;
                        }
                        r0 = r1;
                    }
                } else {
                    let mut r0 = z0;
                    while r0 < z1 {
                        let r1 = (r0 + chunk).min(z1);
                        let ds = reader.read_chunk(r0, r1)?;
                        // A send error means compute hung up (errored/aborted); stop
                        // reading — the compute side already owns the real error.
                        if read_tx.send((r0, r1, ChunkMsg::Host(ds))).is_err() {
                            break;
                        }
                        r0 = r1;
                    }
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
                    // Return the spent volume's backing buffer to the compute
                    // thread for reuse (ignore a send error: a finished compute
                    // thread just lets the buffer drop). `into_raw_vec_and_offset`
                    // is a move, not a copy — the volume is standard C-layout here.
                    let _ = out_free_tx.send(vol.array.into_raw_vec_and_offset().0);
                }
                Ok(())
            });

            // Compute on this thread (the backend lives here). Owns write_tx so it
            // drops at the end of the loop, terminating the writer thread, and
            // `free_tx` so spent pinned buffers return to the reader's pool.
            let compute = compute_chunks(
                read_rx,
                write_tx,
                free_tx,
                out_free_rx,
                backend,
                geom,
                algorithm,
                params,
                prep,
            );

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
/// One read chunk crossing reader→compute. Either **raw** projections read
/// straight into a host staging buffer (pinned for CUDA, for the device-resident
/// upload in [`reconstruct_chunk_raw`](crate::backend::StreamingAnalytic::reconstruct_chunk_raw)),
/// or a fully host-assembled [`Dataset`] (CPU/wgpu). The reader picks per
/// [`Backend::wants_raw_chunks`](crate::backend::Backend::wants_raw_chunks).
enum ChunkMsg {
    /// Raw projection chunk `[nproj, rows, nx]` in a staging buffer, plus the
    /// chunk's flat/dark frames and angles.
    Raw {
        data: Box<dyn crate::backend::HostBuffer>,
        dims: (usize, usize, usize),
        flat: Option<Frames<f32>>,
        dark: Option<Frames<f32>>,
        theta: Vec<f32>,
    },
    /// Fully host-assembled dataset (the original owned-array path).
    Host(Dataset<f32>),
}

#[allow(clippy::too_many_arguments)]
fn compute_chunks(
    read_rx: std::sync::mpsc::Receiver<(usize, usize, ChunkMsg)>,
    write_tx: std::sync::mpsc::SyncSender<(usize, usize, Volume<f32>)>,
    free_tx: std::sync::mpsc::SyncSender<Box<dyn crate::backend::HostBuffer>>,
    out_free: std::sync::mpsc::Receiver<Vec<f32>>,
    backend: &dyn crate::backend::Backend,
    geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    prep: &PrepOptions,
) -> Result<()> {
    // Reuse one backend reconstructor across all chunks (see
    // `recon_chunk_reusing`): a many-chunk job pays the cuFFT-plan / f16-texture
    // setup once, not per chunk.
    let mut recon: Option<Option<Box<dyn crate::backend::StreamingAnalytic>>> = None;
    while let Ok((r0, r1, msg)) = read_rx.recv() {
        // Feed any volume buffers the writer has returned back to the
        // reconstructor so this chunk's device→host download reuses a warm
        // allocation (no-op for backends that build the volume on the host).
        while let Ok(buf) = out_free.try_recv() {
            if let Some(Some(r)) = recon.as_mut() {
                r.give_reuse_buffer(buf);
            }
        }
        let chunk_geom = chunk_geometry(geom, r0, r1);
        let vol = match msg {
            ChunkMsg::Raw {
                data,
                dims,
                flat,
                dark,
                theta,
            } => {
                let vol = raw_chunk_to_volume(
                    &mut recon,
                    backend,
                    data.as_ref(),
                    dims,
                    flat,
                    dark,
                    theta,
                    geom,
                    &chunk_geom,
                    algorithm,
                    params,
                    prep,
                )?;
                // Return the pinned buffer to the reader's pool for reuse (the H2D
                // inside `reconstruct_chunk_raw` is synchronous, so it is already
                // free). Ignore a send error: a finished reader just drops it.
                let _ = free_tx.send(data);
                vol
            }
            ChunkMsg::Host(ds) => chunk_to_volume(
                &mut recon,
                backend,
                ds,
                geom,
                &chunk_geom,
                algorithm,
                params,
                prep,
            )?,
        };
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
/// - **Device-resident** (CUDA): the raw projection chunk goes straight to
///   [`StreamingAnalytic::reconstruct_chunk_raw`], which does dark/flat
///   correction, minus-log, the projection→sinogram transpose, and any
///   GPU-ported `stripe` removal on the device — one PCIe upload, one download,
///   no host normalize round-trip or transpose copy. The reconstructor decides
///   whether it can run the requested `stripe` on the GPU; when it cannot (or
///   has no device path at all) it returns `Ok(None)` and the whole chunk falls
///   through to the host route.
/// - **Host** (CPU/wgpu, or a stripe method with no GPU port): normalize →
///   transpose → stripe removal on the host, then `reconstruct_chunk` /
///   stateless `recon`.
#[allow(clippy::too_many_arguments)]
fn chunk_to_volume(
    slot: &mut Option<Option<Box<dyn crate::backend::StreamingAnalytic>>>,
    backend: &dyn crate::backend::Backend,
    ds: Dataset<f32>,
    geom: &Geometry,
    chunk_geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    prep: &PrepOptions,
) -> Result<Volume<f32>> {
    // Build the reusable reconstructor on the first chunk, sized to that chunk's
    // (largest) projection dims.
    ensure_slot(
        slot,
        backend,
        algorithm,
        params,
        geom,
        ds.data.n_cols(),
        ds.data.n_rows(),
    )?;
    // Device-resident fast path. The reconstructor applies any GPU-ported stripe
    // method itself and returns `Ok(None)` when it cannot (no device path, or the
    // stripe method has no GPU port), in which case the whole chunk falls through
    // to the host route below.
    if let Some(Some(s)) = slot.as_mut() {
        let std = ds.data.array.as_standard_layout();
        let raw = std.as_slice().ok_or_else(|| {
            crate::error::Error::InvalidParam("non-contiguous projection chunk".into())
        })?;
        let dims = ds.data.array.dim();
        if let Some(vol) = s.reconstruct_chunk_raw(
            raw,
            dims,
            ds.flat.as_ref(),
            ds.dark.as_ref(),
            chunk_geom,
            prep.stripe,
        )? {
            return Ok(vol);
        }
    }
    host_path(slot, backend, ds, geom, chunk_geom, algorithm, params, prep)
}

/// Reconstruct one **raw** projection chunk delivered in a host staging buffer
/// (pinned for CUDA): the device-resident path uploads the staging slice straight
/// (a direct DMA when pinned), with no owned `ndarray` materialization. Only the
/// rare host fallback (a stripe method with no GPU port) copies the projections
/// into an owned array. Mirrors [`chunk_to_volume`]'s two routes.
#[allow(clippy::too_many_arguments)]
fn raw_chunk_to_volume(
    slot: &mut Option<Option<Box<dyn crate::backend::StreamingAnalytic>>>,
    backend: &dyn crate::backend::Backend,
    data: &dyn crate::backend::HostBuffer,
    dims: (usize, usize, usize),
    flat: Option<Frames<f32>>,
    dark: Option<Frames<f32>>,
    theta: Vec<f32>,
    geom: &Geometry,
    chunk_geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    prep: &PrepOptions,
) -> Result<Volume<f32>> {
    let (nproj, rows, nx) = dims;
    // The staging buffer is sized to the largest chunk; this (possibly smaller,
    // trailing) chunk occupies only its `need`-element prefix.
    let need = nproj * rows * nx;
    let raw = &data.as_slice()[..need];
    // Reconstructor sized to this (largest, first) chunk: ncols = nx, nz = rows.
    ensure_slot(slot, backend, algorithm, params, geom, nx, rows)?;
    if let Some(Some(s)) = slot.as_mut() {
        if let Some(vol) = s.reconstruct_chunk_raw(
            raw,
            dims,
            flat.as_ref(),
            dark.as_ref(),
            chunk_geom,
            prep.stripe,
        )? {
            return Ok(vol);
        }
    }
    // Host fallback: materialize the staging projections into an owned array once,
    // then run the identical host route as `chunk_to_volume`.
    let array = Array3::from_shape_vec((nproj, rows, nx), raw.to_vec()).map_err(|e| {
        crate::error::Error::ShapeMismatch {
            expected: format!("[{nproj}, {rows}, {nx}]"),
            found: e.to_string(),
        }
    })?;
    let ds = Dataset {
        data: Tomo::new(array, Layout::Projection),
        flat,
        dark,
        theta,
    };
    host_path(slot, backend, ds, geom, chunk_geom, algorithm, params, prep)
}

/// Build the reusable streaming reconstructor into `slot` on the first chunk
/// (sized to `ncols`/`nz`), reused for every later chunk. `Some(None)` records
/// that the backend declined for this algorithm (host fallback).
fn ensure_slot(
    slot: &mut Option<Option<Box<dyn crate::backend::StreamingAnalytic>>>,
    backend: &dyn crate::backend::Backend,
    algorithm: Algorithm,
    params: &ReconParams,
    geom: &Geometry,
    ncols: usize,
    nz: usize,
) -> Result<()> {
    if slot.is_none() {
        let built = match backend.analytic_reconstruct() {
            Some(ar) => ar.streaming(algorithm, params, geom, ncols, nz)?,
            None => None,
        };
        *slot = Some(built);
    }
    Ok(())
}

/// Host reconstruction route shared by [`chunk_to_volume`] and
/// [`raw_chunk_to_volume`]: normalize → transpose (to a C-contiguous sinogram so
/// the back-projector can take a flat slice) → stripe removal → reconstruct from
/// the prepared sinogram (`slot` is already built; `recon_chunk_reusing` just
/// dispatches).
#[allow(clippy::too_many_arguments)]
fn host_path(
    slot: &mut Option<Option<Box<dyn crate::backend::StreamingAnalytic>>>,
    backend: &dyn crate::backend::Backend,
    mut ds: Dataset<f32>,
    geom: &Geometry,
    chunk_geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    prep: &PrepOptions,
) -> Result<Volume<f32>> {
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
