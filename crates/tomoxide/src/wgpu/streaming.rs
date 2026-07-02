//! Device-resident streaming reconstruction for the wgpu backend.
//!
//! [`WgpuBackend`] already fuses the analytic filter→recon chain on-device
//! ([`AnalyticReconstruct`]); this adds the *streaming* layer so the CLI's
//! overlapped read‖compute‖write pipeline
//! ([`ReconSteps::run_streaming_pipelined`](crate::pipeline::ReconSteps::run_streaming_pipelined))
//! can drive wgpu, instead of falling back to the whole-volume path.
//!
//! The streaming-ceiling measurement showed the wgpu whole-volume wall is 60-82%
//! HOST cost: the projection-domain minus-log run through the wgpu Elementwise
//! GPU round-trip (uploads the whole projection volume just to run one elementwise
//! kernel, then downloads it) plus the full-volume projection→sinogram host
//! transpose. [`WgpuFbpStream::reconstruct_chunk_raw`] removes both from the
//! critical path: it normalizes each chunk on the **CPU** (the parallel,
//! memory-bound `minus_log` — no GPU round-trip), transposes just that chunk, and
//! runs the fused device recon so the chunk crosses PCIe once up and the volume
//! once down. Running under the pipeline conveyor, the per-chunk host transpose
//! overlaps disk read/write of the neighbouring chunks, and the bounded working
//! set lets fourierrec reconstruct volumes whose whole-volume oversampled grid
//! would exceed the wgpu max-buffer limit.

use crate::backend::{AnalyticReconstruct, FbpFilter, StreamingAnalytic};
use crate::data::{Dataset, Frames, Layout, Tomo, Volume};
use crate::error::Result;
use crate::geometry::Geometry;
use crate::params::{Algorithm, ReconParams, StripeMethod};

use super::kernels::WgpuLprecGrids;
use super::WgpuBackend;

impl WgpuBackend {
    /// Clone the device/queue handles (cheap `Arc` clones in wgpu) into a fresh
    /// backend with its own empty pipeline cache. A [`WgpuFbpStream`] owns the
    /// clone so it can live on the streaming compute thread while sharing the
    /// original's GPU device — buffers it creates are valid on that device. The
    /// clone recompiles each kernel once into its own cache on the first chunk,
    /// then reuses it for every later chunk.
    pub(crate) fn share(&self) -> WgpuBackend {
        WgpuBackend {
            device: self.device.clone(),
            queue: self.queue.clone(),
            pipelines: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }
}

/// Handle-reusing streaming reconstructor for the wgpu analytic methods
/// (fbp / linerec / fourierrec via the fused `AnalyticReconstruct` path, and
/// lprec via its own log-polar dispatch with grids cached across chunks), bound
/// to a fixed `(algorithm, params)`. Built once on the first chunk by
/// [`AnalyticReconstruct::streaming`] and driven on the streaming compute thread.
pub(crate) struct WgpuFbpStream {
    be: WgpuBackend,
    algorithm: Algorithm,
    params: ReconParams,
    /// lprec log-polar grids, built lazily on the first chunk and reused across
    /// all later chunks (nz-independent). `None` for the fused analytic methods.
    lprec_grids: Option<WgpuLprecGrids>,
}

impl WgpuFbpStream {
    pub(crate) fn new(be: WgpuBackend, algorithm: Algorithm, params: ReconParams) -> Self {
        WgpuFbpStream {
            be,
            algorithm,
            params,
            lprec_grids: None,
        }
    }

    /// Reconstruct one already-normalized, C-contiguous sinogram chunk on the
    /// device. fbp/linerec/fourierrec take the fused `AnalyticReconstruct` path;
    /// lprec FBP-filters the chunk, then runs the log-polar core reusing the
    /// grids cached on the first chunk (rebuilding them per chunk would repeat
    /// the ~175-230 ms host precompute — a regression vs the whole-volume path).
    fn recon_sino(&mut self, sino: &Tomo<f32>, geom: &Geometry) -> Result<Volume<f32>> {
        if self.algorithm != Algorithm::Lprec {
            return self
                .be
                .reconstruct(sino, geom, self.algorithm, &self.params);
        }
        let n = self.params.num_gridx.unwrap_or(sino.n_cols());
        if self.lprec_grids.is_none() {
            self.lprec_grids = Some(self.be.build_lprec_grids(geom, n)?);
        }
        // lprec reconstructs from the FBP-filtered sinogram (recon::recon filters
        // first, then calls LpRecReconstruct); mirror that here per chunk.
        let kernel = self
            .be
            .make_filter(self.params.filter_name, sino.n_cols())?;
        let mut filtered = sino.clone();
        self.be.apply(&mut filtered, &kernel, geom)?;
        let grids = self.lprec_grids.as_ref().expect("lprec grids built above");
        self.be.lprec_run(&filtered, grids)
    }
}

impl StreamingAnalytic for WgpuFbpStream {
    /// Reconstruct one already-normalized, already-transposed sinogram chunk on
    /// the device. Used by [`ReconSteps::run`](crate::pipeline::ReconSteps::run),
    /// which normalizes/transposes the whole dataset up front; the fused device
    /// path keeps the sinogram resident across filter and recon.
    fn reconstruct_chunk(&mut self, sino: &Tomo<f32>, geom: &Geometry) -> Result<Volume<f32>> {
        self.recon_sino(sino, geom)
    }

    /// Device fast path from the raw, un-normalized projection chunk
    /// `[nproj, nz, ncols]`: CPU normalize (dark/flat + parallel minus-log — a
    /// memory-bound op kept off the GPU to avoid the whole-chunk upload/download
    /// round-trip the wgpu `Elementwise` path pays), host transpose to a
    /// C-contiguous sinogram, then the fused device recon (sinogram crosses PCIe
    /// once up, volume once down).
    ///
    /// Returns `Ok(None)` — deferring the whole chunk to the host route — when a
    /// stripe method is requested, since wgpu has no on-device stripe removal
    /// here; the caller then runs `remove_stripe` on the CPU. `StripeMethod::None`
    /// (the common case) takes the device path.
    fn reconstruct_chunk_raw(
        &mut self,
        data: &[f32],
        dims: (usize, usize, usize),
        flat: Option<&Frames<f32>>,
        dark: Option<&Frames<f32>>,
        geom: &Geometry,
        stripe: StripeMethod,
    ) -> Result<Option<Volume<f32>>> {
        if stripe != StripeMethod::None {
            return Ok(None);
        }
        let (nproj, nz, ncols) = dims;
        let need = nproj * nz * ncols;
        let arr = ndarray::Array3::from_shape_vec((nproj, nz, ncols), data[..need].to_vec())
            .map_err(|e| crate::error::Error::ShapeMismatch {
                expected: format!("[{nproj}, {nz}, {ncols}]"),
                found: e.to_string(),
            })?;
        let mut ds = Dataset {
            data: Tomo::new(arr, Layout::Projection),
            flat: flat.cloned(),
            dark: dark.cloned(),
            theta: vec![0.0; nproj], // unused by normalize/recon (geom carries angles)
        };
        // Normalize on the CPU: minus-log (and dark/flat when present) is a
        // memory-bound elementwise op, so the parallel CPU path beats offloading
        // it to the GPU with a full-chunk round-trip.
        let cpu = crate::cpu::CpuBackend::new();
        crate::prep::normalize_dataset(&mut ds, &cpu)?;
        // Projection→sinogram transpose; force standard C-layout so the device
        // recon cores can take a flat slice (matches the whole-volume host path).
        let mut sino = ds.data.to_layout(Layout::Sinogram);
        sino.array = sino.array.as_standard_layout().to_owned();
        let vol = self.recon_sino(&sino, geom)?;
        Ok(Some(vol))
    }
}
