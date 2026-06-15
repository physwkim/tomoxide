//! High-level reconstruction pipelines.
//!
//! [`reconstruct`] is the in-memory ("full") path: preprocess → reconstruct.
//! [`ReconSteps`] is the chunked/streaming path (port of tomocupy
//! `rec_steps.py`); its overlapped read→H2D→compute→D2H→write loop lands in
//! milestone M5 (see `docs/ARCHITECTURE.md` §5).

use tomoxide_core::data::{Dataset, Layout, Volume};
use tomoxide_core::error::{Error, Result};
use tomoxide_core::geometry::Geometry;
use tomoxide_core::params::{Algorithm, PhaseMethod, ReconParams, StripeMethod};

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
/// [`tomoxide_recon::recon`]; it surfaces `NotImplemented` from the first
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
    tomoxide_prep::normalize_dataset(&mut ds, backend)?;

    // 2. Optional projection-domain corrections.
    tomoxide_prep::retrieve_phase(&mut ds.data, prep.phase)?;

    // 3. To sinogram order, then stripe removal, then reconstruct.
    let mut sino = ds.data.to_layout(Layout::Sinogram);
    tomoxide_prep::remove_stripe(&mut sino, prep.stripe)?;
    tomoxide_recon::recon(&sino, geom, algorithm, params, backend)
}

/// Chunked/streaming reconstruction driver (port target: tomocupy
/// `rec_steps.py::GPURecSteps`). Scaffold only.
pub struct ReconSteps;

impl ReconSteps {
    /// Run the streaming pipeline over a reader/writer pair.
    ///
    /// The overlapped multi-stream loop (sinogram/projection chunking, double
    /// buffering, read/write thread pools) is milestone M5.
    pub fn run(&self) -> Result<()> {
        Err(Error::todo(
            "pipeline::ReconSteps::run",
            "tomocupy rec_steps.py:116 recon_steps_all",
        ))
    }
}
