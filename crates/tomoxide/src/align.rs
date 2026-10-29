//! Laminography alignment on one prepped projection stack: the rings estimate,
//! the centre probe sweep and the tilt scan, run in the order
//! `docs/LAMINOGRAPHY_ALIGNMENT.md` gives. `tomoxide align` and the GUI both
//! drive these; judging a sweep ([`judge_sweep`](crate::recon::center::judge_sweep))
//! and reporting it stay with them.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use ndarray::{Array3, Axis};

use crate::backend::Backend;
use crate::cuda::LaminoTileFn;
use crate::data::{Layout, Tomo};
use crate::error::{Error, Result};
use crate::geometry::{Angles, Beam, Center, Detector, Geometry};
use crate::params::{Algorithm, ReconParams};
use crate::pipeline::PrepOptions;
use crate::recon::center::{RingCenter, SampleBand, TiltFocus, TiltViews};
use crate::recon::tilt_flow::TiltFlow;

/// The whole flat/dark-corrected, minus-log projection stack, held in host RAM.
///
/// Every alignment step consumes all of it — the rings average the projections,
/// and both sweeps back-project them — so there is no band to read lazily.
/// Loading it is the expensive part (`nproj·nz·nx·4` bytes, and a full pass of
/// prep), and the steps run one after another on the same data, so a caller
/// loads it once and keeps it.
pub struct PreppedStack {
    path: PathBuf,
    data: Tomo<f32>,
    theta: Vec<f32>,
    nproj: usize,
    nz: usize,
    nx: usize,
}

/// A finished centre probe sweep.
pub struct CenterSweep {
    /// The output slice every candidate was scored on.
    pub slice: usize,
    /// `[candidate, ny, nx]`: the slice as each candidate centre reconstructs it.
    pub frames: Array3<f32>,
    /// Each candidate's [`slice_focus`](crate::recon::center::slice_focus).
    pub focus: Vec<f64>,
}

impl PreppedStack {
    /// Read the DXchange file at `path` and prep it (flat/dark, minus-log).
    pub fn load(backend: &dyn Backend, path: &Path) -> Result<Self> {
        let mut reader = crate::io::open_dxchange(&path.to_string_lossy())?;
        let theta = reader.read_theta()?;
        let mut ds = reader.read_all()?;
        crate::prep::normalize_dataset(&mut ds, backend)?;
        let (nproj, nz, nx) = ds.data.as_layout(Layout::Projection).array.dim();
        Ok(PreppedStack {
            path: path.to_path_buf(),
            data: ds.data,
            theta,
            nproj,
            nz,
            nx,
        })
    }

    /// The file this stack was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The prepped projections.
    pub fn data(&self) -> &Tomo<f32> {
        &self.data
    }

    /// `(nproj, nz, nx)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.nproj, self.nz, self.nx)
    }

    /// Host bytes held.
    pub fn bytes(&self) -> usize {
        self.data.array.len() * std::mem::size_of::<f32>()
    }

    /// The stack of only the projections at `indices`, in their order: a
    /// half (even/odd) for a resolution check, or a subset of the turn.
    pub fn select(&self, indices: &[usize]) -> Result<PreppedStack> {
        if let Some(&i) = indices.iter().find(|&&i| i >= self.nproj) {
            return Err(Error::InvalidParam(format!(
                "projection {i} out of range (nproj {})",
                self.nproj
            )));
        }
        let proj = self.data.as_layout(Layout::Projection);
        Ok(PreppedStack {
            path: self.path.clone(),
            data: Tomo::new(proj.array.select(Axis(0), indices), Layout::Projection),
            theta: indices.iter().map(|&i| self.theta[i]).collect(),
            nproj: indices.len(),
            nz: self.nz,
            nx: self.nx,
        })
    }

    /// Laminography reconstruction of this stack at `geom` (from
    /// [`geometry`](Self::geometry)), streaming its rh-tiles to `on_tile` as
    /// [`reconstruct_lamino_streaming`](crate::reconstruct_lamino_streaming)
    /// does for a file, with `prep` applied after the normalization this stack
    /// already holds. Returns `(rh, n, n)`. CUDA-only.
    pub fn reconstruct(
        &self,
        backend: &dyn Backend,
        geom: &Geometry,
        algorithm: Algorithm,
        params: &ReconParams,
        prep: &PrepOptions,
        on_tile: &mut LaminoTileFn,
    ) -> Result<(usize, usize, usize)> {
        crate::pipeline::lamino_streaming_normalized(
            Cow::Borrowed(&self.data),
            geom,
            algorithm,
            params,
            prep,
            backend,
            on_tile,
        )
    }

    /// Geometry at `center`/`tilt_deg`. `tilt_deg = None` is parallel beam.
    pub fn geometry(&self, center: f32, tilt_deg: Option<f32>) -> Geometry {
        Geometry {
            angles: Angles(self.theta.clone()),
            center: Center::Scalar(center),
            beam: match tilt_deg {
                Some(d) => Beam::laminography(d),
                None => Beam::Parallel,
            },
            detector: Detector {
                width: self.nx,
                height: self.nz,
                pixel_size: 1.0,
            },
        }
    }

    /// Rows of the output volume at `tilt_deg`: `nz` for parallel beam, `rh` for
    /// laminography, whose depth exceeds the detector's because the tilt
    /// stretches it.
    pub fn depth(&self, tilt_deg: Option<f32>) -> usize {
        match tilt_deg {
            Some(d) => crate::cuda::lamino_recon_height(self.nz, d),
            None => self.nz,
        }
    }

    /// The output slice a centre sweep scores by default. The centre response
    /// lives where the sample is, and which plane holds the sample is a property
    /// of the scan, not of the geometry — so a stated sample band (`band`, read
    /// off a reconstruction at `tilt_deg`) supplies its midpoint. Without one the
    /// only neutral default is the middle of the volume, which is not the
    /// detector's middle row under a tilt.
    pub fn default_slice(&self, tilt_deg: Option<f32>, band: Option<(usize, usize)>) -> usize {
        match band {
            Some((lo, hi)) => (lo + hi) / 2,
            None => self.depth(tilt_deg) / 2,
        }
    }

    /// The axis read off the mean of every `step`th projection.
    pub fn rings(&self, backend: &dyn Backend, step: usize) -> Result<RingCenter> {
        crate::recon::center::find_center_rings(&self.data, backend, step)
    }

    /// The tilt read off the vertical flow between neighbouring projections,
    /// before any reconstruction; see
    /// [`estimate_tilt_flow`](crate::recon::tilt_flow::estimate_tilt_flow).
    pub fn tilt_flow(&self, backend: &dyn Backend) -> Result<TiltFlow> {
        crate::recon::tilt_flow::estimate_tilt_flow(&self.data, &self.theta, backend)
    }

    /// Reconstruct output slice `slice` at every candidate centre in one probe
    /// sweep and score each. CUDA-only.
    pub fn center_sweep(
        &self,
        tilt_deg: Option<f32>,
        slice: usize,
        centers: &[f32],
    ) -> Result<CenterSweep> {
        // The probe anchors each candidate on its own nominal, so the
        // geometry's centre is not read.
        let frames = crate::cuda::center_probe_sweep(
            &self.data,
            &self.geometry(self.nx as f32 / 2.0, tilt_deg),
            &ReconParams::default(),
            centers,
            slice as i32,
        )?;
        let focus = crate::recon::center::focus_by_z(frames.view());
        Ok(CenterSweep {
            slice,
            frames,
            focus,
        })
    }

    /// Fully reconstruct at every tilt and score each by its best slice; see
    /// [`lamino_tilt_scan`](crate::recon::center::lamino_tilt_scan), which this
    /// runs with Fourierrec and default parameters. CUDA-only.
    pub fn tilt_scan(
        &self,
        center: f32,
        tilts_deg: &[f32],
        band: Option<SampleBand>,
        on_tilt: &mut dyn FnMut(&TiltFocus, TiltViews<'_>) -> Result<()>,
    ) -> Result<Vec<TiltFocus>> {
        // Default parameters are the tuned analytic path (parzen filter) the
        // laminography goldens reconstruct with. `lamino_rh` left `None` gives
        // every tilt its own natural depth, which is the point of scoring the
        // max over all of it.
        crate::recon::center::lamino_tilt_scan(
            &self.data,
            &self.geometry(center, Some(0.0)), // beam ignored — `tilts_deg` supplies it
            Algorithm::Fourierrec,
            &ReconParams::default(),
            tilts_deg,
            band,
            on_tilt,
        )
    }
}

/// `center ± half_width` in steps of `step`, always including `center` itself:
/// a sweep refines the estimate it is centred on, and a grid that skipped it
/// could only ever move away. A non-positive step or negative width leaves only
/// `center`.
pub fn grid(center: f32, half_width: f32, step: f32) -> Vec<f32> {
    if step <= 0.0 || half_width < 0.0 {
        return vec![center];
    }
    let n = (half_width / step).floor() as i32;
    (-n..=n).map(|k| center + k as f32 * step).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/streaming_dxchange.h5"
        ))
    }

    #[test]
    fn grid_is_centred_and_symmetric() {
        assert_eq!(
            grid(396.0, 1.0, 0.5),
            vec![395.0, 395.5, 396.0, 396.5, 397.0]
        );
        // A step wider than the half-width leaves only the centre itself.
        assert_eq!(grid(10.0, 0.2, 0.5), vec![10.0]);
        assert_eq!(grid(396.0, 0.0, 0.25), vec![396.0]);
        assert_eq!(grid(396.0, 8.0, 0.0), vec![396.0]);
    }

    /// `rh/2` and `nz/2` are the same row only at zero tilt: the tilt stretches
    /// the reconstruction deeper than the detector is tall, and the sweep's
    /// default slice indexes the volume, not the detector. A stated band wins
    /// over both.
    #[test]
    fn default_slice_is_the_band_midpoint_else_the_middle_of_the_volume() {
        let st = PreppedStack::load(&crate::CpuBackend::new(), &fixture()).unwrap();
        let (_nproj, nz, _nx) = st.shape();
        assert_eq!(st.default_slice(None, None), nz / 2);
        let tilted = st.default_slice(Some(44.0), None);
        assert_eq!(tilted, crate::cuda::lamino_recon_height(nz, 44.0) / 2);
        assert!(
            tilted > nz / 2,
            "a 44° tilt must push the middle of the volume past the detector's middle row: \
             got {tilted} for nz {nz}"
        );
        assert_eq!(st.default_slice(Some(44.0), Some((2, 7))), 4);
    }
}
