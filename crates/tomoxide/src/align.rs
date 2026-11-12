//! Laminography alignment on one prepped projection stack: the rings estimate,
//! the centre probe sweep and the tilt scan, run in the order
//! `docs/LAMINOGRAPHY_ALIGNMENT.md` gives. `tomoxide align` and the GUI both
//! drive these; judging a sweep ([`judge_sweep`](crate::recon::center::judge_sweep))
//! and reporting it stay with them.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use ndarray::{Array3, Axis};
use rayon::prelude::*;

use crate::backend::Backend;
use crate::cuda::{BandFrames, LaminoBandFocus, LaminoTileFn};
use crate::data::{Layout, Tomo};
use crate::error::{Error, Result};
use crate::geometry::{Angles, Beam, Center, Detector, Geometry};
use crate::params::{Algorithm, FilterName, ReconParams};
use crate::pipeline::{CancelToken, PrepOptions};
use crate::recon::center::{RingCenter, SampleBand, TiltCandidates, TiltFocus, TiltViews};
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
    /// The output slice `frames` shows.
    pub slice: usize,
    /// The slices scored, inclusive: the band, else `slice` alone.
    pub scored: (usize, usize),
    /// `[candidate, ny, nx]`: `slice` as each candidate centre reconstructs it.
    pub frames: Array3<f32>,
    /// Each candidate's mean [`slice_focus`](crate::recon::center::slice_focus)
    /// over `scored`.
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

    /// This stack with every projection moved by
    /// [`move_frames`](crate::prep::alignment::move_frames): `dy`/`dx` per
    /// projection, one `roll_deg` for all. The move is made on the
    /// transmission `exp(−p)` and logged again, as moving the raw frames
    /// before prep would.
    pub fn moved(
        &self,
        backend: &dyn Backend,
        dy: &[f32],
        dx: &[f32],
        roll_deg: f32,
    ) -> Result<PreppedStack> {
        let mut proj = self.data.to_layout(Layout::Projection);
        proj.array
            .as_slice_mut()
            .expect("to_layout is standard")
            .par_iter_mut()
            .for_each(|p| *p = (-*p).exp());
        crate::prep::alignment::move_frames(&mut proj, dy, dx, roll_deg)?;
        crate::prep::minus_log(&mut proj, backend)?;
        Ok(PreppedStack {
            path: self.path.clone(),
            data: proj,
            theta: self.theta.clone(),
            nproj: self.nproj,
            nz: self.nz,
            nx: self.nx,
        })
    }

    /// This stack binned `f`×`f` on the detector and cropped to whole blocks,
    /// so column `c` of the result is columns `c·f .. c·f + f`. The block
    /// mean is taken of the transmission `exp(−p)` and logged again, as
    /// binning the normalized frames would.
    pub fn binned(&self, backend: &dyn Backend, f: usize) -> Result<PreppedStack> {
        if f == 0 || f > self.nz || f > self.nx {
            return Err(Error::InvalidParam(format!(
                "bin {f} for {}×{} frames",
                self.nz, self.nx
            )));
        }
        let (nz, nx) = (self.nz / f, self.nx / f);
        let proj = self.data.as_layout(Layout::Projection);
        let src = proj.array.as_standard_layout();
        let src = src.as_slice().expect("as_standard_layout is standard");
        let mut out = vec![0.0f32; self.nproj * nz * nx];
        let inv = 1.0 / (f * f) as f32;
        out.par_chunks_mut(nz * nx)
            .zip(src.par_chunks(self.nz * self.nx))
            .for_each(|(o, p)| {
                for (r, row) in o.chunks_mut(nx).enumerate() {
                    for (c, v) in row.iter_mut().enumerate() {
                        let block = (r * f..(r + 1) * f)
                            .flat_map(|y| &p[y * self.nx + c * f..y * self.nx + (c + 1) * f]);
                        *v = block.map(|q| (-q).exp()).sum::<f32>() * inv;
                    }
                }
            });
        let array = Array3::from_shape_vec((self.nproj, nz, nx), out).expect("sized above");
        let mut data = Tomo::new(array, Layout::Projection);
        crate::prep::minus_log(&mut data, backend)?;
        Ok(PreppedStack {
            path: self.path.clone(),
            data,
            theta: self.theta.clone(),
            nproj: self.nproj,
            nz,
            nx,
        })
    }

    /// This stack held in `frames` to score slices `z.0 ..= z.1` of its
    /// linerec reconstruction at `tilt_deg`, filtered with `filter`, for
    /// per-projection shifts and a centre. CUDA-only.
    pub fn band_focus(
        &self,
        filter: FilterName,
        tilt_deg: f32,
        z: (usize, usize),
        frames: BandFrames,
    ) -> Result<LaminoBandFocus> {
        let proj = self.data.as_layout(Layout::Projection);
        LaminoBandFocus::new(proj.array.view(), &self.theta, filter, tilt_deg, z, frames)
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

    /// The output slice a centre sweep shows by default, and scores without a band. The centre response
    /// lives where the sample is, and which plane holds the sample is a property
    /// of the scan, not of the geometry — so a stated sample band (`band`, read
    /// off a reconstruction at `tilt_deg`) supplies its midpoint. Without one the
    /// only neutral default is the middle of the volume, which is not the
    /// detector's middle row under a tilt. It is the layer
    /// [`tilt_scan`](Self::tilt_scan) shows, so a sweep and a scan agree on it.
    pub fn default_slice(&self, tilt_deg: Option<f32>, band: Option<(usize, usize)>) -> usize {
        crate::recon::center::middle_slice(band, self.depth(tilt_deg))
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
    /// sweep, and score each candidate by its mean slice focus over `band`
    /// (inclusive slices of the volume at `tilt_deg`), else on `slice` alone.
    /// Each slice peaks at its own centre by what it holds (on a quartz scan,
    /// eleven slices of one band from 1045 to 1063 px, their mean at 1055), so
    /// one slice answers for its own content. Under a tilt the band is scored
    /// by [`band_focus`](Self::band_focus), a linerec back-projection of it
    /// per candidate; in parallel beam by a probe sweep per slice. A fired
    /// `cancel` stops it at the next probe or candidate with
    /// [`Error::Cancelled`]. `on_scored(done, of)` runs after each candidate
    /// of a tilted band, or each slice of a parallel one, is scored; a sweep
    /// without a band is one probe and never calls it. CUDA-only.
    pub fn center_sweep(
        &self,
        tilt_deg: Option<f32>,
        slice: usize,
        band: Option<(usize, usize)>,
        centers: &[f32],
        cancel: Option<&CancelToken>,
        on_scored: &mut dyn FnMut(usize, usize),
    ) -> Result<CenterSweep> {
        // The probe anchors each candidate on its own nominal, so the
        // geometry's centre is not read.
        let geom = self.geometry(self.nx as f32 / 2.0, tilt_deg);
        let params = ReconParams::default();
        let probe = |z: usize| {
            crate::cuda::center_probe_sweep(&self.data, &geom, &params, centers, z as i32, cancel)
        };
        let frames = probe(slice)?;
        let focus = match (band, tilt_deg) {
            (None, _) => crate::recon::center::focus_by_z(frames.view()),
            (Some(z), Some(tilt)) => {
                let mut scorer =
                    self.band_focus(params.filter_name, tilt, z, BandFrames::Device)?;
                let still = vec![0.0; self.nproj];
                centers
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| {
                        crate::pipeline::check_cancel(cancel)?;
                        let profile = scorer.profile(&still, &still, c)?;
                        on_scored(i + 1, centers.len());
                        Ok(profile.iter().sum::<f64>() / profile.len() as f64)
                    })
                    .collect::<Result<_>>()?
            }
            (Some((lo, hi)), None) => {
                if lo > hi {
                    return Err(Error::Backend(format!(
                        "centre sweep: band {lo}..={hi} holds no slices"
                    )));
                }
                let mut sum = vec![0.0; centers.len()];
                for z in lo..=hi {
                    let scored = crate::recon::center::focus_by_z(probe(z)?.view());
                    sum.iter_mut().zip(scored).for_each(|(s, f)| *s += f);
                    on_scored(z - lo + 1, hi - lo + 1);
                }
                sum.iter().map(|s| s / (hi - lo + 1) as f64).collect()
            }
        };
        Ok(CenterSweep {
            slice,
            scored: band.unwrap_or((slice, slice)),
            frames,
            focus,
        })
    }

    /// Fully reconstruct at every tilt and score each by its mean slice focus; see
    /// [`lamino_tilt_scan`](crate::recon::center::lamino_tilt_scan), which this
    /// runs with Fourierrec and default parameters. CUDA-only.
    pub fn tilt_scan(
        &self,
        center: f32,
        tilts_deg: &[f32],
        band: Option<SampleBand>,
        cancel: Option<&CancelToken>,
        on_tilt: &mut dyn FnMut(&TiltFocus, TiltViews<'_>) -> Result<()>,
    ) -> Result<Vec<TiltFocus>> {
        // Default parameters are the tuned analytic path (parzen filter) the
        // laminography goldens reconstruct with. `lamino_rh` left `None` gives
        // every tilt its own natural depth, so the scored range is its own
        // volume's.
        crate::recon::center::lamino_tilt_scan(
            &self.data,
            &self.geometry(center, Some(0.0)), // beam ignored — `tilts_deg` supplies it
            Algorithm::Fourierrec,
            &ReconParams::default(),
            TiltCandidates { tilts_deg, band },
            cancel,
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

    /// No move gives back the stack: the exp/log around the resampling is
    /// the identity on prepped data.
    #[test]
    fn moved_by_nothing_is_the_same_stack() {
        let cpu = crate::CpuBackend::new();
        let st = PreppedStack::load(&cpu, &fixture()).unwrap();
        let (nproj, _, _) = st.shape();
        let zero = vec![0.0; nproj];
        let m = st.moved(&cpu, &zero, &zero, 0.0).unwrap();
        assert_eq!(m.shape(), st.shape());
        assert_eq!(m.theta, st.theta);
        let a = st.data.as_layout(Layout::Projection);
        let b = m.data.as_layout(Layout::Projection);
        let peak = a.array.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (x, y) in a.array.iter().zip(b.array.iter()) {
            assert!((x - y).abs() <= 1e-4 * peak.max(1.0), "{x} vs {y}");
        }
    }

    /// Each binned pixel is −log of its block's mean transmission, and the
    /// rows and columns past the last whole block are dropped.
    #[test]
    fn binned_is_the_logged_block_mean_of_the_transmission() {
        let cpu = crate::CpuBackend::new();
        let st = PreppedStack::load(&cpu, &fixture()).unwrap();
        let (nproj, nz, nx) = st.shape();
        let f = 3;
        let b = st.binned(&cpu, f).unwrap();
        assert_eq!(b.shape(), (nproj, nz / f, nx / f));
        assert_eq!(b.theta, st.theta);
        let a = st.data.as_layout(Layout::Projection);
        let out = b.data.as_layout(Layout::Projection);
        for &(i, r, c) in &[
            (0, 0, 0),
            (nproj - 1, nz / f - 1, nx / f - 1),
            (nproj / 2, 1, 2),
        ] {
            let mut s = 0.0f64;
            for y in r * f..(r + 1) * f {
                for x in c * f..(c + 1) * f {
                    s += f64::from(-a.array[[i, y, x]]).exp();
                }
            }
            let want = -(s / (f * f) as f64).ln();
            let got = f64::from(out.array[[i, r, c]]);
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "{got} vs {want}"
            );
        }
        assert!(st.binned(&cpu, 0).is_err());
    }
}
