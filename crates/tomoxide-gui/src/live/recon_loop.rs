//! Per-loop Z-slice reconstruction from the ring buffer (docs/GUI.md §2.6).
//!
//! Snapshots the live parameters, assembles the selected slice's sinogram from
//! the [`ProjRing`], removes stripes, and reconstructs one horizontal slice with
//! the analytic backend. tomoxide reconstructs Z (horizontal) slices cheaply
//! (analytic recon is per-slice independent); X/Y ortho panes need dedicated
//! kernels (docs/GUI.md §6 #7) and are out of scope for this pass.

use tomoxide::backend::Backend;
use tomoxide::{
    Algorithm, Angles, Center, Error, FilterName, Geometry, Layout, ReconParams, StripeMethod, Tomo,
};

use super::ring::ProjRing;

/// Live reconstruction parameters, re-read every loop (tomostream semantics — a
/// center tweak or filter change applies on the next iteration).
#[derive(Clone, Debug, PartialEq)]
pub struct LiveReconParams {
    /// Detector row (horizontal slice) to reconstruct.
    pub slice: usize,
    /// Rotation-axis column; `None` ⇒ detector midline.
    pub center: Option<f32>,
    pub filter: FilterName,
    pub stripe: StripeMethod,
    /// Analytic algorithm (FBP/Gridrec/Fourierrec/Lprec/Linerec). Iterative
    /// methods are not used live.
    pub algorithm: Algorithm,
    /// Truncated-projection support extension (`ReconParams::ext_pad`).
    pub ext_pad: bool,
    /// Ring-buffer capacity (~180° of projections).
    pub capacity: usize,
}

impl Default for LiveReconParams {
    fn default() -> Self {
        Self {
            slice: 0,
            center: None,
            filter: FilterName::Parzen,
            stripe: StripeMethod::None,
            algorithm: Algorithm::Fbp,
            ext_pad: false,
            capacity: 180,
        }
    }
}

/// Reconstruct the selected slice from the ring. Returns
/// `(ny, nx, pixels, nproj)` — the reconstructed image and the projection count
/// it was built from.
pub fn reconstruct_slice(
    ring: &ProjRing,
    params: &LiveReconParams,
    backend: &dyn Backend,
) -> tomoxide::Result<(usize, usize, Vec<f32>, usize)> {
    let sino = ring.sinogram(params.slice).ok_or_else(|| {
        Error::InvalidParam(format!(
            "no projections buffered for slice {} (ring empty or slice out of range)",
            params.slice
        ))
    })?;
    let nproj = sino.dim().1;
    let nx = sino.dim().2;

    let mut one = Tomo::new(sino, Layout::Sinogram);
    tomoxide::prep::remove_stripe(&mut one, params.stripe)?;

    // The ring stores angles in degrees (tomoScanStream/DXchange convention);
    // `Angles` is radians. Convert at this boundary, mirroring
    // `H5DxchangeReader::read_theta` (which owns the deg→rad conversion for the
    // offline path) — otherwise a 0–180° sweep is reconstructed as 0–180 rad.
    let thetas: Vec<f32> = ring.thetas().iter().map(|t| t.to_radians()).collect();
    let mut geom = Geometry::parallel(Angles(thetas), nx, 1, 1.0);
    if let Some(c) = params.center {
        geom.center = Center::Scalar(c);
    }

    let rp = ReconParams {
        num_gridx: Some(nx),
        filter_name: params.filter,
        ext_pad: params.ext_pad,
        ..Default::default()
    };
    let vol = tomoxide::recon::recon(&one, &geom, params.algorithm, &rp, backend)?;
    let (_nz, ny, nxo) = vol.dims();
    Ok((ny, nxo, vol.array.iter().copied().collect(), nproj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// A CPU FBP of a small ring buffer produces a finite, correctly-sized
    /// image — the end-to-end ring → prep → recon path.
    #[test]
    fn reconstructs_a_finite_slice_from_the_ring() {
        let backend = tomoxide::CpuBackend;
        let nx = 32;
        let nproj = 60;
        let mut ring = ProjRing::new(nproj);
        ring.set_geometry(1, nx);
        // A centered absorbing bar: transmission dips in the middle columns.
        for p in 0..nproj {
            let _theta = 180.0 * p as f64 / nproj as f64;
            let mut frame = Array2::<f32>::from_elem((1, nx), 1.0);
            for x in (nx / 2 - 3)..(nx / 2 + 3) {
                frame[[0, x]] = 0.2;
            }
            ring.push(180.0 * p as f64 / nproj as f64, frame);
        }
        let params = LiveReconParams {
            capacity: nproj,
            ..Default::default()
        };
        let (ny, nxo, data, used) = reconstruct_slice(&ring, &params, &backend).unwrap();
        assert_eq!((ny, nxo), (nx, nx));
        assert_eq!(used, nproj);
        assert!(data.iter().all(|v| v.is_finite()));
        assert!(data.iter().any(|&v| v.abs() > 0.0));
    }

    #[test]
    fn empty_ring_errors() {
        let backend = tomoxide::CpuBackend;
        let ring = ProjRing::new(4);
        let err = reconstruct_slice(&ring, &LiveReconParams::default(), &backend);
        assert!(matches!(err, Err(Error::InvalidParam(_))));
    }

    /// Regression: the ring stores angles in degrees (tomoScanStream/DXchange
    /// convention) but `Angles` is radians, so `reconstruct_slice` must convert
    /// at the boundary — exactly as `H5DxchangeReader::read_theta` does. Feeding
    /// degrees straight into `Angles` scrambles the sweep (0–180° read as
    /// 0–180 rad ≈ 28 turns) and yields a different, wrong image. This pins the
    /// output to a radian-converted reference recon; it fails if the conversion
    /// is dropped.
    #[test]
    fn ring_angles_are_reconstructed_in_radians() {
        use tomoxide::{Layout, ReconParams, Tomo, recon};
        let backend = tomoxide::CpuBackend;
        let nx = 32;
        let nproj = 45;
        let mut ring = ProjRing::new(nproj);
        ring.set_geometry(1, nx);
        for p in 0..nproj {
            let mut frame = Array2::<f32>::from_elem((1, nx), 1.0);
            for x in (nx / 2 - 3)..(nx / 2 + 3) {
                frame[[0, x]] = 0.2;
            }
            ring.push(180.0 * p as f64 / nproj as f64, frame);
        }
        let params = LiveReconParams {
            capacity: nproj,
            ..Default::default()
        };
        let (_ny, _nx, got, _used) = reconstruct_slice(&ring, &params, &backend).unwrap();

        // Reference: identical FBP but with the ring's degrees converted to
        // radians (the fix's boundary conversion) done explicitly here.
        let sino = ring.sinogram(0).unwrap();
        let one = Tomo::new(sino, Layout::Sinogram);
        let thetas: Vec<f32> = ring.thetas().iter().map(|t| t.to_radians()).collect();
        let geom = Geometry::parallel(Angles(thetas), nx, 1, 1.0);
        let rp = ReconParams {
            num_gridx: Some(nx),
            filter_name: params.filter,
            ext_pad: params.ext_pad,
            ..Default::default()
        };
        let vol = recon::recon(&one, &geom, params.algorithm, &rp, &backend).unwrap();
        let want: Vec<f32> = vol.array.iter().copied().collect();

        assert_eq!(got.len(), want.len());
        assert!(
            got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-4),
            "reconstruct_slice must build Angles from radians, matching read_theta"
        );
    }
}
