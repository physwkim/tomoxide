//! Pins the laminography volume z-centring convention across algorithms.
//!
//! The Fourier/USFFT laminography reconstruction (`recon::lamino`, tomocupy
//! `LamFourierRec`) has always centred the volume z on `rh/2` — the volume's
//! own midpoint. The linerec-family CUDA kernels (`backprojection_ker` and the
//! forward projector) centred it on `nz/2`, the detector-row midpoint, which
//! for a tilted axis (`rh > nz`) shifted every Linerec/Fbp/SIRT laminography
//! volume by `(rh − nz)/2` slices against the Fourierrec volume of the same
//! data — and against the try-probe `sz`, which is how a centre sweep's answer
//! came to depend on which slice it was pointed at. On the real pouch scan the
//! offset was 200 slices (rh 1424, nz 1024). These tests fail under that old
//! convention (the offset here is (48−32)/2 = 8 slices, the slab 2 thick) and
//! pin the unified one: a feature sits at the SAME volume z in every algorithm,
//! and the probe's `sz` IS that z.
//!
//! Own test binary (touches CUDA device state) per the suite convention.
#![cfg(feature = "cuda")]

use ndarray::{Array3, Axis};
use tomoxide::{
    cuda, recon, sim, Algorithm, Angles, Beam, Center, CudaBackend, Detector, Geometry,
    ReconParams, Volume,
};

/// Per-slice energy profile (sum of squares) over z.
fn z_profile(vol: &Array3<f32>) -> Vec<f64> {
    vol.axis_iter(Axis(0))
        .map(|s| s.iter().map(|&v| (v as f64) * (v as f64)).sum())
        .collect()
}

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

/// A 2-slice bright slab at volume z `z0`, forward-projected with the tilted
/// geometry, reconstructs with its z-energy peak back at `z0` — in BOTH the
/// linerec back-projection and the independent Fourier/USFFT algorithm, and the
/// two peaks agree. `rh − nz = 16`, so the pre-fix `nz/2` centring would land
/// the Linerec peak 8 slices away from both `z0` and the Fourierrec peak.
#[test]
fn cuda_lamino_linerec_and_fourierrec_agree_on_the_feature_z() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let (n, nproj, nz) = (64usize, 90usize, 32usize);
    let rh = 48usize;
    let z0 = 20usize; // off-centre (rh/2 = 24) so a symmetric error cannot hide

    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = Array3::<f32>::zeros((rh, n, n));
    for z in z0..z0 + 2 {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let vol = Volume::new(stack);

    let angles = Angles::uniform(nproj, 0.0, 2.0 * std::f32::consts::PI);
    let tilt_deg = 20.0f32;
    let phi = std::f32::consts::FRAC_PI_2 + tilt_deg * std::f32::consts::PI / 180.0;
    let geom = Geometry {
        angles: angles.clone(),
        center: Center::Scalar(n as f32 / 2.0),
        beam: Beam::Laminography { phi },
        detector: Detector {
            width: n,
            height: nz,
            pixel_size: 1.0,
        },
    };
    let params = ReconParams {
        lamino_rh: Some(rh),
        ..Default::default()
    };

    let sino = sim::project(&vol, &geom, &cuda).unwrap();

    let line = recon::recon(&sino, &geom, Algorithm::Linerec, &params, &cuda).unwrap();
    assert_eq!(line.array.dim(), (rh, n, n));
    let four = recon::recon(&sino, &geom, Algorithm::Fourierrec, &params, &cuda).unwrap();
    assert_eq!(four.array.dim(), (rh, n, n));

    let pl = z_profile(&line.array);
    let pf = z_profile(&four.array);
    let (zl, zf) = (argmax(&pl), argmax(&pf));
    eprintln!(
        "slab at z {z0}..{}: Linerec peak z = {zl}, Fourierrec peak z = {zf}",
        z0 + 1
    );
    assert!(
        (zl as i64 - z0 as i64).unsigned_abs() <= 1,
        "Linerec z peak {zl} not at the slab z {z0} — kernel z-centring broke"
    );
    assert!(
        (zf as i64 - z0 as i64).unsigned_abs() <= 1,
        "Fourierrec z peak {zf} not at the slab z {z0}"
    );
    assert!(
        (zl as i64 - zf as i64).unsigned_abs() <= 1,
        "Linerec (z {zl}) and Fourierrec (z {zf}) place the same feature at \
         different volume z — the (rh−nz)/2 convention offset is back"
    );
}

/// The centre probe's `sz` is the SAME volume z the reconstruction's slice
/// index means, in laminography: probing at the slab's z sees the slab, probing
/// `(rh − nz)/2` away (where the pre-fix detector-row centring would have put
/// it) sees an empty slice.
#[test]
fn cuda_lamino_center_probe_sz_is_the_volume_z() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let (n, nproj, nz) = (64usize, 90usize, 32usize);
    let rh = 48usize;
    let z0 = 20usize;

    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = Array3::<f32>::zeros((rh, n, n));
    for z in z0..z0 + 2 {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let angles = Angles::uniform(nproj, 0.0, 2.0 * std::f32::consts::PI);
    let tilt_deg = 20.0f32;
    let phi = std::f32::consts::FRAC_PI_2 + tilt_deg * std::f32::consts::PI / 180.0;
    let geom = Geometry {
        angles: angles.clone(),
        center: Center::Scalar(n as f32 / 2.0),
        beam: Beam::Laminography { phi },
        detector: Detector {
            width: n,
            height: nz,
            pixel_size: 1.0,
        },
    };
    let params = ReconParams {
        lamino_rh: Some(rh),
        ..Default::default()
    };
    let sino = sim::project(&Volume::new(stack), &geom, &cuda).unwrap();

    let energy = |sz: usize| -> f64 {
        let probe =
            cuda::center_probe(&sino, &geom, &params, &[n as f32 / 2.0], sz as i32).unwrap();
        probe.iter().map(|&v| (v as f64) * (v as f64)).sum()
    };

    let on_slab = energy(z0);
    let off_slab = energy(z0 - (rh - nz) / 2); // where nz/2-centring would look
    eprintln!(
        "probe energy at slab z {z0}: {on_slab:.3e}; at z {} (old offset): {off_slab:.3e}",
        z0 - (rh - nz) / 2
    );
    assert!(
        on_slab > 10.0 * off_slab,
        "probe at the slab's volume z ({on_slab:.3e}) is not decisively brighter \
         than at the old detector-centred z ({off_slab:.3e}) — probe sz is not \
         the reconstruction's volume z"
    );
}
