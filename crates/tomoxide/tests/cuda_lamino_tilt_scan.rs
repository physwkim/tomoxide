//! `recon::center::lamino_tilt_scan` — the laminography tilt search.
//!
//! The scan is expensive by construction: one full reconstruction per candidate,
//! scored by the max focus over every slice. `cuda::lamino_tilt_probe` would make
//! a whole sweep one launch, and the tests next door prove the probe reproduces
//! the reconstruction exactly — so the only thing that justifies not using it here
//! is that a fixed slice cannot rank tilts at all. That claim is what these tests
//! have to carry, and it needs an object with structure **in z**: the probe suite's
//! phantom is one 2-D slice stacked into every row, which has no in-focus layer to
//! move and would let a fixed-slice sweep look correct.
#![cfg(feature = "cuda")]

use ndarray::{Array2, Axis};
use tomoxide::recon::center::{lamino_tilt_scan, slice_focus, SampleBand, TiltFocus};
use tomoxide::{
    cuda, sim, Algorithm, Angles, Beam, Center, CudaBackend, Detector, Geometry, ReconParams,
    Volume,
};

const TRUE_TILT: f32 = 20.0;

fn geom_lamino(angles: &Angles, n: usize, nz: usize, tilt_deg: f32) -> Geometry {
    Geometry {
        angles: angles.clone(),
        center: Center::Scalar(n as f32 / 2.0),
        beam: Beam::Laminography {
            phi: std::f32::consts::FRAC_PI_2 + tilt_deg * std::f32::consts::PI / 180.0,
        },
        detector: Detector {
            width: n,
            height: nz,
            pixel_size: 1.0,
        },
    }
}

/// A phantom that occupies a **slab**, deliberately off the middle of the volume.
/// The whole point of scoring the max over z is that the in-focus layer is not
/// where you assume, so a phantom centred in z would hide a scan that only ever
/// looked at the middle.
fn slab_volume(n: usize, rh: usize, z_lo: usize, z_hi: usize) -> Volume<f32> {
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = ndarray::Array3::<f32>::zeros((rh, n, n));
    for z in z_lo..z_hi {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    Volume::new(stack)
}

/// The scan, the phantom, and the tilts the data was actually projected at.
/// Returns the scores, the in-focus slice `on_tilt` handed over for each, and
/// the slab bounds.
fn scan(tilts: &[f32]) -> (Vec<TiltFocus>, Vec<Array2<f32>>, usize, usize) {
    scan_banded(tilts, None)
}

fn scan_banded(
    tilts: &[f32],
    band: Option<SampleBand>,
) -> (Vec<TiltFocus>, Vec<Array2<f32>>, usize, usize) {
    let cuda = CudaBackend::new().expect("checked by the caller");
    let (n, nproj, nz) = (64usize, 90usize, 32usize);
    let rh = cuda::lamino_recon_height(nz, TRUE_TILT);
    // Off-centre in z, and inside the reconstructed depth by a margin.
    let (z_lo, z_hi) = (rh / 4, rh / 4 + 4);

    let angles = Angles::uniform(nproj, 0.0, 2.0 * std::f32::consts::PI);
    let geom_true = geom_lamino(&angles, n, nz, TRUE_TILT);
    let sino = sim::project(&slab_volume(n, rh, z_lo, z_hi), &geom_true, &cuda).unwrap();

    let params = ReconParams::default();
    let mut slices = Vec::new();
    let scores = lamino_tilt_scan(
        &sino,
        &geom_lamino(&angles, n, nz, 0.0), // beam ignored — `tilts` supplies it
        Algorithm::Fourierrec,
        &params,
        tilts,
        band,
        &mut |r, slice| {
            assert_eq!(
                slice.dim(),
                (n, n),
                "the slice handed to on_tilt is not an [n, n] reconstruction plane"
            );
            eprintln!(
                "  {:5.1}°  focus {:.4e}  z_peak {} of {}",
                r.tilt_deg, r.focus, r.z_peak, r.depth
            );
            slices.push(slice.to_owned());
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(
        slices.len(),
        tilts.len(),
        "on_tilt was not called once per candidate"
    );
    (scores, slices, z_lo, z_hi)
}

/// Reconstructing at the tilt the data was projected at is what maximises the
/// focus. This is the assertion the whole subcommand rests on: without it the
/// scan is an expensive way to return its own starting guess.
#[test]
fn cuda_lamino_tilt_scan_recovers_the_tilt_the_data_was_projected_at() {
    if let Err(e) = CudaBackend::new() {
        eprintln!("skipping CUDA test: {e}");
        return;
    }
    let tilts: Vec<f32> = (0..5).map(|k| TRUE_TILT - 10.0 + k as f32 * 5.0).collect();
    let (scores, _, _, _) = scan(&tilts);

    let best = scores
        .iter()
        .max_by(|a, b| a.focus.partial_cmp(&b.focus).unwrap())
        .unwrap();
    assert_eq!(
        best.tilt_deg, TRUE_TILT,
        "the scan peaked at {}°, not the {TRUE_TILT}° the sinogram was projected at",
        best.tilt_deg
    );
}

/// `z_peak` is the layer the sample is on, and the sample is not in the middle.
/// A fixed-slice sweep has to pick a row before it knows this — that is the
/// ill-posedness the scan pays a full reconstruction per tilt to avoid.
#[test]
fn cuda_lamino_tilt_scan_finds_the_layer_the_sample_is_on_not_the_middle() {
    if let Err(e) = CudaBackend::new() {
        eprintln!("skipping CUDA test: {e}");
        return;
    }
    let (scores, _, z_lo, z_hi) = scan(&[TRUE_TILT]);
    let r = &scores[0];
    let mid = r.depth / 2;
    eprintln!("slab z {z_lo}..{z_hi}, z_peak {}, middle {mid}", r.z_peak);
    assert!(
        r.z_peak >= z_lo.saturating_sub(2) && r.z_peak < z_hi + 2,
        "z_peak {} is not on the slab at z {z_lo}..{z_hi}",
        r.z_peak
    );
    assert_ne!(
        r.z_peak, mid,
        "the slab was placed off-centre precisely so that a scan that only looked \
         at the middle would fail here"
    );
    assert!(r.focus > 0.0, "focus is not positive — the volume is empty");
}

/// The focus the scan reports is `slice_focus` of the slice it names, not a
/// number the streaming tiles quietly changed. `on_tile` sees the volume in
/// row-order tiles and only a running max survives; an off-by-one in the tile
/// offset would name a neighbouring slice while still reporting a plausible peak.
#[test]
fn cuda_lamino_tilt_scan_focus_belongs_to_the_slice_it_names() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let (scores, slices, _, _) = scan(&[TRUE_TILT]);
    let r = &scores[0];

    // Re-reconstruct the same tilt whole and score the slice the scan named.
    let (n, nproj, nz) = (64usize, 90usize, 32usize);
    let rh = cuda::lamino_recon_height(nz, TRUE_TILT);
    let angles = Angles::uniform(nproj, 0.0, 2.0 * std::f32::consts::PI);
    let geom = geom_lamino(&angles, n, nz, TRUE_TILT);
    let sino = sim::project(&slab_volume(n, rh, rh / 4, rh / 4 + 4), &geom, &cuda).unwrap();
    let vol = tomoxide::recon::recon(
        &sino,
        &geom,
        Algorithm::Fourierrec,
        &ReconParams::default(),
        &cuda,
    )
    .unwrap();
    assert_eq!(vol.array.dim().0, r.depth, "scan and recon disagree on rh");

    let named = vol.array.index_axis(Axis(0), r.z_peak);
    let want = slice_focus(&named);
    eprintln!("scan focus {:.6e}, recon slice focus {want:.6e}", r.focus);
    assert!(
        (r.focus - want).abs() <= want.abs() * 1e-6,
        "scan reported {:.6e} for slice {} but that slice scores {want:.6e}",
        r.focus,
        r.z_peak
    );
    // The image handed to `on_tilt` is that same slice — the streamed tile is
    // dropped long before the scan knows it won, so this is what says the right
    // one was kept and not its neighbour. Not bit-equality: the scan streams the
    // volume in rh-tiles while this reference reconstructs it whole, and the two
    // batch their FFTs differently. A neighbouring slice would miss by ~100 % of
    // peak, not by round-off.
    let peak = named.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let d = slices[0]
        .iter()
        .zip(named.iter())
        .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
    eprintln!(
        "handed-over slice vs recon slice {}: {:.2e} of peak",
        r.z_peak,
        d / peak
    );
    assert!(
        d / peak < 1e-5,
        "the slice on_tilt handed over differs from the z_peak slice it named by \
         {:.2e} of peak",
        d / peak
    );
    // ...and it really is the best one, i.e. the running max kept the right tile.
    for z in 0..r.depth {
        let f = slice_focus(&vol.array.index_axis(Axis(0), z));
        assert!(
            f <= r.focus * (1.0 + 1e-6),
            "slice {z} scores {f:.6e}, above the reported max {:.6e}",
            r.focus
        );
    }
}

/// The band is what gets scored, and nothing else answers. A band enclosing the
/// slab reproduces the whole-volume winner; a band on the empty far side of the
/// volume forces the winner inside itself, away from the slab a whole-volume max
/// would pick — which is the point: on real data the plane a whole-volume max
/// picks is noise, and the band is how the caller keeps the score on the sample.
#[test]
fn cuda_lamino_tilt_scan_scores_only_the_sample_band() {
    if let Err(e) = CudaBackend::new() {
        eprintln!("skipping CUDA test: {e}");
        return;
    }
    let (unbanded, _, z_lo, z_hi) = scan(&[TRUE_TILT]);

    let on_slab = SampleBand {
        z: (z_lo, z_hi - 1),
        tilt_deg: TRUE_TILT,
    };
    let (scores, _, _, _) = scan_banded(&[TRUE_TILT], Some(on_slab));
    let r = &scores[0];
    // At its own tilt the band maps to itself, and the slab holds the winner.
    assert_eq!(
        r.band,
        Some((z_lo, z_hi - 1)),
        "band not carried to TiltFocus"
    );
    assert_eq!(
        r.z_peak, unbanded[0].z_peak,
        "a band enclosing the slab changed the winner"
    );
    // Two independent scans, so the FFT batching can differ — round-off, not
    // bit-equality, same as the whole-vs-streamed comparison above.
    assert!(
        (r.focus - unbanded[0].focus).abs() <= unbanded[0].focus * 1e-6,
        "banded {} vs unbanded {}",
        r.focus,
        unbanded[0].focus
    );

    let far = (3 * r.depth / 4, 3 * r.depth / 4 + 4);
    let off_slab = SampleBand {
        z: far,
        tilt_deg: TRUE_TILT,
    };
    let (scores, _, _, _) = scan_banded(&[TRUE_TILT], Some(off_slab));
    let r = &scores[0];
    assert!(
        (far.0..=far.1).contains(&r.z_peak),
        "the winner {} escaped the scored band {}..{}",
        r.z_peak,
        far.0,
        far.1
    );
    assert!(
        r.focus < unbanded[0].focus,
        "an empty band scored {} — at least as high as the slab's {} — so the band \
         did not restrict anything",
        r.focus,
        unbanded[0].focus
    );
    // The profile is never truncated to the band: the next band is read off it.
    assert_eq!(r.focus_by_z.len(), r.depth);
}
