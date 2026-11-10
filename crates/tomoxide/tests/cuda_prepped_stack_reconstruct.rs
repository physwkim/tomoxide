//! `PreppedStack::reconstruct` against `reconstruct_lamino_streaming` on the
//! same file: the stack's held normalization plus the shared phase/stripe tail
//! must give the volume the file path gives.
#![cfg(feature = "cuda")]

use tomoxide::align::PreppedStack;
use tomoxide::recon::center::focus_by_z;
use tomoxide::{
    io, reconstruct_lamino_streaming, Algorithm, BackendKind, Engine, PhaseMethod, PrepOptions,
    ReconParams, StripeMethod,
};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/streaming_dxchange.h5"
);

fn collect(
    rows: &mut Vec<f32>,
) -> impl FnMut(usize, &tomoxide::Volume<f32>) -> tomoxide::Result<()> + '_ {
    move |_rh0, tile| {
        rows.extend_from_slice(tile.array.as_slice().unwrap());
        Ok(())
    }
}

fn nrmse(test: &[f32], reference: &[f32]) -> f64 {
    let (mut se, mut sr) = (0.0f64, 0.0f64);
    for (&t, &r) in test.iter().zip(reference) {
        se += (t as f64 - r as f64).powi(2);
        sr += (r as f64).powi(2);
    }
    (se / sr).sqrt()
}

#[test]
fn stack_reconstruct_matches_the_file_path() {
    let engine = match Engine::new(BackendKind::Cuda) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let stack = PreppedStack::load(engine.backend(), FIXTURE.as_ref()).unwrap();
    let (_nproj, _nz, nx) = stack.shape();
    let geom = stack.geometry(nx as f32 / 2.0, Some(20.0));
    let params = ReconParams::default();
    let paganin = PhaseMethod::Paganin {
        pixel_size: 2.64e-5,
        dist: 10.0,
        energy: 8.333,
        alpha: 1e-3,
    };
    // None takes the file path's fused normalization; with prep both take the
    // same steps and differ only by the GPU FFTs' run-to-run floor.
    for (prep, tol) in [
        (PrepOptions::default(), 1e-4),
        (
            PrepOptions {
                stripe: StripeMethod::Fw {
                    sigma: 2.0,
                    level: None,
                },
                phase: paganin,
            },
            1e-6,
        ),
    ] {
        let ds = io::open_dxchange(FIXTURE).unwrap().read_all().unwrap();
        let mut file = Vec::new();
        let want = reconstruct_lamino_streaming(
            ds,
            &geom,
            Algorithm::Fourierrec,
            &params,
            &prep,
            &engine,
            &mut collect(&mut file),
        )
        .unwrap();
        let mut held = Vec::new();
        let got = stack
            .reconstruct(
                engine.backend(),
                &geom,
                Algorithm::Fourierrec,
                &params,
                &prep,
                &mut collect(&mut held),
            )
            .unwrap();
        assert_eq!(got, want);
        let err = nrmse(&held, &file);
        assert!(err <= tol, "{prep:?}: NRMSE {err:.3e} > {tol:.0e}");
    }
}

#[test]
fn select_keeps_the_named_projections_in_order() {
    let engine = match Engine::new(BackendKind::Cuda) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let stack = PreppedStack::load(engine.backend(), FIXTURE.as_ref()).unwrap();
    let (nproj, nz, nx) = stack.shape();
    let odd: Vec<usize> = (1..nproj).step_by(2).collect();
    let half = stack.select(&odd).unwrap();
    assert_eq!(half.shape(), (odd.len(), nz, nx));
    let all = stack.data().as_layout(tomoxide::Layout::Projection);
    let sel = half.data().as_layout(tomoxide::Layout::Projection);
    for (k, &i) in odd.iter().enumerate() {
        assert_eq!(
            sel.array.index_axis(ndarray::Axis(0), k),
            all.array.index_axis(ndarray::Axis(0), i)
        );
    }
    let geom = half.geometry(0.0, None);
    let theta = stack.geometry(0.0, None).angles.0;
    assert_eq!(
        geom.angles.0,
        odd.iter().map(|&i| theta[i]).collect::<Vec<_>>()
    );
    let Err(e) = stack.select(&[nproj]) else {
        panic!("projection {nproj} of {nproj} selected");
    };
    assert!(e.to_string().contains("out of range"));
}

/// A centre sweep scores a candidate by its mean slice focus over the band:
/// under a tilt the band's own back-projection, in parallel beam a probe per
/// slice. Without a band it scores the slice it shows, and the band never
/// changes the frames.
#[test]
fn center_sweep_scores_the_band_by_its_mean() {
    let engine = match Engine::new(BackendKind::Cuda) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let stack = PreppedStack::load(engine.backend(), FIXTURE.as_ref()).unwrap();
    let (nproj, _nz, nx) = stack.shape();
    let centers = [
        nx as f32 / 2.0 - 1.0,
        nx as f32 / 2.0,
        nx as f32 / 2.0 + 1.0,
    ];
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    for tilt in [Some(20.0), None] {
        let depth = stack.depth(tilt);
        let (slice, band) = (depth / 2, (1, depth - 2));
        let one = stack
            .center_sweep(tilt, slice, None, &centers, None)
            .unwrap();
        assert_eq!(one.scored, (slice, slice));
        assert_eq!(one.focus, focus_by_z(one.frames.view()));
        let banded = stack
            .center_sweep(tilt, slice, Some(band), &centers, None)
            .unwrap();
        assert_eq!(banded.scored, band);
        let err = nrmse(
            banded.frames.as_slice().unwrap(),
            one.frames.as_slice().unwrap(),
        );
        assert!(err < 1e-5, "{tilt:?}: the band changed the frames by {err}");
        let want: Vec<f64> = match tilt {
            Some(t) => {
                let filter = ReconParams::default().filter_name;
                let mut scorer = stack.band_focus(filter, t, band).unwrap();
                let still = vec![0.0; nproj];
                centers
                    .iter()
                    .map(|&c| mean(&scorer.profile(&still, &still, c).unwrap()))
                    .collect()
            }
            None => {
                let by_slice: Vec<Vec<f64>> = (band.0..=band.1)
                    .map(|z| {
                        stack
                            .center_sweep(None, z, None, &centers, None)
                            .unwrap()
                            .focus
                    })
                    .collect();
                (0..centers.len())
                    .map(|i| mean(&by_slice.iter().map(|f| f[i]).collect::<Vec<_>>()))
                    .collect()
            }
        };
        for (got, want) in banded.focus.iter().zip(&want) {
            assert!(
                (got / want - 1.0).abs() < 1e-6,
                "{tilt:?}: {:?} is not the band's mean {want:?}",
                banded.focus
            );
        }
        assert_ne!(banded.focus, one.focus);
        let fired = tomoxide::CancelToken::new();
        fired.cancel();
        let stopped = stack.center_sweep(tilt, slice, Some(band), &centers, Some(&fired));
        assert!(matches!(stopped, Err(tomoxide::Error::Cancelled)));
    }
    let Err(e) = stack.center_sweep(None, 0, Some((3, 2)), &centers, None) else {
        panic!("an empty band scored");
    };
    assert!(e.to_string().contains("holds no slices"));
}
