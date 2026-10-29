//! `PreppedStack::reconstruct` against `reconstruct_lamino_streaming` on the
//! same file: the stack's held normalization plus the shared phase/stripe tail
//! must give the volume the file path gives.
#![cfg(feature = "cuda")]

use tomoxide::align::PreppedStack;
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
