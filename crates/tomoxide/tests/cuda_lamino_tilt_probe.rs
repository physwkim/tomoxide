//! `cuda::lamino_tilt_probe` — the laminography tilt-search primitive.
//!
//! `cfunc_linerec`'s `backprojection_try_lamino` indexes `phi` per output slot,
//! so one launch reconstructs one slice at N tilts. That is only worth anything
//! if a probe image IS the slice the reconstruction produces at that tilt: a
//! focus metric ranking a differently-scaled or y-flipped proxy would pick a
//! tilt for the wrong image. tomocupy's kernel writes `(n-ty-1)` and scales by
//! 4/nproj; the convention unification (`docs/ARCHITECTURE.md` §4.1) moved this
//! file's other kernels off both and left the lamino one behind — not because it
//! was uncalled (`backprojection_try_ker` is uncalled too and was migrated) but
//! because §4.1 then declared laminography exempt. So equality against
//! `Algorithm::Linerec` is the thing to pin: it is the assertion that does not
//! care why the two ever diverged.
#![cfg(feature = "cuda")]

use ndarray::{Array2, Axis};
use tomoxide::{
    cuda, recon, sim, Algorithm, Angles, Beam, Center, CpuBackend, CudaBackend, Detector, Geometry,
    ReconParams, Volume,
};

fn lamino_sino(
    n: usize,
    nproj: usize,
    nz: usize,
    cpu: &CpuBackend,
) -> (tomoxide::Tomo<f32>, Angles) {
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = ndarray::Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let angles = Angles::uniform(nproj, 0.0, std::f32::consts::PI);
    let geom_p = Geometry::parallel(angles.clone(), n, nz, 1.0);
    let sino = sim::project(&Volume::new(stack), &geom_p, cpu).unwrap();
    (sino, angles)
}

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

fn max_abs(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// A probe image at tilt `t`, slice `z`, equals the `Algorithm::Linerec`
/// laminography reconstruction's slice `z` at the same tilt — exactly. Same
/// filter, same back-projection math; the probe's only difference is that its
/// output index selects the tilt instead of z, so any mismatch is a convention
/// bug (the y-flip or the 4/nproj gain), not round-off.
#[test]
fn cuda_lamino_tilt_probe_matches_linerec_recon() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (n, nproj, nz) = (64usize, 90usize, 32usize);
    let (sino, angles) = lamino_sino(n, nproj, nz, &cpu);

    let rh = 48usize;
    let params = ReconParams {
        lamino_rh: Some(rh),
        ..Default::default()
    };
    // Well inside `rh` and off-centre, so a z-offset error cannot hide.
    let z = 17usize;
    let tilts = [12.0f32, 20.0, 31.5];

    let probe = cuda::lamino_tilt_probe(
        &sino,
        &geom_lamino(&angles, n, nz, 0.0), // beam ignored — `tilts` supplies it
        &params,
        &tilts,
        z as i32,
    )
    .unwrap();
    assert_eq!(probe.dim(), (tilts.len(), n, n));

    for (i, &t) in tilts.iter().enumerate() {
        let geom = geom_lamino(&angles, n, nz, t);
        let vol = recon::recon(&sino, &geom, Algorithm::Linerec, &params, &cuda).unwrap();
        assert_eq!(vol.array.dim(), (rh, n, n));
        let want = vol.array.index_axis(Axis(0), z).to_owned();
        let got = probe.index_axis(Axis(0), i).to_owned();

        let peak = want.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = max_abs(&want, &got);
        eprintln!("tilt {t}°: peak {peak:e}, max abs diff vs recon slice {z} = {d:e}");
        assert!(peak > 0.0, "tilt {t}°: reference slice is all zeros");
        assert_eq!(want, got, "tilt {t}°: probe ≠ Linerec recon slice {z}");
    }
}

/// One launch, N tilts: every row of the probe is the tilt it was asked for, not
/// a copy of the first. `phi` is per-output-slot inside the kernel, so a wiring
/// that passed a scalar (or bound the array wrong) would still return the right
/// shape while every row held the same image.
#[test]
fn cuda_lamino_tilt_probe_rows_are_distinct_tilts() {
    // `lamino_tilt_probe` is a free function over the selected devices, so this
    // only needs to know a device exists — no backend handle is used below.
    if let Err(e) = CudaBackend::new() {
        eprintln!("skipping CUDA test: {e}");
        return;
    }
    let cpu = CpuBackend::new();
    let (n, nproj, nz) = (64usize, 90usize, 32usize);
    let (sino, angles) = lamino_sino(n, nproj, nz, &cpu);
    let params = ReconParams {
        lamino_rh: Some(48),
        ..Default::default()
    };
    let tilts = [5.0f32, 25.0];

    let probe = cuda::lamino_tilt_probe(
        &sino,
        &geom_lamino(&angles, n, nz, 0.0),
        &params,
        &tilts,
        17,
    )
    .unwrap();

    let a = probe.index_axis(Axis(0), 0).to_owned();
    let b = probe.index_axis(Axis(0), 1).to_owned();
    let d = max_abs(&a, &b);
    eprintln!("tilt 5° vs 25° max abs diff = {d:e}");
    assert!(d > 0.0, "both probe rows identical — phi is not per-tilt");
}

/// The tilt sweep it exists for: probing N tilts costs one pass, and the focus
/// peak it finds is the peak the full reconstructions agree on.
#[test]
fn cuda_lamino_tilt_probe_finds_the_same_peak_as_full_recons() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (n, nproj, nz) = (64usize, 90usize, 32usize);
    let (sino, angles) = lamino_sino(n, nproj, nz, &cpu);
    let params = ReconParams {
        lamino_rh: Some(48),
        ..Default::default()
    };
    let z = 24usize; // mid-volume
    let tilts: Vec<f32> = (0..7).map(|k| k as f32 * 5.0).collect();

    // Mean |∇|² — the focus metric an autofocus sweep scores with.
    let focus = |img: &Array2<f32>| -> f64 {
        let mut acc = 0.0f64;
        for y in 1..n - 1 {
            for x in 1..n - 1 {
                let gx = (img[[y, x + 1]] - img[[y, x - 1]]) as f64;
                let gy = (img[[y + 1, x]] - img[[y - 1, x]]) as f64;
                acc += gx * gx + gy * gy;
            }
        }
        acc / ((n - 2) * (n - 2)) as f64
    };

    let probe = cuda::lamino_tilt_probe(
        &sino,
        &geom_lamino(&angles, n, nz, 0.0),
        &params,
        &tilts,
        z as i32,
    )
    .unwrap();
    let from_probe: Vec<f64> = (0..tilts.len())
        .map(|i| focus(&probe.index_axis(Axis(0), i).to_owned()))
        .collect();

    let from_recon: Vec<f64> = tilts
        .iter()
        .map(|&t| {
            let geom = geom_lamino(&angles, n, nz, t);
            let vol = recon::recon(&sino, &geom, Algorithm::Linerec, &params, &cuda).unwrap();
            focus(&vol.array.index_axis(Axis(0), z).to_owned())
        })
        .collect();

    for (i, &t) in tilts.iter().enumerate() {
        eprintln!(
            "tilt {t:>5}°  probe {:.6e}  recon {:.6e}",
            from_probe[i], from_recon[i]
        );
    }
    let argmax = |v: &[f64]| {
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
    };
    assert_eq!(
        argmax(&from_probe),
        argmax(&from_recon),
        "probe sweep peaks at a different tilt than the full reconstructions"
    );
    for (p, r) in from_probe.iter().zip(&from_recon) {
        assert!(
            (p - r).abs() <= r.abs() * 1e-6,
            "probe focus {p:e} ≠ recon focus {r:e}"
        );
    }
}
