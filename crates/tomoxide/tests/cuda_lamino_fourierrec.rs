//! Verifies the device-resident Fourier/USFFT laminography path
//! (`analytic_lamino_fourierrec`, tomocupy `LamFourierRec`) reproduces the CPU
//! golden [`tomoxide::recon::lamino::lamino`] — the same algorithm, the same
//! ramp filter, the same Gaussian USFFT gridding. The only differences are the
//! `atomicAdd` accumulation order (gather/wrap adjoints) and `expf` vs the host
//! `f32::exp`, so the two agree to ~f32 rounding, not bit-exactly (the CPU golden
//! is itself validated against tomocupy at Pearson 0.99995).
//!
//! Own test binary (touches CUDA device state) per the suite convention.

use ndarray::{Array2, Axis};
use tomoxide::{
    recon, sim, Algorithm, Angles, Beam, Center, CpuBackend, CudaBackend, Detector, Geometry,
    Layout, ReconParams, Volume,
};

fn pearson(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.sum() / n, b.sum() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (dx, dy) = (x - ma, y - mb);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

fn nrmse(test: &[f32], reference: &[f32]) -> f32 {
    let mut se = 0.0f64;
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    for (&t, &r) in test.iter().zip(reference.iter()) {
        se += (t as f64 - r as f64).powi(2);
        mn = mn.min(r as f64);
        mx = mx.max(r as f64);
    }
    let range = (mx - mn).max(1e-12);
    ((se / test.len() as f64).sqrt() / range) as f32
}

#[test]
fn cuda_fourierrec_lamino_matches_cpu_golden() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();

    // Small stack that fits one GPU comfortably. Physical fidelity is validated
    // elsewhere (CPU golden vs tomocupy); here we only test GPU == CPU golden for
    // the identical whole-volume USFFT algorithm. A stacked Shepp phantom
    // parallel-projected then reconstructed as laminography exercises the tilt.
    let (n, nproj, nz) = (64usize, 90usize, 32usize);
    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = ndarray::Array3::<f32>::zeros((nz, n, n));
    for z in 0..nz {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let vol = Volume::new(stack);
    let angles = Angles::uniform(nproj, 0.0, std::f32::consts::PI);
    let geom_p = Geometry::parallel(angles.clone(), n, nz, 1.0);
    let sino = sim::project(&vol, &geom_p, &cpu).unwrap();

    let lamino_angle_deg = 20.0f32;
    let phi = std::f32::consts::FRAC_PI_2 + lamino_angle_deg * std::f32::consts::PI / 180.0;
    let geom_lam = Geometry {
        angles: angles.clone(),
        center: Center::Scalar(n as f32 / 2.0),
        beam: Beam::Laminography { phi },
        detector: Detector {
            width: n,
            height: nz,
            pixel_size: 1.0,
        },
    };
    let rh = 48usize;
    let params = ReconParams {
        lamino_rh: Some(rh),
        ..Default::default()
    };

    // GPU: device-resident LamFourierRec via the analytic-reconstruct dispatch.
    let gpu = recon::recon(&sino, &geom_lam, Algorithm::Fourierrec, &params, &cuda).unwrap();
    assert_eq!(gpu.array.dim(), (rh, n, n));

    // CPU golden: the standalone 3-D `recon::lamino::lamino`, fed the SAME raw
    // (unfiltered — both ramp-filter internally) projections in projection layout
    // `[nproj, nz, n]`.
    let pj = sino.to_layout(Layout::Projection);
    let proj = pj.array.as_slice().unwrap();
    let theta = &angles.0;
    let cpu_vol = recon::lamino::lamino(proj, theta, lamino_angle_deg, n, rh, &cpu).unwrap();
    assert_eq!(cpu_vol.len(), rh * n * n);
    let cpu_arr = ndarray::Array3::from_shape_vec((rh, n, n), cpu_vol).unwrap();

    // Per-rh-slice correlation.
    let mut min_r = 1.0f32;
    for z in 0..rh {
        let a = gpu.array.index_axis(Axis(0), z).to_owned();
        let b = cpu_arr.index_axis(Axis(0), z).to_owned();
        let r = pearson(&a, &b);
        min_r = min_r.min(r);
        assert!(
            r > 0.999,
            "GPU lamino slice {z} disagrees with CPU golden: r = {r:.6}"
        );
    }
    let err = nrmse(gpu.array.as_slice().unwrap(), cpu_arr.as_slice().unwrap());
    eprintln!(
        "cuda LamFourierRec vs CPU golden: min per-slice Pearson = {min_r:.6}, global NRMSE = {err:.3e}"
    );
    assert!(
        err < 5e-3,
        "GPU lamino NRMSE too large vs CPU golden: {err:.3e}"
    );
}
