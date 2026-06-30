//! Cross-backend parity: CUDA analytic reconstruction vs the CPU reference,
//! correcting for CUDA's documented tomocupy output convention.
//!
//! The CUDA analytic kernels (`cfunc_rec`/`cfunc_linerec`/`cfunc_fourierrec`)
//! emit each slice with a **fixed handedness and scale inherited from tomocupy**
//! — the reconstruction is vertically flipped (image rows reversed) and scaled by
//! a per-algorithm constant — relative to the CPU/wgpu path (which follows
//! tomopy). See the note at `cuda/mod.rs` (`analytic_fbp_chunk` / `cfunc_linerec`
//! doc) and `docs/ARCHITECTURE.md`. CUDA's `lprec` kernel does **not** flip.
//!
//! The pre-existing CUDA streaming tests only compared CUDA-streaming against
//! CUDA-whole-volume (same convention) or against tomocupy — never against the
//! CPU/wgpu reference — so a real divergence in the CUDA recon (wrong geometry,
//! angle, filter, centre) beyond the known flip+scale would have been invisible.
//! This test closes that blind spot: it reconstructs the same sinogram on both
//! backends, undoes the documented convention (flip per the table below; scale is
//! absorbed by the scale-invariant Pearson metric), and asserts the result
//! matches the CPU reference. It also pins the *orientation* by asserting the
//! WRONG handedness correlates clearly worse — so if the CUDA convention ever
//! changes, this test fails and forces a deliberate update here, rather than the
//! change passing silently.
//!
//! Only built under `cuda`; needs a real CUDA device (skipped otherwise).
//! Run: `cargo test -p tomoxide --features cuda`.
#![cfg(feature = "cuda")]

use ndarray::{Array2, Axis};
use tomoxide::{
    recon, sim, Algorithm, Angles, CpuBackend, CudaBackend, Geometry, ReconParams, Volume,
};

/// Pearson correlation over a centered disk (amplitude-scale invariant), inside
/// the phantom support away from clipped corners.
fn pearson_disk(a: &Array2<f32>, b: &Array2<f32>, n: usize, radius_frac: f32) -> f64 {
    let c = (n as f32 - 1.0) / 2.0;
    let r2 = (radius_frac * (n as f32 / 2.0)).powi(2);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for iy in 0..n {
        for ix in 0..n {
            let (dy, dx) = (iy as f32 - c, ix as f32 - c);
            if dx * dx + dy * dy <= r2 {
                xs.push(a[[iy, ix]] as f64);
                ys.push(b[[iy, ix]] as f64);
            }
        }
    }
    let nn = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / nn;
    let my = ys.iter().sum::<f64>() / nn;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let (dx, dy) = (x - mx, y - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Best-fit scale `a` minimizing `||cuda − a·cpu||` over the disk (for logging the
/// per-algorithm convention scale; not asserted, since it differs per algorithm:
/// 2/π for fbp/linerec, ≈2·n² for fourierrec, 1 for lprec — the CUDA analytic
/// filter carries tomocupy's net gain, half tomopy's).
fn fit_scale(cuda: &Array2<f32>, cpu: &Array2<f32>, n: usize, radius_frac: f32) -> f64 {
    let c = (n as f32 - 1.0) / 2.0;
    let r2 = (radius_frac * (n as f32 / 2.0)).powi(2);
    let (mut num, mut den) = (0.0, 0.0);
    for iy in 0..n {
        for ix in 0..n {
            let (dy, dx) = (iy as f32 - c, ix as f32 - c);
            if dx * dx + dy * dy <= r2 {
                let (g, p) = (cuda[[iy, ix]] as f64, cpu[[iy, ix]] as f64);
                num += g * p;
                den += p * p;
            }
        }
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Reverse the image rows (the documented CUDA y-flip).
fn yflip(a: &Array2<f32>, n: usize) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((n, n));
    for r in 0..n {
        for col in 0..n {
            out[[n - 1 - r, col]] = a[[r, col]];
        }
    }
    out
}

/// One slice each from the CPU reference and the CUDA reconstruction of the same
/// Shepp–Logan sinogram. Shepp–Logan is vertically asymmetric, so a flip is
/// detectable. Two identical slices: the CUDA fourierrec path requires an even
/// slice count (it packs slice pairs for the real-FFT), and the analytic fbp/
/// linerec paths return garbage for a degenerate single-slice stack.
fn recon_both(algorithm: Algorithm, n: usize, nang: usize) -> (Array2<f32>, Array2<f32>) {
    let cpu = CpuBackend::new();
    let cuda = CudaBackend::new().expect("cuda backend");

    let phantom = sim::shepp2d(n).unwrap();
    let mut stack = ndarray::Array3::<f32>::zeros((2, n, n));
    stack.index_axis_mut(Axis(0), 0).assign(&phantom);
    stack.index_axis_mut(Axis(0), 1).assign(&phantom);
    let vol = Volume::new(stack);
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, 2, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();

    let params = ReconParams {
        num_gridx: Some(n),
        ..Default::default()
    };
    let rc = recon::recon(&sino, &geom, algorithm, &params, &cpu).unwrap();
    let rg = recon::recon(&sino, &geom, algorithm, &params, &cuda).unwrap();
    (
        rg.array.index_axis(Axis(0), 0).to_owned(),
        rc.array.index_axis(Axis(0), 0).to_owned(),
    )
}

/// Assert the CUDA recon matches the CPU reference once the documented
/// convention is undone, and that the opposite handedness does not.
fn check(algorithm: Algorithm, flipped: bool) {
    if CudaBackend::new().is_err() {
        eprintln!("skipping CUDA test ({algorithm:?}): no usable CUDA device");
        return;
    }
    let (n, nang) = (128usize, 180usize);
    let (cuda, cpu) = recon_both(algorithm, n, nang);

    // Apply the documented orientation, then compare scale-invariantly.
    let corrected = if flipped {
        yflip(&cuda, n)
    } else {
        cuda.clone()
    };
    let wrong = if flipped {
        cuda.clone()
    } else {
        yflip(&cuda, n)
    };

    let r_ok = pearson_disk(&corrected, &cpu, n, 0.8);
    let r_wrong = pearson_disk(&wrong, &cpu, n, 0.8);
    let scale = fit_scale(&corrected, &cpu, n, 0.8);
    eprintln!(
        "{algorithm:?}: corrected r = {r_ok:.5}, wrong-handedness r = {r_wrong:.5}, \
         convention scale (cuda/cpu) = {scale:.5}"
    );

    // (1) With the documented convention undone, CUDA == CPU to the f32/cuFFT
    //     floor. Catches any divergence beyond the known flip+scale.
    assert!(
        r_ok > 0.999,
        "{algorithm:?}: CUDA disagrees with CPU after undoing the documented \
         convention: r = {r_ok:.5} (expected > 0.999). If the CUDA convention \
         changed, update `flipped` for this algorithm here."
    );
    // (2) Orientation pin: the opposite handedness must correlate clearly worse,
    //     so a flip regression cannot pass. (Shepp–Logan is y-asymmetric.)
    assert!(
        r_wrong < 0.99,
        "{algorithm:?}: opposite handedness also matches (r = {r_wrong:.5}); the \
         orientation pin is ineffective for this phantom"
    );
}

#[test]
fn cuda_fbp_matches_cpu_under_convention() {
    check(Algorithm::Fbp, true);
}

#[test]
fn cuda_linerec_matches_cpu_under_convention() {
    check(Algorithm::Linerec, true);
}

#[test]
fn cuda_fourierrec_matches_cpu_under_convention() {
    check(Algorithm::Fourierrec, true);
}

#[test]
fn cuda_lprec_matches_cpu_under_convention() {
    // CUDA lprec does NOT flip — it already matches the CPU/wgpu orientation.
    check(Algorithm::Lprec, false);
}
