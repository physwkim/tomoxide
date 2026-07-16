//! Cross-backend parity: CUDA analytic reconstruction vs the CPU reference.
//!
//! Since the orientation unification (Phase 1) the CUDA analytic kernels
//! (`cfunc_rec`/`cfunc_linerec`/`cfunc_fourierrec`) emit each slice in the **same
//! handedness as the CPU/wgpu path** (which follows tomopy) — no vertical flip.
//! Since the scale unification (Phase 2) they also emit the **same amplitude** as
//! the CPU path: fbp/linerec back-project at `π/nproj` (was tomocupy's `4/nproj`),
//! the tomocupy filter ½-gain is removed, fourierrec normalizes its unnormalized
//! cuFFT inverse by `(2n)²`, and lprec matches directly. So this test now asserts
//! both `cuda == cpu` orientation (Pearson) AND `cuda/cpu ≈ 1` scale; the residual
//! ~1.6 % is the ramp-SHAPE gap (CUDA `_wint` order-12 quadrature ramp vs the CPU
//! linear ramp) plus the fourierrec USFFT deapodization, not a convention scale.
//! See `cuda/mod.rs` (`cfunc_linerec` doc) and `docs/ARCHITECTURE.md`.
//!
//! Laminography is deliberately NOT exercised here: the CUDA lamino path
//! (`cfunc_linerec` tilted back-projector) and the CPU lamino path
//! (`recon::lamino`, a USFFT algorithm) are *different reconstruction algorithms*
//! with different filter frameworks, so their amplitudes are not comparable and
//! are excluded from the *scale* unification (each is validated against its own
//! reference — CUDA vs tomocupy, CPU vs wgpu). Only the amplitude is exempt: both
//! lamino paths emit the same CPU/tomopy handedness as the algorithms asserted
//! below, so neither is y-flipped here.
//!
//! The pre-existing CUDA streaming tests only compared CUDA-streaming against
//! CUDA-whole-volume (same convention) or against tomocupy — never against the
//! CPU/wgpu reference — so a real divergence in the CUDA recon (wrong geometry,
//! angle, filter, centre) would have been invisible. This test closes that blind
//! spot: it reconstructs the same sinogram on both backends and asserts the CUDA
//! result matches the CPU reference directly (scale absorbed by Pearson). It also
//! pins the *orientation* by asserting the WRONG (y-flipped) handedness correlates
//! clearly worse — so a flip regression fails here rather than passing silently.
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

/// Best-fit scale `a` minimizing `||cuda − a·cpu||` over the disk. After the scale
/// unification (Phase 2) this is `≈ 1` for every path (fbp/linerec/fourierrec/
/// lprec), so `check` asserts it; the residual ~1.6 % is the ramp-shape + USFFT
/// deapodization gap, not a convention scale. (Pre-unification it was 2/π for
/// fbp/linerec, ≈4·n² for fourierrec, ½ for lprec.)
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
    // (3) Scale pin (Phase 2): CUDA emits the same amplitude as CPU, `cuda/cpu ≈ 1`.
    //     The ~1.6 % residual is the ramp-shape + USFFT-deapodization gap; the 5 %
    //     bar is far below any convention regression (½, 2/π ≈ 0.64, ≈4n²) so those
    //     fail here instead of hiding behind the scale-invariant Pearson.
    assert!(
        (scale - 1.0).abs() < 0.05,
        "{algorithm:?}: CUDA/CPU scale = {scale:.5}, expected ≈ 1 (|Δ| < 0.05). A \
         per-algorithm convention scale has re-appeared — the Phase 2 unification \
         regressed."
    );
}

// After the orientation + scale unification, NO CUDA reconstruction path flips or
// rescales vs CPU — every analytic method matches the CPU/wgpu handedness
// (`flipped = false`) AND amplitude (`scale ≈ 1`, asserted in `check`).
// Laminography is a different algorithm per backend (see the module doc) and is
// not exercised here.
#[test]
fn cuda_fbp_matches_cpu_under_convention() {
    check(Algorithm::Fbp, false);
}

#[test]
fn cuda_linerec_matches_cpu_under_convention() {
    check(Algorithm::Linerec, false);
}

#[test]
fn cuda_fourierrec_matches_cpu_under_convention() {
    check(Algorithm::Fourierrec, false);
}

#[test]
fn cuda_lprec_matches_cpu_under_convention() {
    check(Algorithm::Lprec, false);
}
