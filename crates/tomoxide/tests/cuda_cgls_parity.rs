//! CUDA CGLS parity: the device-resident CGLS solver must match the same CGLS
//! math run through the generic host solver on the *same* CUDA kernels. Only the
//! per-slice dot products differ (device shared-mem tree reduce vs host ndarray
//! sequential sum), and because CGLS is a Krylov method those tiny
//! summation-order differences propagate through the conjugate recurrence — so
//! the two agree to ~1e-2 relative (not the ~1e-8 of contractive SIRT/grad)
//! while staying visually identical (Pearson ≈ 1). Device-residency changes the
//! transfer schedule, not the algorithm.
//!
//! Only built under `cuda`; needs a real CUDA device (skipped otherwise).
#![cfg(feature = "cuda")]

use ndarray::{Array2, Axis};
use tomoxide::{
    recon, sim, Algorithm, Angles, Backend, CpuBackend, CudaBackend, DeviceKind, Dtype,
    FilteredBackproject, ForwardProject, Geometry, IterativeReconstruct, ReconParams, Volume,
};

/// A `CudaBackend` with the device-resident path hidden, so `recon` composes the
/// generic host solver from the CUDA projector/backprojector (per-iteration).
struct PerIterCuda<'a>(&'a CudaBackend);
impl Backend for PerIterCuda<'_> {
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn device(&self) -> DeviceKind {
        self.0.device()
    }
    fn supports(&self, dt: Dtype) -> bool {
        self.0.supports(dt)
    }
    fn projector(&self) -> Option<&dyn ForwardProject> {
        self.0.projector()
    }
    fn backprojector(&self) -> Option<&dyn FilteredBackproject> {
        self.0.backprojector()
    }
    fn iterative_reconstruct(&self) -> Option<&dyn IterativeReconstruct> {
        None
    }
}

fn rel_l2(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(&x, &y)| (x - y).powi(2)).sum();
    let den: f32 = b.iter().map(|&y| y * y).sum();
    (num / den).sqrt()
}

fn pearson(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.sum() / n, b.sum() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in a.iter().zip(b) {
        sxy += (x - ma) * (y - mb);
        sxx += (x - ma).powi(2);
        syy += (y - mb).powi(2);
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Stack a 2-D phantom into `nz` identical slices (CUDA drops slice 0 by the
/// `vr < nz-1` boundary guard, so a parity check needs an interior slice).
fn stack(slice2d: &Array2<f32>, nz: usize) -> ndarray::Array3<f32> {
    let (h, w) = slice2d.dim();
    let mut v = ndarray::Array3::<f32>::zeros((nz, h, w));
    for z in 0..nz {
        v.index_axis_mut(Axis(0), z).assign(slice2d);
    }
    v
}

#[test]
fn cgls_device_matches_per_iteration_cuda() {
    let (n, nang, iters, nz) = (128, 120, 40, 3);
    let cpu = CpuBackend::new();
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no CUDA device, skipping: {e}");
            return;
        }
    };
    let periter = PerIterCuda(&cuda);

    let phantom = sim::shepp2d(n).unwrap();
    let vol = Volume::new(stack(&phantom, nz));
    let geom = Geometry::parallel(Angles::uniform(nang, 0.0, std::f32::consts::PI), n, nz, 1.0);
    let sino = sim::project(&vol, &geom, &cpu).unwrap();
    let p = ReconParams {
        num_gridx: Some(n),
        num_iter: iters,
        ..Default::default()
    };

    // Interior slice 1 (slice 0 is dropped by the CUDA ≥2-slice boundary rule).
    let dev = recon::recon(&sino, &geom, Algorithm::Cgls, &p, &cuda).unwrap();
    let host = recon::recon(&sino, &geom, Algorithm::Cgls, &p, &periter).unwrap();
    let dev2d = dev.array.index_axis(Axis(0), 1).to_owned();
    let host2d = host.array.index_axis(Axis(0), 1).to_owned();

    let r = rel_l2(&dev2d, &host2d);
    let corr = pearson(&dev2d, &host2d);
    eprintln!(
        "CGLS device vs per-iteration CUDA ({iters} iters): relL2 = {r:.2e}, pearson = {corr:.6}"
    );
    assert!(
        dev2d.iter().all(|v| v.is_finite()),
        "device-resident CGLS produced non-finite values"
    );
    // Krylov float-order sensitivity, not a logic error: a real divergence is
    // O(1)/NaN, not ~1e-2. Pearson is the correctness gate; relL2 bounds drift.
    assert!(
        corr > 0.9999,
        "device CGLS decorrelated from per-iteration CUDA: r = {corr:.6}"
    );
    assert!(
        r < 2e-2,
        "device-resident CGLS diverged from per-iteration CUDA: relL2 = {r:.3e}"
    );
}
