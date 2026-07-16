//! `cuda::center_probe` / `center_probe_sweep` — the rotation-centre search
//! primitive, and the lattice invariant that makes it trustworthy.
//!
//! `cfunc_linerec`'s `backprojection_try` indexes its shift array per output
//! slot, so one launch reconstructs one slice at N candidate centres off a single
//! filtering. The catch is that a full reconstruction moves its centre with a
//! Fourier linear phase while the probe moves the back-projection sampling
//! coordinate with linear interpolation: the two agree *exactly* when the offset
//! is a whole number of columns, and differ by ~1.6 % of peak at half a column.
//! An unguarded sub-pixel sweep therefore ranks integer offsets artificially
//! sharp. `center_probe_sweep` removes that failure by construction (one probe
//! per fractional lattice), and these tests pin both halves: the raw probe's
//! integer exactness, and the sweep's exactness at arbitrary spacing.
#![cfg(feature = "cuda")]

use ndarray::{Array2, Array3, Axis};
use tomoxide::{
    cuda, recon, sim, Algorithm, Angles, Beam, Center, CpuBackend, CudaBackend, Detector, Geometry,
    ReconParams, Tomo, Volume,
};

const N: usize = 128;
const NPROJ: usize = 180;
const NZ: usize = 4;
const Z: usize = 1;

fn setup(cpu: &CpuBackend) -> (Tomo<f32>, Angles) {
    let phantom = sim::shepp2d(N).unwrap();
    let mut stack = Array3::<f32>::zeros((NZ, N, N));
    for z in 0..NZ {
        stack.index_axis_mut(Axis(0), z).assign(&phantom);
    }
    let angles = Angles::uniform(NPROJ, 0.0, std::f32::consts::PI);
    let geom = Geometry::parallel(angles.clone(), N, NZ, 1.0);
    let sino = sim::project(&Volume::new(stack), &geom, cpu).unwrap();
    (sino, angles)
}

fn geom_at(angles: &Angles, center: f32) -> Geometry {
    Geometry {
        angles: angles.clone(),
        center: Center::Scalar(center),
        beam: Beam::Parallel,
        detector: Detector {
            width: N,
            height: NZ,
            pixel_size: 1.0,
        },
    }
}

/// max |a−b| / peak(a) over the reconstruction disk. The FOV edge is excluded on
/// purpose: a centre shift moves where the sampling runs off the detector, so the
/// boundary voxels are legitimately not the same set (~12 % of peak there at any
/// shift). Everything the user reconstructs for lives inside.
fn interior_rel(want: &Array2<f32>, got: &Array2<f32>) -> f32 {
    let c = N as f32 / 2.0;
    let rin = 0.45 * N as f32;
    let peak = want.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(peak > 0.0, "reference slice is all zeros");
    let mut d = 0.0f32;
    for y in 0..N {
        for x in 0..N {
            let rr = ((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt();
            if rr < rin {
                d = d.max((want[[y, x]] - got[[y, x]]).abs());
            }
        }
    }
    d / peak
}

fn recon_slice(sino: &Tomo<f32>, angles: &Angles, center: f32, cuda: &CudaBackend) -> Array2<f32> {
    let vol = recon::recon(
        sino,
        &geom_at(angles, center),
        Algorithm::Fbp,
        &ReconParams::default(),
        cuda,
    )
    .unwrap();
    vol.array.index_axis(Axis(0), Z).to_owned()
}

/// The raw probe is the reconstruction, exactly, wherever the offset from the
/// nominal is a whole number of columns — and that holds for a *fractional*
/// nominal too, which is what lets the sweep anchor anywhere.
#[test]
fn cuda_center_probe_is_exact_on_the_integer_lattice() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (sino, angles) = setup(&cpu);
    let params = ReconParams::default();

    for &nominal in &[64.0f32, 64.25, 63.75, 61.3] {
        let cands: Vec<f32> = (-3..=3).map(|k| nominal - k as f32).collect();
        let probe =
            cuda::center_probe(&sino, &geom_at(&angles, nominal), &params, &cands, Z as i32)
                .unwrap();
        assert_eq!(probe.dim(), (cands.len(), N, N));
        for (i, &c) in cands.iter().enumerate() {
            let want = recon_slice(&sino, &angles, c, &cuda);
            let got = probe.index_axis(Axis(0), i).to_owned();
            let rel = interior_rel(&want, &got);
            eprintln!(
                "nominal {nominal}, c {c} (sh {:+}): rel {rel:e}",
                nominal - c
            );
            assert!(
                rel < 1e-5,
                "nominal {nominal}, centre {c}: integer-shift probe should equal the \
                 reconstruction, got {rel:e} of peak"
            );
        }
    }
}

/// The reason `center_probe_sweep` exists: off the lattice the raw probe is a
/// *smoothed* reconstruction, not the reconstruction. If this ever stops holding,
/// the sweep's grouping is dead weight and should go — so pin the defect too, not
/// just the fix.
#[test]
fn cuda_center_probe_is_smoothed_off_the_lattice() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (sino, angles) = setup(&cpu);
    let params = ReconParams::default();

    let nominal = 64.0f32;
    let cands: Vec<f32> = (-3..=3).map(|k| nominal - k as f32 - 0.5).collect();
    let probe =
        cuda::center_probe(&sino, &geom_at(&angles, nominal), &params, &cands, Z as i32).unwrap();
    let mut worst = 0.0f32;
    for (i, &c) in cands.iter().enumerate() {
        let want = recon_slice(&sino, &angles, c, &cuda);
        let got = probe.index_axis(Axis(0), i).to_owned();
        worst = worst.max(interior_rel(&want, &got));
    }
    eprintln!("half-integer offsets: worst rel {worst:e}");
    assert!(
        worst > 1e-3,
        "half-integer probes agreed with the reconstruction to {worst:e}; if the \
         kernel gained a band-limited shift, center_probe_sweep's per-fraction \
         grouping is no longer needed"
    );
}

/// The sweep's contract: every candidate is exact, at arbitrary spacing —
/// including the sub-pixel grid the raw probe cannot serve.
#[test]
fn cuda_center_probe_sweep_is_exact_at_sub_pixel_spacing() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (sino, angles) = setup(&cpu);
    let params = ReconParams::default();

    // 0.25 px grid straddling several integers: 4 distinct fractions, 13 candidates.
    let cands: Vec<f32> = (0..13).map(|k| 62.0 + k as f32 * 0.25).collect();
    let got = cuda::center_probe_sweep(
        &sino,
        &geom_at(&angles, 64.0), // deliberately not any candidate's anchor
        &params,
        &cands,
        Z as i32,
    )
    .unwrap();
    assert_eq!(got.dim(), (cands.len(), N, N));

    for (i, &c) in cands.iter().enumerate() {
        let want = recon_slice(&sino, &angles, c, &cuda);
        let rel = interior_rel(&want, &got.index_axis(Axis(0), i).to_owned());
        eprintln!("sweep c {c}: rel {rel:e}");
        assert!(
            rel < 1e-5,
            "sweep centre {c} should equal the reconstruction, got {rel:e} of peak"
        );
    }
}

/// Ordering: the returned slices follow `centers`, not the internal grouping.
/// The grouping reorders work behind the caller's back, so a scatter bug here
/// would hand back a correct-looking image for the wrong centre.
#[test]
fn cuda_center_probe_sweep_preserves_candidate_order() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (sino, angles) = setup(&cpu);
    let params = ReconParams::default();

    // Interleaved fractions, so grouping must reorder to do its job.
    let cands = [64.0f32, 62.5, 65.0, 63.5, 61.0];
    let got = cuda::center_probe_sweep(&sino, &geom_at(&angles, 64.0), &params, &cands, Z as i32)
        .unwrap();
    for (i, &c) in cands.iter().enumerate() {
        let want = recon_slice(&sino, &angles, c, &cuda);
        let rel = interior_rel(&want, &got.index_axis(Axis(0), i).to_owned());
        assert!(rel < 1e-5, "slot {i} should hold centre {c}, rel {rel:e}");
    }
}

/// A sweep the raw probe would mis-rank must now rank correctly: the sharpest
/// slice is the sharpest reconstruction, on a sub-pixel grid, regardless of where
/// the caller's nominal happened to sit.
#[test]
fn cuda_center_probe_sweep_argmax_matches_full_recons() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    let cpu = CpuBackend::new();
    let (sino, angles) = setup(&cpu);
    let params = ReconParams::default();

    let focus = |img: &Array2<f32>| -> f64 {
        let c = N as f32 / 2.0;
        let rin = 0.45 * N as f32;
        let (mut acc, mut cnt) = (0.0f64, 0usize);
        for y in 1..N - 1 {
            for x in 1..N - 1 {
                let rr = ((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt();
                if rr >= rin {
                    continue;
                }
                let gx = (img[[y, x + 1]] - img[[y, x - 1]]) as f64;
                let gy = (img[[y + 1, x]] - img[[y - 1, x]]) as f64;
                acc += gx * gx + gy * gy;
                cnt += 1;
            }
        }
        acc / cnt.max(1) as f64
    };
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

    let cands: Vec<f32> = (0..17).map(|k| 62.0 + k as f32 * 0.25).collect();
    // Nominals that put the true optimum at a half-integer offset — exactly the
    // case where the raw probe's argmax slides onto the integer lattice.
    for &nominal in &[64.0f32, 61.5, 65.5] {
        let sweep =
            cuda::center_probe_sweep(&sino, &geom_at(&angles, nominal), &params, &cands, Z as i32)
                .unwrap();
        let sf: Vec<f64> = (0..cands.len())
            .map(|i| focus(&sweep.index_axis(Axis(0), i).to_owned()))
            .collect();
        let rf: Vec<f64> = cands
            .iter()
            .map(|&c| focus(&recon_slice(&sino, &angles, c, &cuda)))
            .collect();
        eprintln!(
            "nominal {nominal}: sweep argmax {:.2}, recon argmax {:.2}",
            cands[argmax(&sf)],
            cands[argmax(&rf)]
        );
        assert_eq!(
            argmax(&sf),
            argmax(&rf),
            "nominal {nominal}: sweep picked {:.2}, full recons picked {:.2}",
            cands[argmax(&sf)],
            cands[argmax(&rf)]
        );
    }
}

/// The probe must not depend on the whole padded stack fitting the device.
///
/// It used to: the padded stack was one `cudaMalloc`, which at 1800×1024×1024
/// with a 4096 pad asks for 30 GB and fails outright — on the very dataset the
/// probe was written to align. Filtering now runs through `lamino_filter_to_host`
/// in angle chunks and the try kernel accumulates each chunk into one output.
/// This forces that path with a tiny memory budget and requires the answer not to
/// move: the candidate images, and therefore the centre the sweep picks, are the
/// same whether the stack took one chunk or many.
#[test]
fn cuda_center_probe_is_unchanged_when_the_stack_is_angle_chunked() {
    let cpu = CpuBackend::new();
    // `cuda::center_probe` selects the device itself; this only asks whether
    // there is one to select.
    if let Err(e) = CudaBackend::new() {
        eprintln!("skipping: no CUDA device ({e})");
        return;
    }
    let (sino, angles) = setup(&cpu);
    let nominal = N as f32 / 2.0;
    let cands: Vec<f32> = (-4..=4).map(|k| nominal + k as f32).collect();
    let params = ReconParams::default();

    // Real budget: `lamino_ncproj` returns nproj → a single angle chunk.
    let single =
        cuda::center_probe(&sino, &geom_at(&angles, nominal), &params, &cands, Z as i32).unwrap();

    // Tiny budget: ncproj < nproj → the accumulate-per-angle-chunk path.
    std::env::set_var("TOMOXIDE_CUDA_MAX_FREE_BYTES", "500000");
    let chunked = cuda::center_probe(&sino, &geom_at(&angles, nominal), &params, &cands, Z as i32);
    std::env::remove_var("TOMOXIDE_CUDA_MAX_FREE_BYTES");
    let chunked = chunked.unwrap();

    assert_eq!(chunked.dim(), single.dim());
    for (i, &c) in cands.iter().enumerate() {
        let a = single.index_axis(Axis(0), i).to_owned();
        let b = chunked.index_axis(Axis(0), i).to_owned();
        // Chunking changes only the float summation order of the angle sum and
        // the cuFFT batch algorithm, so this is a rounding-level bound, not a
        // correlation one.
        let rel = interior_rel(&a, &b);
        assert!(
            rel < 1e-4,
            "candidate {c}: angle-chunked probe differs from single-chunk by {rel:.2e} of peak"
        );
    }
}
