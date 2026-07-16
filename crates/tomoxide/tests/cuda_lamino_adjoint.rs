//! Verifies the tilted (laminography) CUDA projector pair is a true adjoint:
//! `⟨A f, g⟩ == ⟨f, Aᵀ g⟩` where `A` = tilted forward projection
//! (`vol [rh,n,n] → sino [nz,nproj,n]`, `forwardprojection_ker` with phi, sz=0,
//! ncz=rh) and `Aᵀ` = tilted back-projection (`cfunc_linerec` with the same
//! phi/rh/nz). This is the foundational primitive iterative laminography needs —
//! the generic SIRT/CGLS/TV solvers only converge to the physical μ when {A, Aᵀ}
//! is an exact transpose. `rh != nz` (the tilt raises the recon height above the
//! detector-row count), which exercises the `sz`/`ncz` generalization.
//!
//! Own test binary (touches CUDA device state) per the suite convention.
#![cfg(feature = "cuda")]

use tomoxide::{
    Angles, Beam, Center, CudaBackend, Detector, FilteredBackproject, ForwardProject, Geometry,
    Layout, Tomo, Volume,
};

// Deterministic pseudo-random fill in [-1, 1] (xorshift; no rand dep).
fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

#[test]
fn cuda_lamino_forward_backward_are_adjoint() {
    let cuda = match CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };

    let (n, nproj, nz, rh) = (64usize, 90usize, 32usize, 48usize);
    let lamino_angle_deg = 20.0f32;
    let phi = std::f32::consts::FRAC_PI_2 + lamino_angle_deg * std::f32::consts::PI / 180.0;
    let angles = Angles::uniform(nproj, 0.0, std::f32::consts::PI);
    let geom = Geometry {
        angles,
        center: Center::Scalar(n as f32 / 2.0),
        beam: Beam::Laminography { phi },
        detector: Detector {
            width: n,
            height: nz,
            pixel_size: 1.0,
        },
    };

    // Random volume f [rh,n,n] and random sinogram g [nz,nproj,n].
    let f = Volume::new(ndarray::Array3::from_shape_vec((rh, n, n), fill(1, rh * n * n)).unwrap());
    let g_arr = ndarray::Array3::from_shape_vec((nz, nproj, n), fill(2, nz * nproj * n)).unwrap();
    let g = Tomo::new(g_arr, Layout::Sinogram);

    // A f  (forward): [nz, nproj, n].
    let mut af = Tomo::new(ndarray::Array3::zeros((nz, nproj, n)), Layout::Sinogram);
    cuda.project(&f, &geom, &mut af).unwrap();

    // Aᵀ g (back-projection): [rh, n, n].
    let mut atg = Volume::new(ndarray::Array3::zeros((rh, n, n)));
    cuda.backproject(&g, &geom, &mut atg).unwrap();

    let lhs = dot(af.array.as_slice().unwrap(), g.array.as_slice().unwrap());
    let rhs = dot(f.array.as_slice().unwrap(), atg.array.as_slice().unwrap());
    let rel = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-12);
    eprintln!("lamino adjoint: ⟨Af,g⟩ = {lhs:.6}, ⟨f,Aᵀg⟩ = {rhs:.6}, rel = {rel:.2e}");
    assert!(
        rel < 1e-4,
        "tilted forward/back not adjoint: rel = {rel:.2e}"
    );
}
