//! Every parallel-beam projector against the one edge rule they share: a
//! detector coordinate `t` taps samples `⌊t⌋` (weight `1 − frac`) and `⌊t⌋ + 1`
//! (weight `frac`), and a tap outside the detector is zero. `ref_project` states
//! that rule in f64; the angles put voxels on the last column (θ = 0, π/2), in
//! `(n − 1, n)`, and in `(−1, 0)`, and `nz = 260` puts detector rows past 256,
//! where `r − 1e-5` rounds back to `r` in f32.
//!
//! CUDA cases need a device (skipped otherwise); wgpu cases need an adapter.

use ndarray::Array3;
use tomoxide::backend::RayProject;
use tomoxide::{
    Angles, CpuBackend, FilteredBackproject, ForwardProject, Geometry, Layout, Tomo, Volume,
};

const N: usize = 8;

fn angles() -> Vec<f32> {
    vec![
        0.0,
        0.1,
        1.0,
        std::f32::consts::FRAC_PI_2,
        2.5,
        std::f32::consts::PI - 0.05,
    ]
}

fn fill(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The detector taps of pixel `(ix, iy)` at angle `a`, out-of-detector taps dropped.
fn taps(ix: usize, iy: usize, a: f32, n: usize) -> Vec<(usize, f64)> {
    let h = (n / 2) as f64;
    let (s, c) = (a as f64).sin_cos();
    let t = (ix as f64 - h) * c + (iy as f64 - h) * s + h;
    let t0 = t.floor();
    let i0 = t0 as i64;
    let f = t - t0;
    [(i0, 1.0 - f), (i0 + 1, f)]
        .into_iter()
        .filter(|&(i, _)| i >= 0 && (i as usize) < n)
        .map(|(i, w)| (i as usize, w))
        .collect()
}

fn ref_project(vol: &Array3<f32>, ang: &[f32]) -> Array3<f32> {
    let (nz, n, _) = vol.dim();
    let mut g = Array3::<f64>::zeros((nz, ang.len(), n));
    for z in 0..nz {
        for iy in 0..n {
            for ix in 0..n {
                for (t, &a) in ang.iter().enumerate() {
                    for (k, w) in taps(ix, iy, a, n) {
                        g[[z, t, k]] += vol[[z, iy, ix]] as f64 * w;
                    }
                }
            }
        }
    }
    g.mapv(|x| x as f32)
}

fn ref_backproject(sino: &Array3<f32>, ang: &[f32]) -> Array3<f32> {
    let (nz, _, n) = sino.dim();
    let mut f = Array3::<f64>::zeros((nz, n, n));
    for z in 0..nz {
        for iy in 0..n {
            for ix in 0..n {
                for (t, &a) in ang.iter().enumerate() {
                    for (k, w) in taps(ix, iy, a, n) {
                        f[[z, iy, ix]] += sino[[z, t, k]] as f64 * w;
                    }
                }
            }
        }
    }
    f.mapv(|x| x as f32)
}

fn assert_close(what: &str, got: &Array3<f32>, want: &Array3<f32>) {
    assert_eq!(got.dim(), want.dim(), "{what}: shape");
    let scale = want.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let (idx, err) = got
        .indexed_iter()
        .map(|(i, &x)| (i, (x - want[i]).abs()))
        .fold(((0, 0, 0), 0.0f32), |m, e| if e.1 > m.1 { e } else { m });
    assert!(
        err <= 1e-4 * scale,
        "{what}: max |err| {err:.3e} at {idx:?} (got {}, want {}, scale {scale:.3e})",
        got[idx],
        want[idx]
    );
}

fn check_parallel(label: &str, fwd: &dyn ForwardProject, bp: &dyn FilteredBackproject) {
    let ang = angles();
    for nz in [1usize, 3, 260] {
        let geom = Geometry::parallel(Angles(ang.clone()), N, nz, 1.0);
        let vol = Array3::from_shape_vec((nz, N, N), fill(nz as u64, nz * N * N)).unwrap();
        let mut g = Tomo::new(Array3::zeros((nz, ang.len(), N)), Layout::Sinogram);
        fwd.project(&Volume::new(vol.clone()), &geom, &mut g)
            .unwrap();
        assert_close(
            &format!("{label} forward nz={nz}"),
            &g.array,
            &ref_project(&vol, &ang),
        );

        let sino =
            Array3::from_shape_vec((nz, ang.len(), N), fill(7 + nz as u64, nz * ang.len() * N))
                .unwrap();
        let mut f = Volume::new(Array3::zeros((nz, N, N)));
        bp.backproject(&Tomo::new(sino.clone(), Layout::Sinogram), &geom, &mut f)
            .unwrap();
        assert_close(
            &format!("{label} back nz={nz}"),
            &f.array,
            &ref_backproject(&sino, &ang),
        );
    }
}

/// Infinite samples on both edge columns: a tap that exists multiplies its
/// sample even at weight 0 (∞·0 = NaN), a tap off the detector adds nothing.
/// θ = 0 puts every voxel on an integer column, so the weights are exactly 1
/// and 0 on every backend and the NaN/∞ pattern is the rule's, not rounding's.
fn check_infinite_edges(label: &str, bp: &dyn FilteredBackproject) {
    let ang = vec![0.0f32];
    for nz in [1usize, 3] {
        let geom = Geometry::parallel(Angles(ang.clone()), N, nz, 1.0);
        let mut sino = Array3::from_shape_vec((nz, 1, N), fill(11 + nz as u64, nz * N)).unwrap();
        for z in 0..nz {
            sino[[z, 0, 0]] = f32::INFINITY;
            sino[[z, 0, N - 1]] = f32::NEG_INFINITY;
        }
        let mut f = Volume::new(Array3::zeros((nz, N, N)));
        bp.backproject(&Tomo::new(sino.clone(), Layout::Sinogram), &geom, &mut f)
            .unwrap();
        for (i, &want) in ref_backproject(&sino, &ang).indexed_iter() {
            let got = f.array[i];
            let same = if want.is_finite() {
                (got - want).abs() <= 1e-4
            } else {
                got.is_nan() == want.is_nan() && (want.is_nan() || got == want)
            };
            assert!(same, "{label} nz={nz} voxel {i:?}: got {got}, want {want}");
        }
    }
}

#[test]
fn cpu_projectors_follow_the_edge_rule() {
    let cpu = CpuBackend::new();
    check_parallel("cpu", &cpu, &cpu);
    check_infinite_edges("cpu", &cpu);
}

/// The ART/BART rows are the forward projector's rows.
#[test]
fn cpu_ray_rows_follow_the_edge_rule() {
    let ang = angles();
    let geom = Geometry::parallel(Angles(ang.clone()), N, 1, 1.0);
    let rows = CpuBackend::new().ray_rows(&geom, N).unwrap();
    for (t, &a) in ang.iter().enumerate() {
        let mut want = vec![vec![0.0f64; N * N]; N];
        for iy in 0..N {
            for ix in 0..N {
                for (k, w) in taps(ix, iy, a, N) {
                    want[k][iy * N + ix] += w;
                }
            }
        }
        for (k, row) in rows[t].iter().enumerate() {
            let mut got = vec![0.0f64; N * N];
            for (&p, &w) in row.pixels.iter().zip(&row.weights) {
                got[p as usize] += w as f64;
            }
            for (p, (&x, &y)) in got.iter().zip(&want[k]).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-5,
                    "ray row angle {t} column {k} pixel {p}: got {x}, want {y}"
                );
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_projectors_follow_the_edge_rule() {
    let cuda = match tomoxide::CudaBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping CUDA test: {e}");
            return;
        }
    };
    check_parallel("cuda", &cuda, &cuda);
    check_infinite_edges("cuda", &cuda);
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn wgpu_projectors_follow_the_edge_rule() {
    let gpu = match tomoxide::wgpu::WgpuBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping wgpu test: {e}");
            return;
        }
    };
    check_parallel("wgpu", &gpu, &gpu);
    check_infinite_edges("wgpu", &gpu);
}
