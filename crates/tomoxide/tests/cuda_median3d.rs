//! CUDA 3D median: against a host sort of each voxel's cube, with and
//! without a mask.
#![cfg(feature = "cuda")]

use ndarray::Array3;
use tomoxide::cuda::median3d;
use tomoxide::CudaBackend;

fn fold(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    while i < 0 || i >= n {
        i = if i < 0 { -i - 1 } else { 2 * n - i - 1 };
    }
    i as usize
}

fn host(v: &Array3<f32>, mask: Option<&Array3<u8>>, size: usize) -> Array3<f32> {
    let (nz, ny, nx) = v.dim();
    let h = (size / 2) as isize;
    Array3::from_shape_fn((nz, ny, nx), |(z, y, x)| {
        if mask.is_some_and(|m| m[(z, y, x)] == 0) {
            return f32::NAN;
        }
        let mut near = Vec::new();
        for dz in -h..=h {
            for dy in -h..=h {
                for dx in -h..=h {
                    let at = (
                        fold(z as isize + dz, nz),
                        fold(y as isize + dy, ny),
                        fold(x as isize + dx, nx),
                    );
                    if mask.is_none_or(|m| m[at] != 0) {
                        near.push(v[at]);
                    }
                }
            }
        }
        near.sort_by(f32::total_cmp);
        (near[(near.len() - 1) / 2] + near[near.len() / 2]) * 0.5
    })
}

#[test]
fn median3d_is_the_sorted_cubes_middle_among_the_masks_voxels() {
    if let Err(e) = CudaBackend::new() {
        eprintln!("skipping CUDA test: {e}");
        return;
    }
    // A volume narrower than the widest cube on one axis, so the fold repeats.
    let dim = (3usize, 17, 23);
    let v = Array3::from_shape_fn(dim, |(z, y, x)| {
        ((z * 7919 + y * 104729 + x * 1299709) % 1009) as f32 * 0.37 - 150.0
    });
    let mask = Array3::from_shape_fn(dim, |(z, y, x)| u8::from((z * 5 + y * 3 + x * 11) % 7 < 3));
    for size in [1usize, 3, 5, 9] {
        for m in [None, Some(&mask)] {
            let got = median3d(v.view(), m.map(|m| m.view()), size).unwrap();
            let want = host(&v, m, size);
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                    "size {size} mask {}: {g} vs {w}",
                    m.is_some()
                );
            }
        }
    }
    assert!(median3d(v.view(), None, 4).is_err());
    assert!(median3d(v.view(), None, 11).is_err());
}
