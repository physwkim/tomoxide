//! At zero tilt laminography is parallel-beam tomography stacked along z, so
//! tomopy's parallel projector pins the in-plane orientation of the Fourier
//! laminography pair: `lamino` must give back the phantom and `lamino_project`
//! its sinogram, neither flipped in y. A round trip through the pair alone
//! cannot see a flip both sides share.

use ndarray::{Array2, Array3, Axis};
use tomoxide::{recon, sim, Angles, CpuBackend, FilterName, Geometry, Layout, Volume};

const N: usize = 64;
const NZ: usize = 4;
const NPROJ: usize = 90;

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x - ma, y - mb);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx * syy).sqrt()
}

/// Shepp–Logan is nearly symmetric in y; a bright disk in its top half is not.
fn phantom() -> Array2<f32> {
    let mut p = sim::shepp2d(N).unwrap();
    for ((y, x), v) in p.indexed_iter_mut() {
        if (y as f32 - 14.0).powi(2) + (x as f32 - 40.0).powi(2) < 25.0 {
            *v += 1.0;
        }
    }
    p
}

fn flipud(a: &Array2<f32>) -> Array2<f32> {
    let mut f = a.clone();
    f.invert_axis(Axis(0));
    f.as_standard_layout().to_owned()
}

fn stack(slice: &Array2<f32>) -> Array3<f32> {
    let mut s = Array3::<f32>::zeros((NZ, N, N));
    for mut z in s.axis_iter_mut(Axis(0)) {
        z.assign(slice);
    }
    s
}

fn theta() -> Vec<f32> {
    Angles::uniform(NPROJ, 0.0, std::f32::consts::PI).0
}

#[test]
fn lamino_at_zero_tilt_reconstructs_the_phantom_unflipped() {
    let cpu = CpuBackend::new();
    let phantom = phantom();
    let geom = Geometry::parallel(Angles(theta()), N, NZ, 1.0);
    let sino = sim::project(&Volume::new(stack(&phantom)), &geom, &cpu).unwrap();
    let proj = sino.to_layout(Layout::Projection);
    let vol = recon::lamino::lamino(
        proj.array.as_slice().unwrap(),
        &theta(),
        0.0,
        N,
        NZ,
        FilterName::Ramp,
        &cpu,
    )
    .unwrap();
    let mid = &vol[NZ / 2 * N * N..(NZ / 2 + 1) * N * N];
    let (r, r_flip) = (
        pearson(mid, phantom.as_slice().unwrap()),
        pearson(mid, flipud(&phantom).as_slice().unwrap()),
    );
    eprintln!("lamino vs phantom r = {r:.3}, vs its y-flip r = {r_flip:.3}");
    assert!(
        r > 0.9 && r > r_flip + 0.3,
        "lamino reconstructs the phantom at r = {r:.3}, its y-flip at {r_flip:.3}"
    );
}

#[test]
fn lamino_project_at_zero_tilt_is_the_parallel_projection() {
    let cpu = CpuBackend::new();
    let phantom = phantom();
    let geom = Geometry::parallel(Angles(theta()), N, NZ, 1.0);
    let parallel = |p: &Array2<f32>| {
        let s = sim::project(&Volume::new(stack(p)), &geom, &cpu).unwrap();
        s.to_layout(Layout::Projection)
            .array
            .into_raw_vec_and_offset()
            .0
    };
    let vol = stack(&phantom);
    let lam =
        recon::lamino::lamino_project(vol.as_slice().unwrap(), &theta(), 0.0, N, NZ, &cpu).unwrap();
    let (r, r_flip) = (
        pearson(&lam, &parallel(&phantom)),
        pearson(&lam, &parallel(&flipud(&phantom))),
    );
    eprintln!("lamino_project vs parallel r = {r:.3}, vs the y-flip's r = {r_flip:.3}");
    assert!(
        r > 0.9 && r > r_flip + 0.3,
        "lamino_project matches the phantom's projections at r = {r:.3}, its \
         y-flip's at {r_flip:.3}"
    );
}
