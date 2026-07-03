//! Absolute-amplitude pin for the analytic reconstructors (FBP, gridrec,
//! fourierrec).
//!
//! Companion to `tests/iterative_amplitude.rs`. The cross-method ratio pins
//! (`fourierrec_parity::{gridrec,fourierrec}_matches_fbp_amplitude`) only fix
//! the methods *relative to each other* — a shared scale error moves them all
//! together and passes. This pins the *absolute* scale against
//! operator-independent ground truth: the analytic line integral of a unit disk
//! is `p(θ, s) = 2√(R² − s²)`, and a correct FBP inverts it back to the disk's
//! own value μ = 1.0 in the interior.
//!
//! This is what the `make_fbp_filter` ramp carries: the physical `|ω|` filter
//! (peak 0.5 at Nyquist) reconstructs μ; tomopy/tomocupy's doubled ramp (peak 1)
//! would land every method at 2×μ and fail this by a factor of two. The gridrec
//! `ramp_scale` (`π/nang`, no empirical ×2) is pinned on the same footing.

use ndarray::Array3;
use tomoxide::data::{Layout, Tomo};
use tomoxide::recon::recon;
use tomoxide::{Algorithm, Angles, CpuBackend, FilterName, Geometry, ReconParams, Volume};

const N: usize = 128;
const NANG: usize = 180;
const RADIUS: f32 = 0.3 * N as f32;

/// Exact line integrals of the centred unit disk: `p(θ, s) = 2√(R² − s²)`,
/// identical for every angle, `s` measured from the axis at column n/2.
fn analytic_disk_sino() -> Tomo<f32> {
    let c = N as f32 / 2.0;
    let mut s = Array3::<f32>::zeros((1, NANG, N));
    for ia in 0..NANG {
        for ix in 0..N {
            let d = ix as f32 - c;
            if d.abs() < RADIUS {
                s[[0, ia, ix]] = 2.0 * (RADIUS * RADIUS - d * d).sqrt();
            }
        }
    }
    Tomo::new(s, Layout::Sinogram)
}

fn geom() -> Geometry {
    Geometry::parallel(Angles::uniform(NANG, 0.0, std::f32::consts::PI), N, 1, 1.0)
}

/// Mean over the disk core (radius < R/2), well inside any edge blur.
fn core_mean(vol: &Volume<f32>) -> f32 {
    let c = N as f32 / 2.0;
    let (mut sum, mut cnt) = (0.0f64, 0u32);
    for iy in 0..N {
        for ix in 0..N {
            let (dx, dy) = (ix as f32 - c, iy as f32 - c);
            if dx * dx + dy * dy < (RADIUS / 2.0) * (RADIUS / 2.0) {
                sum += vol.array[[0, iy, ix]] as f64;
                cnt += 1;
            }
        }
    }
    (sum / cnt as f64) as f32
}

/// FBP, gridrec and fourierrec each reconstruct the unit disk to μ ≈ 1.0 (not
/// 2.0, the doubled-ramp scale). The ramp filter is the sharp reference; the
/// small shortfall is the filter/deapodization shape, not scale.
#[test]
fn analytic_recon_is_physical_mu() {
    let sino = analytic_disk_sino();
    let g = geom();
    for alg in [Algorithm::Fbp, Algorithm::Gridrec, Algorithm::Fourierrec] {
        let params = ReconParams {
            num_gridx: Some(N),
            filter_name: FilterName::Ramp,
            ..Default::default()
        };
        let vol = recon(&sino, &g, alg, &params, &CpuBackend::new()).unwrap();
        let m = core_mean(&vol);
        assert!(
            (m - 1.0).abs() < 0.05,
            "{alg:?} core mean {m} ≠ 1.0 (physical μ) — a ramp/normalization scale regressed"
        );
    }
}
