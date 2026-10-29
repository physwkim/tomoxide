//! `estimate_tilt_flow` on projections whose only motion is the laminography
//! shear: projection `j` is a texture displaced vertically by
//! `sin τ·(x − x_c)·θ_j`, so the flow between any two is a plane in `x` with
//! slope `sin τ·dθ` and zero at `x_c`.

use ndarray::Array3;
use tomoxide::data::{Layout, Tomo};
use tomoxide::recon::tilt_flow::estimate_tilt_flow;
use tomoxide::{CpuBackend, Error};

const NY: usize = 256;
const NX: usize = 384;

/// A sum of sinusoids at pseudo-random frequencies (0.02–0.15 cycles/px) and
/// phases: broadband enough to correlate in every block, and exact at the
/// fractional rows a shear asks for.
fn waves() -> Vec<(f64, f64, f64)> {
    let mut s = 0x9e37_79b9u32;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s as f64 / u32::MAX as f64
    };
    (0..32)
        .map(|_| {
            let (f, a, ph) = (0.02 + 0.13 * next(), std::f64::consts::TAU * next(), next());
            (f * a.cos(), f * a.sin(), std::f64::consts::TAU * ph)
        })
        .collect()
}

fn sheared(theta: &[f32], tilt_deg: f64, axis: f64) -> Tomo<f32> {
    use std::f64::consts::TAU;
    let s = tilt_deg.to_radians().sin();
    let mut a = vec![0f32; theta.len() * NY * NX];
    // Down a column every wave advances by a fixed phasor, so a column is one
    // sin_cos per wave and the rest multiplications.
    for (fx, fy, ph) in waves() {
        let (si, ci) = (TAU * fy).sin_cos();
        for (j, &t) in theta.iter().enumerate() {
            let frame = &mut a[j * NY * NX..(j + 1) * NY * NX];
            for x in 0..NX {
                let y0 = -s * (x as f64 - axis) * f64::from(t);
                let (mut sn, mut cs) = (TAU * (fx * x as f64 + fy * y0) + ph).sin_cos();
                for v in frame[x..].iter_mut().step_by(NX) {
                    *v += cs as f32;
                    (cs, sn) = (cs * ci - sn * si, sn * ci + cs * si);
                }
            }
        }
    }
    let a = Array3::from_shape_vec((theta.len(), NY, NX), a).unwrap();
    Tomo::new(a, Layout::Projection)
}

/// 0.9° steps put the frame-edge sideways motion of a 4-step gap at the
/// correlation's ±12 px, so the pairs are 2, 3 and 4 steps apart — the gaps
/// the 261003 bin-2 scan uses.
fn angles(n: usize) -> Vec<f32> {
    (0..n).map(|j| (j as f32 * 0.9).to_radians()).collect()
}

#[test]
fn tilt_flow_recovers_the_tilt_and_axis_of_a_pure_shear() {
    let cpu = CpuBackend::new();
    // Starts at every 33/8th projection reuse frames across pairs; the
    // estimator does not care, and the fixture is half the size.
    let theta = angles(33);
    for (tilt, axis) in [(20.0, 170.0), (35.0, 230.0)] {
        let r = estimate_tilt_flow(&sheared(&theta, tilt, axis), &theta, &cpu).unwrap();
        eprintln!(
            "truth {tilt}° @ {axis}: {:.2} ± {:.2}°, axis {:.1} ± {:.1}, roll {:.2}°, {} pairs",
            r.tilt_deg,
            r.tilt_sigma_deg,
            r.axis_col,
            r.axis_col_sigma,
            r.roll_deg,
            r.pairs.len()
        );
        let gaps: Vec<usize> = r.pairs.iter().take(3).map(|p| p.gap).collect();
        assert_eq!(gaps, [2, 3, 4]);
        assert_eq!(r.pairs.len(), 24);
        assert!((r.tilt_deg - tilt).abs() < 0.5, "tilt {}", r.tilt_deg);
        assert!((r.axis_col - axis).abs() < 3.0, "axis {}", r.axis_col);
        assert!(r.roll_deg.abs() < 1.0, "roll {}", r.roll_deg);
    }
}

#[test]
fn tilt_flow_refuses_small_frames_and_mismatched_angles() {
    let cpu = CpuBackend::new();
    let theta = angles(8);
    let small = Tomo::new(Array3::<f32>::zeros((8, 255, NX)), Layout::Projection);
    assert!(matches!(
        estimate_tilt_flow(&small, &theta, &cpu),
        Err(Error::InvalidParam(_))
    ));
    let tomo = Tomo::new(Array3::<f32>::zeros((8, NY, NX)), Layout::Projection);
    assert!(matches!(
        estimate_tilt_flow(&tomo, &theta[..7], &cpu),
        Err(Error::ShapeMismatch { .. })
    ));
}
