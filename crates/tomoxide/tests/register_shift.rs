//! `register_shift` — the phase-correlation registration behind `find_center_pc`,
//! public for the drift check. Its tomopy parity is held by `center_pc_parity`;
//! these pin the sign convention and the sub-pixel step a caller relies on.

use ndarray::Array2;
use tomoxide::backend::Backend;
use tomoxide::dtype::Complex32;
use tomoxide::recon::center::register_shift;
use tomoxide::{CpuBackend, Error};

const NY: usize = 64;
const NX: usize = 96;

/// Periodic white noise displaced by `(dy, dx)` through the Fourier shift
/// theorem, so a sub-pixel displacement is exact. Phase correlation weights
/// every frequency equally, so the test image has to fill the spectrum the way
/// a projection does; a few smooth blobs leave it to round-off and fail.
fn texture(dy: f64, dx: f64) -> Array2<f32> {
    let mut s = 0x2545_f491u32;
    let mut buf: Vec<Complex32> = (0..NY * NX)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            Complex32::new(s as f32 / u32::MAX as f32, 0.0)
        })
        .collect();
    let cpu = CpuBackend::new();
    let fft = cpu.fft().unwrap();
    fft.fft_2d(&mut buf, NY, NX, 1, false).unwrap();
    let freq = |k: usize, n: usize| {
        let k = if k <= n / 2 {
            k as f64
        } else {
            k as f64 - n as f64
        };
        k / n as f64
    };
    for ky in 0..NY {
        for kx in 0..NX {
            let ph = -std::f64::consts::TAU * (freq(ky, NY) * dy + freq(kx, NX) * dx);
            let (sn, cs) = ph.sin_cos();
            buf[ky * NX + kx] *= Complex32::new(cs as f32, sn as f32);
        }
    }
    fft.fft_2d(&mut buf, NY, NX, 1, true).unwrap();
    Array2::from_shape_fn((NY, NX), |(y, x)| buf[y * NX + x].re)
}

/// `moving` displaced by `(dy, dx)` registers back with `(−dy, −dx)`.
#[test]
fn register_shift_returns_the_shift_that_moves_moving_onto_reference() {
    let cpu = CpuBackend::new();
    let reference = texture(0.0, 0.0);
    for (dy, dx) in [(3.0, -5.0), (-2.0, 7.0)] {
        let (r, c) = register_shift(&reference, &texture(dy, dx), &cpu, 1.0).unwrap();
        assert_eq!((r, c), (-dy, -dx), "displaced by ({dy}, {dx})");
    }
}

#[test]
fn register_shift_resolves_to_one_over_upsample() {
    let cpu = CpuBackend::new();
    let (r, c) = register_shift(&texture(0.0, 0.0), &texture(1.25, -2.5), &cpu, 4.0).unwrap();
    assert!(
        (r + 1.25).abs() <= 0.125 && (c - 2.5).abs() <= 0.125,
        "got ({r}, {c}), want (-1.25, 2.5)"
    );
}

#[test]
fn register_shift_refuses_mismatched_shapes_and_upsample_below_one() {
    let cpu = CpuBackend::new();
    let a = texture(0.0, 0.0);
    let b = Array2::<f32>::zeros((NY, NX - 1));
    assert!(matches!(
        register_shift(&a, &b, &cpu, 1.0),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        register_shift(&a, &a, &cpu, 0.5),
        Err(Error::InvalidParam(_))
    ));
}
