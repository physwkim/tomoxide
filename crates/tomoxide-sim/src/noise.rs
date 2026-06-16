//! Additive noise models for forward-simulated data (tomopy
//! `sim/project.py:110` `add_gaussian`, `:136` `add_poisson`).
//!
//! ## Parity scope: distribution, not bit-stream
//!
//! tomopy draws from numpy's *global* legacy generator
//! (`np.random.randn`, `np.random.poisson`), which is the MT19937 stream
//! plus numpy's polar-method Gaussian and Knuth/PTRS Poisson. We cannot
//! reproduce that bit-stream from Rust, so — unlike the projector-independent
//! image-domain ports (`remove_ring`, `find_center_pc`) that hit true tomopy
//! numeric parity — these are held to **distribution parity**: the samples
//! follow the *same* defined distribution (matched mean / variance /
//! skewness), verified statistically rather than by array Δ. For
//! reproducibility we take an explicit `seed` instead of relying on a global
//! generator.
//!
//! The Poisson sampler ports numpy's algorithm *selection* — Knuth's
//! multiplication method for `λ < 10` and Hörmann's transformed-rejection
//! (PTRS) for `λ ≥ 10` — so the sample *shape* (including skew, which a
//! normal approximation would flatten) matches a true Poisson, not just its
//! first two moments.

use ndarray::{Array2, Array3};
use tomoxide_core::data::{Layout, Tomo};
use tomoxide_core::error::{Error, Result};

/// SplitMix64 — a tiny, fast, seedable PRNG (Steele, Lea & Flood 2014).
///
/// Used only to drive the noise models; its statistical quality is more than
/// enough for sampling noise for robustness tests. Kept self-contained so the
/// sim crate needs no `rand` dependency.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform double on the *open* interval `(0, 1)` — never 0 or 1, so it is
    /// safe to pass to `ln()` in Box–Muller and PTRS.
    fn next_open01(&mut self) -> f64 {
        // 53 random bits, shifted to the centre of each 2^-53 bucket.
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    /// One standard-normal sample via the Box–Muller transform.
    fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_open01();
        let u2 = self.next_open01();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// One Poisson sample with mean `lam` (`lam >= 0`).
    fn poisson(&mut self, lam: f64) -> i64 {
        if lam == 0.0 {
            return 0;
        }
        if lam < 10.0 {
            self.poisson_mult(lam)
        } else {
            self.poisson_ptrs(lam)
        }
    }

    /// Knuth's multiplication method — exact, used for small `lam` (numpy's
    /// `random_poisson_mult`). `exp(-lam)` is well-conditioned here.
    fn poisson_mult(&mut self, lam: f64) -> i64 {
        let enlam = (-lam).exp();
        let mut x = 0i64;
        let mut prod = 1.0;
        loop {
            prod *= self.next_open01();
            if prod > enlam {
                x += 1;
            } else {
                return x;
            }
        }
    }

    /// Hörmann's transformed-rejection sampler (PTRS), used for `lam >= 10`
    /// (numpy's `random_poisson_ptrs`). Exact and O(1) per accepted sample.
    fn poisson_ptrs(&mut self, lam: f64) -> i64 {
        let slam = lam.sqrt();
        let loglam = lam.ln();
        let b = 0.931 + 2.53 * slam;
        let a = -0.059 + 0.02483 * b;
        let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        loop {
            let u = self.next_open01() - 0.5;
            let v = self.next_open01();
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + lam + 0.43).floor() as i64;
            if us >= 0.07 && v <= vr {
                return k;
            }
            if k < 0 || (us < 0.013 && v > us) {
                continue;
            }
            if (v.ln() + inv_alpha.ln() - (a / (us * us) + b).ln())
                <= -lam + (k as f64) * loglam - loggam((k + 1) as f64)
            {
                return k;
            }
        }
    }
}

/// `log Γ(x)` for `x > 0` via the Stirling series (numpy's `loggam`). Only
/// needed by the PTRS acceptance test, always with `x = k + 1 >= 1`.
fn loggam(x: f64) -> f64 {
    const A: [f64; 10] = [
        8.333_333_333_333_333e-2,
        -2.777_777_777_777_778e-3,
        7.936_507_936_507_937e-4,
        -5.952_380_952_380_953e-4,
        8.417_508_417_508_417e-4,
        -1.917_526_917_526_918e-3,
        6.410_256_410_256_41e-3,
        -2.955_065_359_477_124e-2,
        1.796_443_723_688_307e-1,
        -1.392_432_216_905_9e0,
    ];
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    let mut x0 = x;
    let mut n = 0i64;
    if x <= 7.0 {
        n = (7.0 - x) as i64;
        x0 = x + n as f64;
    }
    let x2 = 1.0 / (x0 * x0);
    let mut gl0 = A[9];
    for k in (0..9).rev() {
        gl0 = gl0 * x2 + A[k];
    }
    let mut gl = gl0 / x0 + 0.5 * std::f64::consts::TAU.ln() + (x0 - 0.5) * x0.ln() - x0;
    if x <= 7.0 {
        let mut xx = x0;
        for _ in 0..n {
            gl -= (xx - 1.0).ln();
            xx -= 1.0;
        }
    }
    gl
}

/// Add Gaussian noise in place: `data += std * N(0, 1) + mean`, element-wise
/// (tomopy `sim/project.py:110`).
///
/// When `std` is `None` it defaults to `data.max() * 0.05`, matching tomopy's
/// `std=None` branch. `seed` makes the draw reproducible (tomopy uses the
/// global numpy generator instead). See the module docs for the
/// distribution-parity scope.
pub fn add_gaussian(data: &mut Tomo<f32>, mean: f32, std: Option<f32>, seed: u64) -> Result<()> {
    let std = std.unwrap_or_else(|| array_max(&data.array) * 0.05);
    let (std, mean) = (std as f64, mean as f64);
    let mut rng = SplitMix64::new(seed);
    for v in data.array.iter_mut() {
        *v += (std * rng.standard_normal() + mean) as f32;
    }
    Ok(())
}

/// Replace each element with a Poisson draw whose mean is that element
/// (tomopy `sim/project.py:136` → `np.random.poisson(tomo)`).
///
/// Every value must be `>= 0` (a Poisson mean); a negative entry yields
/// [`Error::InvalidParam`] *before any element is mutated*, mirroring numpy's
/// rejection of negative `lam`. `seed` makes the draw reproducible. See the
/// module docs for the distribution-parity scope.
pub fn add_poisson(data: &mut Tomo<f32>, seed: u64) -> Result<()> {
    // Validate up front so the operation is all-or-nothing.
    if let Some(neg) = data.array.iter().copied().find(|&v| v < 0.0) {
        return Err(Error::InvalidParam(format!(
            "add_poisson: negative intensity {neg} (a Poisson mean must be >= 0)"
        )));
    }
    let mut rng = SplitMix64::new(seed);
    for v in data.array.iter_mut() {
        *v = rng.poisson(*v as f64) as f32;
    }
    Ok(())
}

/// Multiply each detector pixel by a fixed per-pixel sensitivity drawn from
/// `N(1, std)`, modelling the inconsistent pixel response that produces ring
/// artifacts (tomopy `sim/project.py:153` `add_rings`).
///
/// The sensitivity is sampled once per detector `(row, col)` and held constant
/// across every projection angle — so a pixel that consistently reads high or
/// low traces a ring after reconstruction. This is the structural difference
/// from [`add_gaussian`], whose noise is independent per element. It holds in
/// either [`Layout`]: the sensitivity is keyed on the detector `(row, col)`,
/// never the angle axis (tomopy broadcasts a `(1, ny, nx)` array over `theta`).
/// `seed` makes the draw reproducible (tomopy uses the global numpy generator).
/// See the module docs for the distribution-parity scope.
pub fn add_rings(data: &mut Tomo<f32>, std: f32, seed: u64) -> Result<()> {
    let (n_rows, n_cols) = (data.n_rows(), data.n_cols());
    let std = std as f64;
    let mut rng = SplitMix64::new(seed);
    // One sensitivity per detector pixel, drawn in (row, col) C-order to mirror
    // numpy's `size=(1, ny, nx)` traversal.
    let mut sens = Array2::<f64>::zeros((n_rows, n_cols));
    for s in sens.iter_mut() {
        *s = 1.0 + std * rng.standard_normal();
    }
    // numpy promotes f32 * f64 → f64; we store the product back as f32.
    let layout = data.layout;
    for ((a0, a1, col), v) in data.array.indexed_iter_mut() {
        let row = match layout {
            Layout::Projection => a1, // [angle, row, col]
            Layout::Sinogram => a0,   // [row, angle, col]
        };
        *v = (*v as f64 * sens[[row, col]]) as f32;
    }
    Ok(())
}

/// Saturate a random fraction `f` of elements to `sat`, modelling stray X-rays
/// that flare individual pixels (tomopy `sim/project.py:211` `add_zingers`).
///
/// Each element is independently saturated with probability `f` (tomopy draws
/// `U(0, 1) <= f`). `f <= 0` saturates nothing and `f >= 1` saturates
/// everything, matching numpy's comparison. `seed` makes the draw reproducible.
/// See the module docs for the distribution-parity scope.
pub fn add_zingers(data: &mut Tomo<f32>, f: f32, sat: f32, seed: u64) -> Result<()> {
    let f = f as f64;
    let mut rng = SplitMix64::new(seed);
    for v in data.array.iter_mut() {
        if rng.next_open01() <= f {
            *v = sat;
        }
    }
    Ok(())
}

/// `data.max()` (tomopy's `tomo.max()`); `f32::NEG_INFINITY` for an empty array
/// (unused there — the mutate loop is a no-op).
fn array_max(a: &Array3<f32>) -> f32 {
    a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}
