//! Minimal self-contained complex FFT, f64, no external dependency.
//!
//! An iterative radix-2 Cooley–Tukey core ([`fft_pow2`]) handles power-of-two
//! lengths; [`fft`]/[`ifft`] wrap it with Bluestein's chirp-z algorithm so any
//! length (including primes) transforms in `O(n log n)`. This matches numpy
//! `fft`/`ifft` to f64 round-off — far below the f32 floor the callers cast to.
//!
//! The consumers are the Fourier-Wavelet stripe damping, which needs
//! `real(ifft(fft(col) · d))` for arbitrary-length detail-band columns (exposed
//! as [`filter_real_column`]), and the fitting-based stripe filter, which needs
//! the 2-D `real(ifft2(fft2(mat) · win))` (exposed as [`filter_real_2d`]). Both
//! keep the complex type private.

use std::f64::consts::PI;

use ndarray::Array2;

/// A complex number in f64 (kept private; callers use [`filter_real_column`]).
#[derive(Clone, Copy)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    const ZERO: Cx = Cx { re: 0.0, im: 0.0 };

    #[inline]
    fn new(re: f64, im: f64) -> Self {
        Cx { re, im }
    }
    #[inline]
    fn add(self, o: Cx) -> Cx {
        Cx::new(self.re + o.re, self.im + o.im)
    }
    #[inline]
    fn sub(self, o: Cx) -> Cx {
        Cx::new(self.re - o.re, self.im - o.im)
    }
    #[inline]
    fn mul(self, o: Cx) -> Cx {
        Cx::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    #[inline]
    fn conj(self) -> Cx {
        Cx::new(self.re, -self.im)
    }
    #[inline]
    fn scale(self, s: f64) -> Cx {
        Cx::new(self.re * s, self.im * s)
    }
}

/// In-place radix-2 forward FFT, `X[k] = Σ_n x[n]·exp(−2πi·kn/N)`. `a.len()`
/// MUST be a power of two.
fn fft_pow2(a: &mut [Cx]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    // Butterflies. Twiddles for each stage are reduced via `k² mod` is not
    // needed here; the per-stage angle is small (≤ π) so cos/sin stay accurate.
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let wlen = Cx::new(ang.cos(), ang.sin());
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let mut w = Cx::new(1.0, 0.0);
            for k in 0..half {
                let u = a[i + k];
                let v = a[i + k + half].mul(w);
                a[i + k] = u.add(v);
                a[i + k + half] = u.sub(v);
                w = w.mul(wlen);
            }
            i += len;
        }
        len <<= 1;
    }
}

/// In-place radix-2 inverse FFT (`ifft = conj(fft(conj))/N`). Power-of-two len.
fn ifft_pow2(a: &mut [Cx]) {
    for c in a.iter_mut() {
        *c = c.conj();
    }
    fft_pow2(a);
    let s = 1.0 / a.len() as f64;
    for c in a.iter_mut() {
        *c = c.conj().scale(s);
    }
}

/// Bluestein chirp-z forward DFT for an arbitrary length `n`.
///
/// Uses `nk = (n² + k² − (k−n)²)/2` to turn the DFT into a circular convolution
/// of length `m = next_pow2(2n−1)`: with `w[k] = exp(−iπ·k²/n)`,
/// `X[k] = conj(w[k])·(a ⊛ b)[k]` where `a[k] = x[k]·w[k]` and `b` is the
/// length-`m` even extension of `conj(w)`. The chirp argument is reduced by
/// `k² mod 2n` (the period of `exp(−iπ·k²/n)`) so cos/sin stay accurate for
/// large `k`.
fn fft_bluestein(x: &[Cx]) -> Vec<Cx> {
    let n = x.len();
    let mut m = 1usize;
    while m < 2 * n - 1 {
        m <<= 1;
    }
    let two_n = 2 * n as u128;
    let mut wk = vec![Cx::ZERO; n];
    let mut a = vec![Cx::ZERO; m];
    for k in 0..n {
        let kk = ((k as u128 * k as u128) % two_n) as f64;
        let ang = -PI * kk / n as f64;
        wk[k] = Cx::new(ang.cos(), ang.sin());
        a[k] = x[k].mul(wk[k]);
    }
    // b: length-m even extension of conj(w) (b[k] and b[m-k] for k in 1..n).
    let mut b = vec![Cx::ZERO; m];
    b[0] = wk[0].conj();
    for k in 1..n {
        let bk = wk[k].conj();
        b[k] = bk;
        b[m - k] = bk;
    }
    fft_pow2(&mut a);
    fft_pow2(&mut b);
    for i in 0..m {
        a[i] = a[i].mul(b[i]);
    }
    ifft_pow2(&mut a);
    // X[k] = w[k]·conv[k] (the convolution kernel `b` already carries conj(w)).
    (0..n).map(|k| wk[k].mul(a[k])).collect()
}

/// Forward DFT of any length: radix-2 when `n` is a power of two, else Bluestein.
fn fft(x: &[Cx]) -> Vec<Cx> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if n.is_power_of_two() {
        let mut a = x.to_vec();
        fft_pow2(&mut a);
        a
    } else {
        fft_bluestein(x)
    }
}

/// Inverse DFT of any length (`ifft = conj(fft(conj))/N`).
fn ifft(x: &[Cx]) -> Vec<Cx> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let conj: Vec<Cx> = x.iter().map(|c| c.conj()).collect();
    let f = fft(&conj);
    let s = 1.0 / n as f64;
    f.iter().map(|c| c.conj().scale(s)).collect()
}

/// `real(ifft(fft(col) · d))` for a real column `col` and a real per-frequency
/// multiplier `d` (both length `n`). This is the Fourier-Wavelet damping kernel.
pub(crate) fn filter_real_column(col: &[f64], d: &[f64]) -> Vec<f64> {
    let n = col.len();
    if n == 0 {
        return Vec::new();
    }
    let cx: Vec<Cx> = col.iter().map(|&v| Cx::new(v, 0.0)).collect();
    let spec = fft(&cx);
    let filtered: Vec<Cx> = spec.iter().zip(d).map(|(c, &dv)| c.scale(dv)).collect();
    ifft(&filtered).iter().map(|c| c.re).collect()
}

/// `real(ifft2(fft2(input) · mult))` for a real 2-D `input` and a real
/// per-frequency multiplier `mult` (same shape). The 2-D, separable analogue of
/// [`filter_real_column`]: the forward transform runs along rows then columns,
/// the multiplier is applied in the frequency domain, and the inverse runs along
/// columns then rows. Used by the fitting-based stripe filter (`_2d_filter`).
pub(crate) fn filter_real_2d(input: &Array2<f64>, mult: &Array2<f64>) -> Array2<f64> {
    let (nrow, ncol) = input.dim();
    if nrow == 0 || ncol == 0 {
        return Array2::zeros((nrow, ncol));
    }
    // Complex working grid; row lanes (axis 1) and column lanes (axis 0) are
    // transformed in place via ndarray's lane iterators.
    let mut grid: Array2<Cx> = input.mapv(|v| Cx::new(v, 0.0));
    // fft2: FFT along each row (axis 1), then each column (axis 0).
    for mut lane in grid.rows_mut() {
        let f = fft(&lane.iter().copied().collect::<Vec<_>>());
        for (dst, src) in lane.iter_mut().zip(f) {
            *dst = src;
        }
    }
    for mut lane in grid.columns_mut() {
        let f = fft(&lane.iter().copied().collect::<Vec<_>>());
        for (dst, src) in lane.iter_mut().zip(f) {
            *dst = src;
        }
    }
    // Apply the real frequency multiplier.
    grid.zip_mut_with(mult, |g, &m| *g = g.scale(m));
    // ifft2: inverse FFT along each column, then each row; the per-axis 1/N
    // factors compose to numpy's 1/(nrow·ncol).
    for mut lane in grid.columns_mut() {
        let f = ifft(&lane.iter().copied().collect::<Vec<_>>());
        for (dst, src) in lane.iter_mut().zip(f) {
            *dst = src;
        }
    }
    let mut out = Array2::<f64>::zeros((nrow, ncol));
    for (mut orow, glane) in out.rows_mut().into_iter().zip(grid.rows()) {
        let f = ifft(&glane.iter().copied().collect::<Vec<_>>());
        for (dst, src) in orow.iter_mut().zip(f) {
            *dst = src.re;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive O(n²) DFT, the reference the FFT must reproduce.
    fn dft_naive(x: &[Cx]) -> Vec<Cx> {
        let n = x.len();
        let nf = n as f64;
        (0..n)
            .map(|k| {
                let mut acc = Cx::ZERO;
                for (m, &xv) in x.iter().enumerate() {
                    let ang = -2.0 * PI * (k as f64) * (m as f64) / nf;
                    acc = acc.add(xv.mul(Cx::new(ang.cos(), ang.sin())));
                }
                acc
            })
            .collect()
    }

    fn make(seed: u64, n: usize) -> Vec<Cx> {
        // Deterministic pseudo-random complex samples (no rng dependency).
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        (0..n).map(|_| Cx::new(next(), next())).collect()
    }

    #[test]
    fn fft_matches_naive_dft_all_lengths() {
        // Powers of two, primes, and composites all go through fft().
        for &n in &[1usize, 2, 3, 5, 7, 8, 11, 16, 17, 31, 64, 105, 128, 257] {
            let x = make(n as u64 + 1, n);
            let f = fft(&x);
            let r = dft_naive(&x);
            let err = f.iter().zip(&r).fold(0.0f64, |m, (a, b)| {
                m.max((a.re - b.re).abs().max((a.im - b.im).abs()))
            });
            let scale = r
                .iter()
                .fold(1e-300f64, |m, c| m.max(c.re.abs().max(c.im.abs())));
            assert!(err / scale < 1e-11, "n={n}: rel err {}", err / scale);
        }
    }

    #[test]
    fn ifft_inverts_fft() {
        for &n in &[1usize, 3, 5, 8, 17, 105, 128] {
            let x = make(n as u64 + 99, n);
            let back = ifft(&fft(&x));
            let err = x.iter().zip(&back).fold(0.0f64, |m, (a, b)| {
                m.max((a.re - b.re).abs().max((a.im - b.im).abs()))
            });
            assert!(err < 1e-11, "n={n}: round-trip err {err}");
        }
    }

    #[test]
    fn filter_real_column_matches_direct() {
        // The FFT path must equal the naive real(ifft(fft(col)·d)) it replaces.
        for &n in &[6usize, 11, 64, 105] {
            let col: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 0.37).sin() + 0.1 * i as f64)
                .collect();
            let d: Vec<f64> = (0..n)
                .map(|k| 1.0 - (-((k as f64) - 3.0).powi(2) / 8.0).exp())
                .collect();
            // Direct reference.
            let cx: Vec<Cx> = col.iter().map(|&v| Cx::new(v, 0.0)).collect();
            let spec = dft_naive(&cx);
            let g: Vec<Cx> = spec.iter().zip(&d).map(|(c, &dv)| c.scale(dv)).collect();
            // ifft via naive: conj(dft(conj))/n.
            let gc: Vec<Cx> = g.iter().map(|c| c.conj()).collect();
            let inv = dft_naive(&gc);
            let nf = n as f64;
            let want: Vec<f64> = inv.iter().map(|c| c.conj().scale(1.0 / nf).re).collect();

            let got = filter_real_column(&col, &d);
            let err = got
                .iter()
                .zip(&want)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(err < 1e-11, "n={n}: filter err {err}");
        }
    }

    #[test]
    fn filter_real_2d_matches_naive() {
        use ndarray::Array2;

        // Naive inverse DFT (conj(dft(conj))/N), applied lane-wise.
        let inv = |v: &[Cx]| -> Vec<Cx> {
            let conj: Vec<Cx> = v.iter().map(|c| c.conj()).collect();
            let f = dft_naive(&conj);
            let s = 1.0 / v.len() as f64;
            f.iter().map(|c| c.conj().scale(s)).collect()
        };

        // Reference: real(ifft2(fft2(mat)·mult)) via naive per-axis DFTs, using
        // the same lane structure as filter_real_2d.
        for &(nr, nc) in &[(4usize, 6usize), (5, 5), (8, 3), (7, 9)] {
            let mut mat = Array2::<f64>::zeros((nr, nc));
            let mut mult = Array2::<f64>::zeros((nr, nc));
            for ((r, c), cell) in mat.indexed_iter_mut() {
                *cell = ((r * 7 + c * 3) as f64 * 0.21).sin() + 0.05 * (r + c) as f64;
                mult[[r, c]] =
                    1.0 - (-((r as f64 - 1.5).powi(2) + (c as f64 - 2.0).powi(2)) / 5.0).exp();
            }

            let mut g: Array2<Cx> = mat.mapv(|v| Cx::new(v, 0.0));
            for mut lane in g.rows_mut() {
                let f = dft_naive(&lane.iter().copied().collect::<Vec<_>>());
                for (dst, src) in lane.iter_mut().zip(f) {
                    *dst = src;
                }
            }
            for mut lane in g.columns_mut() {
                let f = dft_naive(&lane.iter().copied().collect::<Vec<_>>());
                for (dst, src) in lane.iter_mut().zip(f) {
                    *dst = src;
                }
            }
            g.zip_mut_with(&mult, |gv, &m| *gv = gv.scale(m));
            for mut lane in g.columns_mut() {
                let f = inv(&lane.iter().copied().collect::<Vec<_>>());
                for (dst, src) in lane.iter_mut().zip(f) {
                    *dst = src;
                }
            }
            let mut want = Array2::<f64>::zeros((nr, nc));
            for (mut wrow, glane) in want.rows_mut().into_iter().zip(g.rows()) {
                let f = inv(&glane.iter().copied().collect::<Vec<_>>());
                for (dst, src) in wrow.iter_mut().zip(f) {
                    *dst = src.re;
                }
            }

            let got = filter_real_2d(&mat, &mult);
            let err = want
                .iter()
                .zip(got.iter())
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(err < 1e-11, "({nr},{nc}): 2d filter err {err}");
        }
    }
}
