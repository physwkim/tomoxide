//! Minimal Daubechies-5 (`db5`) 2-D discrete wavelet transform matching
//! PyWavelets (`pywt`) exactly — only what `remove_stripe_fw` needs: single-level
//! `dwt2`/`idwt2` with the default `symmetric` (half-sample) boundary mode, in
//! f64.
//!
//! Conventions were pinned against pywt 1.8.0 and are exercised by the unit
//! tests below (the oracle values come straight from `pywt`):
//!
//! - **Forward** (`mode='symmetric'`): pad the signal by `F-1` samples each side
//!   with half-sample symmetry, full-convolve with the decomposition filter, and
//!   take `full[F : F + 2·outlen : 2]`, where `outlen = (n + F - 1) / 2`.
//!   Equivalently `out[i] = Σ_k filt[k]·x[sym(2i − k + 1)]`.
//! - **Inverse**: upsample (zeros *after* each coefficient), full-convolve the
//!   approximation/detail with `rec_lo`/`rec_hi`, sum, and crop
//!   `[F-2 : F-2 + (2L − F + 2)]`.
//! - **2-D**: separable — `dwt` along axis 1 (columns within a row) then axis 0;
//!   bands map as `cA = LL`, `cH = (lo-col, hi-row)`, `cV = (hi-col, lo-row)`,
//!   `cD = HH`.

use ndarray::Array2;

/// db5 filter length.
const F: usize = 10;

/// `pywt.Wavelet('db5').dec_lo` (decomposition low-pass).
const DEC_LO: [f64; F] = [
    0.003_335_725_285_473_771_2,
    -0.012_580_751_999_081_999,
    -0.006_241_490_212_798_274,
    0.077_571_493_840_045_72,
    -0.032_244_869_584_638_375,
    -0.242_294_887_066_382_03,
    0.138_428_145_901_320_74,
    0.724_308_528_437_772_9,
    0.603_829_269_797_189_6,
    0.160_102_397_974_192_93,
];

/// `pywt.Wavelet('db5').dec_hi` (decomposition high-pass).
const DEC_HI: [f64; F] = [
    -0.160_102_397_974_192_93,
    0.603_829_269_797_189_6,
    -0.724_308_528_437_772_9,
    0.138_428_145_901_320_74,
    0.242_294_887_066_382_03,
    -0.032_244_869_584_638_375,
    -0.077_571_493_840_045_72,
    -0.006_241_490_212_798_274,
    0.012_580_751_999_081_999,
    0.003_335_725_285_473_771_2,
];

/// `pywt.Wavelet('db5').rec_lo` (reconstruction low-pass).
const REC_LO: [f64; F] = [
    0.160_102_397_974_192_93,
    0.603_829_269_797_189_6,
    0.724_308_528_437_772_9,
    0.138_428_145_901_320_74,
    -0.242_294_887_066_382_03,
    -0.032_244_869_584_638_375,
    0.077_571_493_840_045_72,
    -0.006_241_490_212_798_274,
    -0.012_580_751_999_081_999,
    0.003_335_725_285_473_771_2,
];

/// `pywt.Wavelet('db5').rec_hi` (reconstruction high-pass).
const REC_HI: [f64; F] = [
    0.003_335_725_285_473_771_2,
    0.012_580_751_999_081_999,
    -0.006_241_490_212_798_274,
    -0.077_571_493_840_045_72,
    -0.032_244_869_584_638_375,
    0.242_294_887_066_382_03,
    0.138_428_145_901_320_74,
    -0.724_308_528_437_772_9,
    0.603_829_269_797_189_6,
    -0.160_102_397_974_192_93,
];

/// Half-sample symmetric index (numpy `pad(mode='symmetric')`): reflect `t` into
/// `[0, n)` repeating the boundary sample.
fn sym_index(t: isize, n: isize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * n;
    let mut m = t % period;
    if m < 0 {
        m += period;
    }
    if m >= n {
        m = period - 1 - m;
    }
    m as usize
}

/// 1-D forward DWT (`symmetric` mode): `(cA, cD)`, each of length `(n + F - 1)/2`.
fn dwt1(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let outlen = (n + F - 1) / 2;
    let ni = n as isize;
    let mut ca = vec![0.0f64; outlen];
    let mut cd = vec![0.0f64; outlen];
    for i in 0..outlen {
        let (mut a, mut d) = (0.0f64, 0.0f64);
        for k in 0..F {
            let xv = x[sym_index(2 * i as isize - k as isize + 1, ni)];
            a += DEC_LO[k] * xv;
            d += DEC_HI[k] * xv;
        }
        ca[i] = a;
        cd[i] = d;
    }
    (ca, cd)
}

/// `np.convolve(a, b)` (`mode='full'`), length `a.len() + b.len() - 1`.
fn convolve_full(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0f64; a.len() + b.len() - 1];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    out
}

/// 1-D inverse DWT: reconstruct a signal of length `2·L − F + 2` from
/// `(cA, cD)` (each length `L`).
fn idwt1(ca: &[f64], cd: &[f64]) -> Vec<f64> {
    let l = ca.len();
    if l == 0 {
        return Vec::new();
    }
    // Upsample: zeros AFTER each coefficient ([c0, 0, c1, 0, ...]).
    let mut ua = vec![0.0f64; 2 * l];
    let mut ud = vec![0.0f64; 2 * l];
    for t in 0..l {
        ua[2 * t] = ca[t];
        ud[2 * t] = cd[t];
    }
    let a = convolve_full(&ua, &REC_LO);
    let d = convolve_full(&ud, &REC_HI);
    let r = 2 * l + 2 - F; // reconstruction length
    let off = F - 2; // crop offset
    (0..r).map(|m| a[off + m] + d[off + m]).collect()
}

/// 2-D forward DWT (single level) → `(cA, cH, cV, cD)`, each
/// `[(nrow+F-1)/2, (ncol+F-1)/2]`.
pub(crate) fn dwt2(a: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let (nrow, ncol) = a.dim();
    let outcol = (ncol + F - 1) / 2;
    let outrow = (nrow + F - 1) / 2;
    // Step 1: dwt along axis 1 (each row) → cols_a / cols_d, shape [nrow, outcol].
    let mut cols_a = Array2::<f64>::zeros((nrow, outcol));
    let mut cols_d = Array2::<f64>::zeros((nrow, outcol));
    let mut row = vec![0.0f64; ncol];
    for r in 0..nrow {
        for (c, v) in row.iter_mut().enumerate() {
            *v = a[[r, c]];
        }
        let (ra, rd) = dwt1(&row);
        for c in 0..outcol {
            cols_a[[r, c]] = ra[c];
            cols_d[[r, c]] = rd[c];
        }
    }
    // Step 2: dwt along axis 0 (each column) → the four bands [outrow, outcol].
    let mut ca = Array2::<f64>::zeros((outrow, outcol));
    let mut ch = Array2::<f64>::zeros((outrow, outcol));
    let mut cv = Array2::<f64>::zeros((outrow, outcol));
    let mut cd = Array2::<f64>::zeros((outrow, outcol));
    let mut col = vec![0.0f64; nrow];
    for c in 0..outcol {
        for (r, v) in col.iter_mut().enumerate() {
            *v = cols_a[[r, c]];
        }
        let (aa, da) = dwt1(&col);
        for (r, v) in col.iter_mut().enumerate() {
            *v = cols_d[[r, c]];
        }
        let (ad, dd) = dwt1(&col);
        for r in 0..outrow {
            ca[[r, c]] = aa[r]; // LL
            ch[[r, c]] = da[r]; // lo-col, hi-row
            cv[[r, c]] = ad[r]; // hi-col, lo-row
            cd[[r, c]] = dd[r]; // HH
        }
    }
    (ca, ch, cv, cd)
}

/// 2-D inverse DWT (single level). All four bands must share one shape
/// `[L0, L1]`; the result is `[2·L0 − F + 2, 2·L1 − F + 2]`.
pub(crate) fn idwt2(
    ca: &Array2<f64>,
    ch: &Array2<f64>,
    cv: &Array2<f64>,
    cd: &Array2<f64>,
) -> Array2<f64> {
    let (l0, l1) = ca.dim();
    let rrow = 2 * l0 + 2 - F;
    // Invert axis 0: cols_a = idwt(cA, cH), cols_d = idwt(cV, cD), shape [rrow, l1].
    let mut cols_a = Array2::<f64>::zeros((rrow, l1));
    let mut cols_d = Array2::<f64>::zeros((rrow, l1));
    let (mut a_col, mut h_col, mut v_col, mut d_col) =
        (vec![0.0; l0], vec![0.0; l0], vec![0.0; l0], vec![0.0; l0]);
    for c in 0..l1 {
        for r in 0..l0 {
            a_col[r] = ca[[r, c]];
            h_col[r] = ch[[r, c]];
            v_col[r] = cv[[r, c]];
            d_col[r] = cd[[r, c]];
        }
        let ra = idwt1(&a_col, &h_col);
        let rd = idwt1(&v_col, &d_col);
        for r in 0..rrow {
            cols_a[[r, c]] = ra[r];
            cols_d[[r, c]] = rd[r];
        }
    }
    // Invert axis 1: result row = idwt(cols_a row, cols_d row), shape [rrow, 2*l1-F+2].
    let rcol = 2 * l1 + 2 - F;
    let mut out = Array2::<f64>::zeros((rrow, rcol));
    let (mut arow, mut drow) = (vec![0.0; l1], vec![0.0; l1]);
    for r in 0..rrow {
        for c in 0..l1 {
            arow[c] = cols_a[[r, c]];
            drow[c] = cols_d[[r, c]];
        }
        let rr = idwt1(&arow, &drow);
        for c in 0..rcol {
            out[[r, c]] = rr[c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Oracle values from pywt 1.8.0, db5, mode='symmetric'.
    #[test]
    fn dwt1_matches_pywt() {
        // N = 6.
        let x: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let (ca, cd) = dwt1(&x);
        let ca_exp = [
            8.53855754780706,
            6.8161338031278556,
            3.8970170902159254,
            1.3609373888046061,
            3.08336113348381,
            6.002477846395741,
            8.53855754780706,
        ];
        let cd_exp = [
            0.10093414801910747,
            -0.13606787624540015,
            -0.010042127484714626,
            -0.10093414801910763,
            0.13606787624539993,
            0.01004212748471412,
            0.10093414801910754,
        ];
        assert_eq!(ca.len(), 7);
        for (g, e) in ca.iter().zip(ca_exp) {
            assert!((g - e).abs() < 1e-12, "cA {g} vs {e}");
        }
        for (g, e) in cd.iter().zip(cd_exp) {
            assert!((g - e).abs() < 1e-12, "cD {g} vs {e}");
        }
    }

    #[test]
    fn idwt1_inverts_to_pywt_recon() {
        // N = 9 → recon length 10 (pywt's idwt of dwt(x)).
        let x: Vec<f64> = (1..=9).map(|v| v as f64).collect();
        let (ca, cd) = dwt1(&x);
        let rec = idwt1(&ca, &cd);
        let exp = [
            1.0000000000000004,
            2.0000000000000004,
            3.0000000000000004,
            4.000000000000001,
            5.0,
            6.0,
            6.999999999999998,
            8.0,
            9.0,
            9.000000000000002,
        ];
        assert_eq!(rec.len(), 10);
        for (g, e) in rec.iter().zip(exp) {
            assert!((g - e).abs() < 1e-12, "recon {g} vs {e}");
        }
    }

    #[test]
    fn dwt2_idwt2_match_pywt() {
        let a = Array2::from_shape_fn((5, 4), |(r, c)| (r * 4 + c + 1) as f64);
        let (ca, ch, cv, cd) = dwt2(&a);
        assert_eq!(ca.dim(), (7, 6));
        // Spot-check a few band values against pywt.
        assert!((ca[[0, 0]] - 32.046313561434275).abs() < 1e-10);
        assert!((ch[[0, 0]] - 0.426202580246038).abs() < 1e-10);
        assert!((cv[[0, 0]] - 0.1888883078383144).abs() < 1e-10);
        assert!(cd[[0, 0]].abs() < 1e-12);
        // idwt2 reconstructs to pywt's (6, 4): the first 5 rows recover the
        // original `a` exactly; row 5 is the symmetric boundary repeat of row 4.
        let r = idwt2(&ca, &ch, &cv, &cd);
        assert_eq!(r.dim(), (6, 4));
        for row in 0..5 {
            for col in 0..4 {
                let want = a[[row, col]];
                assert!(
                    (r[[row, col]] - want).abs() < 1e-9,
                    "recon[{row},{col}] {} vs {want}",
                    r[[row, col]]
                );
            }
        }
        assert!((r[[5, 3]] - 20.0).abs() < 1e-9);
    }
}
