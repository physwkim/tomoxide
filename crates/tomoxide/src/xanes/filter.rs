//! 1-D spectral smoothers for per-voxel XANES fitting.
//!
//! Ported from `txm_pal_core::filter` (`~/codes/TXM-Pal-core/src/filter.rs`) —
//! the reference pipeline these reconstructions are chemically mapped with. The
//! signatures were relaxed from `&Vec<f64>` to `&[f64]`; the arithmetic is
//! otherwise unchanged, with ONE deliberate divergence: [`medfilt`] fills the
//! right-edge samples with their windowed medians (SciPy `medfilt` semantics)
//! instead of leaving them zeroed as the upstream port did — see [`medfilt`].

use median::Filter;

/// Sliding-window median filter, matching SciPy's `medfilt` shape.
///
/// `edge_mode` is `"extension"` (repeat the edge sample) or `"zeropadding"`
/// (pad with 0) — the reference pipeline calls it with `"zeropadding"`.
///
/// Every output index — both edges and the interior — is filled with its
/// padded windowed median. This diverges from the upstream `txm_pal_core`
/// port, which advanced the sliding window across the right edge but never
/// stored those medians, leaving the last `window_size / 2` samples at `0.0`
/// and silently corrupting spectra whose absorption peak sits near the
/// high-energy end.
pub fn medfilt(arr: Vec<f64>, window_size: usize, edge_mode: &str) -> Vec<f64> {
    let radius = window_size / 2;
    let mut filtered: Vec<f64> = vec![0.0; arr.len()];
    let mut filter = Filter::new(window_size);

    for i in 0..arr.len() {
        // left edge handling
        if i == 0 {
            // insert arr[0] into filter when (i-radius) < 0
            for _ in 0..radius {
                if edge_mode == "extension" {
                    filter.consume(arr[0]);
                } else if edge_mode == "zeropadding" {
                    filter.consume(0.0);
                }
            }

            // insert arr[0..radius+1] into filter
            for &v in arr.iter().take(radius + 1) {
                filter.consume(v);
            }
            filtered[i] = filter.median();

        // right edge handling
        } else if i >= arr.len().saturating_sub(radius) {
            // Slide in the pad sample and STORE the resulting windowed median,
            // matching SciPy `medfilt` (which pads and still computes the tail
            // medians) and the left-edge branch above. The upstream port
            // consumed the pad but never assigned `filtered[i]`, leaving the
            // last `radius` outputs at their `0.0` init — which corrupted any
            // peak sitting near the high-energy end of a spectrum.
            let pad = if edge_mode == "extension" {
                arr[arr.len() - 1]
            } else {
                0.0
            };
            filtered[i] = filter.consume(pad);
        } else {
            filtered[i] = filter.consume(arr[i + radius]);
        }
    }
    filtered
}

/// Iterated 3-point moving average (endpoints held fixed).
pub fn multi_3point_average(arr: &[f64], iteration: usize) -> Vec<f64> {
    let one_third: f64 = 1.0 / 3.0;
    let mut filtered: Vec<f64> = arr.to_vec();
    let mut buffer: Vec<f64> = arr.to_vec();

    for _ in 0..iteration {
        for i in 1..filtered.len() - 1 {
            filtered[i] = one_third * (buffer[i - 1] + buffer[i] + buffer[i + 1]);
        }
        buffer = filtered.clone();
    }
    filtered
}

/// Centred boxcar (uniform) filter of width `kernel_size`.
pub fn boxcar(arr: &[f64], kernel_size: usize) -> Vec<f64> {
    let kernel_weight: f64 = 1.0 / kernel_size as f64;
    let half_kernel: usize = kernel_size / 2;

    (0..arr.len())
        .map(|i| {
            let start = i.saturating_sub(half_kernel);
            let end = usize::min(i + half_kernel + 1, arr.len());
            arr[start..end].iter().sum::<f64>() * kernel_weight
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: the zero-padded windowed median at every index (odd window).
    fn brute_zeropad_median(arr: &[f64], window: usize) -> Vec<f64> {
        let n = arr.len();
        let r = window / 2;
        (0..n)
            .map(|i| {
                let mut w: Vec<f64> = (0..window)
                    .map(|k| {
                        let idx = i as isize + k as isize - r as isize;
                        if idx < 0 || idx as usize >= n {
                            0.0
                        } else {
                            arr[idx as usize]
                        }
                    })
                    .collect();
                w.sort_by(|a, b| a.partial_cmp(b).unwrap());
                w[window / 2]
            })
            .collect()
    }

    /// Regression for the right-edge zeroing: every output — including the last
    /// `radius` samples — must be the zero-padded windowed median (SciPy
    /// `medfilt` semantics), not the `0.0` init the upstream port left there.
    #[test]
    fn medfilt_zeropadding_matches_windowed_median_including_the_tail() {
        let arr: Vec<f64> = (1..=12).map(|v| v as f64).collect();
        let window = 5usize;
        let got = medfilt(arr.clone(), window, "zeropadding");
        let want = brute_zeropad_median(&arr, window);
        assert_eq!(
            got, want,
            "medfilt must equal the zero-padded windowed median at every index"
        );
        // The tail specifically must be the windowed medians, not zeros.
        assert!(
            got[arr.len() - window / 2..].iter().all(|&v| v > 0.0),
            "right-edge samples must be filled with medians, not left at 0.0"
        );
    }
}
