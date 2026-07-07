//! 1-D spectral smoothers for per-voxel XANES fitting.
//!
//! Ported verbatim (bit-for-bit numerics) from `txm_pal_core::filter`
//! (`~/codes/TXM-Pal-core/src/filter.rs`) — the reference pipeline these
//! reconstructions are chemically mapped with. Only the signatures were
//! relaxed from `&Vec<f64>` to `&[f64]`; the arithmetic is unchanged.

use median::Filter;

/// Sliding-window median filter, matching SciPy's `medfilt` shape.
///
/// `edge_mode` is `"extension"` (repeat the edge sample) or `"zeropadding"`
/// (pad with 0) — the reference pipeline calls it with `"zeropadding"`.
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
            if edge_mode == "extension" {
                filter.consume(arr[arr.len() - 1]);
            } else if edge_mode == "zeropadding" {
                filter.consume(0.0);
            }
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
