//! Per-mode screens. Each view owns its siplot widgets (constructed once with
//! the wgpu `RenderState`) and receives worker [`Event`](crate::worker::Event)s
//! routed by the app shell.
//!
//! `PlotId` allocation convention (siplot ids must not collide; some widgets
//! reserve a small range): Data 0–29, Tune 30–49, Center 50–69.

pub mod data;
pub mod tune;

/// Viridis scaled to the finite min/max of `data` (fallback 0..1).
pub(crate) fn autoscale_viridis(data: &[f32]) -> siplot::Colormap {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in data {
        if v.is_finite() {
            let v = v as f64;
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    if !(lo.is_finite() && hi.is_finite() && lo < hi) {
        (lo, hi) = (0.0, 1.0);
    }
    siplot::Colormap::viridis(lo, hi)
}
