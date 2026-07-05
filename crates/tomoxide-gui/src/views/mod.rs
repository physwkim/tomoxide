//! Per-mode screens. Each view owns its rsplot widgets (constructed once with
//! the wgpu `RenderState`) and receives worker [`Event`](crate::worker::Event)s
//! routed by the app shell.
//!
//! `PlotId` allocation convention (rsplot ids must not collide; some widgets
//! reserve a small range): Data 0–29, Tune 30–49, Center 50–69, Run 70–89,
//! Output 90–109.

pub mod center;
pub mod data;
pub mod output;
pub mod run;
pub mod tune;

/// Render the "value under the cursor" readout for an [`rsplot::ImageView`],
/// mirroring the silx `PositionInfo` "Data" column. `ImageView`'s own readout
/// shows the cursor's x, y only; the pixel value under the cursor is a separate
/// query ([`rsplot::ImageView::value_changed`]), so it is drawn here. A fixed
/// placeholder keeps the row height stable when the cursor is off the image.
pub(crate) fn value_readout(ui: &mut rsplot::egui::Ui, value: Option<(f64, f64, f64)>) {
    let text = match value {
        Some((col, row, v)) => format!("value @ ({col:.0}, {row:.0}) = {v:.6}"),
        None => "value @ —".to_owned(),
    };
    ui.label(text);
}

/// Decode a single-image f32 tiff (the CLI's slice output format). Shared by
/// the Run live view and the Output browser.
pub(crate) fn load_tiff_f32(path: &std::path::Path) -> anyhow::Result<(u32, u32, Vec<f32>)> {
    let file = std::fs::File::open(path)?;
    let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(file))?;
    let (w, h) = dec.dimensions()?;
    match dec.read_image()? {
        tiff::decoder::DecodingResult::F32(v) => Ok((w, h, v)),
        _ => anyhow::bail!("not an f32 tiff"),
    }
}

/// Viridis scaled to the robust 0.5–99.5 percentile range of `data`
/// (fallback 0..1). Absolute min/max let a handful of extreme pixels own the
/// whole colormap — on real truncated-FOV data the iterative methods put a
/// huge edge ring / out-of-disk values in the frame, and with min/max scaling
/// the intact interior structure was left ~1 % of the gray range ("smeared").
/// Percentile clipping is the silx/ImageJ-style autoscale.
pub(crate) fn autoscale_viridis(data: &[f32]) -> rsplot::Colormap {
    let (lo, hi) = robust_range(data);
    rsplot::Colormap::viridis(lo, hi)
}

/// Finite 0.5th and 99.5th percentiles of `data`; falls back to (0, 1) when
/// fewer than two distinct finite values remain.
pub(crate) fn robust_range(data: &[f32]) -> (f64, f64) {
    let mut finite: Vec<f32> = data.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    let last = finite.len() - 1;
    let ilo = last / 200; // 0.5 %
    let ihi = last - ilo; // 99.5 %
    let (_, lo, _) = finite.select_nth_unstable_by(ilo, f32::total_cmp);
    let lo = *lo as f64;
    let (_, hi, _) = finite.select_nth_unstable_by(ihi, f32::total_cmp);
    let hi = *hi as f64;
    if lo < hi { (lo, hi) } else { (0.0, 1.0) }
}

#[cfg(test)]
mod tests {
    use super::robust_range;

    #[test]
    fn robust_range_clips_outliers() {
        // 1000 samples in 0..1 plus two extreme outliers: min/max scaling
        // would return (-1e6, 1e6); the percentile range must stay near 0..1.
        let mut data: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        data.push(-1e6);
        data.push(1e6);
        let (lo, hi) = robust_range(&data);
        assert!((0.0..0.05).contains(&lo), "lo = {lo}");
        assert!((0.95..1.0).contains(&hi), "hi = {hi}");
    }

    #[test]
    fn robust_range_degenerate_falls_back() {
        assert_eq!(robust_range(&[]), (0.0, 1.0));
        assert_eq!(robust_range(&[f32::NAN, f32::INFINITY]), (0.0, 1.0));
        assert_eq!(robust_range(&[3.0; 16]), (0.0, 1.0));
    }
}
