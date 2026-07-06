//! Output mode (docs/GUI.md §2.5): browse a reconstructed volume (tiff slice
//! directory, `.h5`, or `.zarr` store), a downsampled 3-D isosurface view,
//! and the Octopus-style rescale export — a volume histogram drives a
//! min/max window for 8/16-bit tiff export on a common gray scale.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::Receiver;

use rsplot::egui_wgpu::RenderState;
use rsplot::{
    CurveData, Frame, FrameLoader, ImageStack, ItemHandle, Plot1D, ScalarFieldView, egui,
};

/// A resolved reconstruction volume and how to read one slice of it.
#[derive(Clone)]
enum VolumeSource {
    /// One f32 tiff per slice, sorted (the CLI's `{base}_{i:05}.tiff` layout).
    TiffDir(Arc<Vec<PathBuf>>),
    /// `exchange/data` of a `.h5` output.
    H5 { path: PathBuf, nz: usize },
    /// Zarr v2 store: raw `<f4` chunk file `exchange/data/{z}.0.0` per slice.
    Zarr {
        root: PathBuf,
        nz: usize,
        ny: usize,
        nx: usize,
    },
}

impl VolumeSource {
    fn nz(&self) -> usize {
        match self {
            VolumeSource::TiffDir(files) => files.len(),
            VolumeSource::H5 { nz, .. } | VolumeSource::Zarr { nz, .. } => *nz,
        }
    }

    fn describe(&self) -> String {
        match self {
            VolumeSource::TiffDir(files) => format!("tiff — {} slices", files.len()),
            VolumeSource::H5 { path, nz } => {
                format!(
                    "h5 — {} slices ({})",
                    nz,
                    path.file_name().unwrap_or_default().to_string_lossy()
                )
            }
            VolumeSource::Zarr { nz, ny, nx, .. } => {
                format!("zarr — {nz} slices of {ny}×{nx}")
            }
        }
    }
}

/// Resolve a user-supplied path into a browsable volume. Accepts a `.h5`
/// file, a `.zarr` store root, a directory of slice tiffs, one slice tiff
/// (its whole directory is taken), or a CLI output *base* like `…/scan_rec`
/// (`{base}_*.tiff` siblings are collected).
fn resolve_volume(path: &Path) -> Result<VolumeSource, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if path.is_file() && (ext == "h5" || ext == "hdf5") {
        let (nz, _ny, _nx) = tomoxide::io::read_h5_sizes(&path.to_string_lossy(), "/exchange/data")
            .map_err(|e| e.to_string())?;
        return Ok(VolumeSource::H5 {
            path: path.to_path_buf(),
            nz,
        });
    }
    if path.is_dir() && path.join(".zgroup").is_file() {
        return resolve_zarr(path);
    }
    if path.is_dir() {
        return tiff_dir(path, None);
    }
    if path.is_file() && (ext == "tiff" || ext == "tif") {
        let dir = path.parent().ok_or("slice tiff has no parent directory")?;
        return tiff_dir(dir, None);
    }
    // Not an existing file/dir: a CLI output base — collect `{stem}_*.tiff`.
    let dir = path
        .parent()
        .filter(|d| d.is_dir())
        .ok_or_else(|| format!("{} does not exist", path.display()))?;
    let stem = path
        .file_name()
        .ok_or("empty output base")?
        .to_string_lossy()
        .into_owned();
    tiff_dir(dir, Some(&format!("{stem}_")))
}

/// Collect the sorted slice tiffs of `dir`, optionally restricted to a
/// filename prefix. Zero-padded CLI names sort correctly lexicographically.
fn tiff_dir(dir: &Path, prefix: Option<&str>) -> Result<VolumeSource, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("read dir {}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            (ext == "tiff" || ext == "tif") && prefix.is_none_or(|pre| name.starts_with(pre))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("no slice tiffs found in {}", dir.display()));
    }
    Ok(VolumeSource::TiffDir(Arc::new(files)))
}

/// Parse a Zarr v2 store as written by the CLI (`exchange/data`, raw `<f4`,
/// one z-slice per chunk). Anything else is an honest error, not a guess.
fn resolve_zarr(root: &Path) -> Result<VolumeSource, String> {
    let zarray_path = root.join("exchange").join("data").join(".zarray");
    let text = std::fs::read_to_string(&zarray_path)
        .map_err(|e| format!("read {}: {e}", zarray_path.display()))?;
    let meta: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse .zarray: {e}"))?;
    if meta["dtype"] != "<f4" || !meta["compressor"].is_null() {
        return Err(format!(
            "unsupported zarr array (need raw <f4, got dtype {} compressor {})",
            meta["dtype"], meta["compressor"]
        ));
    }
    let dim = |i: usize| -> Result<usize, String> {
        meta["shape"][i]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!(".zarray shape[{i}] missing"))
    };
    let (nz, ny, nx) = (dim(0)?, dim(1)?, dim(2)?);
    if meta["chunks"] != serde_json::json!([1, ny, nx]) {
        return Err(format!(
            "unsupported zarr chunking {} (need one z-slice per chunk)",
            meta["chunks"]
        ));
    }
    Ok(VolumeSource::Zarr {
        root: root.to_path_buf(),
        nz,
        ny,
        nx,
    })
}

/// Read slice `index` as `(ny, nx, row-major f32)`.
fn read_slice(src: &VolumeSource, index: usize) -> anyhow::Result<(usize, usize, Vec<f32>)> {
    match src {
        VolumeSource::TiffDir(files) => {
            let path = files
                .get(index)
                .ok_or_else(|| anyhow::anyhow!("slice {index} out of range"))?;
            let (w, h, data) = super::load_tiff_f32(path)?;
            Ok((h as usize, w as usize, data))
        }
        VolumeSource::H5 { path, .. } => Ok(tomoxide::io::read_h5_frame(
            &path.to_string_lossy(),
            "/exchange/data",
            index,
        )?),
        VolumeSource::Zarr { root, ny, nx, .. } => {
            let chunk = root
                .join("exchange")
                .join("data")
                .join(format!("{index}.0.0"));
            let bytes = std::fs::read(&chunk)?;
            if bytes.len() != ny * nx * 4 {
                anyhow::bail!(
                    "{}: {} bytes, expected {}",
                    chunk.display(),
                    bytes.len(),
                    ny * nx * 4
                );
            }
            let data = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok((*ny, *nx, data))
        }
    }
}

/// Lazy slice loader for the browser stack (sources are slice indices).
struct VolumeFrameLoader(VolumeSource);

impl FrameLoader for VolumeFrameLoader {
    fn load(&self, source: &str) -> Option<Frame> {
        let index = source.parse::<usize>().ok()?;
        let (ny, nx, data) = read_slice(&self.0, index).ok()?;
        Some(Frame::new(
            nx as u32,
            ny as u32,
            data,
            Some(format!("slice {index}")),
        ))
    }
}

/// One background pass over the volume: a stride-sampled ≤`DOWN_MAX`³ copy
/// (the 3-D view input) plus a histogram and robust range computed from it.
struct Scan {
    /// Downsampled volume, row-major `(dz, dy, dx)`.
    down: Vec<f32>,
    dims: (usize, usize, usize),
    /// 256-bin histogram over `[min, max]`.
    hist: Vec<f64>,
    min: f32,
    max: f32,
    /// Robust display window (0.5–99.5 percentile).
    window: (f32, f32),
}

const DOWN_MAX: usize = 192;
const HIST_BINS: usize = 256;

/// Build [`Scan`] by reading every `sz`-th slice and sampling every
/// `(sy, sx)`-th pixel. Pure function so tests can run it synchronously.
fn scan_volume(src: &VolumeSource) -> anyhow::Result<Scan> {
    let nz = src.nz();
    if nz == 0 {
        anyhow::bail!("empty volume");
    }
    let sz = nz.div_ceil(DOWN_MAX);
    let mut down = Vec::new();
    let mut dims_yx = None;
    let mut dz = 0;
    for z in (0..nz).step_by(sz) {
        let (ny, nx, data) = read_slice(src, z)?;
        let (sy, sx) = (ny.div_ceil(DOWN_MAX), nx.div_ceil(DOWN_MAX));
        let (dy, dx) = (ny.div_ceil(sy), nx.div_ceil(sx));
        match dims_yx {
            None => dims_yx = Some((sy, sx, dy, dx)),
            Some(d) if d == (sy, sx, dy, dx) => {}
            Some(_) => anyhow::bail!("slice {z} cross-section differs"),
        }
        for y in (0..ny).step_by(sy) {
            for x in (0..nx).step_by(sx) {
                down.push(data[y * nx + x]);
            }
        }
        dz += 1;
    }
    let (_, _, dy, dx) = dims_yx.unwrap();

    let finite: Vec<f32> = down.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        anyhow::bail!("volume has no finite values");
    }
    let min = finite.iter().copied().fold(f32::INFINITY, f32::min);
    let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut hist = vec![0.0_f64; HIST_BINS];
    let span = (max - min).max(f32::MIN_POSITIVE);
    for &v in &finite {
        let bin = (((v - min) / span) * HIST_BINS as f32) as usize;
        hist[bin.min(HIST_BINS - 1)] += 1.0;
    }
    let (lo, hi) = super::robust_range(&down);
    Ok(Scan {
        down,
        dims: (dz, dy, dx),
        hist,
        min,
        max,
        window: (lo as f32, hi as f32),
    })
}

/// Map `v` into the `[lo, hi]` window as a `0..=max_out` gray level.
fn quantize(v: f32, lo: f32, hi: f32, max_out: u32) -> u32 {
    if !v.is_finite() {
        return 0;
    }
    let t = ((v - lo) / (hi - lo).max(f32::MIN_POSITIVE)).clamp(0.0, 1.0);
    (t * max_out as f32).round() as u32
}

/// Progress/outcome of a background job (scan or export).
enum JobMsg {
    ScanDone(Box<Scan>),
    ExportProgress(usize),
    Finished(Result<String, String>),
}

/// Export every slice as an 8- or 16-bit gray tiff under `dir`, windowed to
/// `[lo, hi]` (one common scale for the whole stack).
fn export_volume(
    src: &VolumeSource,
    dir: &Path,
    bits16: bool,
    lo: f32,
    hi: f32,
    mut progress: impl FnMut(usize),
) -> anyhow::Result<String> {
    std::fs::create_dir_all(dir)?;
    let nz = src.nz();
    for z in 0..nz {
        let (ny, nx, data) = read_slice(src, z)?;
        let path = dir.join(format!("export_{z:05}.tiff"));
        let file = std::io::BufWriter::new(std::fs::File::create(&path)?);
        let mut enc = tiff::encoder::TiffEncoder::new(file)?;
        if bits16 {
            let buf: Vec<u16> = data
                .iter()
                .map(|&v| quantize(v, lo, hi, u16::MAX as u32) as u16)
                .collect();
            enc.write_image::<tiff::encoder::colortype::Gray16>(nx as u32, ny as u32, &buf)?;
        } else {
            let buf: Vec<u8> = data
                .iter()
                .map(|&v| quantize(v, lo, hi, u8::MAX as u32) as u8)
                .collect();
            enc.write_image::<tiff::encoder::colortype::Gray8>(nx as u32, ny as u32, &buf)?;
        }
        progress(z + 1);
    }
    Ok(format!(
        "exported {nz} × {}-bit slices → {}",
        if bits16 { 16 } else { 8 },
        dir.display()
    ))
}

/// Central-area display mode.
#[derive(PartialEq, Clone, Copy)]
enum Pane {
    Slices,
    ThreeD,
}

pub struct OutputView {
    path_input: String,
    source: Option<VolumeSource>,
    /// Human-readable summary of the opened source.
    info: String,
    pane: Pane,

    stack: ImageStack,
    field: ScalarFieldView,
    /// Data was handed to the 3-D view (show it instead of a hint).
    field_ready: bool,
    /// 3-D widgets take a `RenderState` on data upload, not only at
    /// construction — keep a clone (it is `Arc`-backed).
    render_state: RenderState,

    hist_plot: Plot1D,
    hist_curve: Option<ItemHandle>,
    scan: Option<Scan>,

    /// Export window (gray scale end points) and depth.
    pub lo: f32,
    pub hi: f32,
    bits16: bool,
    export_dir: String,
    export_done: usize,

    /// In-flight background job (scan or export; one at a time).
    job: Option<Receiver<JobMsg>>,
    job_label: &'static str,
}

impl OutputView {
    pub fn new(render_state: &RenderState) -> Self {
        let mut stack = ImageStack::new(render_state, 90);
        stack.set_table_visible(false);
        stack.set_n_prefetch(2);
        let mut hist_plot = Plot1D::new(render_state, 100);
        hist_plot.set_graph_title("histogram — click sets the nearer bound");
        let mut field = ScalarFieldView::new(render_state, 0);
        field.add_auto_isosurface(
            render_state,
            rsplot::mean_plus_std,
            egui::Color32::from_rgb(255, 70, 90),
        );
        OutputView {
            path_input: String::new(),
            source: None,
            info: String::new(),
            pane: Pane::Slices,
            stack,
            field,
            field_ready: false,
            render_state: render_state.clone(),
            hist_plot,
            hist_curve: None,
            scan: None,
            lo: 0.0,
            hi: 1.0,
            bits16: false,
            export_dir: String::new(),
            export_done: 0,
            job: None,
            job_label: "",
        }
    }

    /// Open `path` as a volume: point the browser stack at it and start the
    /// background scan (histogram + downsampled 3-D copy).
    fn open(&mut self, path: &Path, ctx: &egui::Context, log: &mut Vec<String>) {
        match resolve_volume(path) {
            Ok(src) => {
                self.path_input = path.display().to_string();
                self.info = src.describe();
                self.stack
                    .set_loader(Arc::new(VolumeFrameLoader(src.clone())));
                self.stack
                    .set_sources((0..src.nz()).map(|i| i.to_string()).collect());
                self.scan = None;
                self.field_ready = false;
                self.export_dir = format!("{}_export", path.display());
                log.push(format!("output: opened {} ({})", path.display(), self.info));
                self.spawn_scan(src.clone(), ctx.clone());
                self.source = Some(src);
            }
            Err(e) => log.push(format!("output: {e}")),
        }
    }

    fn spawn_scan(&mut self, src: VolumeSource, ctx: egui::Context) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.job = Some(rx);
        self.job_label = "scanning volume";
        std::thread::spawn(move || {
            let result = scan_volume(&src);
            match result {
                Ok(scan) => {
                    let _ = tx.send(JobMsg::ScanDone(Box::new(scan)));
                    let _ = tx.send(JobMsg::Finished(Ok(String::new())));
                }
                Err(e) => {
                    let _ = tx.send(JobMsg::Finished(Err(format!("volume scan: {e}"))));
                }
            }
            ctx.request_repaint();
        });
    }

    fn spawn_export(&mut self, ctx: egui::Context, log: &mut Vec<String>) {
        let Some(src) = self.source.clone() else {
            return;
        };
        if self.hi <= self.lo {
            log.push("export not started: window max must exceed min".into());
            return;
        }
        let dir = PathBuf::from(&self.export_dir);
        let (bits16, lo, hi) = (self.bits16, self.lo, self.hi);
        let (tx, rx) = std::sync::mpsc::channel();
        self.job = Some(rx);
        self.job_label = "exporting";
        self.export_done = 0;
        std::thread::spawn(move || {
            let progress_tx = tx.clone();
            let progress_ctx = ctx.clone();
            let result = export_volume(&src, &dir, bits16, lo, hi, move |done| {
                let _ = progress_tx.send(JobMsg::ExportProgress(done));
                progress_ctx.request_repaint();
            });
            let _ = tx.send(JobMsg::Finished(result.map_err(|e| e.to_string())));
            ctx.request_repaint();
        });
    }

    /// Drain background-job messages; apply a finished scan to the widgets.
    fn poll(&mut self, log: &mut Vec<String>) {
        let Some(rx) = &self.job else { return };
        let mut done = false;
        let mut scans = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            match msg {
                JobMsg::ScanDone(scan) => scans.push(scan),
                JobMsg::ExportProgress(n) => self.export_done = n,
                JobMsg::Finished(result) => {
                    match result {
                        Ok(s) if !s.is_empty() => log.push(format!("output: {s}")),
                        Ok(_) => {}
                        Err(e) => log.push(format!("output FAILED: {e}")),
                    }
                    done = true;
                }
            }
        }
        for scan in scans {
            self.apply_scan(*scan);
        }
        if done {
            self.job = None;
        }
    }

    fn apply_scan(&mut self, scan: Scan) {
        (self.lo, self.hi) = scan.window;
        self.stack
            .set_colormap(rsplot::Colormap::viridis(self.lo as f64, self.hi as f64));
        let (dz, dy, dx) = scan.dims;
        self.field_ready = self
            .field
            .set_data(&self.render_state, &scan.down, dz, dy, dx);

        // Histogram curve at bin centers.
        let span = (scan.max - scan.min) as f64;
        let x: Vec<f64> = (0..HIST_BINS)
            .map(|i| scan.min as f64 + span * (i as f64 + 0.5) / HIST_BINS as f64)
            .collect();
        let curve = CurveData::new(x, scan.hist.clone(), egui::Color32::LIGHT_GREEN);
        match self.hist_curve {
            Some(h) => {
                self.hist_plot.update_curve_data(h, &curve);
            }
            None => {
                self.hist_curve = Some(self.hist_plot.add_curve_data_with_legend(&curve, "counts"));
            }
        }
        self.scan = Some(scan);
    }

    /// Histogram with click-to-set: a click moves whichever window bound is
    /// nearer to the clicked gray value (the doc's draggable-range concept).
    fn hist_panel(&mut self, ui: &mut egui::Ui) {
        let resp = self.hist_plot.show(ui);
        if self.scan.is_some()
            && resp.response.clicked()
            && let Some(pos) = resp.response.interact_pointer_pos()
        {
            let (x, _y) = resp.transform.pixel_to_data(pos);
            let x = x as f32;
            if (x - self.lo).abs() <= (x - self.hi).abs() {
                self.lo = x.min(self.hi);
            } else {
                self.hi = x.max(self.lo);
            }
        }
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        last_run: Option<&(String, String)>,
        log: &mut Vec<String>,
    ) {
        self.poll(log);
        if self.job.is_some() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(200));
        }

        ui.horizontal(|ui| {
            if ui.button("File…").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("volume", &["h5", "hdf5", "tiff", "tif"])
                    .pick_file()
            {
                self.open(&path, ui.ctx(), log);
            }
            if ui
                .button("Folder…")
                .on_hover_text("a directory of slice tiffs, or a .zarr store root")
                .clicked()
                && let Some(dir) = rfd::FileDialog::new().pick_folder()
            {
                self.open(&dir, ui.ctx(), log);
            }
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.path_input)
                    .hint_text("output base / .h5 / .zarr / tiff directory")
                    .desired_width(f32::INFINITY),
            );
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                let path = PathBuf::from(self.path_input.clone());
                self.open(&path, ui.ctx(), log);
            }
        });
        if let Some((base, format)) = last_run
            && ui
                .button(format!("Open last run ({base}, {format})"))
                .clicked()
        {
            let path = match format.as_str() {
                "h5" => PathBuf::from(format!("{base}.h5")),
                "zarr" => PathBuf::from(format!("{base}.zarr")),
                _ => PathBuf::from(base),
            };
            self.open(&path, ui.ctx(), log);
        }
        ui.separator();

        if self.source.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label("Open a reconstruction output (tiff directory, .h5, or .zarr).");
            });
            return;
        }

        egui::Panel::left("output_side")
            .resizable(true)
            .default_size(360.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.side_panel(ui, log));
            });

        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.pane, Pane::Slices, "Slices");
            ui.selectable_value(&mut self.pane, Pane::ThreeD, "3D");
        });
        match self.pane {
            Pane::Slices => {
                self.stack.ui(ui);
            }
            Pane::ThreeD => {
                if self.field_ready {
                    self.field.show(ui);
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("3D view appears when the volume scan finishes.");
                    });
                }
            }
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, log: &mut Vec<String>) {
        ui.heading("Volume");
        ui.label(&self.info);
        if self.job.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(self.job_label);
            });
        }
        ui.separator();

        ui.heading("Rescale export");
        self.hist_panel(ui);
        let idle = self.job.is_none();
        ui.add_enabled_ui(idle && self.scan.is_some(), |ui| {
            ui.horizontal(|ui| {
                ui.label("min");
                ui.add(egui::DragValue::new(&mut self.lo).speed(0.01));
                ui.label("max");
                ui.add(egui::DragValue::new(&mut self.hi).speed(0.01));
                if ui
                    .button("auto")
                    .on_hover_text("0.5–99.5 percentile of the scanned volume")
                    .clicked()
                    && let Some(scan) = &self.scan
                {
                    (self.lo, self.hi) = scan.window;
                }
            });
            ui.horizontal(|ui| {
                ui.label("depth");
                ui.selectable_value(&mut self.bits16, false, "8-bit");
                ui.selectable_value(&mut self.bits16, true, "16-bit");
            });
            ui.horizontal(|ui| {
                ui.label("directory");
                ui.add(egui::TextEdit::singleline(&mut self.export_dir).desired_width(180.0));
                if ui.button("…").clicked()
                    && let Some(dir) = rfd::FileDialog::new().pick_folder()
                {
                    self.export_dir = dir.display().to_string();
                }
            });
            if ui.button("Export").clicked() {
                self.spawn_export(ui.ctx().clone(), log);
            }
        });
        if self.job_label == "exporting"
            && self.job.is_some()
            && let Some(src) = &self.source
        {
            let total = src.nz().max(1);
            ui.add(
                egui::ProgressBar::new(self.export_done as f32 / total as f32)
                    .text(format!("{}/{total} slices", self.export_done)),
            );
        }
        if let Some(scan) = &self.scan {
            ui.label(
                egui::RichText::new(format!(
                    "scan: min {:.4}, max {:.4}, 3D {}×{}×{}",
                    scan.min, scan.max, scan.dims.0, scan.dims.1, scan.dims.2
                ))
                .small()
                .weak(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a small volume with the tomoxide writer in the given format and
    /// return the CLI-style output base.
    fn write_volume(dir: &Path, format: tomoxide::io::SaveFormat) -> String {
        let base = dir.join("rec").to_string_lossy().into_owned();
        let vol = tomoxide::Volume::new(ndarray::Array3::from_shape_fn((4, 3, 5), |(z, y, x)| {
            (z * 100 + y * 10 + x) as f32
        }));
        let mut w = tomoxide::io::create_writer(&base, format).unwrap();
        w.reserve(4).unwrap();
        w.write_chunk(&vol, 0, 4).unwrap();
        w.finalize().unwrap();
        base
    }

    fn tmp(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("tomoxide-gui-out-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// All three writer formats resolve and read back the same slices —
    /// via the directory, the output base, and the container paths.
    #[test]
    fn resolve_and_read_all_writer_formats() {
        let dir = tmp("resolve");
        let base = write_volume(&dir, tomoxide::io::SaveFormat::Tiff);
        write_volume(&dir, tomoxide::io::SaveFormat::H5);
        write_volume(&dir, tomoxide::io::SaveFormat::Zarr);

        let sources = [
            resolve_volume(&dir.join(format!("{}.h5", "rec"))).unwrap(),
            resolve_volume(&dir.join("rec.zarr")).unwrap(),
            resolve_volume(Path::new(&base)).unwrap(), // tiff output base
        ];
        for src in &sources {
            assert_eq!(src.nz(), 4);
            let (ny, nx, data) = read_slice(src, 2).unwrap();
            assert_eq!((ny, nx), (3, 5));
            assert_eq!(data[0], 200.0);
            assert_eq!(data[14], 224.0);
        }
        // A directory containing loose tiffs also resolves (whole-dir mode
        // picks up the slice files regardless of prefix).
        let via_dir = resolve_volume(&dir);
        assert!(
            matches!(via_dir, Ok(VolumeSource::TiffDir(ref f)) if f.len() == 4),
            "directory-of-tiffs resolution failed"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A missing base and an empty directory produce errors, not panics.
    #[test]
    fn resolve_rejects_missing_and_empty() {
        let dir = tmp("empty");
        assert!(resolve_volume(&dir).is_err(), "empty dir must not resolve");
        assert!(resolve_volume(&dir.join("nothing/here")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The scan downsample covers the volume, and the histogram counts every
    /// finite sample of the downsampled copy.
    #[test]
    fn scan_volume_histogram_and_downsample() {
        let dir = tmp("scan");
        let base = write_volume(&dir, tomoxide::io::SaveFormat::Tiff);
        let src = resolve_volume(Path::new(&base)).unwrap();
        let scan = scan_volume(&src).unwrap();
        // 4×3×5 is below DOWN_MAX in every axis: the copy is the full volume.
        assert_eq!(scan.dims, (4, 3, 5));
        assert_eq!(scan.down.len(), 60);
        assert_eq!(scan.hist.iter().sum::<f64>(), 60.0);
        assert_eq!(scan.min, 0.0);
        // Max sample: z=3, y=2, x=4 → 3·100 + 2·10 + 4.
        assert_eq!(scan.max, 324.0);
        assert!(scan.window.0 < scan.window.1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Window quantization: clamped ends, linear middle, NaN → 0.
    #[test]
    fn quantize_windows_and_clamps() {
        assert_eq!(quantize(-1.0, 0.0, 2.0, 255), 0);
        assert_eq!(quantize(0.0, 0.0, 2.0, 255), 0);
        assert_eq!(quantize(1.0, 0.0, 2.0, 255), 128);
        assert_eq!(quantize(2.0, 0.0, 2.0, 255), 255);
        assert_eq!(quantize(5.0, 0.0, 2.0, 255), 255);
        assert_eq!(quantize(f32::NAN, 0.0, 2.0, 255), 0);
        assert_eq!(quantize(1.0, 0.0, 2.0, u16::MAX as u32), 32768);
    }

    /// Export writes one decodable gray tiff per slice on a common scale.
    #[test]
    fn export_writes_decodable_gray_tiffs() {
        let dir = tmp("export");
        let base = write_volume(&dir, tomoxide::io::SaveFormat::Tiff);
        let src = resolve_volume(Path::new(&base)).unwrap();
        let out = dir.join("u8");
        let mut seen = 0;
        export_volume(&src, &out, false, 0.0, 324.0, |n| seen = n.max(seen)).unwrap();
        assert_eq!(seen, 4);

        let file = std::fs::File::open(out.join("export_00003.tiff")).unwrap();
        let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).unwrap();
        assert_eq!(dec.dimensions().unwrap(), (5, 3));
        let tiff::decoder::DecodingResult::U8(v) = dec.read_image().unwrap() else {
            panic!("expected u8 tiff");
        };
        // Slice 3 pixel (0,0) = 300 → round(300/324·255) = 236.
        assert_eq!(v[0], 236);
        // Last pixel = 324 (the window max) → 255.
        assert_eq!(*v.last().unwrap(), 255);

        // 16-bit variant decodes as u16 with the wider scale.
        let out16 = dir.join("u16");
        export_volume(&src, &out16, true, 0.0, 324.0, |_| {}).unwrap();
        let file = std::fs::File::open(out16.join("export_00000.tiff")).unwrap();
        let mut dec = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).unwrap();
        let tiff::decoder::DecodingResult::U16(v) = dec.read_image().unwrap() else {
            panic!("expected u16 tiff");
        };
        assert_eq!(v[0], 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
