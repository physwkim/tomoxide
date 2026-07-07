//! XANES mode (docs/GUI.md §2.7): fit a per-voxel absorption-edge peak-energy
//! map over a registered multi-energy reconstruction stack — a chemical-state
//! map. The stack is streamed a `z`-band at a time through
//! [`tomoxide::xanes::fit_map`] on a background thread (cancellable); the map is
//! browsed slice-by-slice, a click reads that voxel's spectrum, and the result
//! saves to an interoperable HDF5.
//!
//! The stack loader (a single combined `registered.h5` with an `energies` axis
//! and `reconstructions/{energy}` volumes) is wired separately; everything here
//! operates on a [`MultiEnergyVolume`] once one is set.

use std::path::Path;
use std::sync::mpsc::Receiver;

use ndarray::{Array1, Array3, Array4, s};
use rsplot::egui_wgpu::RenderState;
use rsplot::{Colormap, CurveData, ImageView, ItemHandle, Plot1D, VolumeRaycaster, egui};
use tomoxide::CancelToken;
use tomoxide::xanes::{
    FitMethod, FitParams, MagnificationParams, MultiEnergyVolume, SmoothAlgo, apply_magnification,
    fit_map, magnification_corr_factors, write_peak_map_h5,
};

const HIST_BINS: usize = 128;

/// Longest edge of the 3-D volume texture; larger maps are decimated to fit so
/// the GPU upload stays bounded (a raw RGBA8 volume is `d·h·w·4` bytes).
const MAX_3D_DIM: usize = 192;

/// Build an RGBA8 volume (row-major `(depth, height, width)`) for the ray-caster
/// from a peak-energy map, decimated so its longest edge is at most
/// [`MAX_3D_DIM`]. Finite voxels take the viridis colour of their peak energy
/// (window `disp`) at full opacity; non-finite (unfitted / masked) voxels are
/// left fully transparent.
fn build_volume_rgba(map: &Array3<f64>, disp: (f32, f32)) -> (Vec<u8>, usize, usize, usize) {
    let (nz, ny, nx) = map.dim();
    if nz == 0 || ny == 0 || nx == 0 {
        return (Vec::new(), 0, 0, 0);
    }
    let stride = nz.max(ny).max(nx).div_ceil(MAX_3D_DIM).max(1);
    let (od, oh, ow) = (
        nz.div_ceil(stride),
        ny.div_ceil(stride),
        nx.div_ceil(stride),
    );
    let cmap = Colormap::viridis(disp.0 as f64, disp.1 as f64);
    let mut rgba = vec![0u8; od * oh * ow * 4];
    let mut i = 0;
    for z in (0..nz).step_by(stride) {
        for y in (0..ny).step_by(stride) {
            for x in (0..nx).step_by(stride) {
                let v = map[[z, y, x]];
                if v.is_finite() {
                    let c = cmap.color_at(v);
                    rgba[i] = c[0];
                    rgba[i + 1] = c[1];
                    rgba[i + 2] = c[2];
                    rgba[i + 3] = 255;
                }
                i += 4;
            }
        }
    }
    (rgba, od, oh, ow)
}

/// A completed `z`-band, progress tick, or terminal status from the fit thread.
enum FitMsg {
    /// One finished band `[z0, z0+band)` of the peak-energy map plus its
    /// edge-jump band, both row-major `(band, ny, nx)`.
    Band {
        z0: usize,
        band: usize,
        ny: usize,
        nx: usize,
        data: Vec<f64>,
        edge: Vec<f64>,
    },
    /// Slices completed so far.
    Progress(usize),
    /// Terminal: `Ok` on completion, `Err` on failure or cancellation.
    Finished(Result<(), String>),
}

/// Where the fit loop reads its `z`-bands from. The default streams a band at a
/// time straight off disk (memory-bounded). Magnification correction couples the
/// `z` axis (each energy's volume is rescaled about its centre), so it cannot be
/// applied band-by-band; when enabled the whole corrected stack is held in
/// memory and bands are sliced from it. One `band()` call, two backings.
enum FitSource {
    Streamed(MultiEnergyVolume),
    Corrected(Array4<f32>),
}

impl FitSource {
    /// The `z`-band `[z0, z1)` as `(E, band, ny, nx)` f32.
    fn band(&self, z0: usize, z1: usize) -> Result<Array4<f32>, String> {
        match self {
            FitSource::Streamed(v) => v.read_band(z0, z1).map_err(|e| e.to_string()),
            FitSource::Corrected(a) => Ok(a.slice(s![.., z0..z1, .., ..]).to_owned()),
        }
    }
}

/// Pre/post-edge image count for the edge-jump reduction (matches the reference
/// `calculate_edge_jump` default of 3).
const EDGE_JUMP_N: usize = 3;

/// Compute the edge-jump band from a `(E, band, ny, nx)` volume band:
/// `mean(last n energies) - mean(first n energies)` per voxel, row-major
/// `(band, ny, nx)`. `n` is clamped to the energy count. Because edge jump is a
/// per-voxel reduction over the energy axis (every band carries all energies),
/// it is exact band-by-band — and it rides the same (possibly magnification-
/// corrected) `volband` as the fit, so the two always agree.
fn edge_jump_band(volband: ndarray::ArrayView4<f32>) -> Vec<f64> {
    let (ne, bh, byn, bxn) = volband.dim();
    let n = EDGE_JUMP_N.min(ne).max(1);
    let mut out = vec![0.0_f64; bh * byn * bxn];
    let mut i = 0;
    for z in 0..bh {
        for y in 0..byn {
            for x in 0..bxn {
                let mut pre = 0.0_f64;
                let mut post = 0.0_f64;
                for e in 0..n {
                    pre += volband[[e, z, y, x]] as f64;
                    post += volband[[ne - 1 - e, z, y, x]] as f64;
                }
                out[i] = post / n as f64 - pre / n as f64;
                i += 1;
            }
        }
    }
    out
}

/// Read the full stack and apply per-energy zone-plate magnification correction,
/// returning a corrected `(E, nz, ny, nx)` volume. Each energy's `(nz, ny, nx)`
/// sub-volume is scaled about its centre by that energy's correction factor
/// (normalised to the first energy), matching the reference workflow that writes
/// corrected `reconstructions/{energy}` datasets.
fn build_corrected(
    vol: &MultiEnergyVolume,
    energies: &[f64],
    mag: &MagnificationParams,
) -> Result<Array4<f32>, String> {
    let (ne, nz, ny, nx) = vol.dims();
    let cf = magnification_corr_factors(energies, mag);
    if cf.len() != ne {
        return Err(format!(
            "magnification: {} factors for {ne} energies",
            cf.len()
        ));
    }
    let mut out = Array4::<f32>::zeros((ne, nz, ny, nx));
    // One energy resident at a time: read energy e, resample it, store, drop.
    // Peak memory is the corrected stack plus a single energy volume, not a
    // second full copy of the whole stack.
    for (e, &factor) in cf.iter().enumerate() {
        let raw = vol.read_energy(e).map_err(|e| e.to_string())?;
        let corrected = apply_magnification(raw.view(), factor);
        out.slice_mut(s![e, .., .., ..]).assign(&corrected);
    }
    Ok(out)
}

pub struct XanesView {
    /// Loaded registered stack (set by the loader, wired separately).
    volume: Option<MultiEnergyVolume>,
    energies: Vec<f64>,
    info: String,

    // Fit parameters.
    method: FitMethod,
    points: usize,
    start_e: f64,
    stop_e: f64,
    smooth: SmoothAlgo,
    smooth_width: usize,
    smooth_order: usize,
    /// Skip voxels whose mean absorption over energy is below this (0 = fit all).
    mask_threshold: f64,
    /// Slices fitted per streamed band.
    band_size: usize,
    /// Apply per-energy zone-plate magnification correction before fitting.
    mag_correct: bool,
    /// Zone-plate geometry driving the magnification correction.
    mag: MagnificationParams,

    // Fit state.
    job: Option<Receiver<FitMsg>>,
    cancel: Option<CancelToken>,
    progress: (usize, usize),

    // Result.
    map: Option<Array3<f64>>,
    /// Edge-jump volume (post-edge minus pre-edge absorption), filled band by
    /// band alongside `map`; the viewer's opacity/thickness channel.
    edge_jump: Option<Array3<f64>>,
    map_z: usize,
    /// Display window for the peak-energy colormap.
    disp: (f32, f32),
    map_view: ImageView,
    hist_plot: Plot1D,
    hist_curve: Option<ItemHandle>,
    spectrum_plot: Plot1D,
    spectrum_curve: Option<ItemHandle>,
    /// The voxel whose spectrum is shown (z, row, col).
    picked: Option<(usize, usize, usize)>,

    // 3-D direct volume rendering of the chemical map.
    render_state: RenderState,
    raycaster: VolumeRaycaster,
    /// Show the ray-cast 3-D volume instead of the 2-D slice browser.
    show_3d: bool,
    /// The map changed since the volume texture was last built.
    vol_dirty: bool,
    /// Opacity multiplier for the volume rendering.
    alpha_scale: f32,

    save_path: String,
}

impl XanesView {
    pub fn new(render_state: &RenderState) -> Self {
        let mut map_view = ImageView::new(render_state, 110);
        map_view.set_side_histogram_displayed(false);
        map_view.image_plot_mut().set_graph_title("peak energy");
        let mut hist_plot = Plot1D::new(render_state, 114);
        hist_plot.set_graph_title("peak-energy histogram");
        let mut spectrum_plot = Plot1D::new(render_state, 116);
        spectrum_plot.set_graph_title("spectrum — click a voxel");
        let raycaster = VolumeRaycaster::new(render_state, 118);
        XanesView {
            volume: None,
            energies: Vec::new(),
            info: String::new(),
            method: FitMethod::Quadratic,
            points: 7,
            start_e: 0.0,
            stop_e: 0.0,
            smooth: SmoothAlgo::None,
            smooth_width: 5,
            smooth_order: 2,
            mask_threshold: 0.0,
            band_size: 16,
            mag_correct: false,
            mag: MagnificationParams::default(),
            job: None,
            cancel: None,
            progress: (0, 0),
            map: None,
            edge_jump: None,
            map_z: 0,
            disp: (0.0, 1.0),
            map_view,
            hist_plot,
            hist_curve: None,
            spectrum_plot,
            spectrum_curve: None,
            picked: None,
            render_state: render_state.clone(),
            raycaster,
            show_3d: false,
            vol_dirty: false,
            alpha_scale: 1.0,
            save_path: String::new(),
        }
    }

    /// Open a combined registered stack (`registered.h5`: an `energies` axis and
    /// `reconstructions/{energy}` volumes) and adopt it for fitting.
    fn open_registered(&mut self, path: &Path, log: &mut Vec<String>) {
        match resolve_registered_stack(path) {
            Ok((volume, info)) => {
                log.push(format!("xanes: loaded {info}"));
                self.set_volume(volume, info);
            }
            Err(e) => log.push(format!("xanes: load FAILED — {e}")),
        }
    }

    /// Adopt a freshly loaded stack: reset fit range to its energy span and
    /// clear any previous result.
    fn set_volume(&mut self, volume: MultiEnergyVolume, info: String) {
        self.energies = volume.energies();
        if let (Some(&first), Some(&last)) = (self.energies.first(), self.energies.last()) {
            self.start_e = first;
            self.stop_e = last;
        }
        let (_e, nz, _ny, _nx) = volume.dims();
        self.map_z = nz / 2;
        self.map = None;
        self.picked = None;
        self.info = info;
        self.volume = Some(volume);
    }

    fn fit_params(&self) -> FitParams {
        FitParams {
            method: self.method,
            points: self.points.max(3),
            start_e: self.start_e,
            stop_e: self.stop_e,
            smooth: self.smooth,
            smooth_width: self.smooth_width.max(1),
            smooth_order: self.smooth_order,
        }
    }

    fn spawn_fit(&mut self, ctx: &egui::Context, log: &mut Vec<String>) {
        let Some(vol) = self.volume.clone() else {
            return;
        };
        if self.stop_e <= self.start_e {
            log.push("xanes: fit not started — stop energy must exceed start".into());
            return;
        }
        let (_e, nz, ny, nx) = vol.dims();
        let params = self.fit_params();
        let energies = self.energies.clone();
        let energy = Array1::from(energies.clone());
        let band_size = self.band_size.max(1);
        let mask_threshold = self.mask_threshold;
        let mag_correct = self.mag_correct;
        let mag = self.mag;
        let cancel = CancelToken::new();
        self.cancel = Some(cancel.clone());
        self.map = Some(Array3::from_elem((nz, ny, nx), f64::NAN));
        self.edge_jump = Some(Array3::from_elem((nz, ny, nx), f64::NAN));
        self.progress = (0, nz);
        let ctx = ctx.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        self.job = Some(rx);

        std::thread::spawn(move || {
            // Build the band source: stream off disk, or read + magnification-
            // correct the whole stack up front (z-coupled, so not band-local).
            let source = if mag_correct {
                match build_corrected(&vol, &energies, &mag) {
                    Ok(a) => FitSource::Corrected(a),
                    Err(e) => {
                        let _ = tx.send(FitMsg::Finished(Err(format!(
                            "magnification correction: {e}"
                        ))));
                        ctx.request_repaint();
                        return;
                    }
                }
            } else {
                FitSource::Streamed(vol)
            };

            let mut z0 = 0;
            while z0 < nz {
                if cancel.is_cancelled() {
                    let _ = tx.send(FitMsg::Finished(Err("cancelled".into())));
                    ctx.request_repaint();
                    return;
                }
                let z1 = (z0 + band_size).min(nz);
                let volband = match source.band(z0, z1) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx.send(FitMsg::Finished(Err(format!(
                            "read band [{z0}, {z1}): {e}"
                        ))));
                        ctx.request_repaint();
                        return;
                    }
                };
                let (ne, bh, byn, bxn) = volband.dim();
                let mut maskband = Array3::<u8>::from_elem((bh, byn, bxn), 1u8);
                if mask_threshold > 0.0 {
                    for z in 0..bh {
                        for y in 0..byn {
                            for x in 0..bxn {
                                let mut sum = 0.0;
                                for e in 0..ne {
                                    sum += volband[[e, z, y, x]] as f64;
                                }
                                if sum / (ne as f64) < mask_threshold {
                                    maskband[[z, y, x]] = 0;
                                }
                            }
                        }
                    }
                }
                let mapband = match fit_map(
                    energy.view(),
                    volband.view(),
                    maskband.view(),
                    &params,
                    Some(&cancel),
                ) {
                    Ok(m) => m,
                    Err(tomoxide::Error::Cancelled) => {
                        let _ = tx.send(FitMsg::Finished(Err("cancelled".into())));
                        ctx.request_repaint();
                        return;
                    }
                    Err(e) => {
                        let _ = tx.send(FitMsg::Finished(Err(e.to_string())));
                        ctx.request_repaint();
                        return;
                    }
                };
                let data: Vec<f64> = mapband.iter().copied().collect();
                let edge = edge_jump_band(volband.view());
                let _ = tx.send(FitMsg::Band {
                    z0,
                    band: bh,
                    ny: byn,
                    nx: bxn,
                    data,
                    edge,
                });
                let _ = tx.send(FitMsg::Progress(z1));
                ctx.request_repaint();
                z0 = z1;
            }
            let _ = tx.send(FitMsg::Finished(Ok(())));
            ctx.request_repaint();
        });
    }

    /// Drain fit-thread messages; fill completed bands into the map.
    fn poll(&mut self, log: &mut Vec<String>) {
        let Some(rx) = &self.job else { return };
        let mut got_band = false;
        let mut finished = None;
        let msgs: Vec<FitMsg> = rx.try_iter().collect();
        for msg in msgs {
            match msg {
                FitMsg::Band {
                    z0,
                    band,
                    ny,
                    nx,
                    data,
                    edge,
                } => {
                    if let Some(map) = &mut self.map
                        && let Ok(slab) = Array3::from_shape_vec((band, ny, nx), data)
                    {
                        map.slice_mut(s![z0..z0 + band, .., ..]).assign(&slab);
                        got_band = true;
                    }
                    if let Some(ej) = &mut self.edge_jump
                        && let Ok(slab) = Array3::from_shape_vec((band, ny, nx), edge)
                    {
                        ej.slice_mut(s![z0..z0 + band, .., ..]).assign(&slab);
                    }
                }
                FitMsg::Progress(done) => self.progress.0 = done,
                FitMsg::Finished(result) => finished = Some(result),
            }
        }
        if got_band {
            self.refresh_display();
        }
        if let Some(result) = finished {
            match result {
                Ok(()) => log.push(format!("xanes: fit complete — {} slices", self.progress.1)),
                Err(e) => log.push(format!("xanes: fit stopped — {e}")),
            }
            self.job = None;
            self.cancel = None;
            self.refresh_display();
        }
    }

    /// Recompute the colormap window + histogram from the finite peak energies
    /// and redraw the current slice.
    fn refresh_display(&mut self) {
        let Some(map) = &self.map else { return };
        let finite: Vec<f32> = map
            .iter()
            .filter(|v| v.is_finite())
            .map(|&v| v as f32)
            .collect();
        if !finite.is_empty() {
            self.disp = {
                let (lo, hi) = super::robust_range(&finite);
                (lo as f32, hi as f32)
            };
        }
        self.redraw_slice();
        self.rebuild_histogram(&finite);
        self.vol_dirty = true;
    }

    /// Rebuild and re-upload the 3-D volume texture from the current map and
    /// display window, if it changed since the last upload. The transfer
    /// function is: hue = the same viridis peak-energy colormap as the 2-D map,
    /// alpha = opaque where a voxel was fitted (finite peak), transparent where
    /// it was masked or the fit left the window — so fitted material shows its
    /// chemical state and background falls away.
    fn rebuild_volume_if_dirty(&mut self) {
        if !self.vol_dirty {
            return;
        }
        let Some(map) = &self.map else { return };
        let (rgba, d, h, w) = build_volume_rgba(map, self.disp);
        if d > 0 && h > 0 && w > 0 {
            self.raycaster
                .set_volume(&self.render_state, &rgba, d, h, w);
        }
        self.vol_dirty = false;
    }

    fn redraw_slice(&mut self) {
        let Some(map) = &self.map else { return };
        let (nz, ny, nx) = map.dim();
        if nz == 0 {
            return;
        }
        let z = self.map_z.min(nz - 1);
        let slice: Vec<f32> = map.slice(s![z, .., ..]).iter().map(|&v| v as f32).collect();
        let cmap = Colormap::viridis(self.disp.0 as f64, self.disp.1 as f64);
        let _ = self.map_view.set_image(nx as u32, ny as u32, &slice, cmap);
        self.map_view
            .image_plot_mut()
            .set_graph_title(format!("peak energy — z {z}"));
    }

    fn rebuild_histogram(&mut self, finite: &[f32]) {
        let (lo, hi) = self.disp;
        let span = (hi - lo).max(f32::MIN_POSITIVE);
        let mut counts = vec![0.0_f64; HIST_BINS];
        for &v in finite {
            let bin = (((v - lo) / span) * HIST_BINS as f32) as isize;
            if (0..HIST_BINS as isize).contains(&bin) {
                counts[bin as usize] += 1.0;
            }
        }
        let x: Vec<f64> = (0..HIST_BINS)
            .map(|i| lo as f64 + (hi - lo) as f64 * (i as f64 + 0.5) / HIST_BINS as f64)
            .collect();
        let curve = CurveData::new(x, counts, egui::Color32::LIGHT_GREEN);
        match self.hist_curve {
            Some(h) => {
                self.hist_plot.update_curve_data(h, &curve);
            }
            None => {
                self.hist_curve = Some(self.hist_plot.add_curve_data_with_legend(&curve, "voxels"));
            }
        }
    }

    /// Read the clicked voxel's absorption spectrum and plot energy vs value.
    fn pick_spectrum(&mut self, z: usize, row: usize, col: usize, log: &mut Vec<String>) {
        let Some(vol) = &self.volume else { return };
        let band = match vol.read_band(z, z + 1) {
            Ok(b) => b,
            Err(e) => {
                log.push(format!("xanes: spectrum read failed — {e}"));
                return;
            }
        };
        let (ne, _b, ny, nx) = band.dim();
        if row >= ny || col >= nx {
            return;
        }
        let y: Vec<f64> = (0..ne).map(|e| band[[e, 0, row, col]] as f64).collect();
        let curve = CurveData::new(self.energies.clone(), y, egui::Color32::LIGHT_BLUE);
        match self.spectrum_curve {
            Some(h) => {
                self.spectrum_plot.update_curve_data(h, &curve);
            }
            None => {
                self.spectrum_curve = Some(
                    self.spectrum_plot
                        .add_curve_data_with_legend(&curve, "absorption"),
                );
            }
        }
        let peak = self
            .map
            .as_ref()
            .map(|m| m[[z, row, col]])
            .unwrap_or(f64::NAN);
        self.spectrum_plot
            .set_graph_title(format!("spectrum @ ({col}, {row}, z{z}) — peak {peak:.4}"));
        self.picked = Some((z, row, col));
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, log: &mut Vec<String>) {
        self.poll(log);
        if self.job.is_some() {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(150));
        }

        ui.horizontal(|ui| {
            if ui
                .button("Load registered stack…")
                .on_hover_text("a combined registered.h5 (energies + reconstructions/{energy})")
                .clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("HDF5", &["h5", "hdf5"])
                    .pick_file()
            {
                self.open_registered(&path, log);
            }
            ui.label(if self.info.is_empty() {
                "no stack loaded".to_owned()
            } else {
                self.info.clone()
            });
        });
        ui.separator();

        if self.volume.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label("Load a registered multi-energy stack to fit a peak-energy chemical map.");
            });
            return;
        }

        egui::Panel::left("xanes_side")
            .resizable(true)
            .default_size(380.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.side_panel(ui, log));
            });

        // Central: slice browser + peak-energy map.
        let Some(map) = &self.map else {
            ui.centered_and_justified(|ui| {
                ui.label("Set the fit range and press Run to compute the map.");
            });
            return;
        };
        let (nz, _ny, _nx) = map.dim();

        // View mode: 2-D slice browser or 3-D ray-cast volume.
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.show_3d, false, "2-D slice");
            ui.selectable_value(&mut self.show_3d, true, "3-D volume");
            if self.show_3d {
                ui.separator();
                ui.label("opacity");
                if ui
                    .add(egui::Slider::new(&mut self.alpha_scale, 0.05..=4.0).logarithmic(true))
                    .changed()
                {
                    self.raycaster.set_alpha_scale(self.alpha_scale);
                }
                if ui.button("reset view").clicked() {
                    self.raycaster.reset_view();
                }
            }
        });

        if self.show_3d {
            self.rebuild_volume_if_dirty();
            self.raycaster.set_alpha_scale(self.alpha_scale);
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                self.raycaster.show(ui);
            });
            return;
        }

        let mut z = self.map_z.min(nz.saturating_sub(1));
        ui.horizontal(|ui| {
            ui.label("z");
            if ui
                .add(egui::Slider::new(&mut z, 0..=nz.saturating_sub(1)))
                .changed()
            {
                self.map_z = z;
                self.redraw_slice();
            }
        });

        // Value-under-cursor readout, then the image; a click over the image
        // (value present) picks that voxel's spectrum.
        let hovered = self.map_view.value_changed();
        super::value_readout(ui, hovered);
        self.map_view.show(ui, None, None);
        if let Some((col, row, _)) = hovered
            && ui.input(|i| i.pointer.primary_clicked())
        {
            self.pick_spectrum(self.map_z, row as usize, col as usize, log);
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, log: &mut Vec<String>) {
        let running = self.job.is_some();

        ui.heading("Fit");
        ui.add_enabled_ui(!running, |ui| {
            egui::Grid::new("xanes_fit_grid").show(ui, |ui| {
                ui.label("method");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.method, FitMethod::Quadratic, "quadratic");
                    ui.selectable_value(&mut self.method, FitMethod::Gaussian, "gaussian");
                });
                ui.end_row();

                ui.label("fit points");
                ui.add(egui::DragValue::new(&mut self.points).range(3..=99));
                ui.end_row();

                ui.label("energy range");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.start_e).speed(0.001));
                    ui.label("…");
                    ui.add(egui::DragValue::new(&mut self.stop_e).speed(0.001));
                });
                ui.end_row();

                ui.label("smoothing");
                egui::ComboBox::from_id_salt("xanes_smooth")
                    .selected_text(smooth_label(self.smooth))
                    .show_ui(ui, |ui| {
                        for a in [
                            SmoothAlgo::None,
                            SmoothAlgo::SavGol,
                            SmoothAlgo::Median,
                            SmoothAlgo::ThreePoint,
                            SmoothAlgo::Boxcar,
                        ] {
                            ui.selectable_value(&mut self.smooth, a, smooth_label(a));
                        }
                    });
                ui.end_row();

                if self.smooth != SmoothAlgo::None {
                    ui.label("smooth width");
                    ui.add(egui::DragValue::new(&mut self.smooth_width).range(1..=99));
                    ui.end_row();
                }
                if self.smooth == SmoothAlgo::SavGol {
                    ui.label("poly order");
                    ui.add(egui::DragValue::new(&mut self.smooth_order).range(1..=9));
                    ui.end_row();
                }

                ui.label("mask threshold");
                ui.add(egui::DragValue::new(&mut self.mask_threshold).speed(0.001))
                    .on_hover_text("skip voxels whose mean absorption is below this (0 = fit all)");
                ui.end_row();

                ui.label("band slices");
                ui.add(egui::DragValue::new(&mut self.band_size).range(1..=256))
                    .on_hover_text("z-slices fitted per streamed band");
                ui.end_row();
            });
        });

        ui.add_enabled_ui(!running, |ui| {
            egui::CollapsingHeader::new("Magnification correction")
                .default_open(false)
                .show(ui, |ui| {
                    ui.checkbox(
                        &mut self.mag_correct,
                        "correct per-energy zone-plate magnification",
                    )
                    .on_hover_text(
                        "focal length grows with energy, so each energy images the \
                         sample at a slightly different magnification; rescale every \
                         energy's volume about its centre before fitting. z-coupled, \
                         so this reads the whole stack into memory.",
                    );
                    ui.add_enabled_ui(self.mag_correct, |ui| {
                        egui::Grid::new("xanes_mag_grid").show(ui, |ui| {
                            ui.label("magnification");
                            ui.add(
                                egui::DragValue::new(&mut self.mag.magnification)
                                    .speed(0.01)
                                    .range(1.0..=10000.0),
                            );
                            ui.end_row();

                            ui.label("ZP diameter (µm)");
                            let mut d_um = self.mag.zp_diameter_m * 1e6;
                            if ui
                                .add(
                                    egui::DragValue::new(&mut d_um)
                                        .speed(1.0)
                                        .range(1.0..=100_000.0),
                                )
                                .changed()
                            {
                                self.mag.zp_diameter_m = d_um * 1e-6;
                            }
                            ui.end_row();

                            ui.label("outer zone (nm)");
                            let mut w_nm = self.mag.zp_outermost_width_m * 1e9;
                            if ui
                                .add(
                                    egui::DragValue::new(&mut w_nm)
                                        .speed(0.5)
                                        .range(1.0..=10_000.0),
                                )
                                .changed()
                            {
                                self.mag.zp_outermost_width_m = w_nm * 1e-9;
                            }
                            ui.end_row();
                        });
                        // Preview the resulting per-energy factors (relative to the
                        // first energy) so the correction is not a black box.
                        if self.energies.len() >= 2 {
                            let cf = magnification_corr_factors(&self.energies, &self.mag);
                            let lo = cf.iter().copied().fold(f64::INFINITY, f64::min);
                            let hi = cf.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            ui.label(format!(
                                "factors {lo:.4}…{hi:.4} over {} energies",
                                cf.len()
                            ));
                        }
                    });
                });
        });

        ui.separator();
        ui.horizontal(|ui| {
            if running {
                if ui.button("Cancel").clicked()
                    && let Some(c) = &self.cancel
                {
                    c.cancel();
                }
            } else if ui.button("Run fit").clicked() {
                let ctx = ui.ctx().clone();
                self.spawn_fit(&ctx, log);
            }
        });
        if running {
            let (done, total) = self.progress;
            ui.add(
                egui::ProgressBar::new(done as f32 / total.max(1) as f32)
                    .text(format!("{done}/{total} slices")),
            );
        }

        ui.separator();
        ui.heading("Peak-energy histogram");
        self.hist_plot.show(ui);

        ui.separator();
        ui.heading("Spectrum");
        self.spectrum_plot.show(ui);

        ui.separator();
        ui.heading("Save result");
        ui.add_enabled_ui(self.map.is_some() && !running, |ui| {
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.save_path)
                        .hint_text("result.h5")
                        .desired_width(200.0),
                );
                if ui.button("…").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("HDF5", &["h5", "hdf5"])
                        .set_file_name("xanes_result.h5")
                        .save_file()
                {
                    self.save_path = path.display().to_string();
                }
            });
            if ui.button("Save peak map").clicked() {
                self.save(log);
            }
        });
    }

    fn save(&mut self, log: &mut Vec<String>) {
        let Some(map) = &self.map else { return };
        let Some(edge_jump) = &self.edge_jump else {
            return;
        };
        if self.save_path.is_empty() {
            log.push("xanes: set a save path first".into());
            return;
        }
        match write_peak_map_h5(
            &self.save_path,
            &self.energies,
            map.view(),
            edge_jump.view(),
        ) {
            Ok(()) => log.push(format!("xanes: saved peak map → {}", self.save_path)),
            Err(e) => log.push(format!("xanes: save FAILED — {e}")),
        }
    }
}

/// Resolve a combined `registered.h5` into a [`MultiEnergyVolume`].
///
/// Discovers the per-energy volumes by listing datasets and parsing the leaf of
/// each `reconstructions/{energy}` key as its energy — so the exact float
/// formatting the writer used is never assumed. Returns the volume and a short
/// description.
fn resolve_registered_stack(path: &Path) -> Result<(MultiEnergyVolume, String), String> {
    let p = path.to_string_lossy();
    let names = tomoxide::io::list_h5_datasets(&p).map_err(|e| e.to_string())?;
    let mut entries: Vec<(f64, String)> = names
        .iter()
        .filter_map(|n| {
            let key = n.strip_prefix('/').unwrap_or(n);
            let rest = key.strip_prefix("reconstructions/")?;
            // The energy is the first path segment after the prefix.
            let energy: f64 = rest.split('/').next()?.parse().ok()?;
            Some((energy, key.to_string()))
        })
        .collect();
    if entries.is_empty() {
        return Err(format!(
            "{}: no reconstructions/{{energy}} datasets found",
            path.display()
        ));
    }
    entries.sort_by(|a, b| a.0.total_cmp(&b.0));
    let volume = MultiEnergyVolume::from_combined(path, &entries).map_err(|e| e.to_string())?;
    let (ne, nz, ny, nx) = volume.dims();
    Ok((
        volume,
        format!("registered — {ne} energies, {nz}×{ny}×{nx}"),
    ))
}

fn smooth_label(a: SmoothAlgo) -> &'static str {
    match a {
        SmoothAlgo::None => "none",
        SmoothAlgo::SavGol => "Savitzky–Golay",
        SmoothAlgo::Median => "median",
        SmoothAlgo::ThreePoint => "3-point",
        SmoothAlgo::Boxcar => "boxcar",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small in-range map (edge < `MAX_3D_DIM`) is uploaded 1:1, finite voxels
    /// are opaque, and a NaN voxel stays fully transparent.
    #[test]
    fn volume_transfer_marks_finite_opaque_and_nan_transparent() {
        let (nz, ny, nx) = (2, 2, 2);
        let mut map = Array3::from_elem((nz, ny, nx), 0.5_f64);
        map[[1, 1, 1]] = f64::NAN; // unfitted voxel
        let (rgba, d, h, w) = build_volume_rgba(&map, (0.0, 1.0));
        assert_eq!((d, h, w), (nz, ny, nx), "small map is not decimated");
        assert_eq!(rgba.len(), nz * ny * nx * 4);
        // Finite voxel (0,0,0) opaque; NaN voxel (1,1,1) — last one — transparent.
        assert_eq!(rgba[3], 255, "finite voxel must be opaque");
        assert_eq!(
            rgba[rgba.len() - 1],
            0,
            "NaN voxel must be fully transparent"
        );
    }

    /// A map whose longest edge exceeds `MAX_3D_DIM` is decimated so every edge
    /// fits, and the RGBA buffer matches the decimated dimensions.
    #[test]
    fn volume_transfer_decimates_oversized_maps() {
        let big = MAX_3D_DIM * 2 + 1; // forces stride 3 (div_ceil)
        let map = Array3::from_elem((big, 1, 1), 0.5_f64);
        let (rgba, d, h, w) = build_volume_rgba(&map, (0.0, 1.0));
        assert!(d <= MAX_3D_DIM, "decimated depth {d} must fit MAX_3D_DIM");
        assert_eq!((h, w), (1, 1));
        assert_eq!(d, big.div_ceil(3), "stride-3 decimation along z");
        assert_eq!(rgba.len(), d * h * w * 4);
    }
}
