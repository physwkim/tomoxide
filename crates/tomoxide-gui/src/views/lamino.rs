//! Laminography alignment (docs/LAMINOGRAPHY_ALIGNMENT.md), the second half of
//! the Center screen.
//!
//! **Why this is not the Center screen's auto buttons.** Under a laminographic
//! tilt the axis has a component along the beam, so a 180° rotation is not a
//! mirror of the object and the 0°/180° symmetry that `find_center_vo`,
//! `find_center_pc` and `find_center_sift` all assume is gone — on the scan the
//! doc is written from, mirror registration scattered 395…607 against a known
//! 396. So the beam does not merely add a parameter here, it invalidates the
//! estimators next door. The toggle picks the family that is valid for the beam.
//!
//! The three steps below are the doc's, in its order, and the screen exists
//! because each one is a *picture* the CLI can only summarise as a number:
//!
//! 1. **Rings** — the mean projection is a bullseye centred on the axis. §1 says
//!    read it by eye, and treat eye-vs-correlation disagreement as the
//!    misalignment flag: closed rings mean the scan was aligned at acquisition,
//!    arcs that never close mean it was not, and no reconstruction geometry
//!    repairs that. The number cannot show you an arc.
//! 2. **Centre** — one probe launch sweeps every candidate, because an in-plane
//!    shift leaves the in-focus layer where it is and its response is sharp
//!    (~40 % per 50 px).
//! 3. **Tilt** — a FULL reconstruction per candidate, scored by the max focus
//!    over every slice (§2). The tilt drags the in-focus layer through z, so a
//!    fixed slice cannot rank it. Minutes per candidate, hence the cancel.

use std::sync::mpsc::Sender;

use rsplot::egui_wgpu::RenderState;
use rsplot::{CurveData, Frame, ImageStack, ImageView, ItemHandle, Plot1D, egui};

use tomoxide::recon::center::{SweepVerdict, judge_sweep};

use crate::worker::{DatasetMeta, Job};

/// A finished sweep over one axis: candidates and their focus.
struct Sweep {
    cands: Vec<f32>,
    focus: Vec<f64>,
}

impl Sweep {
    /// What the curve established — an answer, or one of the ways it has none.
    fn judge(&self) -> Option<SweepVerdict> {
        judge_sweep(&self.cands, &self.focus)
    }
}

/// The verdict as a line to show: colour and text. `None` when the sweep resolved
/// nothing *and* has nothing to say, which cannot happen — every variant speaks.
fn verdict_line(
    v: &SweepVerdict,
    axis: &str,
    unit: &str,
    cands: &[f32],
    scored_on: &str,
) -> (egui::Color32, String) {
    let (lo, hi) = (cands[0], cands[cands.len() - 1]);
    match v {
        SweepVerdict::Resolved { value, .. } => (
            egui::Color32::LIGHT_GREEN,
            format!("Sharpest at {axis} {value:.2}{unit}{scored_on}."),
        ),
        SweepVerdict::Railed { value, .. } => (
            egui::Color32::LIGHT_RED,
            format!(
                "The sweep peaked at {value:.2}{unit} without ever coming back down inside its \
                 own range [{lo:.2}, {hi:.2}] — that is the range running out, not an optimum. \
                 Widen ± or recentre it."
            ),
        ),
        SweepVerdict::Ambiguous { value, rivals, .. } => (
            egui::Color32::LIGHT_RED,
            format!(
                "No answer over [{lo:.2}, {hi:.2}]: {value:.2}{unit} won but {:.2}{unit} is only \
                 {:.2} % behind, and neither is at an edge — the metric does not separate them. \
                 This sweep refines a prior, it does not find one: narrow ± around an axis you \
                 trust (step 1), and confirm by eye.",
                rivals[0].value,
                rivals[0].deficit * 100.0,
            ),
        ),
        SweepVerdict::Flat { .. } => (
            egui::Color32::LIGHT_RED,
            format!(
                "Every one of the {} candidates scored the same. The metric never responded, so \
                 no range will help — the reconstruction is uniform, or the focus is being \
                 measured on empty field.",
                cands.len()
            ),
        ),
    }
}

/// What the ring step found, plus the image it found it in.
struct Rings {
    center: f32,
    prominence: f32,
    trustworthy: bool,
    bytes: usize,
}

/// One finished tilt candidate, including the in-focus slice its reconstruction
/// peaked on — kept here because the montage is rebuilt as candidates land and
/// `ImageStack` does not hand its frames back.
struct Tilt {
    tilt_deg: f32,
    focus: f64,
    z_peak: usize,
    depth: usize,
    focus_by_z: Vec<f64>,
    ny: usize,
    nx: usize,
    slice: Vec<f32>,
}

pub struct LaminoView {
    /// Ring-estimate subsampling: average every Nth projection.
    ring_step: usize,
    rings: Option<Rings>,
    rings_pending: bool,
    mean_view: ImageView,

    /// The working rotation axis, seeded from the rings and refined by step 2.
    center: f32,
    /// The working tilt, seeded by hand and refined by step 3.
    tilt: f32,
    /// Output slice the centre sweep scores; `None` ⇒ the middle of the volume.
    slice: Option<usize>,

    center_half: f32,
    center_step: f32,
    center_pending: bool,
    center_sweep: Option<Sweep>,
    center_slice: usize,
    center_stack: ImageStack,
    center_plot: Plot1D,
    center_curve: Option<ItemHandle>,

    tilt_half: f32,
    tilt_step: f32,
    /// `Some((done, total))` while a scan is running.
    tilt_progress: Option<(usize, usize)>,
    tilts: Vec<Tilt>,
    tilt_stack: ImageStack,
    tilt_plot: Plot1D,
    tilt_curve: Option<ItemHandle>,
    /// Focus of every slice of the selected tilt's reconstruction — the profile
    /// that says whether `z_peak` is a hump on the sample or an edge spike.
    z_plot: Plot1D,
    z_curve: Option<ItemHandle>,
    /// Set by "Use in Tune"; the app shell takes it and applies it.
    accepted: Option<f32>,
}

impl LaminoView {
    pub fn new(render_state: &RenderState) -> Self {
        let mut mean_view = ImageView::new(render_state, 150);
        mean_view.set_side_histogram_displayed(false);
        mean_view
            .image_plot_mut()
            .set_graph_title("mean projection — the rings are centred on the axis");
        *mean_view.position_info_mut() = rsplot::PositionInfo::new(Vec::new());

        let mut center_stack = ImageStack::new(render_state, 152);
        center_stack.set_table_visible(false);
        let mut center_plot = Plot1D::new(render_state, 154);
        center_plot.set_graph_title("focus vs centre — click to pick");

        let mut tilt_stack = ImageStack::new(render_state, 156);
        tilt_stack.set_table_visible(false);
        let mut tilt_plot = Plot1D::new(render_state, 158);
        tilt_plot.set_graph_title("focus vs tilt — click to pick");
        let mut z_plot = Plot1D::new(render_state, 160);
        z_plot.set_graph_title("focus by slice");

        LaminoView {
            ring_step: 10,
            rings: None,
            rings_pending: false,
            mean_view,
            center: 0.0,
            tilt: 45.0,
            slice: None,
            center_half: 8.0,
            center_step: 0.25,
            center_pending: false,
            center_sweep: None,
            center_slice: 0,
            center_stack,
            center_plot,
            center_curve: None,
            tilt_half: 3.0,
            tilt_step: 1.0,
            tilt_progress: None,
            tilts: Vec::new(),
            tilt_stack,
            tilt_plot,
            tilt_curve: None,
            z_plot,
            z_curve: None,
            accepted: None,
        }
    }

    pub fn on_dataset(&mut self, meta: &DatasetMeta) {
        self.center = meta.nx as f32 / 2.0;
        self.slice = None;
        self.rings = None;
        self.rings_pending = false;
        self.center_pending = false;
        self.center_sweep = None;
        self.center_stack.set_frames(Vec::new());
        self.tilt_progress = None;
        self.tilts.clear();
        self.tilt_stack.set_frames(Vec::new());
    }

    #[allow(clippy::too_many_arguments)]
    pub fn on_rings(
        &mut self,
        center: f32,
        prominence: f32,
        trustworthy: bool,
        ny: usize,
        nx: usize,
        mean: &[f32],
        bytes: usize,
    ) {
        self.rings_pending = false;
        let _ =
            self.mean_view
                .set_image(nx as u32, ny as u32, mean, super::autoscale_viridis(mean));
        // Seed the centre, but only from an estimate worth seeding with: on a
        // mis-aligned scan the lopsided arcs drag the correlation peak (281.7
        // against a known 138), so adopting it silently would hand step 2 a
        // starting guess that is 143 px out.
        if trustworthy {
            self.center = center;
        }
        self.rings = Some(Rings {
            center,
            prominence,
            trustworthy,
            bytes,
        });
    }

    pub fn on_center_sweep(
        &mut self,
        centers: Vec<f32>,
        ny: usize,
        nx: usize,
        frames: &[f32],
        focus: Vec<f64>,
        slice: usize,
    ) {
        self.center_pending = false;
        if centers.is_empty() || ny * nx == 0 {
            return;
        }
        let size = ny * nx;
        let cmap = super::autoscale_viridis(frames);
        let stack_frames: Vec<Option<Frame>> = centers
            .iter()
            .enumerate()
            .map(|(i, c)| {
                Some(Frame::new(
                    nx as u32,
                    ny as u32,
                    frames[i * size..(i + 1) * size].to_vec(),
                    Some(format!("centre {c:.2}")),
                ))
            })
            .collect();
        self.center_stack.set_frames(stack_frames);
        self.center_stack.set_colormap(cmap);
        self.center_slice = slice;

        let x: Vec<f64> = centers.iter().map(|&c| c as f64).collect();
        let curve = CurveData::new(x, focus.clone(), egui::Color32::LIGHT_BLUE);
        match self.center_curve {
            Some(h) => {
                self.center_plot.update_curve_data(h, &curve);
            }
            None => {
                self.center_curve = Some(
                    self.center_plot
                        .add_curve_data_with_legend(&curve, "mean |∇|² in a 0.92-FOV disk"),
                );
            }
        }
        let sweep = Sweep {
            cands: centers,
            focus,
        };
        if let Some(v) = sweep.judge() {
            self.center_stack.set_current(v.best().0);
            // Only a resolved sweep hands back a value; a railed, ambiguous or
            // flat one is shown, never adopted.
            if let Some(value) = v.resolved() {
                self.center = value;
            }
        }
        self.center_sweep = Some(sweep);
    }

    /// One tilt candidate landed. Results arrive one full reconstruction apart,
    /// so the screen grows as they come rather than waiting for the set.
    #[allow(clippy::too_many_arguments)]
    pub fn on_tilt(
        &mut self,
        tilt_deg: f32,
        focus: f64,
        z_peak: usize,
        depth: usize,
        focus_by_z: Vec<f64>,
        ny: usize,
        nx: usize,
        slice: &[f32],
        done: usize,
        total: usize,
    ) {
        self.tilt_progress = Some((done, total));
        self.tilts.push(Tilt {
            tilt_deg,
            focus,
            z_peak,
            depth,
            focus_by_z,
            ny,
            nx,
            slice: slice.to_vec(),
        });
        let frames: Vec<Option<Frame>> = self
            .tilts
            .iter()
            .map(|t| {
                Some(Frame::new(
                    t.nx as u32,
                    t.ny as u32,
                    t.slice.clone(),
                    Some(format!(
                        "{:.2}° — slice {} of {}",
                        t.tilt_deg, t.z_peak, t.depth
                    )),
                ))
            })
            .collect();
        self.tilt_stack.set_frames(frames);
        self.tilt_stack
            .set_colormap(super::autoscale_viridis(slice));
        self.tilt_stack.set_current(self.tilts.len() - 1);
        self.refresh_tilt_curves();
    }

    pub fn on_tilt_done(&mut self, _cancelled: bool) {
        self.tilt_progress = None;
        if let Some(value) = self
            .tilt_sweep()
            .and_then(|s| s.judge())
            .and_then(|v| v.resolved())
        {
            self.tilt = value;
        }
    }

    pub fn on_failed(&mut self) {
        self.rings_pending = false;
        self.center_pending = false;
        self.tilt_progress = None;
    }

    pub fn take_accepted(&mut self) -> Option<f32> {
        self.accepted.take()
    }

    fn tilt_sweep(&self) -> Option<Sweep> {
        if self.tilts.is_empty() {
            return None;
        }
        Some(Sweep {
            cands: self.tilts.iter().map(|t| t.tilt_deg).collect(),
            focus: self.tilts.iter().map(|t| t.focus).collect(),
        })
    }

    fn refresh_tilt_curves(&mut self) {
        let Some(sweep) = self.tilt_sweep() else {
            return;
        };
        let x: Vec<f64> = sweep.cands.iter().map(|&t| t as f64).collect();
        let curve = CurveData::new(x, sweep.focus.clone(), egui::Color32::LIGHT_GREEN);
        match self.tilt_curve {
            Some(h) => {
                self.tilt_plot.update_curve_data(h, &curve);
            }
            None => {
                self.tilt_curve = Some(
                    self.tilt_plot
                        .add_curve_data_with_legend(&curve, "max focus over all slices"),
                );
            }
        }
        self.refresh_z_curve();
    }

    /// Focus by slice for the tilt the montage is showing. This is the panel that
    /// separates a real optimum from the failure the method exists to avoid: a
    /// broad hump over the sample versus a spike at a z-edge, where few
    /// projections contribute and the streaks are the gradient.
    fn refresh_z_curve(&mut self) {
        let i = self.tilt_stack.current();
        let Some(t) = self.tilts.get(i) else {
            return;
        };
        let x: Vec<f64> = (0..t.focus_by_z.len()).map(|z| z as f64).collect();
        let curve = CurveData::new(x, t.focus_by_z.clone(), egui::Color32::LIGHT_RED);
        match self.z_curve {
            Some(h) => {
                self.z_plot.update_curve_data(h, &curve);
            }
            None => {
                self.z_curve = Some(
                    self.z_plot
                        .add_curve_data_with_legend(&curve, "focus of each slice"),
                );
            }
        }
        self.z_plot.set_graph_title(format!(
            "focus by slice at {:.2}° — peak at {} of {}",
            t.tilt_deg, t.z_peak, t.depth
        ));
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        jobs: &Sender<Job>,
        cancel: &std::sync::atomic::AtomicBool,
        meta: Option<&std::sync::Arc<DatasetMeta>>,
    ) {
        let Some(meta) = meta.cloned() else {
            ui.label("Open a dataset on the Data screen first.");
            return;
        };
        egui::ScrollArea::vertical().show(ui, |ui| {
            self.step_rings(ui, jobs, &meta);
            ui.separator();
            self.step_center(ui, jobs);
            ui.separator();
            self.step_tilt(ui, jobs, cancel);
        });
    }

    fn step_rings(
        &mut self,
        ui: &mut egui::Ui,
        jobs: &Sender<Job>,
        meta: &std::sync::Arc<DatasetMeta>,
    ) {
        ui.heading("1 · Rings — was the scan aligned at acquisition?");
        ui.label(
            "Over a full turn every point traces a circle around the axis, so the mean \
             projection is a bullseye centred on it. Read the centre column by eye; the \
             number below is the second opinion.",
        );
        ui.horizontal(|ui| {
            ui.label("average every");
            ui.add(egui::DragValue::new(&mut self.ring_step).range(1..=100));
            ui.label("th projection");
            let bytes = meta.nproj * meta.nz * meta.nx * 4;
            let idle = !self.rings_pending;
            if ui
                .add_enabled(idle, egui::Button::new("Read the rings"))
                .on_hover_text(format!(
                    "Loads and preps the whole projection stack — {:.1} GB of host RAM, \
                     kept for steps 2 and 3. Every step here consumes all the projections, \
                     so there is no band to read lazily.",
                    bytes as f64 / 1e9
                ))
                .clicked()
                && jobs
                    .send(Job::LaminoRings {
                        step: self.ring_step,
                    })
                    .is_ok()
            {
                self.rings_pending = true;
            }
            if self.rings_pending {
                ui.spinner();
                ui.label("loading + prepping the whole stack…");
            }
        });

        if let Some(r) = &self.rings {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "centre {:.2}   prominence {:.2} (trustworthy ≥ 8.0)   stack {:.1} GB",
                    r.center,
                    r.prominence,
                    r.bytes as f64 / 1e9
                ));
            });
            if r.trustworthy {
                ui.colored_label(
                    egui::Color32::LIGHT_GREEN,
                    "Closed concentric rings — the estimate is worth refining.",
                );
            } else {
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    "NOT trustworthy — no bullseye, so the rings never closed. That is the \
                     signature of a scan mis-aligned at acquisition, which no reconstruction \
                     geometry repairs. Look at the image before spending a sweep on it.",
                );
            }
            ui.allocate_ui(egui::vec2(ui.available_width(), 320.0), |ui| {
                super::show_image_view_with_value(ui, &mut self.mean_view);
            });
        }
    }

    fn step_center(&mut self, ui: &mut egui::Ui, jobs: &Sender<Job>) {
        ui.heading("2 · Centre — one launch for the whole sweep");
        ui.label(
            "The centre is an in-plane shift: it leaves the in-focus layer where it is, and \
             its response is sharp, so a single slice ranks every candidate.",
        );
        ui.horizontal(|ui| {
            ui.label("tilt");
            ui.add(egui::DragValue::new(&mut self.tilt).speed(0.1).suffix("°"));
            ui.label("centre");
            ui.add(egui::DragValue::new(&mut self.center).speed(0.1));
            ui.label("± ");
            ui.add(egui::DragValue::new(&mut self.center_half).speed(0.5));
            ui.label("step");
            ui.add(
                egui::DragValue::new(&mut self.center_step)
                    .speed(0.05)
                    .range(0.01..=8.0),
            );
        });
        ui.horizontal(|ui| {
            let mut fixed = self.slice.is_some();
            if ui
                .checkbox(&mut fixed, "slice")
                .on_hover_text(
                    "The output slice the sweep scores. Default: the middle of the volume, \
                     which under a tilt is rh/2 and NOT the detector's middle row — the tilt \
                     stretches the reconstruction deeper than the detector is tall. Set it by \
                     hand when the sample is not on the default plane; a flat focus curve is \
                     how an empty slice looks.",
                )
                .changed()
            {
                self.slice = fixed.then_some(0);
            }
            if let Some(s) = &mut self.slice {
                ui.add(egui::DragValue::new(s));
            } else {
                ui.label("auto (middle of the volume)");
            }
            let idle = !self.center_pending && self.rings.is_some();
            if ui
                .add_enabled(idle, egui::Button::new("Sweep the centre"))
                .on_hover_text(if self.rings.is_some() {
                    "One probe launch for every candidate."
                } else {
                    "Read the rings first — they load the stack this sweep runs on."
                })
                .clicked()
            {
                let cands = grid(self.center, self.center_half, self.center_step);
                if jobs
                    .send(Job::LaminoCenterSweep {
                        tilt_deg: Some(self.tilt),
                        slice: self.slice,
                        centers: cands,
                    })
                    .is_ok()
                {
                    self.center_pending = true;
                }
            }
            if self.center_pending {
                ui.spinner();
            }
        });

        let Some(sweep) = &self.center_sweep else {
            return;
        };
        if let Some(v) = sweep.judge() {
            let scored_on = format!(" (scored on slice {})", self.center_slice);
            let (colour, text) = verdict_line(&v, "centre", "", &sweep.cands, &scored_on);
            ui.colored_label(colour, text);
        }
        ui.horizontal(|ui| {
            ui.allocate_ui(egui::vec2(ui.available_width() * 0.5, 300.0), |ui| {
                self.center_stack.ui(ui);
            });
            ui.allocate_ui(egui::vec2(ui.available_width(), 300.0), |ui| {
                let resp = self.center_plot.show(ui);
                if resp.response.clicked()
                    && let Some(pos) = resp.response.interact_pointer_pos()
                {
                    let (x, _y) = resp.transform.pixel_to_data(pos);
                    if let Some(sweep) = &self.center_sweep
                        && let Some(i) = nearest_index(&sweep.cands, x as f32)
                    {
                        self.center_stack.set_current(i);
                    }
                }
            });
        });
    }

    fn step_tilt(
        &mut self,
        ui: &mut egui::Ui,
        jobs: &Sender<Job>,
        cancel: &std::sync::atomic::AtomicBool,
    ) {
        use std::sync::atomic::Ordering;

        ui.heading("3 · Tilt — a full reconstruction per candidate");
        ui.label(
            "The tilt drags the in-focus layer through z (measured: slice 800 → 1120 as the \
             tilt went 40° → 58°) while its own response is only ~2 % per degree, so a fixed \
             slice scores a plane whose error swamps the signal. Only the max over the whole \
             z range ranks it — and that needs the whole reconstruction.",
        );
        let cands = grid(self.tilt, self.tilt_half, self.tilt_step);
        ui.horizontal(|ui| {
            ui.label("tilt");
            ui.add(egui::DragValue::new(&mut self.tilt).speed(0.1).suffix("°"));
            ui.label("± ");
            ui.add(egui::DragValue::new(&mut self.tilt_half).speed(0.5));
            ui.label("step");
            ui.add(
                egui::DragValue::new(&mut self.tilt_step)
                    .speed(0.1)
                    .range(0.1..=10.0),
            );
            ui.label(format!("= {} reconstructions", cands.len()));
        });
        ui.horizontal(|ui| {
            let idle = self.tilt_progress.is_none() && self.rings.is_some();
            if ui
                .add_enabled(idle, egui::Button::new("Scan the tilt"))
                .on_hover_text(
                    "Minutes per candidate. Do the centre first: the tilt is scored at the \
                     centre below, and a wrong centre blurs every candidate equally.",
                )
                .clicked()
            {
                self.tilts.clear();
                self.tilt_stack.set_frames(Vec::new());
                if jobs
                    .send(Job::LaminoTiltScan {
                        center: self.center,
                        tilts: cands,
                    })
                    .is_ok()
                {
                    self.tilt_progress = Some((0, 0));
                }
            }
            ui.label(format!("at centre {:.2}", self.center));
            if let Some((done, total)) = self.tilt_progress {
                ui.spinner();
                ui.label(format!("{done} of {total}"));
                if ui.button("Cancel").clicked() {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
        });

        if self.tilts.is_empty() {
            return;
        }
        if let Some(v) = self.tilt_sweep().and_then(|s| s.judge()) {
            let cands: Vec<f32> = self.tilts.iter().map(|t| t.tilt_deg).collect();
            let (colour, text) = verdict_line(&v, "tilt", "°", &cands, "");
            ui.horizontal(|ui| {
                ui.colored_label(colour, text);
                // The centre is the screen's output, and it is worth carrying over
                // whatever the tilt scan concluded — but only if the centre step
                // itself resolved one.
                if v.resolved().is_some()
                    && ui
                        .button("Use centre in Tune")
                        .on_hover_text(
                            "Confirm on the montage first: at a wrong centre the particles \
                             smear into dashes, at the right one they are round.",
                        )
                        .clicked()
                {
                    self.accepted = Some(self.center);
                }
            });
        }
        ui.horizontal(|ui| {
            ui.allocate_ui(egui::vec2(ui.available_width() * 0.5, 300.0), |ui| {
                let before = self.tilt_stack.current();
                self.tilt_stack.ui(ui);
                if self.tilt_stack.current() != before {
                    self.refresh_z_curve();
                }
            });
            ui.vertical(|ui| {
                ui.allocate_ui(egui::vec2(ui.available_width(), 148.0), |ui| {
                    let resp = self.tilt_plot.show(ui);
                    if resp.response.clicked()
                        && let Some(pos) = resp.response.interact_pointer_pos()
                    {
                        let (x, _y) = resp.transform.pixel_to_data(pos);
                        let cands: Vec<f32> = self.tilts.iter().map(|t| t.tilt_deg).collect();
                        if let Some(i) = nearest_index(&cands, x as f32) {
                            self.tilt_stack.set_current(i);
                            self.refresh_z_curve();
                        }
                    }
                });
                ui.allocate_ui(egui::vec2(ui.available_width(), 148.0), |ui| {
                    self.z_plot.show(ui);
                });
            });
        });
    }
}

/// Candidates covering `center ± half` in `step` increments, `center` included.
/// Shared shape with the CLI's `align`, so a sweep here and a sweep there ask the
/// same question.
fn grid(center: f32, half: f32, step: f32) -> Vec<f32> {
    if step <= 0.0 || half < 0.0 {
        return vec![center];
    }
    let n = (half / step).floor() as i32;
    (-n..=n).map(|k| center + k as f32 * step).collect()
}

/// Index of the candidate closest to `x` (`None` on an empty list).
fn nearest_index(cands: &[f32], x: f32) -> Option<usize> {
    cands
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - x).abs().total_cmp(&(*b - x).abs()))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_centred_and_symmetric() {
        assert_eq!(grid(10.0, 1.0, 0.5), vec![9.0, 9.5, 10.0, 10.5, 11.0]);
        // A step wider than the half-width leaves only the centre itself.
        assert_eq!(grid(10.0, 0.2, 0.5), vec![10.0]);
        assert_eq!(grid(10.0, 1.0, 0.0), vec![10.0]);
    }

    #[test]
    fn nearest_index_snaps_to_closest_candidate() {
        let cands = [44.0_f32, 45.0, 46.0];
        assert_eq!(nearest_index(&cands, 45.4), Some(1));
        assert_eq!(nearest_index(&cands, 0.0), Some(0));
        assert_eq!(nearest_index(&[], 1.0), None);
    }

    /// The verdict a sweep cannot answer must read as a refusal that names its
    /// rival — not as a quieter version of the sharpest-at line. This is the
    /// screen's half of the defect: the old UI printed the ±40 px winner (417) in
    /// green because it was not at an edge.
    #[test]
    fn an_unresolved_sweep_never_reads_as_an_answer() {
        let cands: Vec<f32> = (0..81).map(|k| 356.0 + k as f32).collect();
        let ambiguous = SweepVerdict::Ambiguous {
            index: 61,
            value: 417.0,
            rivals: vec![tomoxide::recon::center::Rival {
                index: 40,
                value: 396.0,
                prominence: 1.77e-7,
                deficit: 0.00335,
            }],
        };
        let (colour, text) = verdict_line(&ambiguous, "centre", "", &cands, "");
        assert_eq!(colour, egui::Color32::LIGHT_RED);
        assert!(text.contains("No answer"), "{text}");
        assert!(text.contains("396"), "the rival is not named: {text}");
        assert!(!text.contains("Sharpest"), "{text}");

        let (colour, text) = verdict_line(
            &SweepVerdict::Resolved {
                index: 40,
                value: 396.0,
            },
            "centre",
            "",
            &cands,
            "",
        );
        assert_eq!(colour, egui::Color32::LIGHT_GREEN);
        assert!(text.contains("Sharpest at centre 396.00"), "{text}");
    }
}
