//! Center mode (docs/GUI.md §2 Center): rotation-axis finding — auto methods,
//! the candidate-sweep montage (`write_center` browser, the Octopus
//! parameter-evaluator concept), and ±0.5/±0.25 px fine tweak (tomostream
//! CenterTweak concept).

use std::sync::Arc;
use std::sync::mpsc::Sender;

use siplot::egui_wgpu::RenderState;
use siplot::{CurveData, Frame, ImageStack, ItemHandle, Plot1D, egui};

use crate::worker::{CenterMethod, DatasetMeta, Job};

/// One auto-detection result kept in the on-screen history.
struct Estimate {
    method: CenterMethod,
    center: f32,
    millis: u128,
}

/// A finished sweep: candidate centers and their per-frame sharpness metric
/// (the frames themselves live in the [`ImageStack`]).
struct Sweep {
    centers: Vec<f32>,
    /// Standard deviation of each trial reconstruction — sharper (better
    /// centered) frames have more contrast (docs/GUI.md §2.3).
    metric: Vec<f64>,
    step: f32,
}

pub struct CenterView {
    /// Sinogram row fed to the Vo/Entropy methods and the sweep.
    row: usize,
    /// The working center value: latest estimate, then hand-tweaked.
    candidate: f32,
    /// Method of the in-flight job (one at a time).
    pending: Option<CenterMethod>,
    history: Vec<Estimate>,
    /// Set by "Use in Tune"; the app shell takes it and applies it.
    accepted: Option<f32>,

    // --- sweep montage ---
    /// Sweep half-range (px): candidates cover `candidate ± half`.
    sweep_half: f32,
    /// Candidate spacing (px).
    sweep_step: f32,
    sweep_pending: bool,
    sweep: Option<Sweep>,
    stack: ImageStack,
    metric_plot: Plot1D,
    metric_curve: Option<ItemHandle>,
}

impl CenterView {
    pub fn new(render_state: &RenderState) -> Self {
        let mut stack = ImageStack::new(render_state, 50);
        stack.set_table_visible(false);
        let mut metric_plot = Plot1D::new(render_state, 60);
        metric_plot.set_graph_title("sharpness (std) — click to pick");
        CenterView {
            row: 0,
            candidate: 0.0,
            pending: None,
            history: Vec::new(),
            accepted: None,
            // tomopy write_center default range shape: ±5 px in 0.5 steps.
            sweep_half: 5.0,
            sweep_step: 0.5,
            sweep_pending: false,
            sweep: None,
            stack,
            metric_plot,
            metric_curve: None,
        }
    }

    pub fn on_dataset(&mut self, meta: &DatasetMeta) {
        self.row = meta.nz / 2;
        self.candidate = (meta.nx as f32) / 2.0;
        self.pending = None;
        self.history.clear();
        self.accepted = None;
        self.sweep_pending = false;
        self.sweep = None;
        self.stack.set_frames(Vec::new());
    }

    pub fn on_center(&mut self, method: CenterMethod, center: f32, millis: u128) {
        self.pending = None;
        self.candidate = center;
        self.history.push(Estimate {
            method,
            center,
            millis,
        });
    }

    /// Route a finished sweep here: build the montage frames and the
    /// sharpness curve, and jump the stack to the sharpest candidate.
    pub fn on_sweep(&mut self, centers: Vec<f32>, ny: usize, nx: usize, frames: &[f32]) {
        self.sweep_pending = false;
        if centers.is_empty() || ny * nx == 0 {
            return;
        }
        let size = ny * nx;
        let cmap = super::autoscale_viridis(frames);
        let mut metric = Vec::with_capacity(centers.len());
        let stack_frames: Vec<Option<Frame>> = centers
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let data = &frames[i * size..(i + 1) * size];
                metric.push(std_dev(data));
                Some(Frame::new(
                    nx as u32,
                    ny as u32,
                    data.to_vec(),
                    Some(format!("center {c:.2}")),
                ))
            })
            .collect();
        self.stack.set_frames(stack_frames);
        self.stack.set_colormap(cmap);

        let x: Vec<f64> = centers.iter().map(|&c| c as f64).collect();
        let curve = CurveData::new(x, metric.clone(), egui::Color32::LIGHT_BLUE);
        match self.metric_curve {
            Some(h) => {
                self.metric_plot.update_curve_data(h, &curve);
            }
            None => {
                self.metric_curve = Some(
                    self.metric_plot
                        .add_curve_data_with_legend(&curve, "std per candidate"),
                );
            }
        }

        // Start on the sharpest frame; picking it is still the user's call.
        let best = metric
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.stack.set_current(best);
        self.sweep = Some(Sweep {
            centers,
            metric,
            step: self.sweep_step,
        });
    }

    /// A worker center job failed: clear the in-flight flags.
    pub fn on_failed(&mut self) {
        self.pending = None;
        self.sweep_pending = false;
    }

    /// Issue a sweep job over `center ± half` in `step` increments. The stop
    /// is nudged by `step/2` so both endpoints land inside the `arange`
    /// (stop-exclusive) enumeration.
    fn send_sweep(&mut self, jobs: &Sender<Job>, center: f32, half: f32, step: f32) {
        if self.sweep_pending || step <= 0.0 || half < step {
            return;
        }
        self.sweep_step = step;
        if jobs
            .send(Job::CenterSweep {
                row: self.row,
                range: (center - half, center + half + step * 0.5, step),
            })
            .is_ok()
        {
            self.sweep_pending = true;
        }
    }

    /// Candidate accepted since the last call (app applies it to Tune).
    pub fn take_accepted(&mut self) -> Option<f32> {
        self.accepted.take()
    }

    fn method_button(
        &mut self,
        ui: &mut egui::Ui,
        jobs: &Sender<Job>,
        method: CenterMethod,
        label: &str,
        hint: &str,
    ) {
        let idle = self.pending.is_none();
        if ui
            .add_enabled(idle, egui::Button::new(label))
            .on_hover_text(hint)
            .clicked()
            && jobs
                .send(Job::FindCenter {
                    method,
                    row: self.row,
                    init: Some(self.candidate),
                })
                .is_ok()
        {
            self.pending = Some(method);
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, jobs: &Sender<Job>, meta: Option<&Arc<DatasetMeta>>) {
        let Some(meta) = meta.cloned() else {
            ui.label("Open a dataset in Data mode first.");
            return;
        };

        egui::Panel::left("center_side")
            .resizable(true)
            .default_size(340.0)
            .show_inside(ui, |ui| self.side_panel(ui, jobs, &meta));

        egui::Panel::bottom("center_metric")
            .resizable(true)
            .default_size(220.0)
            .show_inside(ui, |ui| self.metric_panel(ui));

        // Remaining central space: the candidate montage.
        if self.sweep.is_some() {
            self.stack.ui(ui);
        } else {
            ui.centered_and_justified(|ui| {
                ui.label("Run a sweep to browse trial reconstructions per candidate center.");
            });
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, jobs: &Sender<Job>, meta: &Arc<DatasetMeta>) {
        ui.heading("Rotation axis");
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("sinogram row");
            ui.add(egui::Slider::new(
                &mut self.row,
                0..=meta.nz.saturating_sub(1),
            ));
            ui.label(egui::RichText::new("(Vo / Entropy input)").small().weak());
        });
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            self.method_button(
                ui,
                jobs,
                CenterMethod::Vo,
                "Vo",
                "Nghia Vo's sinogram-domain Fourier method (tomopy find_center_vo)",
            );
            self.method_button(
                ui,
                jobs,
                CenterMethod::Entropy,
                "Entropy",
                "entropy of trial reconstructions, seeded by the current value (tomopy find_center)",
            );
            self.method_button(
                ui,
                jobs,
                CenterMethod::Pc,
                "Phase corr.",
                "phase correlation of the 0°/180° projection pair — reads the whole dataset",
            );
            self.method_button(
                ui,
                jobs,
                CenterMethod::Sift,
                "SIFT",
                "SIFT registration of the 0°/180° pair (sift-center feature, on by default); \
                 reads the whole dataset",
            );
            if let Some(m) = self.pending {
                ui.spinner();
                ui.label(egui::RichText::new(m.label()).small().weak());
            }
        });
        ui.add_space(8.0);
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("center");
            ui.add(
                egui::DragValue::new(&mut self.candidate)
                    .speed(0.25)
                    .range(0.0..=meta.nx as f32),
            );
            for (label, delta) in [
                ("−0.5", -0.5_f32),
                ("−0.25", -0.25),
                ("+0.25", 0.25),
                ("+0.5", 0.5),
            ] {
                if ui.button(label).clicked() {
                    self.candidate += delta;
                }
            }
            ui.label(
                egui::RichText::new(format!("midline {:.1}", meta.nx as f32 / 2.0))
                    .small()
                    .weak(),
            );
        });
        ui.add_space(4.0);
        if ui
            .button("Use in Tune")
            .on_hover_text("set this center on the Tune screen (turns auto-center off)")
            .clicked()
        {
            self.accepted = Some(self.candidate);
        }
        ui.add_space(8.0);
        ui.separator();

        ui.label("Sweep montage");
        ui.horizontal(|ui| {
            ui.label("± range");
            ui.add(
                egui::DragValue::new(&mut self.sweep_half)
                    .speed(0.5)
                    .range(0.25..=meta.nx as f32 / 2.0),
            );
            ui.label("step");
            ui.add(
                egui::DragValue::new(&mut self.sweep_step)
                    .speed(0.05)
                    .range(0.05..=8.0),
            );
        });
        ui.horizontal(|ui| {
            let idle = !self.sweep_pending;
            if ui
                .add_enabled(idle, egui::Button::new("Sweep"))
                .on_hover_text(
                    "reconstruct this sinogram row once per candidate center \
                     (write_center) and browse the results",
                )
                .clicked()
            {
                let (c, h, s) = (self.candidate, self.sweep_half, self.sweep_step);
                self.send_sweep(jobs, c, h, s);
            }
            let refinable = idle && self.sweep.is_some();
            if ui
                .add_enabled(refinable, egui::Button::new("Refine"))
                .on_hover_text("re-sweep around the selected candidate at step/4")
                .clicked()
                && let Some(c) = self.selected_center()
            {
                let s = self.sweep.as_ref().map_or(self.sweep_step, |sw| sw.step);
                self.send_sweep(jobs, c, s, s / 4.0);
            }
            if ui
                .add_enabled(
                    idle && self.sweep.is_some(),
                    egui::Button::new("Use selected"),
                )
                .on_hover_text("adopt the selected candidate as the working center")
                .clicked()
                && let Some(c) = self.selected_center()
            {
                self.candidate = c;
            }
            if self.sweep_pending {
                ui.spinner();
            }
        });
        if let Some(c) = self.selected_center() {
            ui.label(
                egui::RichText::new(format!("selected candidate: {c:.2}"))
                    .small()
                    .weak(),
            );
        }

        if !self.history.is_empty() {
            ui.add_space(8.0);
            ui.separator();
            ui.label("Estimates");
            egui::Grid::new("center_history")
                .striped(true)
                .show(ui, |ui| {
                    for e in self.history.iter().rev() {
                        ui.label(e.method.label());
                        ui.monospace(format!("{:.3}", e.center));
                        ui.label(egui::RichText::new(format!("{} ms", e.millis)).weak());
                        ui.end_row();
                    }
                });
        }
    }

    /// Center of the montage frame the stack currently shows.
    fn selected_center(&self) -> Option<f32> {
        let sweep = self.sweep.as_ref()?;
        sweep.centers.get(self.stack.current()).copied()
    }

    /// Sharpness curve under the montage; a click snaps the stack to the
    /// nearest candidate (docs/GUI.md §2.3 click-to-pick).
    fn metric_panel(&mut self, ui: &mut egui::Ui) {
        let resp = self.metric_plot.show(ui);
        let Some(sweep) = &self.sweep else {
            return;
        };
        if resp.response.clicked()
            && let Some(pos) = resp.response.interact_pointer_pos()
        {
            let (x, _y) = resp.transform.pixel_to_data(pos);
            if let Some(i) = nearest_index(&sweep.centers, x as f32) {
                self.stack.set_current(i);
            }
        }
        // Title doubles as the metric readout for the shown frame.
        let i = self.stack.current();
        if let (Some(c), Some(m)) = (sweep.centers.get(i), sweep.metric.get(i)) {
            self.metric_plot.set_graph_title(format!(
                "sharpness (std) — click to pick — center {c:.2}: {m:.4}"
            ));
        }
    }
}

/// Standard deviation of a frame (the montage sharpness metric).
fn std_dev(data: &[f32]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let n = data.len() as f64;
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = data
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    var.sqrt()
}

/// Index of the candidate closest to `x` (`None` on an empty list).
fn nearest_index(centers: &[f32], x: f32) -> Option<usize> {
    centers
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| (*a - x).abs().total_cmp(&(*b - x).abs()))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn std_dev_matches_hand_calc() {
        assert_eq!(std_dev(&[]), 0.0);
        assert_eq!(std_dev(&[3.0, 3.0, 3.0]), 0.0);
        // Values {1, 3}: mean 2, population variance 1.
        assert!((std_dev(&[1.0, 3.0, 1.0, 3.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nearest_index_snaps_to_closest_candidate() {
        let centers = [10.0_f32, 10.5, 11.0];
        assert_eq!(nearest_index(&centers, 10.6), Some(1));
        assert_eq!(nearest_index(&centers, 9.0), Some(0));
        assert_eq!(nearest_index(&centers, 100.0), Some(2));
        assert_eq!(nearest_index(&[], 1.0), None);
    }
}
