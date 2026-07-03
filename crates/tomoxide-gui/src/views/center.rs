//! Center mode (docs/GUI.md §2 Center): rotation-axis finding — auto methods
//! plus ±0.5/±0.25 px fine tweak (tomostream CenterTweak concept). The
//! candidate-sweep montage (write_center browser) is M2.

use std::sync::Arc;
use std::sync::mpsc::Sender;

use siplot::egui;

use crate::worker::{CenterMethod, DatasetMeta, Job};

/// One auto-detection result kept in the on-screen history.
struct Estimate {
    method: CenterMethod,
    center: f32,
    millis: u128,
}

#[derive(Default)]
pub struct CenterView {
    /// Sinogram row fed to the Vo/Entropy methods.
    row: usize,
    /// The working center value: latest estimate, then hand-tweaked.
    candidate: f32,
    /// Method of the in-flight job (one at a time).
    pending: Option<CenterMethod>,
    history: Vec<Estimate>,
    /// Set by "Use in Tune"; the app shell takes it and applies it.
    accepted: Option<f32>,
}

impl CenterView {
    pub fn on_dataset(&mut self, meta: &DatasetMeta) {
        self.row = meta.nz / 2;
        self.candidate = (meta.nx as f32) / 2.0;
        self.pending = None;
        self.history.clear();
        self.accepted = None;
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

    /// A worker center job failed: clear the in-flight flag.
    pub fn on_failed(&mut self) {
        self.pending = None;
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
                "SIFT registration of the 0°/180° pair — needs the sift-center build feature; \
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
}
