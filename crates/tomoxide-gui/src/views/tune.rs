//! Tune mode (docs/GUI.md §2 Tune): the single-slice parameter tuning loop.
//! Pick a slice, adjust parameters, reconstruct that one slice in memory, and
//! A/B-compare against a pinned earlier result.

use std::sync::Arc;
use std::sync::mpsc::Sender;

use siplot::egui_wgpu::RenderState;
use siplot::{ColormapDialog, CompareImages, Plot2D, egui};
use tomoxide::{Algorithm, StripeMethod};

use crate::worker::{DatasetMeta, Job, PreviewSpec};

/// Algorithms offered in the combo, in FromStr spelling.
const ALGORITHMS: &[&str] = &[
    "fbp",
    "gridrec",
    "fourierrec",
    "lprec",
    "linerec",
    "art",
    "bart",
    "sirt",
    "mlem",
    "osem",
    "ospml_hybrid",
    "ospml_quad",
    "pml_hybrid",
    "pml_quad",
    "tv",
    "grad",
    "tikh",
    "cgls",
];
const FILTERS: &[&str] = &[
    "none", "ramp", "shepp", "cosine", "cosine2", "hamming", "hann", "parzen",
];
const STRIPES: &[&str] = &["none", "fw", "ti", "sf", "vo-all"];
const PHASES: &[&str] = &["none", "paganin", "Gpaganin", "farago"];

/// One finished preview kept for display / pinning.
struct PreviewImage {
    ny: usize,
    nx: usize,
    data: Vec<f32>,
    /// Short parameter summary shown in the compare legend.
    summary: String,
}

pub struct TuneView {
    // --- parameters (Config-field spellings; recipe mapping is task #8) ---
    pub algorithm: String,
    pub filter: String,
    pub center_auto: bool,
    pub center: f32,
    pub num_iter: usize,
    /// Comma-separated reg_par list, parsed on request.
    pub reg_par: String,
    /// Truncated-projection support extension for iterative methods
    /// (`ReconParams::ext_pad`). On by default: real beamline samples routinely
    /// overhang the FOV, and without it the iterative preview is swamped by
    /// the truncation edge ring.
    pub ext_pad: bool,
    pub stripe: String,
    pub fw_sigma: f32,
    pub fw_level: usize,
    pub ti_nblock: usize,
    pub ti_beta: f32,
    pub sf_size: usize,
    pub vo_snr: f32,
    pub vo_la_size: usize,
    pub vo_sm_size: usize,
    /// Phase-retrieval method (Config spelling: `none`/`paganin`/`Gpaganin`/
    /// `farago`). Previews read a row band sized by the kernel support.
    pub phase: String,
    pub pixel_size: f32,
    pub propagation_distance: f32,
    pub energy: f32,
    pub alpha: f32,
    pub db: f32,
    pub w: f32,

    pub slice: usize,
    auto_recon: bool,
    /// Parameters changed since the last issued preview.
    dirty: bool,
    /// Monotone id per issued preview; one in flight at a time.
    generation: u64,
    pending: bool,
    last_millis: Option<u128>,

    current: Option<PreviewImage>,
    pinned: Option<PreviewImage>,

    preview_plot: Plot2D,
    compare: CompareImages,
    cmap_dialog: ColormapDialog,
    preview_image: Option<siplot::ItemHandle>,
}

impl TuneView {
    pub fn new(render_state: &RenderState) -> Self {
        let mut preview_plot = Plot2D::new(render_state, 30);
        preview_plot.set_graph_title("preview");
        preview_plot.set_graph_cursor(true);
        let compare = CompareImages::new(render_state, 40);
        TuneView {
            algorithm: "fbp".into(),
            filter: "parzen".into(),
            center_auto: true,
            center: 0.0,
            num_iter: 10,
            reg_par: String::new(),
            ext_pad: true,
            stripe: "none".into(),
            fw_sigma: 2.0,
            fw_level: 0,
            ti_nblock: 0,
            ti_beta: 1.5,
            sf_size: 5,
            vo_snr: 3.0,
            vo_la_size: 61,
            vo_sm_size: 21,
            // Physics defaults = Config::default() (the CLI template).
            phase: "none".into(),
            pixel_size: 1e-4,
            propagation_distance: 50.0,
            energy: 30.0,
            alpha: 1e-3,
            db: 1000.0,
            w: 2e-4,
            slice: 0,
            auto_recon: false,
            dirty: false,
            generation: 0,
            pending: false,
            last_millis: None,
            current: None,
            pinned: None,
            preview_plot,
            compare,
            cmap_dialog: ColormapDialog::new(),
            preview_image: None,
        }
    }

    pub fn on_dataset(&mut self, meta: &DatasetMeta) {
        self.slice = meta.nz / 2;
        self.center = (meta.nx as f32) / 2.0;
        self.current = None;
        self.pinned = None;
        self.pending = false;
        // A fresh dataset makes any shown preview stale; with no preview at
        // all this also fires the initial one (see ui()).
        self.dirty = true;
    }

    pub fn on_preview(
        &mut self,
        generation: u64,
        ny: usize,
        nx: usize,
        data: Vec<f32>,
        millis: u128,
    ) {
        if generation != self.generation {
            return; // superseded while in flight
        }
        self.pending = false;
        self.last_millis = Some(millis);
        let image = PreviewImage {
            ny,
            nx,
            data,
            summary: self.summary(),
        };
        let cmap = super::autoscale_viridis(&image.data);
        match self.preview_image {
            Some(h) => {
                let _ = self.preview_plot.try_update_image(
                    h,
                    image.nx as u32,
                    image.ny as u32,
                    &image.data,
                    cmap,
                );
            }
            None => {
                if let Ok(h) = self.preview_plot.try_add_image(
                    image.nx as u32,
                    image.ny as u32,
                    &image.data,
                    cmap,
                ) {
                    self.preview_image = Some(h);
                }
            }
        }
        self.preview_plot
            .set_graph_title(format!("slice {} — {}", self.slice, image.summary));
        self.current = Some(image);
        self.update_compare();
    }

    /// A worker preview failed: clear the in-flight flag so the loop resumes.
    pub fn on_preview_failed(&mut self) {
        self.pending = false;
    }

    /// Wall time of the last finished preview — the Run screen's pre-flight
    /// panel extrapolates its full-volume time estimate from it.
    pub fn last_preview_millis(&self) -> Option<u128> {
        self.last_millis
    }

    fn summary(&self) -> String {
        let mut s = self.algorithm.clone();
        if self.is_iterative() {
            s.push_str(&format!(":{}", self.num_iter));
            if self.ext_pad {
                s.push_str("+pad");
            }
        } else {
            s.push_str(&format!("/{}", self.filter));
        }
        if self.stripe != "none" {
            s.push_str(&format!(" stripe={}", self.stripe));
        }
        if self.phase != "none" {
            s.push_str(&format!(" phase={}", self.phase));
        }
        if !self.center_auto {
            s.push_str(&format!(" c={:.2}", self.center));
        }
        s
    }

    /// The panel's phase fields as a typed [`tomoxide::PhaseMethod`].
    fn build_phase(&self) -> Result<tomoxide::PhaseMethod, String> {
        use tomoxide::PhaseMethod;
        Ok(match self.phase.as_str() {
            "none" => PhaseMethod::None,
            "paganin" => PhaseMethod::Paganin {
                pixel_size: self.pixel_size,
                dist: self.propagation_distance,
                energy: self.energy,
                alpha: self.alpha,
            },
            "Gpaganin" => PhaseMethod::GPaganin {
                pixel_size: self.pixel_size,
                dist: self.propagation_distance,
                energy: self.energy,
                db: self.db,
                w: self.w,
            },
            "farago" => PhaseMethod::Farago {
                pixel_size: self.pixel_size,
                dist: self.propagation_distance,
                energy: self.energy,
                db: self.db,
            },
            other => return Err(format!("unknown phase method '{other}'")),
        })
    }

    fn is_iterative(&self) -> bool {
        !matches!(
            self.algorithm.as_str(),
            "fbp" | "gridrec" | "fourierrec" | "lprec" | "linerec"
        )
    }

    /// Fill the shared CLI config with this panel's parameters (recipe save).
    pub fn write_config(&self, cfg: &mut tomoxide::config::Config) -> Result<(), String> {
        cfg.algorithm = self.algorithm.clone();
        cfg.filter_name = self.filter.clone();
        cfg.rotation_axis = (!self.center_auto).then_some(self.center);
        cfg.num_iter = self.num_iter;
        cfg.ext_pad = self.ext_pad;
        cfg.reg_par = parse_reg_par(&self.reg_par)?;
        cfg.remove_stripe_method = self.stripe.clone();
        cfg.fw_sigma = self.fw_sigma;
        cfg.fw_level = self.fw_level;
        cfg.ti_nblock = self.ti_nblock;
        cfg.ti_beta = self.ti_beta;
        cfg.sf_size = self.sf_size;
        cfg.vo_snr = self.vo_snr;
        cfg.vo_la_size = self.vo_la_size;
        cfg.vo_sm_size = self.vo_sm_size;
        cfg.retrieve_phase_method = self.phase.clone();
        cfg.pixel_size = self.pixel_size as f64;
        cfg.propagation_distance = self.propagation_distance as f64;
        cfg.energy = self.energy as f64;
        cfg.alpha = self.alpha as f64;
        cfg.db = self.db as f64;
        cfg.w = self.w as f64;
        Ok(())
    }

    /// Adopt a loaded recipe's parameters (recipe load). Marks the panel
    /// dirty so an enabled auto-recon refreshes the preview.
    pub fn apply_config(&mut self, cfg: &tomoxide::config::Config) {
        self.algorithm = cfg.algorithm.clone();
        self.filter = cfg.filter_name.clone();
        self.center_auto = cfg.rotation_axis.is_none();
        if let Some(c) = cfg.rotation_axis {
            self.center = c;
        }
        self.num_iter = cfg.num_iter.max(1);
        self.ext_pad = cfg.ext_pad;
        self.reg_par = cfg
            .reg_par
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        self.stripe = cfg.remove_stripe_method.clone();
        self.fw_sigma = cfg.fw_sigma;
        self.fw_level = cfg.fw_level;
        self.ti_nblock = cfg.ti_nblock;
        self.ti_beta = cfg.ti_beta;
        self.sf_size = cfg.sf_size;
        self.vo_snr = cfg.vo_snr;
        self.vo_la_size = cfg.vo_la_size;
        self.vo_sm_size = cfg.vo_sm_size;
        self.phase = cfg.retrieve_phase_method.clone();
        self.pixel_size = cfg.pixel_size as f32;
        self.propagation_distance = cfg.propagation_distance as f32;
        self.energy = cfg.energy as f32;
        self.alpha = cfg.alpha as f32;
        self.db = cfg.db as f32;
        self.w = cfg.w as f32;
        self.dirty = true;
    }

    /// Resolve the panel state into a fully-typed spec (errors → session log).
    fn build_spec(&self) -> Result<PreviewSpec, String> {
        let algorithm: Algorithm = self.algorithm.parse().map_err(|e| format!("{e}"))?;
        let filter = self.filter.parse().map_err(|e| format!("{e}"))?;
        let reg_par = parse_reg_par(&self.reg_par)?;
        let stripe = match self.stripe.as_str() {
            "none" => StripeMethod::None,
            "fw" => StripeMethod::Fw {
                sigma: self.fw_sigma,
                level: (self.fw_level != 0).then_some(self.fw_level),
            },
            "ti" => StripeMethod::Ti {
                nblock: self.ti_nblock,
                beta: self.ti_beta,
            },
            "sf" => StripeMethod::Sf { size: self.sf_size },
            "vo-all" => StripeMethod::VoAll {
                snr: self.vo_snr,
                la_size: self.vo_la_size,
                sm_size: self.vo_sm_size,
            },
            other => return Err(format!("unknown stripe method '{other}'")),
        };
        Ok(PreviewSpec {
            slice: self.slice,
            algorithm,
            center: (!self.center_auto).then_some(self.center),
            filter,
            num_iter: self.num_iter,
            reg_par,
            ext_pad: self.ext_pad,
            stripe,
            phase: self.build_phase()?,
        })
    }

    fn request(&mut self, jobs: &Sender<Job>, log: &mut Vec<String>) {
        match self.build_spec() {
            Ok(spec) => {
                self.generation += 1;
                if jobs
                    .send(Job::Preview {
                        generation: self.generation,
                        spec,
                    })
                    .is_ok()
                {
                    self.pending = true;
                    self.dirty = false;
                }
            }
            Err(e) => {
                log.push(format!("preview parameters: {e}"));
                self.dirty = false;
            }
        }
    }

    fn update_compare(&mut self) {
        let (Some(a), Some(b)) = (&self.pinned, &self.current) else {
            return;
        };
        let mut all = a.data.clone();
        all.extend_from_slice(&b.data);
        let cmap = super::autoscale_viridis(&all);
        let _ = self.compare.set_images(
            (a.nx as u32, a.ny as u32),
            &a.data,
            (b.nx as u32, b.ny as u32),
            &b.data,
            cmap,
        );
    }

    fn params_panel(&mut self, ui: &mut egui::Ui, meta: &DatasetMeta) {
        let iterative = self.is_iterative();
        ui.heading("Parameters");
        egui::CollapsingHeader::new("Algorithm")
            .default_open(true)
            .show(ui, |ui| {
                let mut changed = false;
                changed |= combo(ui, "algorithm", &mut self.algorithm, ALGORITHMS);
                ui.add_enabled_ui(!iterative, |ui| {
                    changed |= combo(ui, "filter", &mut self.filter, FILTERS);
                });
                ui.add_enabled_ui(iterative, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("iterations");
                        changed |= ui
                            .add(egui::DragValue::new(&mut self.num_iter).range(1..=500))
                            .changed();
                    });
                    ui.horizontal(|ui| {
                        ui.label("reg_par");
                        changed |= ui
                            .add(
                                egui::TextEdit::singleline(&mut self.reg_par)
                                    .hint_text("0.5,0.01")
                                    .desired_width(90.0),
                            )
                            .changed();
                    });
                    changed |= ui
                        .checkbox(&mut self.ext_pad, "extend FOV")
                        .on_hover_text(
                            "solve on an edge-extended lane and crop back, so samples \
                             overhanging the field of view don't produce an edge ring \
                             (truncated projections); ~2.25\u{d7} slower per iteration",
                        )
                        .changed();
                });
                self.dirty |= changed;
            });
        egui::CollapsingHeader::new("Geometry")
            .default_open(true)
            .show(ui, |ui| {
                let mut changed = false;
                ui.horizontal(|ui| {
                    ui.label("slice");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.slice,
                            0..=meta.nz.saturating_sub(1),
                        ))
                        .changed();
                });
                ui.horizontal(|ui| {
                    changed |= ui.checkbox(&mut self.center_auto, "auto center").changed();
                    ui.add_enabled_ui(!self.center_auto, |ui| {
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut self.center)
                                    .speed(0.25)
                                    .range(0.0..=meta.nx as f32),
                            )
                            .changed();
                    });
                });
                self.dirty |= changed;
            });
        egui::CollapsingHeader::new("Stripe removal")
            .default_open(false)
            .show(ui, |ui| {
                let mut changed = combo(ui, "method", &mut self.stripe, STRIPES);
                match self.stripe.as_str() {
                    "fw" => {
                        changed |= drag(ui, "sigma", &mut self.fw_sigma, 0.1);
                        changed |= drag_usize(ui, "level (0=auto)", &mut self.fw_level);
                    }
                    "ti" => {
                        changed |= drag_usize(ui, "nblock", &mut self.ti_nblock);
                        changed |= drag(ui, "beta", &mut self.ti_beta, 0.1);
                    }
                    "sf" => {
                        changed |= drag_usize(ui, "size", &mut self.sf_size);
                    }
                    "vo-all" => {
                        changed |= drag(ui, "snr", &mut self.vo_snr, 0.1);
                        changed |= drag_usize(ui, "la_size", &mut self.vo_la_size);
                        changed |= drag_usize(ui, "sm_size", &mut self.vo_sm_size);
                    }
                    _ => {}
                }
                self.dirty |= changed;
            });
        egui::CollapsingHeader::new("Phase retrieval")
            .default_open(false)
            .show(ui, |ui| {
                let mut changed = combo(ui, "phase method", &mut self.phase, PHASES);
                if self.phase != "none" {
                    changed |= drag(ui, "pixel_size (cm)", &mut self.pixel_size, 1e-5);
                    changed |= drag(
                        ui,
                        "propagation_distance (cm)",
                        &mut self.propagation_distance,
                        1.0,
                    );
                    changed |= drag(ui, "energy (keV)", &mut self.energy, 1.0);
                    match self.phase.as_str() {
                        "paganin" => changed |= drag(ui, "alpha", &mut self.alpha, 1e-4),
                        "Gpaganin" => {
                            changed |= drag(ui, "delta/beta", &mut self.db, 10.0);
                            changed |= drag(ui, "W (cm)", &mut self.w, 1e-5);
                        }
                        "farago" => changed |= drag(ui, "delta/beta", &mut self.db, 10.0),
                        _ => {}
                    }
                    // Preview cost hint: the retrieval couples rows, so the
                    // preview reads a band this many rows each side.
                    if let Ok(method) = self.build_phase() {
                        let m = tomoxide::prep::phase::margin_rows(&method);
                        ui.label(
                            egui::RichText::new(format!(
                                "preview reads a ±{m}-row band around the slice \
                                 (Fresnel kernel support)"
                            ))
                            .small()
                            .weak(),
                        );
                    }
                }
                self.dirty |= changed;
            });
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        jobs: &Sender<Job>,
        meta: Option<&Arc<DatasetMeta>>,
        log: &mut Vec<String>,
    ) {
        let Some(meta) = meta.cloned() else {
            ui.label("Open a dataset in Data mode first.");
            return;
        };

        // Auto-recon loop: one preview in flight; re-issue when params moved.
        // The very first preview of a dataset fires without the auto toggle,
        // so entering Tune shows a slice instead of an empty panel.
        if (self.auto_recon || self.current.is_none()) && self.dirty && !self.pending {
            self.request(jobs, log);
        }

        egui::Panel::left("tune_params")
            .resizable(true)
            .default_size(320.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.params_panel(ui, &meta);
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(!self.pending, egui::Button::new("Reconstruct"))
                            .clicked()
                        {
                            self.request(jobs, log);
                        }
                        ui.checkbox(&mut self.auto_recon, "auto");
                        if self.pending {
                            ui.spinner();
                        }
                    });
                    if let Some(ms) = self.last_millis {
                        ui.label(egui::RichText::new(format!("last: {ms} ms")).small().weak());
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        let can_pin = self.current.is_some();
                        if ui
                            .add_enabled(can_pin, egui::Button::new("Pin A"))
                            .clicked()
                        {
                            self.pinned = self.current.take();
                            // keep showing the pinned data as current too
                            if let Some(p) = &self.pinned {
                                self.current = Some(PreviewImage {
                                    ny: p.ny,
                                    nx: p.nx,
                                    data: p.data.clone(),
                                    summary: p.summary.clone(),
                                });
                            }
                            self.update_compare();
                        }
                        if self.pinned.is_some() && ui.button("Unpin").clicked() {
                            self.pinned = None;
                        }
                    });
                    if let Some(p) = &self.pinned {
                        ui.label(egui::RichText::new(format!("A: {}", p.summary)).small());
                    }
                    if let Some(c) = &self.current {
                        ui.label(egui::RichText::new(format!("B: {}", c.summary)).small());
                    }
                });
            });

        if self.pinned.is_some() {
            // A/B comparison of the pinned and the latest preview.
            self.compare.show_toolbar(ui);
            self.compare.show(ui);
            self.compare.show_status_bar(ui);
        } else {
            self.preview_plot.show_toolbar_with(ui, |ui, _plot| {
                ui.separator();
                self.cmap_dialog.toggle_button(ui);
            });
            if self.current.is_some() {
                self.preview_plot.show(ui);
                self.cmap_dialog.show(ui.ctx(), &mut self.preview_plot);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Press Reconstruct to preview the selected slice.");
                });
            }
        }
    }
}

/// Parse a comma-separated `reg_par` list (empty entries skipped).
fn parse_reg_par(s: &str) -> Result<Vec<f32>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f32>().map_err(|e| format!("reg_par: {e}")))
        .collect()
}

fn combo(ui: &mut egui::Ui, label: &str, value: &mut String, options: &[&str]) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_salt(label)
            .selected_text(value.clone())
            .show_ui(ui, |ui| {
                for opt in options {
                    changed |= ui
                        .selectable_value(value, (*opt).to_string(), *opt)
                        .changed();
                }
            });
    });
    changed
}

fn drag(ui: &mut egui::Ui, label: &str, value: &mut f32, speed: f32) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed = ui.add(egui::DragValue::new(value).speed(speed)).changed();
    });
    changed
}

fn drag_usize(ui: &mut egui::Ui, label: &str, value: &mut usize) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        changed = ui.add(egui::DragValue::new(value)).changed();
    });
    changed
}
