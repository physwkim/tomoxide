//! Application shell: left mode rail, per-mode central panel, session log
//! pane, status bar (docs/GUI.md §2).

use siplot::egui;

/// The six workflow modes on the rail. Order = top-to-bottom rail order =
/// the operator's usual left-to-right workflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Open a dataset, inspect projections/theta/sinograms.
    Data,
    /// Single-slice parameter tuning loop with A/B compare.
    Tune,
    /// Rotation-axis finding: auto methods + fine tweak.
    Center,
    /// Full-volume reconstruction (subprocess) — M2.
    Run,
    /// Browse reconstruction results — M2.
    Output,
    /// Live streaming reconstruction (EPICS PVA) — M3.
    Live,
}

impl Mode {
    const ALL: [Mode; 6] = [
        Mode::Data,
        Mode::Tune,
        Mode::Center,
        Mode::Run,
        Mode::Output,
        Mode::Live,
    ];

    fn label(self) -> &'static str {
        match self {
            Mode::Data => "Data",
            Mode::Tune => "Tune",
            Mode::Center => "Center",
            Mode::Run => "Run",
            Mode::Output => "Output",
            Mode::Live => "Live",
        }
    }
}

/// One session-log line with its wall-clock timestamp.
struct LogLine {
    at: std::time::SystemTime,
    text: String,
}

/// Append-only session log shown in the bottom pane.
#[derive(Default)]
pub struct SessionLog {
    lines: Vec<LogLine>,
}

impl SessionLog {
    pub fn push(&mut self, text: impl Into<String>) {
        self.lines.push(LogLine {
            at: std::time::SystemTime::now(),
            text: text.into(),
        });
    }
}

pub struct App {
    mode: Mode,
    log: SessionLog,
    log_open: bool,
    worker: crate::worker::Worker,
    /// Backend name reported by the worker's Engine (`cpu`/`cuda`/…).
    backend: Option<String>,
    meta: Option<std::sync::Arc<crate::worker::DatasetMeta>>,
    data: crate::views::data::DataView,
    tune: crate::views::tune::TuneView,
    center: crate::views::center::CenterView,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Every siplot widget constructor needs this; fail loudly at startup
        // rather than per-view if the renderer is misconfigured.
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must use the wgpu renderer (NativeOptions.renderer = Wgpu)");
        let mut log = SessionLog::default();
        log.push(format!("tomoxide-gui {}", env!("CARGO_PKG_VERSION")));
        App {
            mode: Mode::Data,
            log,
            log_open: true,
            worker: crate::worker::Worker::spawn(cc.egui_ctx.clone()),
            backend: None,
            meta: None,
            data: crate::views::data::DataView::new(render_state),
            tune: crate::views::tune::TuneView::new(render_state),
            center: crate::views::center::CenterView::default(),
        }
    }

    /// Drain worker events and route them to the log / owning view.
    fn handle_events(&mut self) {
        use crate::worker::Event;
        let events: Vec<Event> = self.worker.events.try_iter().collect();
        for event in events {
            match event {
                Event::BackendReady(name) => {
                    self.log.push(format!("backend: {name}"));
                    self.backend = Some(name);
                }
                Event::DatasetOpened(meta) => {
                    self.log.push(format!(
                        "opened {} — {}×{}×{} (proj×rows×cols), {} flat / {} dark",
                        meta.path.display(),
                        meta.nproj,
                        meta.nz,
                        meta.nx,
                        meta.nflat,
                        meta.ndark
                    ));
                    self.tune.on_dataset(&meta);
                    self.center.on_dataset(&meta);
                    self.meta = Some(meta.clone());
                    self.data.on_dataset(meta);
                }
                Event::Sinogram {
                    row,
                    nproj,
                    nx,
                    data,
                } => self.data.on_sinogram(row, nproj, nx, &data),
                Event::Preview {
                    generation,
                    slice,
                    ny,
                    nx,
                    data,
                    millis,
                } => {
                    self.log.push(format!("preview slice {slice}: {millis} ms"));
                    self.tune.on_preview(generation, ny, nx, data, millis);
                }
                Event::CenterFound {
                    method,
                    center,
                    millis,
                } => {
                    self.log.push(format!(
                        "center ({}): {center:.3} — {millis} ms",
                        method.label()
                    ));
                    self.center.on_center(method, center, millis);
                }
                Event::JobFailed { what, error } => {
                    self.log.push(format!("FAILED {what}: {error}"));
                    if what.starts_with("preview") {
                        self.tune.on_preview_failed();
                    }
                    if what.starts_with("center") {
                        self.center.on_failed();
                    }
                }
            }
        }
    }

    fn mode_rail(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        for mode in Mode::ALL {
            let selected = self.mode == mode;
            if ui
                .selectable_label(selected, egui::RichText::new(mode.label()).size(15.0))
                .clicked()
            {
                self.mode = mode;
            }
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(format!("mode: {}", self.mode.label()));
            ui.separator();
            ui.label(format!(
                "backend: {}",
                self.backend.as_deref().unwrap_or("starting…")
            ));
            ui.separator();
            ui.label(
                self.meta
                    .as_ref()
                    .map(|m| m.path.display().to_string())
                    .unwrap_or_else(|| "no dataset".into()),
            );
            ui.separator();
            ui.toggle_value(&mut self.log_open, "log");
        });
    }

    fn log_pane(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for line in &self.log.lines {
                    let t = line
                        .at
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| {
                            let s = d.as_secs();
                            format!("{:02}:{:02}:{:02}", (s / 3600) % 24, (s / 60) % 60, s % 60)
                        })
                        .unwrap_or_default();
                    ui.monospace(format!("[{t}] {}", line.text));
                }
            });
    }

    fn central(&mut self, ui: &mut egui::Ui) {
        match self.mode {
            Mode::Data => self.data.ui(ui, &self.worker.jobs),
            Mode::Tune => {
                let mut msgs = Vec::new();
                self.tune
                    .ui(ui, &self.worker.jobs, self.meta.as_ref(), &mut msgs);
                for m in msgs {
                    self.log.push(m);
                }
            }
            Mode::Center => {
                self.center.ui(ui, &self.worker.jobs, self.meta.as_ref());
                if let Some(c) = self.center.take_accepted() {
                    self.tune.center = c;
                    self.tune.center_auto = false;
                    self.log.push(format!("center {c:.3} → Tune"));
                }
            }
            Mode::Run | Mode::Output => {
                ui.heading(self.mode.label());
                ui.label("Planned for M2 (docs/GUI.md §7): full-volume runs are spawned as tomoxide-cli subprocesses; results browse via StackView.");
            }
            Mode::Live => {
                ui.heading("Live");
                ui.label("Planned for M3 (docs/GUI.md §7): EPICS PVA streaming reconstruction.");
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.handle_events();
        egui::Panel::left("mode_rail")
            .resizable(false)
            .default_size(84.0)
            .show_inside(ui, |ui| self.mode_rail(ui));
        egui::Panel::bottom("status_bar").show_inside(ui, |ui| self.status_bar(ui));
        if self.log_open {
            egui::Panel::bottom("session_log")
                .resizable(true)
                .default_size(120.0)
                .show_inside(ui, |ui| self.log_pane(ui));
        }
        egui::CentralPanel::default().show_inside(ui, |ui| self.central(ui));
    }
}
