//! Run mode (docs/GUI.md §2.4): full-volume reconstruction, always as a
//! subprocess. The GUI spawns one `tomoxide` CLI process with the current
//! parameters as a recipe TOML (`--config`) plus `--progress_json`; the CLI's
//! own multi-GPU fan-out shards across the selected devices (children inherit
//! stdout, so their per-chunk JSON lines arrive here with global ranges
//! against one total). Cancel kills the process tree root; partial tiff
//! output stays on disk.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};

use siplot::egui_wgpu::RenderState;
use siplot::{Plot2D, egui};

use crate::views::load_tiff_f32;
use crate::views::tune::TuneView;
use crate::worker::DatasetMeta;

const FORMATS: &[&str] = &["tiff", "h5", "zarr"];
const DTYPES: &[&str] = &["float32", "float16"];
const BACKENDS: &[&str] = &["auto", "cpu", "cuda"];

/// One parsed `--progress_json` stdout line (the shape pinned by the CLI's
/// `progress_line_format` test).
#[derive(serde::Deserialize)]
struct ProgressLine {
    start: usize,
    end: usize,
    total: usize,
    #[allow(dead_code)]
    secs: f64,
}

/// Messages from the child's stdout/stderr reader threads.
enum RunMsg {
    Progress(ProgressLine),
    /// A non-JSON stdout line or any stderr line, for the session log.
    Line(String),
}

/// A running (or just-finished, not yet reaped) reconstruction subprocess.
struct ActiveRun {
    child: std::process::Child,
    rx: Receiver<RunMsg>,
    /// Sum of completed chunk heights (chunks/shards cover disjoint ranges).
    rows_done: usize,
    /// Full output slice count, from the first progress line's `total`.
    total: Option<usize>,
    /// Global `end` of the newest completed chunk (drives the tiff live view).
    latest_end: Option<usize>,
    started: std::time::Instant,
    /// Output base path — tiff slices land at `{output}_{i:05}.tiff`.
    output: String,
    tiff: bool,
    save_format: String,
    cancelled: bool,
    /// Recipe consumed via `--config`.
    recipe: PathBuf,
    /// Whether `recipe` is this view's own temp file (removed after a
    /// successful run, kept for inspection after a failed one). Queue
    /// recipes are the user's files and are never removed.
    remove_recipe: bool,
    /// Position in `queue` when this run is a batch item.
    queue_index: Option<usize>,
}

/// One batch entry: a recipe TOML that is exactly the file the CLI consumes
/// (docs/GUI.md §2.4 batch queue) — the dataset comes from its `file_name`.
struct QueueItem {
    recipe: PathBuf,
    status: QueueStatus,
}

#[derive(Clone, Copy, PartialEq)]
enum QueueStatus {
    Pending,
    Running,
    Done,
    Failed,
}

impl QueueStatus {
    fn icon(self) -> &'static str {
        match self {
            QueueStatus::Pending => "•",
            QueueStatus::Running => "▶",
            QueueStatus::Done => "✔",
            QueueStatus::Failed => "✘",
        }
    }
}

pub struct RunView {
    /// Output base path (each writer adds its suffix). Reset per dataset to
    /// the CLI's own default, `<input-without-extension>_rec`.
    pub output: String,
    pub save_format: String,
    pub dtype: String,
    pub backend: String,
    /// CUDA device picker: `(index, selected)`. Empty without the `cuda`
    /// feature or when no device answers — the picker is then hidden.
    devices: Vec<(i32, bool)>,
    active: Option<ActiveRun>,
    /// Outcome line of the last finished run, shown until the next start.
    last_outcome: Option<String>,
    /// `(output base, save_format)` of the last *successful* run — the Output
    /// screen offers it as a one-click source.
    completed: Option<(String, String)>,
    /// Batch queue, run sequentially while `queue_running`.
    queue: Vec<QueueItem>,
    queue_running: bool,

    plot: Plot2D,
    image: Option<siplot::ItemHandle>,
    /// `latest_end` already loaded into the plot (skip redundant decodes).
    shown_end: Option<usize>,
    /// Throttle tiff decodes to ~2/s even when chunks complete faster.
    last_load: Option<std::time::Instant>,
}

impl RunView {
    pub fn new(render_state: &RenderState) -> Self {
        let mut plot = Plot2D::new(render_state, 70);
        plot.set_graph_title("latest slice");
        RunView {
            output: String::new(),
            save_format: "tiff".into(),
            dtype: "float32".into(),
            backend: "auto".into(),
            devices: tomoxide::cuda::selected_devices()
                .into_iter()
                .map(|d| (d, true))
                .collect(),
            active: None,
            last_outcome: None,
            completed: None,
            queue: Vec::new(),
            queue_running: false,
            plot,
            image: None,
            shown_end: None,
            last_load: None,
        }
    }

    pub fn on_dataset(&mut self, meta: &DatasetMeta) {
        self.output = format!("{}_rec", meta.path.with_extension("").display());
    }

    /// `(output base, save_format)` of the last successful run, if any.
    pub fn completed_output(&self) -> Option<&(String, String)> {
        self.completed.as_ref()
    }

    /// Build the recipe (shared CLI config) for this run from the Tune panel
    /// plus this panel's output settings.
    fn build_config(
        &self,
        tune: &TuneView,
        meta: &DatasetMeta,
    ) -> Result<tomoxide::config::Config, String> {
        let mut cfg = tomoxide::config::Config::default();
        tune.write_config(&mut cfg)?;
        cfg.file_name = meta.path.display().to_string();
        cfg.backend = self.backend.clone();
        cfg.save_format = self.save_format.clone();
        cfg.dtype = self.dtype.clone();
        cfg.output = Some(self.output.clone());
        Ok(cfg)
    }

    fn start(&mut self, tune: &TuneView, meta: &DatasetMeta, log: &mut Vec<String>) {
        let cfg = match self.build_config(tune, meta) {
            Ok(c) => c,
            Err(e) => {
                log.push(format!("run not started: {e}"));
                return;
            }
        };
        static RUN_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let recipe = std::env::temp_dir().join(format!(
            "tomoxide-gui-run-{}-{}.toml",
            std::process::id(),
            RUN_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        if let Err(e) = cfg.write(&recipe) {
            log.push(format!("run not started: recipe write failed: {e}"));
            return;
        }
        self.start_recipe(recipe, None, true, log);
    }

    /// Spawn the CLI on a recipe file (the shared path for single runs and
    /// batch items). Returns whether a child is now running.
    fn start_recipe(
        &mut self,
        recipe: PathBuf,
        queue_index: Option<usize>,
        remove_recipe: bool,
        log: &mut Vec<String>,
    ) -> bool {
        let (dataset, output, save_format, backend) =
            match tomoxide::config::Config::load(&recipe).map_err(|e| e.to_string()) {
                Ok(cfg) => match recipe_run_params(&cfg) {
                    Ok(p) => p,
                    Err(e) => {
                        log.push(format!("run not started: {}: {e}", recipe.display()));
                        return false;
                    }
                },
                Err(e) => {
                    log.push(format!("run not started: {}: {e}", recipe.display()));
                    return false;
                }
            };
        let Some(cli) = find_cli() else {
            log.push(
                "run not started: no `tomoxide` CLI found (checked $TOMOXIDE_CLI, the GUI's \
                 directory, $PATH)"
                    .into(),
            );
            return false;
        };
        let selected: Vec<String> = self
            .devices
            .iter()
            .filter(|(_, on)| *on)
            .map(|(d, _)| d.to_string())
            .collect();
        if !self.devices.is_empty() && selected.is_empty() && backend != "cpu" {
            log.push("run not started: no GPU selected (pick at least one, or backend=cpu)".into());
            return false;
        }

        let mut cmd = std::process::Command::new(&cli);
        cmd.arg("--backend")
            .arg(&backend)
            .arg("recon")
            .arg(&dataset)
            .arg("--config")
            .arg(&recipe)
            .arg("--progress_json")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if !self.devices.is_empty() {
            cmd.env("TOMOXIDE_CUDA_DEVICES", selected.join(","));
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                log.push(format!("run not started: spawn {}: {e}", cli.display()));
                return false;
            }
        };
        log.push(format!(
            "run started: {} recon {} → {output} ({})",
            cli.display(),
            dataset.display(),
            recipe.display(),
        ));

        let (tx, rx) = std::sync::mpsc::channel();
        spawn_line_reader(child.stdout.take(), tx.clone(), true);
        spawn_line_reader(child.stderr.take(), tx, false);
        self.last_outcome = None;
        self.shown_end = None;
        self.active = Some(ActiveRun {
            child,
            rx,
            rows_done: 0,
            total: None,
            latest_end: None,
            started: std::time::Instant::now(),
            output,
            tiff: save_format == "tiff",
            save_format,
            cancelled: false,
            recipe,
            remove_recipe,
            queue_index,
        });
        true
    }

    /// While the queue is running and nothing is active, launch the next
    /// pending item; items that fail to even start are marked and skipped.
    fn advance_queue(&mut self, log: &mut Vec<String>) {
        while self.queue_running && self.active.is_none() {
            let Some(i) = self
                .queue
                .iter()
                .position(|q| q.status == QueueStatus::Pending)
            else {
                self.queue_running = false;
                log.push("batch queue finished".into());
                return;
            };
            let recipe = self.queue[i].recipe.clone();
            if self.start_recipe(recipe, Some(i), false, log) {
                self.queue[i].status = QueueStatus::Running;
            } else {
                self.queue[i].status = QueueStatus::Failed;
            }
        }
    }

    /// Drain progress messages and reap the child if it exited. Returns log
    /// lines for the session log.
    fn poll(&mut self, log: &mut Vec<String>) {
        let Some(run) = &mut self.active else { return };
        while let Ok(msg) = run.rx.try_recv() {
            match msg {
                RunMsg::Progress(p) => {
                    run.rows_done += p.end - p.start;
                    run.total = Some(p.total);
                    run.latest_end = Some(p.end);
                }
                RunMsg::Line(l) => log.push(format!("run: {l}")),
            }
        }
        match run.child.try_wait() {
            Ok(None) => {}
            Ok(Some(status)) => {
                // Reader threads may still hold buffered tail lines; drain
                // what has arrived by now (recv would race thread shutdown).
                while let Ok(msg) = run.rx.try_recv() {
                    match msg {
                        RunMsg::Progress(p) => {
                            run.rows_done += p.end - p.start;
                            run.total = Some(p.total);
                            run.latest_end = Some(p.end);
                        }
                        RunMsg::Line(l) => log.push(format!("run: {l}")),
                    }
                }
                let secs = run.started.elapsed().as_secs_f64();
                let success = status.success() && !run.cancelled;
                let outcome = if run.cancelled {
                    format!(
                        "run cancelled after {secs:.1}s — partial output kept at {}",
                        run.output
                    )
                } else if success {
                    if run.remove_recipe {
                        let _ = std::fs::remove_file(&run.recipe);
                    }
                    self.completed = Some((run.output.clone(), run.save_format.clone()));
                    format!(
                        "run finished in {secs:.1}s → {} ({} rows)",
                        run.output, run.rows_done
                    )
                } else {
                    format!(
                        "run FAILED ({status}) after {secs:.1}s — recipe at {}",
                        run.recipe.display()
                    )
                };
                let queue_index = run.queue_index;
                log.push(outcome.clone());
                self.last_outcome = Some(outcome);
                self.active = None;
                if let Some(i) = queue_index
                    && let Some(item) = self.queue.get_mut(i)
                {
                    item.status = if success {
                        QueueStatus::Done
                    } else {
                        QueueStatus::Failed
                    };
                }
            }
            Err(e) => {
                log.push(format!("run: wait failed: {e}"));
                self.active = None;
            }
        }
    }

    /// Load the newest completed slice into the plot (tiff output only).
    fn update_live_view(&mut self, log: &mut Vec<String>) {
        let (output, end) = match &self.active {
            Some(run) if run.tiff => match run.latest_end {
                Some(end) => (run.output.clone(), end),
                None => return,
            },
            _ => return,
        };
        if self.shown_end == Some(end)
            || self
                .last_load
                .is_some_and(|t| t.elapsed().as_millis() < 500)
        {
            return;
        }
        let path = PathBuf::from(format!("{output}_{:05}.tiff", end - 1));
        match load_tiff_f32(&path) {
            Ok((w, h, data)) => {
                let mut updated = false;
                if let Some(hnd) = self.image {
                    let cmap = super::autoscale_viridis(&data);
                    updated = self.plot.try_update_image(hnd, w, h, &data, cmap).is_ok();
                }
                if !updated {
                    let cmap = super::autoscale_viridis(&data);
                    if let Ok(hnd) = self.plot.try_add_image(w, h, &data, cmap) {
                        self.image = Some(hnd);
                    }
                }
                self.plot.set_graph_title(format!("slice {}", end - 1));
                self.shown_end = Some(end);
                self.last_load = Some(std::time::Instant::now());
            }
            Err(e) => {
                // First failure only — a shard may briefly outpace the fs.
                if self.last_load.is_none() {
                    log.push(format!("live view: {}: {e}", path.display()));
                    self.last_load = Some(std::time::Instant::now());
                }
            }
        }
    }

    fn preflight(&self, ui: &mut egui::Ui, tune: &TuneView, meta: &DatasetMeta) {
        ui.heading("Pre-flight");
        // Output volume: nz slices of num_gridx² f32 (the GUI preview grid is
        // nx; all writers store f32).
        let bytes = meta.nz as u64 * meta.nx as u64 * meta.nx as u64 * 4;
        match available_space_for(Path::new(&self.output)) {
            Some(avail) => {
                let ok = avail > bytes;
                let text = format!(
                    "disk: needs {} — {} free",
                    fmt_bytes(bytes),
                    fmt_bytes(avail)
                );
                ui.colored_label(
                    if ok {
                        egui::Color32::from_rgb(0, 160, 0)
                    } else {
                        egui::Color32::from_rgb(200, 0, 0)
                    },
                    text,
                );
            }
            None => {
                ui.label(format!(
                    "disk: needs {} (free space unknown — output directory missing?)",
                    fmt_bytes(bytes)
                ));
            }
        }
        match tune.last_preview_millis() {
            Some(ms) => {
                let est = ms as f64 * meta.nz as f64 / 1000.0;
                ui.label(
                    egui::RichText::new(format!(
                        "time: ~{est:.0}s extrapolated from the last {ms} ms preview \
                         (chunking/backend can change this considerably)"
                    ))
                    .small()
                    .weak(),
                );
            }
            None => {
                ui.label(
                    egui::RichText::new("time: no preview yet — run one in Tune for an estimate")
                        .small()
                        .weak(),
                );
            }
        }
        ui.label(
            egui::RichText::new("chunk: auto (CLI tune cache / safe default)")
                .small()
                .weak(),
        );
    }

    /// Batch queue (docs/GUI.md §2.4): ordered recipe TOMLs run sequentially,
    /// each being exactly the file the CLI consumes via `--config`.
    fn queue_ui(
        &mut self,
        ui: &mut egui::Ui,
        tune: &TuneView,
        meta: Option<&DatasetMeta>,
        log: &mut Vec<String>,
    ) {
        ui.heading("Batch queue");
        ui.horizontal(|ui| {
            if ui
                .button("Add recipes…")
                .on_hover_text("recipe TOMLs; the dataset comes from each file's file_name")
                .clicked()
                && let Some(files) = rfd::FileDialog::new()
                    .add_filter("recipe TOML", &["toml"])
                    .pick_files()
            {
                for recipe in files {
                    self.queue.push(QueueItem {
                        recipe,
                        status: QueueStatus::Pending,
                    });
                }
            }
            if let Some(meta) = meta
                && ui
                    .button("Queue current")
                    .on_hover_text("save the current parameters as {output}.toml and append it")
                    .clicked()
            {
                match self.build_config(tune, meta) {
                    Ok(cfg) => {
                        let recipe = free_recipe_path(&self.output);
                        match cfg.write(&recipe) {
                            Ok(()) => {
                                log.push(format!("queued {}", recipe.display()));
                                self.queue.push(QueueItem {
                                    recipe,
                                    status: QueueStatus::Pending,
                                });
                            }
                            Err(e) => log.push(format!("queue: recipe write failed: {e}")),
                        }
                    }
                    Err(e) => log.push(format!("queue: {e}")),
                }
            }
        });
        if self.queue.is_empty() {
            ui.label(
                egui::RichText::new("empty — queued recipes also run headlessly via the CLI")
                    .small()
                    .weak(),
            );
            return;
        }

        let mut remove = None;
        egui::Grid::new("run_queue").striped(true).show(ui, |ui| {
            for (i, item) in self.queue.iter().enumerate() {
                ui.label(item.status.icon());
                ui.label(
                    item.recipe
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                )
                .on_hover_text(item.recipe.display().to_string());
                if item.status != QueueStatus::Running && ui.small_button("✕").clicked() {
                    remove = Some(i);
                }
                ui.end_row();
            }
        });
        if let Some(i) = remove {
            self.queue.remove(i);
            // Keep the active batch item pointing at its (shifted) entry.
            if let Some(run) = &mut self.active
                && let Some(qi) = &mut run.queue_index
                && *qi > i
            {
                *qi -= 1;
            }
        }

        let pending = self
            .queue
            .iter()
            .filter(|q| q.status == QueueStatus::Pending)
            .count();
        ui.horizontal(|ui| {
            if self.queue_running {
                if ui.button("Pause queue").clicked() {
                    self.queue_running = false;
                    log.push("batch queue paused (the running item continues)".into());
                }
            } else if ui
                .add_enabled(pending > 0, egui::Button::new("Run queue"))
                .clicked()
            {
                self.queue_running = true;
            }
            ui.label(
                egui::RichText::new(format!("{pending} pending"))
                    .small()
                    .weak(),
            );
        });
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        tune: &TuneView,
        meta: Option<&Arc<DatasetMeta>>,
        log: &mut Vec<String>,
    ) {
        let meta = meta.cloned();
        self.poll(log);
        self.advance_queue(log);
        self.update_live_view(log);
        if self.active.is_some() {
            // Progress arrives without user input; keep polling at ~5 Hz.
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(200));
        }

        egui::Panel::left("run_params")
            .resizable(true)
            .default_size(340.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let idle = self.active.is_none();
                    if let Some(meta) = &meta {
                        ui.heading("Output");
                        ui.add_enabled_ui(idle, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("base path");
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.output)
                                        .desired_width(180.0),
                                );
                                if ui.button("…").clicked()
                                    && let Some(dir) = rfd::FileDialog::new().pick_folder()
                                {
                                    let stem = meta
                                        .path
                                        .file_stem()
                                        .map(|s| s.to_string_lossy().into_owned())
                                        .unwrap_or_else(|| "recon".into());
                                    self.output =
                                        format!("{}", dir.join(format!("{stem}_rec")).display());
                                }
                            });
                            combo(ui, "format", &mut self.save_format, FORMATS);
                            combo(ui, "dtype", &mut self.dtype, DTYPES);
                            combo(ui, "backend", &mut self.backend, BACKENDS);
                            if !self.devices.is_empty() {
                                ui.label("GPUs");
                                ui.horizontal_wrapped(|ui| {
                                    for (dev, on) in &mut self.devices {
                                        ui.checkbox(on, format!("cuda:{dev}"));
                                    }
                                });
                            }
                        });
                        ui.separator();
                        self.preflight(ui, tune, meta);
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!("parameters: {}", tune_summary(tune)))
                                .small(),
                        );
                    } else {
                        ui.label(
                            "Open a dataset in Data mode for single runs; queued recipes run \
                             without one.",
                        );
                    }
                    ui.separator();
                    match &self.active {
                        None => {
                            if let Some(meta) = &meta {
                                let meta = meta.clone();
                                if ui.button("Start full reconstruction").clicked() {
                                    self.start(tune, &meta, log);
                                }
                            }
                            if let Some(outcome) = &self.last_outcome {
                                ui.label(egui::RichText::new(outcome).small());
                            }
                        }
                        Some(_) => {
                            if ui.button("Cancel").clicked()
                                && let Some(run) = &mut self.active
                            {
                                run.cancelled = true;
                                // A cancel also pauses the queue: don't march
                                // on to the next item against the user.
                                self.queue_running = false;
                                if let Err(e) = run.child.kill() {
                                    log.push(format!("run: kill failed: {e}"));
                                }
                            }
                        }
                    }
                    ui.separator();
                    self.queue_ui(ui, tune, meta.as_deref(), log);
                    if let Some(run) = &self.active {
                        let (frac, text) = match run.total {
                            Some(total) if total > 0 => {
                                let f = run.rows_done as f32 / total as f32;
                                let eta = if run.rows_done > 0 {
                                    let secs = run.started.elapsed().as_secs_f64();
                                    let left = secs * (total - run.rows_done) as f64
                                        / run.rows_done as f64;
                                    format!(" — ETA {left:.0}s")
                                } else {
                                    String::new()
                                };
                                (f, format!("{}/{total} rows{eta}", run.rows_done))
                            }
                            _ => (0.0, "waiting for first chunk…".into()),
                        };
                        ui.add(egui::ProgressBar::new(frac).text(text));
                        if !run.tiff {
                            ui.label(
                                egui::RichText::new(
                                    "h5/zarr: progress only (the container is not readable \
                                     until finalize)",
                                )
                                .small()
                                .weak(),
                            );
                        }
                    }
                });
            });

        if self.image.is_some() {
            self.plot.show_toolbar(ui);
            self.plot.show(ui);
        } else {
            ui.centered_and_justified(|ui| {
                ui.label(if self.active.is_some() {
                    "reconstructing — the latest finished slice appears here (tiff output)"
                } else {
                    "Start a run; with tiff output the latest finished slice is shown here."
                });
            });
        }
    }
}

/// First non-existing `{output}.toml` / `{output}-N.toml` — "Queue current"
/// must not overwrite a recipe already queued for the same output base.
fn free_recipe_path(output: &str) -> PathBuf {
    let plain = PathBuf::from(format!("{output}.toml"));
    if !plain.exists() {
        return plain;
    }
    (1..)
        .map(|n| PathBuf::from(format!("{output}-{n}.toml")))
        .find(|p| !p.exists())
        .unwrap()
}

/// Extract what a spawn needs from a recipe: `(dataset, output base,
/// save_format, backend)`. The output default mirrors the CLI
/// (`<input-without-extension>_rec`).
fn recipe_run_params(
    cfg: &tomoxide::config::Config,
) -> Result<(PathBuf, String, String, String), String> {
    if cfg.file_name.is_empty() {
        return Err("recipe has no file_name (input dataset)".into());
    }
    let dataset = PathBuf::from(&cfg.file_name);
    let output = cfg
        .output
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{}_rec", dataset.with_extension("").display()));
    Ok((
        dataset,
        output,
        cfg.save_format.clone(),
        cfg.backend.clone(),
    ))
}

/// One-line parameter summary for the panel (mirrors the Tune legend fields).
fn tune_summary(tune: &TuneView) -> String {
    let mut s = tune.algorithm.clone();
    if tune.center_auto {
        s.push_str(" c=auto");
    } else {
        s.push_str(&format!(" c={:.2}", tune.center));
    }
    if tune.stripe != "none" {
        s.push_str(&format!(" stripe={}", tune.stripe));
    }
    s
}

/// Locate the `tomoxide` CLI binary: `$TOMOXIDE_CLI` override, then next to
/// this executable (deployed together), then `$PATH`, then the repo's own
/// debug build (dev convenience; the path is baked at compile time).
fn find_cli() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("TOMOXIDE_CLI") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let c = dir.join("tomoxide");
        if c.is_file() {
            return Some(c);
        }
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let c = dir.join("tomoxide");
            if c.is_file() {
                return Some(c);
            }
        }
    }
    let dev = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/debug/tomoxide"
    ));
    dev.is_file().then_some(dev)
}

/// Forward each line of a child stream as a [`RunMsg`]. On the stdout stream
/// (`parse_json`), lines starting with `{` are `--progress_json` records;
/// anything else (both streams) goes to the session log.
fn spawn_line_reader<R: std::io::Read + Send + 'static>(
    stream: Option<R>,
    tx: Sender<RunMsg>,
    parse_json: bool,
) {
    let Some(stream) = stream else { return };
    std::thread::spawn(move || {
        use std::io::BufRead as _;
        for line in std::io::BufReader::new(stream).lines() {
            let Ok(line) = line else { break };
            let msg = if parse_json && line.starts_with('{') {
                match serde_json::from_str::<ProgressLine>(&line) {
                    Ok(p) => RunMsg::Progress(p),
                    Err(_) => RunMsg::Line(line),
                }
            } else {
                RunMsg::Line(line)
            };
            if tx.send(msg).is_err() {
                break;
            }
        }
    });
}

/// Free space on the filesystem holding `output` (walking up to the nearest
/// existing ancestor, since the output directory may not exist yet).
fn available_space_for(output: &Path) -> Option<u64> {
    let mut dir = output.parent()?;
    loop {
        if dir.exists() {
            return fs4::available_space(dir).ok();
        }
        dir = dir.parent()?;
    }
}

fn fmt_bytes(b: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let g = b as f64 / GB;
    if g >= 1.0 {
        format!("{g:.1} GiB")
    } else {
        format!("{:.0} MiB", b as f64 / (1024.0 * 1024.0))
    }
}

fn combo(ui: &mut egui::Ui, label: &str, value: &mut String, options: &[&str]) {
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_salt(format!("run_{label}"))
            .selected_text(value.clone())
            .show_ui(ui, |ui| {
                for opt in options {
                    ui.selectable_value(value, (*opt).to_string(), *opt);
                }
            });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CLI's pinned line shape parses into `ProgressLine`.
    #[test]
    fn progress_line_parses() {
        let p: ProgressLine =
            serde_json::from_str("{\"start\":8,\"end\":16,\"total\":128,\"secs\":1.235}").unwrap();
        assert_eq!((p.start, p.end, p.total), (8, 16, 128));
        assert!((p.secs - 1.235).abs() < 1e-9);
    }

    /// `$TOMOXIDE_CLI` pointing at a real file wins; pointing at a missing
    /// file yields None rather than falling through to a wrong binary.
    #[test]
    fn find_cli_env_override() {
        let dir = std::env::temp_dir().join(format!("tomoxide-gui-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("tomoxide");
        std::fs::write(&fake, b"").unwrap();

        // Serialize the two env states within this one test (tests in one
        // binary share the process environment).
        unsafe { std::env::set_var("TOMOXIDE_CLI", &fake) };
        assert_eq!(find_cli(), Some(fake.clone()));
        unsafe { std::env::set_var("TOMOXIDE_CLI", dir.join("missing")) };
        assert_eq!(find_cli(), None);
        unsafe { std::env::remove_var("TOMOXIDE_CLI") };
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A queue spawn takes dataset/output/format/backend from the recipe:
    /// explicit output wins, an absent one derives the CLI default, and a
    /// recipe without file_name is rejected.
    #[test]
    fn recipe_run_params_resolves_from_config() {
        let mut cfg = tomoxide::config::Config {
            file_name: "/data/scan.h5".into(),
            output: Some("/out/vol".into()),
            save_format: "zarr".into(),
            backend: "cpu".into(),
            ..Default::default()
        };
        let (dataset, output, format, backend) = recipe_run_params(&cfg).unwrap();
        assert_eq!(dataset, PathBuf::from("/data/scan.h5"));
        assert_eq!(output, "/out/vol");
        assert_eq!((format.as_str(), backend.as_str()), ("zarr", "cpu"));

        cfg.output = None;
        let (_, output, ..) = recipe_run_params(&cfg).unwrap();
        assert_eq!(output, "/data/scan_rec");

        cfg.file_name = String::new();
        assert!(recipe_run_params(&cfg).is_err());
    }

    /// "Queue current" never overwrites an existing recipe for the same base.
    #[test]
    fn free_recipe_path_skips_existing() {
        let dir = std::env::temp_dir().join(format!("tomoxide-gui-queue-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("vol").to_string_lossy().into_owned();
        assert_eq!(
            free_recipe_path(&base),
            PathBuf::from(format!("{base}.toml"))
        );
        std::fs::write(format!("{base}.toml"), "").unwrap();
        std::fs::write(format!("{base}-1.toml"), "").unwrap();
        assert_eq!(
            free_recipe_path(&base),
            PathBuf::from(format!("{base}-2.toml"))
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn available_space_walks_to_existing_ancestor() {
        let missing = std::env::temp_dir().join("tomoxide-gui-nonexistent/deep/path/rec");
        assert!(available_space_for(&missing).is_some());
    }

    #[test]
    fn fmt_bytes_scales() {
        assert_eq!(fmt_bytes(3 << 30), "3.0 GiB");
        assert_eq!(fmt_bytes(512 << 20), "512 MiB");
    }

    /// The live view must decode exactly what the CLI's tiff writer produces
    /// (same `tiff` major; f32 grayscale, `{base}_{i:05}.tiff` naming).
    #[test]
    fn load_tiff_f32_decodes_writer_output() {
        let dir = std::env::temp_dir().join(format!("tomoxide-gui-tiff-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("rec");

        let vol = tomoxide::Volume::new(ndarray::Array3::from_shape_fn((2, 3, 4), |(z, y, x)| {
            (z * 100 + y * 10 + x) as f32
        }));
        let mut w =
            tomoxide::io::create_writer(&base.to_string_lossy(), tomoxide::io::SaveFormat::Tiff)
                .unwrap();
        w.reserve(2).unwrap();
        w.write_chunk(&vol, 0, 2).unwrap();
        w.finalize().unwrap();

        let (width, height, data) = load_tiff_f32(&dir.join("rec_00001.tiff")).unwrap();
        assert_eq!((width, height), (4, 3));
        assert_eq!(data[0], 100.0); // slice 1, row 0, col 0
        assert_eq!(data[11], 123.0); // slice 1, row 2, col 3
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
