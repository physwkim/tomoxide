//! Live mode — streaming ortho-slice reconstruction (docs/GUI.md §2.6).
//!
//! The tomostream operating model, in-process: a connection panel of PVA channel
//! addresses feeds an rsdm data engine (its own tokio runtime) whose frames land
//! in a fixed-capacity ring buffer; a live thread reconstructs the selected Z
//! slice each loop, re-reading the parameters every iteration so a center tweak
//! or filter change applies on the next pass. tomoxide reconstructs Z
//! (horizontal) slices cheaply, so this ships Z-only — the X/Y ortho panes need
//! dedicated backprojection kernels (docs/GUI.md §6 #7) and are out of scope.
//!
//! Threading: the live thread owns the rsdm engine, the ring buffer, and a
//! tomoxide backend; it exchanges [`LiveCmd`]/[`LiveEvent`] with the UI over
//! mpsc channels. It holds no egui `Context` and never drives repaints itself:
//! the Live view's `ui()` schedules `request_repaint_after(200 ms)` while a
//! session runs, so redraws happen only while the Live tab is the visible mode
//! rather than continuously in the background. Otherwise it is the same
//! self-contained pattern the Output/XANES views use, kept off the shared
//! frame-serving worker (which owns `!Send` HDF5 handles).

use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rsplot::egui_wgpu::RenderState;
use rsplot::{ImageView, egui};
use tomoxide::{Algorithm, BackendKind, Engine, FilterName, StripeMethod};

use crate::live::recon_loop::{LiveReconParams, reconstruct_slice};
use crate::live::ring::ProjRing;
use crate::live::source::{LiveSource, ManualCfg, PvaAddrs};

/// UI → live thread.
enum LiveCmd {
    SetParams(LiveReconParams),
    Stop,
}

/// Live thread → UI. Small, bounded control events only — the big
/// reconstructed-slice payload rides a latest-wins slot ([`LiveImage`]), not this
/// queue, so it cannot accumulate when the UI is not draining.
enum LiveEvent {
    Connected,
    ConnectFailed(String),
    Status {
        fill: usize,
        connected: bool,
        det: (usize, usize),
        darkflat: bool,
    },
    Error(String),
}

/// The latest reconstructed slice. Overwritten in place every loop — only the
/// newest slice is meaningful — so a backgrounded Live tab holds at most one
/// image (~one detector slice) instead of an unbounded queue of them.
struct LiveImage {
    ny: usize,
    nx: usize,
    data: Vec<f32>,
    ms: u128,
    nproj: usize,
}

/// A running live session: the command/event channels and the thread handle.
struct LiveHandle {
    cmd: Sender<LiveCmd>,
    evt: Receiver<LiveEvent>,
    /// Latest-wins slot for the reconstructed slice (see [`LiveImage`]).
    image: Arc<Mutex<Option<LiveImage>>>,
    join: Option<JoinHandle<()>>,
}

pub struct LiveView {
    addrs: PvaAddrs,
    /// Manual overrides for frame geometry / angle (edited when disconnected).
    manual: ManualCfg,
    params: LiveReconParams,
    /// UI toggle for ring removal: `false` ⇒ none, `true` ⇒ Fourier-wavelet.
    stripe_fw: bool,
    /// UI toggle: reconstruct on the detector midline (`center == None`).
    center_auto: bool,
    /// Center value edited when `center_auto` is off.
    center_val: f32,

    handle: Option<LiveHandle>,
    connecting: bool,
    connected: bool,
    fill: usize,
    det: (usize, usize),
    darkflat: bool,
    recon_ms: u128,
    last_nproj: usize,

    image: Option<(usize, usize, Vec<f32>)>,
    view: ImageView,
}

impl LiveView {
    pub fn new(render_state: &RenderState) -> Self {
        let mut view = ImageView::new(render_state, 130);
        view.set_side_histogram_displayed(false);
        view.image_plot_mut().set_graph_title("live slice");
        LiveView {
            addrs: default_addrs(),
            manual: ManualCfg::default(),
            params: LiveReconParams::default(),
            stripe_fw: false,
            center_auto: true,
            center_val: 0.0,
            handle: None,
            connecting: false,
            connected: false,
            fill: 0,
            det: (0, 0),
            darkflat: false,
            recon_ms: 0,
            last_nproj: 0,
            image: None,
            view,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, log: &mut Vec<String>) {
        self.poll(log);
        if self.handle.is_some() {
            ui.ctx().request_repaint_after(Duration::from_millis(200));
        }

        ui.horizontal(|ui| {
            ui.heading("Live");
            ui.separator();
            ui.label(self.status_line());
        });
        ui.separator();

        egui::Panel::left("live_controls")
            .resizable(true)
            .default_size(360.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.control_panel(ui, log));
            });

        // Central: the reconstructed slice with its cursor value readout.
        if self.image.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label("Connect a PVA projection stream to reconstruct the selected slice live.");
            });
            return;
        }
        let hovered = self.view.value_changed();
        super::value_readout(ui, hovered);
        self.view.show(ui, None, None);
    }

    /// Drain live-thread events and fold them into the view state.
    fn poll(&mut self, log: &mut Vec<String>) {
        let mut events = Vec::new();
        if let Some(h) = &self.handle {
            while let Ok(ev) = h.evt.try_recv() {
                events.push(ev);
            }
        }
        for ev in events {
            match ev {
                LiveEvent::Connected => {
                    self.connecting = false;
                    self.connected = true;
                    log.push("live: engine connected (channels resolving)".to_owned());
                }
                LiveEvent::ConnectFailed(e) => {
                    log.push(format!("live: connect FAILED — {e}"));
                    self.teardown();
                }
                LiveEvent::Status {
                    fill,
                    connected,
                    det,
                    darkflat,
                } => {
                    // capacity is UI-owned (the DragValue below flows to the
                    // thread via LiveCmd); do NOT echo the thread's copy back
                    // into params.capacity or it fights the user's live edits.
                    self.fill = fill;
                    self.connected = connected;
                    self.det = det;
                    self.darkflat = darkflat;
                }
                LiveEvent::Error(e) => log.push(format!("live: {e}")),
            }
        }

        // Pull the latest reconstructed slice from the latest-wins slot. Only
        // one is ever buffered, so this stays bounded even if the tab was
        // inactive for the whole scan.
        let latest = self
            .handle
            .as_ref()
            .and_then(|h| h.image.lock().unwrap().take());
        if let Some(img) = latest {
            self.recon_ms = img.ms;
            self.last_nproj = img.nproj;
            self.image = Some((img.ny, img.nx, img.data));
            self.redraw();
        }
    }

    fn redraw(&mut self) {
        if let Some((ny, nx, data)) = &self.image {
            let cmap = super::autoscale_viridis(data);
            let _ = self.view.set_image(*nx as u32, *ny as u32, data, cmap);
        }
    }

    fn status_line(&self) -> String {
        if self.connecting {
            "connecting…".to_owned()
        } else if self.handle.is_some() {
            let link = if self.connected {
                "streaming"
            } else {
                "waiting for frames"
            };
            let det = if self.det.0 > 0 {
                format!("{}×{}", self.det.1, self.det.0)
            } else {
                "—".to_owned()
            };
            let norm = if self.darkflat { "flat/dark" } else { "raw" };
            format!(
                "{link} · buffer {}/{} · det {det} · {norm} · {} proj · {} ms",
                self.fill, self.params.capacity, self.last_nproj, self.recon_ms
            )
        } else {
            "not connected".to_owned()
        }
    }

    fn control_panel(&mut self, ui: &mut egui::Ui, log: &mut Vec<String>) {
        let running = self.handle.is_some();

        ui.strong("Connection");
        ui.add_enabled_ui(!running, |ui| {
            let manual = &self.manual;
            egui::Grid::new("live_addrs").num_columns(2).show(ui, |ui| {
                addr_row(ui, "image", &mut self.addrs.image, true);
                addr_row(ui, "theta", &mut self.addrs.theta, !manual.theta_manual);
                addr_row(ui, "width", &mut self.addrs.width, !manual.dims_manual);
                addr_row(ui, "height", &mut self.addrs.height, !manual.dims_manual);
                addr_row(ui, "dark", &mut self.addrs.dark, true);
                addr_row(ui, "flat", &mut self.addrs.flat, true);
            });

            // Manual frame size — for streams that publish no width/height PV.
            // rsdm hands over a flat pixel array, so the width is required; the
            // height is derived from the frame length when left at 0.
            ui.checkbox(&mut self.manual.dims_manual, "manual frame size");
            if self.manual.dims_manual {
                egui::Grid::new("live_manual_dims")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("width (nx)");
                        ui.add(egui::DragValue::new(&mut self.manual.nx).range(1..=65535));
                        ui.end_row();
                        ui.label("height (ny)");
                        ui.add(
                            egui::DragValue::new(&mut self.manual.ny)
                                .range(0..=65535)
                                .prefix("0 = from image: "),
                        );
                        ui.end_row();
                    });
            }

            // Manual theta — assign a constant angular step per frame when no
            // theta PV is available.
            ui.checkbox(&mut self.manual.theta_manual, "manual theta (deg/frame)");
            if self.manual.theta_manual {
                egui::Grid::new("live_manual_theta")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("start (deg)");
                        ui.add(egui::DragValue::new(&mut self.manual.theta_start).speed(0.1));
                        ui.end_row();
                        ui.label("step (deg)");
                        ui.add(
                            egui::DragValue::new(&mut self.manual.theta_step)
                                .speed(0.01)
                                .range(0.0..=360.0),
                        );
                        ui.end_row();
                    });
            }
        });
        ui.horizontal(|ui| {
            if !running {
                if ui.button("Connect").clicked() {
                    self.connect(log);
                }
            } else if ui.button("Disconnect").clicked() {
                self.teardown();
                log.push("live: disconnected".to_owned());
            }
        });

        ui.separator();
        ui.strong("Reconstruction");
        let before = self.params.clone();

        egui::Grid::new("live_params")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("slice z");
                // Valid slice indices are 0..ny (det.0 is the detector row count
                // / recon Z extent), so the max index is ny - 1. The old `.max(1)`
                // forced zmax >= 1 even when ny == 1, offering slice 1 for a
                // single-row detector — out of range (reconstruct_slice then errs
                // with "slice out of range" instead of reconstructing row 0). A
                // zero-width 0..=0 range is fine for egui's Slider.
                let zmax = self.det.0.saturating_sub(1);
                ui.add(egui::Slider::new(&mut self.params.slice, 0..=zmax));
                ui.end_row();

                ui.label("center");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.center_auto, "midline");
                    ui.add_enabled_ui(!self.center_auto, |ui| {
                        ui.add(egui::DragValue::new(&mut self.center_val).speed(0.25));
                    });
                });
                ui.end_row();

                if !self.center_auto {
                    ui.label("tweak");
                    ui.horizontal(|ui| {
                        for d in [-0.5f32, -0.25, 0.25, 0.5] {
                            if ui.button(format!("{d:+}")).clicked() {
                                self.center_val += d;
                            }
                        }
                    });
                    ui.end_row();
                }

                ui.label("algorithm");
                egui::ComboBox::from_id_salt("live_algo")
                    .selected_text(algo_label(self.params.algorithm))
                    .show_ui(ui, |ui| {
                        for a in ANALYTIC {
                            ui.selectable_value(&mut self.params.algorithm, a, algo_label(a));
                        }
                    });
                ui.end_row();

                ui.label("filter");
                egui::ComboBox::from_id_salt("live_filter")
                    .selected_text(filter_label(self.params.filter))
                    .show_ui(ui, |ui| {
                        for f in FILTERS {
                            ui.selectable_value(&mut self.params.filter, f, filter_label(f));
                        }
                    });
                ui.end_row();

                ui.label("ring removal");
                ui.checkbox(&mut self.stripe_fw, "Fourier-wavelet");
                ui.end_row();

                ui.label("ext. pad");
                ui.checkbox(&mut self.params.ext_pad, "");
                ui.end_row();

                ui.label("buffer (proj)");
                ui.add(egui::DragValue::new(&mut self.params.capacity).range(2..=4096));
                ui.end_row();
            });

        // Fold the UI-only toggles back into the parameter struct.
        self.params.center = if self.center_auto {
            None
        } else {
            Some(self.center_val)
        };
        self.params.stripe = if self.stripe_fw {
            StripeMethod::Fw {
                sigma: 1.0,
                level: None,
            }
        } else {
            StripeMethod::None
        };

        if self.params != before && running {
            self.send_params();
        }
    }

    fn connect(&mut self, log: &mut Vec<String>) {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (evt_tx, evt_rx) = std::sync::mpsc::channel();
        let image = Arc::new(Mutex::new(None));
        let image_thread = Arc::clone(&image);
        let addrs = self.addrs.clone();
        let manual = self.manual.clone();
        let params = self.params.clone();
        let join = std::thread::Builder::new()
            .name("tomoxide-live".to_owned())
            .spawn(move || live_thread(addrs, manual, params, cmd_rx, evt_tx, image_thread))
            .expect("spawning the live thread");
        self.handle = Some(LiveHandle {
            cmd: cmd_tx,
            evt: evt_rx,
            image,
            join: Some(join),
        });
        self.connecting = true;
        self.connected = false;
        self.fill = 0;
        log.push("live: connecting…".to_owned());
    }

    fn send_params(&self) {
        if let Some(h) = &self.handle {
            let _ = h.cmd.send(LiveCmd::SetParams(self.params.clone()));
        }
    }

    fn teardown(&mut self) {
        if let Some(mut h) = self.handle.take() {
            let _ = h.cmd.send(LiveCmd::Stop);
            if let Some(j) = h.join.take() {
                let _ = j.join();
            }
        }
        self.connecting = false;
        self.connected = false;
        self.fill = 0;
    }
}

impl Drop for LiveView {
    fn drop(&mut self) {
        self.teardown();
    }
}

fn addr_row(ui: &mut egui::Ui, label: &str, value: &mut String, enabled: bool) {
    ui.label(label);
    ui.add_enabled(
        enabled,
        egui::TextEdit::singleline(value).desired_width(240.0),
    );
    ui.end_row();
}

/// Placeholder addresses in tomoScanStream / areaDetector convention — the user
/// edits these to the beamline's actual channels.
fn default_addrs() -> PvaAddrs {
    PvaAddrs {
        image: "pva://TOMO:Image".to_owned(),
        theta: "pva://TOMO:Theta".to_owned(),
        width: "pva://TOMO:ArraySize0_RBV".to_owned(),
        // Height left empty: derived from the frame length (len / width) unless a
        // PV is supplied. rsdm flattens the frame, so only the width is required.
        height: String::new(),
        dark: String::new(),
        flat: String::new(),
    }
}

/// Analytic algorithms offered live (iterative methods are not used streaming).
const ANALYTIC: [Algorithm; 5] = [
    Algorithm::Fbp,
    Algorithm::Gridrec,
    Algorithm::Fourierrec,
    Algorithm::Lprec,
    Algorithm::Linerec,
];

const FILTERS: [FilterName; 8] = [
    FilterName::Parzen,
    FilterName::Ramp,
    FilterName::Shepp,
    FilterName::Cosine,
    FilterName::Cosine2,
    FilterName::Hamming,
    FilterName::Hann,
    FilterName::None,
];

fn algo_label(a: Algorithm) -> &'static str {
    match a {
        Algorithm::Fbp => "FBP",
        Algorithm::Gridrec => "Gridrec",
        Algorithm::Fourierrec => "Fourierrec",
        Algorithm::Lprec => "Lprec",
        Algorithm::Linerec => "Linerec",
        _ => "FBP",
    }
}

fn filter_label(f: FilterName) -> &'static str {
    match f {
        FilterName::None => "none",
        FilterName::Ramp => "ramp",
        FilterName::Shepp => "shepp",
        FilterName::Cosine => "cosine",
        FilterName::Cosine2 => "cosine2",
        FilterName::Hamming => "hamming",
        FilterName::Hann => "hann",
        FilterName::Parzen => "parzen",
    }
}

/// The live loop: rsdm frames → ring buffer → per-loop Z-slice recon.
fn live_thread(
    addrs: PvaAddrs,
    manual: ManualCfg,
    mut params: LiveReconParams,
    cmd_rx: Receiver<LiveCmd>,
    evt_tx: Sender<LiveEvent>,
    image: Arc<Mutex<Option<LiveImage>>>,
) {
    // The thread deliberately holds no egui Context: repaint scheduling is
    // owned solely by the Live view's ui(), which calls request_repaint_after
    // every 200 ms while a session runs. A thread-driven request_repaint()
    // would fire on every event/recon regardless of which mode is visible,
    // forcing the whole app to repaint continuously while the Live tab is in
    // the background. Dropping the Context makes that storm unconstructable.
    let send = |e: LiveEvent| {
        let _ = evt_tx.send(e);
    };

    let engine = match Engine::new(BackendKind::Auto) {
        Ok(e) => e,
        Err(e) => {
            send(LiveEvent::ConnectFailed(format!("backend init: {e}")));
            return;
        }
    };
    let mut source = match LiveSource::connect(&addrs, manual) {
        Ok(s) => s,
        Err(e) => {
            send(LiveEvent::ConnectFailed(e));
            return;
        }
    };
    send(LiveEvent::Connected);

    let mut ring = ProjRing::new(params.capacity);
    let mut last_status: Option<(usize, bool, (usize, usize), bool)> = None;
    let mut stop = false;

    while !stop {
        let mut changed = false;
        // Pace the loop on the command channel: wake immediately on a command,
        // otherwise every 30 ms to poll for frames.
        match cmd_rx.recv_timeout(Duration::from_millis(30)) {
            Ok(LiveCmd::Stop) => break,
            Ok(LiveCmd::SetParams(p)) => {
                if p.capacity != params.capacity {
                    ring.set_capacity(p.capacity);
                }
                params = p;
                changed = true;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                LiveCmd::Stop => stop = true,
                LiveCmd::SetParams(p) => {
                    if p.capacity != params.capacity {
                        ring.set_capacity(p.capacity);
                    }
                    params = p;
                    changed = true;
                }
            }
        }
        if stop {
            break;
        }

        let n = source.poll_into(&mut ring);
        source.poll_darkflat(&mut ring);

        let status = (
            ring.len(),
            source.image_connected(),
            ring.dims(),
            ring.has_darkflat(),
        );
        if last_status != Some(status) {
            send(LiveEvent::Status {
                fill: status.0,
                connected: status.1,
                det: status.2,
                darkflat: status.3,
            });
            last_status = Some(status);
        }

        if (n > 0 || changed) && ring.len() >= 2 {
            let t0 = Instant::now();
            match reconstruct_slice(&ring, &params, engine.backend()) {
                Ok((ny, nx, data, nproj)) => {
                    // Overwrite the latest-wins slot rather than queueing: a
                    // backgrounded UI keeps at most this one slice, not a
                    // frame's worth per loop.
                    *image.lock().unwrap() = Some(LiveImage {
                        ny,
                        nx,
                        data,
                        ms: t0.elapsed().as_millis(),
                        nproj,
                    });
                }
                Err(e) => send(LiveEvent::Error(e.to_string())),
            }
        }
    }
}
