//! Data mode (docs/GUI.md §2 Data): open a DXchange file, inspect its
//! metadata, browse projections, plot theta, and inspect raw sinograms.

use std::sync::Arc;
use std::sync::mpsc::Sender;

use rsplot::egui_wgpu::RenderState;
use rsplot::{CurveData, Frame, FrameLoader, ImageStack, ImageView, ItemHandle, Plot1D, egui};

use crate::worker::{DatasetMeta, Job};

/// Projection-browser loader: same `"file::dataset::index"` source format as
/// rsplot's `Hdf5FrameLoader`, but reading through tomoxide's dtype-dispatching
/// HDF5 frame read — real beamline stacks are usually `uint16`, which rsplot's
/// own loader rejects (it reads only 4/8-byte float datasets).
struct DxFrameLoader;

impl FrameLoader for DxFrameLoader {
    fn load(&self, source: &str) -> Option<Frame> {
        let parts: Vec<&str> = source.split("::").collect();
        let [path, data_path, index] = parts.as_slice() else {
            return None;
        };
        let index = index.parse::<usize>().ok()?;
        let (ny, nx, data) = tomoxide::io::read_h5_frame(path, data_path, index).ok()?;
        Some(Frame::new(
            nx as u32,
            ny as u32,
            data,
            Some(source.to_string()),
        ))
    }
}

pub struct DataView {
    path_input: String,
    meta: Option<Arc<DatasetMeta>>,
    /// Projection browser: lazy per-frame HDF5 loads on background threads.
    stack: ImageStack,
    theta_plot: Plot1D,
    theta_curve: Option<ItemHandle>,
    sino_plot: ImageView,
    /// Detector row selected by the slider.
    sino_row: usize,
    /// Row of an in-flight ReadSinogram job (one outstanding request at a
    /// time; re-issued when the slider moved past it — a natural debounce).
    sino_pending: Option<usize>,
    /// Row of the sinogram currently displayed.
    sino_shown: Option<usize>,
}

impl DataView {
    pub fn new(render_state: &RenderState) -> Self {
        let mut stack = ImageStack::new(render_state, 0);
        stack.set_loader(Arc::new(DxFrameLoader));
        stack.set_n_prefetch(2);
        stack.set_table_visible(false);
        let mut theta_plot = Plot1D::new(render_state, 10);
        theta_plot.set_graph_title("theta");
        // An ImageView (not a bare Plot2D) so the crosshair readout can show the
        // pixel value under the cursor via value_changed() — the silx
        // PositionInfo "Data" column. Side histograms and the dedicated colorbar
        // are off so the inspector stays a plain image + readout (the aspect
        // ratio is freed too: sinograms are [nproj × nx], shown stretched).
        let mut sino_plot = ImageView::new(render_state, 20);
        sino_plot.set_side_histogram_displayed(false);
        sino_plot.set_show_colorbar(false);
        sino_plot.image_plot_mut().set_keep_data_aspect_ratio(false);
        sino_plot.image_plot_mut().set_graph_title("sinogram");
        DataView {
            path_input: String::new(),
            meta: None,
            stack,
            theta_plot,
            theta_curve: None,
            sino_plot,
            sino_row: 0,
            sino_pending: None,
            sino_shown: None,
        }
    }

    /// Route a finished open here (from the app's event loop).
    pub fn on_dataset(&mut self, meta: Arc<DatasetMeta>) {
        self.path_input = meta.path.display().to_string();
        // Projection browser sources: one frame per angle.
        let sources: Vec<String> = (0..meta.nproj)
            .map(|i| {
                format!(
                    "{}::{}::{i}",
                    meta.path.display(),
                    tomoxide::io::dxchange::DATA
                )
            })
            .collect();
        self.stack.set_sources(sources);
        // Raw-count display range from frame 0 (the stack has no autoscale).
        self.stack.set_colormap(rsplot::Colormap::viridis(
            meta.data_range.0 as f64,
            meta.data_range.1 as f64,
        ));
        // Theta curve in degrees over the projection index.
        let x: Vec<f64> = (0..meta.theta.len()).map(|i| i as f64).collect();
        let y: Vec<f64> = meta
            .theta
            .iter()
            .map(|&r| (r as f64).to_degrees())
            .collect();
        let color = egui::Color32::LIGHT_BLUE;
        match self.theta_curve {
            Some(h) => {
                self.theta_plot
                    .update_curve_data(h, &CurveData::new(x, y, color));
            }
            None => {
                self.theta_curve =
                    Some(
                        self.theta_plot
                            .add_curve_with_legend(&x, &y, color, "theta (deg)"),
                    );
            }
        }
        // Fresh dataset: reset the sinogram inspector to the mid row.
        self.sino_row = meta.nz / 2;
        self.sino_pending = None;
        self.sino_shown = None;
        self.meta = Some(meta);
    }

    /// Route a finished sinogram read here (from the app's event loop).
    pub fn on_sinogram(&mut self, row: usize, nproj: usize, nx: usize, data: &[f32]) {
        self.sino_pending = None;
        self.sino_shown = Some(row);
        let cmap = super::autoscale_viridis(data);
        let _ = self
            .sino_plot
            .set_image(nx as u32, nproj as u32, data, cmap);
        self.sino_plot
            .image_plot_mut()
            .set_graph_title(format!("sinogram — row {row}"));
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, jobs: &Sender<Job>) {
        // One outstanding sinogram request at a time; catch up when idle.
        if self.meta.is_some()
            && self.sino_pending.is_none()
            && self.sino_shown != Some(self.sino_row)
            && jobs.send(Job::ReadSinogram { row: self.sino_row }).is_ok()
        {
            self.sino_pending = Some(self.sino_row);
        }

        ui.horizontal(|ui| {
            if ui.button("Open…").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("HDF5", &["h5", "hdf5"])
                    .pick_file()
            {
                self.path_input = path.display().to_string();
                let _ = jobs.send(Job::OpenDataset(path));
            }
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.path_input)
                    .hint_text("path/to/dxchange.h5")
                    .desired_width(f32::INFINITY),
            );
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                let _ = jobs.send(Job::OpenDataset(self.path_input.clone().into()));
            }
        });
        ui.separator();

        let Some(meta) = self.meta.clone() else {
            ui.label("Open a DXchange HDF5 file to browse projections, theta, and sinograms.");
            return;
        };

        egui::Panel::left("data_side")
            .resizable(true)
            .default_size(360.0)
            .show_inside(ui, |ui| {
                ui.heading("Dataset");
                egui::Grid::new("data_meta_grid")
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("file");
                        ui.monospace(
                            meta.path
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default(),
                        );
                        ui.end_row();
                        ui.label("projections");
                        ui.monospace(meta.nproj.to_string());
                        ui.end_row();
                        ui.label("rows (nz)");
                        ui.monospace(meta.nz.to_string());
                        ui.end_row();
                        ui.label("columns (nx)");
                        ui.monospace(meta.nx.to_string());
                        ui.end_row();
                        ui.label("flat / dark");
                        ui.monospace(format!("{} / {}", meta.nflat, meta.ndark));
                        ui.end_row();
                        if let (Some(first), Some(last)) = (meta.theta.first(), meta.theta.last()) {
                            ui.label("theta range");
                            ui.monospace(format!(
                                "{:.2}° … {:.2}°",
                                (*first as f64).to_degrees(),
                                (*last as f64).to_degrees()
                            ));
                            ui.end_row();
                        }
                    });
                ui.separator();
                self.theta_plot.show(ui);
            });

        egui::Panel::bottom("data_sino")
            .resizable(true)
            .default_size(320.0)
            .show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("detector row");
                    ui.add(egui::Slider::new(
                        &mut self.sino_row,
                        0..=meta.nz.saturating_sub(1),
                    ));
                    if self.sino_pending.is_some() {
                        ui.spinner();
                    }
                });
                self.sino_plot.show(ui, None, None);
                super::value_readout(ui, self.sino_plot.value_changed());
            });

        // Remaining central space: the projection browser.
        self.stack.ui(ui);
    }
}
