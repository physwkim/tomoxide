//! tomoxide-gui — desktop front-end for tomoxide, built on rsplot (egui+wgpu).
//!
//! Design: docs/GUI.md. Seven modes (Data / Tune / Center / Run / Output /
//! XANES / Live) behind a left mode rail, with a session log pane and a status
//! bar. M1 implements the offline preview loop (Data, Tune, Center + recipes);
//! M2 adds Run/Output; M4 adds the XANES chemical-mapping screen; M3's first
//! cut adds Live (rsdm PVA streaming Z-slice reconstruction).

mod app;
mod live;
mod project;
mod views;
mod worker;

fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    // `tomoxide-gui [FILE] [--mode data|tune|center|run|output|xanes|live]`:
    // optionally open a DXchange .h5 and/or start on a specific mode.
    let mut open: Option<std::path::PathBuf> = None;
    let mut mode: Option<app::Mode> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--mode" {
            match args.next().as_deref().map(app::Mode::from_name) {
                Some(Some(m)) => mode = Some(m),
                _ => log::warn!("--mode: expected data|tune|center|run|output|xanes|live"),
            }
        } else {
            open = Some(std::path::PathBuf::from(arg));
        }
    }
    let options = eframe::NativeOptions {
        // rsplot widgets require the wgpu renderer (cc.wgpu_render_state).
        renderer: eframe::Renderer::Wgpu,
        viewport: rsplot::egui::ViewportBuilder::default()
            .with_title("tomoxide")
            .with_inner_size([1440.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native(
        "tomoxide",
        options,
        Box::new(move |cc| Ok(Box::new(app::App::new(cc, open, mode)) as Box<dyn eframe::App>)),
    )
}
