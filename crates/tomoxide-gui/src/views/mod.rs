//! Per-mode screens. Each view owns its siplot widgets (constructed once with
//! the wgpu `RenderState`) and receives worker [`Event`](crate::worker::Event)s
//! routed by the app shell.
//!
//! `PlotId` allocation convention (siplot ids must not collide; some widgets
//! reserve a small range): Data 0–29, Tune 30–49, Center 50–69.

pub mod data;
