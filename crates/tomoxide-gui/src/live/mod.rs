//! Live streaming reconstruction subsystem (docs/GUI.md §2.6): the projection
//! ring buffer, the rsdm PVA source that fills it, and the per-loop Z-slice
//! reconstruction. Driven by [`crate::views::live::LiveView`], which owns the
//! live thread these pieces run on.

pub mod recon_loop;
pub mod ring;
pub mod source;
