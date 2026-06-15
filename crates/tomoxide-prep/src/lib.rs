//! # tomoxide-prep
//!
//! Preprocessing for tomoxide: flat/dark [`normalize`]ation and minus-log,
//! [`stripe`] removal, [`phase`] retrieval, beam [`hardening`], and misc
//! [`filters`] (circular mask, NaN/neg scrubbing, median/outlier). Depends on
//! `tomoxide-core` only; device kernels are reached through backend traits.
//!
//! Real in this scaffold: `normalize`/`minus_log` (via the CPU backend),
//! `filters::{circ_mask, remove_nan, remove_neg}`. The rest are stubs that name
//! their upstream `file:line` (see `docs/PORTING.md` §D/§E).
#![forbid(unsafe_code)]

pub mod filters;
pub mod hardening;
pub mod normalize;
pub mod phase;
pub mod stripe;

pub use normalize::{minus_log, normalize, normalize_dataset};
pub use phase::retrieve_phase;
pub use stripe::remove_stripe;
