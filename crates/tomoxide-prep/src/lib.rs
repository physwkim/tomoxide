//! # tomoxide-prep
//!
//! Preprocessing for tomoxide: flat/dark [`normalize`]ation and minus-log,
//! [`stripe`] removal, [`phase`] retrieval, beam [`hardening`], and misc
//! [`filters`] (circular mask, NaN/neg scrubbing, median/outlier). Depends on
//! `tomoxide-core` only; device kernels are reached through backend traits.
//!
//! Real in this scaffold: `normalize`/`minus_log` (via the CPU backend),
//! `filters::{circ_mask, remove_nan, remove_neg, median_filter_nonfinite,
//! adjust_range, median_filter, remove_outlier1d}`, `alignment::scale`. The rest
//! are stubs that name their upstream `file:line` (see `docs/PORTING.md` §D/§E).
#![forbid(unsafe_code)]

pub mod alignment;
mod fft;
pub mod filters;
pub mod hardening;
pub mod morph;
pub mod normalize;
pub mod phase;
pub mod stripe;
pub mod stripe3d;
mod wavelet;

pub use alignment::scale;
pub use morph::{downsample, pad, sino_360_to_180, upsample, PadMode, Rotation};
pub use normalize::{
    minus_log, normalize, normalize_bg, normalize_dataset, normalize_nf, Averaging,
};
pub use phase::retrieve_phase;
pub use stripe::remove_stripe;
pub use stripe3d::{stripes_detect3d, stripes_mask3d};
