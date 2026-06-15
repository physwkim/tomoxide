//! Rotation-center finding (ports tomopy `recon/rotation.py` + tomocupy
//! `find_center.py`). All stubs in this scaffold; see `docs/PORTING.md` §C.

use tomoxide_core::data::Tomo;
use tomoxide_core::error::{Error, Result};

/// Entropy-based center finding (tomopy `rotation.py:82`).
pub fn find_center(
    _sino: &Tomo<f32>,
    _theta: &[f32],
    _init: Option<f32>,
    _tol: f32,
) -> Result<f32> {
    Err(Error::todo(
        "center::find_center",
        "tomopy recon/rotation.py:82",
    ))
}

/// Nghia Vo's coarse+fine center search — the workhorse (tomopy `rotation.py:205`).
pub fn find_center_vo(
    _sino: &Tomo<f32>,
    _smin: f32,
    _smax: f32,
    _srad: f32,
    _step: f32,
) -> Result<f32> {
    Err(Error::todo(
        "center::find_center_vo",
        "tomopy recon/rotation.py:205 (_search_coarse/_search_fine)",
    ))
}

/// Phase-correlation between a 0°/180° pair (tomopy `rotation.py:391`).
pub fn find_center_pc(_proj0: &[f32], _proj180: &[f32], _tol: f32) -> Result<f32> {
    Err(Error::todo(
        "center::find_center_pc",
        "tomopy recon/rotation.py:391",
    ))
}

/// SIFT-feature center detection (tomocupy `find_center.py:99`).
pub fn find_center_sift(_proj0: &[f32], _proj180: &[f32]) -> Result<f32> {
    Err(Error::todo(
        "center::find_center_sift",
        "tomocupy find_center.py:99",
    ))
}
