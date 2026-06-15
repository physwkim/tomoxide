//! Beam-hardening correction (ports tomocupy
//! `processing/external/hardening.py`). Stub; see `docs/PORTING.md` §D.

use tomoxide_core::data::Tomo;
use tomoxide_core::error::{Error, Result};

/// Per-row spectral beam-hardening correction.
///
/// The real corrector is parameterized by scintillator/sample/filter
/// materials and the source spectrum (tomocupy config `beam-hardening-*`).
pub fn beam_correct(_data: &mut Tomo<f32>, _start_row: usize, _end_row: usize) -> Result<()> {
    Err(Error::todo(
        "hardening::beam_correct",
        "tomocupy processing/external/hardening.py:50",
    ))
}
