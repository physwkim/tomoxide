//! Phase retrieval (ports tomopy `prep/phase.py` + tomocupy
//! `processing/retrieve_phase.py`). Stub; see `docs/PORTING.md` §D.

use tomoxide_core::data::Tomo;
use tomoxide_core::error::{Error, Result};
use tomoxide_core::params::PhaseMethod;

/// Single-step phase retrieval on a projection stack.
///
/// Paganin params (`pixel_size` cm, `dist` cm, `energy` keV, `alpha`) live in
/// [`PhaseMethod::Paganin`].
pub fn retrieve_phase(_data: &mut Tomo<f32>, method: PhaseMethod) -> Result<()> {
    match method {
        PhaseMethod::None => Ok(()),
        PhaseMethod::Paganin { .. } => Err(Error::todo(
            "phase::retrieve_phase (Paganin)",
            "tomopy prep/phase.py:80; tomocupy retrieve_phase.paganin_filter:59",
        )),
        PhaseMethod::GPaganin => Err(Error::todo(
            "phase::retrieve_phase (Gpaganin)",
            "tomocupy retrieve_phase (Gpaganin)",
        )),
        PhaseMethod::Farago => Err(Error::todo(
            "phase::retrieve_phase (farago)",
            "tomocupy retrieve_phase.farago_filter:110",
        )),
    }
}
