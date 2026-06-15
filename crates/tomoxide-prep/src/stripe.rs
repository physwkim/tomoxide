//! Stripe-artifact removal (ports tomopy `prep/stripe.py` + tomocupy
//! `processing/remove_stripe.py`). Stubs in this scaffold; see
//! `docs/PORTING.md` §D. Dispatch on [`StripeMethod`].

use tomoxide_core::data::Tomo;
use tomoxide_core::error::{Error, Result};
use tomoxide_core::params::StripeMethod;

/// Remove stripes from a sinogram stack using the selected method.
pub fn remove_stripe(_data: &mut Tomo<f32>, method: StripeMethod) -> Result<()> {
    match method {
        StripeMethod::None => Ok(()),
        StripeMethod::Fw { .. } => Err(Error::todo(
            "stripe::remove_stripe_fw",
            "tomopy prep/stripe.py:88 (Fourier-Wavelet)",
        )),
        StripeMethod::Ti { .. } => Err(Error::todo(
            "stripe::remove_stripe_ti",
            "tomopy prep/stripe.py:179 (Titarenko)",
        )),
        StripeMethod::Sf { .. } => Err(Error::todo(
            "stripe::remove_stripe_sf",
            "tomopy libtomo/prep/stripe.c (smoothing filter)",
        )),
        StripeMethod::VoAll { .. } => Err(Error::todo(
            "stripe::remove_all_stripe",
            "tomocupy remove_stripe.remove_all_stripe (vo-all)",
        )),
    }
}
