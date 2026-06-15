//! Stripe-artifact removal (ports tomopy `prep/stripe.py` + tomocupy
//! `processing/remove_stripe.py`). The smoothing-filter method (`Sf`) is
//! implemented; the rest are stubs. See `docs/PORTING.md` §D. Dispatch on
//! [`StripeMethod`].

use tomoxide_core::data::{Layout, Tomo};
use tomoxide_core::error::{Error, Result};
use tomoxide_core::params::StripeMethod;

/// Remove stripes from a sinogram stack using the selected method.
pub fn remove_stripe(data: &mut Tomo<f32>, method: StripeMethod) -> Result<()> {
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
        StripeMethod::Sf { size } => remove_stripe_sf(data, size),
        StripeMethod::VoAll { .. } => Err(Error::todo(
            "stripe::remove_all_stripe",
            "tomocupy remove_stripe.remove_all_stripe (vo-all)",
        )),
    }
}

/// Smoothing-filter stripe removal — a direct port of tomopy
/// `libtomo/prep/stripe.c::remove_stripe_sf`.
///
/// For each reconstruction slice (the `row` axis) the average sinogram row
/// (column-wise mean over projections) is computed, smoothed by a width-`size`
/// moving average with clamp-to-edge boundaries, and the residual
/// `average − smoothed` is subtracted from every projection in that column. All
/// arithmetic is f32 in the upstream summation order, so the result matches
/// tomopy bit-for-bit. Projector-independent.
fn remove_stripe_sf(data: &mut Tomo<f32>, size: usize) -> Result<()> {
    let target = data.layout;
    // tomopy's `remove_stripe_sf` indexes `data[j + s*dz + p*dy*dz]` over
    // `(dx=proj, dy=row, dz=col)` — i.e. the `[angle, row, col]` projection
    // layout.
    let mut proj = data.to_layout(Layout::Projection);
    let (dx, dy, dz) = proj.array.dim();
    if dx == 0 || dy == 0 || dz == 0 || size == 0 {
        return Ok(());
    }
    let arr = &mut proj.array;
    let half = (size / 2) as isize; // C: `size / 2`, integer division
    let last = dz as isize - 1;
    let dxf = dx as f32;
    let sizef = size as f32;
    let mut average_row = vec![0.0f32; dz];
    let mut smooth_row = vec![0.0f32; dz];

    for s in 0..dy {
        // Average row: column-wise mean over projections (each term divided by
        // `dx` before summing, exactly as the C does, to match rounding).
        for j in 0..dz {
            let mut acc = 0.0f32;
            for p in 0..dx {
                acc += arr[[p, s, j]] / dxf;
            }
            average_row[j] = acc;
        }
        // Smooth the average row with a width-`size` moving average, clamping
        // out-of-range taps to the nearest edge.
        for (i, sv) in smooth_row.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for jj in 0..size {
                let mut k = i as isize + jj as isize - half;
                if k < 0 {
                    k = 0;
                }
                if k > last {
                    k = last;
                }
                acc += average_row[k as usize];
            }
            *sv = acc / sizef;
        }
        // Subtract the column residual from every projection in this slice.
        for p in 0..dx {
            for j in 0..dz {
                arr[[p, s, j]] -= average_row[j] - smooth_row[j];
            }
        }
    }

    *data = proj.to_layout(target);
    Ok(())
}
