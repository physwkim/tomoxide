#!/usr/bin/env python
"""Generate the golden for tomoxide's Vo fitting-based stripe-removal test.

Reference: tomopy `prep/stripe.py:520` `remove_stripe_based_fitting` (Nghia Vo's
algorithm 1, for low-pass stripes). Calls the REAL tomopy function
(tomopy 1.15.3 in the tomopy-golden env) — no transcription — so the golden is
exactly tomopy's output.

`_rs_fit` divides each sinogram by its Savitzky-Golay polynomial fit along the
projection axis (`savgol_filter(..., axis=0, mode='mirror')`), then re-multiplies
by a mean-matched 2-D Gaussian-smoothed copy of that fit (`_2d_filter`, an
`ifft2(fft2(.)·win2d)` band-pass with `(-1)^(x+y)` modulation and edge/mean
padding). The 2-D Fourier filter runs in float64, so it is held to the f32
round-off floor (like the Fourier-Wavelet and VoFilter paths), not bit-exactness.

The input is a smooth moving-sinusoid sinogram (strictly positive, so the
divide-by-fit is well-conditioned) plus a low-pass stripe — a smooth
column-dependent gain ramp — and tiny noise.

Run under the tomopy-golden env:
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_stripe_vofit_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - stripe_vofit_input.npy        (nproj, nslices, ncol) input, float32
  - tomopy_stripe_vofit_def.npy   order=3 sigma=(5,20) output, float32
  - tomopy_stripe_vofit_o1.npy    order=1 sigma=(3,10) output, float32
"""
import multiprocessing as mp
import os

mp.set_start_method("fork", force=True)  # tomopy distribute_jobs spawns a Manager

import numpy as np
import tomopy.prep.stripe as stripe

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

nproj, nrow, ncol = 180, 2, 128


def main():
    rng = np.random.default_rng(20260616)
    ang = np.linspace(0.0, np.pi, nproj, endpoint=False)[:, None, None]
    col = np.linspace(0.0, 1.0, ncol)[None, None, :]
    row = (1.0 + 0.1 * np.arange(nrow))[None, :, None]

    # Smooth, strictly-positive sinogram (so sinogram/sinofit is well-conditioned).
    base = (2.0
            + 0.4 * np.sin(5.0 * col + 2.0 * ang)
            + 0.2 * np.cos(9.0 * col - ang))
    data = (base * row).astype("float32")
    data = data + (1e-3 * rng.standard_normal(data.shape)).astype("float32")

    # Stripes algorithm 1 targets: a multiplicative per-column gain that is
    # CONSTANT across all projections/rows (a detector-gain stripe) and sharp
    # across the detector axis — so it sits in the column-Gaussian's stop-band
    # and is removed, unlike a slowly-varying gain (which is legitimate signal).
    gain = np.ones(ncol, dtype="float32")
    for c, g in [(30, 1.30), (75, 0.78), (100, 1.22), (101, 0.85)]:
        gain[c] = np.float32(g)
    data = (data * gain[None, None, :]).astype("float32")
    data = np.ascontiguousarray(data)

    out_def = stripe.remove_stripe_based_fitting(
        data.copy(), order=3, sigma=(5, 20), ncore=1).astype("float32")
    out_o1 = stripe.remove_stripe_based_fitting(
        data.copy(), order=1, sigma=(3, 10), ncore=1).astype("float32")

    np.save(os.path.join(OUT, "stripe_vofit_input.npy"), data)
    np.save(os.path.join(OUT, "tomopy_stripe_vofit_def.npy"), out_def)
    np.save(os.path.join(OUT, "tomopy_stripe_vofit_o1.npy"), out_o1)
    print("tomopy", __import__("tomopy").__version__)
    print("input", data.shape, "range", float(data.min()), float(data.max()))
    print("order=3 max|Δ| from input =", float(np.max(np.abs(out_def - data))))
    print("order=1 max|Δ| from input =", float(np.max(np.abs(out_o1 - data))))
    print("wrote fixtures to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
