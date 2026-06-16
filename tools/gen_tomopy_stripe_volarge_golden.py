#!/usr/bin/env python
"""Generate the golden for tomoxide's Vo large-stripe-removal test.

Reference: tomopy `prep/stripe.py:653` `remove_large_stripe` (Nghia Vo's
algorithm 5). Calls the REAL tomopy function (tomopy 1.15.3 in the
tomopy-golden env) — no transcription — so the golden is exactly tomopy's output.

`_rs_large` sorts each detector column over projections, median-smooths the
sorted profile along the column axis (footprint `(1, size)`), estimates a
per-column intensity factor from the central rows, detects the wide-stripe
columns (`_detect_stripe` + 1-px binary dilation), and overwrites *only* those
columns with the rank-smoothed profile mapped back through the sort order. The
smoothed values are pure rank-filter selections of existing f32 samples:
  * `norm=False` — the whole result is selections/copies of the input → Δ = 0.
  * `norm=True`  — the unmasked columns are additionally divided by their f32
    intensity factor, so it is held to the f32 round-off floor.

Distinct per-element noise is added so no column has two projections with the
same value (numpy-quicksort tie order is not portable, mirroring the VoAll/VoSort
goldens).

Run under the tomopy-golden env:
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_stripe_volarge_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - stripe_volarge_input.npy        (nproj, nslices, ncol) input, float32
  - tomopy_stripe_volarge_norm.npy  snr=3 size=51 drop_ratio=0.1 norm=True
  - tomopy_stripe_volarge_raw.npy   snr=3 size=51 drop_ratio=0.1 norm=False
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

    # Smooth sinogram-like data (a couple of moving sinusoids) plus tiny noise so
    # every column has distinct row values (no argsort ties).
    base = (1.0
            + 0.4 * np.sin(5.0 * col + 2.0 * ang)
            + 0.2 * np.cos(9.0 * col - ang))
    data = (base * row).astype("float32")
    data = data + (1e-3 * rng.standard_normal(data.shape)).astype("float32")

    # Injected large stripes: constant additive offset on a few columns, identical
    # across all angles/rows (the classic stripe → ring source).
    for c, amp in [(30, 0.6), (75, -0.5), (100, 0.45)]:
        data[:, :, c] += np.float32(amp)

    data = np.ascontiguousarray(data.astype("float32"))

    out_norm = stripe.remove_large_stripe(
        data.copy(), snr=3, size=51, drop_ratio=0.1, norm=True, ncore=1,
    ).astype("float32")
    out_raw = stripe.remove_large_stripe(
        data.copy(), snr=3, size=51, drop_ratio=0.1, norm=False, ncore=1,
    ).astype("float32")

    np.save(os.path.join(OUT, "stripe_volarge_input.npy"), data)
    np.save(os.path.join(OUT, "tomopy_stripe_volarge_norm.npy"), out_norm)
    np.save(os.path.join(OUT, "tomopy_stripe_volarge_raw.npy"), out_raw)
    print("tomopy", __import__("tomopy").__version__)
    print("input", data.shape, "range", float(data.min()), float(data.max()))
    print("norm=True  max|Δ| from input =", float(np.max(np.abs(out_norm - data))))
    print("norm=False max|Δ| from input =", float(np.max(np.abs(out_raw - data))))
    print("wrote fixtures to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
