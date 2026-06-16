#!/usr/bin/env python
"""Generate the golden for tomoxide's Vo dead-stripe-removal test.

Reference: tomopy `prep/stripe.py:762` `remove_dead_stripe` (Nghia Vo's
algorithm 6). Calls the REAL tomopy function (tomopy 1.15.3 in the
tomopy-golden env) — no transcription — so the golden is exactly tomopy's output.

`_rs_dead` smooths each detector column over projections (`uniform_filter1d`
width 10), scores each column by its summed deviation from that smooth, detects
the unresponsive/fluctuating columns, and fills them by per-row linear
interpolation across the good columns (`RectBivariateSpline`, kx=ky=1). When
`norm=True` a residual `_rs_large` pass then removes wide stripes. The bilinear
fill is arithmetic, so it is held to the f32 round-off floor.

The input carries:
  * a smooth moving-sinusoid sinogram plus tiny distinct noise (no argsort ties),
  * injected large stripes (constant offset on cols 30/75/100), and
  * a near-dead, slowly-drifting column 55 (strictly monotonic → distinct values,
    so no ties even if it is left unfilled).

`snr=2` is used so the dead-column detection fires (at the tomopy default snr=3
the val2 gate caps below threshold for this fixture — same as the VoAll golden).
The two cases then differ structurally:
  * `norm=True`  — dead column filled AND the large stripes removed (residual pass),
  * `norm=False` — dead column filled only; the large stripes remain.

Run under the tomopy-golden env:
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_stripe_vodead_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - stripe_vodead_input.npy        (nproj, nslices, ncol) input, float32
  - tomopy_stripe_vodead_norm.npy  snr=2 size=51 norm=True
  - tomopy_stripe_vodead_raw.npy   snr=2 size=51 norm=False
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

    base = (1.0
            + 0.4 * np.sin(5.0 * col + 2.0 * ang)
            + 0.2 * np.cos(9.0 * col - ang))
    data = (base * row).astype("float32")
    data = data + (1e-3 * rng.standard_normal(data.shape)).astype("float32")

    # Injected large stripes (constant additive offset, identical across angles).
    for c, amp in [(30, 0.6), (75, -0.5), (100, 0.45)]:
        data[:, :, c] += np.float32(amp)

    # Near-dead, slowly-drifting column 55: strictly monotonic ramp → distinct
    # values (a perfectly flat column would be a non-portable 180-way argsort tie).
    data[:, :, 55] = (np.float32(1.2)
                      + np.linspace(0.0, 1e-2, nproj).astype("float32"))[:, None]

    data = np.ascontiguousarray(data.astype("float32"))

    out_norm = stripe.remove_dead_stripe(
        data.copy(), snr=2, size=51, norm=True, ncore=1,
    ).astype("float32")
    out_raw = stripe.remove_dead_stripe(
        data.copy(), snr=2, size=51, norm=False, ncore=1,
    ).astype("float32")

    np.save(os.path.join(OUT, "stripe_vodead_input.npy"), data)
    np.save(os.path.join(OUT, "tomopy_stripe_vodead_norm.npy"), out_norm)
    np.save(os.path.join(OUT, "tomopy_stripe_vodead_raw.npy"), out_raw)

    def col_change(out, c):
        return float(np.max(np.abs(out[:, :, c] - data[:, :, c])))

    print("tomopy", __import__("tomopy").__version__)
    print("input", data.shape, "range", float(data.min()), float(data.max()))
    print("dead col 55 change: norm=True", col_change(out_norm, 55),
          "norm=False", col_change(out_raw, 55))
    for c in (30, 75, 100):
        print(f"large col {c} change: norm=True {col_change(out_norm, c):.4g}"
              f"  norm=False {col_change(out_raw, c):.4g}")
    print("wrote fixtures to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
