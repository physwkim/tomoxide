#!/usr/bin/env python
"""Generate the golden for tomoxide's `stripes_detect3d` test.

Reference: tomopy `prep/stripe.py:984` `stripes_detect3d` (Daniil Kazantsev's
3D stripe-detection algorithm, :cite:`Kazantsev:23`), backed by the libtomo C
kernel `libtomo/prep/stripes_detect3d.c::stripesdetect3d_main_float`. Calls the
REAL tomopy function (tomopy 1.15.3 in the tomopy-golden env) — no
transcription — so the golden is exactly tomopy's output.

The kernel is pure float32 arithmetic with no FFT (6-stencil mean smoothing →
horizontal forward gradient with step 2 → parallel/orthogonal mean-ratio map →
vertical median filter), so the Rust port reproduces it BIT-EXACTLY (Δ = 0).

Input is a normalized-projection-like 3D stack in [angle, detY(depth),
detX(horizontal)] orientation, smooth and ~[0, 1], with a few constant-across-
angle detector-gain stripes (full and partial) so the returned weights dip
below 0.5 at the stripe edges.

Run under the tomopy-golden env:
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_stripe_detect3d_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - stripe_detect3d_input.npy   (dz=angle, dy=detY, dx=detX) input, float32
  - tomopy_detect3d_def.npy     size=10 radius=3 weights, float32
  - tomopy_detect3d_alt.npy     size=5  radius=2 weights, float32
"""
import multiprocessing as mp
import os

mp.set_start_method("fork", force=True)  # tomopy distribute_jobs spawns a Manager

import numpy as np
import tomopy.prep.stripe as stripe

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

dz, dy, dx = 40, 8, 64  # angle, detY(depth), detX(horizontal)


def main():
    rng = np.random.default_rng(20260616)
    ang = np.linspace(0.0, np.pi, dz, endpoint=False)[:, None, None]
    dety = np.linspace(0.0, 1.0, dy)[None, :, None]
    detx = np.linspace(0.0, 1.0, dx)[None, None, :]

    # Smooth, ~[0, 1] normalized-projection-like stack.
    base = (0.5
            + 0.15 * np.sin(4.0 * detx + 1.5 * ang)
            + 0.10 * np.cos(7.0 * detx - ang)
            + 0.05 * np.sin(3.0 * dety + ang))
    data = np.broadcast_to(base, (dz, dy, dx)).astype("float32").copy()
    data += (5e-3 * rng.standard_normal(data.shape)).astype("float32")

    # Constant-across-angle additive stripes (sharp detX edges). Full stripes
    # span all detY; the col-50 stripe is partial (only the first half of detY).
    data[:, :, 20] += np.float32(0.20)
    data[:, :, 45] -= np.float32(0.18)
    data[:, : dy // 2, 50] += np.float32(0.25)
    data = np.ascontiguousarray(data)

    out_def = stripe.stripes_detect3d(data.copy(), size=10, radius=3,
                                      ncore=1).astype("float32")
    out_alt = stripe.stripes_detect3d(data.copy(), size=5, radius=2,
                                      ncore=1).astype("float32")

    np.save(os.path.join(OUT, "stripe_detect3d_input.npy"), data)
    np.save(os.path.join(OUT, "tomopy_detect3d_def.npy"), out_def)
    np.save(os.path.join(OUT, "tomopy_detect3d_alt.npy"), out_alt)
    print("tomopy", __import__("tomopy").__version__)
    print("input", data.shape, "range", float(data.min()), float(data.max()))
    print("def weights range", float(out_def.min()), float(out_def.max()))
    print("alt weights range", float(out_alt.min()), float(out_alt.max()))
    # Stripe-edge columns (19/21, 44/46, 49/51) should carry sub-0.5 weights.
    for c in (19, 20, 21, 44, 45, 46, 49, 50, 51):
        print(f"  col {c}: def min weight = {float(out_def[:, :, c].min()):.4f}")
    print("wrote fixtures to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
