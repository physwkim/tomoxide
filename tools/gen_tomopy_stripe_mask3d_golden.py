#!/usr/bin/env python
"""Generate the golden for tomoxide's `stripes_mask3d` test.

Reference: tomopy `prep/stripe.py:1058` `stripes_mask3d` (Daniil Kazantsev's
3D stripe-mask algorithm, :cite:`Kazantsev:23`), backed by the libtomo C kernel
`libtomo/prep/stripes_detect3d.c::stripesmask3d_main_float`. Calls the REAL
tomopy function (tomopy 1.15.3 in the tomopy-golden env) — no transcription.

`stripes_mask3d` consumes the `[0, 1]` weights produced by `stripes_detect3d`
and returns a binary `bool` mask: threshold the weights, then enforce
consistency in depth and vertical (along-angle) directions, drop short stripes,
and iteratively merge nearby ones. It is pure integer/bool logic with a single
`f32` threshold compare and `(int)(0.01*sensitivity*len)` thresholds, so the
Rust port reproduces it BIT-EXACTLY (identical bool mask).

The weights are themselves generated here from `stripes_detect3d` on a stack
with constant-across-angle full stripes (so they are long along the angle axis
and survive the consistency/short-stripe filters), plus a partial-depth stripe.

Run under the tomopy-golden env:
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_stripe_mask3d_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - stripe_mask3d_weights.npy   (dz=angle, dy=detY, dx=detX) weights, float32
  - tomopy_mask3d_def.npy       defaults mask, bool
  - tomopy_mask3d_alt.npy       alt-param mask, bool
"""
import multiprocessing as mp
import os

mp.set_start_method("fork", force=True)  # tomopy distribute_jobs spawns a Manager

import numpy as np
import tomopy.prep.stripe as stripe

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

dz, dy, dx = 60, 16, 64  # angle, detY(depth), detX(horizontal)


def main():
    rng = np.random.default_rng(20260616)
    ang = np.linspace(0.0, np.pi, dz, endpoint=False)[:, None, None]
    dety = np.linspace(0.0, 1.0, dy)[None, :, None]
    detx = np.linspace(0.0, 1.0, dx)[None, None, :]

    base = (0.5
            + 0.15 * np.sin(4.0 * detx + 1.5 * ang)
            + 0.10 * np.cos(7.0 * detx - ang)
            + 0.05 * np.sin(3.0 * dety + ang))
    data = np.broadcast_to(base, (dz, dy, dx)).astype("float32").copy()
    data += (5e-3 * rng.standard_normal(data.shape)).astype("float32")

    # Full stripes (all angle, all detY) and one partial-depth stripe.
    data[:, :, 20] += np.float32(0.22)
    data[:, :, 45] -= np.float32(0.20)
    data[:, : dy // 2, 50] += np.float32(0.25)
    data = np.ascontiguousarray(data)

    weights = stripe.stripes_detect3d(data.copy(), size=10, radius=3,
                                      ncore=1).astype("float32")
    weights = np.ascontiguousarray(weights)

    mask_def = stripe.stripes_mask3d(
        weights.copy(), threshold=0.6, min_stripe_length=20,
        min_stripe_depth=10, min_stripe_width=5, sensitivity_perc=85.0,
        ncore=1)
    mask_alt = stripe.stripes_mask3d(
        weights.copy(), threshold=0.5, min_stripe_length=10,
        min_stripe_depth=4, min_stripe_width=3, sensitivity_perc=60.0,
        ncore=1)

    np.save(os.path.join(OUT, "stripe_mask3d_weights.npy"), weights)
    np.save(os.path.join(OUT, "tomopy_mask3d_def.npy"), mask_def)
    np.save(os.path.join(OUT, "tomopy_mask3d_alt.npy"), mask_alt)
    print("tomopy", __import__("tomopy").__version__)
    print("weights", weights.shape, "range", float(weights.min()), float(weights.max()))
    print("mask_def dtype", mask_def.dtype, "True count", int(mask_def.sum()),
          "of", mask_def.size)
    print("mask_alt dtype", mask_alt.dtype, "True count", int(mask_alt.sum()))
    # Per-column True totals — the stripe columns should dominate.
    cols = mask_def.sum(axis=(0, 1))
    hot = np.argsort(cols)[::-1][:6]
    print("def hottest detX cols:", [(int(c), int(cols[c])) for c in hot])
    print("wrote fixtures to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
