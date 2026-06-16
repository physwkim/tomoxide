#!/usr/bin/env python
"""Generate the golden for tomoxide's Vo sorting-based stripe-removal test.

Reference: tomopy `prep/stripe.py:363` `remove_stripe_based_sorting` (Nghia Vo's
algorithm 3). This calls the REAL tomopy function (tomopy 1.15.3 in the
tomopy-golden env) — no transcription — so the golden is exactly tomopy's output.

`_rs_sort` is a pure rank-filter selection (median_filter on the per-column
argsorted values, then unsort), so on tie-free columns the result is bit-exact in
float32. The input therefore carries small distinct per-element noise so no column
has two projections with the same value (numpy-quicksort tie order is not
portable, mirroring the VoAll golden's caveat).

Run under the tomopy-golden env:
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_stripe_vosort_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - stripe_vosort_input.npy        (nproj, nslices, ncol) input, float32
  - tomopy_stripe_vosort_dim1.npy  size=None (->5), dim=1 output, float32
  - tomopy_stripe_vosort_dim2.npy  size=5,    dim=2 output, float32
"""
import os

import numpy as np
import tomopy

NPROJ, NS, NCOL = 180, 2, 64


def main():
    rng = np.random.default_rng(0)

    # Deterministic smooth sinogram stack with injected full/partial stripes.
    y = np.linspace(-1.0, 1.0, NCOL, dtype=np.float64)
    stack = np.empty((NPROJ, NS, NCOL), dtype=np.float32)
    for s in range(NS):
        for p in range(NPROJ):
            stack[p, s, :] = 1.0 + 0.5 * np.cos(2.0 * np.pi * y + 0.05 * p + s)

    stripes = np.zeros(NCOL, dtype=np.float32)
    stripes[10] = 0.3   # full stripe
    stripes[30] = -0.2  # full stripe
    stripes[50] = 0.15  # full stripe
    stack += stripes[None, None, :]
    # Partial stripe (only part of the projection range) on another column.
    stack[: NPROJ // 2, :, 40] += 0.25

    # Distinct per-element noise so no column has tied values across projections.
    stack += (rng.standard_normal(stack.shape) * 1e-3).astype(np.float32)
    stack = stack.astype(np.float32)

    out_dim1 = tomopy.remove_stripe_based_sorting(stack.copy(), size=None, dim=1).astype("float32")
    out_dim2 = tomopy.remove_stripe_based_sorting(stack.copy(), size=5, dim=2).astype("float32")

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")
    np.save(os.path.join(out, "stripe_vosort_input.npy"), stack)
    np.save(os.path.join(out, "tomopy_stripe_vosort_dim1.npy"), out_dim1)
    np.save(os.path.join(out, "tomopy_stripe_vosort_dim2.npy"), out_dim2)
    print("tomopy", tomopy.__version__)
    print("input", stack.shape, "range", float(stack.min()), float(stack.max()))
    print("dim1 range", float(out_dim1.min()), float(out_dim1.max()))
    print("dim2 range", float(out_dim2.min()), float(out_dim2.max()))
    print("wrote fixtures to", os.path.normpath(out))


if __name__ == "__main__":
    main()
