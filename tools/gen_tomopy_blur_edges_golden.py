#!/usr/bin/env python
"""Golden for `blur_edges` (tomopy `prep/alignment.py:482`).

Multiplies every projection image by a radial feather mask: within a projection
`rad = sqrt((row - dy/2)**2 + (col - dz/2)**2)`, mask is 1 where `rad < low*rmax`,
0 where `rad > high*rmax`, and a linear ramp `(rmax - rad)/(rmax - rmin)` in
between. `sqrt` is IEEE-correctly-rounded and the rest is plain f64 arithmetic;
the final in-place `float32 *= float64` is the f64 product cast to f32, so the
result is bit-exact (delta=0).

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_blur_edges_golden.py
"""
import os

import numpy as np
from tomopy.prep.alignment import blur_edges

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260616)
# (dx, dy, dz) = (proj, y, z); even and odd image dims to exercise the centre.
n0, n1, n2 = 4, 18, 23
base = (1.0 + 0.4 * rng.standard_normal((n0, n1, n2))).astype("float32")

# (low, high): the upstream default plus a non-zero inner radius.
cases = [
    (0.0, 0.8),
    (0.2, 0.9),
]

inputs, outputs, params = [], [], []
for low, high in cases:
    inp = base.copy()
    out = np.asarray(blur_edges(base.copy(), low=low, high=high), dtype="float32")
    inputs.append(inp.astype("float32"))
    outputs.append(out)
    params.append((float(low), float(high)))
    print(f"low={low} high={high}: out range [{float(out.min()):.4f}, "
          f"{float(out.max()):.4f}] zeros={int(np.count_nonzero(out == 0))}")

np.save(os.path.join(OUT, "blur_edges_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "blur_edges_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "blur_edges_params.npy"),
        np.asarray(params, dtype="float64"))
print("cases", len(cases), "shape", (n0, n1, n2))
