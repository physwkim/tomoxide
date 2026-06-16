#!/usr/bin/env python
"""Golden for `scale` (tomopy `prep/alignment.py:460`).

Linearly scales a projection stack into [-1, 1] by its peak magnitude:
    scl = max(abs(prj.max()), abs(prj.min()))
    prj /= scl
returning (prj, scl). Pure order statistics (max/min/abs) plus an elementwise
f32 divide — no summation or transcendental — so the result is bit-exact
(delta=0) vs tomopy.

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_scale_golden.py
"""
import os

import numpy as np
from tomopy.prep.alignment import scale

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260616)
n0, n1, n2 = 5, 6, 7

# Several stacks with different peak-magnitude sign and scale, stored flat with
# a shapes/scl table (all share one shape here, but keep the flat layout general).
cases = []
# positive-dominated
cases.append((2.0 + 3.0 * rng.standard_normal((n0, n1, n2))).astype("float32"))
# negative-dominated (|min| is the peak)
cases.append((-5.0 + 2.0 * rng.standard_normal((n0, n1, n2))).astype("float32"))
# symmetric around zero
cases.append((4.0 * rng.standard_normal((n0, n1, n2))).astype("float32"))
# already small
cases.append((0.01 * rng.standard_normal((n0, n1, n2))).astype("float32"))

inputs, outputs, scls = [], [], []
for arr in cases:
    inp = arr.copy()
    out, scl = scale(arr.copy())
    inputs.append(inp.astype("float32"))
    outputs.append(np.asarray(out, dtype="float32"))
    scls.append(float(np.float32(scl)))
    print(f"scl={np.float32(scl)!r} out range "
          f"[{float(np.min(out)):.4f}, {float(np.max(out)):.4f}]")

np.save(os.path.join(OUT, "scale_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "scale_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "scale_scl.npy"), np.asarray(scls, dtype="float64"))
print("cases", len(cases), "shape", (n0, n1, n2))
