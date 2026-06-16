#!/usr/bin/env python
"""Golden for `add_drift` (tomopy `sim/project.py:80`).

Applies a sinusoidal illumination drift along the rotation (axis-0) dimension:
each angle `i` is scaled by `drift[i] = amp·sin(2π·i/period) + mean +
linspace(0,1)[i]`, constant across the detector. This is DETERMINISTIC (no RNG),
so — unlike the add_gaussian/poisson/... noise models held to distribution
parity — it reaches true tomopy numeric parity.

tomopy multiplies an f64 drift by the f32 input, so the result is f64; tomoxide
stores f32, so the golden is the tomopy output cast to f32 and the port computes
the f64 product then casts to f32 (Δ=0 in f32 expected).

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_add_drift_golden.py
"""
import os

import numpy as np
from tomopy.sim.project import add_drift

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260616)
n0, n1, n2 = 32, 8, 12
img = (3.0 + np.sin(np.linspace(0, 5, n1)[:, None] + np.linspace(0, 4, n2)[None, :])).astype("float32")
arr0 = np.broadcast_to(img, (n0, n1, n2)).astype("float32").copy()
arr0 += (0.4 * rng.standard_normal((n0, n1, n2))).astype("float32")

# (amp, period, mean): defaults, then a few variations (incl. non-integer period).
cases = [
    (0.2, 50.0, 1.0),    # tomopy defaults
    (0.5, 13.0, 0.8),    # shorter period
    (0.1, 27.5, 1.2),    # fractional period
]

inputs, outputs, params = [], [], []
for amp, period, mean in cases:
    inp = arr0.copy()
    out = add_drift(arr0.copy(), amp=amp, period=period, mean=mean)
    inputs.append(inp.astype("float32"))
    outputs.append(np.asarray(out, dtype="float32"))  # tomopy returns f64 -> cast to f32
    params.append((amp, period, mean))
    print(f"amp={amp} period={period} mean={mean}: "
          f"out dtype(orig)={np.asarray(out).dtype} range "
          f"[{float(np.min(out)):.4f}, {float(np.max(out)):.4f}]")

np.save(os.path.join(OUT, "add_drift_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "add_drift_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "add_drift_params.npy"),
        np.asarray(params, dtype="float64"))
print("cases", len(cases), "shape", (n0, n1, n2))
