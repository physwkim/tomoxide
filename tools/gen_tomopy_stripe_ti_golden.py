#!/usr/bin/env python
"""Generate tomopy golden data for the Titarenko (`remove_stripe_ti`) parity test.

Runs tomopy 1.15.3 `remove_stripe_ti` on a fixed projection stack
`[proj, row, col]` carrying a smooth sinogram plus injected additive stripes.
The Titarenko method solves a finite-difference normal-equations system per
slice by conjugate gradient (f64) and combines the first/second-difference
corrected sinograms as `sqrt(d1*d2 + alpha*|min|)`, rounding each `_ring` to
f32. tomoxide reimplements the same f64 CG + f32 cast, so this is held to the
f32 round-off floor, not bit-exactness.

Only the default `nblock=0` (whole-sinogram) path is generated: tomopy's block
path `_ringb` (nblock>0) is unrunnable on modern numpy — its NaN guard
`np.where(np.isnan(mysino) is True)` is an always-False identity comparison that
raises on a 0-d array — so there is no reference output to compare against.

Run with the tomopy-enabled env:
    /Users/stevek/mamba/envs/tomopy-golden/bin/python \
        tools/gen_tomopy_stripe_ti_golden.py
"""
import multiprocessing as mp
import os

mp.set_start_method("fork", force=True)  # tomopy distribute_jobs spawns a Manager

import numpy as np
import tomopy.prep.stripe as stripe

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260615)
nproj, nrow, ncol = 180, 2, 128
ang = np.linspace(0.0, np.pi, nproj, endpoint=False)[:, None, None]
col = np.linspace(0.0, 1.0, ncol)[None, None, :]
row = (1.0 + 0.1 * np.arange(nrow))[None, :, None]

base = (1.0
        + 0.4 * np.sin(5.0 * col + 2.0 * ang)
        + 0.2 * np.cos(9.0 * col - ang))
data = (base * row).astype("float32")
data = data + (1e-3 * rng.standard_normal(data.shape)).astype("float32")

# Injected stripes: constant additive offset on a few columns across all angles.
for c, amp in [(30, 0.6), (75, -0.5), (100, 0.45)]:
    data[:, :, c] += np.float32(amp)

data = np.ascontiguousarray(data.astype("float32"))

out = stripe.remove_stripe_ti(
    data.copy(), nblock=0, alpha=1.5, ncore=1,
).astype("float32")
np.save(os.path.join(OUT, "tomopy_stripe_ti_nblock0.npy"), out)
print(f"nblock=0: max|Δ from input| = {float(np.max(np.abs(out - data))):.6g}")

np.save(os.path.join(OUT, "stripe_ti_input.npy"), data)
print("input", data.shape)
