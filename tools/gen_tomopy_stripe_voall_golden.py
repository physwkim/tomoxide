#!/usr/bin/env python
"""Generate tomopy golden data for the Vo all-stripe parity test.

Runs tomopy 1.15.3 `remove_all_stripe` on a fixed projection stack
`[proj, row, col]` that carries a smooth sinogram plus injected large additive
stripes and one dead (flat) column — exercising `_rs_dead` (detection +
bilinear fill + `_rs_large`) and `_rs_sort`. tomoxide processes the SAME array
offline; this composite is held to a tolerance (it reimplements scipy
primitives — uniform_filter1d, median_filter, polyfit, RectBivariateSpline —
whose numerics differ slightly), not bit-exactness.

Run with the tomopy-enabled env:
    /Users/stevek/mamba/envs/tomopy-golden/bin/python \
        tools/gen_tomopy_stripe_voall_golden.py
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

# One near-dead (low-response) column: a near-flat, slowly drifting level (a
# small monotonic ramp over angles) so `_rs_dead`'s detection fires on it. The
# ramp is STRICTLY MONOTONIC, hence every angle value is DISTINCT — a perfectly
# constant column would be an exact 180-way tie, and `_rs_sort`/`_rs_large`
# scatter rank-smoothed values back through `np.argsort`, whose tie order
# (unstable quicksort) is implementation- and even numpy-version-defined, i.e.
# outside the well-defined parity domain. Distinct values make the output
# unambiguous, matching the tie-avoidance already applied to the base data.
data[:, :, 55] = (np.float32(1.2)
                  + np.linspace(0.0, 1e-2, nproj).astype("float32"))[:, None]

data = np.ascontiguousarray(data.astype("float32"))

# Two parity cases over the SAME input, exercising different code paths:
#   snr=3 (tomopy default): large-stripe removal (cols 30/75/100) + sorting.
#                           `_rs_dead`'s val2 gate caps below 3, so the
#                           dead-column bilinear fill does NOT fire (neither
#                           here nor in tomopy).
#   snr=2:                  the same plus the dead-column path — cols 54/55/56
#                           are detected and filled by the kx=ky=1
#                           RectBivariateSpline, exercising that branch.
for snr in (3, 2):
    out = stripe.remove_all_stripe(
        data.copy(), snr=snr, la_size=61, sm_size=21, dim=1, ncore=1,
    ).astype("float32")
    np.save(os.path.join(OUT, f"tomopy_stripe_voall_snr{snr}.npy"), out)
    print(f"snr={snr}: max|Δ| from input = {float(np.max(np.abs(out - data))):.6g}")

np.save(os.path.join(OUT, "stripe_voall_input.npy"), data)
print("input", data.shape)
