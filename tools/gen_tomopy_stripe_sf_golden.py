#!/usr/bin/env python
"""Generate tomopy golden data for the smoothing-filter stripe-removal parity test.

Runs tomopy 1.15.3 `remove_stripe_sf` on a fixed 3-D projection stack
(`[angle, row, col]`) seeded with vertical stripes (constant per-column
offsets that survive across all angles — the classic ring-artifact source).
tomoxide processes the SAME array offline and must match bit-for-bit
(`remove_stripe_sf` is pure per-column f32 arithmetic — no projector, no FFT).

Run with the tomopy-enabled env:
    /Users/stevek/mamba/envs/tomopy-golden/bin/python \
        tools/gen_tomopy_stripe_sf_golden.py
"""
import multiprocessing as mp
import os

mp.set_start_method("fork", force=True)  # tomopy distribute_jobs spawns a Manager

import numpy as np
import tomopy.prep.stripe as stripe

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

# Projection stack [angle, row, col] with smooth structure plus injected
# per-column stripes (same offset at every angle/row → a vertical stripe).
rng = np.random.default_rng(20260615)
nang, nrow, ncol = 16, 8, 40
ang = np.linspace(0.0, np.pi, nang, endpoint=False)[:, None, None]
row = np.linspace(0.0, 1.0, nrow)[None, :, None]
col = np.linspace(0.0, 1.0, ncol)[None, None, :]
base = (0.6 + 0.25 * np.sin(4.0 * col + ang) + 0.15 * row).astype("float32")
data = np.ascontiguousarray(np.broadcast_to(base, (nang, nrow, ncol)).astype("float32"))

# Stripes: a random constant offset per column, identical across all angles.
stripe_offset = (0.3 * rng.standard_normal(ncol)).astype("float32")
data = (data + stripe_offset[None, None, :]).astype("float32")
data = np.ascontiguousarray(data)

out3 = stripe.remove_stripe_sf(data.copy(), size=3, ncore=1).astype("float32")
out5 = stripe.remove_stripe_sf(data.copy(), size=5, ncore=1).astype("float32")

np.save(os.path.join(OUT, "stripe_sf_input.npy"), data)
np.save(os.path.join(OUT, "tomopy_stripe_sf3.npy"), out3)
np.save(os.path.join(OUT, "tomopy_stripe_sf5.npy"), out5)

print("input", data.shape)
print("sf size=3 max|Δ| from input:", float(np.max(np.abs(out3 - data))))
print("sf size=5 max|Δ| from input:", float(np.max(np.abs(out5 - data))))
