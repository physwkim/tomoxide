#!/usr/bin/env python
"""Generate tomopy golden data for the ring-removal parity test.

Runs tomopy 1.15.3 `remove_ring` (int_mode='WRAP') on a fixed reconstructed
stack `[slice, y, x]` containing a disk phantom plus an injected concentric
ring artifact. tomoxide processes the SAME array offline and must match within
the f32/f64-rounding floor (`remove_ring` operates on the reconstructed image —
projector-independent — with nearest-pixel polar transforms and median/mean
filters whose trig and sqrt go through the same libm).

Run with the tomopy-enabled env:
    /Users/stevek/mamba/envs/tomopy-golden/bin/python \
        tools/gen_tomopy_ring_golden.py
"""
import os

import numpy as np
import tomopy.misc.corr as corr

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

# Reconstructed-domain stack [slice, y, x]: a smooth disk phantom plus a bright
# concentric ring (the artifact remove_ring is meant to suppress).
nslice, ny, nx = 3, 80, 80
cy = (ny - 1) / 2.0
cx = (nx - 1) / 2.0
yy, xx = np.mgrid[0:ny, 0:nx]
radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

disk = np.where(radius < 28.0, 1.0, 0.0).astype("float32")
disk += 0.3 * np.cos(radius / 4.0).astype("float32")     # some radial structure
ring = (0.5 * np.exp(-((radius - 18.0) ** 2) / 2.0)).astype("float32")  # ring at r≈18

base = (disk + ring).astype("float32")
# Per-slice variation so the slices are not identical.
rec = np.empty((nslice, ny, nx), dtype="float32")
for s in range(nslice):
    rec[s] = (base * (1.0 + 0.1 * s)).astype("float32")
rec = np.ascontiguousarray(rec)

# center_x/center_y default to (nx-1)/2, (ny-1)/2 inside tomopy; pass them
# explicitly so the Rust side uses identical centers.
out2 = corr.remove_ring(
    rec.copy(), center_x=cx, center_y=cy,
    thresh=300.0, thresh_max=300.0, thresh_min=-100.0,
    theta_min=30, rwidth=2, int_mode="WRAP", ncore=1,
).astype("float32")
out4 = corr.remove_ring(
    rec.copy(), center_x=cx, center_y=cy,
    thresh=300.0, thresh_max=300.0, thresh_min=-100.0,
    theta_min=30, rwidth=4, int_mode="WRAP", ncore=1,
).astype("float32")

np.save(os.path.join(OUT, "ring_input.npy"), rec)
np.save(os.path.join(OUT, "tomopy_ring_rw2.npy"), out2)
np.save(os.path.join(OUT, "tomopy_ring_rw4.npy"), out4)

print("input", rec.shape, "center", (cx, cy))
print("rw=2 max|Δ| from input:", float(np.max(np.abs(out2 - rec))))
print("rw=4 max|Δ| from input:", float(np.max(np.abs(out4 - rec))))
