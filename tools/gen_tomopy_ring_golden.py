#!/usr/bin/env python
"""Generate tomopy golden data for the ring-removal parity test.

Runs tomopy 1.15.3 `remove_ring` (both int_mode='WRAP' and 'REFLECT') on a fixed
reconstructed stack `[slice, y, x]` containing a disk phantom plus an injected
concentric ring artifact. tomoxide processes the SAME array offline and must
match within the f32/f64-rounding floor (`remove_ring` operates on the
reconstructed image — projector-independent — with nearest-pixel polar
transforms and median/mean filters whose trig and sqrt go through the same
libm). WRAP wraps the azimuthal mean at 0/2π; REFLECT mirrors each polar half.

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
def ring(rwidth, int_mode):
    return corr.remove_ring(
        rec.copy(), center_x=cx, center_y=cy,
        thresh=300.0, thresh_max=300.0, thresh_min=-100.0,
        theta_min=30, rwidth=rwidth, int_mode=int_mode, ncore=1,
    ).astype("float32")


out2 = ring(2, "WRAP")
out4 = ring(4, "WRAP")
ref2 = ring(2, "REFLECT")
ref4 = ring(4, "REFLECT")

np.save(os.path.join(OUT, "ring_input.npy"), rec)
np.save(os.path.join(OUT, "tomopy_ring_rw2.npy"), out2)
np.save(os.path.join(OUT, "tomopy_ring_rw4.npy"), out4)
np.save(os.path.join(OUT, "tomopy_ring_reflect_rw2.npy"), ref2)
np.save(os.path.join(OUT, "tomopy_ring_reflect_rw4.npy"), ref4)

print("input", rec.shape, "center", (cx, cy))
print("WRAP    rw=2 max|Δ| from input:", float(np.max(np.abs(out2 - rec))))
print("WRAP    rw=4 max|Δ| from input:", float(np.max(np.abs(out4 - rec))))
print("REFLECT rw=2 max|Δ| from input:", float(np.max(np.abs(ref2 - rec))))
print("REFLECT rw=4 max|Δ| from input:", float(np.max(np.abs(ref4 - rec))))
print("WRAP vs REFLECT rw=2 max|Δ|:", float(np.max(np.abs(out2 - ref2))))
