#!/usr/bin/env python
"""Golden for `shepp3d` (tomopy `misc/phantom.py:284`).

A modified 3-D Shepp-Logan phantom: 10 ellipsoids rasterised on
`np.mgrid[-1:1:size*j]` per axis. Each ellipsoid rotates the coords by an Euler
matrix (`np.tensordot`), shifts by its centre, scales by its semi-axes, and a
voxel is inside when `sum(((R r - c)/s)**2) <= 1`; inside voxels accumulate the
amplitude A in float32, and the cube is finally `clip(0, inf)`. All coordinate
math is f64 (libm sin/cos), so the inclusion mask — and hence the f32 cube — is
reproducible bit-for-bit.

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_shepp3d_golden.py
"""
import os

import numpy as np
from tomopy.misc.phantom import shepp3d

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

# Even, odd, and a slightly larger size to exercise the mgrid endpoint and a
# range of boundary voxels. Kept small so the fixtures stay light.
sizes = [16, 17, 32]

for size in sizes:
    vol = np.asarray(shepp3d(size=size), dtype="float32")
    np.save(os.path.join(OUT, f"shepp3d_{size}.npy"), np.ascontiguousarray(vol))
    print(f"size={size}: shape={vol.shape} range=[{float(vol.min()):.4f}, "
          f"{float(vol.max()):.4f}] nonzero={int(np.count_nonzero(vol))}")

print("sizes", sizes)
