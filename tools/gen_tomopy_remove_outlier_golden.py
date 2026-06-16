#!/usr/bin/env python
"""Golden for `remove_outlier` (tomopy `misc/corr.py:559`).

The axis-chunked 2-D dezinger: for each index along `axis` the orthogonal 2-D
image's size*size median is taken via scipy.ndimage.median_filter (default
mode='reflect'), then a pixel is replaced by that median only where
``arr - median >= dif``. The median filter selects a single order statistic
(rank size*size//2, never an average), so the result is bit-exact (delta=0) vs
tomopy on finite input.

Distinct from remove_outlier3d (3-D cube) and remove_outlier1d (1-D mirror).
This wrapper uses `arr[tuple(slc)]`, so it runs unmodified on numpy 2.x.

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6, scipy 1.17.1):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_remove_outlier_golden.py
"""
import os

import numpy as np
from tomopy.misc.corr import remove_outlier

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260616)
n0, n1, n2 = 6, 9, 13
base = (1.0 + 0.4 * rng.standard_normal((n0, n1, n2))).astype("float32")
# Inject bright spikes so the dezinger has real work along every chunk axis.
spikes = rng.integers(0, [n0, n1, n2], size=(25, 3))
for z, y, x in spikes:
    base[z, y, x] += np.float32(6.0)
base = base.astype("float32")

# (dif, size, axis): odd & even sizes, all three axes, a near-zero threshold.
cases = [
    (0.5, 3, 0),
    (0.5, 3, 1),
    (0.5, 3, 2),
    (1.0, 5, 0),
    (0.2, 4, 2),   # even footprint -> single order statistic, not a mean
    (0.0, 3, 1),   # dif=0: replace wherever arr >= median
]

inputs, outputs, params = [], [], []
for dif, size, axis in cases:
    inp = base.copy()
    out = remove_outlier(base.copy(), dif, size=size, axis=axis)
    inputs.append(inp.astype("float32"))
    outputs.append(np.asarray(out, dtype="float32"))
    params.append((float(dif), float(size), float(axis)))
    nchanged = int(np.count_nonzero(inp != np.asarray(out, dtype="float32")))
    print(f"dif={dif} size={size} axis={axis}: changed {nchanged} px, "
          f"out dtype={np.asarray(out).dtype}")

np.save(os.path.join(OUT, "remove_outlier_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "remove_outlier_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "remove_outlier_params.npy"),
        np.asarray(params, dtype="float64"))
print("cases", len(cases), "shape", (n0, n1, n2))
