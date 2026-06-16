#!/usr/bin/env python
"""Golden for `median_filter` (tomopy `misc/corr.py:167`).

Median-filters every 2-D slice along `axis` with a size×size footprint via
scipy.ndimage.median_filter (default mode='reflect', half-sample reflection).
Every pixel is replaced by its local median — there is no threshold. The median
filter selects a single order statistic (rank size*size//2, never an average),
so the result is bit-exact (delta=0) vs tomopy on finite input.

This wrapper uses `arr[tuple(slc)]`, so it runs unmodified on numpy 2.x
(unlike the sibling `remove_outlier1d`).

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6, scipy 1.17.1):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_median_filter_golden.py
"""
import os

import numpy as np
from tomopy.misc.corr import median_filter

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260616)
n0, n1, n2 = 6, 9, 13
base = (1.0 + 0.4 * rng.standard_normal((n0, n1, n2))).astype("float32")
# Inject bright/dark spikes so the median actually moves pixels.
spikes = rng.integers(0, [n0, n1, n2], size=(25, 3))
for j, (z, y, x) in enumerate(spikes):
    base[z, y, x] += np.float32(6.0 if j % 2 == 0 else -6.0)
base = base.astype("float32")

# (size, axis): odd & even footprints, all three axes.
cases = [
    (3, 0),
    (3, 1),
    (3, 2),
    (5, 0),
    (4, 2),   # even footprint -> single order statistic, not a mean
    (2, 1),   # smallest even footprint
]

inputs, outputs, params = [], [], []
for size, axis in cases:
    inp = base.copy()
    out = median_filter(base.copy(), size=size, axis=axis)
    inputs.append(inp.astype("float32"))
    outputs.append(np.asarray(out, dtype="float32"))
    params.append((float(size), float(axis)))
    nchanged = int(np.count_nonzero(inp != np.asarray(out, dtype="float32")))
    print(f"size={size} axis={axis}: changed {nchanged} px, "
          f"out dtype={np.asarray(out).dtype}")

np.save(os.path.join(OUT, "median_filter_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "median_filter_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "median_filter_params.npy"),
        np.asarray(params, dtype="float64"))
print("cases", len(cases), "shape", (n0, n1, n2))
