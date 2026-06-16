#!/usr/bin/env python
"""Golden for `normalize_roi` (tomopy `prep/normalize.py:168`).

For each projection the mean `bg` of `proj[r0:r2, r1:r3]` is computed and, if
`bg != 0`, the whole projection is divided by it. The ROI mean uses numpy's f32
pairwise summation; reproducing that accumulation tree makes `bg` — and the
elementwise f32 divide — bit-exact (delta=0).

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_normalize_roi_golden.py
"""
import os

import numpy as np
from tomopy.prep.normalize import _normalize_roi
from tomopy.util import dtype

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)


def normalize_roi(arr, roi):
    """tomopy's normalize_roi body applied in-process, bypassing the flaky
    macOS multiprocessing pool in `mproc.distribute_jobs`. `_normalize_roi` is
    tomopy's verbatim per-projection kernel (corr.py: `bg = proj[roi].mean();
    if bg != 0: proj /= bg`); the distribute_jobs wrapper chunks axis 0 with no
    cross-projection coupling, so applying the kernel per projection is
    numerically identical. `_normalize_roi` is at normalize.py:200."""
    arr = dtype.as_float32(arr)
    for p in range(arr.shape[0]):
        _normalize_roi(arr[p], roi)   # in place
    return arr

rng = np.random.default_rng(20260616)
n0, n1, n2 = 5, 32, 40
base = (100.0 + 20.0 * rng.standard_normal((n0, n1, n2))).astype("float32")

# roi = [r0, r1, r2, r3] = [row_start, col_start, row_end, col_end]: the default
# 10x10, a larger >128-element ROI (recursion path), and an offset non-square ROI.
cases = [
    [0, 0, 10, 10],     # tomopy default (n=100, base case)
    [4, 6, 24, 30],     # 20x24 = 480 (recursion path), offset
    [2, 5, 14, 13],     # 12x8 = 96, offset non-square
]

inputs, outputs, params = [], [], []
for roi in cases:
    inp = base.copy()
    out = normalize_roi(base.copy(), roi=roi)
    inputs.append(inp.astype("float32"))
    outputs.append(np.asarray(out, dtype="float32"))
    params.append([float(v) for v in roi])
    r0, r1, r2, r3 = roi
    print(f"roi={roi} (n={(r2-r0)*(r3-r1)}): "
          f"out range [{float(np.min(out)):.4f}, {float(np.max(out)):.4f}]")

np.save(os.path.join(OUT, "normalize_roi_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "normalize_roi_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "normalize_roi_params.npy"),
        np.asarray(params, dtype="float64"))
print("cases", len(cases), "shape", (n0, n1, n2))
