#!/usr/bin/env python
"""Golden for `gaussian_filter` (tomopy `misc/corr.py:118`).

Gaussian-filters every 2-D slice along `axis` via scipy.ndimage.gaussian_filter
(default mode='reflect', half-sample reflection; truncate=4.0). The kernel is
`exp(-x^2/2sigma^2)` normalised by numpy's f64 pairwise sum, reversed for
correlation, and the convolution accumulates in f64 (scipy's line buffers are
double) with the intermediate stored in f32 between the two separable passes.
Because the kernel uses `exp` (a transcendental, where numpy's vectorised f64
`exp` and libm differ by <=1 ULP), the port is held to the f32 round-off floor.

This wrapper uses `arr[tuple(slc)]`, so it runs unmodified on numpy 2.x.

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6, scipy 1.17.1):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_gaussian_filter_golden.py
"""
import os

import numpy as np
from tomopy.misc.corr import gaussian_filter

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260616)
n0, n1, n2 = 6, 17, 21
base = (1.0 + 0.4 * rng.standard_normal((n0, n1, n2))).astype("float32")

# (sigma, order, axis): the default sigma=3 order=0 on each axis, a small sigma
# (tiny radius), and the derivative orders 1 and 2 (anti-symmetric / symmetric
# branch of NI_Correlate1D).
cases = [
    (3.0, 0, 0),
    (3.0, 0, 1),
    (3.0, 0, 2),
    (0.8, 0, 0),   # lw = int(4*0.8+0.5) = 3, small kernel
    (2.0, 1, 0),   # first derivative -> anti-symmetric branch
    (2.0, 2, 1),   # second derivative -> symmetric branch
]

inputs, outputs, params = [], [], []
for sigma, order, axis in cases:
    inp = base.copy()
    out = gaussian_filter(base.copy(), sigma=sigma, order=order, axis=axis)
    out = np.asarray(out, dtype="float32")
    inputs.append(inp.astype("float32"))
    outputs.append(out)
    params.append((float(sigma), float(order), float(axis)))
    print(f"sigma={sigma} order={order} axis={axis}: "
          f"out range [{float(out.min()):.4f}, {float(out.max()):.4f}] "
          f"dtype={out.dtype}")

np.save(os.path.join(OUT, "gaussian_filter_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "gaussian_filter_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "gaussian_filter_params.npy"),
        np.asarray(params, dtype="float64"))
print("cases", len(cases), "shape", (n0, n1, n2))
