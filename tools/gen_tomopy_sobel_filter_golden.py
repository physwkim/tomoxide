#!/usr/bin/env python
"""Golden for `sobel_filter` (tomopy `misc/corr.py:474`).

Applies scipy.ndimage.sobel to every 2-D slice along `axis`: a [-1,0,1]
central-difference correlation along the slice's last axis, then a [1,2,1]
smoothing correlation along the other axis (both mode='reflect'). The weights
are exact small integers and f32 inputs are exact in scipy's f64 accumulator, so
the result is bit-exact (delta=0).

tomopy's published `sobel_filter` (1.15.3) cannot run as written: (1) it
references a bare `filters.sobel` but `corr.py` never binds `filters`
(NameError), and (2) it indexes `arr[slc]` with a *list*, which raises
IndexError on numpy 2.x. Both are compat bugs; the intended op is unambiguous
(`scipy.ndimage.filters.sobel is scipy.ndimage.sobel`, default axis=-1, per 2-D
slice). This generator inlines tomopy's verbatim body with exactly those two
one-token fixes (`filters.sobel` -> `scipy.ndimage.sobel`, `arr[slc]` ->
`arr[tuple(slc)]`) — same dtype cast, same per-slice scipy call.

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6, scipy 1.17.1):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_sobel_filter_golden.py
"""
import os

import numpy as np
import scipy.ndimage
from tomopy.util import dtype

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)


def sobel_filter(arr, axis=0):
    """tomopy `sobel_filter` body, applied in-process with the two compat fixes
    (`filters.sobel` -> `scipy.ndimage.sobel`, `arr[slc]` -> `arr[tuple(slc)]`)
    and serially (the ThreadPoolExecutor only parallelises independent slices)."""
    arr = dtype.as_float32(arr)
    out = np.empty_like(arr)
    slc = [slice(None)] * arr.ndim
    for i in range(arr.shape[axis]):
        slc[axis] = i
        scipy.ndimage.sobel(arr[tuple(slc)], output=out[tuple(slc)])
    return out


rng = np.random.default_rng(20260616)
n0, n1, n2 = 6, 11, 15
base = (1.0 + 0.4 * rng.standard_normal((n0, n1, n2))).astype("float32")

cases = [0, 1, 2]   # stacking axis: take 2-D slices along each of the 3 axes

inputs, outputs, params = [], [], []
for axis in cases:
    inp = base.copy()
    out = sobel_filter(base.copy(), axis=axis)
    out = np.asarray(out, dtype="float32")
    inputs.append(inp.astype("float32"))
    outputs.append(out)
    params.append((float(axis),))
    print(f"axis={axis}: out range [{float(out.min()):.4f}, {float(out.max()):.4f}] "
          f"dtype={out.dtype}")

np.save(os.path.join(OUT, "sobel_filter_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "sobel_filter_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "sobel_filter_params.npy"),
        np.asarray(params, dtype="float64"))
print("cases", len(cases), "shape", (n0, n1, n2))
