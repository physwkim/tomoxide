#!/usr/bin/env python
"""Golden for `remove_outlier1d` (tomopy `misc/corr.py:615`).

Removes bright outliers (zingers) with a 1-D median filter along `axis`
(scipy.ndimage.median_filter, mode='mirror'), replacing a pixel by the local
median only where ``arr - median >= dif``. The median filter selects a single
order statistic (rank size//2, never an average), so the result is bit-exact
(delta=0) vs tomopy on finite input.

NOTE: tomopy 1.15.3's public ``remove_outlier1d`` raises ``IndexError`` on
numpy 2.x — corr.py:660 indexes ``arr[slc]`` with a *list* of slices, which
modern numpy rejects (the sibling ``remove_outlier`` at corr.py:600 already
uses the correct ``arr[tuple(slc)]``). We therefore inline tomopy's exact body
below with the single one-character compat fix (``arr[slc]`` ->
``arr[tuple(slc)]``), so the golden is tomopy's own code path — same chunking,
dtype casts, scipy.ndimage call, and ``ne.evaluate`` ``where`` — unbroken for
numpy 2.x.

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6, scipy 1.17.1):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_remove_outlier1d_golden.py
"""
import concurrent.futures as cf
import os

import numexpr as ne
import numpy as np
import scipy.ndimage
from tomopy.util import dtype, mproc

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)


def remove_outlier1d(arr, dif, size=3, axis=0, ncore=None, out=None):
    """Verbatim copy of tomopy `misc/corr.py:615` with the numpy-2.x compat
    fix `arr[slc]` -> `arr[tuple(slc)]` (lines 660, 662)."""
    arr = dtype.as_float32(arr)
    dif = np.float32(dif)

    tmp = np.empty_like(arr)

    other_axes = [i for i in range(arr.ndim) if i != axis]
    largest = np.argmax([arr.shape[i] for i in other_axes])
    lar_axis = other_axes[largest]
    ncore, chnk_slices = mproc.get_ncore_slices(arr.shape[lar_axis], ncore=ncore)
    filt_size = [1] * arr.ndim
    filt_size[axis] = size

    with cf.ThreadPoolExecutor(ncore) as e:
        slc = [slice(None)] * arr.ndim
        for i in range(ncore):
            slc[lar_axis] = chnk_slices[i]
            e.submit(
                scipy.ndimage.median_filter,
                arr[tuple(slc)],
                size=filt_size,
                output=tmp[tuple(slc)],
                mode="mirror",
            )

    with mproc.set_numexpr_threads(ncore):
        out = ne.evaluate("where(arr-tmp>=dif,tmp,arr)", out=out)

    return out


rng = np.random.default_rng(20260616)
n0, n1, n2 = 9, 7, 11
base = (1.0 + 0.3 * rng.standard_normal((n0, n1, n2))).astype("float32")
# Inject bright spikes so the dezinger has real work along every axis.
spikes = rng.integers(0, [n0, n1, n2], size=(20, 3))
for z, y, x in spikes:
    base[z, y, x] += np.float32(5.0)
base = base.astype("float32")

# (dif, size, axis): odd & even sizes, all three axes, a near-zero threshold.
cases = [
    (0.5, 3, 0),
    (0.5, 3, 1),
    (0.5, 3, 2),
    (1.0, 5, 0),
    (0.2, 4, 2),   # even size -> single order statistic, not a mean
    (0.0, 3, 1),   # dif=0: replace wherever arr >= median
]

inputs, outputs, params = [], [], []
for dif, size, axis in cases:
    inp = base.copy()
    out = remove_outlier1d(base.copy(), dif, size=size, axis=axis)
    inputs.append(inp.astype("float32"))
    outputs.append(np.asarray(out, dtype="float32"))
    params.append((float(dif), float(size), float(axis)))
    nchanged = int(np.count_nonzero(inp != np.asarray(out, dtype="float32")))
    print(f"dif={dif} size={size} axis={axis}: changed {nchanged} px, "
          f"out dtype={np.asarray(out).dtype}")

np.save(os.path.join(OUT, "remove_outlier1d_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "remove_outlier1d_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "remove_outlier1d_params.npy"),
        np.asarray(params, dtype="float64"))
print("cases", len(cases), "shape", (n0, n1, n2))
