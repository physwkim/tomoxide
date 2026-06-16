#!/usr/bin/env python
"""Golden for `median_filter_nonfinite` (tomopy `misc/corr.py:281`).

Replaces every non-finite value (NaN/±inf) in a 3-D stack with the median of the
finite values in its `size×size` neighbourhood along the last two axes. Per
projection the medians read a snapshot taken before any correction, so adjacent
bad pixels do not influence each other. A kernel with no finite value raises.

Pure NumPy (np.isfinite / np.nonzero / np.median) → bit-exact parity is
achievable. This script also probes np.median's dtype/rounding on a float32 even
count, which decides the even-case arithmetic in the Rust port.

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_median_nonfinite_golden.py
"""
import os

import numpy as np
from tomopy.misc.corr import median_filter_nonfinite

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

# Probe: np.median dtype + even-count rounding for float32 input.
probe = np.array([1.0, 2.0], dtype="float32")
m = np.median(probe)
print(f"np.median(float32[1,2]) = {m!r} dtype={np.asarray(m).dtype}")
# A pair whose float32 mean differs from a float64 mean cast to float32.
a, b = np.float32(0.1), np.float32(0.3)
m2 = np.median(np.array([a, b], dtype="float32"))
f32_mean = np.float32((a + b) / np.float32(2.0))
f64_then_cast = np.float32((float(a) + float(b)) / 2.0)
print(f"even-case: np.median={m2!r} f32mean={f32_mean!r} f64cast={f64_then_cast!r}")

rng = np.random.default_rng(20260616)
n0, n1, n2 = 4, 18, 22
base = (10.0 + 5.0 * np.sin(np.linspace(0, 6, n1)[:, None]
                            + np.linspace(0, 4, n2)[None, :])).astype("float32")

inputs, outputs, sizes = [], [], []
for case, size in enumerate([3, 5]):
    arr = np.broadcast_to(base, (n0, n1, n2)).astype("float32").copy()
    arr += (0.3 * rng.standard_normal((n0, n1, n2))).astype("float32")
    # Scatter NaN and ±inf, but never enough to leave a whole kernel non-finite:
    # ~6% corruption with size>=3 keeps every neighbourhood mostly finite.
    bad = rng.random((n0, n1, n2)) < 0.06
    kinds = rng.integers(0, 3, size=arr.shape)
    arr[bad & (kinds == 0)] = np.nan
    arr[bad & (kinds == 1)] = np.inf
    arr[bad & (kinds == 2)] = -np.inf
    # Guard: ensure the [0,0] corner of each projection stays finite so the
    # smallest (clamped 2×2 / 3×3) corner kernels always have a finite value.
    arr[:, 0, 0] = base[0, 0]

    inp = arr.copy()
    out = median_filter_nonfinite(arr.copy(), size=size)
    assert out.dtype == np.float32, out.dtype
    n_bad = int(np.count_nonzero(~np.isfinite(inp)))
    n_out_bad = int(np.count_nonzero(~np.isfinite(out)))
    assert n_out_bad == 0, n_out_bad
    print(f"case {case}: size={size} corrupted={n_bad} -> remaining non-finite={n_out_bad}")
    inputs.append(inp)
    outputs.append(out)
    sizes.append(size)

np.save(os.path.join(OUT, "median_nonfinite_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "median_nonfinite_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "median_nonfinite_sizes.npy"),
        np.asarray(sizes, dtype="int64"))
print("cases", len(sizes), "shape", (n0, n1, n2))
