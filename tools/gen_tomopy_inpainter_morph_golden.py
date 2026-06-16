#!/usr/bin/env python
"""Golden for `inpainter_morph` (tomopy `misc/corr.py:996`, C
`libtomo/misc/inpainter.c` `Inpainter_morph_main`).

Morphological inpainter over a boolean mask. Only the deterministic modes are
captured here: ``inpainting_type='mean'`` (Gaussian-distance-weighted mean,
`eucl_weighting_inpainting`) and ``'median'`` (median order statistic with the C
buffer-sort quirks). ``'random'`` is intentionally excluded — it draws from C
`rand()` inside an OpenMP-parallel loop, so tomopy's own output is not
reproducible run-to-run (verified: two runs differ), making a golden meaningless.

`exp`/`powf` (the Gaussian weights) match macOS libm bit-for-bit and every
accumulation is fixed-order f32, so the Rust port reproduces these cases
**bit-exactly (delta=0)**. Cases cover the 3-D kernel (`axis=None`) and the
2-D-per-slice kernel (`axis=0`), even/odd dims, and `iterations` 0/1/2.

The mask is saved as uint8 (0/1) for `ndarray-npy` compatibility.

Run with the tomopy env (tomopy 1.15.3, numpy 2.4.6):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_inpainter_morph_golden.py
"""
import os

import numpy as np
from tomopy.misc.corr import inpainter_morph

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260616)
# (d0, d1, d2) with mixed even/odd dims; the masked block is interior so it is
# fully surrounded by non-empty data and completes.
shape = (5, 6, 7)
base = (1.0 + 0.3 * rng.standard_normal(shape)).astype("float32")
mask = np.zeros(shape, dtype=bool)
mask[1:4, 2:4, 2:5] = True   # interior block

# type_code: 0 = mean, 1 = median.  axis_code: -1 = None, else the axis int.
cases = [
    ("mean",   2, 2, None),
    ("median", 2, 2, None),
    ("mean",   3, 0, None),
    ("median", 2, 0, None),
    ("mean",   2, 1, 0),
    ("median", 3, 0, 0),
]

inputs, masks, outputs, params = [], [], [], []
for ty, size, iters, axis in cases:
    out = inpainter_morph(base.copy(), mask.copy(), size=size,
                          iterations=iters, inpainting_type=ty, axis=axis)
    out = np.asarray(out, dtype="float32")
    inputs.append(base.astype("float32"))
    masks.append(mask.astype("uint8"))
    outputs.append(out)
    type_code = 0.0 if ty == "mean" else 1.0
    axis_code = -1.0 if axis is None else float(axis)
    params.append((float(size), float(iters), type_code, axis_code))
    n_masked = int(mask.sum())
    n_changed = int(np.count_nonzero(out != base))
    print(f"{ty:6s} size={size} iter={iters} axis={axis}: "
          f"masked={n_masked} changed={n_changed} "
          f"out range [{float(out.min()):.4f}, {float(out.max()):.4f}]")

np.save(os.path.join(OUT, "inpainter_morph_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "inpainter_morph_mask.npy"),
        np.ascontiguousarray(np.stack(masks)))
np.save(os.path.join(OUT, "inpainter_morph_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "inpainter_morph_params.npy"),
        np.asarray(params, dtype="float64"))
print("cases", len(cases), "shape", shape)
