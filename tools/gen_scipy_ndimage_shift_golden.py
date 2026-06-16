#!/usr/bin/env python
"""Golden for the cubic-spline `scipy.ndimage.shift` port used by
`find_center_pc`'s `rotc_guess` pre-alignment.

`find_center_pc` (tomopy `rotation.py:422-423`) calls
`ndimage.shift(proj, [0, -imgshift], order=3, mode='constant', cval=0)` on each
projection before phase correlation. tomoxide ports that exact transform
(spline prefilter both axes to float64 with mirror-boundary initialisation, then
a 16-tap separable cubic resample; out-of-bounds output centres collapse to
`cval`, in-bounds taps are whole-sample mirror-reflected). This script captures
scipy's own output so the port can be held to Δ ≈ 0 in isolation — a far tighter
check than the quarter-pixel-quantised `find_center_pc` center.

The shift list mixes fractional and integer, both signs, and magnitudes large
enough to push some output centres out of bounds (exercising the cval path) and
others near the edge (exercising the mirror-reflected taps). Row shifts are
included even though `find_center_pc` only shifts columns, to exercise the
axis-0 spline as well.

Run with the tomopy/scipy env (scipy 1.17.1):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_scipy_ndimage_shift_golden.py
"""
import os

import numpy as np
from scipy import ndimage

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide-recon", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

nrow, ncol = 24, 40
yy = np.linspace(0.0, 1.0, nrow)[:, None]
xx = np.linspace(0.0, 1.0, ncol)[None, :]

# Asymmetric smooth structure plus a localized blob so the spline has real
# curvature to reproduce (a flat image would hide prefilter/reconstruct errors).
img = (0.7 * np.sin(5.0 * xx + 1.1 * yy)
       + 0.3 * np.cos(9.0 * xx - 1.7 * yy)
       + 0.6 * np.exp(-(((xx - 0.3) / 0.08) ** 2 + ((yy - 0.55) / 0.18) ** 2)))
img = img.astype("float32")

# (shift_row, shift_col): pure-column (find_center_pc shape), plus 2-D cases.
shifts = [
    (0.0, 2.4),     # fractional column, content moves right
    (0.0, -3.7),    # fractional column, other sign
    (0.0, 0.37),    # sub-pixel column
    (0.0, 3.0),     # integer column (spline at integer coords, not a roll)
    (1.3, -2.6),    # 2-D fractional (axis-0 spline too)
    (-0.8, 5.2),    # large column shift: left edge samples out of bounds -> cval
]

outs = []
for sr, sc in shifts:
    o = ndimage.shift(img, [sr, sc], order=3, mode="constant", cval=0.0)
    assert o.dtype == np.float32, o.dtype
    outs.append(o.astype("float32"))

np.save(os.path.join(OUT, "ndimage_shift_input.npy"),
        np.ascontiguousarray(img))
np.save(os.path.join(OUT, "ndimage_shift_outputs.npy"),
        np.ascontiguousarray(np.stack(outs)))
np.save(os.path.join(OUT, "ndimage_shift_params.npy"),
        np.asarray(shifts, dtype="float64"))
print("cases", len(shifts), "shape", (nrow, ncol))
for (sr, sc), o in zip(shifts, outs):
    print(f"  shift=({sr:+.2f},{sc:+.2f}) "
          f"min={o.min():.5f} max={o.max():.5f}")
