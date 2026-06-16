#!/usr/bin/env python
"""Golden centers for `find_center_pc` with a non-trivial `rotc_guess`.

`find_center_pc(proj1, proj2, tol, rotc_guess)` (tomopy `rotation.py:391`) shifts
both projections by `[0, -imgshift]` with `imgshift = rotc_guess - (ncol-1)/2`
via `scipy.ndimage.shift(order=3, mode='constant', cval=0)` before the phase
correlation, then adds `imgshift` back to the recovered center. The default
`rotc_guess=None` path (imgshift == 0) is covered by
`gen_tomopy_center_pc_golden.py`; this script exercises the spline pre-alignment
with `imgshift != 0` (fractional and integer, both signs).

This is the end-to-end parity for tomoxide's `find_center_pc` rotc_guess path:
the only new component vs the None path is the cubic-spline `ndimage.shift`
(isolated golden: `gen_scipy_ndimage_shift_golden.py`).

Run with the tomopy-enabled env (tomopy 1.15.3, scipy 1.17.1):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_center_pc_rotc_golden.py
"""
import os

import numpy as np
from tomopy.recon.rotation import find_center_pc

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260616)
nrow, ncol = 48, 160
yy = np.linspace(0.0, 1.0, nrow)[:, None]
xx = np.linspace(0.0, 1.0, ncol)[None, :]

img = (0.6 * np.sin(6.0 * xx + 1.3 * yy)
       + 0.4 * np.cos(11.0 * xx - 2.0 * yy)
       + 0.5 * np.exp(-(((xx - 0.34) / 0.07) ** 2 + ((yy - 0.6) / 0.2) ** 2)))
img = img.astype("float32")


def fourier_shift_cols(a, d):
    f = np.fft.fft(a, axis=1)
    k = np.fft.fftfreq(a.shape[1])[None, :]
    f = f * np.exp(-2j * np.pi * k * d)
    return np.real(np.fft.ifft(f, axis=1)).astype("float32")


# (column shift baked into proj180, tol, rotc_guess). cen_fliplr = (ncol-1)/2 =
# 79.5, so imgshift = rotc_guess - 79.5 spans fractional (+3.5, +2.2), integer
# (-3.0) and negative.
cases = [
    (3.0, 0.5, 83.0),    # imgshift = +3.5 (fractional)
    (4.5, 0.5, 76.5),    # imgshift = -3.0 (integer)
    (-6.0, 0.5, 81.7),   # imgshift = +2.2 (fractional)
    (2.4, 0.5, 74.8),    # imgshift = -4.7 (fractional)
]

proj0_list, proj180_list, centers, guesses, tols = [], [], [], [], []
for shift_cols, tol, rotc_guess in cases:
    n0 = (5e-4 * rng.standard_normal((nrow, ncol))).astype("float32")
    n1 = (5e-4 * rng.standard_normal((nrow, ncol))).astype("float32")
    proj0 = (img + n0).astype("float32")
    shifted = fourier_shift_cols(img, shift_cols)
    proj180 = np.fliplr(shifted + n1).astype("float32")
    c = float(find_center_pc(proj0, proj180, tol=tol, rotc_guess=rotc_guess))
    proj0_list.append(proj0)
    proj180_list.append(proj180)
    centers.append(c)
    guesses.append(rotc_guess)
    tols.append(tol)
    print(f"shift={shift_cols:+.1f} tol={tol} rotc_guess={rotc_guess} "
          f"(imgshift={rotc_guess - (ncol - 1.0) / 2.0:+.1f}): center = {c}")

np.save(os.path.join(OUT, "center_pc_rotc_proj0.npy"),
        np.ascontiguousarray(np.stack(proj0_list)))
np.save(os.path.join(OUT, "center_pc_rotc_proj180.npy"),
        np.ascontiguousarray(np.stack(proj180_list)))
np.save(os.path.join(OUT, "center_pc_rotc_centers.npy"),
        np.asarray(centers, dtype="float64"))
np.save(os.path.join(OUT, "center_pc_rotc_guess.npy"),
        np.asarray(guesses, dtype="float64"))
np.save(os.path.join(OUT, "center_pc_rotc_tols.npy"),
        np.asarray(tols, dtype="float64"))
print("cases", len(centers), "shape", (nrow, ncol))
