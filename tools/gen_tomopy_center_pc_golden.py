#!/usr/bin/env python
"""Generate tomopy golden centers for the phase-correlation `find_center_pc`.

`find_center_pc(proj0, proj180, tol)` registers proj0 against the mirrored
proj180 by skimage `phase_cross_correlation` (normalization='phase',
upsample_factor=1/tol) and maps the column shift to a rotation center. It never
touches a projector, so tomoxide can match it exactly: with tol=0.5 the shift is
quantized to half a pixel and the center to a quarter pixel.

Several 2-D projection pairs (distinct row/col sizes, integer and fractional
column offsets) exercise the whole-pixel argmax and the 3x3 upsampled-DFT
subpixel refinement. proj0/proj180 are float32 so scipy runs the FFT in
complex64 (the precision class of tomoxide's Complex32 FFT).

Run with the tomopy-enabled env:
    /Users/stevek/mamba/envs/tomopy-golden/bin/python \
        tools/gen_tomopy_center_pc_golden.py
"""
import os

import numpy as np
from tomopy.recon.rotation import find_center_pc

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260615)
nrow, ncol = 48, 160
yy = np.linspace(0.0, 1.0, nrow)[:, None]
xx = np.linspace(0.0, 1.0, ncol)[None, :]

# Asymmetric smooth structure + a localized blob so the correlation peak is
# sharp and unambiguous (no argmax ties).
img = (0.6 * np.sin(6.0 * xx + 1.3 * yy)
       + 0.4 * np.cos(11.0 * xx - 2.0 * yy)
       + 0.5 * np.exp(-(((xx - 0.34) / 0.07) ** 2 + ((yy - 0.6) / 0.2) ** 2)))
img = img.astype("float32")


def fourier_shift_cols(a, d):
    """Clean cyclic shift of `a` along columns by `d` (a phase ramp in the FFT),
    integer or fractional — exactly what phase correlation recovers."""
    f = np.fft.fft(a, axis=1)
    k = np.fft.fftfreq(a.shape[1])[None, :]
    f = f * np.exp(-2j * np.pi * k * d)
    return np.real(np.fft.ifft(f, axis=1)).astype("float32")


proj0_list, proj180_list, centers, tols = [], [], [], []
for shift_cols, tol in [(3.0, 0.5), (4.5, 0.5), (-6.0, 0.5), (2.4, 0.5)]:
    n0 = (5e-4 * rng.standard_normal((nrow, ncol))).astype("float32")
    n1 = (5e-4 * rng.standard_normal((nrow, ncol))).astype("float32")
    proj0 = (img + n0).astype("float32")
    # proj180 mirrored so that fliplr(proj180) is img cyclically shifted by
    # `shift_cols`; integer cases land on a half-pixel center, fractional cases
    # exercise the 3x3 upsampled-DFT subpixel refinement.
    shifted = fourier_shift_cols(img, shift_cols)
    proj180 = np.fliplr(shifted + n1).astype("float32")
    c = float(find_center_pc(proj0, proj180, tol=tol))
    proj0_list.append(proj0)
    proj180_list.append(proj180)
    centers.append(c)
    tols.append(tol)
    print(f"shift={shift_cols:+.1f} tol={tol}: center = {c}")

np.save(os.path.join(OUT, "center_pc_proj0.npy"),
        np.ascontiguousarray(np.stack(proj0_list)))
np.save(os.path.join(OUT, "center_pc_proj180.npy"),
        np.ascontiguousarray(np.stack(proj180_list)))
np.save(os.path.join(OUT, "center_pc_centers.npy"),
        np.asarray(centers, dtype="float64"))
np.save(os.path.join(OUT, "center_pc_tols.npy"),
        np.asarray(tols, dtype="float64"))
print("cases", len(centers), "shape", (nrow, ncol))
