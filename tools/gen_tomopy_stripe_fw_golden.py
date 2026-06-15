#!/usr/bin/env python
"""Generate tomopy golden data for the Fourier-Wavelet (`remove_stripe_fw`) parity test.

Runs tomopy 1.15.3 `remove_stripe_fw` on a fixed projection stack
`[proj, row, col]` carrying a smooth sinogram plus injected additive stripes.
The Fourier-Wavelet method (Münch 2009) runs a `level`-deep `db5` 2-D wavelet
decomposition per slice (float32 `pywt`), damps the vertical-detail bands along
the projection axis in Fourier space (numpy promotes to complex128/float64),
reconstructs (float64 `pywt`), and casts back to float32. tomoxide reimplements
the same db5 transform (validated against `pywt`) with the matching float32
forward / float64 damp+inverse dtype flow, so this is held to the f32 round-off
floor, not bit-exactness.

Defaults: `wname='db5'`, `sigma=2`, `pad=True`, `level=None`
(auto = `ceil(log2(max(shape)))`). tomoxide's `StripeMethod::Fw` fixes db5 and
always pads, matching these defaults.

Run with the tomopy-enabled env:
    /Users/stevek/mamba/envs/tomopy-golden/bin/python \
        tools/gen_tomopy_stripe_fw_golden.py
"""
import multiprocessing as mp
import os

mp.set_start_method("fork", force=True)  # tomopy distribute_jobs spawns a Manager

import numpy as np
import tomopy.prep.stripe as stripe

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260616)
nproj, nrow, ncol = 180, 2, 128
ang = np.linspace(0.0, np.pi, nproj, endpoint=False)[:, None, None]
col = np.linspace(0.0, 1.0, ncol)[None, None, :]
row = (1.0 + 0.1 * np.arange(nrow))[None, :, None]

base = (1.0
        + 0.4 * np.sin(5.0 * col + 2.0 * ang)
        + 0.2 * np.cos(9.0 * col - ang))
data = (base * row).astype("float32")
data = data + (1e-3 * rng.standard_normal(data.shape)).astype("float32")

# Injected stripes: constant additive offset on a few columns across all angles.
for c, amp in [(30, 0.6), (75, -0.5), (100, 0.45)]:
    data[:, :, c] += np.float32(amp)

data = np.ascontiguousarray(data.astype("float32"))

out = stripe.remove_stripe_fw(
    data.copy(), level=None, wname="db5", sigma=2, pad=True, ncore=1,
).astype("float32")
np.save(os.path.join(OUT, "tomopy_stripe_fw_db5.npy"), out)
print(f"fw db5: max|Δ from input| = {float(np.max(np.abs(out - data))):.6g}")

np.save(os.path.join(OUT, "stripe_fw_input.npy"), data)
print("input", data.shape)
