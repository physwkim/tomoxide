#!/usr/bin/env python
"""Generate tomopy golden data for tomoxide's Paganin phase-retrieval test.

Runs under the `tomopy-golden` env:

    /Users/stevek/mamba/envs/tomopy-golden/bin/python tools/gen_tomopy_phase_golden.py

retrieve_phase is a Fourier low-pass on each radiograph (projector-independent),
so the Rust port is held to tomopy parity. A small deterministic radiograph
stack (a few Gaussian blobs per frame, edge-asymmetric so the pad value matters)
is filtered with tomopy 1.15.3 `retrieve_phase` at default params. Writes, under
crates/tomoxide/tests/fixtures/:
  - phase_input.npy   (nproj, dy, dz)  input radiograph stack, float32
  - tomopy_phase.npy  (nproj, dy, dz)  tomopy retrieve_phase output, float32

The golden is checked in, so the test runs offline; regeneration needs tomopy.
Params are tomopy defaults: pixel_size=1e-4, dist=50, energy=20, alpha=1e-3.
"""
import os
import multiprocessing as mp

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import numpy as np
import tomopy

NPROJ, DY, DZ = 6, 48, 64

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")

# Deterministic radiograph stack: a moving Gaussian blob plus a baseline ramp,
# asymmetric left/right so the edge-pad value is exercised.
yy, xx = np.mgrid[0:DY, 0:DZ].astype("float32")
stack = np.empty((NPROJ, DY, DZ), dtype="float32")
for m in range(NPROJ):
    cy = DY * (0.3 + 0.4 * m / NPROJ)
    cx = DZ * (0.25 + 0.5 * m / NPROJ)
    blob = np.exp(-(((yy - cy) ** 2) / (2 * 7.0**2) + ((xx - cx) ** 2) / (2 * 9.0**2)))
    ramp = 0.2 + 0.1 * (xx / DZ)  # left/right asymmetry
    stack[m] = (ramp + blob).astype("float32")

phase = tomopy.retrieve_phase(
    stack.copy(), pixel_size=1e-4, dist=50, energy=20, alpha=1e-3, ncore=1
).astype("float32")

np.save(os.path.join(out, "phase_input.npy"), stack)
np.save(os.path.join(out, "tomopy_phase.npy"), phase)
print("input", stack.shape, "output", phase.shape)
print("input range", float(stack.min()), float(stack.max()))
print("phase range", float(phase.min()), float(phase.max()))
print("wrote fixtures to", os.path.normpath(out))
