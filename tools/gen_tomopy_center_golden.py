#!/usr/bin/env python
"""Generate tomopy golden data for tomoxide's center-finding parity test.

Runs under the `tomopy-golden` env (tomopy is conda-forge only):

    /Users/stevek/mamba/envs/tomopy-golden/bin/python tools/gen_tomopy_center_golden.py

Reads the checked-in sinogram (crates/tomoxide/tests/fixtures/sino.npy, written
by gen_tomopy_golden.py) and runs tomopy's `find_center_vo` on it across a few
parameter sets and on an off-center (left-padded) variant. Writes, under
crates/tomoxide/tests/fixtures/:
  - center_sino_pad.npy   (nang, 1, ncol+PAD)  left-padded off-center sinogram
  - tomopy_center_vo.npy  (ncases,)            tomopy find_center_vo per case

find_center_vo is a sinogram-domain Fourier method (projector-independent), so
the Rust port is held to true parity, not just self-consistency. The golden is
checked in, so the test runs offline; only regeneration needs tomopy.

Case order (kept in sync with crates/tomoxide/tests/center_parity.rs):
  [0] base sino,      default params
  [1] base sino,      ratio=0.7, drop=10
  [2] base sino,      smin=-30, smax=30
  [3] left-pad-8 sino default params
"""
import os
import multiprocessing as mp

# tomopy's distribute_jobs() spins up an mp.Manager(); "fork" keeps it working
# in this headless run (see gen_tomopy_golden.py).
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import numpy as np
from tomopy.recon.rotation import find_center_vo

PAD = 8

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")

sino = np.load(os.path.join(out, "sino.npy")).astype("float32")  # (nang, 1, ncol)
sino_pad = np.pad(sino, ((0, 0), (0, 0), (PAD, 0)), mode="edge").astype("float32")

cases = [
    float(find_center_vo(sino, ncore=1)),
    float(find_center_vo(sino, ratio=0.7, drop=10, ncore=1)),
    float(find_center_vo(sino, smin=-30, smax=30, ncore=1)),
    float(find_center_vo(sino_pad, ncore=1)),
]
golden = np.asarray(cases, dtype="float32")

np.save(os.path.join(out, "center_sino_pad.npy"), sino_pad)
np.save(os.path.join(out, "tomopy_center_vo.npy"), golden)
print("base ncol", sino.shape[2], "pad ncol", sino_pad.shape[2])
print("golden find_center_vo:", golden.tolist())
print("wrote fixtures to", os.path.normpath(out))
