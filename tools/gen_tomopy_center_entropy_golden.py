#!/usr/bin/env python
"""Generate tomopy golden data for tomoxide's entropy center-finder parity test.

Runs under the `tomopy-golden` env (tomopy is conda-forge only):

    /Users/stevek/mamba/envs/tomopy-golden/bin/python tools/gen_tomopy_center_entropy_golden.py

Reads the checked-in sinogram (crates/tomoxide/tests/fixtures/sino.npy, written
by gen_tomopy_golden.py) and its angles, then runs tomopy's `find_center`
(entropy + Nelder-Mead) on the base sinogram and on a left-padded off-center
variant. Writes, under crates/tomoxide/tests/fixtures/:
  - center_fc_pad.npy      (nang, 1, ncol+PAD)  left-padded off-center sinogram
  - tomopy_center_fc.npy   (ncases,)            tomopy find_center per case
  - center_fc_true.npy     (ncases,)            tomopy find_center_vo (true axis)

Unlike find_center_vo / find_center_pc, find_center reconstructs (gridrec) at
each candidate center and minimises reconstruction entropy, so it goes THROUGH
the projector and inherits the linear-interp-vs-Siddon gridrec gap (see
PORTING). The Rust port is therefore held to ±~1 px of this golden, not bit
parity. find_center_vo (projector-independent, accurate) is printed as the
"true center" reference for each case.

Case order (kept in sync with crates/tomoxide/tests/center_entropy_parity.rs):
  [0] base sino       (true center ~ ncol/2)
  [1] left-pad-8 sino (true center ~ ncol/2 + 8)
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
from tomopy.recon.rotation import find_center, find_center_vo

PAD = 8

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")

sino = np.load(os.path.join(out, "sino.npy")).astype("float32")  # (nang, 1, ncol)
ang = np.load(os.path.join(out, "angles.npy")).astype("float32")  # (nang,)
sino_pad = np.pad(sino, ((0, 0), (0, 0), (PAD, 0)), mode="edge").astype("float32")

cases = [("base", sino), ("pad8", sino_pad)]
golden, true = [], []
for name, s in cases:
    fc = float(find_center(s, ang, tol=0.5)[0])
    vo = float(find_center_vo(s, ncore=1))  # accurate, projector-independent
    golden.append(fc)
    true.append(vo)
    print(f"{name:5s}: ncol={s.shape[2]:3d}  find_center={fc:.4f}  "
          f"find_center_vo(true)={vo:.4f}")

golden = np.asarray(golden, dtype="float32")
true = np.asarray(true, dtype="float32")
np.save(os.path.join(out, "center_fc_pad.npy"), sino_pad)
np.save(os.path.join(out, "tomopy_center_fc.npy"), golden)
np.save(os.path.join(out, "center_fc_true.npy"), true)
print("golden find_center:", golden.tolist())
print("true   find_center_vo:", true.tolist())
print("wrote fixtures to", os.path.normpath(out))
