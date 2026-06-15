#!/usr/bin/env python
"""Generate tomopy golden data for tomoxide's parity tests.

Runs under the `tomopy-golden` micromamba env (tomopy is conda-forge only):

    micromamba run -n tomopy-golden python tools/gen_tomopy_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - phantom.npy          (N, N)        Shepp-Logan phantom
  - angles.npy           (nang,)       projection angles, radians, float32
  - sino.npy             (nang, 1, N)  forward projection (tomopy proj order)
  - tomopy_fbp.npy       (N, N)        tomopy FBP reconstruction of sino
  - tomopy_gridrec.npy   (N, N)        tomopy gridrec reconstruction of sino

The Rust parity test (crates/tomoxide/tests/tomopy_parity.rs) reconstructs the
SAME sino and asserts it matches tomopy's reconstruction. Golden is checked in,
so the test runs offline; only regeneration needs tomopy.
"""
import os
import multiprocessing as mp

# tomopy's distribute_jobs() always spins up an mp.Manager(); under macOS the
# default "spawn" start method dies with EOFError in this headless run. fork
# keeps the manager in-process and works for pure-compute jobs.
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import numpy as np
import tomopy

N = 128
NANG = 180

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")
os.makedirs(out, exist_ok=True)

ang = tomopy.angles(NANG)  # 0..pi, radians
phantom = tomopy.shepp2d(size=N)  # (1, N, N)
# Forward project; pad=False keeps the detector width == N. ncore=1 avoids
# tomopy's multiprocessing Manager (it fails under `micromamba run`).
sino = tomopy.project(phantom, ang, pad=False, ncore=1)  # (NANG, 1, ndet)
center = sino.shape[2] / 2.0

# Pin the FBP filter to a pure ramp so it compares against tomoxide's default
# ramp (tomopy's own default is 'shepp'). gridrec uses its built-in density
# compensation, so its filter_name is left at tomopy's default.
rec_fbp = tomopy.recon(
    sino, ang, center=center, algorithm="fbp", filter_name="ramp", ncore=1
)
rec_gridrec = tomopy.recon(sino, ang, center=center, algorithm="gridrec", ncore=1)

print("phantom", phantom.shape, "sino", sino.shape, "center", center)
print("rec_fbp", rec_fbp.shape, "rec_gridrec", rec_gridrec.shape)

np.save(os.path.join(out, "phantom.npy"), phantom[0].astype("float32"))
np.save(os.path.join(out, "angles.npy"), ang.astype("float32"))
np.save(os.path.join(out, "sino.npy"), sino.astype("float32"))
np.save(os.path.join(out, "tomopy_fbp.npy"), rec_fbp[0].astype("float32"))
np.save(os.path.join(out, "tomopy_gridrec.npy"), rec_gridrec[0].astype("float32"))
print("wrote fixtures to", os.path.normpath(out))
