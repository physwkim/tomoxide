#!/usr/bin/env python3
"""Build a multi-row DXchange HDF5 fixture for the out-of-core streaming test.

`ReconSteps::run_streaming` reads only each chunk's detector rows via
`DatasetReader::read_chunk` (an HDF5 hyperslab), so a fixture with several
detector rows (nz>1) is needed to exercise chunking. We reuse the committed
phantom sinogram (`sino.npy`, nz=1) and tile it into nz rows with a small
per-row gain so the rows differ, then turn it into a raw intensity acquisition
exactly like the M3 pipeline fixture:

  transmission T = exp(-scale * sino_row)     in (0, 1]
  data           = dark + (flat - dark) * T   raw intensity, [nproj, nz, nx]
  data_white     = flat = 1000;  data_dark = dark = 10
  theta          = angles in DEGREES

The streaming and full (`read_all`) paths must reconstruct this identically.

Run:  micromamba run -n tomo python tools/gen_dxchange_streaming_fixture.py
"""
import os
import numpy as np
import h5py

FIX = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")

sino = np.load(os.path.join(FIX, "sino.npy"))        # (nproj, 1, nx) float32
angles = np.load(os.path.join(FIX, "angles.npy"))    # (nproj,) radians
nproj, _, nx = sino.shape
NZ = 6

PEAK_ATTEN = 6.0
FLAT0, DARK0 = 1000.0, 10.0
scale = PEAK_ATTEN / float(sino.max())

# Tile to NZ rows with a per-row gain so the slices are not identical.
base = sino[:, 0, :]                                   # (nproj, nx)
rows = np.empty((nproj, NZ, nx), dtype=np.float64)
for z in range(NZ):
    rows[:, z, :] = base * (1.0 + 0.05 * z)

trans = np.exp(-scale * rows)                          # (0, 1]
data = (DARK0 + (FLAT0 - DARK0) * trans).astype(np.float32)
white = np.full((2, NZ, nx), FLAT0, dtype=np.float32)
dark = np.full((2, NZ, nx), DARK0, dtype=np.float32)
theta_deg = (angles * 180.0 / np.pi).astype(np.float32)

out = os.path.join(FIX, "streaming_dxchange.h5")
with h5py.File(out, "w") as f:
    g = f.create_group("exchange")
    g.create_dataset("data", data=data, compression="gzip", chunks=True)
    g.create_dataset("data_white", data=white, compression="gzip", chunks=True)
    g.create_dataset("data_dark", data=dark, compression="gzip", chunks=True)
    g.create_dataset("theta", data=theta_deg)

print(f"wrote {out}: data {data.shape} {data.dtype}, nz={NZ}")
