#!/usr/bin/env python3
"""Generate a small DXchange-format HDF5 fixture + sidecar .npy goldens.

Exercises the tomoxide-io reader end to end:
  * uint16 image data (the common raw-detector dtype) -> reader casts to f32,
  * gzip-compressed, chunked storage -> exercises rust-hdf5's deflate path,
  * /exchange/theta in DEGREES -> reader converts to radians (deg/180*pi).

Layout matches tomocupy dataio/reader.py:
  /exchange/data        [nproj, nz, nx]
  /exchange/data_white  [nflat, nz, nx]
  /exchange/data_dark   [ndark, nz, nx]
  /exchange/theta       [nproj]  (degrees)

Run:  python3 tools/gen_dxchange_fixture.py
"""
import os
import numpy as np
import h5py

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide-io", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

NPROJ, NZ, NX = 5, 3, 4
NFLAT, NDARK = 2, 2


def ramp(nframe, base):
    """Distinct, f32-exact values: base + f*100 + z*10 + x (fits in u16)."""
    a = np.empty((nframe, NZ, NX), dtype=np.uint16)
    for f in range(nframe):
        for z in range(NZ):
            for x in range(NX):
                a[f, z, x] = base + f * 100 + z * 10 + x
    return a


data = ramp(NPROJ, 0)
white = ramp(NFLAT, 1000)
dark = ramp(NDARK, 50)
theta_deg = np.linspace(0.0, 180.0, NPROJ, dtype=np.float32)  # [0,45,90,135,180]

h5_path = os.path.join(OUT, "dxchange_small.h5")
with h5py.File(h5_path, "w") as f:
    g = f.create_group("exchange")
    g.create_dataset("data", data=data, compression="gzip", chunks=True)
    g.create_dataset("data_white", data=white, compression="gzip", chunks=True)
    g.create_dataset("data_dark", data=dark, compression="gzip", chunks=True)
    g.create_dataset("theta", data=theta_deg)

# Sidecar goldens (what the Rust reader must reproduce, bit-exact).
np.save(os.path.join(OUT, "dxchange_data_f32.npy"), data.astype(np.float32))
np.save(os.path.join(OUT, "dxchange_white_f32.npy"), white.astype(np.float32))
np.save(os.path.join(OUT, "dxchange_dark_f32.npy"), dark.astype(np.float32))
# read_theta semantics: tomocupy reader.py:314 -> deg.astype(f32)/180*pi.
theta_rad = (theta_deg.astype("float32") / 180 * np.pi).astype(np.float32)
np.save(os.path.join(OUT, "dxchange_theta_rad.npy"), theta_rad)

print(f"wrote {h5_path}")
print(f"  data {data.shape} {data.dtype}, white {white.shape}, dark {dark.shape}")
print(f"  theta_deg {theta_deg.tolist()} -> theta_rad {theta_rad.tolist()}")
