#!/usr/bin/env python3
"""Build a realistic DXchange HDF5 fixture for the M3 end-to-end pipeline test.

Reuses the committed phantom sinogram (`sino.npy`, a tomopy parallel-beam
sinogram of `phantom.npy`) and turns it back into a raw-style acquisition.

`phantom.npy` is the Shepp-Logan phantom scaled to 0..255, so its line
integrals reach ~8400 — physically nonsensical as an attenuation and far past
the f32 `exp` underflow knee (~88). We therefore rescale the sinogram by a
single constant into a realistic attenuation range (peak ~6, well under the
`minus_log` clamp of -ln(1e-6)=13.8 and nowhere near underflow). FBP is linear
and `pearson_disk` is amplitude-scale invariant, so reconstructing the scaled
sinogram recovers the SAME phantom correlation as the unscaled parity test.

  transmission T = exp(-scale * sino)        in [exp(-6), 1]
  data           = dark + (flat - dark) * T  raw intensity, [nproj, nz, nx]
  data_white     = flat = 1000               flat/open-beam field
  data_dark      = dark = 10                  dark field
  theta          = angles in DEGREES          -> reader converts back to radians

So `normalize` (= (data-dark)/(flat-dark)) recovers `T` exactly via real
subtraction and division (not an identity flat/dark), `minus_log` recovers
`scale * sino`, and `find_center_vo -> fbp` then recovers the phantom. The
uint16 dtype-cast read path is covered separately by dxchange_read.rs.

Run:  python3 tools/gen_dxchange_pipeline_fixture.py
"""
import os
import numpy as np
import h5py

FIX = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")

sino = np.load(os.path.join(FIX, "sino.npy"))        # (nproj, nz, nx) float32
angles = np.load(os.path.join(FIX, "angles.npy"))    # (nproj,) radians
nproj, nz, nx = sino.shape

# Rescale the huge 0..255-phantom line integrals into a realistic attenuation
# peak; constant scale keeps FBP linearity so phantom recovery is unchanged.
PEAK_ATTEN = 6.0
FLAT0, DARK0 = 1000.0, 10.0
scale = PEAK_ATTEN / float(sino.max())

trans = np.exp(-scale * sino).astype(np.float64)     # transmission in (0, 1]
data = (DARK0 + (FLAT0 - DARK0) * trans).astype(np.float32)
white = np.full((2, nz, nx), FLAT0, dtype=np.float32)
dark = np.full((2, nz, nx), DARK0, dtype=np.float32)
theta_deg = (angles * 180.0 / np.pi).astype(np.float32)

out = os.path.join(FIX, "pipeline_dxchange.h5")
with h5py.File(out, "w") as f:
    g = f.create_group("exchange")
    g.create_dataset("data", data=data, compression="gzip", chunks=True)
    g.create_dataset("data_white", data=white, compression="gzip", chunks=True)
    g.create_dataset("data_dark", data=dark, compression="gzip", chunks=True)
    g.create_dataset("theta", data=theta_deg)

recovered_sino_max = scale * float(sino.max())
print(f"wrote {out}")
print(f"  scale={scale:.6g}  recovered sino max={recovered_sino_max:.4g} (clamp at 13.8)")
print(f"  data {data.shape} {data.dtype} in [{data.min():.4g}, {data.max():.4g}]")
print(f"  theta_deg [{theta_deg[0]:.3f} .. {theta_deg[-1]:.3f}], nproj={nproj}, nx={nx}")
