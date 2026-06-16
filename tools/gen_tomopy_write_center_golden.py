#!/usr/bin/env python
"""Generate the center-enumeration golden for tomoxide's `write_center` test.

`write_center` (tomopy `recon/rotation.py:438`) reconstructs one slice across a
range of rotation centers. The reconstruction *content* goes through gridrec,
which tomoxide implements with a gridrec-*family* kernel (Kaiser-Bessel, ramp
weight) rather than tomopy's PSWF + parzen, so the slice pixels are NOT
bit-identical to tomopy and are not goldened here. The one part held to exact
parity is the **center enumeration**, which is pure `numpy.arange`:

    cen_range is None -> np.arange(dx/2 - 5, dx/2 + 5, 0.5)   # rotation.py:548
    else              -> np.arange(*cen_range)                # rotation.py:550

This script writes those center arrays (float32) for the test's cases so the Rust
`arange` is checked against numpy's exact length/values.

Run under any env with numpy (e.g. the tomopy-golden env):
    /Users/stevek/mamba/envs/tomopy-golden/bin/python \
        tools/gen_tomopy_write_center_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - write_center_centers_default.npy   centers for ncol=64, cen_range=None
  - write_center_centers_range.npy     centers for cen_range=(28.0, 36.0, 0.5)
"""
import os

import numpy as np

NCOL = 64  # detector width (tomopy `dx`) used by the test sinogram

# tomopy `write_center`: default center range when `cen_range is None`.
centers_default = np.arange(NCOL / 2 - 5, NCOL / 2 + 5, 0.5).astype("float32")

# tomopy `write_center`: explicit `cen_range` -> `np.arange(*cen_range)`.
centers_range = np.arange(28.0, 36.0, 0.5).astype("float32")

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")
np.save(os.path.join(out, "write_center_centers_default.npy"), centers_default)
np.save(os.path.join(out, "write_center_centers_range.npy"), centers_range)
print("default centers", centers_default.shape, centers_default[0], centers_default[-1])
print("range centers", centers_range.shape, centers_range[0], centers_range[-1])
print("wrote fixtures to", os.path.normpath(out))
