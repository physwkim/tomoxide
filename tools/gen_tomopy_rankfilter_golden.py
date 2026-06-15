#!/usr/bin/env python
"""Generate tomopy golden data for the CPU RankFilter parity test.

Runs tomopy 1.15.3 `median_filter3d` and `remove_outlier3d` on a fixed 3-D
volume seeded with salt-and-pepper spikes, and saves the input plus both
outputs as .npy fixtures. tomoxide reconstructs the SAME arrays offline and
must match bit-for-bit (these are pure neighbourhood ops — no projector, no
FFT — so true tomopy numeric parity is expected).

Run with the tomopy-enabled env:
    /Users/stevek/mamba/envs/tomopy-golden/bin/python \
        tools/gen_tomopy_rankfilter_golden.py
"""
import os

import numpy as np
import tomopy.misc.corr as corr

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

# Deterministic 3-D volume: a smooth ramp/blob base so the median has real
# structure to preserve, plus injected salt-and-pepper spikes the filter must
# remove. Shape (dz, dy, dx) — the same array fed to tomopy and to tomoxide.
rng = np.random.default_rng(20260615)
dz, dy, dx = 12, 20, 24
z = np.linspace(0.0, 1.0, dz)[:, None, None]
y = np.linspace(0.0, 1.0, dy)[None, :, None]
x = np.linspace(0.0, 1.0, dx)[None, None, :]
base = (0.3 + 0.5 * z + 0.4 * y - 0.2 * x
        + 0.3 * np.sin(6.0 * x + 3.0 * y)).astype("float32")
base = np.ascontiguousarray(np.broadcast_to(base, (dz, dy, dx)).astype("float32"))

vol = base.copy()
# ~5% of voxels become spikes (very high or very low) to exercise both the
# pure-median and the dezinger-threshold paths.
n_spike = int(0.05 * vol.size)
idx = rng.choice(vol.size, size=n_spike, replace=False)
flat = vol.ravel()
flat[idx[: n_spike // 2]] = 10.0     # bright spikes
flat[idx[n_spike // 2:]] = -10.0     # dark spikes
vol = flat.reshape(dz, dy, dx).astype("float32")
vol = np.ascontiguousarray(vol)

# median_filter3d: size=3 (radius 1) and size=5 (radius 2).
med3 = corr.median_filter3d(vol.copy(), size=3, ncore=1).astype("float32")
med5 = corr.median_filter3d(vol.copy(), size=5, ncore=1).astype("float32")

# remove_outlier3d (dezinger): only spikes exceeding `dif` from the local
# median are replaced; smooth structure is left untouched.
out_small = corr.remove_outlier3d(vol.copy(), dif=0.5, size=3, ncore=1).astype("float32")
out_large = corr.remove_outlier3d(vol.copy(), dif=5.0, size=3, ncore=1).astype("float32")

np.save(os.path.join(OUT, "rankfilter_input.npy"), vol)
np.save(os.path.join(OUT, "tomopy_median3.npy"), med3)
np.save(os.path.join(OUT, "tomopy_median5.npy"), med5)
np.save(os.path.join(OUT, "tomopy_dezinger_small.npy"), out_small)
np.save(os.path.join(OUT, "tomopy_dezinger_large.npy"), out_large)

print("input", vol.shape, "spikes", n_spike)
print("median3 changed voxels:", int(np.count_nonzero(med3 != vol)))
print("median5 changed voxels:", int(np.count_nonzero(med5 != vol)))
print("dezinger dif=0.5 changed:", int(np.count_nonzero(out_small != vol)))
print("dezinger dif=5.0 changed:", int(np.count_nonzero(out_large != vol)))
