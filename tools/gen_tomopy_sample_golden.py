#!/usr/bin/env python
"""Generate the golden for tomoxide's downsample/upsample test.

Reference: tomopy `misc/morph.py::downsample`/`upsample` →
`libtomo/misc/morph.c` (`c_sample`). These call the REAL tomopy functions
(tomopy 1.15.3) — no transcription.

downsample bins by `2^level` along `axis`, each output = mean of the bin
accumulated as Σ(data/binsize) in f32; upsample replicates each value `2^level`
times along the axis. Both are f32 in the upstream order → bit-exact (Δ = 0).

Run under the tomopy-golden env:
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_sample_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - sample_input.npy             (4, 4, 8) input, float32
  - tomopy_downsample_ax2_l1.npy downsample axis=2 level=1 -> (4,4,4)
  - tomopy_downsample_ax0_l1.npy downsample axis=0 level=1 -> (2,4,8)
  - tomopy_upsample_ax2_l1.npy   upsample   axis=2 level=1 -> (4,4,16)
  - tomopy_upsample_ax1_l1.npy   upsample   axis=1 level=1 -> (4,8,8)
"""
import os

import numpy as np
import tomopy


def main():
    rng = np.random.default_rng(5)
    arr = (0.2 + rng.random((4, 4, 8))).astype(np.float32)

    ds2 = tomopy.misc.morph.downsample(arr.copy(), level=1, axis=2).astype("float32")
    ds0 = tomopy.misc.morph.downsample(arr.copy(), level=1, axis=0).astype("float32")
    us2 = tomopy.misc.morph.upsample(arr.copy(), level=1, axis=2).astype("float32")
    us1 = tomopy.misc.morph.upsample(arr.copy(), level=1, axis=1).astype("float32")

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")
    np.save(os.path.join(out, "sample_input.npy"), arr)
    np.save(os.path.join(out, "tomopy_downsample_ax2_l1.npy"), ds2)
    np.save(os.path.join(out, "tomopy_downsample_ax0_l1.npy"), ds0)
    np.save(os.path.join(out, "tomopy_upsample_ax2_l1.npy"), us2)
    np.save(os.path.join(out, "tomopy_upsample_ax1_l1.npy"), us1)
    print("tomopy", tomopy.__version__)
    print("ds2", ds2.shape, "ds0", ds0.shape, "us2", us2.shape, "us1", us1.shape)
    print("wrote fixtures to", os.path.normpath(out))


if __name__ == "__main__":
    main()
