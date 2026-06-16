#!/usr/bin/env python
"""Generate the golden for tomoxide's morph.pad test.

Reference: tomopy `misc/morph.py::pad`. Calls the REAL tomopy function
(tomopy 1.15.3) — no transcription.

`pad` widens `axis` by `npad` on each side (npad=None → ⌈(dim·√2−dim)/2⌉); the
flanks are a constant (`mode='constant'`, `constant_values`) or the replicated
edge slab (`mode='edge'`). Pure copy/fill → bit-exact (Δ = 0).

Run under the tomopy-golden env:
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_pad_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - pad_input.npy                (4, 4, 8) input, float32
  - tomopy_pad_ax2_const_def.npy axis=2 constant npad=default
  - tomopy_pad_ax2_edge_def.npy  axis=2 edge     npad=default
  - tomopy_pad_ax0_const_n3.npy  axis=0 constant npad=3 constant_values=0.5
  - tomopy_pad_ax1_edge_n2.npy   axis=1 edge     npad=2
"""
import os

import numpy as np
import tomopy


def main():
    rng = np.random.default_rng(6)
    arr = (0.2 + rng.random((4, 4, 8))).astype(np.float32)

    p_c_def = tomopy.misc.morph.pad(arr.copy(), 2, mode="constant").astype("float32")
    p_e_def = tomopy.misc.morph.pad(arr.copy(), 2, mode="edge").astype("float32")
    p_c_n3 = tomopy.misc.morph.pad(
        arr.copy(), 0, npad=3, mode="constant", constant_values=0.5).astype("float32")
    p_e_n2 = tomopy.misc.morph.pad(arr.copy(), 1, npad=2, mode="edge").astype("float32")

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")
    np.save(os.path.join(out, "pad_input.npy"), arr)
    np.save(os.path.join(out, "tomopy_pad_ax2_const_def.npy"), p_c_def)
    np.save(os.path.join(out, "tomopy_pad_ax2_edge_def.npy"), p_e_def)
    np.save(os.path.join(out, "tomopy_pad_ax0_const_n3.npy"), p_c_n3)
    np.save(os.path.join(out, "tomopy_pad_ax1_edge_n2.npy"), p_e_n2)
    print("tomopy", tomopy.__version__)
    print("shapes", p_c_def.shape, p_e_def.shape, p_c_n3.shape, p_e_n2.shape)
    print("wrote fixtures to", os.path.normpath(out))


if __name__ == "__main__":
    main()
