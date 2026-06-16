#!/usr/bin/env python
"""Generate the golden for tomoxide's sino_360_to_180 test.

Reference: tomopy `misc/morph.py::sino_360_to_180`. This calls the REAL tomopy
function (tomopy 1.15.3 in the tomopy-golden env) — no transcription — so the
golden is exactly tomopy's output.

The first n = dx//2 projections (0–180°) are kept; the next n (180–360°) are
reversed along the detector-column axis and stitched on to widen the detector,
overlapping by `overlap` columns where the two are linearly cross-faded. Direct
regions are exact f32 copies; the seam blend is computed in float64 (numpy
promotes float64-weights · float32-data) and cast back to float32, so the port
matches bit-for-bit (Δ = 0). Both rotation sides and an overlap=0 (seam-less)
case are emitted.

Run under the tomopy-golden env:
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_sino360_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - sino360_input.npy            (dx, dy, dz) input, float32
  - tomopy_sino360_left_o4.npy   rotation='left',  overlap=4, float32
  - tomopy_sino360_right_o4.npy  rotation='right', overlap=4, float32
  - tomopy_sino360_left_o0.npy   rotation='left',  overlap=0, float32
"""
import os

import numpy as np
import tomopy

DX, DY, DZ = 8, 2, 16


def main():
    rng = np.random.default_rng(3)

    stack = np.empty((DX, DY, DZ), dtype=np.float32)
    for p in range(DX):
        for r in range(DY):
            stack[p, r, :] = 1.0 + 0.1 * np.arange(DZ) + 0.5 * p + 0.2 * r
    stack += (rng.standard_normal(stack.shape) * 1e-3).astype(np.float32)
    stack = stack.astype(np.float32)

    out_left = tomopy.misc.morph.sino_360_to_180(
        stack.copy(), overlap=4, rotation="left").astype("float32")
    out_right = tomopy.misc.morph.sino_360_to_180(
        stack.copy(), overlap=4, rotation="right").astype("float32")
    out_left0 = tomopy.misc.morph.sino_360_to_180(
        stack.copy(), overlap=0, rotation="left").astype("float32")

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")
    np.save(os.path.join(out, "sino360_input.npy"), stack)
    np.save(os.path.join(out, "tomopy_sino360_left_o4.npy"), out_left)
    np.save(os.path.join(out, "tomopy_sino360_right_o4.npy"), out_right)
    np.save(os.path.join(out, "tomopy_sino360_left_o0.npy"), out_left0)
    print("tomopy", tomopy.__version__)
    print("input", stack.shape)
    print("left_o4", out_left.shape, "right_o4", out_right.shape, "left_o0", out_left0.shape)
    print("wrote fixtures to", os.path.normpath(out))


if __name__ == "__main__":
    main()
