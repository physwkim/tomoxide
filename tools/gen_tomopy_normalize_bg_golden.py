#!/usr/bin/env python
"""Generate the golden for tomoxide's background (air-region) normalization test.

Reference: tomopy `prep/normalize.py:207` `normalize_bg` →
`libtomo/prep/prep.c::normalize_bg`. This calls the REAL tomopy function
(tomopy 1.15.3 in the tomopy-golden env) — no transcription — so the golden is
exactly tomopy's output.

`normalize_bg` is a per-projection-row reduction: the mean of the `air`
left-boundary pixels and the `air` right-boundary pixels gives an air baseline
that is linearly interpolated across the detector width, then every pixel is
divided by its local baseline (scaling the air boundaries to one). The C does
all arithmetic in float32 in a fixed accumulation order, so the port matches
bit-for-bit (Δ = 0).

The synthetic stack has an air baseline that rises left→right (so the per-row
slope is non-trivial, `air_left != air_right`), a central absorbing-object dip,
a small per-projection/row offset, and distinct noise. All values stay positive
so the `<= 0 → 1` clamp is not exercised here.

Run under the tomopy-golden env:
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_normalize_bg_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - normalize_bg_input.npy       (nproj, nrows, ncol) input, float32
  - tomopy_normalize_bg_air1.npy air=1 output, float32
  - tomopy_normalize_bg_air4.npy air=4 output, float32
"""
import os

import numpy as np
import tomopy

NPROJ, NROWS, NCOL = 8, 3, 64


def main():
    rng = np.random.default_rng(2)

    x = np.linspace(0.0, 1.0, NCOL, dtype=np.float64)
    obj = 0.5 * np.exp(-((x - 0.5) ** 2) / (2.0 * 0.05 ** 2))  # central dip
    stack = np.empty((NPROJ, NROWS, NCOL), dtype=np.float32)
    for p in range(NPROJ):
        for s in range(NROWS):
            base = 1.0 + 0.3 * x + 0.02 * p + 0.01 * s  # rises left->right
            stack[p, s, :] = base - obj

    stack += (rng.standard_normal(stack.shape) * 1e-3).astype(np.float32)
    stack = stack.astype(np.float32)

    out_air1 = tomopy.normalize_bg(stack.copy(), air=1).astype("float32")
    out_air4 = tomopy.normalize_bg(stack.copy(), air=4).astype("float32")

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")
    np.save(os.path.join(out, "normalize_bg_input.npy"), stack)
    np.save(os.path.join(out, "tomopy_normalize_bg_air1.npy"), out_air1)
    np.save(os.path.join(out, "tomopy_normalize_bg_air4.npy"), out_air4)
    print("tomopy", tomopy.__version__)
    print("input", stack.shape, "range", float(stack.min()), float(stack.max()))
    print("air1 range", float(out_air1.min()), float(out_air1.max()))
    print("air4 range", float(out_air4.min()), float(out_air4.max()))
    print("wrote fixtures to", os.path.normpath(out))


if __name__ == "__main__":
    main()
