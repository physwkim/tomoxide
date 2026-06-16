#!/usr/bin/env python
"""Generate the golden for tomoxide's nearest-flat-fields normalization test.

Reference: tomopy `prep/normalize.py:245` `normalize_nf` (averaging='mean', the
default — averaging='median' passes `dtype=np.float32` to `np.median`, which
raises TypeError on modern numpy, so it has no reference output). This calls the
REAL tomopy function (tomopy 1.15.3 in the tomopy-golden env) — no transcription.

Each of the `len(flat_loc)` flat groups (`flats.shape[0] // len(flat_loc)`
frames each) contributes its per-pixel median as the flat for the projections
nearest its `flat_loc` position; `dark` is the per-pixel mean of the dark frames.
Each projection is `(proj - dark) / max(flat - dark, 1e-6)`, optionally clamped
above by `cutoff`. All f32 in the upstream order → bit-exact (Δ = 0).

Two cases are emitted: an even group size (2 → median averages two values) with
no cutoff, and an odd group size (3 → median selects) with a cutoff. In both,
the (0,0) pixel of every flat is forced equal to the dark mean so `flat-dark`
hits the `< 1e-6 → 1e-6` clamp.

Run under the tomopy-golden env:
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_normalize_nf_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - normalize_nf_tomo.npy         (nproj, ny, nx) projections, float32
  - normalize_nf_dark.npy         (ndark, ny, nx) dark frames, float32
  - normalize_nf_flatsA.npy       4 flats (2 groups x 2), float32
  - normalize_nf_flatsB.npy       6 flats (2 groups x 3), float32
  - tomopy_normalize_nf_A.npy     flat_loc=[0,7], cutoff=None, float32
  - tomopy_normalize_nf_B.npy     flat_loc=[1,6], cutoff=1.5,  float32
"""
import os

import numpy as np
import tomopy

NPROJ, NY, NX = 8, 2, 4


def main():
    rng = np.random.default_rng(4)

    tomo = (0.3 + 0.5 * rng.random((NPROJ, NY, NX))).astype(np.float32)
    dark = (0.1 + 0.05 * rng.random((2, NY, NX))).astype(np.float32)
    dark_mean = np.mean(dark, axis=0, dtype=np.float32)

    flats_a = (1.0 + 0.3 * rng.random((4, NY, NX))).astype(np.float32)
    flats_b = (1.0 + 0.3 * rng.random((6, NY, NX))).astype(np.float32)
    # Force the (0,0) flat to the dark mean so flat-dark hits the 1e-6 clamp.
    flats_a[:, 0, 0] = dark_mean[0, 0]
    flats_b[:, 0, 0] = dark_mean[0, 0]

    out_a = tomopy.prep.normalize.normalize_nf(
        tomo.copy(), flats_a.copy(), dark.copy(), [0, 7]).astype("float32")
    out_b = tomopy.prep.normalize.normalize_nf(
        tomo.copy(), flats_b.copy(), dark.copy(), [1, 6], cutoff=1.5).astype("float32")

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")
    np.save(os.path.join(out, "normalize_nf_tomo.npy"), tomo)
    np.save(os.path.join(out, "normalize_nf_dark.npy"), dark)
    np.save(os.path.join(out, "normalize_nf_flatsA.npy"), flats_a)
    np.save(os.path.join(out, "normalize_nf_flatsB.npy"), flats_b)
    np.save(os.path.join(out, "tomopy_normalize_nf_A.npy"), out_a)
    np.save(os.path.join(out, "tomopy_normalize_nf_B.npy"), out_b)
    print("tomopy", tomopy.__version__)
    print("A range", float(out_a.min()), float(out_a.max()))
    print("B range", float(out_b.min()), float(out_b.max()))
    print("wrote fixtures to", os.path.normpath(out))


if __name__ == "__main__":
    main()
