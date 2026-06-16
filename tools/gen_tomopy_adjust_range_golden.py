#!/usr/bin/env python
"""Golden for `adjust_range` (tomopy `misc/corr.py:90`).

Clips an array's dynamic range to `[dmin, dmax]`. `None` bounds default to the
data's own min/max; a bound is only applied when strictly tighter than the data
range (strict `>`/`<`), so both-None is a no-op and looser-than-data bounds are
no-ops too. Pure NumPy (np.max/np.min + boolean-mask assignment) → bit-exact.

Cases cover: both None (no-op), high-only, low-only, both, and looser-than-data
(no-op via the guards). `None` is encoded in the saved param arrays as NaN.

Run with the tomopy env (tomopy 1.15.3):
    export PATH="/opt/homebrew/bin:$PATH"
    micromamba run -n tomopy-golden python3 tools/gen_tomopy_adjust_range_golden.py
"""
import os

import numpy as np
from tomopy.misc.corr import adjust_range

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260616)
n0, n1, n2 = 3, 12, 16
base = (5.0 * np.sin(np.linspace(0, 5, n1)[:, None] + np.linspace(0, 3, n2)[None, :])).astype("float32")
arr0 = np.broadcast_to(base, (n0, n1, n2)).astype("float32").copy()
arr0 += (1.5 * rng.standard_normal((n0, n1, n2))).astype("float32")

dlo = float(arr0.min())
dhi = float(arr0.max())
span = dhi - dlo

# (dmin, dmax) with None encoded as NaN.
cases = [
    (np.nan, np.nan),                       # both None -> no-op
    (np.nan, dhi - 0.25 * span),            # high clip only
    (dlo + 0.25 * span, np.nan),            # low clip only
    (dlo + 0.2 * span, dhi - 0.2 * span),   # both clips
    (dlo - span, dhi + span),               # looser than data -> no-op (guards)
]

inputs, outputs, dmins, dmaxs = [], [], [], []
for dmin, dmax in cases:
    a = arr0.copy()
    inp = a.copy()
    kw = {}
    if not np.isnan(dmin):
        kw["dmin"] = float(dmin)
    if not np.isnan(dmax):
        kw["dmax"] = float(dmax)
    out = adjust_range(a, **kw)
    assert out.dtype == np.float32, out.dtype
    inputs.append(inp)
    outputs.append(out.astype("float32"))
    dmins.append(dmin)
    dmaxs.append(dmax)
    print(f"dmin={dmin} dmax={dmax}: out range [{out.min():.4f}, {out.max():.4f}]")

np.save(os.path.join(OUT, "adjust_range_input.npy"),
        np.ascontiguousarray(np.stack(inputs)))
np.save(os.path.join(OUT, "adjust_range_output.npy"),
        np.ascontiguousarray(np.stack(outputs)))
np.save(os.path.join(OUT, "adjust_range_dmin.npy"), np.asarray(dmins, dtype="float64"))
np.save(os.path.join(OUT, "adjust_range_dmax.npy"), np.asarray(dmaxs, dtype="float64"))
print("cases", len(cases), "shape", (n0, n1, n2))
