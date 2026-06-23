#!/usr/bin/env python
"""Generate tomopy golden data for the vector-tomography parity test.

Runs tomopy 1.15 `tomopy.recon.vector.{vector,vector2,vector3}` on small,
fixed-seed random datasets and saves the inputs and the reconstructed vector
fields. tomoxide's `tomoxide_recon::vector` port reconstructs the SAME inputs
offline and must match bit-for-bit (it is a line-for-line port of the C kernels,
including the mixed float/double arithmetic).

Inputs are in tomopy's public projection order `(dt, dy, dx)` = (angles,
slices, detector). Centers are left at the tomopy default (`dx/2` per slice).
vector2/vector3 use only theta1/center1 internally (tomopy behaviour), but the
extra thetas are still saved so the Rust test passes the same arguments.

Run with the tomopy-enabled env, e.g.:
    micromamba run -n tomo python tools/gen_tomopy_vector_golden.py
"""
import os

import numpy as np
from tomopy.recon.vector import vector, vector2, vector3

OUT = os.path.join(os.path.dirname(__file__), "..",
                   "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20240617)
DT, DY, DX = 16, 2, 16          # angles, slices, detector (single-dataset case)
# vector2/vector3 reconstruct a full 3-D vector field (dx**3 voxels), so their
# axis-1/axis-2 index formulas are only in-bounds when the number of slices
# equals the detector width — tomopy corrupts memory otherwise. Use a cube.
NCUBE = 8
NUM_ITER = 4
theta1 = np.linspace(0.0, np.pi, DT, endpoint=False).astype("float32")
theta2 = (theta1 + 0.013).astype("float32")   # unused by tomopy, saved anyway
theta3 = (theta1 + 0.027).astype("float32")
theta1c = np.linspace(0.0, np.pi, NCUBE, endpoint=False).astype("float32")
theta2c = (theta1c + 0.013).astype("float32")
theta3c = (theta1c + 0.027).astype("float32")


def rand_tomo():
    return rng.standard_normal((DT, DY, DX)).astype("float32")


def rand_cube():
    # (dt, dy, dx) with dt = dy = dx = NCUBE.
    return rng.standard_normal((NCUBE, NCUBE, NCUBE)).astype("float32")


# --- vector (single dataset → 2 components) ---
t1 = rand_tomo()
r1, r2 = vector(t1.copy(), theta1.copy(), center=None, num_iter=NUM_ITER)
np.save(os.path.join(OUT, "vector_tomo1.npy"), t1)
np.save(os.path.join(OUT, "vector_theta1.npy"), theta1)
np.save(os.path.join(OUT, "vector_out1.npy"), r1.astype("float32"))
np.save(os.path.join(OUT, "vector_out2.npy"), r2.astype("float32"))

# --- vector2 (two datasets → 3 components, axis1=1, axis2=2) ---
a1 = rand_cube()
a2 = rand_cube()
v2r1, v2r2, v2r3 = vector2(a1.copy(), a2.copy(), theta1c.copy(), theta2c.copy(),
                           center1=None, center2=None, num_iter=NUM_ITER,
                           axis1=1, axis2=2)
np.save(os.path.join(OUT, "vector2_tomo1.npy"), a1)
np.save(os.path.join(OUT, "vector2_tomo2.npy"), a2)
np.save(os.path.join(OUT, "vector2_theta1.npy"), theta1c)
np.save(os.path.join(OUT, "vector2_out1.npy"), v2r1.astype("float32"))
np.save(os.path.join(OUT, "vector2_out2.npy"), v2r2.astype("float32"))
np.save(os.path.join(OUT, "vector2_out3.npy"), v2r3.astype("float32"))

# --- vector3 (three datasets → 3 components, axis1=0, axis2=1, axis3=2) ---
b1 = rand_cube()
b2 = rand_cube()
b3 = rand_cube()
v3r1, v3r2, v3r3 = vector3(b1.copy(), b2.copy(), b3.copy(),
                           theta1c.copy(), theta2c.copy(), theta3c.copy(),
                           center1=None, center2=None, center3=None,
                           num_iter=NUM_ITER, axis1=0, axis2=1, axis3=2)
np.save(os.path.join(OUT, "vector3_tomo1.npy"), b1)
np.save(os.path.join(OUT, "vector3_tomo2.npy"), b2)
np.save(os.path.join(OUT, "vector3_tomo3.npy"), b3)
np.save(os.path.join(OUT, "vector3_theta1.npy"), theta1c)
np.save(os.path.join(OUT, "vector3_out1.npy"), v3r1.astype("float32"))
np.save(os.path.join(OUT, "vector3_out2.npy"), v3r2.astype("float32"))
np.save(os.path.join(OUT, "vector3_out3.npy"), v3r3.astype("float32"))

print("vector golden written:")
print("  vector :", r1.shape, "x2")
print("  vector2:", v2r1.shape, "x3")
print("  vector3:", v3r1.shape, "x3")
