#!/usr/bin/env python
"""Generate the golden for tomoxide's Farago phase-retrieval test.

The reference is tomocupy `processing/retrieve_phase.py::farago_filter`
(Farago 2024). tomocupy runs on the GPU via cupy, which is unavailable on this
machine, so this script is a FAITHFUL LINE-BY-LINE TRANSCRIPTION of tomocupy's
exact functions with `cp` -> `np`: `_calc_pad`, `_calc_pad_val`,
`_calc_pad_width`, `_reciprocal_coord`, `_reciprocal_grid`,
`_farago_filter_factor`, and `_retrieve_phase`. `cupy.fft` and `numpy.fft`
implement the same DFT (cuFFT vs pocketfft), so this reproduces the GPU result to
the float round-off floor — the same basis on which the standard-Paganin test
holds tomoxide to tomopy parity. Nothing is invented: every formula below is
copied from the cited tomocupy source.

Run under any env with numpy + scipy (e.g. the tomopy-golden env):
    /Users/stevek/mamba/envs/tomopy-golden/bin/python \
        tools/gen_tomocupy_farago_golden.py

Writes, under crates/tomoxide/tests/fixtures/:
  - farago_input.npy      (nproj, dy, dz)  input radiograph stack, float32
  - tomocupy_farago.npy   (nproj, dy, dz)  Farago output, float32

Params are tomocupy defaults: pixel_size=1e-4, dist=50, energy=20, db=1000.
"""
import os

import numpy as np
import scipy.fft

# tomocupy runs the transform on the GPU in SINGLE precision (cupy.fft of a
# float32 array yields complex64). numpy.fft would upcast float32->complex128 and
# make the golden higher-precision than the reference, so use scipy.fft, which
# preserves float32->complex64 — the faithful CPU stand-in for cuFFT.
fft2 = scipy.fft.fft2
ifft2 = scipy.fft.ifft2
fftshift = scipy.fft.fftshift

# --- tomocupy constants (retrieve_phase.py:49-52) ---------------------------
SPEED_OF_LIGHT = 299792458e2  # [cm/s]
PI = 3.14159265359
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]


def _wavelength(energy):
    return 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def _reciprocal_coord(pixel_size, num_grid):
    n = num_grid - 1
    rc = np.arange(-n, num_grid, 2, dtype=np.float32)
    rc *= 0.5 / (n * pixel_size)
    return rc


def _reciprocal_grid(pixel_size, nx, ny):
    indx = _reciprocal_coord(pixel_size, nx)
    indy = _reciprocal_coord(pixel_size, ny)
    np.square(indx, out=indx)
    np.square(indy, out=indy)
    idx, idy = np.meshgrid(indy, indx)
    return idx + idy


def _farago_filter_factor(energy, dist, db, w2):
    return 1 / (
        np.cos(PI * _wavelength(energy) * dist * w2)
        + db * np.sin(PI * _wavelength(energy) * dist * w2)
    )


def _calc_pad_width(dim, pixel_size, wavelength, dist):
    pad_pix = np.ceil(PI * wavelength * dist / pixel_size**2)
    return int((pow(2, np.ceil(np.log2(dim + pad_pix))) - dim) * 0.5)


def _calc_pad_val(data):
    return np.mean((data[..., 0] + data[..., -1]) * 0.5)


def _calc_pad(data, pixel_size, dist, energy, pad):
    dx, dy, dz = data.shape
    wavelength = _wavelength(energy)
    py, pz, val = 0, 0, 0
    if pad:
        val = _calc_pad_val(data)
        py = _calc_pad_width(dy, pixel_size, wavelength, dist)
        pz = _calc_pad_width(dz, pixel_size, wavelength, dist)
    return py, pz, val


def _retrieve_phase(data, phase_filter, px, py, prj, pad):
    dx, dy, dz = data.shape
    num_jobs = data.shape[0]
    normalized_phase_filter = phase_filter / phase_filter.max()
    for m in range(num_jobs):
        prj[px : dy + px, py : dz + py] = data[m]
        prj[:px] = prj[px]
        prj[-px:] = prj[-px - 1]
        prj[:, :py] = prj[:, py][:, np.newaxis]
        prj[:, -py:] = prj[:, -py - 1][:, np.newaxis]
        fproj = fft2(prj)
        fproj *= normalized_phase_filter
        proj = np.real(ifft2(fproj))
        if pad:
            proj = proj[px : dy + px, py : dz + py]
        data[m] = proj


def farago_filter(data, pixel_size, dist, energy, db, pad=True):
    """tomocupy `farago_filter` (retrieve_phase.py:110-150)."""
    py, pz, val = _calc_pad(data, pixel_size, dist, energy, pad)
    dx, dy, dz = data.shape
    w2 = _reciprocal_grid(pixel_size, dy + 2 * py, dz + 2 * pz)
    phase_filter = fftshift(_farago_filter_factor(energy, dist, db, w2))
    prj = np.full((dy + 2 * py, dz + 2 * pz), val, dtype=data.dtype)
    _retrieve_phase(data, phase_filter, py, pz, prj, pad)
    return data


NPROJ, DY, DZ = 6, 48, 64
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "crates", "tomoxide", "tests", "fixtures")

# Same deterministic radiograph stack as the Paganin/GPaganin tests (moving
# Gaussian blob plus a left/right-asymmetric ramp so the edge-pad value is used).
yy, xx = np.mgrid[0:DY, 0:DZ].astype("float32")
stack = np.empty((NPROJ, DY, DZ), dtype="float32")
for m in range(NPROJ):
    cy = DY * (0.3 + 0.4 * m / NPROJ)
    cx = DZ * (0.25 + 0.5 * m / NPROJ)
    blob = np.exp(-(((yy - cy) ** 2) / (2 * 7.0**2) + ((xx - cx) ** 2) / (2 * 9.0**2)))
    ramp = 0.2 + 0.1 * (xx / DZ)
    stack[m] = (ramp + blob).astype("float32")

phase = farago_filter(
    stack.copy(), pixel_size=1e-4, dist=50, energy=20, db=1000, pad=True
).astype("float32")

np.save(os.path.join(out, "farago_input.npy"), stack)
np.save(os.path.join(out, "tomocupy_farago.npy"), phase)
print("input", stack.shape, "output", phase.shape)
print("input range", float(stack.min()), float(stack.max()))
print("phase range", float(phase.min()), float(phase.max()))
print("wrote fixtures to", os.path.normpath(out))
