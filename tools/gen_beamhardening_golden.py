#!/usr/bin/env python
"""Generate golden data for the beam-hardening parity test.

This is a faithful translation of the parts of the `beamhardening` package
(aps-7bm/beamhardening, `beamhardening.py` + `material.py`) that tomocupy's
`processing/external/hardening.py` actually exercises:

  * `compute_interp_values()` — builds the centerline (extinction-length →
    sample-thickness) LUT and the angular (vertical-angle → correction-factor)
    LUT from a set of bending-magnet source spectra, after filtering and
    scintillator absorption.
  * `find_angles()` — per-detector-row vertical angle from the flat field.
  * `correct_centerline()` + `correct_angle()` — the two `np.interp` passes
    tomocupy applies after minus-log (`proc_functions.beamhardening`).

The ONE deviation from upstream is the cross-section source: `beamhardening`
uses `xraydb`, which has no Rust port, so both this generator and the Rust port
(`tomoxide-prep::hardening`) use **xraylib** instead. For pure elements xraydb
and xraylib are bit-identical; for compounds they differ by ~4e-5 (atomic-weight
tables). Using xraylib on both sides makes the Rust port reproduce this golden
to the f64 floor while staying physically within ~1e-4 of real tomocupy.

Run with the tomopy env (xraylib + scipy installed there):
    micromamba run -n tomo python tools/gen_beamhardening_golden.py
"""
import os

import numpy as np
import scipy.integrate
import scipy.signal
import xraylib

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "crates", "tomoxide-prep", "data")
OUT = os.path.join(HERE, "..", "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

# --- fixed test configuration (mirrored verbatim in the Rust test) ---------
SCINT = ("Lu3Al5O12", 6.73, 100.0)      # formula, density g/cc, thickness um
SAMPLE = ("Fe", 7.87)                    # formula, density
FILTERS = [("Al", 2.7, 750.0), ("Cu", 8.96, 50.0), ("Be", 1.85, 250.0)]
REF_TRANS = 0.1
THRESHOLD_TRANS = 1e-5
D_SOURCE_M = 36.0
PIXEL_SIZE_UM = 10.0


def material_mu(formula, density, energies_ev, kind):
    """Linear attenuation (1/cm), xraylib CS_*_CP * density. energies in eV."""
    if kind == "total":
        cs = np.array([xraylib.CS_Total_CP(formula, e / 1000.0) for e in energies_ev])
    else:
        cs = np.array([xraylib.CS_Photo_CP(formula, e / 1000.0) for e in energies_ev])
    return cs * density


def read_spectra():
    """Read the vendored Psi_##urad.dat bending-magnet spectra."""
    spectra = {}
    for fn in sorted(os.listdir(DATA)):
        if fn.startswith("Psi") and fn.endswith(".dat"):
            angle = float(fn.split("_")[1][:2])
            d = np.genfromtxt(os.path.join(DATA, fn), comments="!")
            spectra[angle] = (d[:, 0].copy(), d[:, 1].copy())
    return spectra


def apply_filters(energies, power):
    power = power.copy()
    for formula, dens, thick in FILTERS:
        att = material_mu(formula, dens, energies, "total") * thick * 1e-4
        power = power * np.exp(-att)
    return power


def find_interp_values_one_angle(energies, power):
    sample_thicknesses = np.sort(np.concatenate(
        (-np.logspace(1, -1, 41), [0], np.logspace(-1, 4.5, 111))))
    sample_ext = material_mu(SAMPLE[0], SAMPLE[1], energies, "total") * 1 * 1e-4
    scint_ext = material_mu(SCINT[0], SCINT[1], energies, "photo") * SCINT[2] * 1e-4
    scint_abs = 1 - np.exp(-scint_ext)
    detected = np.zeros_like(sample_thicknesses)
    for i in range(sample_thicknesses.size):
        trans = np.exp(-sample_ext * sample_thicknesses[i])
        detected[i] = scipy.integrate.simpson(power * trans * scint_abs, x=energies)
    absorbed = scipy.integrate.simpson(power * scint_abs, x=energies)
    eff_trans = detected / absorbed
    mask = eff_trans > THRESHOLD_TRANS
    usable_trans = eff_trans[mask]
    usable_thick = sample_thicknesses[mask]
    usable_ext = -np.log(usable_trans)
    inds = np.argsort(usable_ext)
    return usable_ext[inds], usable_thick[inds]


def compute_interp_values(spectra):
    angles, cal_curve = [], []
    centerline = None
    for angle in sorted(spectra.keys()):
        angles.append(float(angle))
        energies, power = spectra[angle]
        filtered = apply_filters(energies, power)
        iv = find_interp_values_one_angle(energies, filtered)
        if angle == 0:
            centerline = iv
        cal_curve.append(np.interp(REF_TRANS, iv[0], iv[1]))
    cal_curve = np.array(cal_curve)
    cal_curve = cal_curve / cal_curve[0]
    return centerline, (np.array(angles), cal_curve)


def find_angles(flat, pixel_size, d_source):
    vertical_slice = np.sum(flat, axis=1, dtype=np.float64)
    g = scipy.signal.windows.gaussian(200, 20)
    filtered = scipy.signal.convolve(vertical_slice, g, mode="same")
    center_row = float(np.argmax(filtered))
    angles = np.abs(np.arange(flat.shape[0]) - center_row)
    angles *= pixel_size / d_source
    return angles


def main():
    spectra = read_spectra()
    centerline, angular = compute_interp_values(spectra)
    centerline_ext, centerline_path = centerline
    angular_angles, angular_corr = angular

    # Synthetic flat: a bright vertical band so the fan centre is well defined.
    nrows, ncols = 64, 8
    rows = np.arange(nrows)
    flat = (1000.0 * np.exp(-((rows - 30.0) ** 2) / (2 * 8.0 ** 2)))[:, None]
    flat = np.repeat(flat, ncols, axis=1).astype(np.float64)
    flat += 5.0  # small pedestal
    row_angles = find_angles(flat, PIXEL_SIZE_UM, D_SOURCE_M)

    # Synthetic minus-log projection chunk over detector rows [start, end).
    start_row, end_row = 28, 36
    rng = np.random.default_rng(7)
    nproj = 3
    data = rng.uniform(0.05, 0.6, size=(nproj, end_row - start_row, ncols)).astype(np.float64)

    # correct_centerline: np.interp(data, ext, path).
    out = np.interp(data, centerline_ext, centerline_path)
    # correct_angle: per-row *= interp(angle, angular_angles, angular_corr).
    current = np.arange(start_row, end_row)
    corr = np.interp(row_angles[current], angular_angles, angular_corr)
    for i in range(corr.shape[0]):
        out[:, i, :] = out[:, i, :] * corr[i]

    np.save(os.path.join(OUT, "bh_centerline_ext.npy"), centerline_ext.astype(np.float64))
    np.save(os.path.join(OUT, "bh_centerline_path.npy"), centerline_path.astype(np.float64))
    np.save(os.path.join(OUT, "bh_angular_angles.npy"), angular_angles.astype(np.float64))
    np.save(os.path.join(OUT, "bh_angular_corr.npy"), angular_corr.astype(np.float64))
    np.save(os.path.join(OUT, "bh_row_angles.npy"), row_angles.astype(np.float64))
    np.save(os.path.join(OUT, "bh_flat.npy"), flat.astype(np.float32))
    np.save(os.path.join(OUT, "bh_data_in.npy"), data.astype(np.float32))
    np.save(os.path.join(OUT, "bh_data_out.npy"), out.astype(np.float32))

    print("beamhardening golden written:")
    print("  centerline LUT points:", centerline_ext.size)
    print("  angular angles:", angular_angles.tolist())
    print("  angular corr:", np.round(angular_corr, 6).tolist())
    print("  center row:", float(np.argmax(scipy.signal.convolve(
        np.sum(flat, axis=1, dtype=np.float64),
        scipy.signal.windows.gaussian(200, 20), mode="same"))))
    print("  data_out range:", float(out.min()), float(out.max()))


if __name__ == "__main__":
    main()
