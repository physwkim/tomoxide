#!/usr/bin/env python
"""Generate golden data for the SIFT center-finding parity test.

Reproduces tomocupy `find_center.py::_register_shift_sift` (cv2 SIFT +
BFMatcher knn + Lowe ratio test) and the center formula `ncol/2 - shift_x/2`.
The Rust port (`tomoxide-recon::center::find_center_sift`, `sift-center`
feature) links the SAME OpenCV (conda libopencv, matching this env's cv2), so
the uint8 images, keypoints, matches, and center match to the f32 floor.

Saves the float inputs, the intermediate uint8 images (to unit-test the
histogram-based normalization in isolation), and the golden shifts + center.

Run in the cv4 env (cv2 == the libopencv the Rust crate links):
    micromamba run -n cv4 python tools/gen_sift_center_golden.py
"""
import os

import numpy as np
import cv2

OUT = os.path.join(os.path.dirname(__file__), "..", "crates", "tomoxide", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

TH = 0.5  # Lowe ratio threshold (tomocupy default)


def _find_min_max(data):
    mmin = np.zeros(data.shape[0], dtype="float32")
    mmax = np.zeros(data.shape[0], dtype="float32")
    for k in range(data.shape[0]):
        h, e = np.histogram(data[k][:], 1000)
        stend = np.where(h > np.max(h) * 0.005)
        st = stend[0][0]
        end = stend[0][-1]
        mmin[k] = e[st]
        mmax[k] = e[end + 1]
    return mmin, mmax


def _to_uint8(img, mmin, mmax):
    tmp = (img - mmin) / (mmax - mmin) * 255
    tmp[tmp > 255] = 255
    tmp[tmp < 0] = 0
    return tmp.astype("uint8")


def register_shift_sift(datap1, datap2, th=0.5):
    mmin, mmax = _find_min_max(datap1)
    sift = cv2.SIFT_create()
    shifts = np.zeros([datap1.shape[0], 2], dtype="float32")
    u1_all, u2_all = [], []
    ngood = 0
    for idx in range(datap1.shape[0]):
        tmp1 = _to_uint8(datap2[idx], mmin[idx], mmax[idx])
        tmp2 = _to_uint8(datap1[idx], mmin[idx], mmax[idx])
        u1_all.append(tmp1)
        u2_all.append(tmp2)
        kp1, des1 = sift.detectAndCompute(tmp1, None)
        kp2, des2 = sift.detectAndCompute(tmp2, None)
        match = cv2.BFMatcher()
        matches = match.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < th * n.distance]
        ngood = len(good)
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        shift = (src - dst)[:, 0, :]
        shifts[idx] = np.mean(shift, axis=0)[::-1]
    return shifts, ngood, np.array(u1_all), np.array(u2_all)


def make_textured(rng, ny, nx, n):
    """Stack of textured images (sum of random Gaussian blobs) — plenty of
    SIFT-detectable corners/blobs."""
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    out = np.zeros((n, ny, nx), dtype=np.float32)
    for k in range(n):
        img = np.zeros((ny, nx), dtype=np.float32)
        for _ in range(40):
            cy, cx = rng.uniform(8, ny - 8), rng.uniform(8, nx - 8)
            s = rng.uniform(2.0, 5.0)
            amp = rng.uniform(0.4, 1.0)
            img += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s)))
        out[k] = img
    return out


def main():
    rng = np.random.default_rng(2024)
    ny, nx, n = 96, 96, 2
    datap1 = make_textured(rng, ny, nx, n)
    # datap2 = datap1 shifted by a known (dy, dx); SIFT should recover it.
    dy, dx = 2, 6
    datap2 = np.empty_like(datap1)
    for k in range(n):
        datap2[k] = np.roll(np.roll(datap1[k], dy, axis=0), dx, axis=1)

    shifts, ngood, u1, u2 = register_shift_sift(datap1, datap2, TH)
    ncol = nx
    centers = ncol / 2 - shifts[:, 1] / 2
    center = float(np.mean(centers))

    np.save(os.path.join(OUT, "sift_datap1.npy"), datap1)
    np.save(os.path.join(OUT, "sift_datap2.npy"), datap2)
    np.save(os.path.join(OUT, "sift_u1.npy"), u1)
    np.save(os.path.join(OUT, "sift_u2.npy"), u2)
    np.save(os.path.join(OUT, "sift_shifts.npy"), shifts.astype("float32"))
    np.save(os.path.join(OUT, "sift_center.npy"), np.array([center], dtype="float32"))

    print("sift golden written:")
    print("  shape:", datap1.shape, "good matches:", ngood)
    print("  shifts (dy,dx):\n", shifts)
    print("  center:", center)


if __name__ == "__main__":
    main()
