# tomoxide — Next Steps

Actionable backlog after the M2 scalar iterative family. This is the working
to-do list; the milestone framing lives in [`ROADMAP.md`](ROADMAP.md) and the
per-module upstream map in [`PORTING.md`](PORTING.md). Every task below cites the
exact stub `file:line` in this repo and the upstream reference to port from.

_Status as of 2026-06-15: M0 (scaffold), M1 (FBP/gridrec, tomopy-verified),
and the M2 **scalar** iterative family (sirt, mlem, osem, pml/ospml quad &
hybrid, grad, tikh, tv, art, bart) are complete and pushed to `origin/main`
(`679828e`)._

---

## Conventions for every task here

Each task is its own commit with a round-trip / parity test, the same way the
M2 family was done. Before declaring any one done:

- `cargo fmt --all`, then `cargo clippy -p <crate> --all-targets -- -D warnings`
  and `cargo nextest run -p <crate>` for the touched crate(s); escalate to
  `--workspace` on cross-crate API changes and before any push.
- **Define "done" as a verifiable check** (stated per task below), not prose.
- **Projector-model caveat carries forward.** Anything compared to tomopy
  numerically inherits the linear-interp-vs-Siddon gap (see PORTING). Prefer
  self-consistency / golden-data parity; `gridrec` is the only model-independent
  numeric reference so far.
- New third-party dependencies (HDF5, TIFF, FFT-of-real helpers) are **not**
  added without asking first.

---

## Option A — Finish M2: vector tomography (deferred)

The only remaining M2 method. Out of scope of the scalar `recon()` contract:
it takes **multiple** tilt datasets in and returns a **vector field** out.

- **Stub:** `crates/tomoxide-recon/src/lib.rs:141` (the `_ =>` arm of the
  iterative dispatch) — `vector` / `vector2` / `vector3`.
- **Upstream:** tomopy `libtomo/recon/vector.c` (`vector`, `vector2`, `vector3`).
- **Blocker / needs sign-off:** a separate API surface (multi-dataset in,
  `Vec`-of-`Volume` or vector-field out) outside `recon()`. Decide the public
  shape before coding — this is an architectural addition, not a drop-in arm.
- **Done =** reconstruct a synthetic vector phantom from ≥2 tilt series; each
  component round-trips to the known field within tolerance.

Niche relative to Option B; recommended only if there is a concrete vector-data
consumer.

---

## Option B — M3: Preprocessing & center finding (CPU)  ← recommended

This is the milestone that makes tomoxide usable on **real** data. ROADMAP goal:
_a full CPU pipeline: HDF in → preprocess → center → FBP → TIFF out._ Ordered
below by dependency and value — the first three close the end-to-end pipeline.

### B1. I/O bookends — `tomoxide-io`  (unblocks real data in/out)

- **Stubs:** `crates/tomoxide-io/src/lib.rs:57` `open_dxchange` (HDF5 reader),
  `:65` `create_writer` (TIFF / HDF5 / Zarr).
- **Upstream:** tomocupy `dataio/reader.py:59`, `dataio/writer.py:103`.
- **Needs sign-off (dependency):** HDF5 binding crate (`hdf5`/`hdf5-metno`) and
  a TIFF crate (`tiff`). Pick TIFF-writer first (no native lib, pure Rust) so a
  reconstruction can be saved before the HDF5 reader lands.
- **Done =** write a `Volume` to per-slice TIFF and read it back bit-equal;
  open a checked-in small DXchange `.h5` and recover the known data/theta shapes.

### B2. Center finding — `tomoxide-recon::center`  (unblocks correct recon)

- ✅ **`find_center_vo` (the workhorse) — done.** Sinogram-domain Vo method,
  matches tomopy 1.15.3 exactly (Δ = 0) on 4 parity cases
  (`center_parity.rs`, golden from `tools/gen_tomopy_center_golden.py`).
- ✅ **`find_center_pc` — done.** Phase-correlation of the 0°/mirrored-180°
  pair: a port of skimage `phase_cross_correlation` (`normalization="phase"`,
  `upsample_factor = 1/tol`) — forward FFTs, phase-normalized cross-power
  spectrum, whole-pixel argmax, then a 3×3 matrix-multiply upsampled-DFT subpixel
  refinement. Projector-independent and (with tol=0.5) quantized to a quarter-
  pixel center, so it matches tomopy 1.15.3 **exactly (Δ = 0)** on 4 cases
  including two subpixel (`center_pc_parity.rs`, golden from
  `tools/gen_tomopy_center_pc_golden.py`). The `rotc_guess` pre-alignment
  (`ndimage.shift`) is not yet ported — `Some(_)` returns `NotImplemented`.
- ✅ **`find_center` — done.** Entropy + Nelder-Mead (`rotation.py:82`):
  reconstructs a slice with gridrec at candidate centers and minimises the masked
  reconstruction's 64-bin histogram entropy with a faithful scalar Nelder-Mead
  (validated to reproduce scipy's result exactly on tomopy's own cost). It goes
  *through* the projector (gridrec), so it is held to recovery, not bit parity:
  it lands on the true axis (`find_center_vo`) within ±0.5 px and agrees with
  tomopy's `find_center` within ±1 px (`center_entropy_parity.rs`, golden from
  `tools/gen_tomopy_center_entropy_golden.py`). Surfaced and fixed a latent
  gridrec defect — the Fourier recentering shift keyed off the raw FFT bin index
  rather than the signed frequency, collapsing reconstructions at sub-pixel
  centers (invisible at the integer default center; `gridrec_subpixel_center.rs`
  regresses it), bit-identical at integer centers.
- **Remaining stubs:** `crates/tomoxide-recon/src/center.rs` —
  `write_center` (`rotation.py:438`), `find_center_sift` (defer to M7, needs
  SIFT/AI; tomocupy `find_center.py:99`).

### B3. Stripe removal — `tomoxide-prep::stripe`  (ring-artifact prevention)

- ✅ **Sf (smoothing-filter) — done.** Direct port of tomopy
  `libtomo/prep/stripe.c::remove_stripe_sf` (per-slice column-mean over angles →
  clamp-to-edge width-`size` moving average → subtract the residual). Same-order
  f32 arithmetic, so it matches tomopy 1.15.3 **bit-for-bit** on size 3/5
  (`stripe_sf_parity.rs`, golden from `tools/gen_tomopy_stripe_sf_golden.py`).
- ✅ **VoAll (Vo all-stripe) — done.** Port of tomopy `prep/stripe.py:843`
  `remove_all_stripe` (Vo algorithms 3+5+6): per slice `_rs_dead` (uniform-filter
  fluctuation detection → bilinear `kx=ky=1` RectBivariateSpline fill of dead
  columns → `_rs_large` rank-smoothing of large stripes) then `_rs_sort`
  (argsort-per-column → median-across-columns → unsort). Composes scipy
  primitives (uniform_filter1d, median_filter, polyfit, RectBivariateSpline) over
  distinct-valued columns, so it matches tomopy 1.15.3 to the **f32 round-off
  floor** (max rel Δ≈5.8e-7) on 2 cases — snr=3 (large+sort) and snr=2
  (adds the dead-column fill path) — `stripe_voall_parity.rs`, golden from
  `tools/gen_tomopy_stripe_voall_golden.py`. Exact-tie columns are deliberately
  avoided in the fixture: argsort tie order is numpy-quicksort-defined (not
  portable), so a perfectly constant column is outside the well-defined parity
  domain; the injected dead column is a strictly monotonic near-flat ramp.
- ✅ **Ti (Titarenko/Miqueles) — done.** Port of tomopy `prep/stripe.py:179`
  `remove_stripe_ti`: per slice solve a finite-difference normal-equations system
  by conjugate gradient (f64) for the per-detector-column offset, then combine
  the first/second-difference corrected sinograms as `sqrt(d1·d2 + β·|min|)`,
  rounding each `_ring` to f32. Reproduces the f64 CG + f32 cast in the upstream
  op order, so it matches tomopy 1.15.3 to the **f32 round-off floor**
  (max rel Δ≈5.2e-7) — `stripe_ti_parity.rs`, golden from
  `tools/gen_tomopy_stripe_ti_golden.py`. Only the default `nblock=0`
  (whole-sinogram) path is supported/verified: tomopy's block path `_ringb`
  (nblock>0) is unrunnable on modern numpy (its NaN guard
  `np.where(np.isnan(...) is True)` raises), so there is no reference output —
  tomoxide returns `NotImplemented` for nblock>0 rather than guessing.
- **Remaining stub:** `crates/tomoxide-prep/src/stripe.rs` — Fw (Fourier-Wavelet).
- **Upstream:** tomopy `prep/stripe.py:88` (Fw). Fw needs a 1-D wavelet
  (db-family) — check if a Rust crate is acceptable or port the lifting scheme
  (sign-off on the dependency).
- **Done (each) =** inject a synthetic stripe into a sinogram; the chosen method
  reduces the column-variance of the stripe by a stated factor without blurring
  legitimate features; reconstruction shows fewer ring artifacts (roughness over
  a flat annulus drops).

### B4. Phase retrieval — `tomoxide-prep::phase`

- ✅ **Paganin — done.** FFT-domain `1/(λ·dist·w2/(4π)+α)` low-pass on
  power-of-2-padded radiographs (reuses `Fft`); matches tomopy 1.15.3 to the f32
  round-off floor (max relative Δ ≈ 2.4e-7), `phase_parity.rs` golden from
  `tools/gen_tomopy_phase_golden.py`.
- **Remaining stubs:** `crates/tomoxide-prep/src/phase.rs` — `GPaganin`
  (tomocupy generalized Paganin), `Farago` (tomocupy
  `retrieve_phase.farago_filter:110`).

### B5. Rank filters — `CpuBackend: RankFilter`  (completes the prep family)

- ✅ **`median3d` + `remove_outlier` — done.** Direct port of tomopy
  `median_filt3d.c::medfilt3D_float` (clamp-to-center boundary, `(2·radius+1)³`
  window, sorted median at `total/2`; one uniform rule covers the pure-median
  and dezinger-threshold paths). Matches tomopy 1.15.3 **bit-for-bit** on 4
  parity cases — median size 3/5 and dezinger dif 0.5/5.0 (`rankfilter_parity.rs`,
  golden from `tools/gen_tomopy_rankfilter_golden.py`).
- **Upstream:** tomopy `libtomo/misc/median_filt3d.c`;
  `misc/corr.py:355,413` (`median_filter3d`, `remove_outlier3d`).

### B6. Ring removal — `tomoxide-recon::ring`

- ✅ **`remove_ring` — done (both `int_mode`).** Full port of tomopy
  `libtomo/misc/remove_ring.c` (polar transform → 3-band radial median →
  subtract/threshold → 3-band azimuthal mean → inverse transform → subtract).
  The exact float/double cast chain plus the shared libm make it **bit-for-bit**
  with tomopy 1.15.3 (Δ = 0) on rwidth 2/4 for both `int_mode` values —
  `WRAP` (default, cyclic azimuth) and `REFLECT` (each polar half mirrored at
  its 0/π and π/2π edges, via `RingIntMode`); `ring_parity.rs`, golden from
  `tools/gen_tomopy_ring_golden.py`.

### B7. Lower-priority polish (M3 tail)

- **Beam hardening** — `crates/tomoxide-prep/src/hardening.rs:11`
  `beam_correct`; tomocupy `processing/external/hardening.py:50`. Needs
  material/spectrum config; defer unless a dataset needs it.
- ✅ **Sim noise** — `add_gaussian` / `add_poisson` (`crates/tomoxide-sim/src/noise.rs`;
  tomopy `sim/project.py:110,136`). Done. Distribution parity (matched moments),
  not Δ=0: numpy's MT19937 stream is not reproducible from Rust. Self-contained
  seeded SplitMix64 (no `rand` dep); Poisson ports numpy's Knuth-mult / Hörmann
  PTRS selection. Tested by moments incl. Poisson skewness in
  `tests/noise_stats.rs`.

**M3 done =** `open_dxchange → normalize/minus_log → remove_stripe → find_center_vo
→ fbp → TIFF out` runs end-to-end on a checked-in small dataset, asserted by a
pipeline integration test.

---

## After M3 (context, not yet actionable)

- **M4 — CUDA backend** (parity: tomocupy): C-ABI shim over `cfunc_*`, `nvcc`
  gated on the `cuda` feature. `crates/tomoxide-cuda` currently advertises the
  device but has no compute path.
- **M5 — Streaming pipeline:** `crates/tomoxide/src/pipeline.rs:60`
  `ReconSteps::run` (tomocupy `rec_steps.py:116`). Chunking, double buffering,
  3-stage overlap.
- **M6 — wgpu/Metal:** WGSL ports of the FBP filter, backprojection,
  elementwise; runs the GPU path on Apple Silicon.
- **M7 — Laminography, beam hardening, AI center (`find_center_sift`), f16,
  zarr, benchmarks.**

---

## Suggested sequence

1. ✅ **B2 `find_center_vo`** — done (tomopy parity Δ=0).
2. ✅ **B4 Paganin** — done (tomopy parity, max rel Δ≈2.4e-7).
3. **B1 TIFF writer** — so any reconstruction is saveable (smallest, no native
   dep).
4. **B1 HDF5 reader** — real data in; closes the bookends.
5. ✅ **B5 rank filters** + ✅ **B3 stripe Sf** + ✅ **B6 ring** + ✅ **B3 stripe
   Vo-all** + ✅ **B3 stripe Ti** — done (tomopy parity; bit-exact for
   rank/Sf/ring, ≈f32 floor for Vo-all/Ti). Remaining artifact-correction
   family: **B3 Fw** (wavelet dependency, needs sign-off).
6. Wire the **M3 end-to-end pipeline integration test**.
7. ✅ **B7 sim noise** — done (distribution parity). Remaining B7: beam
   hardening (needs material/spectrum config). Then M4+.

Each step is one commit + one test, full-workspace pass before any push, and
push only on explicit confirmation.
