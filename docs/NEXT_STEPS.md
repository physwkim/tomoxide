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

- ✅ **HDF5 reader done** — `open_dxchange` (`crates/tomoxide-io/src/lib.rs`)
  via the pure-Rust `rust-hdf5` crate (no libhdf5/C dep). Reads DXchange
  `/exchange/{data,data_white,data_dark,theta}`, casts any on-disk numeric
  dtype to f32, converts theta degrees→radians (or linspace fallback).
  Bit-exact parity test against a gzip-compressed uint16 fixture
  (`tools/gen_dxchange_fixture.py`).
- ✅ **TIFF writer done** — `create_writer(.., SaveFormat::Tiff)` via the
  pure-Rust `tiff` crate (no native libtiff). Per-slice 32-bit-float TIFF
  `{prefix}_{i:05}.tiff` (tomocupy `dataio/writer.py:281`). Bit-exact
  round-trip test; output verified readable by Python tifffile. Both I/O
  bookends are now closed.
- ✅ **HDF5 writer done** — `create_writer(.., SaveFormat::H5)` via pure-Rust
  `rust-hdf5` (no libhdf5). Single `{base}.h5` with one contiguous
  `/exchange/data` f32 `[nz,ny,nx]` dataset + `axes`/`description`/`units`
  attrs (tomocupy `dataio/writer.py` `h5nolinks`); chunks fill `[start,end)`
  via HDF5 hyperslab. Bit-exact round-trip (`tests/h5_write.rs`) + verified
  readable by reference libhdf5 (h5py).
- **Remaining stub:** `create_writer` for `Zarr` output — tomocupy
  `dataio/writer.py:294`. Needs a zarr crate (new dependency, sign-off);
  lower priority (TIFF + H5 cover the M3 pipeline).

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
- ✅ **`write_center` — done.** Port of tomopy `rotation.py:438`: reconstruct the
  `ind`-th slice (default `n_rows/2`) with gridrec across a range of rotation
  centers (`cen_range`, numpy-`arange` semantics; default
  `arange(ncol/2−5, ncol/2+5, 0.5)`), optionally `ratio`-circular-masked, returned
  as a `[len(centers), n, n]` stack + the center values (the I/O-free core, so
  `tomoxide-recon` stays `tomoxide-core`-only; persist as `{center:.2f}.tiff` via
  `tomoxide-io` to mirror tomopy's files). Parity scope: the **center enumeration**
  is held to tomopy exactly (Δ=0 vs a numpy golden, both default and explicit
  range); the reconstruction *content* is gridrec-*family* (Kaiser–Bessel kernel +
  ramp, not tomopy's PSWF + parzen), so the slice pixels are self-consistent
  gridrec reconstructions, not bit-identical to tomopy — validated against an
  independent `recon(Gridrec)` (Δ=0) plus the mask geometry. `write_center_parity.rs`,
  golden from `tools/gen_tomopy_write_center_golden.py`.
- **Remaining stub:** `crates/tomoxide-recon/src/center.rs` — `find_center_sift`
  (defer to M7, needs SIFT/AI; tomocupy `find_center.py:99`).

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
- ✅ **Fw (Fourier-Wavelet) — done.** Port of tomopy `prep/stripe.py:88`
  `_remove_stripe_fw` (Münch 2009): per slice pad the projection axis to
  `nproj + nproj/8`, run a `level`-deep db5 2-D wavelet decomposition, damp the
  vertical-detail bands along the projection axis in Fourier space, reconstruct,
  and crop back. `level=None` → `ceil(log2(max(nproj, nrows, ncol)))`; `pad`
  always on, matching tomopy defaults. The db5 dwt2/idwt2 are **hand-ported** (no
  new dependency) in `crates/tomoxide-prep/src/wavelet.rs`, with the pywt
  `symmetric` convention reverse-engineered and unit-tested against pywt 1.8.0 to
  the f64 floor. The forward decomposition mirrors tomopy's float32 pywt path
  (each band rounded to f32) while damping + reconstruction run in f64 (numpy/pywt
  promotion), so it matches tomopy 1.15.3 to the **f32 round-off floor** (max rel
  Δ≈1.2e-6) — `stripe_fw_parity.rs`, golden from
  `tools/gen_tomopy_stripe_fw_golden.py`. The Münch damping uses a self-contained
  `O(n log n)` FFT (radix-2 + Bluestein for arbitrary length, no FFT dependency)
  in `crates/tomoxide-prep/src/fft.rs`, validated against a naive DFT to the f64
  floor.
- ✅ **VoSort (sorting-based) — done.** Port of tomopy `prep/stripe.py:363`
  `remove_stripe_based_sorting` (Vo 2018 algorithm 3, for partial stripes): per
  sinogram slice `_rs_sort` — argsort each detector column's values over
  projections, median-smooth the sorted matrix, unsort. The median is a pure
  rank-filter **selection** of an existing f32 value (no arithmetic), so it matches
  tomopy 1.15.3 **bit-for-bit (Δ=0)** on tie-free columns for both `dim=1`
  (footprint `(size,1)`) and `dim=2` (`(size,size)`); `size=None` → tomopy default
  `max(5, ⌊0.01·ncol⌋)`. `StripeMethod::VoSort { size, dim }`; the `rs_sort`
  scaffold (sort/perm/unsort) was made smoother-pluggable and is shared with
  `VoAll` (unchanged, still passing). `stripe_vosort_parity.rs`, golden from the
  **real tomopy** `tools/gen_tomopy_stripe_vosort_golden.py`.
- ✅ **VoFilter (filtering-based) — done.** Port of tomopy `prep/stripe.py:437`
  `remove_stripe_based_filtering` (Vo 2018 algorithm 2): per sinogram slice
  `_rs_filter` separates a low-pass (smooth) component with a Gaussian Fourier
  filter along the projection axis (`real(ifft(fft(col·listsign)·window)·listsign)`,
  reflect-padded), runs the `_rs_sort` correction on that component, then adds back
  the high-pass residual. New pieces: `scipy.signal.windows.gaussian` (closed-form
  `exp(-n²/2σ²)`), the `(-1)^n` listsign modulation, and `np.pad` mode=`reflect`
  (whole-sample symmetric — distinct from scipy.ndimage `reflect`). The Fourier
  core reuses the self-contained f64 column FFT in `fft.rs` and the inner sort
  reuses the `rs_sort`/`median_filter_2d` scaffolds from `VoSort`. tomopy runs the
  filter in float64 then casts to f32, so it is held to the **f32 round-off floor**
  like the Fourier-Wavelet path (measured Δ=0 for these fixtures, `dim=1` sigma=3 &
  `dim=2` sigma=5). `StripeMethod::VoFilter { sigma, size, dim }`;
  `stripe_vofilter_parity.rs`, golden from the **real tomopy**
  `tools/gen_tomopy_stripe_vofilter_golden.py`.
- **Done (each) =** inject a synthetic stripe into a sinogram; the chosen method
  reduces the column-variance of the stripe by a stated factor without blurring
  legitimate features; reconstruction shows fewer ring artifacts (roughness over
  a flat annulus drops).

### B4. Phase retrieval — `tomoxide-prep::phase`

- ✅ **Paganin — done.** FFT-domain `1/(λ·dist·w2/(4π)+α)` low-pass on
  power-of-2-padded radiographs (reuses `Fft`); matches tomopy 1.15.3 to the f32
  round-off floor (max relative Δ ≈ 2.4e-7), `phase_parity.rs` golden from
  `tools/gen_tomopy_phase_golden.py`.
- ✅ **GPaganin (generalized Paganin) — done.** Port of tomocupy
  `retrieve_phase.paganin_filter(method='Gpaganin')` (Paganin et al. 2020): same
  single-step Fourier retrieval as Paganin but with a `cos`-based reciprocal grid
  `kf = cos(ix·2π·ps) + cos(iy·2π·ps)` and filter `1/(1 − (2·aph/W²)·(kf − 2))`,
  `aph = db·dist·λ/(4π)`, parameterized by `db`/`W` instead of `alpha`. The
  shared pad/fftshift/normalize/ifft/crop driver is factored as `run_phase`;
  Paganin and GPaganin are thin filter closures over it. The filter is
  ill-conditioned (`scale ≈ 1.2e3`), so the grid/filter are evaluated in f32 to
  mirror cupy's single precision — matching the reference to the **f32 round-off
  floor** (max rel Δ≈4.9e-7) where an f64 evaluation diverged ~25×.
  `gpaganin_parity.rs`, golden from `tools/gen_tomocupy_gpaganin_golden.py` (a
  faithful CPU/`scipy.fft` single-precision transcription of tomocupy's exact
  functions, since tomocupy needs a GPU).
- ✅ **Farago — done.** Port of tomocupy `retrieve_phase.farago_filter`
  (Farago 2024): the same padded-Fourier driver (`run_phase`) as Paganin but with
  the filter `1/(cos θ + db·sin θ)`, `θ = π·λ·dist·(ix² + iy²)` over the **squared**
  reciprocal grid (`_reciprocal_grid` + `_farago_filter_factor`). `run_phase` now
  takes a filter-builder closure; the f64-grid Paganin family goes through a
  `pointwise_filter` adapter (bit-identical to the prior inline path) while Farago
  builds its grid directly in f32. The filter is f32-sensitive — `db ≈ 1e3`
  multiplies `sin θ` so a 1-ULP error in `θ` is amplified ~1e3× — so the grid is
  built from the **exact** f32 reciprocal coordinate (numpy/cupy round the
  `0.5/((n−1)·ps)` scale to f32 *before* the multiply, NEP50 weak scalar;
  `reciprocal_coord_f32`). An f64 grid cast down diverges ~1e-3 (verified); the
  exact-f32 grid makes the filter **bit-identical** to numpy's, leaving only the
  single-precision FFT residual → **f32 round-off floor** (max rel Δ≈4.6e-7).
  `farago_parity.rs`, golden from `tools/gen_tomocupy_farago_golden.py` (faithful
  CPU/`scipy.fft` single-precision transcription of tomocupy's exact functions).
- **Phase family complete:** Paganin, GPaganin, and Farago are all ported; no
  phase-retrieval stubs remain.

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
- ✅ **Background normalization** — `normalize::normalize_bg` (tomopy
  `prep/normalize.py:207` → `libtomo/prep/prep.c::normalize_bg`). Done. Per
  projection row the mean of the `air` left- and right-boundary pixels defines an
  air baseline linearly interpolated across the detector-column axis; every pixel
  is divided by its local baseline (non-positive means clamped to `1`). f32 in the
  upstream accumulation order, with `f32::mul_add` for the baseline
  `air_left + air_slope·j` (a single C statement clang contracts to a fused
  multiply-add under the default `-ffp-contract=on`), so it matches tomopy 1.15.3
  **bit-for-bit (Δ=0)** for both `air=1` (default) and `air=4`.
  `normalize_bg_parity.rs`, golden from the **real tomopy**
  `tools/gen_tomopy_normalize_bg_golden.py`.
- ✅ **360→180 folding** — `morph::sino_360_to_180` (tomopy `misc/morph.py`). Done.
  New `prep::morph` module: the first `n=dx/2` projections (0–180°) are kept and
  the next `n` (180–360°) are column-reversed and stitched on to widen the
  detector, overlapping by `overlap` columns with a linear cross-fade;
  `Rotation::{Left,Right}` selects which half lands on which side. Direct regions
  are exact f32 copies and the seam blend is computed in f64 (numpy `float64`
  linspace weights · `float32` data) then cast to f32, with a faithful numpy
  `linspace`, so it matches tomopy 1.15.3 **bit-for-bit (Δ=0)** for both rotations
  and `overlap=0/4`. `sino360_parity.rs`, golden from the **real tomopy**
  `tools/gen_tomopy_sino360_golden.py`.

**M3 done =** `open_dxchange → normalize/minus_log → remove_stripe → find_center_vo
→ fbp → TIFF out` runs end-to-end on a checked-in small dataset, asserted by a
pipeline integration test. ✅ **Done** — `crates/tomoxide/tests/pipeline_e2e.rs`
(`m3_pipeline_hdf_to_tiff`) wires the whole chain on one DXchange fixture:
`find_center_vo = 63.500` (Δ=0 vs the Vo golden), FBP recovery vs the phantom
Pearson `r = 0.8727`, TIFF round-trip bit-exact. The `.h5` fixture is gitignored
sample data, regenerated by `tools/gen_dxchange_pipeline_fixture.py`.

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
3. ✅ **B1 TIFF writer** — done (`create_writer`, pure-Rust `tiff`, per-slice
   f32, bit-exact). Any reconstruction is now saveable.
4. ✅ **B1 HDF5 reader** — done (`open_dxchange`, pure-Rust `rust-hdf5`,
   bit-exact). Real data in; both I/O bookends are closed.
5. ✅ **B5 rank filters** + ✅ **B3 stripe Sf** + ✅ **B6 ring** + ✅ **B3 stripe
   Vo-all** + ✅ **B3 stripe Ti** + ✅ **B3 stripe Fw** — done (tomopy parity;
   bit-exact for rank/Sf/ring, ≈f32 floor for Vo-all/Ti/Fw). Fw hand-ports the
   db5 wavelet (no new dependency). The B3 stripe family is complete.
6. ✅ **M3 end-to-end pipeline integration test** — done
   (`tests/pipeline_e2e.rs`; HDF in → preprocess → center 63.500 → FBP r=0.8727
   → TIFF out, bit-exact). Closes the M3 end-to-end gate.
7. ✅ **B7 sim noise** — done (distribution parity). Remaining B7: beam
   hardening (needs material/spectrum config). Then M4+.

Each step is one commit + one test, full-workspace pass before any push, and
push only on explicit confirmation.
