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
  `tools/gen_tomopy_center_pc_golden.py`). The `rotc_guess` pre-alignment is
  **done**: both projections are pre-shifted by `[0, -imgshift]`
  (`imgshift = rotc_guess - (ncol-1)/2`) through a line-faithful
  `scipy.ndimage.shift` (order-3 cubic spline, `mode='constant'`, `cval=0`,
  ported from scipy 1.17.1 — mirror-init prefilter + 16-tap separable resample,
  out-of-bounds centres → `cval`, mirror-reflected edge taps), and `imgshift`
  is added back. The isolated shift reproduces scipy **bit-for-bit (0 f32
  mismatches)** across fractional/integer/out-of-bounds cases
  (`tomoxide-recon` `ndimage_shift_matches_scipy`), and the end-to-end centers
  match tomopy on 4 `imgshift != 0` cases (`center_pc_parity.rs`, golden from
  `tools/gen_tomopy_center_pc_rotc_golden.py`). At `imgshift == 0` (default
  `None`) the spline shift is the f32 identity, so it is skipped (None path
  bit-for-bit unchanged).
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
- ✅ **VoLarge (large-stripe) — done.** Port of tomopy `prep/stripe.py:653`
  `remove_large_stripe` (Vo 2018 algorithm 5): per sinogram slice `_rs_large`
  sorts each detector column over projections, median-smooths the sorted profile
  along the column axis (footprint `(1, size)`), estimates a per-column intensity
  factor from the central rows (`drop_ratio` of the extremes dropped), detects the
  wide-stripe columns (`_detect_stripe` + 1-px binary dilation), and overwrites
  *only* those columns with the rank-smoothed profile mapped back through the
  (optionally intensity-normalised) sort order. Reuses the `rs_large` helper
  already shared with `VoAll` (unchanged). The rank-smoothed writes are pure f32
  selections, so `norm=false` matches tomopy **bit-for-bit (Δ=0)**; `norm=true`
  additionally divides the unmasked columns by their f32 factor → **f32 round-off
  floor** (max rel ≤ 1e-5). `StripeMethod::VoLarge { snr, size, drop_ratio, norm }`;
  `stripe_volarge_parity.rs`, golden from the **real tomopy**
  `tools/gen_tomopy_stripe_volarge_golden.py` (snr=3, size=51, drop_ratio=0.1).
- ✅ **VoDead (dead-stripe) — done.** Port of tomopy `prep/stripe.py:762`
  `remove_dead_stripe` (Vo 2018 algorithm 6): per sinogram slice `_rs_dead`
  smooths each detector column over projections (`uniform_filter1d` width 10),
  scores each column by its summed deviation from that smooth, detects the
  unresponsive/fluctuating columns (`_detect_stripe` + 1-px dilation, the two
  border columns never flagged), and fills the flagged columns by per-row linear
  interpolation across the good columns (the `kx=ky=1` `RectBivariateSpline`).
  When `norm` is set a residual `_rs_large` pass then removes wide stripes.
  **Structural change:** tomopy gates that residual pass on `norm`, but the
  `rs_dead` helper previously hardcoded it (it only served `VoAll`, always
  `norm=True`); a `norm` bool is now threaded through `rs_dead` and `VoAll`'s call
  site passes `true` explicitly, so `VoAll` stays bit-identical. The bilinear fill
  is arithmetic, so both cases hold to the **f32 round-off floor** (max rel ≤ 1e-5).
  `snr=2` fires the dead-column detection; the two cases differ structurally on
  the injected large stripes — `norm=true` removes them, `norm=false` leaves them
  bit-identical to the input (Δ=0 on those columns, asserted).
  `StripeMethod::VoDead { snr, size, norm }`; `stripe_vodead_parity.rs`, golden
  from the **real tomopy** `tools/gen_tomopy_stripe_vodead_golden.py`.
- ✅ **VoFit (fitting-based) — done.** Port of tomopy `prep/stripe.py:520`
  `remove_stripe_based_fitting` (Vo 2018 algorithm 1, for low-pass stripes): per
  `[proj, col]` slice `_rs_fit` divides the sinogram by its Savitzky–Golay
  polynomial fit along the projection axis, then re-multiplies by a mean-matched
  2-D Gaussian-smoothed copy of that fit (`_2d_filter`:
  `real(ifft2(fft2(matpad·matsign)·win2d)·matsign)`, `(-1)^(x+y)` modulation,
  edge-pad columns + mean-pad rows, crop). Two new self-contained primitives, **no
  new dependency**: (1) the Savitzky–Golay weights — scipy's `savgol_coeffs` is a
  min-norm `lstsq`, computed here from **scaled normal equations** (fit nodes
  mapped to ≈[-1, 1] so the tiny `(order+1)`-square solve is well-conditioned),
  reproducing scipy's SVD `lstsq` to the f64 floor (verified directly vs scipy);
  `mode='mirror'` reuses the existing whole-sample reflect. (2) the 2-D Fourier
  filter `fft::filter_real_2d` — the separable 2-D analogue of
  `filter_real_column` (fft2 rows-then-cols / ifft2 cols-then-rows over the f64
  complex FFT), unit-tested vs a naive 2-D DFT. The 2-D filter runs in f64, so it
  is held to the **f32 round-off floor** (max rel ≈ 2e-7) like Fw/VoFilter.
  `StripeMethod::VoFit { order, sigma }`; `stripe_vofit_parity.rs`, golden from
  the **real tomopy** `tools/gen_tomopy_stripe_vofit_golden.py` (order=3
  sigma=(5,20) and order=1 sigma=(3,10)).
- ✅ **stripes_detect3d (Kazantsev 2023) — done.** Port of tomopy
  `prep/stripe.py:984` `stripes_detect3d` / libtomo
  `stripes_detect3d.c::stripesdetect3d_main_float`: a 3-D stripe *detector* (not a
  remover) over the `[angle, detY, detX]` stack producing a `[0, 1]` weights
  volume whose smaller values mark stripe edges. Four full-volume passes — a
  6-stencil mean smoothing (kept only as the zero-gradient fallback of pass 3), a
  detX forward-difference gradient (step 2), a per-voxel ratio between the mean
  `|gradient|` in the angle×depth plate parallel to the stripe and the means
  orthogonal to it (smaller of left/right), and a vertical (along-angle) median
  filter with the C kernel's off-by-one median index. Pure f32, **no FFT and no
  new dependency**, so it is **bit-exact (Δ=0)** against tomopy. New `stripe3d`
  module; public API `prep::stripes_detect3d(&Tomo, size, radius) -> Array3<f32>`
  (it returns weights and does not mutate the data, so it is deliberately *not* a
  `StripeMethod` variant). `stripe_detect3d_parity.rs`, golden from the **real
  tomopy** `tools/gen_tomopy_stripe_detect3d_golden.py` (size=10/radius=3 defaults
  and size=5/radius=2), plus `[0,1]`-range and stripe-highlight structural checks.
- ✅ **stripes_mask3d (Kazantsev 2023) — done.** Port of tomopy
  `prep/stripe.py:1058` `stripes_mask3d` / libtomo
  `stripes_detect3d.c::stripesmask3d_main_float`: turns a `stripes_detect3d`
  weights volume into a binary `bool` stripe mask. Threshold the weights (`<=`
  candidate), then prune by stripe consistency in depth (drop deep features —
  stripes are shallow) and along-angle (drop short runs), drop stripes shorter
  than half the min length, and iteratively merge nearby stripes (one pass per
  unit of min width). Each pass reads the previous pass's *full* result and
  writes a fresh buffer, mirroring the C kernel's read/write `mask`/`Output`
  split, so neighbour reads never see a half-updated mask. Pure integer/bool
  logic with a single `f32` threshold compare and `(int)(0.01·sensitivity·len)`
  thresholds (computed in f32 to match the C `float`), so it is **bit-exact
  (Δ=0)** — the bool mask matches tomopy element-for-element. Public API
  `prep::stripes_mask3d(&Array3<f32> weights, threshold, min_stripe_length,
  min_stripe_depth, min_stripe_width, sensitivity_perc) -> Array3<bool>`.
  `stripe_mask3d_parity.rs`, golden from the **real tomopy**
  `tools/gen_tomopy_stripe_mask3d_golden.py` (defaults `0.6/20/10/5/85` and a
  looser `0.5/10/4/3/60`); the weights fixture is the real-tomopy
  `stripes_detect3d` output. This completes the 3-D stripe-family port.
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

- ✅ **`median3d` + `remove_outlier3d` — done.** Direct port of tomopy
  `median_filt3d.c::medfilt3D_float` (clamp-to-center boundary, `(2·radius+1)³`
  window, sorted median at `total/2`; one uniform rule covers the pure-median
  and dezinger-threshold paths). Matches tomopy 1.15.3 **bit-for-bit** on 4
  parity cases — median size 3/5 and dezinger dif 0.5/5.0 (`rankfilter_parity.rs`,
  golden from `tools/gen_tomopy_rankfilter_golden.py`). The facade fn and
  `RankFilter` trait method were renamed `remove_outlier` → `remove_outlier3d`
  (aligning to this docs name and the sibling `median3d`), freeing
  `remove_outlier` for the 2-D corr.py:559 port below.
- **Upstream:** tomopy `libtomo/misc/median_filt3d.c`;
  `misc/corr.py:355,413` (`median_filter3d`, `remove_outlier3d`).
- ✅ **`median_filter_nonfinite` — done.** Port of tomopy `misc/corr.py:281`
  (pure NumPy): replace each non-finite value (NaN/±inf) with the median of the
  finite values in its `size×size` neighbourhood along the last two axes. Per
  projection the medians read a snapshot taken before any write (tomopy's
  `projection_copy`), so adjacent bad pixels don't see each other's fixes; the
  window is clamped to the slice with bounds `i ± size/2` (even `size` →
  odd `2·(size/2)+1` width, as upstream); `np.median` on float32 returns float32
  (odd → middle order statistic, even → f32 mean of the two middles), and an
  all-non-finite kernel errors. Medians are order-free, so it matches tomopy
  1.15.3 **bit-for-bit (Δ=0)** for size 3/5 (`median_nonfinite_parity.rs`, golden
  from `tools/gen_tomopy_median_nonfinite_golden.py`). `prep::filters::median_filter_nonfinite`.
- ✅ **`adjust_range` — done.** Port of tomopy `misc/corr.py:90` (pure NumPy):
  clip a stack's values to `[dmin, dmax]`. `None` bounds default to the data
  min/max, and a bound is applied only when *strictly* tighter than the data
  range (strict `>`/`<`), so both-`None` and looser-than-data bounds are no-ops,
  exactly as upstream (incl. the post-high-clip `np.min` recompute). No
  summation → matches tomopy 1.15.3 **bit-for-bit (Δ=0)** across both-None /
  high-only / low-only / both / looser-than-data cases (`adjust_range_parity.rs`,
  golden from `tools/gen_tomopy_adjust_range_golden.py`). `prep::filters::adjust_range`.
- ✅ **`median_filter` — done.** Port of tomopy `misc/corr.py:167`: median-filter
  every 2-D slice along `axis` with a `size×size` footprint (scipy.ndimage
  default `mode='reflect'`, half-sample reflection — its index map verified
  against scipy `ni_support.c` `NI_EXTEND_REFLECT`). Every pixel is replaced by
  its local median (no threshold). scipy's median filter picks a single order
  statistic (rank `size·size/2`, never an average even for an even footprint), so
  the result is **bit-exact (Δ=0)** on finite input. Golden from real tomopy
  1.15.3 (this wrapper uses `arr[tuple(slc)]`, so unlike `remove_outlier1d` it
  runs unmodified on numpy 2.x). 6 cases (odd/even `size`, all three axes) pass at
  0 bit-mismatches (`median_filter_parity.rs`, golden from
  `tools/gen_tomopy_median_filter_golden.py`). `prep::filters::median_filter`.
  Distinct from `median_filter3d` (3-D cube, backend-routed) and
  `median_filter_nonfinite` (NaN/±inf scrub). It is also the 2-D median primitive
  (`median2d_reflect`) that the 2-D `remove_outlier` reuses with a dezinger
  threshold (below).
- ✅ **`remove_outlier` (2-D) — done.** Port of tomopy `misc/corr.py:559`: the
  axis-chunked 2-D dezinger. For each index along `axis` the orthogonal 2-D
  image's `size×size` median is taken (scipy.ndimage default `mode='reflect'`,
  the shared `median2d_reflect` primitive), then a pixel is replaced by that
  median only where `arr − median ≥ diff`. Single order statistic + plain f32
  `where` → **bit-exact (Δ=0)**. Naming was unblocked by renaming the 3-D cube
  dezinger `remove_outlier` → `remove_outlier3d` (a separate refactor commit);
  `remove_outlier` now matches tomopy's public name. Golden from real tomopy
  1.15.3 (uses `arr[tuple(slc)]`, runs on numpy 2.x). 6 cases (odd/even `size`,
  all three axes, `dif=0`) pass at 0 bit-mismatches (`remove_outlier_parity.rs`,
  golden from `tools/gen_tomopy_remove_outlier_golden.py`).
  `prep::filters::remove_outlier`. Distinct from `remove_outlier3d` (3-D cube)
  and `remove_outlier1d` (1-D mirror).
- ✅ **`remove_outlier1d` — done.** Port of tomopy `misc/corr.py:615`: 1-D
  `size`-tap median along `axis` (scipy.ndimage `mode='mirror'`, whole-sample
  reflection — its index map verified against scipy `ni_support.c`
  `NI_EXTEND_MIRROR`), then replace a pixel by that median only where
  `arr − median ≥ diff` (strict `<` keeps it). scipy's median filter picks a
  single order statistic (rank `size/2`, never an average even for even `size`)
  and the `where` test is a plain f32 subtraction, so the result is **bit-exact
  (Δ=0)** on finite input. The published tomopy 1.15.3 wrapper raises
  `IndexError` on numpy 2.x (corr.py:660 indexes `arr[slc]` with a *list*; the
  sibling `remove_outlier` already uses `arr[tuple(slc)]`), so the golden inlines
  tomopy's verbatim body with that single one-character compat fix — same
  chunking / dtype casts / scipy call / `ne.evaluate` `where`. 6 cases (odd/even
  `size`, all three axes, `dif=0`) pass at 0 bit-mismatches
  (`remove_outlier1d_parity.rs`, golden from
  `tools/gen_tomopy_remove_outlier1d_golden.py`). `prep::filters::remove_outlier1d`.
  Distinct from the 2-D `remove_outlier` (corr.py:559) and the 3-D cube
  `remove_outlier3d` (backend-routed).
- ✅ **`gaussian_filter` — done.** Port of tomopy `misc/corr.py:118`: separable
  Gaussian blur of every 2-D slice along `axis`. A faithful port of scipy.ndimage
  `gaussian_filter1d` + the C `NI_Correlate1D` (verified line-by-line against
  `scipy/ndimage/src/ni_filters.c`): the kernel is `exp(−x²/2σ²)` over
  `x = −lw..lw` with `lw = ⌊4σ+0.5⌋` (`truncate=4`), normalised by **numpy's f64
  pairwise sum** (new `pairwise_sum_f64`, the f64 analogue of `normalize`'s
  `pairwise_sum_f32`) then reversed for correlation; the convolution loads the
  line into an f64 buffer and accumulates in f64 with scipy's *exact* symmetric /
  anti-symmetric / general summation branch (selected by the same `DBL_EPSILON`
  test), `mode='reflect'` (half-sample) boundaries, and the **f32 intermediate**
  cast between the two separable passes; derivative `order ≥ 1` uses scipy's
  `q' + q·p'` kernel recurrence. The only possible divergence is the kernel `exp`
  (numpy's vectorised f64 `exp` vs libm, ≤1 ULP), so the precision class is the
  **f32 round-off floor** — but on the golden it realises **Δ=0** (0/12852 pixels
  differ) across sigma=3 order=0 on each axis, a small-radius sigma, and
  derivative orders 1 & 2. The `correlate1d_2d` branch reproduction is the shared
  primitive that makes the integer-weighted `sobel_filter` bit-exact.
  `gaussian_filter_parity.rs`, golden from the **real tomopy**
  `tools/gen_tomopy_gaussian_filter_golden.py`. `prep::filters::gaussian_filter`.
- ✅ **`sobel_filter` — done.** Port of tomopy `misc/corr.py:474`: scipy.ndimage's
  Sobel transform on every 2-D slice along `axis` — a `[−1,0,1]`
  central-difference correlation along the slice's last axis (the anti-symmetric
  branch of `correlate1d_2d`) then a `[1,2,1]` smoothing correlation along the
  other (the symmetric branch), both `mode='reflect'`. Reuses the f64
  `correlate1d_2d` primitive built for `gaussian_filter`; the weights are exact
  small integers and f32 inputs are exact in the f64 accumulator, so the result
  is **bit-exact (Δ=0)** (0/2970 across all three axes). The published tomopy
  1.15.3 wrapper cannot run — a bare `filters.sobel` (NameError; `corr.py` never
  binds `filters`) *and* the numpy-2.x `arr[slc]` list-index `IndexError` — so the
  golden inlines tomopy's verbatim body with exactly those two one-token compat
  fixes (`filters.sobel → scipy.ndimage.sobel`, `arr[slc] → arr[tuple(slc)]`),
  same dtype cast and per-slice scipy call. `sobel_filter_parity.rs`, golden from
  the **real tomopy** `tools/gen_tomopy_sobel_filter_golden.py`.
  `prep::filters::sobel_filter`.

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
- ✅ **Sim noise** — `add_gaussian` / `add_poisson` / `add_rings` /
  `add_salt_pepper` / `add_zingers` (`crates/tomoxide-sim/src/noise.rs`; tomopy
  `sim/project.py:110,136,153,183,211`). Done. Distribution parity (matched
  moments), not Δ=0: numpy's MT19937 stream is not reproducible from Rust.
  Self-contained seeded SplitMix64 (no `rand` dep); Poisson ports numpy's
  Knuth-mult / Hörmann PTRS selection. `add_rings` draws a fixed
  per-detector-pixel sensitivity `N(1, std)` broadcast across all angles (a
  ring); `add_zingers` saturates each element to `sat` with probability `f`;
  `add_salt_pepper` corrupts each element to `val` with probability `prob` (`<`,
  not `<=`), `val=None` → the original data max. Tested by moments incl. Poisson
  skewness, the add_rings constant-across-angles structural invariant, and the
  add_zingers / add_salt_pepper corrupted-fraction + default-val checks in
  `tests/noise_stats.rs`. The tomopy sim-noise family is complete.
- ✅ **Sim illumination drift** — `add_drift` (`crates/tomoxide-sim/src/noise.rs`;
  tomopy `sim/project.py:80`). Done. Deterministic (no RNG): scales each
  projection angle `i` by `drift[i] = amp·sin(2π·i/period) + mean +
  linspace(0,1)[i]`, constant across the detector. Held to the **f32 round-off
  floor (≤1 ULP per pixel), not Δ=0** — numpy 2.x evaluates `np.sin` on the f64
  angle array with its own vectorized routine that differs from Rust libm
  `f64::sin` by ≤1 ULP for some angles, and that f64 difference survives the f32
  cast in a small fraction of pixels (the f64·f32 product is commutative, so
  `sin` is the sole divergence). Same precision class as the single-precision
  phase-retrieval ports. Golden from real tomopy
  (`tools/gen_tomopy_add_drift_golden.py`; 1493/9216 pixels differ ≤1 ULP, worst
  rel=1.19e-7) in `tests/add_drift_parity.rs`.
- ✅ **3-D Shepp-Logan phantom** — `phantom::shepp3d` (`crates/tomoxide-sim/src/
  phantom.rs`; tomopy `misc/phantom.py:284`). Done. A faithful f64 ellipsoid
  rasterizer: 10 ellipsoids sampled on `np.mgrid[-1:1:size·j]`
  (`= arange·step − 1`, `step = 2/(size−1)`), each rotated by an Euler matrix
  (libm `sin`/`cos` of `to_radians` — `np.radians` and numpy's *scalar* trig both
  verified bit-exact vs Rust), shifted by its centre and scaled by its semi-axes,
  with the inclusion test `Σ((R·r − c)/s)² ≤ 1` in f64 and the amplitudes
  accumulated in f32 exactly like numpy's in-place `obj[mask] += A` (the f64 add
  cast back to f32 after each ellipsoid), then `clip(0, ∞)`. Matches tomopy
  **bit-for-bit (Δ=0)** at sizes 16/17/32. The only step not reproduced is
  numpy's BLAS dot order inside `tensordot` (≤1 ULP), which flips **no** voxel —
  no grid sample lands within f64-ULP of the `≤ 1` boundary, the same
  non-materialisation as the `circ_mask` boundary gap, so a plain sequential dot
  suffices. `shepp3d_parity.rs`, golden from the **real tomopy**
  `tools/gen_tomopy_shepp3d_golden.py`. (The simplified f32 `shepp2d` rasterizer
  is left untouched.)
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
- ✅ **Nearest-flat-fields normalization** — `normalize::normalize_nf` (tomopy
  `prep/normalize.py:245`, `averaging='mean'`). Done. Each flat group's per-pixel
  median normalizes the projections nearest its `flat_loc`; `dark` is the dark-frame
  mean; `(proj−dark)/max(flat−dark,1e-6)` with an optional `cutoff`, group bounds
  at the half-to-even midpoint of consecutive `flat_loc`. f32 in upstream order →
  **bit-exact (Δ=0)** for even/odd group sizes incl. the denom-clamp and cutoff
  paths. `averaging='median'` returns a TODO error because tomopy's
  `np.median(dark, …, dtype=np.float32)` raises on modern numpy (no reference),
  mirroring the `remove_stripe_ti` block-path treatment. `normalize_nf_parity.rs`,
  golden from the **real tomopy** `tools/gen_tomopy_normalize_nf_golden.py`.
- ✅ **ROI normalization** — `normalize::normalize_roi` (tomopy
  `prep/normalize.py:168`). Done. Each projection is divided by the mean `bg` of
  its ROI window `proj[r0:r2, r1:r3]` (skipped when `bg == 0`, matching tomopy's
  `if bg != 0`). The catch is the divisor: numpy's `ndarray.mean` sums the ROI in
  f32 with its **pairwise** accumulation tree (8-accumulator unrolled base case
  for `n ≤ 128`, recursive split rounded down to a multiple of 8 otherwise), so a
  plain sequential f32 sum diverges by ~1 ULP and poisons every divided pixel. A
  new `pairwise_sum_f32` reproduces that tree exactly → `bg` and the elementwise
  divide are **bit-exact (Δ=0)** for the default 10×10 ROI (base case), a
  480-element ROI (recursion path), and an offset non-square ROI. Golden from the
  **real tomopy** `tools/gen_tomopy_normalize_roi_golden.py`, which applies
  tomopy's verbatim per-projection `_normalize_roi` kernel in-process (the macOS
  `mproc.distribute_jobs` pool is flaky and the chunking is per-projection
  independent, so this is numerically identical). `normalize_roi_parity.rs`.
- ✅ **Downsample / upsample** — `morph::{downsample, upsample}` (tomopy
  `misc/morph.py` → `libtomo/misc/morph.c::c_sample`). Done. Power-of-two binning
  (`2^level`) along any axis: downsample = per-bin mean accumulated as
  `Σ(data/binsize)` in f32 with the C's flat running counter (bit-exact even when
  the axis isn't divisible); upsample = `×binsize` replication. **Bit-exact (Δ=0)**
  across axes 0/1/2. `sample_parity.rs`, golden from the **real tomopy**
  `tools/gen_tomopy_sample_golden.py`. (`trim_sinogram` is unrunnable on modern
  numpy — float `ceil/floor` slice bounds + float `diameter` array shape both
  raise `TypeError` — so it has no reference; deferred as a TODO.)
- ✅ **Axis padding** — `morph::pad` (tomopy `misc/morph.py:73`). Done. Widens an
  axis by `npad` on each side (`npad=None` → `⌈(dim·√2−dim)/2⌉`); flanks are a
  constant (`PadMode::Constant`) or the replicated edge slab (`PadMode::Edge`).
  Pure copy/fill → **bit-exact (Δ=0)** for constant/edge on axes 0/1/2 and default
  / explicit `npad`. `pad_parity.rs`, golden from the **real tomopy**
  `tools/gen_tomopy_pad_golden.py`.
- ✅ **Projection scaling** — `alignment::scale` (tomopy `prep/alignment.py:460`).
  Done. Seeds a new `prep::alignment` module mirroring tomopy's `prep/alignment.py`.
  Divides a projection stack in place by `scl = max(|max|, |min|)` to land it in
  `[−1, 1]` and returns `scl` (needed to invert the scaling). Pure order
  statistics (`max`/`min`/`abs`) plus an elementwise f32 divide — no summation or
  transcendental — so both the scaled array and the returned `scl` are
  **bit-exact (Δ=0)**. Matches tomopy's lack of a zero guard (all-zero stack →
  `scl=0`, NaN pixels); errors only on an empty stack. 4 cases
  (positive/negative-dominated, symmetric, small-magnitude) pass at 0
  bit-mismatches. `scale_parity.rs`, golden from the **real tomopy**
  `tools/gen_tomopy_scale_golden.py`.
- ✅ **Edge blurring** — `alignment::blur_edges` (tomopy `prep/alignment.py:482`).
  Done. Multiplies every projection image by a radial feather mask: within a
  projection `rad = √((row−dy/2)² + (col−dz/2)²)`, the mask is `1` where
  `rad < low·rad_max`, `0` where `rad > high·rad_max`, and a linear ramp
  `(rmax−rad)/(rmax−rmin)` between (`rmin = low·rad_max`, `rmax = high·rad_max`).
  Unlike the `sin`/`cos`/`exp` ports, `√` is **IEEE-correctly-rounded** so
  `np.sqrt == f64::sqrt` bit-for-bit (no round-off floor); `arr**2 == x·x` for
  f64 and the in-place `float32 *= float64` is the f64 product cast to f32 (both
  verified in the golden env), so the result is **bit-exact (Δ=0)**. The mask is
  built by the same sequential assignment as upstream, so a degenerate `low>high`
  also matches. `blur_edges_parity.rs` covers the default `(0, 0.8)` and
  `(0.2, 0.9)`, golden from the **real tomopy**
  `tools/gen_tomopy_blur_edges_golden.py`. (The rest of `prep/alignment.py` —
  `shift_images`/`add_jitter` [skimage warp], `align_seq`/`align_joint` — stays
  unported.)
- ✅ **Morphological inpainter** — `filters::inpainter_morph` (tomopy
  `misc/corr.py:996`, C `libtomo/misc/inpainter.c` `Inpainter_morph_main`,
  Kazantsev 2023). Done. Fills a boolean-masked region: zero the mask, grow
  inward from the non-empty boundary (≤`countmask` passes) until filled, then
  `iterations` smoothing passes; `axis=None` runs the symmetric 3-D kernel,
  `axis=Some(a)` inpaints each 2-D slice along array axis `a`. Three modes via
  `InpaintingType`: **Mean** (`eucl_weighting`, Gaussian-distance-weighted) and
  **Median** are deterministic and **bit-exact (Δ=0)** — the mean path's `exp`/
  `powf` match macOS libm bit-for-bit and its f32 multiply-accumulate is fused
  via `mul_add` to match libtomo's FMA-contracted build (a split `*`,`+` drifts
  ≤2 ULP, the one wrinkle this port had to chase); the median path reproduces the
  C buffer-sort quirks exactly (2-D sorts `counter_local−1`, 3-D sorts the whole
  `window_fullength` zero-padded buffer, both pick `_values[counter_local/2]`, so
  a boundary cell can land on the padding and stay exactly `0.0` — one golden case
  exercises this). **Random** (rand-pair + final mean smoothing) is faithfully
  ported but has **no bit-parity reference**: C `rand()` under OpenMP is not
  reproducible run-to-run (verified — tomopy's own output differs each run), so it
  uses an internal deterministic LCG and is covered structurally by unit tests.
  `inpainter_morph_parity.rs` covers mean/median × {`axis=None`, `axis=0`} ×
  `iterations` 0/1/2, golden from the **real tomopy**
  `tools/gen_tomopy_inpainter_morph_golden.py`.

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
- **M6 — wgpu/Metal:** runs the GPU path on Apple Silicon (Metal), Vulkan/DX12
  elsewhere; gated on the `gpu-wgpu` feature.
  - ✅ Device init + GPU compute smoke test (`WgpuBackend::new`, full
    upload→dispatch→readback roundtrip).
  - ✅ Reusable compute primitives (`compute.rs`: storage buffers, uniform,
    1-D dispatch, host readback).
  - ✅ **Elementwise** (`darkflat` + `minus_log`) WGSL ports with CpuBackend
    tolerance-parity tests (GPU f32 ≠ libm bit-exact → rtol 1e-5, not Δ=0).
  - ✅ **FilteredBackproject** (parallel-beam back-projection) WGSL port
    (`backproject.wgsl`): one thread/voxel, linear-interp sum over angles ×
    π/nproj. Host-computed (cosθ,sinθ) + per-row center keep the inclusion
    boundary bit-identical to CPU; tolerance parity (rtol 1e-4), scalar &
    per-row center. (Parity tests reconstruct an interior ROI so no ray
    grazes the detector-edge inclusion cutoff.)
  - ✅ **ForwardProject** (parallel-beam Radon) WGSL port (`project.wgsl`):
    race-free scatter via one thread per `(row, angle)` (disjoint detector
    column span, CPU pixel order). Exact adjoint of the wgpu back-projector,
    shares the host-side `(cosθ,sinθ)`/per-row-center helper. Tolerance parity
    (rtol 1e-4), scalar & per-row center, interior-ROI test.
  - ✅ **RankFilter** (`median3d` + `remove_outlier3d`) WGSL port
    (`medfilt3d.wgsl`): one thread/voxel, clamp-to-center gather + partial-
    selection order statistic. **Bit-exact (Δ=0)** with CPU (pure gather + one
    subtraction). Window capped at 7³ (WGSL private-array size); larger errors
    out. Surfaced + fixed the workgroup-size single-source-of-truth defect
    (dispatch1d injects `const WG`; kernels use `@workgroup_size(WG)`).
  - ✅ **Fft** (batched 1-D + 2-D) WGSL port (`fft.wgsl` bit-reversal +
    per-stage butterflies; `fft_transpose.wgsl` for the 2-D column pass).
    Tolerance parity vs rustfft (GPU cos/sin twiddles); fft_1d & fft_2d
    roundtrip tests.
  - ✅ **Bluestein** chirp-z transform (`bluestein.wgsl`): `fft_1d` now handles
    **any length**, not just power-of-two — non-pow2 lengths run a length-n DFT
    as a power-of-two circular convolution (`m=next_pow2(2n−1)`: 3 radix-2 FFTs
    + spectral multiply), host-side chirps with `j² mod 2n` argument reduction.
    Rel error ≈1e-7 vs rustfft (which uses Bluestein internally), lengths
    3/5/6/12/100. (`fft_2d` stays pow2-only — non-pow2 2-D is a CPU fallback;
    device-level 2-D Bluestein is a follow-up. This unblocks GPU `fourierrec`/
    `lprec`, which need arbitrary-length transforms.)
  - ✅ **FbpFilter** (`apply`) WGSL port (`fbp_filter.wgsl` `apply_filter`):
    each detector lane padded to `pad=filter.len()`, then forward FFT →
    frequency-domain ×filter multiply (complex×real, broadcast across the lane
    batch) → inverse FFT, all one serialized GPU submission chain (no host
    round-trip between transforms), crop to n_cols ×(1/pad). Power-of-two pad
    only (radix-2 FFT; else error→CPU fallback). `make_filter` delegates to the
    shared `tomoxide_core::backend::make_fbp_filter` so CPU/GPU build the
    identical kernel. Tolerance parity vs CpuBackend (2e-3). **This closes the
    full GPU FBP recon path** — filter → back-project both run on-device.
  - ✅ **End-to-end analytic recon on GPU** (`analytic_gpu_parity` test): the
    capabilities *compose* through `recon::recon(.., &dyn Backend)` with no
    recon-crate changes. **FBP** (FbpFilter ∘ FilteredBackproject) GPU↔CPU
    NRMSE 1.2e-6; **gridrec** runs on the GPU for free (needs only `Fft`, all
    transform lengths power-of-two; Kaiser-Bessel gridding stays host-side)
    GPU↔CPU NRMSE 3.4e-7. Both also correlate r≈0.96–0.97 with the phantom.
  - ✅ **Iterative recon on GPU** (`iterative_gpu_parity` test): the whole
    project∘back-project solver family (SIRT, MLEM, OSEM, PML/OSPML, grad,
    tikh, tv) runs on the GPU through the same `&dyn Backend` dispatch — first
    composition coverage of the GPU forward projector. SIRT 100-iter loop:
    GPU↔CPU correlation 1.00000, NRMSE 1.8e-4, GPU recon as accurate as CPU.
    (`Art`/`Bart` stay CPU-only — host-side sparse `RayProject`.)
  - ⬜ Remaining: **device-level 2-D Bluestein** (so `fft_2d` also handles
    arbitrary dims); **`fourierrec`/`lprec`/`lamino`** wgpu ports (USFFT-based,
    reference `~/codes/tomocupy/src/cuda/cfunc_{fourierrec,lprec,usfft*}.cu` —
    stubs on every backend, so they need golden data, not a CPU parity ref);
    sub-pixel rotation-center phase in the filter (`fbp_filter_center`; CPU
    currently centers in the back-projector — a semantic change needing
    sign-off). (`RayProject` is host-side sparse rows for ART/BART — not a GPU
    kernel.) **All GPU-appropriate capability traits are now implemented and
    parity-verified on wgpu.**
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
