# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **`tomoxide-gui` now depends on `rsplot` / `rsdm` `=0.5.2`** (was `=0.5.0`).
  0.5.1 adds the `VolumeRaycaster` widget the XANES 3-D chemical-map view uses
  and 0.5.2 hardens it; the GUI builds against crates.io again, with the
  local-checkout `[patch.crates-io]` block commented out.

### Added

- **`recon::center::judge_sweep` — a focus sweep's verdict, not just its
  argmax.** Returns `SweepVerdict::{Resolved, Railed, Ambiguous, Flat}`, and only
  `Resolved` carries a value (`SweepVerdict::resolved()`); `best()` exists to
  *show* the winner on a plot or montage, never to adopt it. Every caller — `align`,
  the GUI's centre and tilt steps — now refuses rather than reports a number the
  curve did not establish, which matters because a bad axis silently poisons the
  tilt search that consumes it.
  The verdict is one uniform test — does this lobe own at least 20 % of the
  curve's height? — asked of the winner and of every other lobe. It replaces an
  edge check, which measurement showed is blind to both ways a real sweep fails.
  On the aligned pouch scan (known axis 396, tilt 44°), swept ±40 px: on the
  rh/2 plane the curve carries **three** lobes and the highest is 417 — 21 px
  wrong, interior, beating the truth by 0.34 %, so no edge check can fire
  (`Ambiguous`, naming 396 as the strongest rival at 55 % prominence). On the
  sample plane the curve instead rises across the whole window and turns over one
  sample *short* of the edge at 435 — an interior maximum with no rival to
  contradict it, and still nothing but the range running out (`Railed`). Both are
  the same fact: the winner owns no lobe of its own. Narrowed to ±8 px around a
  prior, the same curve resolves 395.75 at 83 %.
- **`tomoxide align` — the alignment workflow as a subcommand**, plus the two
  library pieces it stands on. `recon::center::find_center_rings` is the ring
  estimator from `docs/LAMINOGRAPHY_ALIGNMENT.md` §1: the 360° mean projection is
  a bullseye centred on the rotation axis, registered against its own mirror. It
  reports a `prominence` — `(peak − median)/MAD` of the registration profile —
  and flags the scan when the rings never closed, which is the acquisition-time
  misalignment no reconstruction geometry repairs (measured: aligned scan 397.37
  at prominence 16.2 → trustworthy; misaligned scan 281.72 at 2.22 → flagged, and
  `align` stops there unless `--force`). `cuda::center_probe` / `center_probe_sweep`
  then refine it: `cfunc_linerec`'s `backprojection_try` reconstructs one slice at
  N candidate centres in **one** launch off a single filtering. A probe slice
  equals the reconstruction exactly only when the shift is a whole number of
  columns (measured 2e-7…9e-7 of peak) and differs by ~1.6 % at half a column, so
  a naive sub-pixel sweep ranks integer offsets artificially sharp;
  `center_probe_sweep` removes that by construction, issuing one probe per
  fractional lattice, and never producing the biased slices at all. End to end on
  the aligned pouch scan, swept ±8 px around the axis read off the rings:
  `--center 395.87` against a known 396. Around a *prior* is the whole contract —
  the same sweep widened to ±40 px grows three lobes and picks 417, and that is
  what `judge_sweep` above now refuses rather than returns.
  The two axes are searched by different machinery, and the asymmetry is physics,
  not an optimisation left undone. A fixed slice is sound for the centre — an
  in-plane shift does not move the in-focus layer, so one slice can rank the
  candidates — and unsound for the tilt, whose response is broad
  (~2 % per degree) while the in-focus layer moves in z with it (`z_peak` 800 →
  1120 as tilt went 40° → 58°), so a fixed slice scores the wrong plane by more
  than the tilt signal is worth. Measured: a fixed-slice tilt sweep returns 48° at
  one slice and rails to the top of any range at another. Hence
  `recon::center::lamino_tilt_scan` (`--tilt_width` / `--tilt_step`), which does
  what `docs/LAMINOGRAPHY_ALIGNMENT.md` §2/§4 validates: one **full**
  reconstruction per candidate, scored by the max `slice_focus` over the whole z
  range. That needs every (tilt, z) pair, which is exactly the work
  `lamino_tilt_probe` skips — so the probe buys nothing there and remains for the
  case its saving is real, a plane that is already known. The scan streams the
  volume, so its memory cost is one rh-tile regardless of depth, and it keeps what
  the reconstruction already computed: the winning slice (§3 confirms by eye) and
  `focus_by_z`, the focus of every slice. The profile is what separates a real
  optimum from the failure §2 names — "a MONOTONE focus surface with the argmax
  pinned to a grid corner" — and on the aligned pouch scan at 44° it shows a broad
  hump rising to 1.96e-5 at slice 292 of 1424, not an edge spike.
- **`recon::center::pick_interior_max`** — the argmax of a sweep, plus whether the
  sweep earned the right to call it an optimum. A maximum on the first or last
  candidate is the range running out, and there is no way to tell from inside the
  range whether the true peak is at the edge or beyond it. Both `align` and the
  GUI refuse to adopt a railed pick; measured worth: it caught a real tilt scan
  railing at 47° on the pouch scan, and a centre sweep flat to 1.8 % and railing
  when pointed at an empty slice.
- **`recon::center::slice_focus` and `mean_projection`** — the two measurements
  the alignment method is written in, as library functions rather than copies in
  each consumer. `slice_focus` is §2's score (mean |∇|² inside a 0.92-FOV disk;
  the disk matters — reconstruction geometry decides where the sampling leaves the
  detector, so scoring the full frame scores a boundary that moves with the
  parameter being scored). `mean_projection` is the bullseye `find_center_rings`
  measures, exposed because §1 makes *looking* at it step one and treats
  eye-vs-correlation disagreement as the misalignment flag — which needs the
  image, not only the number.
- **`tomoxide-gui`: laminography alignment on the Center screen** (Beam:
  Parallel | Laminography). The toggle is not a display option: under a
  laminographic tilt the axis leans along the beam, so 0°/180° is not a mirror of
  the object and `find_center_vo` / `_pc` / `_sift` all lose the symmetry they
  assume (mirror registration scattered 395…607 against a known 396). It picks the
  estimator family valid for the beam. The three steps are the doc's, and each is
  a picture the CLI can only summarise as a number: **1 Rings** shows the mean
  projection itself, so closed rings versus arcs that never close is a thing you
  see rather than a prominence you trust; **2 Centre** is the probe sweep as a
  montage plus a focus curve with click-to-pick; **3 Tilt** streams one result per
  full reconstruction — the in-focus slice into a montage, focus-vs-tilt, and
  focus-by-slice for the selected candidate — with a cancel, since it is minutes
  per candidate. Railed picks are shown as such and never adopted.
- **`docs/LAMINOGRAPHY_ALIGNMENT.md` — how to find the laminography rotation
  center and tilt.** A field-tested recipe, validated against a scan whose center
  was already known independently. Leads with reading the center off the raw data
  before reconstructing anything: the mean of all projections over 360° is a
  bullseye of rings centred on the rotation axis, so the center column is
  readable by eye (automated flip-registration measured 397.5 vs a known 396),
  and rings that open into arcs instead of closing diagnose a scan that was
  mis-aligned at acquisition. Also documents the reconstruction sweep that
  follows (full projections, focus scored as the max over the whole z range, not
  a sub-sample over a fixed band) and records the three methods that failed
  validation — 0°/180° mirror registration (the tilted axis breaks the mirror
  symmetry), 1-D column-profile symmetry (truncation destroys it), and
  sub-sampled projections with a fixed z band.
- **Fourier laminography (`LamFourierRec`) now honours `--filter`.** The
  Fourier/USFFT lamino path applied a plain `|f|` ramp and ignored the selected
  apodisation, unlike tomocupy's `LamFourierRec` (which filters with
  `args.fbp_filter`, default parzen) and unlike tomoxide's own fbp/linerec lamino
  paths. The apodisation window is now folded into the projection ramp on all
  three paths (CPU golden, CUDA host, CUDA device) from a single window
  definition shared with `make_fbp_filter`. The default is parzen (tomocupy
  parity); pass `--filter ramp` for the previous plain-ramp behaviour.
- **Multi-GPU angle-split SIRT for laminography.** Laminographic SIRT can now run
  across multiple GPUs by partitioning the projection angles into contiguous
  per-device chunks; each GPU holds the full volume and its angle chunk and
  contributes a partial tilted back-projection, with the correction all-reduced
  each iteration. The result is bit-identical to single-device SIRT.
- **CUDA laminography now streams out-of-core, across all GPUs.** The tilted
  back-projector previously built the whole filtered stack in one shot on a
  single GPU, which overflowed the kernels' 32-bit index (`nz·nproj·ncols ≈
  7.5e9 > 2³¹`, SIGSEGV) and exceeded VRAM at production resolutions. It now
  mirrors tomocupy's laminography chunking: filter the stack once into host
  memory (nz-sub-chunked so the padded scratch stays under VRAM and the index
  ceiling), then a nested output-rh-tile × projection-angle-chunk loop uploads
  each filtered angle chunk and accumulates its back-projection into the tile.
  Multi-GPU shards the output rh axis across every selected device (the whole
  stack is filtered once, then per-device back-projection shards read it
  read-only). Chunk sizes are derived from free VRAM and the index ceiling. The
  fits-in-one-chunk case stays byte-identical to the previous single-shot path.
  The CLI now streams the result to disk as well: `reconstruct_lamino_streaming`
  (library + pipeline) emits the output one rh-tile at a time through a callback,
  so `tomoxide-cli` writes the volume tile-by-tile and never holds the whole
  reconstruction in host RAM. Tiles are computed a round of GPUs at a time and
  written from the main thread (the H5 writer is `!Send`), bounding host peak to
  the sinogram plus the filtered host stack plus one in-flight tile per device.
- **GUI: Live streaming reconstruction screen** (`tomoxide-gui`, the seventh
  mode wired up — docs/GUI.md §2.6, milestone M3 first cut). Connects a
  tomoScanStream-style pvAccess projection stream through rsdm's headless data
  engine: NTNDArray frames land in a fixed-capacity ~180°-of-projections ring
  buffer (with rolling dark/flat), and a self-contained live thread
  reconstructs the selected **Z (horizontal) slice** every loop, re-reading the
  parameters each iteration (tomostream semantics — a center tweak or filter
  change applies on the next pass). Controls: PVA channel addresses
  (image/theta/companion width+height/optional dark+flat), slice index, live
  center + tweak, analytic algorithm, filter, Fourier-wavelet ring removal,
  buffer depth. rsdm delivers frames as a flat pixel array, so the detector
  width comes from a companion width PV (or a manual value) and the height is
  either a PV, a manual value, or **derived from the frame length** (`len /
  width`) — so only the width is ever required. The rotation angle likewise
  comes from a theta PV or a manual constant step per frame (deg/frame), for
  streams that publish no theta PV. When a theta PV is used each frame is paired
  with the latest scalar angle (no NTNDArray `uniqueId` is exposed). X/Y
  ortho panes need dedicated backprojection kernels (§6 #7) and are out of
  scope for this pass — Z-only, as the design's honesty note calls for. Verified
  end-to-end against an in-process `epics_pva_rs::PvaServer` (no beamline
  required).
- **`tomoxide::xanes` — per-voxel XANES peak-energy fitting** (new `xanes`
  feature, off by default). Ports the reference `txm_pal_core` fit core
  (Levenberg–Marquardt quadratic / Gaussian white-line fits + Savitzky–Golay /
  median / 3-point / boxcar smoothers) to pure Rust with no PyO3. `fit_peak_energy`
  is the per-voxel core (smooth → locate peak → windowed curve fit → range-check
  → NaN on failure); `fit_map` is a `rayon` driver over a `(E, z, y, x)` `f32`
  volume view, parallel across `z`, that reads views and never materialises the
  full `f64` stack (so a 40-energy 500³ volume streams a z-band at a time) and
  polls a `CancelToken` between slices. This is library-side milestone-M4 item
  #11.
- **`tomoxide::xanes::MultiEnergyVolume` — multi-energy stack reader** (M4 #12)
  plus **`io::read_h5_band`** (coalesced `[z0, z1)` band hyperslab read, dtype-
  dispatching like `read_h5_frame`). A `MultiEnergyVolume` is a common-grid set
  of per-energy recon volumes — either separate files sharing a dataset key
  (tomoxide's own per-energy output, `from_files`) or one combined file with an
  explicit `(energy, dataset)` per energy (`from_combined`) — validated to one
  grid, sorted by energy, and read a `z`-band at a time into `(E, band, ny, nx)`
  `f32` to feed `fit_map` without holding the full stack. Dataset keys are
  always caller-supplied; the reader never guesses a combined file's per-energy
  naming scheme.
- **`tomoxide::xanes::write_peak_map_h5`** + **`io::list_h5_datasets`**. The
  results writer emits a fitted map (`peak_energies`, `energies`, `edge_jump`,
  finite-voxel `mask`) in the layout the `xanes_tools` Python viewer reads; the dataset
  lister returns every key in a file so a combined-stack loader can discover
  per-energy volumes without guessing the writer's name formatting.
- **`tomoxide::xanes` — zone-plate magnification correction** (M4 #14). Ports the
  reference `magnification_correction.py`: `magnification_corr_factors` derives a
  per-energy scale factor from the zone-plate focal-length model (focal length
  grows with photon energy, so each energy images the sample at a slightly
  different magnification), normalised to the first energy; `apply_magnification`
  rescales a `(z, y, x)` volume about its centre by that factor (bilinear over
  `z`/`x`, the `y` rotation axis untouched, zero-fill outside — matching
  `scipy.ndimage.affine_transform(diag(cf, 1, cf), order=1, mode="constant")`).
  `MagnificationParams` carries the zone-plate geometry with reference defaults.
- **GUI: XANES chemical-mapping screen** (`tomoxide-gui`, a seventh mode). Loads
  a combined registered stack (`registered.h5`: an `energies` axis +
  `reconstructions/{energy}` volumes — per-energy keys discovered by parsing the
  dataset-name leaves, no formatting assumed), streams it a `z`-band at a time
  through `xanes::fit_map` on a cancellable background thread with a live
  progress bar, and browses the peak-energy map slice-by-slice (a click reads
  that voxel's spectrum; a histogram summarises the map). Fit controls cover
  method (quadratic / Gaussian), window width, energy range, smoothing, a
  mean-absorption mask threshold, and band size; the result saves via
  `write_peak_map_h5`. This is milestone-M4 item #13 (viewer); the stack it
  consumes is produced upstream (energy-looped recon + external registration).
  The central panel toggles between the 2-D slice browser and a **3-D direct
  volume rendering** of the chemical map (rsplot's new GPU `VolumeRaycaster`:
  front-to-back alpha-composited ray-march, orbit / pan / wheel-zoom). The
  transfer function reuses the same viridis peak-energy colormap as the 2-D map
  for hue, with fitted voxels opaque and unfitted (masked / out-of-window)
  voxels transparent; an opacity slider scales the composite and the volume is
  decimated to a bounded edge so the GPU upload stays small (docs/GUI.md §2.7).
- **GUI: XANES per-energy magnification correction** (`tomoxide-gui`). A
  side-panel toggle applies `xanes::apply_magnification` to each energy's volume
  before fitting, driven by editable zone-plate geometry (magnification, ZP
  diameter, outermost zone width) with a live preview of the resulting factor
  range. Because the correction scales the `z` axis it cannot be applied to the
  streamed `z`-bands independently, so enabling it reads the whole stack into
  memory and fits from the corrected copy; the default streaming path is
  unchanged (docs/GUI.md §6 #14).
- **GUI design: XANES spectroscopic-mapping screen** (`docs/GUI.md` §2.7, a
  seventh mode). Post-reconstruction per-voxel white-line / edge-shift fitting
  over a multi-energy volume stack, reusing the existing `txm_pal_core` Rust
  fit core (Levenberg–Marquardt + rayon, no GPU); 2-D chemical maps +
  histogram + click-to-spectrum, with a results-HDF5 save interoperable with
  the `xanes_tools` Python viewer. New library additions — fitting-module port,
  multi-energy stack reader, energy-to-energy registration, zone-plate
  magnification prep — are catalogued in §6 (#11–#14) and milestone M4 (§7).
  Per-energy reconstruction reuses the **existing CUDA streaming pipeline**
  (`run_streaming_pipelined_range`) looped over energies with the current
  DXchange preprocessing unchanged — no new recon or prep code. Energy-to-energy
  3-D registration (SimpleITK rigid Mutual-Information) has no pure-Rust
  equivalent and stays an upstream Python step in v1.

### Fixed

- **Fourier laminography reconstruction sign corrected.** The Fourier/USFFT
  lamino path (`recon::lamino::lamino` and its CUDA analytic mirror) reconstructed
  dense material as **negative** attenuation — inverted versus the physical
  minus-log convention that linerec/fbp (direct back-projection) and SIRT (physical
  forward model) follow. The cause was a global `-1` in tomocupy's `irfftshiftc`
  fft2d step, ported faithfully; it is a pure sign, independent of the fftshift
  centering. It is now dropped in `fft2d_fwd` and, symmetrically, in the paired
  `fft2d_inv`, so the reconstruction matches linerec/fbp/SIRT (positive material)
  while the `fft2d_inv(fft2d_fwd) == id` round-trip and CPU↔CUDA parity are
  preserved. Output now differs from tomocupy's raw fourierrec by a global sign
  only (magnitudes identical).

## [0.6.0] - 2026-07-06

### Changed

- **`tomoxide-gui` now depends on `rsplot` 0.5.0 from crates.io** (was the
  `siplot` git dependency pinned to a rev). The plotting crate and its GitHub
  repo were renamed `siplot` → `rsplot` (and the sibling EPICS engine
  `sidm` → `rsdm`) and published to crates.io; the GUI switches to
  `rsplot = "=0.5.0"` with a commented-out `[patch.crates-io]` for local dev.
  Mostly a `use siplot` → `use rsplot` identifier rename. The one behavioural
  gap: the local `siplot` carried an unpushed commit making a `Plot2D`
  crosshair read out the pixel value under the cursor ("x, y, value"); `rsplot`
  0.5.0 instead exposes the value via the higher-level `ImageView`
  (`value_changed`, silx `PositionInfo` "Data"). The Data sinogram inspector and
  the Tune single-slice preview were migrated `Plot2D` → `ImageView` to keep the
  value readout (the preview's `ColormapDialog` is replaced by `ImageView`'s
  interactive colorbar). GUI build/clippy/35 tests green on CPU and CUDA. The
  Data sinogram inspector and Tune preview `ImageView` layout was fixed and
  verified on-screen: the `ImageView` position-info bar is emptied and the value
  readout is pinned to the bottom so the image fills exactly the available
  height, keeping the sinogram's resizable bottom panel stable (an earlier
  version let it grow into / collapse away from the projection browser).
- **`tomoxide-gui` is now published to crates.io** at `0.6.0` (its first
  release, aligned with the library and CLI). To make it installable
  everywhere, its default features are CPU-only (`["sift-center"]`, was
  `["cuda", "sift-center"]`) so `cargo install tomoxide-gui` and the docs.rs
  build succeed without an NVIDIA toolkit — enable the CUDA backend explicitly
  with `--features cuda`. Its `tomoxide` path dependency gained a matching
  `version` (`0.6.0`), which crates.io requires to publish. The crate still
  lives outside the Cargo workspace (edition 2024 / rust 1.92, own lockfile).

### Added

- **CLI `--progress_json`** (`recon`, `recon_steps`) — one flushed JSON line
  per completed output chunk on stdout
  (`{"start":s,"end":e,"total":nz,"secs":t}`; global slice range, full output
  slice count, wall-clock seconds since run start), implemented as a thin
  `VolumeWriter` tee. The multi-GPU shard orchestrator forwards the flag and
  lets children inherit stdout, so shard lines stream through with global
  ranges against one total. Machine progress for wrappers — the GUI Run
  screen tails these from its subprocess runs. Runtime-only (not a config
  key); progress lines are exactly the stdout lines starting with `{`.
- **Cooperative cancellation for the chunked drivers** — new
  `pipeline::CancelToken` (`Clone`-able atomic flag) attachable via
  `ReconSteps::with_cancel`; all three drivers (`run`, `run_streaming`,
  `run_streaming_pipelined_range`) check it at chunk boundaries and stop with
  the new `Error::Cancelled`. Cancellation truncates: chunks already written
  stay on disk and the writer finalizes the partial output.
- **`io::InMemoryWriter`** — a `VolumeWriter` that collects the reconstruction
  into a shared in-memory `InMemoryVolume` (the `Arc<Mutex<…>>` handle
  survives the pipelined driver consuming the writer) with an optional
  `on_chunk` progress callback. Backing store for GUI previews.
- **`tomoxide::config` (feature `config`)** — the CLI's TOML `Config` moved
  into the library behind a default-off feature (optional `serde`/`toml`
  deps) so GUI recipes and CLI configs are one format. Gains three fields:
  `lamino_angle`, `dtype`, and `output` (base path; each writer adds its own
  suffix). The CLI gained the matching `--output` flag, `--config` now feeds
  all three, and the multi-GPU z-shard fan-out forwards the resolved output
  path to its children.
- **`io::read_h5_frame`** — read one `[ny, nx]` frame of a 3-D HDF5 stack as
  `f32` through the reader's full dtype dispatch (u8/i8/u16/i16/u32/i32/f32/
  f64). Real beamline stacks are usually `uint16`; the GUI projection browser
  reads through this (rsplot's own HDF5 loader handles 4/8-byte floats only).
- **`io::read_h5_sizes`** — the `(n, ny, nx)` shape probe paired with
  `read_h5_frame`, so a volume browser can size its frame list without
  reading any data.
- **Vo 2018 single-method stripe variants through config/CLI** — the
  long-implemented `StripeMethod::VoSort`/`VoFilter`/`VoLarge`/`VoDead`/
  `VoFit` are now selectable as `--remove_stripe vo-sort | vo-filter |
  vo-large | vo-dead | vo-fit`, with per-method config fields/flags
  (`vo_sort_*`, `vo_filter_*`, `vo_large_*`, `vo_dead_*`, `vo_fit_*`;
  median sizes use the `fw_level` convention `0` = tomopy auto) and
  multi-GPU shard forwarding.
- **`tomoxide-gui` M1 (offline preview loop)** — new repo-internal but
  workspace-`exclude`d crate (rsplot is edition-2024/rust-1.92; workspace
  membership would raise the repo's effective MSRV above 1.82) implementing
  docs/GUI.md M1: a single worker thread owns the `Engine` and all HDF5
  handles (`!Send`); **Data** (DXchange open + metadata, projection browser —
  frames load through `io::read_h5_frame` so `uint16` beamline stacks render,
  with the colormap scaled to the stack's raw-count range — theta plot, raw
  sinogram inspector), **Tune** (single-slice preview through
  `run_streaming_pipelined_range` into `io::InMemoryWriter`, parameter panel
  with auto-recon, A/B pin compare), **Center** (Vo / entropy /
  phase-correlation / SIFT auto methods, ±0.5/±0.25 px tweak, hand-off to
  Tune), and recipe save/load (recipe file = CLI config TOML plus a `[gui]`
  table the CLI ignores). `tomoxide-gui [FILE] [--mode <mode>]` opens a
  dataset and/or picks the starting mode from the command line, and Tune
  fires the first preview of a fresh dataset by itself (the auto toggle
  still gates re-runs on parameter changes).
- **GUI Tune — λ sweep (L-curve regularization tuner).** For algorithms whose
  `reg_par[0]` is a regularization strength λ (`tv`, `grad`, `tikh`,
  `pml_*`, `ospml_*`), a new worker `Job::LambdaSweep` reconstructs the preview
  slice once per λ across a log-spaced grid and scores each on the L-curve —
  data residual `‖A x − b‖₂` (the fidelity term, via `sim::project`) vs the
  reconstruction's isotropic TV seminorm (roughness). A floating window shows
  the per-λ montage over the L-curve; the max-distance-to-chord corner is the
  suggested λ, a click picks any point, and "Use selected λ" writes it to
  `reg_par[0]`. The guide is the L-curve corner, not a sharpness auto-pick:
  sharpness falls monotonically with λ and real data has no ground truth
  (`docs/BENCHMARKS.md` §10), so the choice stays the user's. Verified finite
  and λ-varying on both CPU and the CUDA device-resident 1-slice path.
- **GUI design document** (`docs/GUI.md`) — design for a `tomoxide-gui`
  desktop application built on rsplot (egui + wgpu) and rsdm (EPICS PVA):
  offline workflow (dataset browsing, single-slice tune loop with A/B
  compare, center finding with a `write_center` sweep montage, subprocess
  full-volume runs, output browsing) plus a tomostream-style live streaming
  mode, with the prioritized list of library additions it requires.

### Added

- **`ext_pad` — truncated-projection support extension for iterative
  methods.** Real samples routinely overhang the field of view, so the
  projections don't end at zero; an iterative forward model whose support is
  the detector-width grid then dumps that inconsistency into a huge FOV-edge
  ring and background offset that swamp the (intact) interior — on real
  800-wide data a 10-iteration CGLS correlated only 0.56 with fbp full-frame
  while agreeing 0.99 in the interior. With `ReconParams::ext_pad` (CLI
  `--ext_pad`, config `ext_pad`, GUI Tune "extend FOV" — on by default in the
  GUI) the sinogram is edge-replicate extended by `ncols/4` per side, the
  solve runs on the wider grid, and the central crop is returned; the wrapper
  sits above the backend dispatch so CPU/CUDA/wgpu see the identical extended
  problem. Real-data result: CGLS-10 full-frame correlation with fbp 0.56 →
  0.996, interior display contrast restored to fbp's level. Off by default in
  the library (tomopy-parity semantics unchanged; ~2.25× cost per iteration).

### Changed

- **Analytic reconstructions now emit the physical μ** — the shared
  `make_fbp_filter` base ramp is the physical `|ω|` inversion filter (peak
  `0.5` at Nyquist) instead of tomopy/tomocupy's doubled ramp (peak `1`), and
  `recon::gridrec`'s `ramp_scale` drops its matching empirical `×2`. Every
  analytic method (FBP, linerec, fourierrec, lprec, gridrec, on all backends)
  therefore reconstructs the attenuation μ per pixel-unit rather than `2×μ` —
  the same scale the iterative solvers converge to, so fbp→iterative
  warm-starts are now scale-consistent. **All analytic output amplitudes are
  halved**; downstream code that hard-coded the old scale (rescale windows,
  8/16-bit export ranges) must adjust. Cross-method and cross-backend ratios
  are unchanged (both sides halve). Pinned by `tests/analytic_amplitude.rs` (a
  unit disk reconstructs to core mean ≈ 1.0). This is a deliberate departure
  from both upstreams, whose absolute analytic amplitude is itself
  convention-dependent (tomopy gridrec ≈ 1.16×μ, tomocupy ≈ 4/π×μ).
- **Iterative reconstructions now converge to the physical μ** — the
  forward/back-projector pair used by the iterative solvers is the plain
  line-integral Radon transform `W` and its pure adjoint `Wᵀ` on every
  backend (CPU, CUDA, wgpu); the `π/nproj` FBP angular-quadrature weight
  previously baked into both operators is now passed by the *analytic* FBP
  call sites only, where it belongs. A converged SIRT/CGLS/MLEM/… solve of
  `W x = p` therefore lands on the attenuation per pixel-unit (pinned by the
  new `tests/iterative_amplitude.rs` against an analytic disk sinogram)
  instead of `(nproj/π)·μ` — e.g. ≈ 143× smaller at 450 projections —
  matching ART/BART, which always solved the ungained ray equations. Iterate
  trajectories are unchanged up to that overall scale for the self-scaling
  methods (SIRT/CGLS/MLEM/OSEM/PML/OSPML); for `grad`/`tikh`/`tv` the fixed
  step and regularization now act on the physical scale, so hand-tuned
  `reg_par` values from before may need retuning. The grad/tv host and CUDA
  gain-compensation machinery (`adj_scale`/`fwd_gain_inv`) is deleted.
  Analytic (FBP/gridrec/…) outputs are unchanged.
- **GUI preview autoscale is percentile-robust** — image colormaps scale to
  the 0.5–99.5 % range instead of the absolute min/max, so a handful of
  extreme pixels (e.g. the FOV-edge ring iterative methods produce on
  truncated-FOV data) no longer own the whole gray range and flatten the
  interior structure.
- **`sift-center` is pure Rust** — `find_center_sift` now runs on the
  `lowe-sift` crate instead of the `opencv` binding, dropping the system
  OpenCV + clang build requirement entirely. The uint8 normalization stays
  bit-exact vs numpy; the SIFT stage is an independent implementation of
  Lowe's paper, so recovered shifts land within ~0.034 px and the center
  within 0.008 px of the cv2 golden (tolerances in `sift_center_parity`
  updated from float-noise to algorithmic bounds). The feature now needs
  rustc ≥ 1.92 (above the 1.82 MSRV, which is unchanged for default builds).
- **`tomoxide-gui` builds with `sift-center` on by default** (alongside
  `cuda`), so the Center screen's SIFT method is always available with no
  extra system packages. A feature-gated smoke test pins the SIFT center
  call chain end-to-end.
- **HDF5/TIFF writers are zero-copy** — `H5Writer`/`TiffWriter` handed each
  chunk through an elementwise gather copy before writing; a standard C-layout
  chunk (what every driver produces) is now passed straight to the write call
  (`as_slice`), with the gather kept only as a non-contiguous fallback. The
  gather was the H5 writer's dominant cost (~2× the raw file write at 512³).
- **The HDF5 reconstruction output is finalized without `fsync`** — new
  `VolumeWriter::finalize` hook (called once by every driver on the success
  path); `H5Writer` implements it with rust-hdf5 0.3.1's `close_no_sync`
  (complete, valid HDF5; durability left to OS page-cache writeback, matching
  the TIFF/Zarr writers). Dropping the writer without `finalize` (error paths)
  still closes durably. Also removes the H5 writer's per-chunk `flush()`
  (a documented no-op in rust-hdf5) and its stale "durable partial output"
  doc claim. Combined effect on a 512³ fbp streaming recon (1 GPU): h5 output
  1.61 s → 0.84 s, on par with tiff (0.83 s), bit-identical output.
  rust-hdf5 0.3.2 supplies the second half of that win: its `create` no longer
  `ftruncate`s a brand-new empty file — that truncate armed ext4
  `auto_da_alloc`, whose implicit writeback inside the final `close(2)`
  (~325 ms at 512³) silently defeated `close_no_sync`.
- **The HDF5 writer unlinks a stale output before creating** — re-running a
  reconstruction over an existing `.h5` output now lands on a fresh inode
  instead of truncating the old file, so the overwrite rerun no longer pays
  the `auto_da_alloc` writeback either (was +0.44 s end-to-end at 512³). The
  unlink trades away rust-hdf5's lock-before-truncate protection for this one
  regenerable output file.
- **The Zarr writer emits `<f4` chunk bytes zero-copy** on little-endian
  targets (bytemuck safe Pod cast — bytemuck core is now an unconditional
  dependency, its `derive` feature still gated behind `gpu-wgpu`); the
  per-element `to_le_bytes` gather remains only for big-endian targets and
  non-contiguous callers.

### Fixed

- **CUDA analytic reconstruction of a single slice was silently all-zero**
  (and an odd Fourierrec slice count a hard error). The z-bilinear
  back-projection kernel samples slice pairs, so it needs a ≥2-slice batch,
  and `cfunc_fourierrec` packs slice pairs, so it needs an even one — but a
  1-slice job (GUI preview, `recon --start_row R --end_row R+1`) built the
  streaming handle at capacity 1 and the one-shot path handed the kernels the
  raw count. Both now pad the batch with zero rows up to the kernel domain
  (≥2, even for Fourierrec) and drop the pad rows from the output, reusing
  the existing partial-chunk machinery; `FourierReconstruct::reconstruct`
  likewise zero-pads an odd stack instead of erroring. Multi-slice outputs
  are unchanged (single-row CLI recon is bit-identical to the same row of a
  multi-row run).
- **CUDA iterative reconstruction of a single slice was garbage** (the same
  batch-domain family as the analytic fix above, via the other kernel pair):
  the device-resident solvers and the `FilteredBackproject`/`ForwardProject`
  wrappers share the z-bilinear projection kernels, so a 1-slice problem
  forward-projected to zero and the solve iterated on nothing — a GUI Tune
  preview of sirt/tv showed garbage. `IterativeReconstruct::solve` and both
  wrappers now duplicate the slice into a 2-slice problem (exact: the z-interp
  weights sum to 1 on identical rows, and EM ratios stay finite where zero-pad
  rows would 0/0) and drop the duplicate; the 1-slice solve equals the same
  slice of a multi-slice solve.
- **fourierrec output was uniformly π·nd² smaller than every other method**
  (read as "all zeros" on real data — ~10⁻⁸ at nd = 800) on all three
  backends: the deapodization missed the Δθ = π/nang angular quadrature
  weight and compensated the inverse-FFT normalization against the wrong
  reference amplitude. Invisible to every parity test because they compare
  Pearson-style (scale-invariant) or fourierrec-to-fourierrec; the best-fit
  fbp/fourierrec amplitude is now pinned ≈1 by a regression test. Host
  (`phi_amp = π·nd²/nang`), CUDA (`divphi` ×π/4 over the unnormalized cuFFT
  inverse), and wgpu (deapodize `norm = π/4`) all land on the unified
  fbp/tomopy scale; cross-backend ratios are unchanged.
- **gridrec disagreed with fbp/fourierrec on real data and sat on an
  arbitrary amplitude.** Three defects in one method: (1) its output was
  never masked to the detector-width disk, so gridding leakage outside the
  field of view dominated the frame (corr vs fbp 0.36 on real 800-wide data
  while agreeing 0.97 inside a 0.9-radius disk); (2) its radial FFT was
  zero-padded, so the nonzero borders of real (truncated-FOV/absorbance)
  projections became a hard step that rang across the FOV-edge annulus — it
  now edge-replicates the padding exactly like `FbpFilter::apply`; (3) its
  ramp weight and deapodization were unnormalized (the Kaiser–Bessel pair's
  W/I₀(β) constant was dropped and no polar density compensation applied),
  leaving a size-dependent scale (~2600× below fbp at n=128) — samples now
  carry `2π·|ρ|/nang` and the true KB constant, landing on the unified
  fbp/tomopy amplitude. Pinned by amplitude and truncated-projection
  regression tests; real 800-wide data now: gridrec↔fourierrec corr 0.992,
  gridrec↔ramp-fbp corr 0.963, scales within 3 %.
- **`recon --start_row/--end_row` was silently ignored by the whole-volume
  paths** (algorithms without a streaming handle — gridrec and the iterative
  set, everything on CPU/wgpu — plus `--algorithm` chains): a 1-row request
  read and wrote all rows. Both whole-volume branches now read only the
  requested detector-row band and write it at its global slice offset
  (matching the streaming/shard semantics); an explicit row range under
  laminography is rejected (the tilt couples all rows) instead of dropped.

### Documentation

- **`docs/BENCHMARKS.md` §10 — FBP vs iterative accuracy against a known-truth
  phantom.** Settles whether FBP's sharper *look* is accuracy or contrast using a
  synthetic Shepp–Logan phantom (piecewise-const + a TV-adversarial textured
  variant), inverse-crime-mitigated (generate at 2×, bin detector, recon at 1×),
  with a transmission-Poisson noise sweep. Finding: a properly-regularised
  iterative recon (TV, λ tuned to the noise) is closest to truth in every regime
  (≈ halves FBP NRMSE at 450 views), FBP is the noise-robust floor whose extra
  detail is contrast not accuracy, and unregularised fixed-iter CGLS is best at
  high SNR but worse than FBP under noise. Also documents that real-data dense
  references are method-dependent (FBP vs iterative disagree at r ≈ 0.73–0.81),
  qualifying the §1 quality-proxy caveat.

## [0.5.1] - 2026-07-02

### Changed

- **HDF5 reads are now rayon-parallel** (rust-hdf5 0.3.0 with the `parallel`
  feature: parallel chunk decode — deflate inflate + chunk gather). Read sits on
  the critical path of the streaming reconstruction (each row-chunk re-decodes
  the whole projection-chunked dataset), so this is a large end-to-end win. On a
  1200×512×512 `fbp` streaming reconstruction (1 GPU): gzip-chunked input
  73.5 s → 4.3 s (**17×**), uncompressed-chunked 9.5 s → 3.1 s (**3.1×**);
  reconstruction output is byte-for-byte identical (the decode is deterministic).

## [0.5.0] - 2026-07-02

Headline: the **wgpu (portable GPU) backend becomes device-resident end to end**,
closing most of the gap to CUDA — analytic recon, streaming, SIRT and the
gridding recons now keep their data on the GPU across the pipeline, and the
scatter kernels use native `f32` atomics (wgpu 30). The workspace is also now
**self-contained** (all dependencies resolve from crates.io) and gains
**CI on every branch push** across Linux, Windows and macOS.

### Added

- **CGLS reconstruction (`--algorithm cgls`).** Conjugate-gradient least squares
  (the standard algorithm; recurrence parity-checked against ASTRA's
  `CglsAlgorithm` but implemented independently — ASTRA is GPL-3.0, no ASTRA code
  is used): a Krylov solver of the same `‖Ax − b‖²` normal equations as SIRT/GRAD
  but with the optimal step and conjugate directions, so it reaches a given
  residual in far fewer iterations (≈4–30× vs SIRT). Runs on every backend via
  the generic solver, plus a CUDA device-resident fast path (per-slice dot/axpy
  kernels; one upload / one download across all iterations). Parameter-free;
  supports warm-start chaining. No built-in regularization, so it needs early
  stopping on ill-posed data.
- **Per-stage iteration budgets in algorithm chains.** An iterative stage in a
  `--algorithm` chain can carry a `:iters` suffix
  (e.g. `--algorithm fbp,sirt:30,tv:10`); stages without one fall back to
  `--num_iter`. Analytic stages reject the suffix. Lets a chain spend, say, 30
  SIRT iterations then 10 TV iterations in one run — previously every stage shared
  a single `--num_iter`.
- **Continuous integration (GitHub Actions).** Runs on every branch push and
  every pull request: `rustfmt`, then a per-platform matrix — Linux/x86_64,
  Windows/x86_64, macOS/arm64 — each running `clippy -D warnings`, the test
  suite and doctests on the default (CPU) build plus a `gpu-wgpu` compile-check,
  and a Linux job that type-checks the CUDA FFI bindings without an nvcc toolkit.

### Changed

- **wgpu backend upgraded to wgpu 30** (from 23). Enables `SHADER_FLOAT32_ATOMIC`
  (Vulkan `VK_EXT_shader_atomic_float`); devices without it fall back to the
  portable compare-exchange emulation automatically.
- **wgpu reconstruction is now device-resident.** A fused analytic path keeps the
  filtered sinogram on-GPU (fbp/linerec/fourierrec); device-resident streaming
  reconstructs chunk-by-chunk (fbp/linerec/fourierrec and lprec, the latter
  caching its log-polar grids across chunks); SIRT keeps volume/sinogram resident
  across iterations; the FBP filter and the fourierrec/lprec gridding run on the
  GPU. The scatter kernels (forward projection, fourierrec gather/wrap, lprec
  gather) use native `f32` `atomicAdd`, replacing the CAS-emulation penalty —
  forward projection ~6.6× and SIRT ~6.4× faster (same-build A/B). Net effect:
  fbp/linerec reach CUDA parity, SIRT and lprec beat CUDA on this hardware.
- **Shared host precompute sped up.** lprec precomputes its FFTs on the CPU
  (rustfft) instead of a GPU round-trip (wgpu and CUDA), `build_grids` coordinate
  loops are parallelised with rayon, and the CPU `minus_log` prep is parallelised
  across the projection volume.
- **`xraylib` (the optional `beam-hardening` dependency) is now sourced from
  crates.io** instead of an out-of-repo path dependency, so a fresh clone / CI
  resolves the workspace without a sibling checkout. Local development against a
  sibling checkout can override via `[patch.crates-io]`.

### Fixed

- **wgpu forward projection** now applies the `π/nproj` adjoint gain, so the
  `{A, Aᵀ}` pair is matched — fixes `forward_project` output scale and the
  iterative solvers built on it (SIRT).
- **stripe TI block-size division** uses `checked_div` (satisfies
  `clippy::manual_checked_ops`, new on stable Rust 1.96).
- **HDF5 test fixtures** were swallowed by the repo-wide `*.h5` gitignore and
  never committed; they are now tracked so the test suite runs on a fresh clone.

## [0.4.0] - 2026-07-01

Two headline themes. First, the **full iterative reconstruction suite now runs on
the GPU**, device-resident (one upload / one download across all iterations).
Second, a **cross-backend convention unification**: CUDA analytic reconstruction
now matches the **CPU/wgpu (tomopy) convention** in both orientation and
amplitude, replacing 0.3.0's deliberate CUDA-matches-tomocupy parity. The CLI
also gains the full preprocessing / iterative / filter composition surface and a
live TOML config.

> **Behaviour changes — read before upgrading.** CUDA analytic output changes
> *orientation* (the tomocupy vertical flip is removed) and *amplitude* (now
> tomopy scale, not tomocupy's) relative to 0.3.0. If you depended on CUDA output
> matching tomocupy, this is a breaking change.

### Added

- **GPU iterative reconstruction suite (device-resident).** `ForwardProject for
  CudaBackend` (an exact adjoint of the `cfunc_linerec` back-projector) unlocks
  tomopy's iterative family on CUDA via the backend-generic solvers. `sirt`,
  `mlem`, `osem`, `ospml_hybrid`/`ospml_quad`, `pml_hybrid`/`pml_quad`, `grad`,
  `tikh`, and `tv` keep the volume and sinogram resident on the device across all
  iterations (H2D once, D2H once, fused per-iteration kernels), 1.3–11.4× faster
  than a per-iteration CUDA loop and 51–95× faster than CPU at 512²; output
  matches the host solvers. `art`/`bart` run on CUDA via shared row-action
  geometry, bit-identical to CPU.
- **Warm-start / algorithm chaining.** `ReconParams.init` seeds a solver from a
  prior volume, so an analytic result can warm-start an iterative refinement
  (e.g. `fbp` → `sirt` converges in fewer iterations). Available across the
  iterative suite on both the host and the CUDA device-resident path.
- **CLI preprocessing / iterative / filter knobs + live config.** `recon` and
  `recon_steps` gain `--filter`, `--remove_stripe`, `--retrieve_phase` (with the
  phase physics flags), `--num_iter`, `--reg_par`, and the per-method stripe/phase
  parameters (`--fw_*`, `--ti_*`, `--sf_size`, `--vo_*`). `--config` (a
  `tomoxide init` TOML) now actually drives reconstruction, with precedence
  `flag > config > default`. `--algorithm a,b` chains stages (warm-start) on the
  whole-volume path.

### Changed

- **CUDA analytic orientation → CPU/tomopy.** The tomocupy y-flip is removed from
  `cfunc_linerec` (back-projection storage index) and `cfunc_fourierrec` (a clean
  output-row flip in `divphi`), so CUDA emits the CPU/wgpu handedness. Back- and
  forward-projectors flip together, so they remain a discrete transpose.
- **CUDA analytic scale → CPU/tomopy.** The `cfunc_linerec` back-projection
  constant `4/nproj` (tomocupy) becomes `π/nproj` (tomopy); the CUDA-only `½` FBP
  filter gain in `build_filter_w` is removed; the CUDA `fourierrec` divides its
  unnormalized cuFFT inverse by `(2n)²` to match the CPU's normalized inverse FFT.
  Net: `cuda/cpu ≈ 1` for `fbp`/`linerec`/`fourierrec`/`lprec`.
- **CPU forward projector is now a true adjoint.** `sim::project` is scaled by
  `π/nproj` so the CPU `{A, Aᵀ}` pair is symmetric at one scale (matching the CUDA
  pair), keeping the iterative solvers well-posed cross-backend. The fixed-step
  `grad`/`tv` solvers gain-normalize the data residual (`nproj/π`) so their
  conditioning is unchanged by the forward-scale change. **`sim::project` output
  values change by `π/nproj`.**

### Not changed (documented exceptions)

- **Laminography is excluded from the unification.** The CUDA lamino path
  (`cfunc_linerec` tilted back-projector) and the CPU `recon::lamino` path (a USFFT
  algorithm) are *different reconstruction algorithms* with different filter
  frameworks, so they are not scale-comparable (measured `cuda/cpu ≈ −0.89`, a sign
  flip plus a filter-gain difference). Each is validated against its own reference
  (CUDA vs tomocupy, CPU vs wgpu); both stay y-flipped, consistently. Do not
  warm-start one lamino backend from the other.
- **`gridrec`** is backend-agnostic (`recon::gridrec` over the `Fft` capability),
  already identical across backends — unaffected.

### Fixed

- **CPU `osem`/`ospml`/`pml` crashed on multi-slice reconstruction.** Their
  subset builder (and the CPU back-projector) indexed with `select(Axis(1))`,
  which is non-contiguous for any `nz > 1`; both now take standard-layout arrays,
  so these methods work for real multi-slice volumes.
- **`tomoxide init` template serialization.** The phase-physics config fields are
  now `f64`, so the template writes clean decimals (`pixel_size = 0.0001`) instead
  of f32→f64 promotion noise (`0.00009999999747…`).

### Removed

- **`docs/ROADMAP.md`.** Superseded by this changelog and the per-release notes;
  all references were removed.

### Documentation

- Rewrote the README to the working v0.4.0 state: accurate two-crate layout,
  feature-gated build instructions, and a detailed command-line usage section
  (all subcommands, options, config precedence, chaining, multi-GPU, examples).
- Added an iterative-algorithm selection guide and a chaining (warm-start)
  section to `docs/ALGORITHMS.md`, and documented the convention unification
  across `docs/ARCHITECTURE.md` and `docs/ALGORITHMS.md`.

## [0.3.0] - 2026-06-30

A filter correctness / convention release. The CUDA backend now matches
tomocupy's analytic reconstruction in **absolute amplitude** (0.2.0's CUDA
analytic output was 2× too large), the **default FBP filter switches to
`parzen`** to match tomocupy, and the CUDA filter ramp is ported to tomocupy's
exact degree-12 quadrature *shape* (not just its scale). Also adds GPU
laminography.

> **Behaviour changes — read before upgrading.** Both the default-filter switch
> and the CUDA amplitude halving change reconstruction *values* relative to
> 0.2.0. See **Changed** below for how to restore the old behaviour.

### Added

- **GPU laminography.** `recon --lamino_angle` runs the analytic linerec path
  with a tilted rotation axis on CUDA (port of tomocupy's scalar-`phi` linerec),
  verified against tomocupy on real leaf data (Pearson 0.99997).
- **tomocupy `_wint` quadrature ramp** (`backend::wint_ramp`) — a faithful port
  of tomocupy's degree-12 Newton–Cotes interpolatory quadrature (inverse
  Vandermonde weights over overlapping order-point windows + the 40-sample
  endpoint correction), so the CUDA analytic filter reproduces tomocupy's ramp
  *shape* bit-for-bit, closing a ~1% straight-line-ramp gap near DC/Nyquist.
- **`backend::RampShape`** — selects the base ramp per backend (`Linear` =
  tomopy for CPU/wgpu, `Wint` = tomocupy for CUDA).

### Changed

- **Default FBP filter is now `parzen`** (was `ramp`), matching tomocupy's
  default. Reconstructions that used the default filter will be smoother than
  under 0.2.0; set `filter_name = FilterName::Ramp` (library) to restore the
  sharp ramp.
- **CUDA analytic output amplitude halved to match tomocupy.** `build_filter_w`
  used `1.0/pad`, making every CUDA analytic method
  (fbp / linerec / fourierrec / lprec / laminography, f32 + fp16) exactly 2×
  tomocupy. It now uses `0.5/pad`: CUDA matches **tomocupy** in absolute
  amplitude while the CPU/wgpu path still matches **tomopy**. The documented
  CUDA↔CPU convention scales become `2/π` (fbp/linerec), `≈2·n²` (fourierrec),
  `½` (lprec); gridrec stays `1`.
- **Per-backend filter ramp shape.** The base ramp is no longer shared between
  backends: CPU/wgpu build tomopy's linear ramp, CUDA builds tomocupy's `_wint`
  quadrature ramp. Apodization, padding, the `≥0` clamp, DC doubling and the
  symmetric FFT layout remain shared in `make_fbp_filter`; all tomocupy-specific
  filter behaviour (the `½` gain and the `_wint` shape) now lives on the CUDA
  side.
- **API:** `backend::make_fbp_filter` gained a `RampShape` argument.

### Fixed

- `docs/ARCHITECTURE.md` §4.1: lprec's CUDA/CPU amplitude-scale row corrected
  `1` → `½` (stale since the `½` normalization landed; the parity test already
  undoes the `½`).

## [0.2.0] - 2026-06-30

This release turns the CUDA backend into a full streaming, multi-GPU
reconstruction engine that beats tomocupy end-to-end on every analytic method,
adds half-precision (fp16) and on-device stripe removal, makes the portable
wgpu backend work at realistic volume sizes, and consolidates the workspace into
a single library crate.

### Added

- **New algorithms / preprocessing**
  - Vector tomography reconstruction (port of tomopy `vector.c`), bit-exact vs
    the reference.
  - Beam-hardening correction backed by xraylib (`beam-hardening` feature).
  - `find_center_sift` via OpenCV (`sift-center` feature).
  - Final two deferred preprocessing paths: `stripe_ti` with `nblock > 0`, and
    median-based `normalize`.

- **CUDA backend**
  - GPU FBP back-projection (`cfunc_linerec`), Fourier reconstruction
    (`cfunc_fourierrec`), and the on-device FBP filter (`cfunc_filter`) — the
    full analytic FBP/fourierrec path runs on-device with no per-stage host
    copies.
  - cuFFT-backed `Fft` capability, unlocking gridrec, lprec, and phase
    retrieval on the GPU.
  - Multi-GPU per-slice reconstruction (device-pinned pools) and multi-GPU
    fused analytic reconstruction (Fbp/Linerec).
  - Memory-aware streaming to lift the large-volume GPU ceiling, with an async
    double-buffered H2D∥compute∥D2H pipeline for the fused Fbp/Linerec path.
  - Half-precision (fp16) analytic reconstruction path, including out-of-core
    fp16 Fbp/Linerec via a tiled async pipeline and device-resident fp16
    fourierrec streaming.
  - Device-resident streaming reconstructors for fourierrec and lprec (one
    upload / one download per chunk; GPU gather/scatter/prefilter for lprec).
  - On-device stripe removal in the streaming raw path: Titarenko,
    Fourier-Wavelet, and Vo all-stripe.

- **CLI**
  - `--dtype float32|float16` flag for `recon` / `recon_steps`.
  - `--save-format` and a per-chunk `VolumeWriter::reserve` contract.
  - `tune_chunk` subcommand to empirically pick the best-fitting pipeline chunk.
  - Multi-GPU z-shard fan-out for streaming `recon` (uses all GPUs).
  - Auto-pipelined GPU recon for analytic methods.

- **Pipeline / IO**
  - Out-of-core streaming reconstruction (`read_chunk`, `ReconSteps::run`) and a
    pipelined read‖compute‖write variant.
  - TIFF writer that streams per-chunk volumes by global index.

- **Tests / docs**
  - Cross-backend parity test for the tomocupy output convention
    (`tests/cuda_cpu_convention_parity.rs`) and `docs/ARCHITECTURE.md` §4.1
    documenting the CUDA analytic orientation/scale convention.
  - wgpu dispatch-overflow regression test.

### Changed

- Consolidated the nine library crates into a single `tomoxide` crate.
- Parallelized FFT-based reconstruction on the CPU (bit-exact), with
  backend-owned per-slice scheduling via `Fft::for_each_slice`.
- CUDA performance work: thread-local cuFFT plan cache + per-thread default
  stream; cached cos/sin(theta) in shared memory; hardware-texture
  back-projection for the fp16 path; sliding-window Vo median filter; pinned-host
  D2H for streaming downloads; lprec log-polar FFT switched from C2C to in-place
  R2C/C2R (2× faster, half the memory); recycled output volume buffers across
  streaming chunks; HDF5 chunks read into pinned host buffers for direct-DMA
  H2D.
- Bumped `rust-hdf5` to 0.2.27 for coalesced hyperslab reads.

### Fixed

- **wgpu**: fold the 1-D dispatch into a 2-D grid to clear the
  65535-workgroup-per-dimension cap, and request the adapter's real limits
  instead of the WebGL downlevel defaults — wgpu now reconstructs
  512²/1024²/2048² volumes.
- **CUDA**: never hand `cfunc_linerec` a <2-slice z-chunk (the fused path
  returned zeros); z-tile the composed FBP filter to lift the lprec large-volume
  ceiling; bound the per-slice pool to the in-flight cap so cuFFT plans cannot
  exhaust VRAM; harden vendored `cfunc_filter`/`cfunc_fourierrec` against OOM
  (no more SIGSEGV).
- **IO**: guarantee `Tomo::to_layout` yields a C-contiguous array; fix
  `TiffWriter::write_chunk` underflow on an inverted range.
- **recon**: cfg-gate `LP_NSPAN` to the `cuda` feature to clear a dead-code
  warning.

## [0.1.0] - 2026-06-25

Initial release: tri-backend (CPU / CUDA / wgpu) tomographic reconstruction
toolkit porting tomopy and tomocupy, with the CPU `libtomo` algorithm set and
the first CUDA FBP back-projection.

[Unreleased]: https://github.com/physwkim/tomoxide/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/physwkim/tomoxide/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/physwkim/tomoxide/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/physwkim/tomoxide/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/physwkim/tomoxide/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/physwkim/tomoxide/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/physwkim/tomoxide/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/physwkim/tomoxide/releases/tag/v0.1.0
