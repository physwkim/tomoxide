# tomoxide — GUI design (`tomoxide-gui`)

This document is the design for a desktop GUI for tomoxide: a new crate
`crates/tomoxide-gui` built on [rsplot] (egui 0.34 + wgpu scientific plotting,
a silx port) for every scientific view, and on rsplot's sibling crate `rsdm`
(a PyDM-style EPICS display/data-engine layer) for live PVAccess data. It
covers three workflows in one application:

1. **Offline** — open a DXchange HDF5, tune parameters on a single-slice
   preview, find the rotation axis, reconstruct the full volume, browse the
   result.
2. **Live streaming** — subscribe to a tomoScanStream-style PVA projection
   stream and reconstruct ortho-slices continuously while the operator nudges
   the rotation center (the tomostream operating model).
3. **Spectroscopic (XANES)** — take a stack of reconstructed volumes acquired
   across an absorption edge, fit each voxel's spectrum, and map the chemical
   state (edge-shift / white-line energy) in 3-D.

[rsplot]: https://github.com/physwkim/rsplot

**Intellectual-property note.** The workflow of the commercial *Octopus
Reconstruction* package (single-slice tune loop, candidate-montage parameter
evaluation, pre-flight resource checks, stack rescaling) and the operator
model of APS *tomostream* (live ortho-slices, live center tweak,
click-to-set) are used as **concept references only**. No text, artwork,
screenshots, or screen layouts are copied; every screen here is an original
composition of rsplot widgets around tomoxide's actual API surface.

---

## 1. Application shell

An `eframe` application (wgpu renderer — required by rsplot) with:

- a **left mode rail** (`egui::SidePanel`) selecting one of seven modes:
  **Data · Tune · Center · Run · Output · Live · XANES**;
- a **persistent bottom log pane** (collapsible): every job with its
  parameters, timing, and outcome — the session history;
- a **status bar**: resolved backend (`Engine::name()`), GPU inventory
  (`cuda::selected_devices()` / `cuda::device_name()`), last recon time.

No docking framework in v1 — plain egui panels; rsplot's detached windows
cover "pop this view out". All modes share one `Project` (the parameter
model, §5) and one recon worker (§4).

---

## 2. Screens

### 2.1 Data — dataset browsing

- Open a DXchange HDF5 via a native file dialog; the worker calls
  `open_dxchange(path)`, `read_sizes()`, `read_theta()`.
- **Metadata card**: nproj / nz / nx / nflat / ndark, theta start/end/step,
  raw size estimate.
- **Projection browser**: rsplot `StackView` over the projections. Its
  `FrameLoader` trait is `Send + Sync` but rust-hdf5's `H5File` is `!Send`,
  so the loader is a *channel client*: `load(i)` sends a request to the
  worker thread (which owns the reader) and blocks on the reply. StackView's
  lazy loading keeps this cheap. Flat/dark frames are viewable through the
  same view via a source selector.
- **Theta plot**: `Plot1D` of the angle array (irregular or missing angle
  sets are visible at a glance).
- **Sinogram inspector**: pick a detector row → `ImageView` of the
  `[nproj × nx]` sinogram (side histograms/colorbar off; an `ImageView`, not a
  bare `Plot2D`, so the crosshair readout can show the pixel value under the
  cursor via `value_changed` — the silx `PositionInfo` "Data" column).

### 2.2 Tune — the single-slice preview loop

The core interaction (the Octopus tune-loop concept mapped onto tomoxide):

- **Left**: a decimated projection thumbnail (`Plot2D`) with a draggable
  horizontal line selecting the preview slice `z`.
- **Middle**: the parameter panel — tabbed groups mirroring the CLI's
  conceptual grouping, with the Octopus conventions of greying out
  irrelevant fields and "auto" checkboxes mapping to `None`:
  - **Geometry** — rotation axis (auto checkbox → `None` = midline),
    `lamino_angle`;
  - **Algorithm** — algorithm combo including the chain syntax
    (`fbp,sirt:30,tv:10`) via a small chain builder; filter combo
    (ramp/shepp/cosine/cosine2/hamming/hann/parzen); `num_iter`/`reg_par`
    greyed out for analytic algorithms;
  - **Prep** — stripe method + only the selected method's parameters enabled
    (`fw_*`, `ti_*`, `sf_*`, `vo_*`); phase-retrieval method + physics
    (`pixel_size`, `propagation_distance`, `energy`, `alpha`, `db`, `w`);
  - **System** — backend (Auto/Cpu/Cuda/Wgpu), dtype (f32/f16), chunk size
    with the tuned-cache value shown, per-slice timing of the last preview
    and the extrapolated full-volume estimate.
- **Right (dominant)**: the current single-slice preview as an rsplot
  `ImageView` — its crosshair reads out the pixel value under the cursor
  (`value_changed`, the silx `PositionInfo` "Data" column) and its interactive
  colorbar (value histogram + draggable `vmin`/`vmax`) sets the display range.
  A "Pin" button (or holding Space) snapshots the current result together with
  its parameters and switches the pane to rsplot `CompareImages` — the current
  preview as image B against the pinned one as image A; split/subtract/A-B modes
  give instant parameter comparison, with rectangle-ROI stats and line profiles.
- **Auto-recon toggle**: debounced (~300 ms) re-reconstruction on any
  parameter change — viable because one analytic slice on the GPU is
  sub-second.
- **λ sweep (L-curve regularization tuner)**: shown only for algorithms whose
  `reg_par[0]` is a regularization strength λ (`tv`, `grad`, `tikh`,
  `pml_*`, `ospml_*`). Reconstructs the current slice once per λ across a
  log-spaced grid (all other parameters fixed), then plots the **L-curve** —
  data residual `‖A x − b‖₂` (the exact fidelity term the solver minimizes)
  against the reconstruction's isotropic TV seminorm (roughness), both in
  log₁₀. A floating window shows the per-λ montage (`ImageStack`) over the
  L-curve (`Plot1D`); the corner is highlighted as the suggested λ and a click
  snaps to any point, "Use selected λ" writes it back to `reg_par[0]`. The
  guidance is deliberately the L-curve corner, **not** a sharpness auto-pick:
  λ is not solvable like the rotation axis, sharpness decreases monotonically
  with λ, and on real data there is no ground truth — so the pick stays the
  user's (the accuracy rationale is [`BENCHMARKS.md` §10](BENCHMARKS.md#10-fbp-vs-iterative-accuracy-against-a-known-truth-phantom)).

Execution path: the worker runs
`ReconSteps::run_streaming_pipelined_range(z, z + 1, …)` into the new
`InMemoryWriter` (§6 #2) for analytic algorithms without phase retrieval.
Phase retrieval is row-coupled and iterative methods want more context, so
those previews run `ReconSteps::run` over a **row band** `[z − m, z + m]`
through the new `RowBandReader` adapter (§6 #5), displaying the center row
(m sized from the Paganin kernel support). Until #5 lands, phase/iterative
previews are disabled with a tooltip, not silently wrong.

### 2.3 Center — finding and evaluating the rotation axis

Three complementary tools on one screen:

1. **Auto buttons** — Vo (`find_center_vo`, the workhorse), Entropy
   (`find_center`), Phase-correlation (`find_center_pc`), and SIFT
   (`find_center_sift`, shown only when the `sift-center` feature is
   compiled in). Each runs on the worker against the current preview slice
   and proposes a value with an explicit "accept" step.
2. **Sweep montage** — the Octopus parameter-evaluator concept, backed by
   the existing `recon::center::write_center(tomo, theta, backend,
   cen_range, ind, mask, ratio) -> (Vec<f32>, Array3<f32>)`: the user sets
   center ± range ± step; the candidate reconstructions become a `StackView`
   (frame label = candidate center) next to a `Plot1D` of a per-frame
   sharpness metric (image standard deviation) with click-to-pick. A
   "refine" button re-runs the sweep centered on the pick with step/4.
3. **Fine tweak** — ±0.5 / ±0.25 px nudge buttons that immediately re-run
   the single-slice preview. The same control reappears in Live mode
   (tomostream's CenterTweak), where it applies on every loop.

### 2.4 Run — full-volume reconstruction

- **Output**: path, format (tiff/h5/zarr), dtype.
- **Pre-flight panel** (the Octopus System-tab concept): disk estimate
  (`nz·ny·nx·dtype` vs. free space), chunk from the CLI's tune cache
  (keyed by file/algorithm/dtype/GPU), wall-time extrapolated from the last
  preview — each with a green/red feasibility indicator. GPU picker
  (checkbox per device).
- **Execution is always a subprocess.** The GUI spawns `tomoxide` (the CLI)
  — one process per selected GPU with `--start_row/--end_row` z-shards,
  exactly the CLI's own multi-GPU fan-out; one process for CPU or a single
  GPU too. Rationale: a single code path, the per-process cuFFT-plan/VRAM
  isolation is already the library's deliberate design, cancellation is
  process kill (no library hook needed on this path), and a CUDA OOM cannot
  take down the GUI. Requires the CLI addition `--progress-json` (§6 #4);
  the GUI tails one JSON line per completed chunk from each child.
- **Live progress**: per-shard and aggregate progress bars, plus a
  latest-slice view (`Plot2D` with `try_update_image`, zoom preserved) when
  the output is tiff (per-slice files are cheap to load as they appear).
  For h5/zarr output only the progress bars are shown — single-container
  writers are not shardable and the container is not valid until finalize;
  this limitation is accepted and documented.
- **Cancel** kills the children; partial tiff output stays on disk and is
  noted in the log.
- **Batch queue** (phase 2): an ordered list of (dataset, recipe TOML) pairs
  run sequentially — each recipe being exactly the file the CLI consumes, so
  a queue tuned in the GUI is scriptable headlessly.

### 2.5 Output — result browsing

- `StackView` over the reconstructed volume with per-format `FrameLoader`s
  implemented in the GUI (tiff directory via the `tiff` crate; h5 through
  the worker-thread channel client; zarr chunk files directly).
- **3D**: rsplot `ScalarFieldView` (isosurface) over a worker-built
  downsampled copy (≤ ~256³).
- **Rescale/export** (the Octopus rescale concept): a volume histogram
  (`Plot1D` with a draggable min/max range) drives export of the stack to
  8/16-bit tiff on a common gray scale. Pure GUI-side conversion.

### 2.6 Live — streaming ortho-slice reconstruction

The tomostream operating model, in-process:

- **Connection panel**: PVA channel addresses for projection / dark / flat /
  theta (tomoScanStream conventions as defaults), connected through the
  `rsdm` data engine (its own tokio runtime). Frames land in a
  fixed-capacity **ring buffer** holding ~180° of projections plus theta and
  rolling dark/flat.
- **Recon loop** on the worker: wakes on new frames or any parameter change,
  assembles sinograms for the selected slice indices from the ring buffer,
  reconstructs, and publishes to the UI (`try_update_image` +
  `request_repaint`).
- **Views**: `Plot2D` panes with crosshair cursors marking the other slice
  positions, and **click-to-set** — clicking a position in one pane moves
  the other slice indices (tomostream's signature interaction, minus its
  NaN-line hack: rsplot draws real markers).
- **Parameters** (re-read every loop, tomostream semantics): center +
  tweak up/down (live nudging), filter, ring removal (none/fw), slice
  indices, Paganin group, Start/Abort, recon-time and buffer-fill readouts.
- **Honesty note**: tomoxide today reconstructs **Z slices** cheaply
  (analytic recon is per-slice independent), so live mode ships **Z-only
  first** — one or several horizontal slices. The X/Y ortho panes require
  new dedicated single-slice backprojection kernels (§6 #7; tomostream has
  `orthox`/`orthoy` CUDA kernels, tomoxide does not). Incremental
  ring-buffer recon (`obj += bp(new) − bp(old)`) is an optimization (§6 #8);
  v1 recomputes the displayed slices each loop, which analytic GPU recon
  sustains at interactive rates for typical detector widths.

### 2.7 XANES — spectroscopic chemical mapping

The post-reconstruction spectroscopy workflow of the existing `xanes_tools`
pipeline, brought in-process. Input is a **stack of reconstructed volumes, one
per photon energy** across an absorption edge (Fe / Co / Ni K-edge, …); output
is a per-voxel **chemical-state map** — the fitted white-line / edge energy —
modulated by the **edge jump** (absorber thickness/density).

- **Input.** Either (a) the **native product of the Run screen's batch
  queue** — one tomoxide output per energy (each a tiff directory / `.h5`
  holding a single `/exchange/data` / `.zarr`) plus the energy list, or
  (b) a **combined multi-energy HDF5** in the existing Python layouts
  (`reconstructions/{energy}` or `entry_1/{energy}/recon` + `energies`).
  Both are read as a lazy 4-D `(E, z, y, x)` stack via the Output screen's
  `VolumeSource` + `read_h5_sizes`/`read_h5_frame`, extended with the energy
  axis (§6 #12); per-energy frames load lazily through the worker channel.
  An **export writer to `reconstructions/{energy}`** covers the handoff to
  the external Python registration step, which consumes that layout. v1
  assumes the stack is **already registered** across energies (see honesty
  note).
- **Producing the stack — preprocessing and recon are unchanged.** The stack is
  the output of an **energy loop over the existing reconstruction path**, not a
  new one. Data preprocessing (flat/dark, −log, stripe removal, binning) stays
  exactly the current DXchange prep, and each energy is reconstructed by
  tomoxide's **existing CUDA streaming pipeline, used as-is**
  (`run_streaming_pipelined_range` — the same read‖compute‖write path the Run
  screen drives), with the rotation center reused from a reference energy or
  re-found per energy (the notebook's practice — the axis is shared, but the
  energy-dependent magnification shifts it slightly in pixels). The only new
  recon-side code is the outer energy loop; there is **no XANES-specific
  preprocessing or reconstruction kernel**.
- **Fit controls** (mirroring `xanes_fitting_3d.py`, one panel group each):
  - *Edge jump* = mean(last `n_post` energies) − mean(first `n_pre` energies)
    → a 3-D thickness map.
  - *Mask* (which voxels to fit): intensity threshold — absolute or percentile
    — over mean / max / post-edge / single-energy intensity; or the *advanced*
    pipeline (median filter → loose threshold → binary closing + hole fill →
    largest connected component).
  - *Smoothing* along energy: savgol / median / 3-point / boxcar with
    window + order.
  - *Fit*: quadratic vertex or Gaussian center over a `fit_points` window
    around each spectrum's max, restricted to an energy range → per-voxel peak
    energy; *concentration* = `(peak − startE)/(stopE − startE)` clipped to
    [0, 1].
- **Compute.** The fit is the existing **`txm_pal_core`** Rust core (per-voxel
  Levenberg–Marquardt + rayon, no GPU), reused as a plain library dependency
  (§6 #11). Voxel fits are independent, so the fit streams in **z-bands**:
  read the `(E, band, ny, nx)` slab, fit it, write the result rows, advance —
  bounded memory. The Python script's materialize-everything approach does
  not transfer: a 40-energy 500³ f64 stack is ~40 GB; a band is `E × ny × nx`.
  The port takes `f32` views (the recon output dtype), converting
  per-spectrum for the f64 LM solver. The fit runs on the rayon pool under
  the same `CancelToken` as recon jobs (checked between bands) — **not** on
  the frame-serving worker thread, so browsing stays responsive during a long
  fit; per-band progress streams back the same way `InMemoryWriter`'s
  `on_chunk` reports full-volume runs (§6 #2).
- **Result views**:
  - **2-D chemical maps**: `ImageView` of peak-energy / concentration /
    edge-jump for the selected Z (and an orthogonal Y cut) under the XANES HSV
    colormap — hue = concentration (red→yellow→green edge-shift ramp),
    value = alpha = normalized edge jump, unfitted voxels transparent. The
    RGBA mapping is pure arithmetic ported from the Python viewer.
  - **Histogram**: `Plot1D` of the fitted peak energies with draggable
    start / center / stop lines — the oxidation-window control; re-binning is
    the interactive threshold.
  - **Click-to-spectrum**: clicking a voxel in a map (rsplot `ImageView`
    value-query gives `(z, y, x)`) plots that voxel's raw spectrum
    `stack[:, z, y, x]` vs energy in a `Plot1D`, with the fitted peak marked —
    the core diagnostic of the Python viewer.
  - **3-D**: the central panel toggles between the 2-D slice browser and a
    **direct volume rendering** of the chemical map through rsplot's
    `VolumeRaycaster` — a GPU front-to-back alpha-composited ray-march (WGSL,
    orbit / pan / wheel-zoom) that replaces the earlier `ScalarFieldView`
    iso-surface plan with the Python viewer's VTK-style ray-cast. The transfer
    function reuses the **same viridis peak-energy colormap as the 2-D map** for
    hue, with fitted voxels opaque and unfitted (masked / out-of-window) voxels
    transparent; an opacity slider scales the composite. The volume is decimated
    to a bounded edge (≤192) so the RGBA8 texture upload stays small. (Alpha is
    fitted-voxel presence in this cut; hue = chemistry + alpha = edge-jump
    thickness is the next transfer-function refinement.)
  - **Save**: results HDF5 with `peak_energies` / `concentration` /
    `edge_jump` / `mask` / `energies` / `histogram`, matching the schema the
    existing `xanes_viewer_3d.py` reads, so the two viewers interoperate.

**Honesty note — registration and magnification are upstream in v1.** Two
XANES-specific corrections precede fitting and are **not** in the v1 screen:
(a) **energy-to-energy 3-D registration** — the reference pipeline uses
SimpleITK rigid Mutual-Information, which has no pure-Rust equivalent, so v1
consumes an **already-registered** stack (produced by the existing Python
`register_volumes.py`), with in-GUI registration a staged addition
(translation-only phase-correlation first — `txm_pal_core` already ships
`phase_cross_correlation` — then rigid MI; §6 #13). (b) **Zone-plate
magnification correction** (the energy-dependent focal length) is now an
**in-GUI toggle** (§6 #14): a side-panel option applies
`xanes::apply_magnification` per energy before fitting, with editable zone-plate
geometry and a live factor-range preview. Because the correction scales the `z`
axis it can't be applied to the streamed `z`-bands independently, so enabling it
reads the whole stack into memory and fits from the corrected copy; the default
path still streams a band at a time. Per-energy reconstruction itself is **not**
new work — it reuses the existing CUDA streaming pipeline as-is (see "Producing
the stack" above).

**Provenance.** The fitting math and workflow are the existing `txm_pal_core`
(Rust) and `xanes_tools` (Python) — reused, not re-derived; the
white-line / edge-shift method is standard XANES analysis. No third-party IP.

---

## 3. Architecture

### 3.1 In the repository, outside the workspace

rsplot is `edition = "2024"`, `rust-version = "1.92"`; the tomoxide workspace
is `edition = "2021"` with a declared MSRV of `1.82` (`rust-toolchain.toml`
pins `stable`, so day-to-day builds are on a current toolchain — the MSRV is
a compatibility promise, not the dev toolchain).

If `tomoxide-gui` were a workspace **member**, rsplot would enter the
workspace dependency graph, and cargo parses every member and path-dependency
manifest on *every* command — an `edition = "2024"` manifest is rejected by
cargo < 1.85, so the repository's effective MSRV would silently become 1.92
for all commands. `default-members` does not help; manifests are parsed at
workspace load regardless of build selection.

**Decision: keep the crate in the repo but exclude it from the workspace.**

- Root `Cargo.toml`: `exclude = ["crates/tomoxide-gui"]`.
- `crates/tomoxide-gui/Cargo.toml`: its own empty `[workspace]`,
  `edition = "2024"`, `rust-version = "1.92"`, its own committed
  `Cargo.lock`.
- `cargo build --workspace` at the root still parses and builds on a 1.82
  toolchain (the MSRV guarantee stays CI-checkable). The GUI builds with
  `cargo build --manifest-path crates/tomoxide-gui/Cargo.toml` (alias in a
  justfile/Makefile; separate CI job on stable).
- Accepted costs: a second lockfile, no `workspace.dependencies`
  inheritance, no `-p tomoxide-gui` from the root.
- **Convergence plan**: when the workspace MSRV is raised to ≥ 1.92, fold
  the crate into the workspace and delete the exclusion.

### 3.2 Dependencies

- `tomoxide = { path = "../tomoxide", features = ["config"] }` (§5).
- `rsplot` / `rsdm`: **crates.io releases** (`rsplot = "=0.5.0"` now; `rsdm`
  when Live/M3 lands), with a commented-out `[patch.crates-io]` block pointing
  at the sibling checkout for local development — CI stays reproducible from the
  registry without requiring the sibling repo. (Both were renamed from
  `siplot`/`sidm` and published to crates.io.)
- `eframe = "0.34"` with the wgpu renderer (rsplot deliberately does not
  depend on eframe itself; the app owns the window).
- **wgpu duplication**: tomoxide's optional `gpu-wgpu` backend pins wgpu 30
  while rsplot renders through egui-wgpu 0.34's re-export (a different
  major). The two coexist (types never cross — the recon `Engine` owns its
  own instance; rsplot renders through eframe's `RenderState`), but at a
  compile-time cost, so the GUI's default features enable tomoxide `cuda`
  and **not** `gpu-wgpu`.

### 3.3 Module layout

```text
crates/tomoxide-gui/
  Cargo.toml            # own workspace, edition 2024, rust-version 1.92
  src/
    main.rs             # eframe bootstrap (wgpu renderer)
    app.rs              # TomoxideApp: mode rail, status bar, log pane
    project.rs          # Project = tomoxide::config::Config + [gui] table
    state.rs            # dataset meta, A/B pins, montage, run status
    worker/
      mod.rs            # worker thread, Job/Event enums, channels
      jobs.rs           # OpenDataset, Preview, AutoCenter, CenterSweep,
                        # LoadFrame, Downsample3D
      runner.rs         # full runs: spawn CLI shards, tail --progress-json
    live/
      ring.rs           # ~180° projection ring buffer (+theta, dark/flat)
      source.rs         # rsdm engine + PVA channels -> ring buffer
      recon_loop.rs     # per-loop param snapshot + slice recon
    views/
      data.rs tune.rs center.rs run.rs output.rs live.rs
      params.rs         # shared tabbed parameter forms (grey-out logic)
      loaders.rs        # FrameLoader impls (channel-client H5, tiff dir, zarr)
```

---

## 4. Threading and data flow

```text
UI thread (eframe/egui, all rsplot widgets)
   │  Job ────────────────► worker thread (owns Engine/Backend + all H5 handles)
   │  ◄──────────── Event      constructs every DatasetReader on this thread
   │   + ctx.request_repaint() (H5File is !Send — the same factory-closure
   │                            discipline as run_streaming_pipelined)
   ├── full runs: spawn one tomoxide CLI child per GPU shard;
   │              tail --progress-json; cancel = kill
   └── live: rsdm engine (tokio) ─ PVA frames ─► ring buffer ─► worker loop
```

- **One long-lived worker thread**; jobs arrive over an mpsc channel,
  results return as `Event`s drained in `App::update`; the worker holds an
  `egui::Context` clone and calls `request_repaint()` after each event.
- **`!Send` H5 handles**: readers are only ever constructed and used on the
  worker (interactive path) or inside the library's own pipeline threads
  (which already take `Send` factory closures). The UI thread never holds a
  reader; `FrameLoader` impls proxy through the worker channel.
- **Backend residency**: `Engine::new(kind)` runs on the worker; CUDA
  context and FFT plans stay thread-local to it. Changing backend rebuilds
  the worker's engine.
- **Preview results**: `InMemoryWriter::write_chunk` doubles as the
  progress signal — each chunk is published to a shared buffer and the UI
  copies row-major `&[f32]` into `Plot2D::try_update_image` (zoom and
  colormap preserved).
- **Cancellation**, two regimes:
  - *in-process* (preview/sweep/auto-center/live): library `CancelToken`
    (§6 #1), checked between chunks. Until it lands, previews are short
    enough that "let it finish, discard by generation counter" is the
    stopgap.
  - *subprocess* (all full-volume runs): kill the children — a main reason
    the subprocess model is chosen for full volumes.
- **Rule**: in-process for everything interactive; subprocess for every
  full-volume run (single- and multi-GPU alike).
- **Live**: the loop snapshots parameters each iteration (so a center tweak
  applies "next loop", matching tomostream's re-read-every-loop semantics).
  Any structural change (filter, ring removal, Paganin) triggers a full
  recompute of the displayed slices.

---

## 5. Parameter model — the GUI recipe *is* the CLI config

The GUI's project file is the same TOML the CLI consumes, so a recipe tuned
in the GUI runs headlessly with `tomoxide recon --config recipe.toml` and
vice versa.

- Move `Config` from `crates/tomoxide-cli/src/config.rs` into the library as
  `tomoxide::config`, behind a **default-off `config` feature** gating
  optional `serde`/`toml` dependencies. The CLI re-exports it (§6 #3). While
  moving, add the fields the CLI currently exposes only as flags — `dtype`,
  `lamino_angle`, output path — so recipes are complete.
- GUI-only state (preview slice, A/B pins, colormap prefs, window layout)
  lives in a trailing `[gui]` table of the same file:
  `struct Project { #[serde(flatten)] config: Config, gui: GuiSection }`
  with `gui` last so the table serializes after the flat keys. The CLI's
  `Config::load` ignores unknown keys (`#[serde(default)]`), so a GUI-saved
  project is directly CLI-consumable. If `flatten` proves brittle with the
  `toml` crate, the fallback is a sidecar `name.gui.toml`.
- Presets are just bundled TOMLs; "Save preset / Load preset" lives on the
  Tune screen.

---

## 6. Required tomoxide additions (prioritized)

| # | Priority | Addition | Shape |
|---|----------|----------|-------|
| 1 | P0 | **Cancellation** | `CancelToken(Arc<AtomicBool>)` + `Error::Cancelled`; checked between chunks in the `ReconSteps` drivers, propagated to the I/O threads via channel close. Per-iteration checks inside iterative solvers are a follow-up. |
| 2 | P0 | **`io::InMemoryWriter`** | Implements `VolumeWriter`: `reserve` sizes a `Vec<f32>` (dims fixed at first chunk), `write_chunk` copies into the global range and invokes an optional `on_chunk: FnMut(usize, usize) + Send` — which is also the in-process progress callback (no new trait). |
| 3 | P0 | **Config into the library** | `tomoxide-cli/src/config.rs` → `tomoxide::config` behind a `config` feature (optional serde/toml); add `dtype`/`lamino_angle`/output fields; CLI re-exports. |
| 4 | P0 (CLI) | **`--progress-json`** | One flushed JSON line per completed chunk on stdout (`{"start":s,"end":e,"total":nz,"secs":t}`), implemented as a thin `VolumeWriter` tee in the CLI. The GUI's progress channel for subprocess runs. |
| 5 | P1 | **`io::RowBandReader`** | Wraps any `DatasetReader`, restricting to rows `[r0, r1)` (remapped `read_sizes`/`read_all`/`read_chunk*`). Enables banded previews for phase retrieval (row-coupled) and iterative methods without reading the whole file. |
| 6 | P1 | **Expose the Vo stripe variants** | `VoSort/VoFilter/VoLarge/VoDead/VoFit` exist in the library but are not wired through params/config/CLI parsing; wire them so the GUI's stripe combo is complete. |
| 7 | P2 | **Ortho X/Y slice kernels** | tomostream-parity `orthox`/`orthoy` single-slice backprojection (CUDA first, CPU reference): `recon_ortho(data, theta, center, axis, idx) -> Array2<f32>`. Without it, Live is Z-only and offline vertical cuts require full-volume recon. The largest new library work in this design. |
| 8 | P2/P3 | **Incremental ring-buffer recon** | `obj += bp(new) − bp(old)` over the circular buffer for the live loop; only needed if per-loop recompute misses the frame budget on large detectors. |
| 9 | P2 | **Dezinger prep** | tomostream exposes Dezinger (median radius 2/3/4); add to `prep` or document its omission from the live parameter set. |
| 10 | P3 | Nice-to-haves | `estimate_run(...) -> {host, vram, disk}` for the pre-flight panel (GUI-side arithmetic is the v1 fallback); `open_volume(path)` readers for reconstructed tiff/h5/zarr (GUI-side loaders are the v1 fallback). |
| 11 | P1 (XANES) | **XANES fitting module** | Port `txm_pal_core`'s PyO3-free core (`fit.rs` Levenberg–Marquardt quad/gaussian, `filter.rs` smoothing) into `tomoxide::xanes` behind a feature: per-voxel peak-energy fit taking banded `ArrayView4<f32>` (E,band,ny,nx — the recon dtype, converted per-spectrum for the f64 solver) + `ArrayView3<u8>` mask, CPU + rayon, `CancelToken`-aware between bands. Reuse, not rewrite — split a non-PyO3 math crate (or add an `rlib` target; `txm_pal_core` is `cdylib`-only today). Align ndarray (`txm_pal_core` 0.15 → tomoxide 0.16) and lift `levenberg-marquardt`/`nalgebra`/`savgol-rs`. |
| 12 | P1 (XANES) | **Multi-energy stack reader** | Energy-axis-aware loader yielding a lazy, z-band-readable 4-D `(E,z,y,x)` view over **either** a set of per-energy tomoxide outputs (tiff dir / `/exchange/data` h5 / zarr each — the batch queue's native product) + energy list, **or** a combined H5 (`reconstructions/{energy}` / `entry_1/{energy}/recon` + `energies`); plus the edge-jump / mask / intensity reductions over the energy axis, and an export writer to `reconstructions/{energy}` for the Python-registration handoff. |
| 13 | P2/P3 (XANES) | **Energy-to-energy 3-D registration** | Translation-only phase-correlation first (reuse `txm_pal_core::phase_cross_correlation`); rigid Mattes-Mutual-Information (SimpleITK parity) is the genuine gap with no Rust equivalent — reimplement or ITK FFI. Until then registration stays an external Python step. |
| 14 | P3 (XANES) | **Zone-plate magnification correction** — *done*. | Energy-dependent affine scale (zone-plate focal length ∝ energy). Implemented in `tomoxide::xanes::magnification` (`magnification_corr_factors` + `apply_magnification`, ported from `magnification_correction.py`) and wired as an in-GUI toggle in the XANES screen (applied per energy before fitting). |

---

## 7. Milestones

- **M1 — MVP (offline preview)**: crate scaffold with the
  excluded-workspace/toolchain setup; additions #1–#3; Data screen (open,
  projection StackView, theta plot); Tune screen with analytic algorithms,
  single-Z preview (ImageView value readout + interactive colorbar),
  ROI/profile, A/B CompareImages;
  auto-center buttons; recipe save/load (CLI-compatible TOML).
- **M2 — Full offline**: `--progress-json` (#4); Run screen with subprocess
  fan-out, GPU picker, pre-flight estimates, latest-slice live view, cancel;
  center sweep montage (`write_center` + StackView + metric Plot1D);
  `RowBandReader` (#5) unlocking phase/iterative previews; Vo stripe
  variants (#6); Output screen (browsing, ScalarFieldView 3D, rescale
  export); batch queue.
- **M3 — Live streaming** *(first cut done)*: rsdm PVA source + ring buffer
  + Z-slice live loop with live center tweak and per-loop parameter re-read
  are implemented (`crates/tomoxide-gui/src/live/{ring,source,recon_loop}.rs`
  + `views/live.rs`), verified against an in-process `epics_pva_rs::PvaServer`.
  Because rsdm delivers a flat frame, the width comes from a companion width
  PV or a manual value and the height is a PV, a manual value, or derived from
  the frame length (`len / width` — only the width is required); the angle
  comes from a theta PV or a manual constant step per frame. Each frame pairs
  with the latest scalar angle (no NTNDArray `uniqueId` exposure). Remaining:
  the ortho X/Y kernels (#7) for the full
  multi-pane click-to-set experience; incremental recon (#8) and dezinger (#9)
  as performance/parity follow-ups.
- **M4 — XANES spectroscopic mapping**: multi-energy stack reader (#12) +
  ported `txm_pal_core` fitting module (#11); XANES screen with the fit
  controls, 2-D chemical maps + histogram + click-to-spectrum, and a
  results-H5 save interoperable with the existing Python viewer. The
  per-energy stack is produced by looping the **existing CUDA streaming recon**
  over energies with the current DXchange preprocessing unchanged — no new
  recon/prep work. The 3-D chemical map now renders through rsplot's
  `VolumeRaycaster` (a GPU direct-volume ray-cast), and **zone-plate
  magnification correction (#14) is an in-GUI toggle** (`xanes::magnification`).
  Registration stays upstream (Python) in v1; in-GUI registration
  (translation → rigid, #13) is the remaining follow-up, along with an
  edge-jump / thickness alpha transfer function for the raycaster.
