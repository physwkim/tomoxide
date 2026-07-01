# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.3.0]: https://github.com/physwkim/tomoxide/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/physwkim/tomoxide/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/physwkim/tomoxide/releases/tag/v0.1.0
