# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
