# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/physwkim/tomoxide/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/physwkim/tomoxide/releases/tag/v0.1.0
