# tomoxide — Roadmap

Milestones from the current scaffold to feature parity. Each milestone is
shippable on its own and keeps `cargo build --workspace` + `cargo nextest run
--workspace` green on a machine with **no** GPU.

## M0 — Scaffold ✅ (this commit)

- Workspace, 10 crates, tri-backend trait skeleton (CPU/CUDA/wgpu).
- Data model, geometry, params/enums, errors in `tomoxide-core`.
- Every algorithm/preprocessing entry point exists and dispatches; bodies
  return `Error::NotImplemented` with an upstream `file:line` doc pointer.
- A handful of real CPU ops (`angles`, `minus_log`, `darkflat`, `circ_mask`,
  `shepp2d`) so there is an end-to-end smoke test.
- Design docs: ARCHITECTURE, PORTING, ROADMAP.

**Done = workspace builds and tests pass with the CPU backend selected.**

## M1 — CPU analytic core (parity target: tomopy FBP/gridrec)

- `tomoxide-cpu`: real `Fft` (rustfft), `FbpFilter`, `FilteredBackproject`.
- `recon::fbp` and `recon::gridrec` produce correct slices.
- `sim::project` forward projector (port `project.c`) for round-trip tests.
- Verification: phantom → project → reconstruct → SSIM/MSE vs phantom; and
  numeric diff against tomopy `recon(..., algorithm='fbp'/'gridrec')` on a
  fixed phantom.

**Done = `fbp`/`gridrec` within tolerance of tomopy on shepp2d.**

## M2 — CPU iterative family (parity target: tomopy ART/SIRT/MLEM…)

- `IterativeSolver` on CPU; port `art/sirt/mlem/osem/bart` then the
  regularized set (`ospml_*`, `pml_*`, `tv`, `tikh`, `grad`).
- Block/ordered-subset handling (`num_block`, `ind_block`).
- Verification: per-algorithm numeric diff vs tomopy at fixed `num_iter`.

## M3 — Preprocessing & center finding (CPU)

- `prep`: `minus_log`, `normalize*`, the stripe-removal family
  (`fw`, `ti`, `sf`, Vo sorting/filtering/fitting), Paganin `retrieve_phase`,
  `remove_ring`, `median_filter3d`, dezinger.
- `center`: `find_center_vo` (primary), `find_center`, `find_center_pc`.
- `tomoxide-io`: DXchange HDF5 reader/writer + TIFF.

**Done = a full CPU pipeline: HDF in → preprocess → center → FBP → TIFF out.**

## M4 — CUDA backend (parity target: tomocupy)

- `tomoxide-cuda`: C-ABI shim over tomocupy's `cfunc_*` classes; `build.rs`
  nvcc compile gated on the `cuda` feature; FFI bindings.
- Backends for `fourierrec`, `lprec`, `linerec`, `cfunc_filter`.
- GPU `Elementwise`/stripe/phase to match tomocupy's `proc_functions`.
- Verification: on a CUDA host, numeric diff vs tomocupy for each method.

## M5 — Streaming pipeline (parity target: tomocupy `rec_steps`)

- `pipeline::ReconSteps`: sinogram/projection chunking, double buffering,
  3-stage overlap (read → H2D → compute → D2H → write), read/write thread
  pools, `try`/`try_lamino` center & laminography sweeps.
- Out-of-core reconstruction of a dataset larger than device memory.

## M6 — Portable GPU (wgpu / Metal)

- `tomoxide-wgpu`: WGSL ports of the FBP filter, backprojection, elementwise.
- Runs the GPU path on Apple Silicon (Metal) and Vulkan/DX12.
- Verification: wgpu vs CPU numeric diff on the same phantom.

## M7 — Laminography, beam hardening, AI center, polish

- USFFT laminography (`lamfourierrec`, `cfunc_usfft1d/2d`, `cfunc_fft2d`).
- Beam-hardening correction; SIFT/AI center finding.
- `f16` paths on GPU backends; zarr output; benchmarks (`hyperfine`/criterion).

---

### Cross-cutting verification harness

A `tests/` parity harness (added in M1) fixes a small phantom, runs the
reference (tomopy/tomocupy via a checked-in `.npy`/`.h5` golden), and asserts
tomoxide is within tolerance — backend-parametrized so CPU/CUDA/wgpu all run
the same assertions.
