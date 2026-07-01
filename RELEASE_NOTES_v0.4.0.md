# tomoxide v0.4.0

tomoxide is a Rust tomographic reconstruction toolkit combining the algorithmic
breadth of tomopy (CPU `libtomo`) with the GPU-accelerated streaming
reconstruction of tomocupy (CUDA), behind a single tri-backend abstraction:
CPU / CUDA / wgpu.

This release has two headline themes: the **full iterative reconstruction suite
now runs on the GPU** (device-resident), and a **cross-backend convention
unification** that makes CUDA analytic output match the CPU/wgpu (tomopy)
convention. The CLI also gains the full preprocessing / iterative / filter
composition surface and a live TOML config.

## Behaviour changes — read before upgrading

- **CUDA analytic output changed orientation and amplitude.** CUDA analytic
  reconstruction (`fbp` / `linerec` / `fourierrec` / `lprec`) now matches the
  CPU/wgpu (tomopy) convention: the tomocupy vertical (y-axis) flip is removed and
  the amplitude is tomopy scale (`cuda/cpu ≈ 1`), not tomocupy's. In 0.3.0 CUDA
  deliberately matched tomocupy; if you depended on that, this is a breaking
  change. Backends now agree up to a deterministic ~1.6% `_wint`-vs-linear ramp
  *shape* residual (Pearson ≈ 1.0).
- **`sim::project` output changed by `π/nproj`.** The CPU forward projector is now
  a true adjoint of the back-projector at a single scale, matching the CUDA pair,
  so the iterative solvers stay well-posed across backends. The fixed-step
  `grad`/`tv` solvers gain-normalize the residual, so their behaviour is unchanged.
- **Laminography is excluded from the unification.** The CUDA (tilted linerec) and
  CPU (USFFT) lamino paths are different algorithms and are not scale-comparable;
  each is validated against its own reference and both stay y-flipped. Do not
  warm-start one lamino backend from the other.

## Highlights

- **The iterative suite runs on the GPU, device-resident.** A CUDA forward
  projector (exact adjoint of the linerec back-projector) unlocks tomopy's
  iterative family on the device: `sirt`, `mlem`, `osem`, `ospml`, `pml`, `grad`,
  `tikh`, `tv` keep the volume and sinogram on the GPU across all iterations
  (one upload, one download) — 1.3–11.4× over a per-iteration CUDA loop and
  51–95× over CPU at 512². `art`/`bart` also run on CUDA (bit-identical to CPU).
- **Warm-start / algorithm chaining.** Seed a solver from a prior volume via
  `ReconParams.init`, or from the CLI with `--algorithm fbp,sirt` — the analytic
  result warm-starts the iterative refinement so it converges in fewer iterations.
- **A composable CLI.** `recon`/`recon_steps` now expose stripe removal, phase
  retrieval (with physics), filter choice, iterations, regularization, and every
  per-method parameter, and `--config` (a `tomoxide init` TOML) drives a run with
  `flag > config > default` precedence.
- **Cross-backend convention unification.** CUDA analytic reconstruction matches
  CPU/wgpu in orientation and amplitude; pinned by
  `tests/cuda_cpu_convention_parity.rs` and documented in `docs/ARCHITECTURE.md`.

## Install

```toml
[dependencies]
tomoxide = "0.4.0"
```

CLI:

```sh
cargo install tomoxide-cli
```

GPU features are opt-in (`cuda`, `gpu-wgpu`); see the README for the feature
matrix, the command-line reference, and the benchmark tables.

See [CHANGELOG.md](CHANGELOG.md) for the complete list of changes.
