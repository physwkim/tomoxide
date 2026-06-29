# tomoxide

> A Rust tomographic reconstruction toolkit — the algorithmic breadth of
> [tomopy](https://github.com/tomopy/tomopy) fused with the GPU-accelerated
> streaming reconstruction of [tomocupy](https://github.com/tomography/tomocupy),
> behind a single **tri-backend** abstraction: **CPU · CUDA · wgpu (Metal)**.

> ⚠️ **Status: scaffold (v0.0.0).** This is a broad module skeleton plus a
> design/porting plan. Most numerical kernels are stubs that return
> `Error::NotImplemented` and carry a doc comment pointing at the exact
> upstream C / CUDA source to port from. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Why

| | tomopy | tomocupy | **tomoxide** |
|---|---|---|---|
| Language | Python + C (`libtomo`) | Python + CUDA (CuPy) | **Rust** |
| Algorithm breadth | ✅ gridrec, FBP, ART, SIRT, MLEM, OSEM, OSPML, TV, TIKH, … | ⚠️ fourierrec, lprec, linerec | ✅ union of both |
| GPU acceleration | ⚠️ partial | ✅ CUDA streaming | ✅ CUDA + portable wgpu |
| Streaming / on-the-fly | ❌ | ✅ chunked, double-buffered | ✅ (port of `rec_steps`) |
| Memory safety | C | CUDA/C++ | ✅ Rust |
| Runs without an NVIDIA GPU | ✅ (CPU) | ❌ | ✅ (CPU or Metal via wgpu) |

## Workspace layout

```
crates/
  tomoxide-core    data model (Projections/Sinogram/Volume), geometry,
                   the Backend trait + capability traits, params/enums, errors
  tomoxide-cpu     CPU backend (ndarray + rayon)         — ports libtomo
  tomoxide-cuda    CUDA backend (FFI to vendored .cu)    — ports tomocupy kernels
  tomoxide-wgpu    portable GPU backend (WGSL/Metal)     — optional `gpu-wgpu`
  tomoxide-recon   reconstruction algorithms + center finding
  tomoxide-prep    normalize, stripe removal, phase retrieval, ring removal, …
  tomoxide-io      DXchange/HDF5 + TIFF/zarr readers & writers
  tomoxide-sim     phantoms + forward projection
  tomoxide         umbrella crate: re-exports + high-level pipelines
  tomoxide-cli     `tomoxide` command-line front-end
```

## Build

```sh
# Default: CPU backend only — builds & tests on any machine (incl. Apple Silicon).
cargo build --workspace
cargo nextest run --workspace      # or: cargo test --workspace

# Portable GPU backend (Metal on macOS, Vulkan/DX12 elsewhere):
cargo build -p tomoxide-wgpu --features gpu-wgpu

# CUDA backend — requires an NVIDIA toolkit (nvcc) and is compiled only when
# the `cuda` feature is enabled. Point it at the kernel sources first:
export TOMOXIDE_CUDA_KERNELS=/path/to/tomocupy/src/cuda
cargo build -p tomoxide-cuda --features cuda
```

The `cuda` feature never compiles on a machine without `nvcc`; the default
build selects the CPU backend so the whole workspace builds on this Mac.

## Choosing a backend (performance)

Not every algorithm is faster on the GPU — there are two reconstruction paths
and they scale very differently:

- **Fused / device-resident path — `Fbp`, `Linerec`, `Fourierrec`, `lprec`.**
  Filtering, back-projection and — for `lprec` — the log-polar spline prefilter,
  gather/FFT/scatter all stay resident on the device (one upload, one download),
  so the GPU wins decisively. On CUDA the `recon`/`recon_steps` path streams
  these per chunk with the cuFFT plans and log-polar grids reused across chunks;
  `Fbp`/`Linerec` additionally scale across multiple GPUs.
- **Composed FFT path — `gridrec`.** Only the per-slice FFT is offloaded
  (cuFFT); the Fourier-grid build, the gather/scatter and the Cartesian
  resampling all run on the host. This is **host-gather bound**: on a strong
  multi-core CPU the GPU's FFT offload does not pay for the host gather plus the
  upload/download round-trip.

### Measured scaling by image size

Reconstruction time per backend on this machine (96-core CPU, 4× RTX 5000 Ada),
sweeping the in-plane image size at fixed depth `nz=128`. GPU columns are the
median of 5 runs (3 at 2048²) after a warmup, with the GPU clocks left dynamic
(unlocked boost — idle 210 MHz, boosting toward 3105 MHz under load), so the GPU
times carry run-to-run variance; CPU columns are clock-independent and carried
over. Times in seconds; **bold** is the fastest backend for that row.

`Fbp` (fused) — GPU wins, and the gap widens with size:

| image | CPU | 1-GPU | 4-GPU |
|---|---|---|---|
| 128²  | 0.078  | **0.008** | 0.012 |
| 256²  | 0.484  | 0.054     | **0.048** |
| 512²  | 2.819  | 0.211     | **0.201** |
| 1024² | 18.84  | 1.156     | **0.601** |
| 2048² | 165.8  | 6.039     | **2.342** |

`Fourierrec` (fused, single-device) — GPU wins ~4–8× at every size:

| image | CPU | 1-GPU | 4-GPU |
|---|---|---|---|
| 128²  | 0.118 | **0.014** | 0.014 |
| 256²  | 0.290 | 0.067     | **0.066** |
| 512²  | 1.087 | **0.250** | 0.250 |
| 1024² | 3.537 | **0.721** | 0.731 |
| 2048² | 16.20 | 4.044     | **3.961** |

`Gridrec` (composed, host-gather bound) — 4-GPU wins at every size; 1-GPU
trails the CPU only at 2048²:

| image | CPU | 1-GPU | 4-GPU |
|---|---|---|---|
| 128²  | 0.191 | 0.110 | **0.097** |
| 256²  | 0.414 | 0.403 | **0.373** |
| 512²  | 2.357 | 1.375 | **1.300** |
| 1024² | 5.642 | 4.778 | **4.391** |
| 2048² | 22.71 | 28.25 | **16.03** |

`lprec` and `Paganin` (per-projection FFT) — CPU wins at small sizes; multi-GPU
overtakes at large sizes. (The `lprec` GPU columns below are recon-only
whole-volume `bench_parallel` numbers that predate the device-resident log-polar
rewrite; the end-to-end CUDA `recon` path is now much faster — see the
head-to-head tables. A recon-only refresh of these cells is pending.):

| image | lprec CPU | lprec 1-GPU | lprec 4-GPU | Paganin CPU | Paganin 1-GPU | Paganin 4-GPU |
|---|---|---|---|---|---|---|
| 128²  | **0.085** | 0.293 | 0.123 | **0.019** | 0.123 | 0.044 |
| 256²  | **0.323** | 0.699 | 0.353 | **0.083** | 0.253 | 0.096 |
| 512²  | 1.074 | 1.528 | **1.033** | **0.278** | 0.674 | 0.306 |
| 1024² | 3.503 | 3.654 | **2.917** | 1.099 | 1.413 | **0.766** |
| 2048² | 18.61 | 17.18 | **15.46** | 5.487 | 5.173 | **3.494** |

What this means in practice:

- **`Fbp`** — use the GPU; the CPU/GPU gap grows from ~6× at 128² to ~70× at
  2048² (CPU dense back-projection scales as `nd²`). 4-GPU is fastest from 256²
  up. Essential for large images.
- **`Fourierrec`** — GPU, ~4–8× over the CPU at all sizes. Single-device by
  design, so a second GPU stays within noise; a single GPU now completes 2048²
  (4.0 s) on the oversampled Fourier grid.
- **`Gridrec`** — host-gather bound, but the GPU still wins: 4-GPU is fastest at
  every size and 1-GPU beats the CPU except at 2048², where one device's
  host-core pool saturates on the gather and the CPU edges ahead.
- **`lprec`, `Paganin`** — CPU at small sizes, where the light per-slice /
  per-projection FFT cannot amortize the GPU's per-call malloc + H2D/D2H + sync.
  The multi-GPU z-split overtakes the CPU once the work is large enough — from
  512² for `lprec`, 1024² for `Paganin`.
- **Multi-GPU** pays off when the work is large: at 2048² for every algorithm,
  and from 512²–1024² on the composed/phase paths. At small sizes the per-device
  fixed cost (and splitting host cores across pools) makes 4-GPU tie or lose to
  1-GPU. Multi-GPU also scales with depth `nz` (deeper stacks = more slices to
  spread). Select GPUs with `TOMOXIDE_CUDA_DEVICES` (comma-separated indices;
  unset = all visible, the default).

The exact numbers are hardware-specific — a weaker CPU or a single faster GPU
shifts each crossover — but the structural reasons (fused vs host-gather bound;
per-call overhead) hold regardless. GPU columns were measured with dynamic
(unlocked boost) clocks, so they reflect real-world boost performance but carry
run-to-run variance; small times and the run-to-run-nondeterministic `Fourierrec`
carry the most ± noise. Locking the clock (`nvidia-smi -lgc <MHz>`) trades some
speed for reproducibility. Reproduce with the `bench_parallel` example (CPU /
single-GPU / all-GPU):

```sh
cargo run --release --features cuda --example bench_parallel -- cpu 1024 1024 128 1
TOMOXIDE_CUDA_DEVICES=0 cargo run --release --features cuda --example bench_parallel -- cuda 1024 1024 128 1
cargo run --release --features cuda --example bench_parallel -- cuda 1024 1024 128 1
```

### Head-to-head vs tomocupy (end-to-end wall)

Full-pipeline wall time — HDF5 read → normalize → reconstruct → TIFF write — for
the three algorithms both tools implement (`linerec`, `fourierrec`, `lprec`), on
the same synthetic DXchange file in `/dev/shm` (RAM I/O, so the comparison is
compute- not disk-bound), `nz=128`, dynamic (unlocked boost) GPU clocks. Both
tools ran under the same clock regime, so the relative comparison is fair;
absolute times carry boost-clock variance. Times in seconds; **bold** is the
faster tool. The `fourierrec` and `lprec` tables were re-measured after the
device-resident streaming rewrite (1-GPU, best of 3 runs, sizes through 4096²);
the `linerec` table below still carries the earlier 2-run, 4-column (1-/4-GPU)
data and predates the Phase 1 auto-pipeline — it is pending a refresh.
tomoxide multi-GPU is one process splitting `z` across 4 devices; tomocupy
1.0.4 is single-GPU per process, so its "4-GPU" is 4 concurrent processes each
pinned to one device over a quarter of the `z` rows (`--start-row/--end-row`,
wall = slowest shard).

`linerec` (fp32):

| image | tomoxide 1-GPU | tomocupy 1-GPU | tomoxide 4-GPU | tomocupy 4-GPU |
|---|---|---|---|---|
| 128²  | **0.52** | 1.14 | **0.96** | 1.83 |
| 256²  | **0.69** | 1.69 | **1.16** | 2.47 |
| 512²  | **1.18** | 2.10 | **1.66** | 2.24 |
| 1024² | **2.24** | 2.28 | **2.73** | 3.12 |
| 2048² | 11.52 | **6.27** | 7.66 | **5.32** |

`fourierrec` (fp32, 1-GPU; refreshed after the device-resident streaming rewrite,
best of 3 runs; 4-GPU cells pending re-measurement):

| image | tomoxide 1-GPU | tomocupy 1-GPU |
|---|---|---|
| 128²  | **0.31** | 1.63 |
| 256²  | **0.36** | 1.63 |
| 512²  | **0.43** | 1.54 |
| 1024² | **1.30** | 3.27 |
| 2048² | **3.14** | 4.61 |
| 4096² | **11.30** | 12.53 |

`lprec` (fp32, 1-GPU; refreshed after the device-resident log-polar rewrite, best
of 3 runs; 4-GPU cells pending re-measurement):

| image | tomoxide 1-GPU | tomocupy 1-GPU |
|---|---|---|
| 128²  | **0.34** | 1.75 |
| 256²  | **0.36** | 1.91 |
| 512²  | **0.54** | 1.84 |
| 1024² | **1.37** | 2.83 |
| 2048² | **4.55** | 5.27 |
| 4096² | 15.67 | **11.32** |

fp16 (`linerec`, `fourierrec`; tomoxide fp16 covers only the analytic paths):

| image | linerec tox/tc 1-GPU | linerec tox/tc 4-GPU | fourierrec tox/tc 1-GPU | fourierrec tox/tc 4-GPU |
|---|---|---|---|---|
| 128²  | **0.44**/1.57 | **0.60**/1.92 | **0.54**/1.27 | **0.60**/2.34 |
| 256²  | **0.66**/1.66 | **0.74**/2.28 | **0.67**/1.66 | **0.73**/2.34 |
| 512²  | **1.14**/2.08 | **1.20**/2.79 | **1.25**/2.12 | **1.24**/2.68 |
| 1024² | **2.96**/3.16 | 2.94/**2.94** | **2.96**/3.17 | **3.02**/3.22 |
| 2048² | 11.25/**5.95** | 11.25/**4.27** | 7.44/**5.39** | 9.38/**4.48** |

What this means:

- **tomoxide wins end-to-end across the sweep for `fourierrec` and `lprec`.**
  After the device-resident streaming rewrite, `recon` on CUDA streams these
  per chunk (GPU normalize/transpose, cuFFT plans + log-polar grids reused), so
  tomoxide leads `fourierrec` at every size up to 4096² and `lprec` up to 2048².
  Two compounding wins: the compiled binary starts in ~0.3 s vs tomocupy's
  ~1.3–1.7 s Python + CuPy/context init, and the per-chunk GPU path keeps the
  device busy without a full-volume host transpose.
- **tomocupy retakes `lprec` only at 4096²** (~1.4×: 11.3 s vs 15.7 s). The
  crossover used to sit at ~512²; the device-resident log-polar path pushed it
  out past 2048². `fourierrec` shows no crossover through 4096². (The `linerec`
  table above predates the Phase 1 auto-pipeline and is pending a refresh — do
  not read its 2048² row as current.)
- **`lprec` is no longer host-gather bound on CUDA.** Its log-polar spline
  prefilter + gather/FFT/scatter now run device-resident (ported from the
  parity-verified Rust math), so the old "tomocupy's strongest case / tomoxide's
  slowest GPU path" no longer holds — at 2048² tomoxide is ~1.16× faster.
- **Multi-GPU.** tomoxide's z-split helps only at large sizes (e.g. linerec
  2048² 11.5→7.7 s). tomocupy's 4-process shard rarely beats its own single GPU
  below 2048² — four cold Python/CuPy inits plus each shard re-reading its slab
  outweigh the split. The `fourierrec`/`lprec` 4-GPU cells are pending
  re-measurement after the streaming rewrite.

Caveat: this is **wall-to-wall** time. tomocupy's own internal "Reconstruction
time" (compute only, excluding Python import) is far smaller (e.g. fourierrec
256² ≈ 0.39 s), so the small/medium wins are tomoxide avoiding interpreter
startup, not a faster core loop; for very large or batched streaming jobs
tomocupy's pipeline dominates. Reproduce by generating a DXchange file with the
`make_synthetic_dxchange` example and timing `tomoxide recon` against
`tomocupy recon` on it.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — data model, backend abstraction, streaming pipeline.
- [docs/PORTING.md](docs/PORTING.md) — upstream tomopy/tomocupy → tomoxide module map with provenance.
- [docs/ROADMAP.md](docs/ROADMAP.md) — milestones from scaffold to feature parity.

## License

BSD-3-Clause. Derived in part from tomopy and tomocupy (both BSD-3-Clause,
UChicago Argonne LLC). See [LICENSE](LICENSE) and [docs/PORTING.md](docs/PORTING.md).
