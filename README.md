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
| 128²  | 0.077  | **0.008** | 0.012 |
| 256²  | 0.485  | 0.050     | **0.040** |
| 512²  | 2.876  | 0.219     | **0.200** |
| 1024² | 18.86  | 1.041     | **0.614** |
| 2048² | 164.2  | 5.681     | **2.245** |

`Fourierrec` (fused, single-device) — GPU wins ~4–8× at every size:

| image | CPU | 1-GPU | 4-GPU |
|---|---|---|---|
| 128²  | 0.103 | **0.012** | 0.012 |
| 256²  | 0.283 | 0.063     | **0.063** |
| 512²  | 0.761 | **0.235** | 0.237 |
| 1024² | 3.427 | **0.666** | 0.675 |
| 2048² | 16.59 | 3.880     | **3.878** |

`Gridrec` (composed, host-gather bound) — 4-GPU wins at every size; 1-GPU
trails the CPU only at 2048²:

| image | CPU | 1-GPU | 4-GPU |
|---|---|---|---|
| 128²  | 0.207 | 0.110 | **0.097** |
| 256²  | 0.656 | 0.414 | **0.374** |
| 512²  | 1.771 | 1.345 | **1.292** |
| 1024² | 6.867 | 4.965 | **4.434** |
| 2048² | 23.00 | 28.18 | **15.88** |

`lprec` and `Paganin` — after the device-resident log-polar rewrite (GPU columns
freshly re-measured), `lprec` now wins on the GPU at **every** size, including the
smallest. `Paganin`'s light per-projection FFT still leaves the CPU ahead at small
sizes, with multi-GPU overtaking from 1024²:

| image | lprec CPU | lprec 1-GPU | lprec 4-GPU | Paganin CPU | Paganin 1-GPU | Paganin 4-GPU |
|---|---|---|---|---|---|---|
| 128²  | 0.091 | **0.051** | 0.052 | **0.023** | 0.114 | 0.038 |
| 256²  | 0.305 | **0.242** | 0.247 | **0.089** | 0.244 | 0.090 |
| 512²  | 1.030 | 0.937 | **0.914** | **0.261** | 0.680 | 0.298 |
| 1024² | 4.516 | **2.362** | 2.382 | 1.145 | 1.511 | **0.981** |
| 2048² | 18.72 | 10.16 | **10.10** | 5.198 | 5.436 | **3.337** |

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
- **`lprec`** — GPU at every size after the device-resident log-polar rewrite;
  the old per-slice-FFT overhead that lost to the CPU below 512² is gone (128²
  0.051 s on one GPU vs 0.091 s CPU). 1-GPU and 4-GPU are within noise — the
  log-polar path is effectively single-device on this whole-volume benchmark.
- **`Paganin`** — CPU at small sizes, where the light per-projection FFT cannot
  amortize the GPU's per-call malloc + H2D/D2H + sync. The multi-GPU z-split
  overtakes the CPU once the work is large enough — from 1024².
- **Multi-GPU** pays off on the z-splittable paths — `Fbp` (2048² 5.7→2.2 s),
  `Gridrec` (28.2→15.9 s), and `Paganin` (5.4→3.3 s) — when the work is large.
  `Fourierrec` and `lprec` are effectively single-device here (4-GPU within noise
  of 1-GPU). At small sizes the per-device fixed cost (and splitting host cores
  across pools) makes 4-GPU tie or lose to 1-GPU. Multi-GPU also scales with depth
  `nz` (deeper stacks = more slices to spread). Select GPUs with
  `TOMOXIDE_CUDA_DEVICES` (comma-separated indices; unset = all visible, the
  default).

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
absolute times carry boost-clock variance. All three fp32 tables and the fp16
table were re-measured together (best of 3 runs, sizes through 4096², 1- and
4-GPU). Times in seconds; **bold** is the faster tool in each 1-GPU / 4-GPU pair.
tomoxide multi-GPU is one process splitting `z` across 4 devices; tomocupy
1.0.4 is single-GPU per process, so its "4-GPU" is 4 concurrent processes each
pinned to one device over a quarter of the `z` rows (`--start-row/--end-row`,
wall = slowest shard).

`linerec` (fp32):

| image | tomoxide 1-GPU | tomocupy 1-GPU | tomoxide 4-GPU | tomocupy 4-GPU |
|---|---|---|---|---|
| 128²  | **0.41** | 1.07 | **0.51** | 1.64 |
| 256²  | **0.44** | 1.18 | **0.55** | 1.76 |
| 512²  | **0.63** | 2.20 | **0.66** | 2.71 |
| 1024² | **1.22** | 3.22 | **1.21** | 3.46 |
| 2048² | **4.28** | 6.42 | **4.23** | 4.84 |
| 4096² | **30.79** | 36.86 | 30.84 | **15.92** |

`fourierrec` (fp32):

| image | tomoxide 1-GPU | tomocupy 1-GPU | tomoxide 4-GPU | tomocupy 4-GPU |
|---|---|---|---|---|
| 128²  | **0.49** | 1.10 | **0.52** | 1.90 |
| 256²  | **0.52** | 1.70 | **0.58** | 2.40 |
| 512²  | **0.63** | 2.16 | **0.69** | 2.33 |
| 1024² | **1.05** | 3.21 | **1.14** | 3.59 |
| 2048² | **2.61** | 4.56 | **2.85** | 5.08 |
| 4096² | **9.24** | 12.56 | **9.29** | 10.58 |

`lprec` (fp32):

| image | tomoxide 1-GPU | tomocupy 1-GPU | tomoxide 4-GPU | tomocupy 4-GPU |
|---|---|---|---|---|
| 128²  | **0.52** | 1.30 | **0.53** | 2.10 |
| 256²  | **0.55** | 1.87 | **0.63** | 2.53 |
| 512²  | **0.70** | 2.33 | **0.72** | 2.87 |
| 1024² | **1.34** | 3.48 | **1.30** | 3.76 |
| 2048² | **3.14** | 5.09 | **3.20** | 5.10 |
| 4096² | **11.12** | 11.84 | 11.59 | **10.75** |

fp16 (`linerec`, `fourierrec`; tomoxide fp16 covers only the analytic paths):

| image | linerec tox/tc 1-GPU | linerec tox/tc 4-GPU | fourierrec tox/tc 1-GPU | fourierrec tox/tc 4-GPU |
|---|---|---|---|---|
| 128²  | **0.56**/1.53 | **0.52**/1.88 | **0.59**/1.59 | **0.69**/2.15 |
| 256²  | **0.57**/1.64 | **0.62**/2.39 | **0.90**/1.72 | **0.85**/2.41 |
| 512²  | **0.65**/2.02 | **0.71**/2.60 | **1.29**/2.09 | **1.30**/2.59 |
| 1024² | **1.13**/3.12 | **1.14**/3.22 | **3.20**/3.27 | **3.11**/3.23 |
| 2048² | **3.49**/6.01 | **3.46**/4.75 | 9.83/**5.41** | 10.16/**5.18** |

What this means:

- **tomoxide wins end-to-end across the sweep for `fourierrec` and `lprec`.**
  `recon` on CUDA streams these per chunk (GPU normalize/transpose, cuFFT plans +
  log-polar grids reused, and output volume buffers recycled across chunks rather
  than re-allocated), so tomoxide leads `fourierrec` at every size through 4096²
  on both 1- and 4-GPU, and leads `lprec` through 4096² on a single GPU. Two
  compounding wins: the compiled binary starts in ~0.3 s vs tomocupy's ~1.3–1.7 s
  Python + CuPy/context init, and the per-chunk GPU path keeps the device busy
  without a full-volume host transpose.
- **`lprec` 4096² 1-GPU is now a tomoxide win** (11.12 s vs 11.84 s). Recycling
  the per-chunk output buffer (no fresh 536 MB allocation + page-faults per chunk)
  cut ~3 s off the wall and closed tomocupy's last single-GPU lead; the crossover
  that used to sit at ~512² is gone through 4096². tomocupy's 4-process shard
  still edges tomoxide's single-process z-split at 4096² (10.75 s vs 11.59 s).
- **`linerec`** — tomoxide wins 1- and 4-GPU through 2048². At 4096² the dense
  back-projection is heavy and the 8.6 GB read+write dominates: tomoxide still
  wins 1-GPU (30.8 s vs 36.9 s) but tomocupy's 4-process shard wins 4-GPU
  (15.9 s vs 30.8 s — tomoxide's single-process z-split barely speeds the wall
  once I/O dominates).
- **Multi-GPU.** tomoxide's single-process z-split pays off on the pure-recon
  scaling benchmark above but little end-to-end at the largest size, where the
  un-split HDF5 read + TIFF write dominate the wall (`fourierrec`/`lprec` are
  effectively single-device here). tomocupy's 4-process shard re-reads its slab
  per process but spreads the I/O across processes, so it pulls ahead at 4096²
  on `linerec`/`lprec`.
- **fp16.** tomoxide wins both algorithms through 1024² on 1- and 4-GPU, and
  `linerec` fp16 stays ahead at 2048². `fourierrec` fp16 is the exception (9.8 s
  vs tomocupy 5.4 s at 2048²): its half-precision path falls back to the
  per-chunk reconstructor instead of the device-resident streaming handle, so it
  runs slower than its own fp32 (2.6 s) — a known open item.

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
