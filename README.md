# tomoxide

[![crates.io](https://img.shields.io/crates/v/tomoxide.svg)](https://crates.io/crates/tomoxide)
[![docs.rs](https://img.shields.io/docsrs/tomoxide)](https://docs.rs/tomoxide)
[![docs site](https://img.shields.io/badge/docs-mdBook-blue)](https://physwkim.github.io/tomoxide/)

> A Rust tomographic reconstruction toolkit — the algorithmic breadth of
> [tomopy](https://github.com/tomopy/tomopy) fused with the GPU-accelerated
> streaming reconstruction of [tomocupy](https://github.com/tomography/tomocupy),
> behind a single **tri-backend** abstraction: **CPU · CUDA · wgpu (Metal)**.

> **Status: working — v0.6.0.** All three backends reconstruct real datasets.
> The **CPU** backend ports tomopy's analytic (`fbp`, `gridrec`, `fourierrec`,
> `lprec`, `linerec`) and iterative (`sirt`, `mlem`, `osem`, `ospml`, `pml`,
> `tv`, `grad`, `tikh`, `art`, `bart`) families; the **CUDA** backend ports
> tomocupy's device-resident streaming kernels (multi-GPU, fp16, laminography,
> and the full iterative suite on-device); the **wgpu** backend runs a portable
> subset (Metal / Vulkan / DX12) with no NVIDIA toolkit. See the
> [documentation site](https://physwkim.github.io/tomoxide/) and
> [CHANGELOG.md](CHANGELOG.md).

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

Two published crates — the `tomoxide` library (everything — all three backends
live in it as modules behind `cuda` / `gpu-wgpu` features) and the `tomoxide-cli`
binary — plus a desktop GUI (`tomoxide-gui`) that lives in the repo but outside
the Cargo workspace (it targets a newer toolchain, so it is not on crates.io).

```
crates/
  tomoxide       library: data model, geometry, the Backend trait, all three
                 backends, reconstruction, preprocessing, I/O, simulation, and
                 the high-level pipelines
  tomoxide-cli   `tomoxide` command-line front-end (init/status/recon/recon_steps/tune_chunk)
  tomoxide-gui   desktop app (rsplot / egui + wgpu); workspace-excluded, not published
```

Inside the library:

```
crates/tomoxide/src/
  backend.rs   Backend trait + capability traits (Fft, FbpFilter,
               FilteredBackproject, ForwardProject, RankFilter, …)
  cpu/         CPU backend (ndarray + rayon)              — ports libtomo
  cuda/        CUDA backend (FFI to the vendored kernels) — ports tomocupy kernels
  wgpu/        portable GPU backend (WGSL)                — feature `gpu-wgpu`
  recon/       reconstruction algorithms + center finding
  prep/        normalize, stripe removal, phase retrieval, …
  io/          DXchange/HDF5 + TIFF/zarr readers & writers
  sim          phantoms + forward projection
  data.rs geometry.rs params.rs dtype.rs engine.rs pipeline.rs error.rs
crates/tomoxide/cuda/   vendored tomocupy .cu/.cuh kernels + shim.cpp
                        (compiled by build.rs via nvcc when `cuda` is enabled)
```

## Install

Both crates are published on crates.io:

```sh
cargo add tomoxide          # library — add to a Rust project
cargo install tomoxide-cli  # the `tomoxide` command-line tool
```

Opt into a GPU backend with a feature (`--features cuda` needs the CUDA toolkit;
`--features gpu-wgpu` is portable Metal / Vulkan / DX12 with no NVIDIA toolkit).

## Build

Backends are Cargo features on the `tomoxide` / `tomoxide-cli` crates (default =
CPU only). Enable one when building.

```sh
# Default: CPU backend only — builds & tests on any machine, no GPU (incl. Apple Silicon).
cargo build --release
cargo nextest run --workspace      # or: cargo test --workspace

# CUDA backend (NVIDIA). The vendored kernels in crates/tomoxide/cuda/ are compiled
# by build.rs via nvcc when the `cuda` feature is on — no env setup needed
# (TOMOXIDE_CUDA_KERNELS defaults to that dir). Requires an NVIDIA toolkit (nvcc):
cargo build --release -p tomoxide-cli --features cuda

# Portable GPU backend (Metal on macOS, Vulkan/DX12 elsewhere) — no NVIDIA toolkit:
cargo build --release -p tomoxide-cli --features gpu-wgpu
```

The `cuda` feature never compiles on a machine without `nvcc`; the default build
selects the CPU backend so the whole workspace builds anywhere (including
GPU-less CI and Apple Silicon).

## Command-line usage

The `tomoxide` binary (crate `tomoxide-cli`) is the front-end. Build it, then run
either the installed binary or through cargo:

```sh
cargo build --release -p tomoxide-cli            # → target/release/tomoxide
cargo build --release -p tomoxide-cli --features cuda   # CUDA-enabled build

# Run directly…
target/release/tomoxide recon scan.h5 --algorithm fbp
# …or via cargo (note the `--` separating cargo args from tomoxide args):
cargo run --release -p tomoxide-cli -- recon scan.h5 --algorithm fbp
```

The input is a DXchange/HDF5 file (`/exchange/{data,data_white,data_dark,theta}`).
Output is written next to the input as `<name>_rec` — a directory of per-slice
TIFFs by default, or a single `.h5` / `.zarr` with `--save_format`. Don't have a
dataset? Generate a synthetic one:

```sh
cargo run --release --example make_synthetic_dxchange -- <nproj> <nz> <nx> scan.h5
```

### Global options

| Option | Values | Meaning |
|---|---|---|
| `--backend` | `auto` (default) · `cpu` · `cuda` · `wgpu` | Compute backend. `auto` picks the best available; a `--config` `backend` is used only when this is left at `auto`. |
| `-v`, `--verbose` | — | Debug logging (prints the resolved preprocessing / iteration settings, chunk source, multi-GPU sharding, …). |

### Subcommands

| Command | Purpose |
|---|---|
| `init` | Write a default TOML config you can edit (`--config tomoxide.toml`). |
| `status` | Print the resolved backend and, with `--config <f>`, the parsed config. |
| `recon` | Full reconstruction. On CUDA the fused methods auto-stream per chunk and fan across GPUs; everything else runs whole-volume. |
| `recon_steps` | Explicit out-of-core streaming reconstruction (tomocupy `recon_steps`), chunked over detector rows. |
| `tune_chunk` | Measure and cache the fastest `--chunk` for a file/algorithm/GPU (CUDA fused methods only). |

### `recon` / `recon_steps` options

Both share the same reconstruction knobs. Every option is optional: an omitted
flag falls back to the `--config` file (if given), then to a built-in default —
**precedence is `flag > config > default`** (tomocupy-style).

**Core**

| Option | Values / default | Meaning |
|---|---|---|
| `<FILE>` | — | Input DXchange HDF5 (positional, required). |
| `--config <F>` | — | TOML config supplying defaults for the options below. |
| `--algorithm <A>` | default `fbp` | Method, or a **comma-separated chain** (see below). |
| `--center <C>` | auto-find | Rotation-axis column. |
| `--dtype <D>` | `float32` (also `float16`) | Precision; `float16` only affects the CUDA/wgpu analytic paths. |
| `--save_format <F>` | `tiff` · `h5` · `zarr` | Output container (default `tiff`). |
| `--filter <F>` | `parzen` (default) | Apodization: `none` `ramp` `shepp` `cosine` `cosine2` `hamming` `hann` `parzen`. |
| `--num_iter <N>` | `1` | Iterations for iterative methods (analytic methods ignore it). |
| `--reg_par <csv>` | — | Regularization parameters for iterative methods, e.g. `--reg_par 0.5,0.01`. |

Analytic methods: `fbp` `gridrec` `fourierrec` `lprec` `linerec`.
Iterative methods: `art` `bart` `sirt` `mlem` `osem` `ospml_hybrid` `ospml_quad`
`pml_hybrid` `pml_quad` `tv` `grad` `tikh` `cgls`.

**Preprocessing — stripe removal** (`--remove_stripe <M>`, applied before recon).
Each method reads only its own parameters (all overridable; defaults shown):

| `<M>` | Parameters (default) |
|---|---|
| `none` (default) | — |
| `fw` (Fourier-wavelet) | `--fw_sigma 2.0` · `--fw_level 0` (0 = auto) |
| `ti` (Titarenko) | `--ti_nblock 0` · `--ti_beta 1.5` |
| `sf` (smoothing filter) | `--sf_size 5` |
| `vo-all` (Vo combined) | `--vo_snr 3.0` · `--vo_la_size 61` · `--vo_sm_size 21` |

**Preprocessing — phase retrieval** (`--retrieve_phase <M>`). Shared physics flags:
`--pixel_size 1e-4` (cm) · `--propagation_distance 50` (cm) · `--energy 30` (keV) ·
`--alpha 1e-3` · `--db 1000` · `--w 2e-4` (cm).

| `<M>` | Uses |
|---|---|
| `none` (default) | — |
| `paganin` | pixel_size, distance, energy, alpha |
| `Gpaganin` | pixel_size, distance, energy, db, w |
| `farago` | pixel_size, distance, energy, db |

**`recon`-only**

| Option | Meaning |
|---|---|
| `--chunk <N>` | Detector rows per CUDA streaming chunk. Omitted → the `tune_chunk` cache, else a safe default. |
| `--start_row`, `--end_row` | Reconstruct only a contiguous z-shard (also used internally by the multi-GPU orchestrator). |
| `--lamino_angle <deg>` | Laminography tilt (CUDA `fbp`/`linerec`, f32); forces the whole-stack path. |
| `--lamino_rh <N>` | Laminography output height (default `ceil(nz / cos(angle) / 2) * 2`). |

**`recon_steps`-only**

| Option | Meaning |
|---|---|
| `--chunk <N>` | Slices per streaming chunk (tomocupy `--nsino-per-chunk`); omitted → config `nsino_per_chunk`, else default. |

### Config file & precedence

`tomoxide init` writes an editable TOML; `recon`/`recon_steps` load it with
`--config` and use it as the defaults, with any CLI flag overriding. So a config
pins the pipeline once, and a flag tweaks one run:

```sh
tomoxide init --config scan.toml            # then edit scan.toml
tomoxide recon scan.h5 --config scan.toml   # config drives everything
tomoxide recon scan.h5 --config scan.toml --filter ramp --center 512.5
#   ^ ramp + center override the config; all other values still come from it
tomoxide status --config scan.toml          # inspect the parsed config
```

### Algorithm chaining (warm-start)

A comma-separated `--algorithm` runs the stages in order, seeding each from the
previous stage's volume — e.g. give an iterative solver a fast analytic starting
point so it converges in fewer iterations:

```sh
tomoxide recon scan.h5 --algorithm fbp,sirt --num_iter 30
#   fbp reconstructs, then sirt (30 iters) is warm-started from the fbp result
```

Each iterative stage can carry its own iteration budget with a `:iters` suffix;
stages without one fall back to `--num_iter`:

```sh
tomoxide recon scan.h5 --algorithm fbp,sirt:30,tv:10
#   fbp seed → sirt 30 iters → tv 10 iters, each warm-started from the previous
```

Analytic stages reject a `:iters` suffix (they take no iterations). `reg_par` /
`filter` still apply to every stage. Chaining uses the whole-volume path, so it
is supported by `recon` only (not the streaming `recon_steps`).

### Multi-GPU, chunking, laminography

```sh
# Tune & cache the fastest streaming chunk for this file/algorithm/GPU:
tomoxide --backend cuda tune_chunk scan.h5 --algorithm fourierrec --dtype float16
# A later recon then auto-applies the cached chunk (see the -v log line).

# Multi-GPU: recon auto-fans one z-shard subprocess per selected GPU for CUDA
# TIFF jobs at width ≥ 2048. Restrict the device set with TOMOXIDE_CUDA_DEVICES:
TOMOXIDE_CUDA_DEVICES=0,1,2,3 tomoxide --backend cuda recon scan.h5 --algorithm fbp

# Laminography (tilted rotation axis), CUDA fbp/linerec:
tomoxide --backend cuda recon scan.h5 --algorithm linerec --lamino_angle 20 --center 512
```

### Examples

```sh
# GPU FBP, half precision, HDF5 output:
tomoxide --backend cuda recon scan.h5 --algorithm fbp --dtype float16 --save_format h5

# CPU gridrec with Vo stripe removal (tightened) + a fixed center:
tomoxide --backend cpu recon scan.h5 --algorithm gridrec \
    --remove_stripe vo-all --vo_snr 4 --center 640.5

# Paganin phase retrieval, then FBP:
tomoxide recon scan.h5 --retrieve_phase paganin \
    --pixel_size 0.65e-4 --propagation_distance 20 --energy 25 --alpha 2e-3

# Out-of-core streaming SIRT, 16 rows per chunk:
tomoxide recon_steps scan.h5 --algorithm sirt --num_iter 50 --chunk 16
```

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
For nx≥2048 tomoxide multi-GPU fans one z-shard process per device (each pinned
via `CUDA_VISIBLE_DEVICES` over a quarter of the `z` rows, `--start_row/--end_row`,
wall = slowest shard) — the same shape as tomocupy 1.0.4, which is single-GPU per
process so its "4-GPU" is 4 concurrent shard processes too. Below nx=2048 tomoxide
stays single-GPU (the extra per-process CUDA init isn't worth the split), so those
"4-GPU" cells equal its 1-GPU streaming path.

`linerec` (fp32):

| image | tomoxide 1-GPU | tomocupy 1-GPU | tomoxide 4-GPU | tomocupy 4-GPU |
|---|---|---|---|---|
| 128²  | **0.41** | 1.07 | **0.51** | 1.64 |
| 256²  | **0.44** | 1.18 | **0.55** | 1.76 |
| 512²  | **0.63** | 2.20 | **0.66** | 2.71 |
| 1024² | **1.22** | 3.22 | **1.21** | 3.46 |
| 2048² | **4.28** | 6.42 | **2.53** | 4.84 |
| 4096² | **30.79** | 36.86 | **10.15** | 15.92 |

`fourierrec` (fp32):

| image | tomoxide 1-GPU | tomocupy 1-GPU | tomoxide 4-GPU | tomocupy 4-GPU |
|---|---|---|---|---|
| 128²  | **0.49** | 1.10 | **0.52** | 1.90 |
| 256²  | **0.52** | 1.70 | **0.58** | 2.40 |
| 512²  | **0.63** | 2.16 | **0.69** | 2.33 |
| 1024² | **1.05** | 3.21 | **1.14** | 3.59 |
| 2048² | **2.61** | 4.56 | **2.05** | 5.08 |
| 4096² | **9.24** | 12.56 | **4.94** | 10.58 |

`lprec` (fp32):

| image | tomoxide 1-GPU | tomocupy 1-GPU | tomoxide 4-GPU | tomocupy 4-GPU |
|---|---|---|---|---|
| 128²  | **0.52** | 1.30 | **0.53** | 2.10 |
| 256²  | **0.55** | 1.87 | **0.63** | 2.53 |
| 512²  | **0.70** | 2.33 | **0.72** | 2.87 |
| 1024² | **1.34** | 3.48 | **1.30** | 3.76 |
| 2048² | **3.14** | 5.09 | **3.46** | 5.10 |
| 4096² | **11.12** | 11.84 | **9.37** | 10.75 |

fp16 (`linerec`, `fourierrec`; tomoxide fp16 covers only the analytic paths):

| image | linerec tox/tc 1-GPU | linerec tox/tc 4-GPU | fourierrec tox/tc 1-GPU | fourierrec tox/tc 4-GPU |
|---|---|---|---|---|
| 128²  | **0.56**/1.53 | **0.52**/1.88 | **0.59**/1.59 | **0.69**/2.15 |
| 256²  | **0.57**/1.64 | **0.62**/2.39 | **0.90**/1.72 | **0.85**/2.41 |
| 512²  | **0.65**/2.02 | **0.71**/2.60 | **1.29**/2.09 | **1.30**/2.59 |
| 1024² | **1.13**/3.12 | **1.14**/3.22 | **3.20**/3.27 | **3.11**/3.23 |
| 2048² | **3.49**/6.01 | **2.23**/4.75 | **2.70**/5.41 | **1.93**/5.18 |

What this means:

- **tomoxide wins end-to-end across the whole sweep for all three algorithms, on
  both 1- and 4-GPU.** `recon` on CUDA streams per chunk (GPU normalize/transpose,
  cuFFT plans + log-polar grids reused, output volume buffers recycled across
  chunks rather than re-allocated). Two compounding wins: the compiled binary
  starts in ~0.3 s vs tomocupy's ~1.3–1.7 s Python + CuPy/context init, and the
  per-chunk GPU path keeps the device busy without a full-volume host transpose.
- **`lprec` 4096² 1-GPU is now a tomoxide win** (11.12 s vs 11.84 s). Recycling
  the per-chunk output buffer (no fresh 536 MB allocation + page-faults per chunk)
  cut ~3 s off the wall and closed tomocupy's last single-GPU lead; the crossover
  that used to sit at ~512² is gone through 4096².
- **Multi-GPU.** At nx≥2048 `recon` fans one z-shard process per GPU
  (`CUDA_VISIBLE_DEVICES`, one contiguous row range each), so the GPU compute *and*
  the HDF5 read / TIFF write parallelize across processes — the same multi-process
  shard tomocupy uses, but with the leaner per-process startup. This closed the
  4096² 4-GPU gap that the old single-device streaming left: `linerec` 30.8→10.2 s
  (vs tomocupy 15.9 s), `fourierrec` 9.3→4.9 s (vs 10.6 s), `lprec` 11.6→9.4 s
  (vs 10.8 s) — all now tomoxide wins, bit-identical to the single-GPU output
  (`fourierrec` Pearson 1.0, atomicAdd floor). Below nx=2048 the extra CUDA-init
  per shard process outweighs the split, so multi-GPU stays on the single-GPU
  streaming path (still ahead of tomocupy, whose 4-process Python startup is
  heavier).
- **fp16.** tomoxide wins both algorithms at every size on 1- and 4-GPU through
  2048². `fourierrec` fp16 used to be the exception (9.8 s at 2048², slower than
  its own fp32) because its half-precision path fell back to the per-chunk
  reconstructor; it now shares the device-resident streaming handle (pack →
  `cfunc_fourierrec` (f16) → unpack reusing one handle set), dropping the 2048²
  1-GPU wall to 2.70 s and beating tomocupy 5.41 s.

Caveat: this is **wall-to-wall** time. tomocupy's own internal "Reconstruction
time" (compute only, excluding Python import) is far smaller (e.g. fourierrec
256² ≈ 0.39 s), so the small/medium wins are tomoxide avoiding interpreter
startup, not a faster core loop; for very large or batched streaming jobs
tomocupy's pipeline dominates. Reproduce by generating a DXchange file with the
`make_synthetic_dxchange` example and timing `tomoxide recon` against
`tomocupy recon` on it.

The narrative docs are also published as a browsable site:
**<https://physwkim.github.io/tomoxide/>**. The API reference is on
[docs.rs/tomoxide](https://docs.rs/tomoxide).

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — data model, backend abstraction, streaming pipeline, cross-backend conventions.
- [docs/ALGORITHMS.md](docs/ALGORITHMS.md) — the analytic and iterative methods and their parameters.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — measured reconstruction quality, speed, and iteration behaviour of the methods on real data (which method/filter to pick).
- [docs/GUI.md](docs/GUI.md) — the `tomoxide-gui` desktop app design (Data / Tune / Center / Run / Output / Live modes, offline + live streaming).
- [docs/PORTING.md](docs/PORTING.md) — upstream tomopy/tomocupy → tomoxide module map with provenance.
- [CHANGELOG.md](CHANGELOG.md) — release-by-release changes.

## License

BSD-3-Clause. Derived in part from tomopy and tomocupy (both BSD-3-Clause,
UChicago Argonne LLC). See [LICENSE](LICENSE) and [docs/PORTING.md](docs/PORTING.md).
