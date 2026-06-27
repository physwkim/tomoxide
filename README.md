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

- **Fused analytic path — `Fbp`, `Linerec`, `Fourierrec`.** Filtering and
  back-projection stay resident on the device (one upload, one download), so
  the GPU wins decisively and scales across multiple GPUs.
- **Composed FFT path — `lprec`, `gridrec`.** Only the per-slice FFT is
  offloaded (cuFFT); the log-polar / Fourier-grid build, the gather/scatter and
  the Cartesian resampling all run on the host. These are **host-gather bound**:
  on a strong multi-core CPU the GPU's FFT offload does not pay for the host
  gather plus the upload/download round-trip.

### Measured scaling by image size

Reconstruction time per backend on this machine (96-core CPU, 4× RTX 5000 Ada),
sweeping the in-plane image size at fixed depth `nz=128`, single-shot (`reps=1`).
Times in seconds; **bold** is the fastest backend for that row.

`Fbp` (fused) — GPU wins, and the gap widens with size:

| image | CPU | 1-GPU | 4-GPU |
|---|---|---|---|
| 128²  | 0.078  | **0.014** | 0.018 |
| 256²  | 0.484  | 0.052     | **0.051** |
| 512²  | 2.819  | 0.200     | **0.194** |
| 1024² | 18.84  | **0.769** | 0.795 |
| 2048² | 165.8  | 5.71 †    | **2.37** |

`Fourierrec` (fused, single-device) — GPU wins ~4–6× at every size:

| image | CPU | 1-GPU | 4-GPU |
|---|---|---|---|
| 128²  | 0.118 | **0.018** | 0.023 |
| 256²  | 0.290 | **0.065** | 0.066 |
| 512²  | 1.087 | 0.245     | **0.229** |
| 1024² | 3.537 | **0.620** | 0.895 |
| 2048² | 16.20 | crash †   | **4.36** |

`Gridrec` (composed, host-gather bound) — CPU ≈ GPU at every size:

| image | CPU | 1-GPU | 4-GPU |
|---|---|---|---|
| 128²  | 0.191 | 0.195 | **0.134** |
| 256²  | **0.414** | 0.607 | 0.433 |
| 512²  | 2.357 | 1.581 | **1.426** |
| 1024² | **5.642** | 5.031 | 5.565 |
| 2048² | 22.71 | 25.54 † | **17.43** |

`lprec` (composed) and `Paganin` (per-projection FFT) — **CPU wins at every size**:

| image | lprec CPU | lprec 1-GPU | lprec 4-GPU | Paganin CPU | Paganin 1-GPU | Paganin 4-GPU |
|---|---|---|---|---|---|---|
| 128²  | **0.085** | 0.523 | 0.189 | **0.019** | 0.223 | 0.067 |
| 256²  | **0.323** | 1.219 | 0.520 | **0.083** | 0.600 | 0.263 |
| 512²  | **1.074** | 3.080 | 1.446 | **0.278** | 1.199 | 0.470 |
| 1024² | **3.503** | 5.248 | 4.974 | **1.099** | 3.213 | 2.275 |
| 2048² | **18.61** | crash † | 22.22 | **5.487** | crash † | 9.101 |

What this means in practice:

- **`Fbp`** — use the GPU; the CPU/GPU gap grows from ~6× at 128² to ~70× at
  2048² (CPU dense back-projection scales as `nd²`). Essential for large images.
- **`Fourierrec`** — GPU, ~4–6× at all sizes. Single-device by design, so a
  second GPU does not help; at 2048² a single GPU runs out of memory on the
  oversampled Fourier grid, and only the multi-GPU z-split completes.
- **`Gridrec`** — host-gather bound: CPU and GPU stay within ~1.7× everywhere,
  so the backend choice barely matters.
- **`lprec`, `Paganin`** — prefer the **CPU** at every size. The light per-slice
  / per-projection FFT does not amortize the GPU's per-call malloc + H2D/D2H +
  sync, so even a single GPU loses to the 96-core host, and more GPUs never close
  the gap.
- **Multi-GPU** pays off only when the work is large *and* GPU-bound — at 2048²
  for `Fbp`/`Gridrec`, and as the only path past single-GPU memory limits.
  Below that, the per-device fixed cost (and splitting host cores across pools)
  makes 4-GPU tie or lose to 1-GPU. Multi-GPU also scales with depth `nz`
  (deeper stacks = more slices to spread). Select GPUs with
  `TOMOXIDE_CUDA_DEVICES` (comma-separated indices; unset = all visible, the
  default).

The exact numbers are hardware-specific — a weaker CPU or a single faster GPU
shifts each crossover — but the structural reasons (fused vs host-gather bound;
per-call overhead) hold regardless. `reps=1` is single-shot, so small times and
the run-to-run-nondeterministic `Fourierrec` carry ± noise. `†` 1-GPU at 2048²
is at the 32 GB memory boundary and crashes on the fused/Fourier path; the
multi-GPU run survives by splitting the z-axis across devices. Reproduce with
the `bench_parallel` example (CPU / single-GPU / all-GPU):

```sh
cargo run --release --features cuda --example bench_parallel -- cpu 1024 1024 128 1
TOMOXIDE_CUDA_DEVICES=0 cargo run --release --features cuda --example bench_parallel -- cuda 1024 1024 128 1
cargo run --release --features cuda --example bench_parallel -- cuda 1024 1024 128 1
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — data model, backend abstraction, streaming pipeline.
- [docs/PORTING.md](docs/PORTING.md) — upstream tomopy/tomocupy → tomoxide module map with provenance.
- [docs/ROADMAP.md](docs/ROADMAP.md) — milestones from scaffold to feature parity.

## License

BSD-3-Clause. Derived in part from tomopy and tomocupy (both BSD-3-Clause,
UChicago Argonne LLC). See [LICENSE](LICENSE) and [docs/PORTING.md](docs/PORTING.md).
