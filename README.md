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

Measured on this machine (96-core CPU, 4× RTX 5000 Ada), `lprec`:

| size | CPU | CUDA (4-GPU) | faster |
|---|---|---|---|
| nd=1024, nz=64  | **2.20 s**  | 3.26 s  | CPU 1.48× |
| nd=1024, nz=256 | **7.43 s**  | 8.20 s  | CPU 1.10× |
| nd=2048, nz=64  | **10.82 s** | 12.51 s | CPU 1.16× |

So for `lprec` (and `gridrec`) prefer the **CPU** backend, unless the data is
already on the GPU from an adjacent stage (e.g. Paganin phase retrieval or an
`Fbp` pass) and moving it back to the host would cost more than the gather. The
exact numbers are hardware-specific — a weaker CPU or a single faster GPU shifts
the crossover — but the structural reason (host-gather bound) holds regardless.
Reproduce with the `bench_parallel` example:

```sh
cargo run --release --features cuda --example bench_parallel -- cpu  1024 1024 256 1 lprec
cargo run --release --features cuda --example bench_parallel -- cuda 1024 1024 256 1 lprec
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — data model, backend abstraction, streaming pipeline.
- [docs/PORTING.md](docs/PORTING.md) — upstream tomopy/tomocupy → tomoxide module map with provenance.
- [docs/ROADMAP.md](docs/ROADMAP.md) — milestones from scaffold to feature parity.

## License

BSD-3-Clause. Derived in part from tomopy and tomocupy (both BSD-3-Clause,
UChicago Argonne LLC). See [LICENSE](LICENSE) and [docs/PORTING.md](docs/PORTING.md).
