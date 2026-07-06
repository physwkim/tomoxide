# tomoxide

**A Rust tomographic reconstruction toolkit** — the algorithmic breadth of
[tomopy](https://github.com/tomopy/tomopy) fused with the GPU-accelerated
streaming reconstruction of
[tomocupy](https://github.com/tomography/tomocupy), behind a single
**tri-backend** abstraction: **CPU · CUDA · wgpu (Metal / Vulkan / DX12)**.

[![crates.io](https://img.shields.io/crates/v/tomoxide.svg)](https://crates.io/crates/tomoxide)
[![docs.rs](https://img.shields.io/docsrs/tomoxide)](https://docs.rs/tomoxide)
[![GitHub](https://img.shields.io/badge/github-physwkim%2Ftomoxide-blue?logo=github)](https://github.com/physwkim/tomoxide)

All three backends reconstruct real beamline datasets. The **CPU** backend
ports tomopy's analytic (`fbp`, `gridrec`, `fourierrec`, `lprec`, `linerec`)
and iterative (`sirt`, `mlem`, `osem`, `ospml`, `pml`, `tv`, `grad`, `tikh`,
`art`, `bart`) families; the **CUDA** backend ports tomocupy's device-resident
streaming kernels (multi-GPU, fp16, laminography, and the full iterative suite
on-device); the **wgpu** backend runs a portable subset with no NVIDIA toolkit.

## Why

|                             | tomopy                | tomocupy                | **tomoxide**                 |
| --------------------------- | --------------------- | ----------------------- | ---------------------------- |
| Language                    | Python + C (`libtomo`) | Python + CUDA (CuPy)   | **Rust**                     |
| Algorithm breadth           | ✅ broad               | ⚠️ fourierrec/lprec/linerec | ✅ union of both         |
| GPU acceleration            | ⚠️ partial            | ✅ CUDA streaming        | ✅ CUDA + portable wgpu       |
| Streaming / on-the-fly      | ❌                     | ✅ chunked, double-buffered | ✅ (port of `rec_steps`)  |
| Memory safety               | C                     | CUDA/C++                | ✅ Rust                       |
| Runs without an NVIDIA GPU  | ✅ (CPU)               | ❌                       | ✅ (CPU or Metal via wgpu)    |

## Install

The library and CLI are published on crates.io:

```sh
# Library — add to a Rust project
cargo add tomoxide

# Command-line reconstruction tool
cargo install tomoxide-cli
```

The default build is pure-CPU with no system dependencies. Opt into the GPU
backends and extras with cargo features:

```sh
# CUDA backend (needs the CUDA toolkit / nvcc)
cargo add tomoxide --features cuda

# Portable GPU backend (wgpu: Metal / Vulkan / DX12)
cargo add tomoxide --features gpu-wgpu
```

Other features: `config` (TOML recipes, shared by the CLI and GUI),
`beam-hardening` (xraylib), `sift-center` (pure-Rust SIFT center finding).

## Workspace layout

Two published crates plus a desktop GUI that lives in the repo but outside the
Cargo workspace:

```text
crates/
  tomoxide       library: data model, geometry, the Backend trait, all three
                 backends, reconstruction, preprocessing, I/O, simulation,
                 and the high-level streaming pipelines
  tomoxide-cli   command-line front-end (init / status / recon / recon_steps / tune_chunk)
  tomoxide-gui   desktop app (rsplot / egui + wgpu) — not published to crates.io
```

## This book

- **[Architecture](ARCHITECTURE.md)** — the `Backend` trait, the tri-backend
  design, threading, and the streaming pipelines.
- **[Algorithms](ALGORITHMS.md)** — the analytic and iterative reconstruction
  families and their filters, geometry, and conventions.
- **[Benchmarks](BENCHMARKS.md)** — cross-backend timing and accuracy, including
  FBP-vs-iterative comparisons against known-truth phantoms.
- **[Desktop GUI](GUI.md)** — the six-mode `tomoxide-gui` (Data / Tune / Center
  / Run / Output / Live) and its offline + live-streaming design.
- **[Porting & tomocupy parity](PORTING.md)** — how the CUDA/CPU backends were
  ported and validated against the upstream tomopy / tomocupy originals.

The full **API reference** is on [docs.rs/tomoxide](https://docs.rs/tomoxide).

## License

See the [repository](https://github.com/physwkim/tomoxide) for license details.
