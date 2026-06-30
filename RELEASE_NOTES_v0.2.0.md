# tomoxide v0.2.0

tomoxide is a Rust tomographic reconstruction toolkit combining the algorithmic
breadth of tomopy (CPU `libtomo`) with the GPU-accelerated streaming
reconstruction of tomocupy (CUDA), behind a single tri-backend abstraction:
CPU / CUDA / wgpu.

## Highlights

- **Full CUDA streaming engine.** Device-resident streaming reconstructors for
  the analytic methods (FBP, linerec, fourierrec) and lprec, with an async
  H2D∥compute∥D2H pipeline and memory-aware chunking that lifts the
  large-volume GPU ceiling. tomoxide now beats tomocupy end-to-end on every
  analytic method on an RTX 5000 Ada.
- **Multi-GPU.** Per-slice and fused analytic reconstruction across all GPUs,
  plus a multi-GPU z-shard fan-out for the streaming `recon` CLI. 4-GPU 4096²
  head-to-head now favors tomoxide.
- **Half precision (fp16).** End-to-end fp16 analytic path including
  out-of-core fp16 Fbp/Linerec and device-resident fp16 fourierrec streaming;
  fourierrec 2048² single-GPU went 9.83 → 2.70 s, ahead of tomocupy. Enable with
  `--dtype float16`.
- **On-device stripe removal.** Titarenko, Fourier-Wavelet, and Vo all-stripe
  removal run inside the streaming raw path on the GPU.
- **Portable wgpu backend now usable at scale.** Cleared the
  65535-workgroup dispatch cap and the WebGL downlevel buffer limits; wgpu
  reconstructs 512²/1024²/2048² volumes and matches the CPU reference at
  Pearson 1.0 (a correct, portable fallback — not a perf path).
- **New algorithms / preprocessing.** Vector tomography (tomopy `vector.c`,
  bit-exact), beam-hardening correction (xraylib, `beam-hardening` feature), and
  SIFT-based center finding (OpenCV, `sift-center` feature).
- **Single-crate workspace.** The nine library crates were consolidated into one
  `tomoxide` crate.

## Cross-backend convention note

The CUDA analytic streaming kernels emit each slice with tomocupy's handedness:
a vertical (y-axis) flip plus a per-algorithm scale (4/π for fbp/linerec, ≈4·n²
for fourierrec; lprec and gridrec keep the CPU/tomopy orientation and scale 1).
The reconstruction is numerically identical to the CPU/wgpu path once the
convention is undone (Pearson 1.0). This is documented in
`docs/ARCHITECTURE.md` §4.1 and pinned by
`tests/cuda_cpu_convention_parity.rs`.

## Install

```toml
[dependencies]
tomoxide = "0.2.0"
```

CLI:

```sh
cargo install tomoxide-cli
```

GPU features are opt-in (`cuda`, `gpu-wgpu`); see the README for the feature
matrix and benchmark tables.

See [CHANGELOG.md](CHANGELOG.md) for the complete list of changes.
