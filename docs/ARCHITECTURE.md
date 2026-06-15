# tomoxide — Architecture

This document defines the data model, the tri-backend abstraction, the
reconstruction taxonomy, and the streaming pipeline. It is the contract that
every module stub in the workspace is written against.

The two upstream projects map onto tomoxide as two complementary halves:

- **tomopy** contributes *algorithmic breadth* on the **CPU** (`libtomo`):
  gridrec, FBP, and the iterative family (ART, BART, SIRT, MLEM, OSEM,
  OSPML/PML, TV, TIKH, GRAD, vector), plus a rich preprocessing/misc library.
- **tomocupy** contributes *GPU streaming throughput* (**CUDA**): Fourier-based
  reconstruction (`fourierrec`), log-polar (`lprec`), direct line integration
  (`linerec`), laminography (USFFT), and a chunked, double-buffered,
  multi-stream pipeline for out-of-core datasets.

tomoxide unifies them: one data model, one set of capability traits, three
interchangeable backends.

---

## 1. Data model (`tomoxide-core::data`)

All bulk arrays are `f32` by default (`f16` is a first-class option on the GPU
backends, mirroring tomocupy's `--dtype float16`). Arrays are 3-D and carry an
explicit **axis layout** so we never silently transpose:

| Type           | Axes (slowest→fastest)        | tomopy/tomocupy name      |
|----------------|-------------------------------|---------------------------|
| `Projections`  | `[angle, row, col]` = (dt,dy,dx) | "projection order" (default) |
| `Sinogram`     | `[row, angle, col]` = (dy,dt,dx) | "sinogram order" (`sinogram_order=True`) |
| `Volume`       | `[z(row), y, x]`              | reconstructed object      |
| `Slice2D`      | `[y, x]`                      | one reconstructed slice   |

`Layout::{Projection, Sinogram}` is tracked at the type level; conversion is an
explicit `swap_axes(0,1)` (the same operation tomopy performs internally). The
column axis `x`/`col` is the detector width; `row`/`y` indexes the slice (the
reconstruction is independent and parallel per row for parallel-beam geometry).

Auxiliary fields (the DXchange triple):

- `data`  — projections, the array above.
- `flat` / `white` — flat-field (open-beam) frames `[nflat, row, col]`.
- `dark` — dark-field frames `[ndark, row, col]`.
- `theta` — projection angles, `[angle]`, radians.

### Geometry (`tomoxide-core::geometry`)

```text
Geometry {
    angles: Angles,          // radians, length = n_proj
    center: Center,          // rotation axis: scalar or per-row
    beam:   Beam,            // Parallel | Cone { source_dist } | Laminography { tilt }
    detector: Detector { width, height, pixel_size }
}
```

`Center` is `Scalar(f32)` or `PerRow(Vec<f32>)` because tomopy accepts a center
array (one per slice) and tomocupy searches it per chunk. Laminography carries
the pitch angle `phi` that tomocupy's `linerec`/USFFT kernels take.

---

## 2. Backend abstraction (`tomoxide-core::backend`)

The central design decision. Algorithms are written **once**, against
capability traits, and dispatched to whichever backend the caller selected.

### 2.1 Device & buffers

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn device(&self) -> DeviceKind;          // Cpu | Cuda | Wgpu
    fn supports(&self, dt: Dtype) -> bool;   // e.g. CPU has no f16 FFT
}

pub trait DeviceBuffer<T: Element> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn copy_from_host(&mut self, src: &[T]) -> Result<()>;
    fn copy_to_host(&self, dst: &mut [T]) -> Result<()>;
}
```

A `Buffer<T>` lives on the device. On the CPU backend it wraps an
`ndarray`/`Vec`; on CUDA it wraps a device pointer + stream; on wgpu it wraps a
`wgpu::Buffer`. Host↔device copies are explicit, matching tomocupy's
pinned-memory staging.

### 2.2 Capability traits

Each numerical primitive the pipeline needs is one trait. A backend implements
the subset it supports; missing ones return `Error::NotImplemented`.

| Trait               | Method(s)                                            | Upstream origin |
|---------------------|------------------------------------------------------|-----------------|
| `Fft`               | `fft_1d`, `ifft_1d`, `fft_2d`, `ifft_2d` (batched, R2C/C2R) | cuFFT (tomocupy `cfunc_fft2d`) / rustfft (CPU) |
| `FbpFilter`         | `make_filter(name, n)`, `apply(sino, filter, center)` | tomocupy `cfunc_filter`; tomopy `fbp.c` |
| `FilteredBackproject` | `backproject(sino, geom, out)`                     | `fourierrec` / `lprec` / `linerec` / `gridrec.c` |
| `ForwardProject`    | `project(volume, geom, out)`                         | tomopy `project.c` |
| `IterativeSolver`   | `step(state, sino, geom)` (one ART/SIRT/MLEM iter)   | tomopy `art.c`, `sirt.c`, … |
| `Elementwise`       | `darkflat`, `minus_log`, `clip`, `axpy`              | tomocupy `proc_functions`; tomopy `normalize` |
| `RankFilter`        | `median3d`, `remove_outlier`                         | tomopy `median_filt3d.c` |

Algorithms in `tomoxide-recon`/`tomoxide-prep` take `&dyn Backend` (plus the
capability trait objects the backend exposes) and never name a concrete device.
This is what lets the same `fbp()` run on CPU, CUDA, or Metal.

### 2.3 The three backends

```
                 ┌──────────────────────── tomoxide-core ───────────────────────┐
                 │   data · geometry · Dtype · Error ·  capability traits        │
                 └───────────────┬───────────────┬───────────────┬──────────────┘
                                 │               │               │
                 ┌───────────────▼──┐  ┌─────────▼─────────┐  ┌──▼───────────────┐
                 │  tomoxide-cpu    │  │  tomoxide-cuda    │  │  tomoxide-wgpu   │
                 │  ndarray+rayon   │  │  FFI → *.cu/nvcc  │  │  WGSL compute    │
                 │  rustfft         │  │  cuFFT, cuda      │  │  Metal/Vulkan    │
                 │  always builds   │  │  feature = "cuda" │  │  feat=`gpu-wgpu` │
                 └──────────────────┘  └───────────────────┘  └──────────────────┘
```

- **CPU** (`tomoxide-cpu`): the parity target for `libtomo`. Pure Rust
  (`ndarray`, `rayon`, `rustfft`). Always compiles; the reference for tests.
- **CUDA** (`tomoxide-cuda`): re-uses tomocupy's *battle-tested* `.cu` kernels.
  A thin `extern "C"` shim wraps the SWIG-exposed `cfunc_*` C++ classes; a
  `build.rs` invokes `nvcc` **only when the `cuda` feature is set**, so a
  machine without a CUDA toolkit still builds the rest of the workspace. The
  kernel directory is given by `TOMOXIDE_CUDA_KERNELS` (defaults to a vendored
  `cuda/` copy). See §4.
- **wgpu** (`tomoxide-wgpu`): a portable compute backend (Metal on macOS,
  Vulkan/DX12/GL elsewhere) so the GPU path runs on hardware tomocupy can't
  target. Kernels are WGSL ports of the CUDA kernels. Optional (`gpu-wgpu`)
  because `wgpu` is a heavy dependency; the default workspace build skips it.

### 2.4 Backend selection

`tomoxide::Engine::auto()` probes at runtime in order: CUDA (if compiled and a
device is present) → wgpu (if compiled and an adapter is present) → CPU. The
CLI exposes `--backend {auto,cpu,cuda,wgpu}`.

---

## 3. Reconstruction taxonomy (`tomoxide-recon`)

Two families, dispatched by a single `recon(sino, geom, Algorithm, backend)`.

### 3.1 Analytic / direct (one pass: filter → backproject)

| Algorithm    | Backend(s)         | Upstream |
|--------------|--------------------|----------|
| `Fbp`        | CPU, CUDA, wgpu    | tomopy `fbp.c`; filter is shared with `gridrec` |
| `Gridrec`    | CPU                | tomopy `gridrec.c` (Fourier grid + convolution) |
| `Fourierrec` | CUDA, wgpu         | tomocupy `cfunc_fourierrec` (USFFT, exponential interpolation) |
| `Lprec`      | CUDA, wgpu         | tomocupy `cfunc_lprec` (log-polar) |
| `Linerec`    | CUDA, wgpu         | tomocupy `cfunc_linerec` (direct line integral; laminography) |

The FBP filter kernel (`ramp`, `shepp`, `cosine`, `cosine2`, `hamming`, `hann`,
`parzen`, `none`) is shared across CPU and GPU — both tomopy and tomocupy expose
the same set.

### 3.2 Iterative (project ↔ backproject loop)

| Algorithm        | params                                   | Upstream |
|------------------|------------------------------------------|----------|
| `Art`            | `num_iter`                               | tomopy `art.c` |
| `Bart`           | `num_iter, num_block, ind_block`         | tomopy `bart.c` |
| `Sirt`           | `num_iter`                               | tomopy `sirt.c` |
| `Mlem`           | `num_iter`                               | tomopy `mlem.c` |
| `Osem`           | `num_iter, num_block, ind_block`         | tomopy `osem.c` |
| `OspmlHybrid`    | `num_iter, reg_par, num_block, ind_block`| tomopy `ospml_hybrid.c` |
| `OspmlQuad`      | `num_iter, reg_par, num_block, ind_block`| tomopy `ospml_quad.c` |
| `PmlHybrid`      | `num_iter, reg_par`                      | tomopy `pml_hybrid.c` |
| `PmlQuad`        | `num_iter, reg_par`                      | tomopy `pml_quad.c` |
| `Tv`             | `num_iter, reg_par`                      | tomopy `tv.c` |
| `Grad`           | `num_iter, reg_par`                      | tomopy `grad.c` |
| `Tikh`           | `num_iter, reg_data, reg_par`            | tomopy `tikh.c` |
| `Vector{,2,3}`   | `num_iter, axis…`                        | tomopy `vector.c` |

All iterative solvers reduce to `ForwardProject` + `FilteredBackproject`
(unfiltered backproject) capability calls, so the same solver loop runs on any
backend that provides those two primitives.

### 3.3 Center finding (`tomoxide-recon::center`)

`find_center` (entropy), `find_center_vo` (Vo coarse+fine, the workhorse),
`find_center_pc` (phase correlation), plus tomocupy's SIFT/AI variants
(`find_center_sift`, `find_center_ai`) behind feature flags.

---

## 4. CUDA FFI boundary (`tomoxide-cuda`)

tomocupy exposes C++ classes through SWIG; we cannot link SWIG's Python module,
so tomoxide adds a **thin C-ABI shim** (`cuda/shim.cpp`) that wraps each
`cfunc_*` class in `extern "C"` create/destroy/call functions:

```c
// generated shim, one set per kernel class
void*  tomoxide_fourierrec_new(size_t nproj, size_t nz, size_t n, const float* theta);
void   tomoxide_fourierrec_backproject(void* h, void* f, const void* g, void* stream);
void   tomoxide_fourierrec_free(void* h);
```

Rust side (`ffi.rs`) declares these as `extern "C"` and `CudaBackend` owns the
opaque handles. `build.rs`:

1. runs only if `cfg!(feature = "cuda")` (guarded by `CARGO_FEATURE_CUDA`);
2. resolves the kernel dir from `$TOMOXIDE_CUDA_KERNELS` (default `./cuda`);
3. shells out to `nvcc` to compile `cfunc_*.cu` + `shim.cpp` into a static lib
   for the gencode arches in `$TOMOXIDE_CUDA_ARCH` (default `75;80;86;89;90`);
4. emits `cargo:rustc-link-lib=cudart`, `cufft`, and the shim archive.

Without the feature, `build.rs` is a no-op and `CudaBackend::new()` returns
`Error::BackendUnavailable`.

Buffer pointers and CUDA stream handles cross the boundary as opaque `void*`
(tomocupy passes them as `size_t`); Rust treats them as `*mut c_void`. The
`f16` variants correspond to tomocupy's `*fp16` classes, selected by `Dtype`.

---

## 5. Streaming pipeline (`tomoxide::pipeline`)

Direct port of tomocupy's two execution modes, layered over the backend traits.

### 5.1 Full, in-memory (`rec.py::GPURec`)

For datasets that fit in device memory. Reads everything, preprocesses,
reconstructs, writes.

### 5.2 Chunked / streaming (`rec_steps.py::GPURecSteps`) — the throughput path

Out-of-core. Two chunking axes and a 3-stage software pipeline that overlaps
disk I/O, host↔device transfer, and compute:

```
 read thread        stream 1 (H2D)      stream 2 (compute)     stream 3 (D2H)     write threads
 ───────────        ──────────────      ──────────────────     ──────────────     ─────────────
 HDF chunk ──▶ pinned[k%2] ──▶ gpu[k%2] ─▶ proc_sino           ─▶ rec_gpu ─▶ pinned ─▶ TIFF/HDF/zarr
 (max_read_threads)            (set)        proc_proj             (.get)
                                            fbp_filter_center
                                            backprojection
```

- **Sinogram chunking** (`proc_sino_parallel`): `ncz` rows per chunk over `z`;
  dark/flat correction, outlier removal (dezinger), stripe removal, minus-log,
  beam hardening.
- **Projection chunking** (`proc_proj_parallel`): `ncproj` angles per chunk;
  phase retrieval, beam hardening, projection rotation, double-FOV handling.
- **Double buffering**: index `k%2` ping-pongs pinned/device buffers so stage
  *k+1* uploads while stage *k* computes and stage *k-1* downloads.

In tomoxide this is expressed with backend-agnostic buffers and Rust threads
(`std::thread` + channels) for the read/write pools; the per-backend "stream"
is an associated `Stream` type (a real `cudaStream_t`, a `wgpu::Queue`
submission, or — on CPU — a `rayon` scope). The orchestration code is written
once.

The `try` / `try_lamino` modes (single-slice reconstruction across a sweep of
candidate centers / laminography angles) reuse the same chunk machinery with a
shift array, feeding `find_center`.

---

## 6. Crate dependency graph

```
tomoxide-core  ◀── everything (types, traits, errors)
      ▲
      ├── tomoxide-cpu      (impl traits, ndarray/rayon/rustfft)
      ├── tomoxide-cuda     (impl traits, FFI; feature "cuda")
      ├── tomoxide-wgpu     (impl traits, WGSL; feature "gpu-wgpu")
      ├── tomoxide-recon ──▶ depends on core only (backend-agnostic algorithms)
      ├── tomoxide-prep  ──▶ depends on core only
      ├── tomoxide-io
      └── tomoxide-sim
                 ▲
   tomoxide (umbrella) ── re-exports all of the above, owns Engine + pipeline
                 ▲
        tomoxide-cli ── clap front-end (init/recon/recon_steps/status)
```

`recon`/`prep` deliberately depend on **core only**, not on any backend crate —
they receive a backend through trait objects. Backend crates are wired together
in the `tomoxide` umbrella, which is the only place that knows all three exist.
