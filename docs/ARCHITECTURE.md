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
| `RayProject`        | `ray_rows(geom, n)` (sparse single-ray rows of `R`)  | tomopy `art.c`, `bart.c` (row-action Kaczmarz) |
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
| `Cgls`           | `num_iter`                               | standard CGLS (Björck); parity-checked vs ASTRA `CglsAlgorithm` |
| `Vector{,2,3}`   | `num_iter, axis…`                        | tomopy `vector.c` |

Most iterative solvers reduce to `ForwardProject` + `FilteredBackproject`
(unfiltered backproject) capability calls, so the same solver loop runs on any
backend that provides those two primitives. The row-action methods (`Art`,
`Bart`) are the exception: they update the reconstruction one ray at a time, so
they use the `RayProject` (single-ray) capability instead.

For each method's strengths, use cases, and a selection guide (plus the
device-resident GPU benchmark), see [`ALGORITHMS.md`](ALGORITHMS.md).

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

### 4.1 CUDA analytic output convention (vs CPU/wgpu)

The vendored tomocupy back-projection / Fourier kernels originally carried
tomocupy's own output convention (y-flipped, `4/nproj` back-projection, a `½`
filter gain), which differed from the CPU/wgpu (tomopy) path. A **cross-backend
convention unification** brought the CUDA analytic output onto the CPU/tomopy
convention, so for the same sinogram, geometry, and centre a CUDA analytic
reconstruction now matches the CPU/wgpu output in **both orientation and scale**:

| algorithm | image orientation | amplitude scale (cuda / cpu) |
|-----------|-------------------|------------------------------|
| `fbp`, `linerec`      | same as CPU | `≈ 1` (back-projection unified `4/nproj → π/nproj`, filter `½` removed) |
| `fourierrec`          | same as CPU | `≈ 1` (filter `½` removed, unnormalized cuFFT inverse normalized by `(2n)²`) |
| `lprec`               | same as CPU | `≈ 1` (filter `½` removed; log-polar back-projection already matched the CPU port) |
| `gridrec`             | same as CPU | `1` (backend-agnostic `recon::gridrec` over the `Fft` capability; never used `build_filter_w`) |

What the unification changed (the CPU/wgpu backends port **tomopy**, the CUDA
backend ports **tomocupy**, and the unification target is tomopy):

1. **Orientation** — the tomocupy y-flip was removed from `cfunc_linerec`
   (storage-index un-flip) and `cfunc_fourierrec` (a clean output-row flip in
   `divphi`), so CUDA emits the CPU/tomopy handedness. Both back- and
   forward-projectors were flipped together, so they remain a discrete transpose.
2. **Filter amplitude** — the CUDA-only `½` in `build_filter_w` (tomocupy's net
   FBP gain) was dropped; the gain is now `1/pad`, matching tomopy.
3. **Back-projection scale** — `cfunc_linerec`'s baked-in `c = 4/nproj`
   (tomocupy) became a caller-supplied gain: the analytic FBP call sites pass the
   `π/nproj` angular-quadrature dθ weight (tomopy scale), while the iterative
   solvers pass `1.0` — the back-projector itself is the *pure* adjoint `Wᵀ` of
   the unweighted line-integral forward projector `W` on every backend, so a
   converged iterative solve yields the physical μ (pinned absolutely by
   `tests/iterative_amplitude.rs`).
4. **fourierrec normalization** — cuFFT does not normalize its inverse transform
   and `plan2d` is `(2n)²`, so the CUDA fourierrec ran `(2n)²`× hot; `divphi` now
   divides by `(2n)²` to match the CPU's normalized inverse FFT.

**Ramp shape (the remaining residual).** The base ramp still differs per backend:
CPU/wgpu build tomopy's plain linear ramp (`2·k/pad`), CUDA builds tomocupy's
degree-12 `_wint` quadrature ramp, selected via `backend::RampShape` (`Linear`
for CPU/wgpu, `Wint` for CUDA) in the single shared `make_fbp_filter`. This leaves
a deterministic **~1.6%** amplitude residual on the unified paths (the measured
`cuda/cpu` scale is ≈1.016, not exactly 1); Pearson stays ≈1.0 because the shape
difference is small and smooth. This is a shape convention, not a numerical error.

**Laminography is excluded.** The CUDA lamino path (`cfunc_linerec` tilted
back-projector) and the CPU lamino path (`recon::lamino`, a USFFT algorithm) are
*different reconstruction algorithms* with different filter frameworks (CUDA:
`make_fbp_filter(Wint)`; CPU: a bare `|f|/ne` ramp), so they are not
scale-comparable and are **not** unified. The measured `cuda/cpu` lamino ratio is
`≈ −0.89` (a sign flip plus a filter-framework gain difference); both remain
y-flipped, consistently. Each lamino path is validated against its own reference
(CUDA vs tomocupy, CPU vs wgpu) rather than against each other. Consumers must not
assume CUDA and CPU laminography agree in absolute amplitude or sign.

The cross-backend regression test `tests/cuda_cpu_convention_parity.rs` pins both
the orientation and the `cuda/cpu ≈ 1` scale (|Δ| < 5%) for fbp/linerec/fourierrec/
lprec, so any re-appearance of a per-algorithm convention scale — or any other CUDA
divergence — fails there rather than hiding behind a scale-invariant metric.

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
