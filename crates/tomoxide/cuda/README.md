# tomoxide-cuda — vendored CUDA kernels

This directory holds the C-ABI **shim** (`shim.cpp`) that exposes tomocupy's
`cfunc_*` C++ kernel classes to Rust, plus the **vendored** kernel sources
`build.rs` compiles when the `cuda` feature is on.

## M4 FBP path (current)

The wired path is parallel-beam FBP back-projection: `cfunc_linerec.cu` +
`cfunc_linerec.cuh`, `kernels_linerec.cuh`, `defs.cuh` are vendored here, and
the shim adds minimal CUDA-runtime helpers (device probe, `cudaMalloc`/memcpy/
free). The FBP **filter** runs on the CPU (the shared `tomoxide-core` filter
definition), so cufft and the other `cfunc_*` classes are not compiled or
linked. `build.rs` globs the `.cu` files in this directory, so adding more
kernels here (fourierrec/lprec/filter) extends the build with no other change.

The notes below describe how to bring in additional kernels.

## Pointing the build at the kernels

The kernel sources (`cfunc_*.cu`) and headers (`cfunc_*.cuh`, `kernels_*.cuh`,
`defs.cuh`) come from tomocupy:

```
tomocupy/src/cuda/      cfunc_fourierrec.cu, cfunc_lprec.cu, cfunc_linerec.cu,
                        cfunc_filter.cu, cfunc_fft2d.cu, cfunc_usfft1d.cu,
                        cfunc_usfft2d.cu
tomocupy/src/include/   matching *.cuh headers
```

Two ways to supply them:

1. **Env var (recommended for development):**
   ```sh
   export TOMOXIDE_CUDA_KERNELS=/path/to/tomocupy/src/cuda
   cargo build -p tomoxide-cuda --features cuda
   ```
   `build.rs` also adds the sibling `../include` to the nvcc include path.

2. **Vendor:** copy the `.cu`/`.cuh` files into this directory next to
   `shim.cpp` and build with just `--features cuda`.

## Gencode arches

Set `TOMOXIDE_CUDA_ARCH` (default `75;80;86;89;90`) to the compute
capabilities you target, e.g. `export TOMOXIDE_CUDA_ARCH=86`.

## Provenance / licence

These kernels are tomocupy's, BSD-3-Clause (UChicago Argonne LLC). When
vendoring, keep tomocupy's licence header in each file. See
`../../../docs/PORTING.md` §A and §4.
