// C-ABI shim over tomocupy's `cfunc_linerec` CUDA back-projection class, plus
// minimal CUDA runtime helpers (device probe + linear device memory) so Rust
// can move host buffers across the boundary without cupy.
//
// tomocupy ships its kernels as C++ classes wrapped for Python by SWIG; tomoxide
// cannot link the SWIG module, so this re-exports the pieces it needs as plain
// `extern "C"` functions that `src/ffi.rs` binds. Device pointers and CUDA
// streams cross the boundary as `void*`; the underlying class takes them as
// `size_t`, so we cast.
//
// Compiled by `build.rs` (via nvcc) only when the `cuda` feature is enabled,
// together with the vendored `cfunc_linerec.cu`. See README.md.
//
// Scope: this is the M4 FBP back-projection path (`cfunc_linerec`, parallel
// beam). The FBP *filter* runs on the CPU (the shared `tomoxide-core` filter
// definition), so cufft and the other `cfunc_*` classes are not linked here.

#include <cstddef>
#include <cuda_runtime.h>

#include "cfunc_linerec.cuh"
#include "cfunc_fourierrec.cuh"
#include "cfunc_filter.cuh"

namespace {
inline size_t as_size(void* p) { return reinterpret_cast<size_t>(p); }
inline size_t as_size(const void* p) { return reinterpret_cast<size_t>(p); }
}  // namespace

extern "C" {

// ---- device runtime helpers ----
int tomoxide_cuda_device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}
// Bind the calling host thread to a device (for multi-GPU pools). 0 on success.
int tomoxide_cuda_set_device(int dev) { return (int) cudaSetDevice(dev); }
// Returns a device pointer (as void*) or null on failure.
void* tomoxide_cuda_malloc(size_t bytes) {
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
  return p;
}
void tomoxide_cuda_free(void* p) { cudaFree(p); }
// 0 on success, non-zero cudaError_t otherwise.
int tomoxide_cuda_memcpy_h2d(void* dst, const void* src, size_t bytes) {
  return (int) cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}
int tomoxide_cuda_memcpy_d2h(void* dst, const void* src, size_t bytes) {
  return (int) cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}
int tomoxide_cuda_memset(void* p, int value, size_t bytes) {
  return (int) cudaMemset(p, value, bytes);
}
int tomoxide_cuda_sync() { return (int) cudaDeviceSynchronize(); }
// Free / total memory on the current device (bytes). 0 on success.
int tomoxide_cuda_mem_info(size_t* free_bytes, size_t* total_bytes) {
  return (int) cudaMemGetInfo(free_bytes, total_bytes);
}

// ---- linerec (cfunc_linerec) ----
void* tomoxide_linerec_new(size_t nproj, size_t nz, size_t n, size_t ncproj, size_t ncz) {
  return new cfunc_linerec(nproj, nz, n, ncproj, ncz);
}
void tomoxide_linerec_backproject(void* h, void* f, const void* g, const float* theta, float phi,
                                  int sz, void* stream) {
  static_cast<cfunc_linerec*>(h)->backprojection(as_size(f), as_size(g), as_size(theta), phi, sz,
                                                 as_size(stream));
}
void tomoxide_linerec_free(void* h) { delete static_cast<cfunc_linerec*>(h); }

// ---- fourierrec (cfunc_fourierrec) ----
void* tomoxide_fourierrec_new(size_t nproj, size_t nz, size_t n, const float* theta) {
  return new cfunc_fourierrec(nproj, nz, n, as_size(theta));
}
void tomoxide_fourierrec_backproject(void* h, void* f, const void* g, void* stream) {
  static_cast<cfunc_fourierrec*>(h)->backprojection(as_size(f), as_size(g), as_size(stream));
}
void tomoxide_fourierrec_free(void* h) { delete static_cast<cfunc_fourierrec*>(h); }

// ---- FBP filter (cfunc_filter) ----
void* tomoxide_filter_new(size_t nproj, size_t nz, size_t n) {
  return new cfunc_filter(nproj, nz, n);
}
void tomoxide_filter_apply(void* h, void* g, const void* w, void* stream) {
  static_cast<cfunc_filter*>(h)->filter(as_size(g), as_size(w), as_size(stream));
}
void tomoxide_filter_free(void* h) { delete static_cast<cfunc_filter*>(h); }

}  // extern "C"
