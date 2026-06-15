// C-ABI shim over tomocupy's SWIG-exposed `cfunc_*` C++ classes.
//
// tomocupy ships its CUDA kernels as C++ classes wrapped for Python by SWIG.
// tomoxide cannot link the SWIG Python module, so this shim re-exports each
// class as plain `extern "C"` create/call/free functions that Rust's FFI
// (`src/ffi.rs`) binds to. Device pointers and CUDA streams cross the boundary
// as `void*`; the underlying classes take them as `size_t`, so we cast.
//
// This file is compiled by `build.rs` (via nvcc) ONLY when the `cuda` feature
// is enabled, together with the vendored kernels. The `#include`s below resolve
// against tomocupy's `include/` (passed on the nvcc `-I` path). See README.md.

#include <cstddef>

#include "cfunc_fourierrec.cuh"
#include "cfunc_lprec.cuh"
#include "cfunc_linerec.cuh"
#include "cfunc_filter.cuh"

namespace {
inline size_t as_size(void* p) { return reinterpret_cast<size_t>(p); }
inline size_t as_size(const void* p) { return reinterpret_cast<size_t>(p); }
}  // namespace

extern "C" {

// ---- fourierrec ----
void* tomoxide_fourierrec_new(size_t nproj, size_t nz, size_t n, const float* theta) {
  return new cfunc_fourierrec(nproj, nz, n, as_size(theta));
}
void tomoxide_fourierrec_backproject(void* h, void* f, const void* g, void* stream) {
  static_cast<cfunc_fourierrec*>(h)->backprojection(as_size(f), as_size(g), as_size(stream));
}
void tomoxide_fourierrec_free(void* h) { delete static_cast<cfunc_fourierrec*>(h); }

// ---- lprec ----
void* tomoxide_lprec_new(size_t nproj, size_t nz, size_t n, size_t ntheta, size_t nrho) {
  return new cfunc_lprec(static_cast<int>(nproj), static_cast<int>(nz), static_cast<int>(n),
                         static_cast<int>(ntheta), static_cast<int>(nrho));
}
void tomoxide_lprec_backproject(void* h, void* f, const void* g, void* stream) {
  static_cast<cfunc_lprec*>(h)->backprojection(as_size(f), as_size(g), as_size(stream));
}
void tomoxide_lprec_free(void* h) { delete static_cast<cfunc_lprec*>(h); }

// ---- linerec ----
void* tomoxide_linerec_new(size_t nproj, size_t nz, size_t n, size_t ncproj, size_t ncz) {
  return new cfunc_linerec(nproj, nz, n, ncproj, ncz);
}
void tomoxide_linerec_backproject(void* h, void* f, const void* g, const float* theta, float phi,
                                  int sz, void* stream) {
  static_cast<cfunc_linerec*>(h)->backprojection(as_size(f), as_size(g), as_size(theta), phi, sz,
                                                 as_size(stream));
}
void tomoxide_linerec_free(void* h) { delete static_cast<cfunc_linerec*>(h); }

// ---- FBP filter ----
void* tomoxide_filter_new(size_t nproj, size_t nz, size_t n) {
  return new cfunc_filter(nproj, nz, n);
}
void tomoxide_filter_apply(void* h, void* g, const void* w, void* stream) {
  static_cast<cfunc_filter*>(h)->filter(as_size(g), as_size(w), as_size(stream));
}
void tomoxide_filter_free(void* h) { delete static_cast<cfunc_filter*>(h); }

}  // extern "C"
