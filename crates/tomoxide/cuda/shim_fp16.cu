// C-ABI shim for the FP16 (half-precision) kernel variants.
//
// tomocupy compiles each `cfunc_*` class twice — once with `real = float` and
// once with `-DHALF` (`real = half`) — and selects the variant by `--dtype`.
// The default f32 build (shim.cpp + cfunc_*.cu compiled standalone) gives the
// global-scope `cfunc_filter` / `cfunc_linerec` / `cfunc_fourierrec`. This TU
// compiles the SAME vendored sources a second time with `HALF` defined, inside
// a private `fp16` namespace, so their symbols (the classes plus the kernels
// `mulw`, `backprojection_ker`, `gather`, …) do not collide with the f32 ones
// in the single static library `build.rs` produces. Only `Dataset`-shaped half
// buffers cross the FFI as opaque `void*`; the half-ness is purely in byte size.
//
// FP16 cuFFT (`cufftXtMakePlanMany` with `CUDA_R_16F`/`CUDA_C_16F`) requires
// power-of-two transform sizes, so the padded width `ne` must be a power of two
// (the Rust side enforces this). theta/shift stay f32.

// System headers OUTSIDE the namespace, so cuFFT/CUDA types stay at global
// scope; the vendored `.cuh` re-include them but their include guards make the
// in-namespace includes no-ops.
#include <cstddef>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <stdio.h>

#define HALF
namespace fp16 {
#include "cfunc_filter.cu"
#include "cfunc_linerec.cu"
#include "cfunc_fourierrec.cu"
}  // namespace fp16
#undef HALF

// ---- half-precision pad/crop/pack/unpack (mirror elementwise.cu, real=half) ----
// These are pure data moves (copy/replicate) — no arithmetic — so they carry
// `__half` payloads through the same index math as the f32 kernels.
__global__ void pad_ker_h(const __half* src, __half* dst, int nz, int nproj, int ncols, int ne,
                          int pad_side) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nproj * ne;
  if (i >= total) return;
  int x = (int) (i % ne);
  long long srow = (i / ne) * ncols;
  int sx = x - pad_side;
  if (sx < 0) sx = 0;
  else if (sx >= ncols) sx = ncols - 1;
  dst[i] = src[srow + sx];
}

__global__ void crop_ker_h(const __half* src, __half* dst, int nz, int nproj, int ncols, int ne,
                           int pad_side) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nproj * ncols;
  if (i >= total) return;
  int x = (int) (i % ncols);
  long long row = i / ncols;
  dst[i] = src[row * ne + pad_side + x];
}

__global__ void pack_ker_h(const __half* src, __half* dst, int nz, int nproj, int ncols) {
  int half = nz / 2;
  long long n = (long long) half * nproj * ncols;
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long per = (long long) nproj * ncols;
  long long s = i / per;
  long long pr = i % per;
  dst[2 * i] = src[s * per + pr];
  dst[2 * i + 1] = src[(s + half) * per + pr];
}

__global__ void unpack_ker_h(const __half* src, __half* dst, int nz, int n) {
  int half = nz / 2;
  long long per = (long long) n * n;
  long long total = (long long) half * per;
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  long long s = i / per;
  long long pr = i % per;
  dst[s * per + pr] = src[2 * i];
  dst[(s + half) * per + pr] = src[2 * i + 1];
}

namespace {
inline size_t as_size(void* p) { return reinterpret_cast<size_t>(p); }
inline size_t as_size(const void* p) { return reinterpret_cast<size_t>(p); }
}  // namespace

extern "C" {

// ---- FBP filter (fp16) ----
void* tomoxide_filter_fp16_new(size_t nproj, size_t nz, size_t n) {
  return new fp16::cfunc_filter(nproj, nz, n);
}
void tomoxide_filter_fp16_apply(void* h, void* g, const void* w, void* stream) {
  static_cast<fp16::cfunc_filter*>(h)->filter(as_size(g), as_size(w), as_size(stream));
}
void tomoxide_filter_fp16_free(void* h) { delete static_cast<fp16::cfunc_filter*>(h); }

// ---- linerec (fp16) ----
void* tomoxide_linerec_fp16_new(size_t nproj, size_t nz, size_t n, size_t ncproj, size_t ncz) {
  return new fp16::cfunc_linerec(nproj, nz, n, ncproj, ncz);
}
void tomoxide_linerec_fp16_backproject(void* h, void* f, const void* g, const float* theta,
                                       float phi, int sz, void* stream) {
  static_cast<fp16::cfunc_linerec*>(h)->backprojection(as_size(f), as_size(g), as_size(theta), phi,
                                                       sz, as_size(stream));
}
void tomoxide_linerec_fp16_free(void* h) { delete static_cast<fp16::cfunc_linerec*>(h); }

// ---- fourierrec (fp16) ----
void* tomoxide_fourierrec_fp16_new(size_t nproj, size_t nz, size_t n, const float* theta) {
  return new fp16::cfunc_fourierrec(nproj, nz, n, as_size(theta));
}
void tomoxide_fourierrec_fp16_backproject(void* h, void* f, const void* g, void* stream) {
  static_cast<fp16::cfunc_fourierrec*>(h)->backprojection(as_size(f), as_size(g), as_size(stream));
}
void tomoxide_fourierrec_fp16_free(void* h) { delete static_cast<fp16::cfunc_fourierrec*>(h); }

// ---- elementwise (fp16) ----
int tomoxide_pad_fp16(const void* src, void* dst, size_t nz, size_t nproj, size_t ncols, size_t ne,
                      size_t pad_side, void* stream) {
  long long total = (long long) nz * nproj * ne;
  int block = 256, grid = (int) ((total + block - 1) / block);
  pad_ker_h<<<grid, block, 0, (cudaStream_t) stream>>>((const __half*) src, (__half*) dst, (int) nz,
                                                       (int) nproj, (int) ncols, (int) ne,
                                                       (int) pad_side);
  return (int) cudaGetLastError();
}
int tomoxide_crop_fp16(const void* src, void* dst, size_t nz, size_t nproj, size_t ncols, size_t ne,
                       size_t pad_side, void* stream) {
  long long total = (long long) nz * nproj * ncols;
  int block = 256, grid = (int) ((total + block - 1) / block);
  crop_ker_h<<<grid, block, 0, (cudaStream_t) stream>>>((const __half*) src, (__half*) dst,
                                                        (int) nz, (int) nproj, (int) ncols,
                                                        (int) ne, (int) pad_side);
  return (int) cudaGetLastError();
}
int tomoxide_pack_pairs_fp16(const void* src, void* dst, size_t nz, size_t nproj, size_t ncols,
                             void* stream) {
  long long n = (long long) (nz / 2) * nproj * ncols;
  int block = 256, grid = (int) ((n + block - 1) / block);
  pack_ker_h<<<grid, block, 0, (cudaStream_t) stream>>>((const __half*) src, (__half*) dst,
                                                        (int) nz, (int) nproj, (int) ncols);
  return (int) cudaGetLastError();
}
int tomoxide_unpack_pairs_fp16(const void* src, void* dst, size_t nz, size_t n, void* stream) {
  long long total = (long long) (nz / 2) * n * n;
  int block = 256, grid = (int) ((total + block - 1) / block);
  unpack_ker_h<<<grid, block, 0, (cudaStream_t) stream>>>((const __half*) src, (__half*) dst,
                                                          (int) nz, (int) n);
  return (int) cudaGetLastError();
}

}  // extern "C"
