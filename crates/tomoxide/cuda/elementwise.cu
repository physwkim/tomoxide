// Elementwise preprocessing kernels (dark/flat correction, minus-log).
//
// tomocupy does these as cupy elementwise expressions rather than `cfunc_*`
// classes, so there is no upstream `.cu` to vendor; these small kernels
// reproduce `proc_functions.darkflat_correction` and `.minus_log` exactly. The
// frame averages (`dark2d`, `flat2d`) and the clamped denominator are computed
// host-side (few frames) and uploaded; the per-projection broadcast and the
// minus-log run here. Compiled by build.rs (nvcc) with the rest of the shim.

#include <cstddef>
#include <cuda_runtime.h>

// data[p, z, x] = (data[p, z, x] - dark2d[z, x]) / denom[z, x]
// data is [nproj, nz, nx] (projection layout); dark2d/denom are [nz, nx].
__global__ void darkflat_ker(float* data, const float* dark2d, const float* denom,
                             int nproj, int nz, int nx) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nproj * nz * nx;
  if (i >= total) return;
  long long frame = i % ((long long) nz * nx);  // (z, x) within a projection
  data[i] = (data[i] - dark2d[frame]) / denom[frame];
}

// data[i] = -ln(max(data[i], 1e-6)); non-finite -> 0  (tomopy minus_log).
__global__ void minuslog_ker(float* data, long long total) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float c = data[i] < 1e-6f ? 1e-6f : data[i];
  float o = -logf(c);
  data[i] = isfinite(o) ? o : 0.0f;
}

// --- device-resident analytic pipeline helpers (no host round-trips) ---

// Edge-replicate pad each [ncols] detector lane into [ne], centred at pad_side
// (tomocupy fbp_filter_center). src [nz,nproj,ncols] -> dst [nz,nproj,ne].
__global__ void pad_ker(const float* src, float* dst, int nz, int nproj, int ncols, int ne,
                        int pad_side) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nproj * ne;
  if (i >= total) return;
  int x = (int) (i % ne);
  long long row = i / ne;            // (z, p) lane index
  long long srow = row * ncols;
  int sx = x - pad_side;
  if (sx < 0) sx = 0;
  else if (sx >= ncols) sx = ncols - 1;
  dst[i] = src[srow + sx];
}

// Crop the centred [pad_side, pad_side+ncols) window: src [nz,nproj,ne] ->
// dst [nz,nproj,ncols].
__global__ void crop_ker(const float* src, float* dst, int nz, int nproj, int ncols, int ne,
                         int pad_side) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nproj * ncols;
  if (i >= total) return;
  int x = (int) (i % ncols);
  long long row = i / ncols;
  dst[i] = src[row * ne + pad_side + x];
}

// Pack slice pairs (s, s+nz/2) into one complex slice: src [nz,nproj,ncols] ->
// dst complex [nz/2,nproj,ncols] (interleaved re/im). re=slice s, im=slice s+nz/2.
__global__ void pack_ker(const float* src, float* dst, int nz, int nproj, int ncols) {
  int half = nz / 2;
  long long n = (long long) half * nproj * ncols;
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // i indexes complex element [s, p, x] with s in [0, half).
  long long per = (long long) nproj * ncols;
  long long s = i / per;
  long long pr = i % per;            // p*ncols + x within a slice
  dst[2 * i] = src[s * per + pr];
  dst[2 * i + 1] = src[(s + half) * per + pr];
}

// De-interleave complex output volume: src complex [nz/2,n,n] -> dst [nz,n,n].
// re -> slice s, im -> slice s+nz/2.
__global__ void unpack_ker(const float* src, float* dst, int nz, int n) {
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

extern "C" {

int tomoxide_pad(const void* src, void* dst, size_t nz, size_t nproj, size_t ncols, size_t ne,
                 size_t pad_side, void* stream) {
  long long total = (long long) nz * nproj * ne;
  int block = 256, grid = (int) ((total + block - 1) / block);
  pad_ker<<<grid, block, 0, (cudaStream_t) stream>>>((const float*) src, (float*) dst, (int) nz,
                                                      (int) nproj, (int) ncols, (int) ne,
                                                      (int) pad_side);
  return (int) cudaGetLastError();
}

int tomoxide_crop(const void* src, void* dst, size_t nz, size_t nproj, size_t ncols, size_t ne,
                  size_t pad_side, void* stream) {
  long long total = (long long) nz * nproj * ncols;
  int block = 256, grid = (int) ((total + block - 1) / block);
  crop_ker<<<grid, block, 0, (cudaStream_t) stream>>>((const float*) src, (float*) dst, (int) nz,
                                                       (int) nproj, (int) ncols, (int) ne,
                                                       (int) pad_side);
  return (int) cudaGetLastError();
}

int tomoxide_pack_pairs(const void* src, void* dst, size_t nz, size_t nproj, size_t ncols,
                        void* stream) {
  long long n = (long long) (nz / 2) * nproj * ncols;
  int block = 256, grid = (int) ((n + block - 1) / block);
  pack_ker<<<grid, block, 0, (cudaStream_t) stream>>>((const float*) src, (float*) dst, (int) nz,
                                                      (int) nproj, (int) ncols);
  return (int) cudaGetLastError();
}

int tomoxide_unpack_pairs(const void* src, void* dst, size_t nz, size_t n, void* stream) {
  long long total = (long long) (nz / 2) * n * n;
  int block = 256, grid = (int) ((total + block - 1) / block);
  unpack_ker<<<grid, block, 0, (cudaStream_t) stream>>>((const float*) src, (float*) dst, (int) nz,
                                                        (int) n);
  return (int) cudaGetLastError();
}

// Returns 0 on success, the cudaError_t otherwise.
int tomoxide_darkflat(void* data, const void* dark2d, const void* denom, size_t nproj,
                      size_t nz, size_t nx, void* stream) {
  long long total = (long long) nproj * nz * nx;
  int block = 256;
  int grid = (int) ((total + block - 1) / block);
  darkflat_ker<<<grid, block, 0, (cudaStream_t) stream>>>(
      (float*) data, (const float*) dark2d, (const float*) denom, (int) nproj, (int) nz,
      (int) nx);
  return (int) cudaGetLastError();
}

int tomoxide_minuslog(void* data, size_t n, void* stream) {
  long long total = (long long) n;
  int block = 256;
  int grid = (int) ((total + block - 1) / block);
  minuslog_ker<<<grid, block, 0, (cudaStream_t) stream>>>((float*) data, total);
  return (int) cudaGetLastError();
}

}  // extern "C"
