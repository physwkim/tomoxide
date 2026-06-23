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

extern "C" {

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
