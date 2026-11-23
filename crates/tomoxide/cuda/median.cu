// 3D median over a cube, optionally among a mask's voxels alone.

#include <cuda_runtime.h>
#include <math.h>

namespace {

const int MEDIAN_SIDE = 9;  // the widest cube: a thread sorts side^3 values of its own

// scipy's `reflect`, numpy's `symmetric`: d c b a | a b c d | d c b a.
__device__ inline int median_fold(int i, int n) {
  while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1;
  return i;
}

__global__ void median3d_ker(const float* v, const unsigned char* mask, float* out, int nz, int ny,
                             int nx, int h) {
  long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long) nz * ny * nx) return;
  if (mask && !mask[idx]) {
    out[idx] = nanf("");
    return;
  }
  int x = (int) (idx % nx);
  long long row = idx / nx;
  int y = (int) (row % ny);
  int z = (int) (row / ny);
  float near[MEDIAN_SIDE * MEDIAN_SIDE * MEDIAN_SIDE];
  int n = 0;
  for (int dz = -h; dz <= h; dz++) {
    long long pz = median_fold(z + dz, nz);
    for (int dy = -h; dy <= h; dy++) {
      long long pzy = (pz * ny + median_fold(y + dy, ny)) * nx;
      for (int dx = -h; dx <= h; dx++) {
        long long j = pzy + median_fold(x + dx, nx);
        if (mask && !mask[j]) continue;
        float a = v[j];
        int k = n++;
        for (; k > 0 && near[k - 1] > a; k--) near[k] = near[k - 1];
        near[k] = a;
      }
    }
  }
  // The voxel itself is counted, so n >= 1; an odd n adds its middle to itself.
  out[idx] = (near[(n - 1) / 2] + near[n / 2]) * 0.5f;
}

}  // namespace

extern "C" {

// out[z, y, x] = the median of v over the (2h+1)^3 cube about each voxel, the
// volume folded at its ends; an even count gives the mean of the two middles.
// With `mask` (u8, same shape; null for none) only its non-zero voxels are
// counted, and a voxel off it gets NaN. h <= 4. Returns 0 once queued, else
// the cudaError_t.
int tomoxide_median3d(const void* v, const void* mask, void* out, int nz, int ny, int nx, int h,
                      void* stream) {
  if (h < 0 || 2 * h + 1 > MEDIAN_SIDE) return (int) cudaErrorInvalidValue;
  const int block = 256;
  long long total = (long long) nz * ny * nx;
  median3d_ker<<<(unsigned int) ((total + block - 1) / block), block, 0, (cudaStream_t) stream>>>(
      (const float*) v, (const unsigned char*) mask, (float*) out, nz, ny, nx, h);
  return (int) cudaGetLastError();
}

}  // extern "C"
