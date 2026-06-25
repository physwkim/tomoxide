// Batched C2C FFT via cuFFT, exposing the tomoxide `Fft` capability on the GPU.
//
// Implementing this one capability lets every Fft-composing method
// (`gridrec`, `fourierrec`, `lprec`, Paganin/GPaganin/Farago phase, the
// Fourier-wavelet stripe filter) run on CUDA through the existing backend-
// agnostic code — the same way they compose onto wgpu. cuFFT leaves the inverse
// unnormalized, so we divide by the transform size to match tomoxide's
// convention (`ifft(fft(x)) == x`), as the CPU/wgpu backends do.
//
// Data crosses as device `cufftComplex*` (== interleaved float2, layout-
// compatible with tomoxide's `Complex32`). Compiled by build.rs (nvcc) and
// linked against cuFFT.

#include <cufft.h>
#include <cuda_runtime.h>

__global__ void cscale_ker(float* d, long long n, float f) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  d[i] *= f;
}

static int run_scale(void* data, long long nfloat, float f) {
  int block = 256;
  int grid = (int) ((nfloat + block - 1) / block);
  cscale_ker<<<grid, block>>>((float*) data, nfloat, f);
  return (int) cudaGetLastError();
}

extern "C" {

// In-place batched 1-D C2C FFT of `batch` transforms of length `n`.
// Returns 0 on success.
int tomoxide_fft_1d(void* data, size_t n, size_t batch, int inverse) {
  cufftHandle plan;
  int dims[1] = {(int) n};
  if (cufftPlanMany(&plan, 1, dims, nullptr, 1, (int) n, nullptr, 1, (int) n, CUFFT_C2C,
                    (int) batch) != CUFFT_SUCCESS)
    return -1;
  cufftResult r = cufftExecC2C(plan, (cufftComplex*) data, (cufftComplex*) data,
                               inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
  if (r != CUFFT_SUCCESS) {
    cufftDestroy(plan);
    return -2;
  }
  if (inverse) {
    long long cnt = 2ll * (long long) n * (long long) batch;
    run_scale(data, cnt, 1.0f / (float) n);
  }
  cufftDestroy(plan);
  return (int) cudaDeviceSynchronize();
}

// In-place batched 2-D C2C FFT of `batch` images of size `rows × cols`.
int tomoxide_fft_2d(void* data, size_t rows, size_t cols, size_t batch, int inverse) {
  cufftHandle plan;
  int dims[2] = {(int) rows, (int) cols};
  int stride = (int) (rows * cols);
  if (cufftPlanMany(&plan, 2, dims, nullptr, 1, stride, nullptr, 1, stride, CUFFT_C2C,
                    (int) batch) != CUFFT_SUCCESS)
    return -1;
  cufftResult r = cufftExecC2C(plan, (cufftComplex*) data, (cufftComplex*) data,
                               inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
  if (r != CUFFT_SUCCESS) {
    cufftDestroy(plan);
    return -2;
  }
  if (inverse) {
    long long cnt = 2ll * (long long) rows * (long long) cols * (long long) batch;
    run_scale(data, cnt, 1.0f / (float) (rows * cols));
  }
  cufftDestroy(plan);
  return (int) cudaDeviceSynchronize();
}

}  // extern "C"
