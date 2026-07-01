// Device-resident iterative-solver elementwise kernels.
//
// The simultaneous iterative solvers (SIRT first) keep the volume and sinogram
// resident on the GPU across every iteration; these fused per-element ops
// replace the host `ndarray` arithmetic so no data crosses the PCIe bus inside
// the iteration loop. Projection/back-projection reuse the existing
// `tomoxide_forwardproject` / `tomoxide_linerec_backproject` kernels (both
// already operate on device pointers). f32 only; built by build.rs's `.cu` glob.

#include <cuda_runtime.h>
#include <math.h>

// ax[i] = (b[i] - ax[i]) * rw[i]  — SIRT weighted residual, in-place into `ax`
// (which held `A x`), so R ∘ (b − A x) needs no extra buffer.
__global__ void iter_residual_ker(float *ax, const float *b, const float *rw, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    ax[i] = (b[i] - ax[i]) * rw[i];
}

// vol[i] += cw[i] * corr[i]  — SIRT sensitivity-weighted update x += C ∘ Aᵀ(…)
__global__ void iter_update_ker(float *vol, const float *cw, const float *corr, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    vol[i] += cw[i] * corr[i];
}

// out[i] = |x[i]| > thr ? 1/x[i] : 0  — the ray-length R = 1/A(1) and
// sensitivity C = 1/Aᵀ(1) weights (matches the host solver's threshold). In
// place (out == x) is safe: each thread reads then writes its own index.
__global__ void iter_recip_thresh_ker(float *out, const float *x, float thr, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    float v = x[i];
    out[i] = (fabsf(v) > thr) ? (1.0f / v) : 0.0f;
}

// ax[i] = |ax[i]| > 1e-6 ? b[i] / ax[i] : 0  — EM ratio b ⊘ A x, in-place into
// `ax` (which held `A x`). Matches the host MLEM/OSEM zero-guard.
__global__ void iter_em_ratio_ker(float *ax, const float *b, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    float a = ax[i];
    ax[i] = (fabsf(a) > 1e-6f) ? (b[i] / a) : 0.0f;
}

// vol[i] = |sens[i]| > 1e-6 ? vol[i]*corr[i]/sens[i] : vol[i]  — EM multiplicative
// update x ∘ Aᵀ(ratio) ⊘ Aᵀ(1) (pixels with zero sensitivity left untouched).
__global__ void iter_em_update_ker(float *vol, const float *corr, const float *sens,
                                   long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    float s = sens[i];
    if (fabsf(s) > 1e-6f)
        vol[i] = vol[i] * corr[i] / s;
}

extern "C" {

int tomoxide_iter_em_ratio(void *ax, const void *b, size_t n, void *stream) {
    long long total = (long long)n;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    iter_em_ratio_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)ax, (const float *)b, total);
    return (int)cudaGetLastError();
}

int tomoxide_iter_em_update(void *vol, const void *corr, const void *sens, size_t n,
                            void *stream) {
    long long total = (long long)n;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    iter_em_update_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)vol, (const float *)corr, (const float *)sens, total);
    return (int)cudaGetLastError();
}

int tomoxide_iter_residual(void *ax, const void *b, const void *rw, size_t n, void *stream) {
    long long total = (long long)n;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    iter_residual_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)ax, (const float *)b, (const float *)rw, total);
    return (int)cudaGetLastError();
}

int tomoxide_iter_update(void *vol, const void *cw, const void *corr, size_t n, void *stream) {
    long long total = (long long)n;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    iter_update_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)vol, (const float *)cw, (const float *)corr, total);
    return (int)cudaGetLastError();
}

int tomoxide_iter_recip_thresh(void *out, const void *x, float thr, size_t n, void *stream) {
    long long total = (long long)n;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    iter_recip_thresh_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)out, (const float *)x, thr, total);
    return (int)cudaGetLastError();
}

}  // extern "C"
