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

// --- ordered-subset penalized ML (ospml / pml) ---

// De Pierro penalized-ML pixel update over the pre-update snapshot `old` (one
// thread per voxel). Solves the quadratic `2F·x'² + G·x' + E = 0` (positive
// root) with E = −x·corr, and over the in-grid 8-neighbours g (cardinal weight
// 1, diagonal 1/√2, normalized by the present-weight sum): F = Σ 2·reg·w_g·γ_g,
// G = sens − Σ 2·reg·w_g·γ_g·(x + x_g). γ_g = 1 (quadratic, has_delta=0) or
// 1/(1+|x−x_g|/δ) (hybrid). reg=0 ⇒ the MLEM/OSEM step −E/G. Verbatim port of
// the host `penalized_ml_update`, same neighbour order (so f/g accumulate
// identically) and same n/2-independent grid indexing.
__global__ void iter_pml_update_ker(float *vol, const float *old, const float *corr,
                                    const float *sens, float reg, float delta, int has_delta, int n,
                                    int nz) {
    const float S = 0.70710678118654752440f; // 1/√2
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    if (ix >= n || iy >= n || z >= nz)
        return;
    long long base = (long long)z * n * n;
    long long idx = base + (long long)iy * n + ix;
    // Host NEIGHBORS order: (di=row, dj=col) cardinal then diagonal.
    const int di[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const int dj[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    const float raw[8] = {1.0f, 1.0f, 1.0f, 1.0f, S, S, S, S};
    float xij = old[idx];
    float e = -xij * corr[idx];
    float f = 0.0f, g = sens[idx];
    if (reg != 0.0f) {
        float wtot = 0.0f;
        for (int k = 0; k < 8; k++) {
            int ni = iy + di[k], nj = ix + dj[k];
            if (ni >= 0 && ni < n && nj >= 0 && nj < n)
                wtot += raw[k];
        }
        for (int k = 0; k < 8; k++) {
            int ni = iy + di[k], nj = ix + dj[k];
            if (ni >= 0 && ni < n && nj >= 0 && nj < n) {
                float xg = old[base + (long long)ni * n + nj];
                float gamma = has_delta ? (1.0f / (1.0f + fabsf((xij - xg) / delta))) : 1.0f;
                float coef = 2.0f * reg * (raw[k] / wtot) * gamma;
                f += coef;
                g -= coef * (xij + xg);
            }
        }
    }
    float out;
    if (f != 0.0f)
        out = (-g + sqrtf(g * g - 8.0f * f * e)) / (4.0f * f);
    else if (fabsf(g) > 1e-6f)
        out = -e / g;
    else
        out = xij;
    vol[idx] = out;
}

// --- gradient descent (grad / tikh) ---

// ax[i] = ax[i]*r - b[i]  — data proximal r·R x − b (in-place into `ax`).
__global__ void iter_grad_prox_ker(float *ax, const float *b, float r, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    ax[i] = ax[i] * r - b[i];
}

// grad[i] = coef * bpv[i]  — data gradient 2r·adj_scale·Rᵀ(…). Fresh write.
__global__ void iter_grad_assemble_ker(float *grad, const float *bpv, float coef, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    grad[i] = coef * bpv[i];
}

// grad[i] += two_reg1 * (vol[i] - prior[i])  — Tikhonov gradient 2·reg1·(x−prior).
__global__ void iter_grad_tikh_ker(float *grad, const float *vol, const float *prior,
                                   float two_reg1, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    grad[i] += two_reg1 * (vol[i] - prior[i]);
}

// x[i] *= s  — final unscale back to the physical domain (x ← r·x).
__global__ void iter_scale_inplace_ker(float *x, float s, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    x[i] *= s;
}

// vol[i] -= lambda[z]*grad[i], z = i/slice_len  — per-slice gradient step.
__global__ void iter_axpy_neg_slice_ker(float *vol, const float *grad, const float *lambda,
                                        long long slice_len, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    int z = (int)(i / slice_len);
    vol[i] -= lambda[z] * grad[i];
}

// Per-slice Barzilai–Borwein reductions: num[z] = Σ(x−x0)(g−g0), den[z] = Σ(g−g0)².
// One block per slice; blockDim must be a power of two (shared-mem tree reduce).
__global__ void iter_bb_reduce_ker(float *num, float *den, const float *x, const float *x0,
                                   const float *g, const float *g0, long long slice_len, int nz) {
    int z = blockIdx.x;
    if (z >= nz)
        return;
    extern __shared__ float sh[];
    float *snum = sh;
    float *sden = sh + blockDim.x;
    long long base = (long long)z * slice_len;
    float ln = 0.0f, ld = 0.0f;
    for (long long i = threadIdx.x; i < slice_len; i += blockDim.x) {
        float dx = x[base + i] - x0[base + i];
        float dg = g[base + i] - g0[base + i];
        ln += dx * dg;
        ld += dg * dg;
    }
    snum[threadIdx.x] = ln;
    sden[threadIdx.x] = ld;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            snum[threadIdx.x] += snum[threadIdx.x + s];
            sden[threadIdx.x] += sden[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        num[z] = snum[0];
        den[z] = sden[0];
    }
}

// lambda[z] = fixed_step≥0 ? fixed_step : (is_first ? 1e-3 : (den≠0 ? num/den : 1e-3)).
__global__ void iter_bb_lambda_ker(float *lambda, const float *num, const float *den,
                                   float fixed_step, int is_first, int nz) {
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= nz)
        return;
    float lam;
    if (fixed_step >= 0.0f)
        lam = fixed_step;
    else if (is_first)
        lam = 1e-3f;
    else
        lam = (den[z] != 0.0f) ? (num[z] / den[z]) : 1e-3f;
    lambda[z] = lam;
}

extern "C" {

int tomoxide_iter_pml_update(void *vol, const void *old, const void *corr, const void *sens,
                             float reg, float delta, int has_delta, size_t n, size_t nz,
                             void *stream) {
    dim3 block(16, 16, 1);
    dim3 grid(((unsigned)n + 15) / 16, ((unsigned)n + 15) / 16, (unsigned)nz);
    iter_pml_update_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)vol, (const float *)old, (const float *)corr, (const float *)sens, reg, delta,
        has_delta, (int)n, (int)nz);
    return (int)cudaGetLastError();
}

int tomoxide_iter_grad_prox(void *ax, const void *b, float r, size_t n, void *stream) {
    long long total = (long long)n;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    iter_grad_prox_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)ax, (const float *)b, r, total);
    return (int)cudaGetLastError();
}

int tomoxide_iter_grad_assemble(void *grad, const void *bpv, float coef, size_t n, void *stream) {
    long long total = (long long)n;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    iter_grad_assemble_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)grad, (const float *)bpv, coef, total);
    return (int)cudaGetLastError();
}

int tomoxide_iter_grad_tikh(void *grad, const void *vol, const void *prior, float two_reg1,
                            size_t n, void *stream) {
    long long total = (long long)n;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    iter_grad_tikh_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)grad, (const float *)vol, (const float *)prior, two_reg1, total);
    return (int)cudaGetLastError();
}

int tomoxide_iter_scale_inplace(void *x, float s, size_t n, void *stream) {
    long long total = (long long)n;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    iter_scale_inplace_ker<<<grid, block, 0, (cudaStream_t)stream>>>((float *)x, s, total);
    return (int)cudaGetLastError();
}

int tomoxide_iter_axpy_neg_slice(void *vol, const void *grad, const void *lambda, size_t slice_len,
                                 size_t total_n, void *stream) {
    long long total = (long long)total_n;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    iter_axpy_neg_slice_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)vol, (const float *)grad, (const float *)lambda, (long long)slice_len, total);
    return (int)cudaGetLastError();
}

int tomoxide_iter_bb_reduce(void *num, void *den, const void *x, const void *x0, const void *g,
                            const void *g0, size_t slice_len, size_t nz, void *stream) {
    int block = 256;
    int grid = (int)nz;
    size_t shmem = 2 * (size_t)block * sizeof(float);
    iter_bb_reduce_ker<<<grid, block, shmem, (cudaStream_t)stream>>>(
        (float *)num, (float *)den, (const float *)x, (const float *)x0, (const float *)g,
        (const float *)g0, (long long)slice_len, (int)nz);
    return (int)cudaGetLastError();
}

int tomoxide_iter_bb_lambda(void *lambda, const void *num, const void *den, float fixed_step,
                            int is_first, size_t nz, void *stream) {
    int block = 256;
    int grid = (int)(((int)nz + block - 1) / block);
    iter_bb_lambda_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)lambda, (const float *)num, (const float *)den, fixed_step, is_first, (int)nz);
    return (int)cudaGetLastError();
}

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
