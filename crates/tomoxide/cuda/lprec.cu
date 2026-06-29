// Device-resident runtime kernels for log-polar (lprec) reconstruction.
//
// Ports the per-slice runtime of `recon/lprec.rs::process_row` to CUDA so the
// cubic-B-spline gather/scatter and the spline prefilter run on the GPU instead
// of the host (the host interpolation, not the FFT, was lprec's bottleneck). The
// geometry grids (kfull convolution kernel, per-span log-polar/Cartesian coords,
// target index sets) are precomputed once on the host by `build_grids` and
// uploaded; these kernels consume them. The 2-D FFT convolution itself reuses
// the shared `tomoxide_fft_2d` (cuFFT) entry point, called between gather and
// scatter from the Rust orchestrator.
//
// Buffer layouts (row-major, batched over `nz` slices):
//   g   [nz, nproj, n]            real, the filtered sinogram chunk (also the
//                                 in-place spline-coefficient buffer)
//   flc [nz, nrho, ntheta_pad]    in-place R2C buffer: row-padded real
//                                 (ntheta_pad = 2*(ntheta/2+1)) overlaying the
//                                 half-complex spectrum [nz, nrho, ntheta/2+1].
//                                 The log-polar function is real, so the forward
//                                 R2C / inverse C2R run at ~half the cost and
//                                 memory of a full C2C transform.
//   f   [nz, n, n]                real, the Cartesian output chunk
// Index sets (`lpids`/`wids` over flc grid, `cids` over the f grid) are span-
// independent; only the float coordinate arrays differ per span.

#include <cuda_runtime.h>

// Cubic-B-spline pole, sqrt(3) - 2 (matches recon/lprec.rs POLE).
#define LPREC_POLE (-0.2679492f)
#define LPREC_BLK 256

// Cubic-B-spline basis weights at fractional offset f in [0,1); taps at
// floor-1 .. floor+2 (matches `bspline_weights`).
__device__ __forceinline__ void lprec_bspline_weights(float f, float w[4]) {
    float one = 1.0f - f;
    float sq = f * f;
    float one_sq = one * one;
    w[0] = (1.0f / 6.0f) * one_sq * one;
    w[1] = 2.0f / 3.0f - 0.5f * sq * (2.0f - f);
    w[2] = 2.0f / 3.0f - 0.5f * one_sq * (2.0f - one);
    w[3] = (1.0f / 6.0f) * sq * f;
}

__device__ __forceinline__ int lprec_wrap(int i, int n) {
    return ((i % n) + n) % n;
}

// 4x4 cubic-B-spline interpolation of a real coeff grid at (x, y), wrap
// addressing over the logical `width x height` grid. `stride` is the physical
// row stride in elements (≥ width): it differs from `width` when the grid is
// row-padded (the in-place R2C/C2R real buffer is padded to `2*(width/2+1)`),
// so wrap uses `width` for the column index but addressing uses `stride`.
// Mirrors `cubic_interp2d`.
__device__ float lprec_cubic_interp2d(const float *coeffs, int width, int height,
                                      int stride, float x, float y) {
    float ixf = floorf(x);
    float iyf = floorf(y);
    float wx[4], wy[4];
    lprec_bspline_weights(x - ixf, wx);
    lprec_bspline_weights(y - iyf, wy);
    int ix = (int)ixf;
    int iy = (int)iyf;
    float sum = 0.0f;
    for (int j = 0; j < 4; ++j) {
        int py = lprec_wrap(iy - 1 + j, height);
        long row = (long)py * stride;
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            int px = lprec_wrap(ix - 1 + i, width);
            acc += wx[i] * coeffs[row + px];
        }
        sum += wy[j] * acc;
    }
    return sum;
}

// In-place cubic-B-spline prefilter of one strided line (samples -> spline
// coefficients), clamped boundary over a 12-sample horizon. Mirrors
// `convert_to_coeffs`. `stride` lets the angle-axis pass run down columns.
__device__ void lprec_convert_coeffs(float *c, int n, int stride) {
    if (n < 2) {
        return;
    }
    float lambda = (1.0f - LPREC_POLE) * (1.0f - 1.0f / LPREC_POLE);
    int horizon = n < 12 ? n : 12;
    float zn = LPREC_POLE;
    float sum = c[0];
    for (int k = 0; k < horizon; ++k) {
        sum += zn * c[(long)k * stride];
        zn *= LPREC_POLE;
    }
    c[0] = lambda * sum;
    float prev = c[0];
    for (int k = 1; k < n; ++k) {
        float v = lambda * c[(long)k * stride] + LPREC_POLE * prev;
        c[(long)k * stride] = v;
        prev = v;
    }
    long last = (long)(n - 1) * stride;
    c[last] *= LPREC_POLE / (LPREC_POLE - 1.0f);
    prev = c[last];
    for (int k = n - 2; k >= 0; --k) {
        float v = LPREC_POLE * (prev - c[(long)k * stride]);
        c[(long)k * stride] = v;
        prev = v;
    }
}

// Spline prefilter along the detector axis: one thread per (slice, angle) line.
__global__ void lprec_prefilter_rows_ker(float *g, int nz, int nproj, int n) {
    long t = (long)blockIdx.x * LPREC_BLK + threadIdx.x;
    long total = (long)nz * nproj;
    if (t >= total) {
        return;
    }
    long s = t / nproj;
    long a = t % nproj;
    lprec_convert_coeffs(g + (s * nproj + a) * n, n, 1);
}

// Spline prefilter along the angle axis: one thread per (slice, detector) column.
__global__ void lprec_prefilter_cols_ker(float *g, int nz, int nproj, int n) {
    long t = (long)blockIdx.x * LPREC_BLK + threadIdx.x;
    long total = (long)nz * n;
    if (t >= total) {
        return;
    }
    long s = t / n;
    long d = t % n;
    lprec_convert_coeffs(g + s * (long)nproj * n + d, nproj, n);
}

// Gather: polar -> log-polar cubic interpolation, accumulated into the padded
// real work buffer. `xs` is the detector coord (width n), `ys` the angle coord
// (height nproj); atomicAdd because the wrapping set can land on the same target
// as the main set. `flc` is the in-place R2C real buffer, row-padded to
// `ntheta_pad = 2*(ntheta/2+1)` per row (per-slice stride `nrho*ntheta_pad`);
// `targets` are the padded within-slice indices (`row*ntheta_pad + col`).
__global__ void lprec_gather_ker(const float *g, float *flc, const int *targets,
                                 const float *xs, const float *ys, int npts,
                                 int nz, int nproj, int n, int nrho,
                                 int ntheta_pad) {
    long t = (long)blockIdx.x * LPREC_BLK + threadIdx.x;
    long total = (long)nz * npts;
    if (t >= total) {
        return;
    }
    long s = t / npts;
    int idx = (int)(t % npts);
    const float *gs = g + s * (long)nproj * n;
    float val = lprec_cubic_interp2d(gs, n, nproj, n, xs[idx], ys[idx]);
    long base = s * (long)nrho * ntheta_pad + targets[idx];
    atomicAdd(&flc[base], val);
}

// Broadcast complex multiply on the half-complex spectrum: flc[s, i] *= kfull[i]
// over the [nrho, ntheta_c] grid (ntheta_c = ntheta/2+1, the R2C half-width).
__global__ void lprec_cmul_ker(float2 *flc, const float2 *kfull, int nz, long ng) {
    long t = (long)blockIdx.x * LPREC_BLK + threadIdx.x;
    long total = (long)nz * ng;
    if (t >= total) {
        return;
    }
    long i = t % ng;
    float2 a = flc[t];
    float2 b = kfull[i];
    flc[t] = make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Scatter: log-polar -> Cartesian disk cubic interpolation, accumulated into f
// (x2 folds tomocupy's 2/(nrho*ntheta) scale; the inverse C2R already applied
// 1/(nrho*ntheta)). `xs` is the theta coord (logical width ntheta), `ys` the rho
// coord (height nrho); `flc` is the padded real buffer (row stride ntheta_pad).
// cids targets are distinct within a span, so a plain += is race free within one
// launch; spans accumulate across successive launches.
__global__ void lprec_scatter_ker(const float *flc, float *f, const int *targets,
                                  const float *xs, const float *ys, int npts,
                                  int nz, int n, int nrho, int ntheta,
                                  int ntheta_pad) {
    long t = (long)blockIdx.x * LPREC_BLK + threadIdx.x;
    long total = (long)nz * npts;
    if (t >= total) {
        return;
    }
    long s = t / npts;
    int idx = (int)(t % npts);
    const float *fs = flc + s * (long)nrho * ntheta_pad;
    float val =
        2.0f * lprec_cubic_interp2d(fs, ntheta, nrho, ntheta_pad, xs[idx], ys[idx]);
    f[s * (long)n * n + targets[idx]] += val;
}

extern "C" {

int tomoxide_lprec_prefilter_rows(void *g, int nz, int nproj, int n, void *stream) {
    long total = (long)nz * nproj;
    int grid = (int)((total + LPREC_BLK - 1) / LPREC_BLK);
    lprec_prefilter_rows_ker<<<grid, LPREC_BLK, 0, (cudaStream_t)stream>>>(
        (float *)g, nz, nproj, n);
    return cudaGetLastError();
}

int tomoxide_lprec_prefilter_cols(void *g, int nz, int nproj, int n, void *stream) {
    long total = (long)nz * n;
    int grid = (int)((total + LPREC_BLK - 1) / LPREC_BLK);
    lprec_prefilter_cols_ker<<<grid, LPREC_BLK, 0, (cudaStream_t)stream>>>(
        (float *)g, nz, nproj, n);
    return cudaGetLastError();
}

int tomoxide_lprec_gather(const void *g, void *flc, const void *targets,
                          const void *xs, const void *ys, int npts, int nz,
                          int nproj, int n, int nrho, int ntheta_pad,
                          void *stream) {
    long total = (long)nz * npts;
    if (total == 0) {
        return cudaSuccess;
    }
    int grid = (int)((total + LPREC_BLK - 1) / LPREC_BLK);
    lprec_gather_ker<<<grid, LPREC_BLK, 0, (cudaStream_t)stream>>>(
        (const float *)g, (float *)flc, (const int *)targets, (const float *)xs,
        (const float *)ys, npts, nz, nproj, n, nrho, ntheta_pad);
    return cudaGetLastError();
}

int tomoxide_lprec_cmul(void *flc, const void *kfull, int nz, int nrho,
                        int ntheta_c, void *stream) {
    long ng = (long)nrho * ntheta_c;
    long total = (long)nz * ng;
    int grid = (int)((total + LPREC_BLK - 1) / LPREC_BLK);
    lprec_cmul_ker<<<grid, LPREC_BLK, 0, (cudaStream_t)stream>>>(
        (float2 *)flc, (const float2 *)kfull, nz, ng);
    return cudaGetLastError();
}

int tomoxide_lprec_scatter(const void *flc, void *f, const void *targets,
                           const void *xs, const void *ys, int npts, int nz,
                           int n, int nrho, int ntheta, int ntheta_pad,
                           void *stream) {
    long total = (long)nz * npts;
    if (total == 0) {
        return cudaSuccess;
    }
    int grid = (int)((total + LPREC_BLK - 1) / LPREC_BLK);
    lprec_scatter_ker<<<grid, LPREC_BLK, 0, (cudaStream_t)stream>>>(
        (const float *)flc, (float *)f, (const int *)targets, (const float *)xs,
        (const float *)ys, npts, nz, n, nrho, ntheta, ntheta_pad);
    return cudaGetLastError();
}

} // extern "C"
