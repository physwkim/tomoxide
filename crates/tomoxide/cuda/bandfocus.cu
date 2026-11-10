// Kernels of `LaminoBandFocus` (src/cuda/mod.rs): a stack's frames moved on
// the device, the focus of a laminography band back-projected from them, and
// that focus's derivative over every frame's shift and the centre.
//
// The move is `PreppedStack::moved` at roll 0 (cubic B-spline of exp(−p) at
// (i + dy, j + dx), then −log), read from coefficients made on the host by
// `prep::alignment::spline_coeffs`. The derivative is exact for the
// reconstruction the band comes from: `framedot3_ker` reads the filtered
// frames at the same taps with the same weights as `backprojection_ker`.

#include <cuda_runtime.h>
#include <math.h>

#define SPLINE_PAD 12 // prep::alignment::SPLINE_PAD

// prep::alignment's bspline_weights, and their derivative over f.
static __device__ __forceinline__ void bspline_w(float f, float *w)
{
    float one = 1.0f - f, sq = f * f, one_sq = one * one;
    w[0] = (1.0f / 6.0f) * one_sq * one;
    w[1] = 2.0f / 3.0f - 0.5f * sq * (2.0f - f);
    w[2] = 2.0f / 3.0f - 0.5f * one_sq * (2.0f - one);
    w[3] = (1.0f / 6.0f) * sq * f;
}

static __device__ __forceinline__ void bspline_dw(float f, float *d)
{
    float one = 1.0f - f;
    d[0] = -0.5f * one * one;
    d[1] = -2.0f * f + 1.5f * f * f;
    d[2] = 2.0f * one - 1.5f * one * one;
    d[3] = 0.5f * f * f;
}

// m [rows][alen][n]: frames a0 .. a0 + alen moved by (dy, dx), as the sinogram
// chunk the filter and back-projection read. With dmx/dmy non-null, also
// dm/d dx and dm/d dy there (0 where minus_log clamps).
static __global__ void move_ker(float *m, float *dmx, float *dmy, const float *coef,
                                const float *dy, const float *dx, int a0, int alen, int rows, int n)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int t = blockIdx.z;
    if (j >= n || i >= rows)
        return;
    int h = rows + 2 * SPLINE_PAD, w = n + 2 * SPLINE_PAD;
    double cy = (rows - 1) / 2.0, cx = (n - 1) / 2.0;
    // move_frames' position, then spline_at's padding offset.
    double r = ((double)i + (double)dy[a0 + t] - cy) + cy + SPLINE_PAD;
    double c = ((double)j + (double)dx[a0 + t] - cx) + cx + SPLINE_PAD;
    double fr = floor(r), fc = floor(c);
    float wr[4], wc[4], dwr[4], dwc[4];
    bspline_w((float)(r - fr), wr);
    bspline_w((float)(c - fc), wc);
    int cols[4];
#pragma unroll
    for (int k = 0; k < 4; k++)
        cols[k] = min(max((int)fc - 1 + k, 0), w - 1);
    const float *cf = coef + (size_t)(a0 + t) * h * w;
    bool grad = dmx != nullptr;
    if (grad)
    {
        bspline_dw((float)(r - fr), dwr);
        bspline_dw((float)(c - fc), dwc);
    }
    float s = 0.0f, sx = 0.0f, sy = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; k++)
    {
        const float *row = cf + (size_t)min(max((int)fr - 1 + k, 0), h - 1) * w;
        float v = wc[0] * row[cols[0]] + wc[1] * row[cols[1]] + wc[2] * row[cols[2]] +
                  wc[3] * row[cols[3]];
        s += wr[k] * v;
        if (grad)
        {
            sx += wr[k] * (dwc[0] * row[cols[0]] + dwc[1] * row[cols[1]] +
                           dwc[2] * row[cols[2]] + dwc[3] * row[cols[3]]);
            sy += dwr[k] * v;
        }
    }
    // minus_log
    float out = -logf(fmaxf(s, 1e-6f));
    bool ok = isfinite(out);
    size_t o = ((size_t)i * alen + t) * n + j;
    m[o] = ok ? out : 0.0f;
    if (grad)
    {
        bool live = ok && s > 1e-6f;
        dmx[o] = live ? -sx / s : 0.0f;
        dmy[o] = live ? -sy / s : 0.0f;
    }
}

// Sum `s` over the block into one atomicAdd on `out`.
static __device__ __forceinline__ void block_add(double *out, double s)
{
    __shared__ double red[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0)
        red[warp] = s;
    __syncthreads();
    if (warp == 0)
    {
        s = lane < (int)(blockDim.x >> 5) ? red[lane] : 0.0;
        for (int o = 16; o > 0; o >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0)
            atomicAdd(out, s);
    }
}

// prof[z] += slice_focus's sum for slice z of `v` [nb][n][n]: gx² + gy² over
// the pixels `mask` [n][n] keeps.
static __global__ void focus_ker(double *prof, const float *v, const unsigned char *mask, int n)
{
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockIdx.y;
    double s = 0.0;
    if (p < n * n && mask[p])
    {
        const float *img = v + (size_t)z * n * n;
        double gx = (double)(img[p + 1] - img[p - 1]);
        double gy = (double)(img[p + n] - img[p - n]);
        s = gx * gx + gy * gy;
    }
    block_add(prof + z, s);
}

// r = scale · d(Σ_z prof[z]) / d v: each kept pixel's gx reaches its x
// neighbours with opposite signs, its gy its y neighbours. The mask keeps no
// border pixel, so a kept neighbour's own neighbours are in the slice.
static __global__ void focus_grad_ker(float *r, const float *v, const unsigned char *mask,
                                      float scale, int n)
{
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockIdx.y;
    if (p >= n * n)
        return;
    int y = p / n, x = p - y * n;
    const float *img = v + (size_t)z * n * n;
    float g = 0.0f;
    if (x > 0 && mask[p - 1])
        g += img[p] - img[p - 2];
    if (x < n - 1 && mask[p + 1])
        g -= img[p + 2] - img[p];
    if (y > 0 && mask[p - n])
        g += img[p] - img[p - 2 * n];
    if (y < n - 1 && mask[p + n])
        g -= img[p + 2 * n] - img[p];
    r[(size_t)z * n * n + p] = scale * g;
}

// out[t][k] += Σ_voxels r · (h_k at angle t, read as backprojection_ker reads
// its sinogram), for h_0..2 = hx, hy, hc, each [nz][nproj][n] like the
// back-projected chunk. FD_ANG angles per block share one read of r.
#define FD_ANG 8
static __global__ void framedot3_ker(double *out, const float *r, const float *hx,
                                     const float *hy, const float *hc, const float *theta,
                                     float phi, float axis, int sz, int ncz, int n, int nz,
                                     int rh, int nproj)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int t0 = FD_ANG * blockIdx.z;
    int tid = threadIdx.x + blockDim.x * threadIdx.y;
    __shared__ float sc[FD_ANG], ss[FD_ANG];
    __shared__ double red[3 * FD_ANG][32];
    if (tid < FD_ANG && t0 + tid < nproj)
    {
        sc[tid] = __cosf(theta[t0 + tid]);
        ss[tid] = __sinf(theta[t0 + tid]);
    }
    __syncthreads();

    float acc[3 * FD_ANG];
#pragma unroll
    for (int k = 0; k < 3 * FD_ANG; k++)
        acc[k] = 0.0f;
    if (tx < n && ty < n)
    {
        float cphi = __cosf(phi);
        float sphi = __sinf(phi);
        size_t nn = (size_t)n * nproj;
        for (int tz = 0; tz < ncz; tz++)
        {
            float val = r[tx + ty * n + (size_t)tz * n * n];
            if (val == 0.0f)
                continue;
#pragma unroll
            for (int a = 0; a < FD_ANG; a++)
            {
                if (t0 + a >= nproj)
                    continue;
                // backprojection_ker's u, v, tap and guard, term for term.
                float ctheta = sc[a];
                float stheta = ss[a];
                float R0 = ctheta, R1 = stheta;
                float R3 = stheta * cphi, R4 = -ctheta * cphi, R5 = sphi;
                float u = R0 * (tx - n / 2) + R1 * (ty - n / 2) + axis;
                float v = R3 * (tx - n / 2) + R4 * (ty - n / 2) + R5 * (tz + sz - rh / 2) + nz / 2;
                int ur = (int)(u - 1e-5f);
                int vr = (int)(v - 1e-5f);
                if ((u > -1e-5f) & (ur < n - 1) & (v > -1e-5f) & (vr < nz - 1))
                {
                    float fu = u - ur;
                    float fv = v - vr;
                    size_t o = (size_t)vr * nn + (size_t)(t0 + a) * n + ur;
                    float w00 = (1 - fu) * (1 - fv), w10 = fu * (1 - fv);
                    float w01 = (1 - fu) * fv, w11 = fu * fv;
                    acc[3 * a] += val * (w00 * __ldg(hx + o) + w10 * __ldg(hx + o + 1) +
                                         w01 * __ldg(hx + o + nn) + w11 * __ldg(hx + o + nn + 1));
                    acc[3 * a + 1] += val * (w00 * __ldg(hy + o) + w10 * __ldg(hy + o + 1) +
                                             w01 * __ldg(hy + o + nn) + w11 * __ldg(hy + o + nn + 1));
                    acc[3 * a + 2] += val * (w00 * __ldg(hc + o) + w10 * __ldg(hc + o + 1) +
                                             w01 * __ldg(hc + o + nn) + w11 * __ldg(hc + o + nn + 1));
                }
            }
        }
    }

    int lane = tid & 31, warp = tid >> 5;
#pragma unroll
    for (int k = 0; k < 3 * FD_ANG; k++)
    {
        double x = acc[k];
        for (int o = 16; o > 0; o >>= 1)
            x += __shfl_down_sync(0xffffffffu, x, o);
        if (lane == 0)
            red[k][warp] = x;
    }
    __syncthreads();
    int nwarps = (blockDim.x * blockDim.y) / 32;
    if (tid < 3 * FD_ANG && t0 + tid / 3 < nproj)
    {
        double s = 0.0;
        for (int w = 0; w < nwarps; w++)
            s += red[tid][w];
        atomicAdd(&out[3 * (size_t)t0 + tid], s);
    }
}

// See move_ker; dmx/dmy both null for the frames alone. Returns cudaGetLastError().
extern "C" int tomoxide_bf_move(void *m, void *dmx, void *dmy, const void *coef, const float *dy,
                                const float *dx, int a0, int alen, int rows, int n, void *stream)
{
    dim3 block(32, 8, 1);
    dim3 grid((unsigned)((n + 31) / 32), (unsigned)((rows + 7) / 8), (unsigned)alen);
    move_ker<<<grid, block, 0, (cudaStream_t)stream>>>((float *)m, (float *)dmx, (float *)dmy,
                                                        (const float *)coef, dy, dx, a0, alen,
                                                        rows, n);
    return (int)cudaGetLastError();
}

// `prof` [nb] (pre-zeroed doubles) += focus_ker's sums. Returns cudaGetLastError().
extern "C" int tomoxide_bf_focus(void *prof, const void *v, const void *mask, int nb, int n,
                                 void *stream)
{
    int block = 256;
    dim3 grid((unsigned)((n * n + block - 1) / block), (unsigned)nb, 1);
    focus_ker<<<grid, block, 0, (cudaStream_t)stream>>>((double *)prof, (const float *)v,
                                                         (const unsigned char *)mask, n);
    return (int)cudaGetLastError();
}

// `r` [nb][n][n] = focus_grad_ker's. Returns cudaGetLastError().
extern "C" int tomoxide_bf_focus_grad(void *r, const void *v, const void *mask, float scale,
                                      int nb, int n, void *stream)
{
    int block = 256;
    dim3 grid((unsigned)((n * n + block - 1) / block), (unsigned)nb, 1);
    focus_grad_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float *)r, (const float *)v, (const unsigned char *)mask, scale, n);
    return (int)cudaGetLastError();
}

// `out` [nproj][3] (pre-zeroed doubles) += framedot3_ker's sums; the geometry
// arguments are tomoxide_linerec_backproject's. Returns cudaGetLastError().
extern "C" int tomoxide_bf_framedot3(void *out, const void *r, const void *hx, const void *hy,
                                     const void *hc, const float *theta, float phi, float axis,
                                     int sz, int ncz, int nz, int n, int rh, int nproj,
                                     void *stream)
{
    dim3 block(32, 8, 1); // whole warps: the shuffles take every lane
    dim3 grid((unsigned)((n + 31) / 32), (unsigned)((n + 7) / 8),
              (unsigned)((nproj + FD_ANG - 1) / FD_ANG));
    framedot3_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (double *)out, (const float *)r, (const float *)hx, (const float *)hy, (const float *)hc,
        theta, phi, axis, sz, ncz, n, nz, rh, nproj);
    return (int)cudaGetLastError();
}
