// Parallel-beam forward projection (Radon transform) — the exact discrete
// transpose of `kernels_linerec.cuh::backprojection_ker`.
//
// The voxel-driven back-projector gathers, for output voxel
//   j = tx + ty*n + tz*n*n,
//   f[j] += c * Σ_t  bilinear(g; u(j,t), v(j,t)),
// i.e. as a matrix `f = (c·W)·g` where `W` is the bilinear gather operator and
// `c` the caller's gain (π/nproj for FBP's angular quadrature, 1 for the
// iterative adjoint — see cfunc_linerec.cu). This forward projector is the
// iterative solvers' `A = Wᵀ`, the *unweighted* scatter transpose: each voxel
// scatters its value into the *same* (u,v) taps with the *same* bilinear
// weights and no gain, so with the iterative back-projection (gain 1) the pair
// {A, Aᵀ} = {Wᵀ, W} is a true adjoint pair AND `A` is the plain line-integral
// Radon transform (unit pixel spacing) — a converged solve of `A x = p` yields
// the physical μ, matching ART/BART and tomopy `project.c`. Keeping the
// geometry byte-identical to `backprojection_ker` — the (un-flipped) row `ty`,
// the centre `axis`, the rotation matrix `R`, the `(int)(·-1e-5)` tap, and the
// in-bounds guard — is what makes the pair exact, so SIRT/MLEM/OSEM/… converge.
// The shared guard `vr < nz-1` drops slice 0 for the parallel beam (`v = tz` ⇒
// `vr = tz-1`), the documented ≥2-slice rule, exactly as the back-projector
// does.
//
// f32 only (the iterative suite is f32); built into libtomoxide_cuda_kernels.a
// by build.rs's `.cu` glob.

#include <cuda_runtime.h>
#include <math.h>

// Scatter transpose of backprojection_ker (see file header). `g` is the output
// sinogram [nz][nproj][n] (must be pre-zeroed; the kernel only atomic-adds),
// `f` the input volume [ncz][n][n]. `ncz` is the volume z-extent (== nz for the
// parallel beam; == the laminography recon height `rh` otherwise) and `sz` the
// global z-offset of this volume block — both matching backprojection_ker so the
// pair {A, Aᵀ} stays an exact transpose for a tilted axis (phi ≠ π/2) and an
// output height that differs from the detector-row count. `axis` is the detector
// column voxel column n/2 lands on, the rotation centre; the pair stays exact
// at any centre backprojection_ker is given too.
static __global__ void forwardprojection_ker(float *g, const float *f, const float *theta,
                                             float phi, float axis, int sz, int ncz, int n, int nz,
                                             int rh, int nproj)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    // cos/sin(theta[t]) cached once per block, identical for every voxel
    // (mirrors backprojection_ker). Filled before the bounds guard so every
    // thread reaches __syncthreads.
    extern __shared__ float ssc[]; // [0,nproj)=cos, [nproj,2*nproj)=sin
    {
        int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
        int nthreads = blockDim.x * blockDim.y * blockDim.z;
        for (int t = tid; t < nproj; t += nthreads)
        {
            ssc[t] = __cosf(theta[t]);
            ssc[nproj + t] = __sinf(theta[t]);
        }
    }
    __syncthreads();
    if (tx >= n || ty >= n || tz >= ncz)
        return;

    // Same (un-flipped) row and centre as the back-projector's accumulation index.
    float val = f[tx + ty * n + tz * n * n];
    if (val == 0.0f)
        return; // a zero voxel scatters nothing

    float cphi = __cosf(phi);
    float sphi = __sinf(phi);
    float R[6] = {};
    for (int t = 0; t < nproj; t++)
    {
        float ctheta = ssc[t];
        float stheta = ssc[nproj + t];
        R[0] = ctheta;       R[1] = stheta;        R[2] = 0;
        R[3] = stheta * cphi; R[4] = -ctheta * cphi; R[5] = sphi;
        float u = R[0] * (tx - n / 2) + R[1] * (ty - n / 2) + axis;
        // Volume z centred on rh/2 (the volume's own midpoint), detector row on
        // nz/2 — mirrors backprojection_ker so {A, Aᵀ} stays an exact transpose.
        float v = R[3] * (tx - n / 2) + R[4] * (ty - n / 2) + R[5] * (tz + sz - rh / 2) + nz / 2;

        int ur = (int)(u - 1e-5f);
        int vr = (int)(v - 1e-5f);

        // Same in-bounds guard as the gather; out-of-bounds taps contribute 0.
        // (int) truncates toward zero, so the lower bound tests u and v, with the
        // 1e-5 margin the tap gives the upper one: u in (-1, 0) would otherwise
        // land on column 0 with a weight above 1.
        if ((u > -1e-5f) & (ur < n - 1) & (v > -1e-5f) & (vr < nz - 1))
        {
            float fu = u - ur;
            float fv = v - vr;
            // Transpose of the 4-tap bilinear gather: scatter val into the same
            // taps with the same weights.
            atomicAdd(&g[ur + 0 + t * n + (vr + 0) * n * nproj], val * (1 - fu) * (1 - fv));
            atomicAdd(&g[ur + 1 + t * n + (vr + 0) * n * nproj], val * (0 + fu) * (1 - fv));
            atomicAdd(&g[ur + 0 + t * n + (vr + 1) * n * nproj], val * (1 - fu) * (0 + fv));
            atomicAdd(&g[ur + 1 + t * n + (vr + 1) * n * nproj], val * (0 + fu) * (0 + fv));
        }
    }
}

// C-ABI host wrapper. `g` (output sinogram [nz][nproj][n]) must be pre-zeroed.
// Returns cudaGetLastError(): 0 once the launch is queued.
// Unweighted (gain 1): the geometry matches backprojection_ker with gain 1 (the
// iterative back-projection call), so {forward, back} are a true adjoint pair
// and the forward is the plain line-integral Radon transform — matching the CPU
// and wgpu forward projectors, so the iterative suite is scale-unified across
// backends at the physical μ. The launch geometry mirrors
// cfunc_linerec::backprojection.
// `ncz` = volume z-extent of this block (== `nz` parallel beam, up to the recon
// height for laminography), `sz` = the volume block's global z-offset, `nz` =
// detector rows, `rh` = FULL reconstruction height the global z is centred on
// (== `nz` parallel beam, `lamino_recon_height` otherwise), `axis` = detector
// column of voxel column n/2 (the rotation centre).
extern "C" int tomoxide_forwardproject(void *g, const void *f, const float *theta, float phi,
                                        float axis, int sz, int ncz, int nz, int n, int rh,
                                        int nproj, void *stream)
{
    dim3 dimBlock(32, 32, 1);
    dim3 grid((unsigned)ceil(n / 32.0), (unsigned)ceil(n / 32.0), (unsigned)ncz);
    size_t shmem = 2 * (size_t)nproj * sizeof(float); // cos/sin(theta) cache
    forwardprojection_ker<<<grid, dimBlock, shmem, (cudaStream_t)stream>>>(
        (float *)g, (const float *)f, theta, phi, axis, sz, ncz, n, nz, rh, nproj);
    return (int)cudaGetLastError();
}

// Each frame's derivative along its columns and along its rows, as numpy's
// `gradient`: central differences inside, one-sided at the edges. `m` is
// [nframes][rows][n]; `d` is [2][nframes][rows][n], columns first.
static __global__ void frame_derivatives_ker(const float *m, float *d, long long nframes, int rows,
                                             int n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = nframes * rows * n;
    if (i >= total)
        return;
    int x = (int)(i % n);
    int y = (int)(i / n % rows);
    long long l = x == 0 ? i : i - 1, r = x == n - 1 ? i : i + 1;
    d[i] = (m[r] - m[l]) / (float)(r - l);
    long long u = y == 0 ? i : i - n, w = y == rows - 1 ? i : i + n;
    d[total + i] = (m[w] - m[u]) / (float)((w - u) / n);
}

// Per-angle inner product of forwardprojection_ker's frame with the given
// frames, without forming the frame: for angle t and c = 0 (hx), 1 (hy),
//   out[2t + c] = Σ_p (A_t f)(p) · h_c[t](p).
// A voxel gathers h at the very taps, with the very weights, it would scatter
// into, so this is that sum up to its order, with no atomics on frames. Each
// thread sums its voxel column over z for FD_ANG angles at once, so the band is
// read once per FD_ANG angles; a block adds its partials in double, one
// atomicAdd per value. `hx`/`hy` are [nproj][nz][n], `out` [nproj][2] pre-zeroed.
#define FD_ANG 8
static __global__ void framedot_ker(double *out, const float *f, const float *hx, const float *hy,
                                    const float *theta, float phi, float axis, int sz, int ncz,
                                    int n, int nz, int rh, int nproj)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int t0 = FD_ANG * blockIdx.z;
    int tid = threadIdx.x + blockDim.x * threadIdx.y;
    __shared__ float sc[FD_ANG], ss[FD_ANG];
    __shared__ double red[2 * FD_ANG][32];
    if (tid < FD_ANG && t0 + tid < nproj)
    {
        sc[tid] = __cosf(theta[t0 + tid]);
        ss[tid] = __sinf(theta[t0 + tid]);
    }
    __syncthreads();

    float ax[FD_ANG], ay[FD_ANG];
#pragma unroll
    for (int a = 0; a < FD_ANG; a++)
        ax[a] = ay[a] = 0.0f;
    if (tx < n && ty < n)
    {
        float cphi = __cosf(phi);
        float sphi = __sinf(phi);
        for (int tz = 0; tz < ncz; tz++)
        {
            float val = f[tx + ty * n + tz * n * n];
            if (val == 0.0f)
                continue;
#pragma unroll
            for (int a = 0; a < FD_ANG; a++)
            {
                if (t0 + a >= nproj)
                    continue;
                // forwardprojection_ker's u, v, tap and guard, term for term.
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
                    size_t o = ((size_t)(t0 + a) * nz + vr) * n + ur;
                    float w00 = (1 - fu) * (1 - fv), w10 = fu * (1 - fv);
                    float w01 = (1 - fu) * fv, w11 = fu * fv;
                    ax[a] += val * (w00 * __ldg(hx + o) + w10 * __ldg(hx + o + 1) +
                                    w01 * __ldg(hx + o + n) + w11 * __ldg(hx + o + n + 1));
                    ay[a] += val * (w00 * __ldg(hy + o) + w10 * __ldg(hy + o + 1) +
                                    w01 * __ldg(hy + o + n) + w11 * __ldg(hy + o + n + 1));
                }
            }
        }
    }

    int lane = tid & 31, warp = tid >> 5;
#pragma unroll
    for (int a = 0; a < FD_ANG; a++)
    {
        double x = ax[a], y = ay[a];
        for (int o = 16; o > 0; o >>= 1)
        {
            x += __shfl_down_sync(0xffffffffu, x, o);
            y += __shfl_down_sync(0xffffffffu, y, o);
        }
        if (lane == 0)
        {
            red[2 * a][warp] = x;
            red[2 * a + 1][warp] = y;
        }
    }
    __syncthreads();
    int nwarps = (blockDim.x * blockDim.y) / 32;
    if (tid < 2 * FD_ANG && t0 + tid / 2 < nproj)
    {
        double s = 0.0;
        for (int w = 0; w < nwarps; w++)
            s += red[tid][w];
        atomicAdd(&out[2 * (size_t)t0 + tid], s);
    }
}

// `d` [2][nframes][rows][n] from `m` [nframes][rows][n]; see
// frame_derivatives_ker. Returns cudaGetLastError().
extern "C" int tomoxide_frame_derivatives(const void *m, void *d, long long nframes, int rows,
                                          int n, void *stream)
{
    int block = 256;
    long long total = nframes * rows * n;
    frame_derivatives_ker<<<(unsigned)((total + block - 1) / block), block, 0,
                            (cudaStream_t)stream>>>((const float *)m, (float *)d, nframes, rows, n);
    return (int)cudaGetLastError();
}

// `out` [nproj][2] (pre-zeroed doubles) += framedot_ker's sums; the geometry
// arguments are tomoxide_forwardproject's. Returns cudaGetLastError().
extern "C" int tomoxide_framedot(void *out, const void *f, const void *hx, const void *hy,
                                 const float *theta, float phi, float axis, int sz, int ncz,
                                 int nz, int n, int rh, int nproj, void *stream)
{
    dim3 block(32, 8, 1); // whole warps: framedot_ker's shuffles take every lane
    dim3 grid((unsigned)((n + 31) / 32), (unsigned)((n + 7) / 8),
              (unsigned)((nproj + FD_ANG - 1) / FD_ANG));
    framedot_ker<<<grid, block, 0, (cudaStream_t)stream>>>(
        (double *)out, (const float *)f, (const float *)hx, (const float *)hy, theta, phi, axis,
        sz, ncz, n, nz, rh, nproj);
    return (int)cudaGetLastError();
}
