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
// geometry byte-identical to the back-projector — the (un-flipped) row `ty`,
// the centre `axis`, the (u, v) formulas, and the zero-padded tap (`proj_tap`)
// — is what makes the pair exact, so SIRT/MLEM/OSEM/… converge; both pairs
// compute their taps through projgeom.cuh.
// The parallel beam is a gather (forwardprojection_par_ker), the transpose of
// backprojection_par_ker; a tilted axis is the atomic scatter
// (forwardprojection_ker), the transpose of backprojection_ker.
//
// f32 only (the iterative suite is f32); built into libtomoxide_cuda_kernels.a
// by build.rs's `.cu` glob.

#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>

#include "projgeom.cuh"

// The voxel tile one block of forwardprojection_ker owns: FWD_X×FWD_Y columns
// of FWD_G slices. At one angle its taps span at most |(FWD_X, FWD_Y)| + 2
// detector columns and that plus FWD_G rows, so they fit the FWD_P² shared
// patch; a tap outside it goes straight to g. cos/sin are cached FWD_TC angles
// at a time, so shared memory does not grow with nproj.
#define FWD_X 32
#define FWD_Y 8
#define FWD_G 16
#define FWD_P 40
#define FWD_TC 512

// Scatter transpose of backprojection_ker (see file header). `g` is the output
// sinogram [nz][nproj][n] (must be pre-zeroed; the kernel only atomic-adds),
// `f` the input volume [ncz][n][n]. `ncz` is the volume z-extent (== nz for the
// parallel beam; == the laminography recon height `rh` otherwise) and `sz` the
// global z-offset of this volume block — both matching backprojection_ker so the
// pair {A, Aᵀ} stays an exact transpose for a tilted axis (phi ≠ π/2) and an
// output height that differs from the detector-row count. `axis` is the detector
// column voxel column n/2 lands on, the rotation centre; the pair stays exact
// at any centre backprojection_ker is given too.
// Per angle a thread sums its column's taps row by row in registers, adds each
// row into the block's shared detector patch, and the block flushes the patch
// to g: one global atomic per touched detector cell instead of four per voxel.
// Taps and weights are the per-voxel scatter's; only the summation order
// differs, which the atomics never fixed.
static __global__ void __launch_bounds__(FWD_X *FWD_Y)
    forwardprojection_ker(float *g, const float *f, const float *theta, float phi, float axis, int sz,
                          int ncz, int n, int nz, int rh, int nproj)
{
    __shared__ float ssc[2 * FWD_TC]; // [0,FWD_TC)=cos, [FWD_TC,2*FWD_TC)=sin of the chunk
    __shared__ float patch[FWD_P * FWD_P];
    const int tid = threadIdx.x + FWD_X * threadIdx.y;
    const int x0 = blockIdx.x * FWD_X, y0 = blockIdx.y * FWD_Y, z0 = blockIdx.z * FWD_G;
    const int tx = x0 + threadIdx.x, ty = y0 + threadIdx.y;
    const int xe = min(x0 + FWD_X, n) - 1, ye = min(y0 + FWD_Y, n) - 1;
    // Volume z centred on rh/2 (the volume's own midpoint), detector row on
    // nz/2 — the taps of backprojection_ker, through the same helpers.
    const int h = n / 2, cnt = min(FWD_G, ncz - z0), zb = z0 + sz - rh / 2;
    const float cphi = __cosf(phi), sphi = __sinf(phi);
    const size_t zs = (size_t)n * nproj; // g's detector-row stride

    // Same (un-flipped) row and centre as the back-projector's accumulation index.
    float val[FWD_G];
    bool any = false;
#pragma unroll
    for (int z = 0; z < FWD_G; z++)
    {
        val[z] = (tx < n && ty < n && z < cnt) ? f[tx + (size_t)ty * n + (size_t)(z0 + z) * n * n] : 0.f;
        any |= val[z] != 0.f;
    }
    if (!__syncthreads_or(any))
        return; // a zero tile scatters nothing
    for (int i = tid; i < FWD_P * FWD_P; i += FWD_X * FWD_Y)
        patch[i] = 0.f;

    for (int t0 = 0; t0 < nproj; t0 += FWD_TC)
    {
        // The previous angle's closing __syncthreads ordered its reads of ssc.
        const int tn = min(FWD_TC, nproj - t0);
        for (int i = tid; i < tn; i += FWD_X * FWD_Y)
        {
            ssc[i] = __cosf(theta[t0 + i]);
            ssc[FWD_TC + i] = __sinf(theta[t0 + i]);
        }
        __syncthreads();
        for (int k = 0; k < tn; k++)
        {
            const int t = t0 + k;
            const float c = ssc[k], s = ssc[FWD_TC + k];
            // u and v are monotone in x, y and z, so the tile's smallest taps,
            // the patch origin, come from its corners.
            const float um = fminf(fminf(proj_u(c, s, x0, y0, h, axis), proj_u(c, s, xe, y0, h, axis)),
                                   fminf(proj_u(c, s, x0, ye, h, axis), proj_u(c, s, xe, ye, h, axis)));
            const float vm = fminf(fminf(lam_vxy(c, s, cphi, x0, y0, h), lam_vxy(c, s, cphi, xe, y0, h)),
                                   fminf(lam_vxy(c, s, cphi, x0, ye, h), lam_vxy(c, s, cphi, xe, ye, h)));
            const int u0 = proj_tap(um, n).i;
            const int v0 = proj_tap(fminf(lam_v(vm, sphi, zb, nz), lam_v(vm, sphi, zb + cnt - 1, nz)), nz).i;
            const Tap cu = proj_tap(proj_u(c, s, tx, ty, h, axis), n);
            if (any & tap_any(cu))
            {
                const float fu = cu.f, vxy = lam_vxy(c, s, cphi, tx, ty, h);
                const int ur = cu.i, pu = ur - u0;
                // Adds detector cells (ur, r) and (ur+1, r), the ones that exist:
                // a tap off the detector is zero padding and is never written,
                // so every cell of the patch lies on the detector.
                auto emit = [&](int r, float wa, float wb) {
                    if (!tap_in(r, nz))
                        return;
                    const int pv = r - v0;
                    if ((pu >= 0) & (pu < FWD_P - 1) & (pv >= 0) & (pv < FWD_P))
                    {
                        float *q = patch + pu + pv * FWD_P;
                        if (cu.lo)
                            atomicAdd(q, wa);
                        if (cu.hi)
                            atomicAdd(q + 1, wb);
                    }
                    else
                    {
                        float *q = g + ur + (size_t)t * n + (size_t)r * zs;
                        if (cu.lo)
                            atomicAdd(q, wa);
                        if (cu.hi)
                            atomicAdd(q + 1, wb);
                    }
                };
                // (r0a, r0b) hold row `row`, (r1a, r1b) row `row`+1. v is
                // monotone in z, so a row the taps have moved past is complete.
                // Row -1 is a real pair (its row 0 exists), so "no row yet" is
                // a value no tap reaches.
                const int NO_ROW = INT_MIN / 2;
                int row = NO_ROW;
                float r0a = 0.f, r0b = 0.f, r1a = 0.f, r1b = 0.f;
#pragma unroll
                for (int z = 0; z < FWD_G; z++)
                {
                    const Tap rv = proj_tap(lam_v(vxy, sphi, zb + z, nz), nz);
                    const int vr = rv.i;
                    if ((val[z] != 0.f) & tap_any(rv))
                    {
                        if (vr == row + 1)
                        {
                            emit(row, r0a, r0b);
                            r0a = r1a, r0b = r1b, r1a = 0.f, r1b = 0.f;
                        }
                        else if (vr == row - 1)
                        {
                            emit(row + 1, r1a, r1b);
                            r1a = r0a, r1b = r0b, r0a = 0.f, r0b = 0.f;
                        }
                        else if (vr != row)
                        {
                            if (row != NO_ROW)
                            {
                                emit(row, r0a, r0b);
                                emit(row + 1, r1a, r1b);
                            }
                            r0a = 0.f, r0b = 0.f, r1a = 0.f, r1b = 0.f;
                        }
                        row = vr;
                        // Transpose of the 4-tap bilinear gather: val into the
                        // same taps with the same weights.
                        const float fv = rv.f, a = val[z];
                        r0a += a * (1 - fu) * (1 - fv);
                        r0b += a * (0 + fu) * (1 - fv);
                        r1a += a * (1 - fu) * (0 + fv);
                        r1b += a * (0 + fu) * (0 + fv);
                    }
                }
                if (row != NO_ROW)
                {
                    emit(row, r0a, r0b);
                    emit(row + 1, r1a, r1b);
                }
            }
            __syncthreads();
            for (int i = tid; i < FWD_P * FWD_P; i += FWD_X * FWD_Y)
            {
                const float a = patch[i];
                if (a != 0.f)
                {
                    patch[i] = 0.f;
                    atomicAdd(&g[(u0 + i % FWD_P) + (size_t)t * n + (size_t)(v0 + i / FWD_P) * zs], a);
                }
            }
            __syncthreads();
        }
    }
}

// Accumulates into acc the PAR_G slices of src (stride n*n) whose voxels tap
// detector column ub at angle (c, s): the column taps of backprojection_par_ker
// read the other way round. src is the volume when |c| >= |s| and its x<->y
// transpose otherwise, so the voxels a ray meets lie along the contiguous index
// i and consecutive ub read consecutive addresses. MASKED skips the slices not
// set in ok.
template <bool MASKED>
static __device__ __forceinline__ void par_gather(float *acc, const float *src, unsigned ok, float c, float s, float axis, int ub, int n)
{
    const int h = n / 2;
    const size_t nn = (size_t)n * n;
    const bool xdom = fabsf(c) >= fabsf(s);
    const float a = xdom ? c : s, b = xdom ? s : c; // u = a*(i-h) + b*(j-h) + axis
    // ub is tapped by voxels with ⌊u⌋ == ub or ⌊u⌋ + 1 == ub, i.e. u in
    // (ub - 1, ub + 1). Column ub exists, so the tap that lands on it is never
    // zero padding.
    const float lo = ub - 1, hi = ub + 1;
    for (int j = 0; j < n; j++)
    {
        float base = b * (j - h) + axis;
        float ia = (lo - base) / a + h, ib = (hi - base) / a + h;
        // One voxel of margin each side absorbs the rounding of ia/ib; proj_tap
        // then decides each candidate exactly as the back-projector does.
        int i0 = max(0, (int)floorf(fminf(ia, ib)) - 1);
        int i1 = min(n - 1, (int)ceilf(fmaxf(ia, ib)) + 1);
        for (int i = i0; i <= i1; i++)
        {
            Tap cu = proj_tap(xdom ? proj_u(c, s, i, j, h, axis) : proj_u(c, s, j, i, h, axis), n);
            float w;
            if (cu.i == ub)
                w = 1 - cu.f;
            else if (cu.i + 1 == ub)
                w = cu.f;
            else
                continue;
            const float *p = src + i + (size_t)j * n;
#pragma unroll
            for (int z = 0; z < PAR_G; z++)
                if (!MASKED || ((ok >> z) & 1u))
                    acc[z] += p[z * nn] * w;
        }
    }
}

// Parallel-beam forward projection: the transpose of backprojection_par_ker as a
// gather, so no atomics. One thread owns detector column ub at angle t for PAR_G
// slices of the zn-slice chunk starting at global slice z0 (f, fT point at that
// chunk); slice tz lands on detector row tz + off. Every (ub, t, row) has one
// owner, so g is accumulated with a plain +=.
static __global__ void forwardprojection_par_ker(float *g, const float *f, const float *fT, const float *theta,
                                                 float axis, int z0, int zn, int off, int n, int nz, int nproj, int nub)
{
    int ub = (blockIdx.x % nub) * blockDim.x + threadIdx.x;
    int t = blockIdx.x / nub;
    int zl = blockIdx.z * PAR_G;
    if (ub >= n)
        return;
    unsigned ok = 0;
    for (int z = 0; z < PAR_G && zl + z < zn; z++)
        if (tap_in(z0 + zl + z + off, nz))
            ok |= 1u << z;
    if (ok == 0)
        return;
    float c = __cosf(theta[t]), s = __sinf(theta[t]);
    const float *src = (fabsf(c) >= fabsf(s) ? f : fT) + (size_t)zl * n * n;
    float acc[PAR_G];
#pragma unroll
    for (int z = 0; z < PAR_G; z++)
        acc[z] = 0.f;
    if (ok == (1u << PAR_G) - 1)
        par_gather<false>(acc, src, ok, c, s, axis, ub, n);
    else
        par_gather<true>(acc, src, ok, c, s, axis, ub, n);
    float *gt = g + ub + (size_t)t * n;
#pragma unroll
    for (int z = 0; z < PAR_G; z++)
        if ((ok >> z) & 1u)
            gt[(size_t)(z0 + zl + z + off) * n * nproj] += acc[z];
}

// dst[z][x][y] = src[z][y][x] for each n x n slice.
static __global__ void transpose_xy_ker(float *dst, const float *src, int n)
{
    __shared__ float tile[32][33];
    size_t o = (size_t)blockIdx.z * n * n;
    int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8)
        if (x < n && y + k < n)
            tile[threadIdx.y + k][threadIdx.x] = src[o + x + (size_t)(y + k) * n];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8)
        if (x < n && y + k < n)
            dst[o + x + (size_t)(y + k) * n] = tile[threadIdx.x][threadIdx.y + k];
}

// Scratch budget of the parallel-beam gather: the transposed copy of one chunk
// of slices, the chunk a multiple of PAR_G.
static const size_t FWD_SCRATCH_BUDGET = (size_t)256 << 20;

// Bytes of scratch tomoxide_forwardproject uses for this geometry when given at
// least that much; 0 for a tilted axis (the scatter needs none).
extern "C" size_t tomoxide_forwardproject_scratch_bytes(float phi, int n, int ncz)
{
    if (!is_parallel_tilt(phi) || n <= 0 || ncz <= 0)
        return 0;
    size_t slice = (size_t)n * n * sizeof(float);
    size_t cap = FWD_SCRATCH_BUDGET / slice / PAR_G * PAR_G;
    if (cap < PAR_G)
        cap = PAR_G;
    if (cap > (size_t)ncz)
        cap = (size_t)ncz;
    return cap * slice;
}

// C-ABI host wrapper. `g` (output sinogram [nz][nproj][n]) is accumulated
// into, so callers pre-zero it.
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
// The parallel beam runs the gather over chunks of as many slices as `scratch`
// (`scratch_bytes`, stream-ordered on `stream`) holds; a tilted axis, or
// scratch smaller than one slice, runs the atomic scatter.
extern "C" int tomoxide_forwardproject(void *g, const void *f, const float *theta, float phi,
                                        float axis, int sz, int ncz, int nz, int n, int rh,
                                        int nproj, void *scratch, size_t scratch_bytes,
                                        void *stream)
{
    cudaStream_t st = (cudaStream_t)stream;
    size_t nn = (size_t)n * n;
    size_t zc = scratch_bytes / (nn * sizeof(float));
    if (is_parallel_tilt(phi) && zc > 0)
    {
        int off = sz - rh / 2 + nz / 2;
        int nub = (n + 127) / 128;
        float *fT = (float *)scratch;
        for (int z0 = 0; z0 < ncz; z0 += (int)zc)
        {
            int zn = ncz - z0 < (int)zc ? ncz - z0 : (int)zc;
            const float *fz = (const float *)f + (size_t)z0 * nn;
            transpose_xy_ker<<<dim3((n + 31) / 32, (n + 31) / 32, zn), dim3(32, 8), 0, st>>>(fT, fz, n);
            forwardprojection_par_ker<<<dim3((unsigned)nub * nproj, 1, (zn + PAR_G - 1) / PAR_G), 128, 0, st>>>(
                (float *)g, fz, fT, theta, axis, z0, zn, off, n, nz, nproj, nub);
        }
        return (int)cudaGetLastError();
    }
    dim3 grid((n + FWD_X - 1) / FWD_X, (n + FWD_Y - 1) / FWD_Y, (ncz + FWD_G - 1) / FWD_G);
    forwardprojection_ker<<<grid, dim3(FWD_X, FWD_Y), 0, st>>>(
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
        const int h = n / 2, zb = sz - rh / 2; // zz of slice tz is zb + tz
        const float cphi = __cosf(phi), sphi = __sinf(phi);
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
                // forwardprojection_ker's taps: the cells it scatters into.
                const float c = sc[a], s = ss[a];
                const Tap cu = proj_tap(proj_u(c, s, tx, ty, h, axis), n);
                const Tap rv = proj_tap(lam_v(lam_vxy(c, s, cphi, tx, ty, h), sphi, zb + tz, nz), nz);
                if (tap_any(cu) & tap_any(rv))
                {
                    const ptrdiff_t o = ((ptrdiff_t)(t0 + a) * nz + rv.i) * n + cu.i;
                    ax[a] += val * lam_bilerp(hx + o, n, cu, rv);
                    ay[a] += val * lam_bilerp(hy + o, n, cu, rv);
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
