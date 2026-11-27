#include "defs.cuh"
#include "projgeom.cuh"

#ifdef HALF
// The f16 back-projection path uses a hardware-interpolated layered texture
// (below). The f32 path keeps the direct-gather kernel (in the #else branch):
// f32 reconstruction runs across multiple GPUs, where each worker's synchronous
// cudaMalloc3DArray for the texture serializes on the CUDA allocator lock and
// erases the multi-GPU overlap — measured a net regression. The single-GPU f16
// path has no such contention, so the texture's collapsed gather is a net win.

// Transpose the filtered sinogram into the layered texture array.
// Source `data` is laid out [z][proj][col] (index col + t*n + z*n*nproj, the
// same order the gather kernel indexed). The layered array wants layer=proj
// with a 2D plane of (x=col, y=z), so we scatter each element into its layer.
// surf2DLayeredwrite takes the x offset in BYTES.
void __global__ fill_tex_ker(cudaSurfaceObject_t surf, real *data, int ncz, int n, int nproj)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.y * blockIdx.y + threadIdx.y;
    int t = blockDim.z * blockIdx.z + threadIdx.z;
    if (col >= n || z >= ncz || t >= nproj)
        return;
    real val = data[col + t * n + z * n * nproj];
    SURF_WRITE_L(val, surf, col * (int)sizeof(real), z, t);
}

// Texture back-projection: identical geometry to the old gather kernel, but
// the 4-tap load + float bilinear interpolation collapses into one hardware
// tex2DLayered linear fetch (col = x, z = y, projection = layer). The texture
// returns the filtered value in float for both dtypes (a 16F array is read and
// filtered in float), so we accumulate in float at no extra conversion cost.
// `rh` is the FULL reconstruction height in slices (== nz for parallel beam,
// `lamino_recon_height` for laminography). The volume z is centred on rh/2 —
// the volume's own midpoint — while the detector row stays centred on nz/2.
// Centring z on nz/2 instead (the pre-fix convention) shifted every lamino
// volume by (rh-nz)/2 slices against the Fourier (USFFT) reconstruction,
// which has always centred on rh/2.
void __global__ backprojection_tex_ker(real *f, cudaTextureObject_t tex, float *theta, float phi, float axis, float c, int sz, int ncz, int n, int nz, int rh, int nproj)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    // cos/sin(theta[t]) is identical for every (tx,ty,tz); compute it once per
    // block into shared memory instead of nproj transcendentals per thread.
    // Done before the bounds guard so all threads reach __syncthreads (threads
    // that would early-return must not skip the barrier).
    // ssc holds 2*nproj floats = 8*nproj bytes of dynamic shared memory; the
    // launch passes that size, bounded by the 48 KB/block default (nproj up to
    // 6144, far beyond any real scan).
    extern __shared__ float ssc[]; // [0,nproj)=cos, [nproj,2*nproj)=sin
    {
        int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
        int nthreads = blockDim.x * blockDim.y * blockDim.z;
        for (int t = tid; t < nproj; t += nthreads)
            theta_cs(theta[t], &ssc[t], &ssc[nproj + t]);
    }
    __syncthreads();
    if (tx >= n || ty >= n || tz >= ncz)
        return;

    float f0 = 0;
    float cphi = __cosf(phi);
    float sphi = __sinf(phi);
    float R[6] = {};

    for (int t = 0; t<nproj; t++)
    {
        float ctheta = ssc[t];
        float stheta = ssc[nproj + t];
        R[0] =  ctheta;       R[1] =  stheta;        R[2] = 0;
        R[3] =  stheta*cphi;  R[4] = -ctheta*cphi;   R[5] = sphi;
        float u = R[0]*(tx-n/2)+R[1]*(ty-n/2)+axis;
        float v = R[3]*(tx-n/2)+R[4]*(ty-n/2)+R[5]*(tz+sz-rh/2) + nz/2;

        // Texel i is sampled at coordinate i+0.5, so u+0.5 reproduces
        // (1-frac(u))*T[⌊u⌋] + frac(u)*T[⌊u⌋+1] (and likewise for v), and the
        // border addressing reads 0 off the detector: proj_tap's rule. The test
        // only skips the fetches of pairs wholly off the detector.
        if (tap_any(proj_tap(u, n)) & tap_any(proj_tap(v, nz)))
            f0 += tex2DLayered<float>(tex, u + 0.5f, v + 0.5f, t);
    }
    f[tx + ty * n + tz * n * n] += static_cast<real>(f0*c);
}

#else // !HALF: f32 direct-gather back-projection (see note above)

// Slices one thread of backprojection_ker covers: the column tap and the (x, y)
// part of the row depend on (x, y, θ) only, so they are computed once per group.
#define LAM_G 8

// Tilted-axis back-projection, bilinear in (u, v). `rh` = full reconstruction
// height; volume z centres on rh/2, detector row on nz/2 (see the note on
// `backprojection_tex_ker`). `axis` = detector column voxel column n/2 reads, as
// in forwardprojection_ker. Each thread accumulates LAM_G slices of one (x, y).
// cos/sin(theta) is cached per block in 2*nproj floats of dynamic shared memory
// (the 48 KB default holds nproj up to 6144), filled before the bounds guard so
// every thread reaches __syncthreads.
void __global__ backprojection_ker(float *f, const float *data, const float *theta, float phi, float axis, float c, int sz, int ncz, int n, int nz, int rh, int nproj)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int z0 = blockIdx.z * LAM_G;
    extern __shared__ float ssc[]; // [0,nproj)=cos, [nproj,2*nproj)=sin
    for (int t = threadIdx.x + blockDim.x * threadIdx.y; t < nproj; t += blockDim.x * blockDim.y)
        theta_cs(theta[t], &ssc[t], &ssc[nproj + t]);
    __syncthreads();
    if (tx >= n || ty >= n)
        return;

    const int h = n / 2;
    const int cnt = min(LAM_G, ncz - z0);
    const int zb = z0 + sz - rh / 2; // zz of the group's first slice
    const float cphi = __cosf(phi);
    const float sphi = __sinf(phi);
    const size_t zs = (size_t)n * nproj;
    float acc[LAM_G];
#pragma unroll
    for (int z = 0; z < LAM_G; z++)
        acc[z] = 0.f;
    for (int t = 0; t < nproj; t++)
    {
        float ct = ssc[t], st = ssc[nproj + t];
        Tap cu = proj_tap(proj_u(ct, st, tx, ty, h, axis), n);
        if (!tap_any(cu))
            continue;
        float vxy = lam_vxy(ct, st, cphi, tx, ty, h);
        const float *p = data + cu.i + (size_t)t * n;
#pragma unroll
        for (int z = 0; z < LAM_G; z++)
        {
            // Branch-free (a per-slice branch cost 3%): lam_bilerp loads only
            // the taps that exist, so a pair off the detector adds 0, and the
            // slices z >= cnt are summed but never stored.
            Tap rv = proj_tap(lam_v(vxy, sphi, zb + z, nz), nz);
            acc[z] += lam_bilerp(p + (ptrdiff_t)rv.i * (ptrdiff_t)zs, zs, cu, rv);
        }
    }
#pragma unroll
    for (int z = 0; z < LAM_G; z++)
        if (z < cnt)
            f[tx + (size_t)ty * n + (size_t)(z0 + z) * n * n] += acc[z] * c;
}

// Accumulates the PAR_G slices whose detector rows start at d0 (stride zs);
// MASKED skips the slices not set in ok, so a group lying wholly inside the
// detector runs the branch-free instantiation.
template <bool MASKED>
static __device__ __forceinline__ void par_accumulate(float *acc, const float *ssc, const float *d0, unsigned ok, int tx, int ty, float axis, int n, int nproj)
{
    const int h = n / 2;
    const size_t zs = (size_t)n * nproj;
    for (int t = 0; t < nproj; t++)
    {
        Tap cu = proj_tap(proj_u(ssc[t], ssc[nproj + t], tx, ty, h, axis), n);
        if (!tap_any(cu))
            continue;
        // With one tap off the detector both loads read the existing column at
        // half its weight: unpredicated loads (~5% faster than reading the
        // absent tap as 0), no 0 · sample term (NaN for an Inf sample), and
        // halving is exact, so the pair sums to the one-tap product.
        const int ca = cu.lo ? cu.i : cu.i + 1, cb = cu.hi ? cu.i + 1 : cu.i;
        const float w = cu.lo ? 1 - cu.f : cu.f;
        const float wa = cu.lo & cu.hi ? w : 0.5f * w, wb = cu.lo & cu.hi ? cu.f : 0.5f * w;
        const float *p = d0 + (size_t)t * n;
#pragma unroll
        for (int z = 0; z < PAR_G; z++)
            if (!MASKED || ((ok >> z) & 1u))
                acc[z] += p[z * zs + ca] * wa + p[z * zs + cb] * wb;
    }
}

// Parallel-beam back-projection (is_parallel_tilt(phi)). The tilted kernel's
// v-interpolation weighs row r = tz + sz - rh/2 + nz/2 by exactly 1 and its
// neighbour by 0 at this tilt, so this kernel reads row r alone and each thread
// reuses one column tap for PAR_G slices: backprojection_ker's operator up to
// summation order, at a quarter of its loads.
void __global__ backprojection_par_ker(float *f, const float *data, const float *theta, float axis, float c, int sz, int ncz, int n, int nz, int rh, int nproj)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int z0 = blockIdx.z * PAR_G;
    extern __shared__ float ssc[]; // [0,nproj)=cos, [nproj,2*nproj)=sin
    for (int t = threadIdx.x + blockDim.x * threadIdx.y; t < nproj; t += blockDim.x * blockDim.y)
        theta_cs(theta[t], &ssc[t], &ssc[nproj + t]);
    __syncthreads();
    if (tx >= n || ty >= n)
        return;

    const int off = sz - rh / 2 + nz / 2; // detector row of slice tz is tz + off
    unsigned ok = 0;
    for (int z = 0; z < PAR_G && z0 + z < ncz; z++)
        if (tap_in(z0 + z + off, nz))
            ok |= 1u << z;
    float acc[PAR_G];
#pragma unroll
    for (int z = 0; z < PAR_G; z++)
        acc[z] = 0.f;
    // Row z0+off may lie outside the stack; only rows with their ok bit set are read.
    const float *d0 = data + (ptrdiff_t)(z0 + off) * n * nproj;
    if (ok == (1u << PAR_G) - 1)
        par_accumulate<false>(acc, ssc, d0, ok, tx, ty, axis, n, nproj);
    else
        par_accumulate<true>(acc, ssc, d0, ok, tx, ty, axis, n, nproj);
#pragma unroll
    for (int z = 0; z < PAR_G; z++)
        if (z0 + z < ncz)
            f[tx + (size_t)ty * n + (size_t)(z0 + z) * n * n] += acc[z] * c;
}

#endif // HALF

// `rh` = full reconstruction height; the probed slice `sz` is a volume z and
// centres on rh/2 (see the note on `backprojection_tex_ker`).
void __global__ backprojection_try_ker(real *f, real *data, float *theta, float phi, float c, int sz, float* sh, int ncz, int n, int nz, int rh, int nproj)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    // cache cos/sin(theta[t]) once per block (see backprojection_ker)
    // ssc holds 2*nproj floats = 8*nproj bytes of dynamic shared memory; the
    // launch passes that size, bounded by the 48 KB/block default (nproj up to
    // 6144, far beyond any real scan).
    extern __shared__ float ssc[]; // [0,nproj)=cos, [nproj,2*nproj)=sin
    {
        int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
        int nthreads = blockDim.x * blockDim.y * blockDim.z;
        for (int t = tid; t < nproj; t += nthreads)
            theta_cs(theta[t], &ssc[t], &ssc[nproj + t]);
    }
    __syncthreads();
    if (tx >= n || ty >= n || tz >= ncz)
        return;
    const int h = n / 2;
    const size_t zs = (size_t)n * nproj;
    float f0 = 0;
    float cphi = __cosf(phi);
    float sphi = __sinf(phi);

    for (int t = 0; t<nproj; t++)
    {
        float ctheta = ssc[t];
        float stheta = ssc[nproj + t];
        Tap cu = proj_tap(__fsub_rn(proj_u(ctheta, stheta, tx, ty, h, (float)h), sh[tz]), n);
        Tap rv = proj_tap(lam_v(lam_vxy(ctheta, stheta, cphi, tx, ty, h), sphi, sz - rh / 2, nz), nz);
        if (tap_any(cu) & tap_any(rv))
            f0 += lam_bilerp(data + cu.i + (size_t)t * n + (ptrdiff_t)rv.i * (ptrdiff_t)zs, zs, cu, rv);
    }
    f[tx + ty * n + tz * n * n] += static_cast<real>(f0*c);
}
// `phi` and `rh` are per-tz: each tz slot probes one tilt candidate, and the
// full reconstruction height depends on the tilt (`lamino_recon_height`), so
// each candidate's slice `sz` centres on its own rh[tz]/2.
void __global__ backprojection_try_lamino_ker(real *f, real *data, float *theta, float* phi, float c, int sz, const int* rh, int ncz, int n, int nz, int nproj)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    // cache cos/sin(theta[t]) once per block (see backprojection_ker). phi is
    // per-tz here, so its cos/sin stays per-thread.
    // ssc holds 2*nproj floats = 8*nproj bytes of dynamic shared memory; the
    // launch passes that size, bounded by the 48 KB/block default (nproj up to
    // 6144, far beyond any real scan).
    extern __shared__ float ssc[]; // [0,nproj)=cos, [nproj,2*nproj)=sin
    {
        int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
        int nthreads = blockDim.x * blockDim.y * blockDim.z;
        for (int t = tid; t < nproj; t += nthreads)
            theta_cs(theta[t], &ssc[t], &ssc[nproj + t]);
    }
    __syncthreads();
    if (tx >= n || ty >= n || tz >= ncz)
        return;
    const int h = n / 2;
    const size_t zs = (size_t)n * nproj;
    float f0 = 0;
    float cphi = __cosf(phi[tz]);
    float sphi = __sinf(phi[tz]);

    for (int t = 0; t<nproj; t++)
    {
        float ctheta = ssc[t];
        float stheta = ssc[nproj + t];
        Tap cu = proj_tap(proj_u(ctheta, stheta, tx, ty, h, (float)h), n);
        Tap rv = proj_tap(lam_v(lam_vxy(ctheta, stheta, cphi, tx, ty, h), sphi, sz - rh[tz] / 2, nz), nz);
        if (tap_any(cu) & tap_any(rv))
            f0 += lam_bilerp(data + cu.i + (size_t)t * n + (ptrdiff_t)rv.i * (ptrdiff_t)zs, zs, cu, rv);
    }
    // No y-flip and the caller's dθ gain, like every other kernel here: tomocupy
    // writes `(n-ty-1)` and scales by 4/nproj, and this file's ports were moved
    // off both by the cross-backend convention unification (see
    // `backprojection_ker`, `docs/ARCHITECTURE.md` §4.1). This kernel alone kept
    // the upstream convention until 32e8ca9. Being uncalled is *not* what spared
    // it — `backprojection_try_ker` is equally uncalled and was migrated. What
    // singles this one out is that it is the file's only laminography-*only*
    // kernel, and §4.1 then claimed laminography was excluded from the
    // unification: a rule that licensed skipping it, and that was already false
    // (the lamino path back-projects through `backprojection_ker`, which *was*
    // migrated — so the two lamino kernels simply disagreed). A tilt probe whose
    // slice is flipped and 4/π off against every volume tomoxide reconstructs
    // would be comparing against an image the user never sees.
    f[tx + ty * n + tz * n * n] += static_cast<real>(f0*c);
}
