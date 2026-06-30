// Parallel-beam forward projection (Radon transform) — the exact discrete
// transpose of `kernels_linerec.cuh::backprojection_ker`.
//
// The voxel-driven back-projector gathers, for output voxel
//   j = tx + (n-ty-1)*n + tz*n*n,
//   f[j] += c * Σ_t  bilinear(g; u(j,t), v(j,t))   with c = 4/nproj,
// i.e. as a matrix `f = (c·W)·g` where `W` is the bilinear gather operator.
// Its adjoint (the forward projector the iterative solvers need as `A`, with the
// back-projector serving as `Aᵀ`) is `g = (c·Wᵀ)·f`: each voxel scatters its
// value into the *same* (u,v) taps with the *same* bilinear weights and the same
// `c`. Keeping the geometry byte-identical to `backprojection_ker` — the y-flip
// `(n-ty-1)`, the `n/2` centre, the rotation matrix `R`, the `(int)(·-1e-5)` tap,
// and the in-bounds guard — makes {project, backproject} a true {A, Aᵀ} pair, so
// SIRT/MLEM/OSEM/… converge. The shared guard `vr < nz-1` drops slice 0 for the
// parallel beam (`v = tz` ⇒ `vr = tz-1`), the documented ≥2-slice rule, exactly
// as the back-projector does.
//
// f32 only (the iterative suite is f32); built into libtomoxide_cuda_kernels.a
// by build.rs's `.cu` glob.

#include <cuda_runtime.h>
#include <math.h>

// Scatter transpose of backprojection_ker (see file header). `g` is the output
// sinogram [nz][nproj][n] (must be pre-zeroed; the kernel only atomic-adds),
// `f` the input volume [nz][n][n].
static __global__ void forwardprojection_ker(float *g, const float *f, const float *theta,
                                             float phi, float c, int n, int nz, int nproj)
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
    if (tx >= n || ty >= n || tz >= nz)
        return;

    // Same y-flip and centre as the back-projector's accumulation index.
    float val = c * f[tx + (n - ty - 1) * n + tz * n * n];
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
        float u = R[0] * (tx - n / 2) + R[1] * (ty - n / 2) + n / 2;
        float v = R[3] * (tx - n / 2) + R[4] * (ty - n / 2) + R[5] * (tz - nz / 2) + nz / 2;

        int ur = (int)(u - 1e-5f);
        int vr = (int)(v - 1e-5f);

        // Same in-bounds guard as the gather; out-of-bounds taps contribute 0.
        if ((ur >= 0) & (ur < n - 1) & (vr >= 0) & (vr < nz - 1))
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
// `c = 4/nproj` matches backprojection_ker exactly so {forward, back} are a true
// adjoint pair. The launch geometry mirrors cfunc_linerec::backprojection.
extern "C" void tomoxide_forwardproject(void *g, const void *f, const float *theta, float phi,
                                        int nz, int n, int nproj, void *stream)
{
    dim3 dimBlock(32, 32, 1);
    dim3 grid((unsigned)ceil(n / 32.0), (unsigned)ceil(n / 32.0), (unsigned)nz);
    size_t shmem = 2 * (size_t)nproj * sizeof(float); // cos/sin(theta) cache
    float c = 4.0f / (float)nproj;
    forwardprojection_ker<<<grid, dimBlock, shmem, (cudaStream_t)stream>>>(
        (float *)g, (const float *)f, theta, phi, c, n, nz, nproj);
}
