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
// the `n/2` centre, the rotation matrix `R`, the `(int)(·-1e-5)` tap, and the
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
// `f` the input volume [nz][n][n].
static __global__ void forwardprojection_ker(float *g, const float *f, const float *theta,
                                             float phi, int n, int nz, int nproj)
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
// Unweighted (gain 1): the geometry matches backprojection_ker with gain 1 (the
// iterative back-projection call), so {forward, back} are a true adjoint pair
// and the forward is the plain line-integral Radon transform — matching the CPU
// and wgpu forward projectors, so the iterative suite is scale-unified across
// backends at the physical μ. The launch geometry mirrors
// cfunc_linerec::backprojection.
extern "C" void tomoxide_forwardproject(void *g, const void *f, const float *theta, float phi,
                                        int nz, int n, int nproj, void *stream)
{
    dim3 dimBlock(32, 32, 1);
    dim3 grid((unsigned)ceil(n / 32.0), (unsigned)ceil(n / 32.0), (unsigned)nz);
    size_t shmem = 2 * (size_t)nproj * sizeof(float); // cos/sin(theta) cache
    forwardprojection_ker<<<grid, dimBlock, shmem, (cudaStream_t)stream>>>(
        (float *)g, (const float *)f, theta, phi, n, nz, nproj);
}
