#include "cfunc_fourierrec.cuh"
#include "kernels_fourierrec.cuh"

cfunc_fourierrec::cfunc_fourierrec(size_t nproj, size_t nz, size_t n, size_t theta_)
    : nproj(nproj), nz(nz), n(n) {
    float eps = 1e-3;
    mu = -log(eps) / (2 * n * n);
    m = ceil(2 * n * 1 / PI * sqrt(-mu * log(eps) + (mu * n) * (mu * n) / 4));
    theta = (float*)theta_;
    // Check every cuFFT/cudaMalloc return: cuFFT auto-allocates the plan work
    // area at cufftXtMakePlanMany time, so any out-of-memory failure here leaves
    // valid_ == false and the factory returns null, instead of leaving an
    // unallocated work area that cufftXtExec later dereferences (SIGSEGV).
    if (cudaMalloc((void **)&fde,
            (2 * n + 2 * m) * (2 * n + 2 * m) * nz * sizeof(real2)) != cudaSuccess) {
        fde = nullptr;
        return;
    }
    if (cudaMalloc((void **)&x, n * nproj * sizeof(float)) != cudaSuccess) {
        x = nullptr;
        return;
    }
    if (cudaMalloc((void **)&y, n * nproj * sizeof(float)) != cudaSuccess) {
        y = nullptr;
        return;
    }

    long long ffts[] = {2*n,2*n};
	  long long idist = (2 * n + 2 * m) * (2 * n + 2 * m);long long odist = (2 * n + 2 * m) * (2 * n + 2 * m);
    long long inembed[] = {2 * n + 2 * m, 2 * n + 2 * m};long long onembed[] = {2 * n + 2 * m, 2 * n + 2 * m};
    size_t workSize = 0;

    if (cufftCreate(&plan2d) != CUFFT_SUCCESS) return;
    plan2d_ok = true;
    if (cufftXtMakePlanMany(plan2d,
        2, ffts,
        inembed, 1, idist, CUDA_C,
        onembed, 1, odist, CUDA_C,
        nz, &workSize, CUDA_C) != CUFFT_SUCCESS) return;
    // fft 1d
    if (cufftCreate(&plan1d) != CUFFT_SUCCESS) return;
    plan1d_ok = true;
    ffts[0] = n;
    idist = n;
    odist = n;
    inembed[0] = n;
    onembed[0] = n;
    if (cufftXtMakePlanMany(plan1d,
        1, ffts,
        inembed, 1, idist, CUDA_C,
        onembed, 1, odist, CUDA_C,
        nproj*nz, &workSize, CUDA_C) != CUFFT_SUCCESS) return;
    valid_ = true;

    // Pre-compute grid dimensions (constant for lifetime of object)
    dim3 dimBlock(32, 32, 1);
    GS2d0 = dim3(ceil(n / 32.0), ceil(nproj / 32.0));
    GS3d0 = dim3(ceil(n / 32.0), ceil(n / 32.0), nz);
    GS3d1 = dim3(ceil(2 * n / 32.0), ceil(2 * n / 32.0), nz);
    GS3d2 = dim3(ceil((2 * n + 2 * m) / 32.0), ceil((2 * n + 2 * m) / 32.0), nz);
    GS3d3 = dim3(ceil(n / 32.0), ceil(nproj / 32.0), nz);

    // Pre-compute x, y once (theta is fixed for lifetime of object)
    takexy <<<GS2d0, dimBlock>>> (x, y, theta, n, nproj);
  }


// destructor, memory deallocation
cfunc_fourierrec::~cfunc_fourierrec() { free(); }

void cfunc_fourierrec::free() {
  if (!is_free) {
    if (fde) cudaFree(fde);
    if (x) cudaFree(x);
    if (y) cudaFree(y);
    if (plan2d_ok) cufftDestroy(plan2d);
    if (plan1d_ok) cufftDestroy(plan1d);
    is_free = true;
  }
}

void cfunc_fourierrec::backprojection(size_t f_, size_t g_, size_t stream_) {
    real2* g = (real2 *)g_;    
    real2* f = (real2 *)f_;
    cudaStream_t stream = (cudaStream_t)stream_;    
    cufftSetStream(plan1d, stream);
    cufftSetStream(plan2d, stream);    

    dim3 dimBlock(32, 32, 1);

    cudaMemsetAsync(fde, 0, (2 * n + 2 * m) * (2 * n + 2 * m) * nz * sizeof(real2), stream);

    ifftshiftc <<<GS3d3, dimBlock, 0, stream>>> (g, n, nproj, nz);
    cufftXtExec(plan1d, g, g, CUFFT_FORWARD);
    ifftshiftc <<<GS3d3, dimBlock, 0, stream>>> (g, n, nproj, nz);    
    
    gather <<<GS3d3, dimBlock, 0, stream>>> (g, fde, x, y, m, mu, n, nproj, nz);    
    
    wrap <<<GS3d2, dimBlock, 0, stream>>> (fde, n, nz, m);
    
    fftshiftc <<<GS3d2, dimBlock, 0, stream>>> (fde, 2 * n + 2 * m, nz);
    cufftXtExec(plan2d, &fde[m + m * (2 * n + 2 * m)],
               &fde[m + m * (2 * n + 2 * m)], CUFFT_INVERSE);
    fftshiftc <<<GS3d2, dimBlock, 0, stream>>> (fde, 2 * n + 2 * m, nz);
    
    divphi <<<GS3d0, dimBlock, 0, stream>>> (fde, f, mu, n, nz, nproj, m);        
    circ <<<GS3d0, dimBlock, 0, stream>>> (f, 0, n, nz);  

}
