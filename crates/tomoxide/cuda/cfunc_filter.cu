#include "cfunc_filter.cuh"
#include "kernels_filter.cuh"
#include<stdio.h>
cfunc_filter::cfunc_filter(size_t nproj, size_t nz, size_t n)
    : nproj(nproj), nz(nz), n(n) {

    //fft filter R<->C
    // cuFFT auto-allocates the plan work area at cufftXtMakePlanMany time, so
    // checking every cuFFT/cudaMalloc return here is what turns an out-of-memory
    // chunk into a clean failure: any failure leaves valid_ == false and the
    // factory returns null, instead of leaving an unallocated work area that
    // cufftXtExec later dereferences (SIGSEGV inside libcufft).
    if (cufftCreate(&plan_filter_fwd) != CUFFT_SUCCESS) return;
    fwd_ok = true;
    if (cufftCreate(&plan_filter_inv) != CUFFT_SUCCESS) return;
    inv_ok = true;

    long long ffts[1] = {};
    long long inembed[1] = {};
    long long onembed[1] = {};
    long long idist = 0;
    long long odist = 0;
    size_t workSize = 0;

    ffts[0] = n;
	  idist = n;odist = n/2+1;
    inembed[0] = n;onembed[0] = n/2+1;


    if (cudaMalloc((void **)&ge,
            (n/2+1) * nproj * nz * sizeof(real2)) != cudaSuccess) {
        ge = nullptr;
        return;
    }
    if (cufftXtMakePlanMany(plan_filter_fwd,
        1, ffts,
        inembed, 1, idist, CUDA_R,
        onembed, 1, odist, CUDA_C,
        nproj*nz, &workSize, CUDA_C) != CUFFT_SUCCESS) return;
    if (cufftXtMakePlanMany(plan_filter_inv,
        1, ffts,
        onembed, 1, odist, CUDA_C,
        inembed, 1, idist, CUDA_R,
        nproj*nz, &workSize, CUDA_C) != CUFFT_SUCCESS) return;
    valid_ = true;
    }


// destructor, memory deallocation
cfunc_filter::~cfunc_filter() { free(); }

void cfunc_filter::free() {
  if (!is_free) {
    if (fwd_ok) cufftDestroy(plan_filter_fwd);
    if (inv_ok) cufftDestroy(plan_filter_inv);
    if (ge) cudaFree(ge);
    is_free = true;
  }
}

void cfunc_filter::filter(size_t g_, size_t w_, size_t stream_) {
    real* g = (real *)g_;    
    real2* w = (real2 *)w_;
    cudaStream_t stream = (cudaStream_t)stream_;    
    cufftSetStream(plan_filter_fwd, stream);
    cufftSetStream(plan_filter_inv, stream);    
    dim3 dimBlock(32,32,1);
    dim3 GS3d2 = dim3(ceil((n/2+1)/32.0), ceil(nproj / 32.0), nz);
    cufftXtExec(plan_filter_fwd, g, ge, CUFFT_FORWARD);
    mulw <<<GS3d2, dimBlock, 0, stream>>> (ge, w, n/2+1, nproj, nz);
    cufftXtExec(plan_filter_inv, ge, g, CUFFT_INVERSE);
    // 1/n normalization is folded into wfilter in fbp_filter.py::calc_filter()
}
