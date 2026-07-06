#include "cfunc_linerec.cuh"
#include "kernels_linerec.cuh"

cfunc_linerec::cfunc_linerec(size_t nproj, size_t nz, size_t n,size_t ncproj, size_t ncz)
    : nproj(nproj), nz(nz), n(n), ncproj(ncproj), ncz(ncz) {    
  }


// destructor, memory deallocation
cfunc_linerec::~cfunc_linerec() { free(); }

void cfunc_linerec::free() {
  if (!is_free) {
#ifdef HALF
    if (tex_init) {
      cudaDestroyTextureObject(tex_obj);
      cudaDestroySurfaceObject(surf_obj);
      cudaFreeArray(tex_array);
      tex_init = false;
    }
#endif
    is_free = true;
  }
}

#ifdef HALF
// Lazily allocate the layered cudaArray that backs the back-projection texture
// (width = n columns, height = ncz slices, layers = ncproj projections) plus
// the texture (linear-filtered read) and surface (transpose write) objects.
// f16 only — see the note in kernels_linerec.cuh on why f32 keeps the gather.
void cfunc_linerec::ensure_texture() {
    if (tex_init) return;
    cudaChannelFormatDesc ch = CUDA_CREATE_CHANNEL_DESC();
    cudaExtent ext = make_cudaExtent(n, ncz, ncproj);
    cudaMalloc3DArray(&tex_array, &ch, ext, cudaArrayLayered | cudaArraySurfaceLoadStore);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = tex_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, nullptr);
    cudaCreateSurfaceObject(&surf_obj, &resDesc);
    tex_init = true;
}
#endif

void cfunc_linerec::backprojection(size_t f_, size_t g_, size_t theta_, float phi, float gain, int sz, size_t stream_) {
    real* g = (real *)g_;
    real* f = (real *)f_;
    float* theta = (float *)theta_;
    cudaStream_t stream = (cudaStream_t)stream_;
    dim3 dimBlock(32,32,1);
    dim3 GS3d0 = dim3(ceil(n / 32.0), ceil(n / 32.0), ncz);
    size_t shmem = 2 * ncproj * sizeof(float); // cos/sin(theta) cache
#ifdef HALF
    ensure_texture();
    // Upload the filtered sinogram into the layered texture array (transpose
    // [z][proj][col] -> layer=proj plane=(col,z)). Separate launch from the
    // back-projection so the texture cache sees the surface writes.
    dim3 fillBlock(32, 8, 1);
    dim3 fillGrid(ceil(n / 32.0), ceil(ncz / 8.0), ncproj);
    fill_tex_ker<<<fillGrid, fillBlock, 0, stream>>>(surf_obj, g, ncz, n, ncproj);
    // Back-projector gain is the CALLER's angular quadrature weight: the analytic
    // FBP paths pass π/nproj (the dθ weight, was tomocupy's 4/nproj — unified to
    // the CPU/tomopy scale), the iterative solvers pass 1 so this kernel is the
    // pure adjoint of forwardproj.cu's unweighted scatter and the pair {A, Aᵀ}
    // converges to the physical μ.
    backprojection_tex_ker <<<GS3d0, dimBlock, shmem, stream>>> (f, tex_obj, theta, phi, gain, sz, ncz, n, nz, ncproj);
#else
    backprojection_ker <<<GS3d0, dimBlock, shmem, stream>>> (f, g, theta, phi, gain, sz, ncz, n, nz, ncproj);
#endif
}

void cfunc_linerec::backprojection_try(size_t f_, size_t g_, size_t theta_, size_t sh_, float phi, int sz,  size_t stream_) {
    real* g = (real *)g_;    
    real* f = (real *)f_;
    float* sh = (float *)sh_;

    float* theta = (float *)theta_;
    cudaStream_t stream = (cudaStream_t)stream_;        
    // set thread block, grid sizes will be computed before cuda kernel execution
    dim3 dimBlock(32,32,1);    
    dim3 GS3d0;  
    GS3d0 = dim3(ceil(n / 32.0), ceil(n / 32.0), ncz);
    size_t shmem = 2 * ncproj * sizeof(float); // cos/sin(theta) cache
    backprojection_try_ker<<<GS3d0, dimBlock, shmem, stream>>> (f, g, theta, phi, 3.14159265358979f/nproj, sz, sh, ncz, n, nz, ncproj);
}                                            

void cfunc_linerec::backprojection_try_lamino(size_t f_, size_t g_, size_t theta_, size_t phi_, int sz,  size_t stream_) {
    real* g = (real *)g_;    
    real* f = (real *)f_;
    float* phi = (float *)phi_;
    float* theta = (float *)theta_;
    cudaStream_t stream = (cudaStream_t)stream_;        
    // set thread block, grid sizes will be computed before cuda kernel execution
    dim3 dimBlock(32,32,1);    
    dim3 GS3d0;  
    GS3d0 = dim3(ceil(n / 32.0), ceil(n / 32.0), ncz);
    size_t shmem = 2 * ncproj * sizeof(float); // cos/sin(theta) cache
    backprojection_try_lamino_ker<<<GS3d0, dimBlock, shmem, stream>>> (f, g, theta, phi, 4.0f/nproj, sz, ncz, n, nz, ncproj);
}                                            