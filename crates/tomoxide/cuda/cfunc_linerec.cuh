#ifndef CFUNC_LINEREC_CUH
#define CFUNC_LINEREC_CUH

#include <cufft.h>
#include <cuda_fp16.h>
#include "defs.cuh"


class cfunc_linerec {
  bool is_free = false;

#ifdef HALF
  // Hardware-texture back-projection state (f16 build only; the f32 build keeps
  // the direct-gather kernel — see kernels_linerec.cuh for why). The filtered
  // sinogram is uploaded into a layered cudaArray so the inner loop can use one
  // tex2DLayered linear fetch instead of a 4-tap gather + float bilinear
  // interpolation. Allocated lazily on first backprojection, freed in the
  // destructor (which the FFI calls only after the stream is synced, so freeing
  // the array post-launch is safe).
  cudaArray_t tex_array = nullptr;
  cudaTextureObject_t tex_obj = 0;
  cudaSurfaceObject_t surf_obj = 0;
  bool tex_init = false;
  void ensure_texture();
#endif

public:
  size_t n;      // width of square slices
  size_t nproj; // number of angles
  size_t nz;    // number of slices
  size_t ncproj;    // number of slices
  size_t ncz;    // number of slices
  cfunc_linerec(size_t nproj, size_t nz, size_t n, size_t ncproj, size_t ncz);
  ~cfunc_linerec();
  void backprojection(size_t f_, size_t g_, size_t theta, float phi, float gain, int sz, size_t stream_);
  void backprojection_try(size_t f_, size_t g_, size_t theta_,size_t sh_,  float phi, int sz, size_t stream_);
  void backprojection_try_lamino(size_t f_, size_t g_, size_t theta_, size_t phi_, int sz,  size_t stream_);
  void free();
};

#endif