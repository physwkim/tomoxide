#ifndef CFUNC_FOURIERREC_CUH
#define CFUNC_FOURIERREC_CUH

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include "defs.cuh"


class cfunc_fourierrec {
  bool is_free = false;

  size_t m;
  float mu;


  float *x = nullptr;
  float *y = nullptr;
  float* theta;
  real2 *fde = nullptr;

  cufftHandle plan2d;
  cufftHandle plan1d;

  // Track which resources the constructor actually created, so a partial
  // failure tears down only what exists and `valid()` gates the factory.
  bool plan2d_ok = false;
  bool plan1d_ok = false;
  bool valid_ = false;

  // pre-computed kernel grid dimensions (constant for lifetime of object)
  dim3 GS2d0, GS3d0, GS3d1, GS3d2, GS3d3;

public:
  size_t n;      // width of square slices
  size_t nproj; // number of angles
  size_t nz;    // number of slices
  cfunc_fourierrec(size_t nproj, size_t nz, size_t n, size_t theta);
  ~cfunc_fourierrec();
  // True only if every plan/buffer was allocated; the factory returns null
  // otherwise so an OOM surfaces as a clean error instead of a later SIGSEGV
  // when cufftXtExec runs on an unallocated work area.
  bool valid() const { return valid_; }
  void backprojection(size_t f, size_t g, size_t stream);
  void free();
};

#endif