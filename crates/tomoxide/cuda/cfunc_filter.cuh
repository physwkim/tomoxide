#ifndef CFUNC_FILTER_CUH
#define CFUNC_FILTER_CUH

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include "defs.cuh"

class cfunc_filter {
  bool is_free = false;

  cufftHandle plan_filter_fwd;
  cufftHandle plan_filter_inv;
  real2* ge = nullptr;
  // Track which resources the constructor actually created, so a partial
  // failure tears down only what exists and `valid()` gates the factory.
  bool fwd_ok = false;
  bool inv_ok = false;
  bool valid_ = false;

public:
  size_t n;      // width of square slices
  size_t nproj; // number of angles
  size_t nz;    // number of slices
  cfunc_filter(size_t nproj, size_t nz, size_t n);
  ~cfunc_filter();
  // True only if every plan/buffer was allocated; the factory returns null
  // otherwise so an OOM surfaces as a clean error instead of a later SIGSEGV
  // when cufftXtExec runs on an unallocated work area.
  bool valid() const { return valid_; }
  void filter(size_t g, size_t w, size_t stream);
  void free();
};

#endif