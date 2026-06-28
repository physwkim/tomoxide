// GPU stripe removal kernels (device-resident, run on the transposed f32
// sinogram [nz, nproj, ncol] in place, before the optional f16 cast). These
// reproduce the CPU goldens in `src/prep/stripe.rs`; because they use parallel
// reductions / cuFFT / sort where the CPU uses serial loops, they are held to
// correlation parity, not bit-exactness (see the parity examples).
//
// Method 1: Titarenko (`remove_stripe_ti`, nblock = 0). Per slice it builds the
// first- and second-difference corrected sinograms d1 = _ring(s,1,1),
// d2 = _ring(s,2,1) and combines them as sqrt(d1*d2 + beta*|min(d1*d2)|).
// Each _ring solves (HᵀH + alpha I) q = f for a per-column offset q via the
// conjugate-gradient method (`_ringCGM`). We run one thread block per slice and
// solve the two CG systems with block-cooperative f64 reductions; the operator
// (HᵀH) and the iteration are identical to the CPU, so each block converges to
// the same q to the 1e-7 residual floor.

#include <cstddef>
#include <cuda_runtime.h>
#include <math.h>

// Block-wide reduction over `local` (block size must be a power of two).
// op: 0 = sum, 1 = min, 2 = max. Returns the result to every thread.
__device__ double blk_reduce(double local, double* sh, int op) {
  int tid = threadIdx.x;
  sh[tid] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (op == 0)
        sh[tid] += sh[tid + s];
      else if (op == 1)
        sh[tid] = fmin(sh[tid], sh[tid + s]);
      else
        sh[tid] = fmax(sh[tid], sh[tid + s]);
    }
    __syncthreads();
  }
  double res = sh[0];
  __syncthreads();
  return res;
}

__device__ double blk_dot(const double* a, const double* b, int n, double* sh) {
  double local = 0.0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) local += a[i] * b[i];
  return blk_reduce(local, sh, 0);
}

// y[i] = (HᵀH x)[i] for the _ringMatXvec operator: conv(conv(x, flip(h))[lh-1 :
// n], h)[i]. h has length lh (<= 3); hf is h reversed. Single output element i.
__device__ double ti_matxvec_elem(const double* x, int n, const double* h, const double* hf,
                                  int lh, int i) {
  double acc = 0.0;
  for (int j = 0; j < lh; j++) {
    int t = i - j;                 // index into the cropped conv `u`
    if (t < 0 || t > n - lh) continue;  // u defined for t in [0, n-lh]
    double uv = 0.0;               // u[t] = sum_k hf[k] * x[t + (lh-1) - k]
    for (int k = 0; k < lh; k++) {
      int idx = t + (lh - 1) - k;
      if (idx >= 0 && idx < n) uv += hf[k] * x[idx];
    }
    acc += h[j] * uv;
  }
  return acc;
}

// One thread block per slice z. `scratch` is nz * 7 * ncol doubles; block z owns
// the 7*ncol window [r | w | z | x | pp | qA | qB].
__global__ void ti_slice_ker(float* sino, int nproj, int ncol, float beta, double* scratch) {
  extern __shared__ double sh[];
  int z = blockIdx.x;
  float* sino_z = sino + (long long) z * nproj * ncol;
  double* base = scratch + (long long) z * 7 * ncol;
  double* r = base;
  double* w = base + ncol;
  double* zc = base + 2 * ncol;
  double* xq = base + 3 * ncol;
  double* pp = base + 4 * ncol;
  double* qA = base + 5 * ncol;
  double* qB = base + 6 * ncol;

  // mysino[col][angle] = sino_z[angle*ncol + col], NaN -> 0 (matches _ring).
  // val(a, c) reads that element with the NaN guard.
#define TI_VAL(a, c) ({ double _v = (double) sino_z[(long long)(a) * ncol + (c)]; isnan(_v) ? 0.0 : _v; })

  for (int variant = 0; variant < 2; variant++) {
    // Finite-difference kernel h and its reverse hf.
    double h[3], hf[3];
    int lh;
    if (variant == 0) {  // _kernel(1,1) = [1, -1]
      lh = 2;
      h[0] = 1.0; h[1] = -1.0;
    } else {             // _kernel(2,1) = [-1, 2, -1]
      lh = 3;
      h[0] = -1.0; h[1] = 2.0; h[2] = -1.0;
    }
    for (int k = 0; k < lh; k++) hf[k] = h[lh - 1 - k];

    // pp[col] = mean over angles of mysino[col][angle].
    for (int c = threadIdx.x; c < ncol; c += blockDim.x) {
      double s = 0.0;
      for (int a = 0; a < nproj; a++) s += TI_VAL(a, c);
      pp[c] = s / (double) nproj;
    }
    __syncthreads();

    // alpha = 1 / (2 * (max - min)) over per-angle column sums.
    double lmin = INFINITY, lmax = -INFINITY;
    for (int a = threadIdx.x; a < nproj; a += blockDim.x) {
      double s = 0.0;
      for (int c = 0; c < ncol; c++) s += TI_VAL(a, c);
      lmin = fmin(lmin, s);
      lmax = fmax(lmax, s);
    }
    double gmin = blk_reduce(lmin, sh, 1);
    double gmax = blk_reduce(lmax, sh, 2);
    double alpha = 1.0 / (2.0 * (gmax - gmin));

    // CG init: r = f = -HᵀH pp, w = -r, x = 0.
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) {
      double mv = ti_matxvec_elem(pp, ncol, h, hf, lh, i);
      r[i] = -mv;
      w[i] = mv;
      xq[i] = 0.0;
    }
    __syncthreads();
    // z = apply(w) = HᵀH w + alpha w.
    for (int i = threadIdx.x; i < ncol; i += blockDim.x)
      zc[i] = ti_matxvec_elem(w, ncol, h, hf, lh, i) + alpha * w[i];
    __syncthreads();
    double a = blk_dot(r, w, ncol, sh) / blk_dot(w, zc, ncol, sh);
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) xq[i] += a * w[i];
    __syncthreads();

    for (int iter = 0; iter < 1000000; iter++) {
      for (int i = threadIdx.x; i < ncol; i += blockDim.x) r[i] -= a * zc[i];
      __syncthreads();
      double norm = sqrt(blk_dot(r, r, ncol, sh));
      if (norm < 1e-7) break;
      double bb = blk_dot(r, zc, ncol, sh) / blk_dot(w, zc, ncol, sh);
      for (int i = threadIdx.x; i < ncol; i += blockDim.x) w[i] = -r[i] + bb * w[i];
      __syncthreads();
      for (int i = threadIdx.x; i < ncol; i += blockDim.x)
        zc[i] = ti_matxvec_elem(w, ncol, h, hf, lh, i) + alpha * w[i];
      __syncthreads();
      a = blk_dot(r, w, ncol, sh) / blk_dot(w, zc, ncol, sh);
      for (int i = threadIdx.x; i < ncol; i += blockDim.x) xq[i] += a * w[i];
      __syncthreads();
    }

    double* q = (variant == 0) ? qA : qB;
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) q[i] = xq[i];
    __syncthreads();
  }

  // p = d1*d2 (f32), shift = beta*|min p|, out = sqrt(p + shift).
  float lpmin = INFINITY;
  long long total = (long long) nproj * ncol;
  for (long long idx = threadIdx.x; idx < total; idx += blockDim.x) {
    int c = (int) (idx % ncol);
    int rr = (int) (idx / ncol);
    double v = TI_VAL(rr, c);
    float d1 = (float) (v + qA[c]);
    float d2 = (float) (v + qB[c]);
    lpmin = fminf(lpmin, d1 * d2);
  }
  float gpmin = (float) blk_reduce((double) lpmin, sh, 1);
  float shift = beta * fabsf(gpmin);
  for (long long idx = threadIdx.x; idx < total; idx += blockDim.x) {
    int c = (int) (idx % ncol);
    int rr = (int) (idx / ncol);
    double v = TI_VAL(rr, c);
    float d1 = (float) (v + qA[c]);
    float d2 = (float) (v + qB[c]);
    sino_z[idx] = sqrtf(d1 * d2 + shift);
  }
#undef TI_VAL
}

extern "C" {

// Titarenko stripe removal on the f32 sinogram [nz, nproj, ncol], in place.
// scratch: nz * 7 * ncol doubles. Returns 0 on success, the cudaError_t else.
int tomoxide_stripe_ti(void* sino, size_t nz, size_t nproj, size_t ncol, float beta, void* scratch,
                       void* stream) {
  int block = 256;
  size_t shmem = (size_t) block * sizeof(double);
  ti_slice_ker<<<(unsigned) nz, block, shmem, (cudaStream_t) stream>>>(
      (float*) sino, (int) nproj, (int) ncol, beta, (double*) scratch);
  return (int) cudaGetLastError();
}

}  // extern "C"
