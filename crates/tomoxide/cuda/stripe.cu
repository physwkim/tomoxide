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

// ===========================================================================
// Method 2: Fourier-Wavelet (`remove_stripe_fw`). Per slice: pad the projection
// axis to nx=nproj+nproj/8, run a `level`-deep db5 (Daubechies-5) 2-D wavelet
// decomposition (each band rounded to f32), damp every vertical-detail band
// along the projection axis with the Münch Fourier filter, then invert. The
// wavelet runs in f64 to match the CPU; the per-column damping FFT reuses the
// f32 cuFFT shim (`tomoxide_fft_1d`) — correlation parity, not bit-exact.
// All kernels are batched over the leading nz (one transform set per slice).
// ===========================================================================

#define FW_F 10  // db5 filter length

// pywt db5 filters (must match src/prep/wavelet.rs exactly).
__device__ const double FW_DEC_LO[FW_F] = {
    0.0033357252854737712,  -0.012580751999081999, -0.006241490212798274,
    0.07757149384004572,    -0.032244869584638375, -0.24229488706638203,
    0.13842814590132074,    0.7243085284377729,    0.6038292697971896,
    0.16010239797419293};
__device__ const double FW_DEC_HI[FW_F] = {
    -0.16010239797419293,  0.6038292697971896,    -0.7243085284377729,
    0.13842814590132074,   0.24229488706638203,   -0.032244869584638375,
    -0.07757149384004572,  -0.006241490212798274, 0.012580751999081999,
    0.0033357252854737712};
__device__ const double FW_REC_LO[FW_F] = {
    0.16010239797419293,   0.6038292697971896,    0.7243085284377729,
    0.13842814590132074,   -0.24229488706638203,  -0.032244869584638375,
    0.07757149384004572,   -0.006241490212798274, -0.012580751999081999,
    0.0033357252854737712};
__device__ const double FW_REC_HI[FW_F] = {
    0.0033357252854737712, 0.012580751999081999,  -0.006241490212798274,
    -0.07757149384004572,  -0.032244869584638375, 0.24229488706638203,
    0.13842814590132074,   -0.7243085284377729,   0.6038292697971896,
    -0.16010239797419293};

// numpy pad(mode='symmetric') half-sample reflection of t into [0, n).
__device__ int fw_sym(long long t, long long n) {
  if (n == 1) return 0;
  long long period = 2 * n;
  long long m = t % period;
  if (m < 0) m += period;
  if (m >= n) m = period - 1 - m;
  return (int) m;
}

// f32 round-trip of every element (emulates tomopy's float32 pywt forward pass).
__global__ void fw_round_ker(double* a, long long n) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  a[i] = (double) (float) a[i];
}

// Pad: f32 sino [nz,nproj,ncol] -> f64 approx [nz,nx,ncol], sinogram at rows
// [xshift, xshift+nproj); elsewhere zero.
__global__ void fw_pad_ker(const float* in, double* approx, int nz, int nproj, int ncol, int nx,
                           int xshift) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nx * ncol;
  if (i >= total) return;
  int c = (int) (i % ncol);
  long long row = i / ncol;
  int r = (int) (row % nx);
  int z = (int) (row / nx);
  int p = r - xshift;
  approx[i] = (p >= 0 && p < nproj) ? (double) in[((long long) z * nproj + p) * ncol + c] : 0.0;
}

// Final crop: f64 sli [nz, sliR, sliC] -> f32 sino [nz,nproj,ncol], reading rows
// [xshift, xshift+nproj) and cols [0, ncol).
__global__ void fw_final_ker(const double* sli, float* out, int nz, int nproj, int ncol, int sliR,
                             int sliC, int xshift) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nproj * ncol;
  if (i >= total) return;
  int c = (int) (i % ncol);
  long long row = i / ncol;
  int p = (int) (row % nproj);
  int z = (int) (row / nproj);
  out[i] = (float) sli[((long long) z * sliR + (xshift + p)) * sliC + c];
}

// Crop top-left [oR,oC] of [nz, inR, inC].
__global__ void fw_crop_ker(const double* in, double* out, int nz, int inR, int inC, int oR,
                            int oC) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * oR * oC;
  if (i >= total) return;
  int c = (int) (i % oC);
  long long row = i / oC;
  int r = (int) (row % oR);
  int z = (int) (row / oR);
  out[i] = in[((long long) z * inR + r) * inC + c];
}

// 1-D forward DWT along the LAST axis (columns within a row): in [nz,R,C] ->
// lo,hi [nz,R,oC], oC=(C+F-1)/2. out[i] = Σ_k filt[k]·x[sym(2i-k+1)].
__global__ void fw_dwt_rows_ker(const double* in, double* lo, double* hi, int nz, int R, int C,
                                int oC) {
  long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * R * oC;
  if (idx >= total) return;
  int i = (int) (idx % oC);
  long long row = idx / oC;
  int r = (int) (row % R);
  int z = (int) (row / R);
  const double* x = in + ((long long) z * R + r) * C;
  double a = 0.0, d = 0.0;
  for (int k = 0; k < FW_F; k++) {
    double xv = x[fw_sym(2 * (long long) i - k + 1, C)];
    a += FW_DEC_LO[k] * xv;
    d += FW_DEC_HI[k] * xv;
  }
  lo[idx] = a;
  hi[idx] = d;
}

// 1-D forward DWT along the MIDDLE axis (rows): in [nz,R,C] -> lo,hi [nz,oR,C],
// oR=(R+F-1)/2.
__global__ void fw_dwt_cols_ker(const double* in, double* lo, double* hi, int nz, int R, int C,
                                int oR) {
  long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * oR * C;
  if (idx >= total) return;
  int c = (int) (idx % C);
  long long row = idx / C;
  int i = (int) (row % oR);
  int z = (int) (row / oR);
  const double* base = in + (long long) z * R * C + c;
  double a = 0.0, d = 0.0;
  for (int k = 0; k < FW_F; k++) {
    double xv = base[(long long) fw_sym(2 * (long long) i - k + 1, R) * C];
    a += FW_DEC_LO[k] * xv;
    d += FW_DEC_HI[k] * xv;
  }
  lo[idx] = a;
  hi[idx] = d;
}

// 1-D inverse DWT along the MIDDLE axis (rows): lo,hi [nz,L0,C] -> out
// [nz,rR,C], rR=2*L0+2-F. out[m] = Σ_t lo[t]·REC_LO[F-2+m-2t] + hi[t]·REC_HI[..].
__global__ void fw_idwt_cols_ker(const double* lo, const double* hi, double* out, int nz, int L0,
                                 int C, int rR) {
  long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * rR * C;
  if (idx >= total) return;
  int c = (int) (idx % C);
  long long row = idx / C;
  int m = (int) (row % rR);
  int z = (int) (row / rR);
  const double* lob = lo + (long long) z * L0 * C + c;
  const double* hib = hi + (long long) z * L0 * C + c;
  double acc = 0.0;
  int tlo = (m - FW_F) / 2;
  if (tlo < 0) tlo = 0;
  int thi = (m + FW_F) / 2;
  if (thi > L0 - 1) thi = L0 - 1;
  for (int t = tlo; t <= thi; t++) {
    int fi = FW_F - 2 + m - 2 * t;
    if (fi < 0 || fi >= FW_F) continue;
    acc += lob[(long long) t * C] * FW_REC_LO[fi] + hib[(long long) t * C] * FW_REC_HI[fi];
  }
  out[idx] = acc;
}

// 1-D inverse DWT along the LAST axis (cols): lo,hi [nz,R,L1] -> out [nz,R,rC],
// rC=2*L1+2-F.
__global__ void fw_idwt_rows_ker(const double* lo, const double* hi, double* out, int nz, int R,
                                 int L1, int rC) {
  long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * R * rC;
  if (idx >= total) return;
  int m = (int) (idx % rC);
  long long row = idx / rC;
  int r = (int) (row % R);
  int z = (int) (row / R);
  const double* lob = lo + ((long long) z * R + r) * L1;
  const double* hib = hi + ((long long) z * R + r) * L1;
  double acc = 0.0;
  int tlo = (m - FW_F) / 2;
  if (tlo < 0) tlo = 0;
  int thi = (m + FW_F) / 2;
  if (thi > L1 - 1) thi = L1 - 1;
  for (int t = tlo; t <= thi; t++) {
    int fi = FW_F - 2 + m - 2 * t;
    if (fi < 0 || fi >= FW_F) continue;
    acc += lob[t] * FW_REC_LO[fi] + hib[t] * FW_REC_HI[fi];
  }
  out[idx] = acc;
}

// Damping helpers. cv is [nz, my, mx] (f64). The FFT runs along `my` (axis 0),
// so we gather each (z, col) lane into a contiguous length-`my` complex
// transform laid out as [nz*mx][my], multiply the spectrum by d[my], invert.
__global__ void fw_damp_gather_ker(const double* cv, float* cplx, int nz, int my, int mx) {
  long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * my * mx;
  if (idx >= total) return;
  int c = (int) (idx % mx);
  long long row = idx / mx;
  int r = (int) (row % my);
  int z = (int) (row / my);
  long long dst = ((long long) z * mx + c) * my + r;  // transform (z,c), element r
  cplx[2 * dst] = (float) cv[idx];
  cplx[2 * dst + 1] = 0.0f;
}

__global__ void fw_damp_apply_ker(float* cplx, const double* d, int nz, int my, int mx) {
  long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * mx * my;
  if (t >= total) return;
  int k = (int) (t % my);  // frequency index
  float dv = (float) d[k];
  cplx[2 * t] *= dv;
  cplx[2 * t + 1] *= dv;
}

__global__ void fw_damp_scatter_ker(const float* cplx, double* cv, int nz, int my, int mx) {
  long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * my * mx;
  if (idx >= total) return;
  int c = (int) (idx % mx);
  long long row = idx / mx;
  int r = (int) (row % my);
  int z = (int) (row / my);
  long long src = ((long long) z * mx + c) * my + r;
  cv[idx] = (double) cplx[2 * src];
}

static inline int fw_grid(long long total, int block) { return (int) ((total + block - 1) / block); }

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

// --- Fourier-Wavelet building blocks (orchestrated from Rust) ---

int tomoxide_fw_pad(const void* in, void* approx, size_t nz, size_t nproj, size_t ncol, size_t nx,
                    size_t xshift, void* stream) {
  int block = 256;
  long long total = (long long) nz * nx * ncol;
  fw_pad_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const float*) in, (double*) approx, (int) nz, (int) nproj, (int) ncol, (int) nx,
      (int) xshift);
  return (int) cudaGetLastError();
}

int tomoxide_fw_final(const void* sli, void* out, size_t nz, size_t nproj, size_t ncol, size_t sliR,
                      size_t sliC, size_t xshift, void* stream) {
  int block = 256;
  long long total = (long long) nz * nproj * ncol;
  fw_final_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const double*) sli, (float*) out, (int) nz, (int) nproj, (int) ncol, (int) sliR, (int) sliC,
      (int) xshift);
  return (int) cudaGetLastError();
}

int tomoxide_fw_crop(const void* in, void* out, size_t nz, size_t inR, size_t inC, size_t oR,
                     size_t oC, void* stream) {
  int block = 256;
  long long total = (long long) nz * oR * oC;
  fw_crop_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const double*) in, (double*) out, (int) nz, (int) inR, (int) inC, (int) oR, (int) oC);
  return (int) cudaGetLastError();
}

int tomoxide_fw_round(void* a, size_t n, void* stream) {
  int block = 256;
  fw_round_ker<<<fw_grid((long long) n, block), block, 0, (cudaStream_t) stream>>>((double*) a,
                                                                                   (long long) n);
  return (int) cudaGetLastError();
}

int tomoxide_fw_dwt_rows(const void* in, void* lo, void* hi, size_t nz, size_t R, size_t C,
                         void* stream) {
  int block = 256, oC = (int) ((C + FW_F - 1) / 2);
  long long total = (long long) nz * R * oC;
  fw_dwt_rows_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const double*) in, (double*) lo, (double*) hi, (int) nz, (int) R, (int) C, oC);
  return (int) cudaGetLastError();
}

int tomoxide_fw_dwt_cols(const void* in, void* lo, void* hi, size_t nz, size_t R, size_t C,
                         void* stream) {
  int block = 256, oR = (int) ((R + FW_F - 1) / 2);
  long long total = (long long) nz * oR * C;
  fw_dwt_cols_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const double*) in, (double*) lo, (double*) hi, (int) nz, (int) R, (int) C, oR);
  return (int) cudaGetLastError();
}

int tomoxide_fw_idwt_cols(const void* lo, const void* hi, void* out, size_t nz, size_t L0, size_t C,
                          void* stream) {
  int block = 256, rR = (int) (2 * L0 + 2 - FW_F);
  long long total = (long long) nz * rR * C;
  fw_idwt_cols_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const double*) lo, (const double*) hi, (double*) out, (int) nz, (int) L0, (int) C, rR);
  return (int) cudaGetLastError();
}

int tomoxide_fw_idwt_rows(const void* lo, const void* hi, void* out, size_t nz, size_t R, size_t L1,
                          void* stream) {
  int block = 256, rC = (int) (2 * L1 + 2 - FW_F);
  long long total = (long long) nz * R * rC;
  fw_idwt_rows_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const double*) lo, (const double*) hi, (double*) out, (int) nz, (int) R, (int) L1, rC);
  return (int) cudaGetLastError();
}

int tomoxide_fw_damp_gather(const void* cv, void* cplx, size_t nz, size_t my, size_t mx,
                            void* stream) {
  int block = 256;
  long long total = (long long) nz * my * mx;
  fw_damp_gather_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const double*) cv, (float*) cplx, (int) nz, (int) my, (int) mx);
  return (int) cudaGetLastError();
}

int tomoxide_fw_damp_apply(void* cplx, const void* d, size_t nz, size_t my, size_t mx,
                           void* stream) {
  int block = 256;
  long long total = (long long) nz * mx * my;
  fw_damp_apply_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (float*) cplx, (const double*) d, (int) nz, (int) my, (int) mx);
  return (int) cudaGetLastError();
}

int tomoxide_fw_damp_scatter(const void* cplx, void* cv, size_t nz, size_t my, size_t mx,
                             void* stream) {
  int block = 256;
  long long total = (long long) nz * my * mx;
  fw_damp_scatter_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const float*) cplx, (double*) cv, (int) nz, (int) my, (int) mx);
  return (int) cudaGetLastError();
}

}  // extern "C"
