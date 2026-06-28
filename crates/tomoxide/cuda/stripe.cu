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

#include <climits>
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

// ===========================================================================
// Method 3: Vo all-stripe (`remove_all_stripe` = _rs_dead then _rs_sort).
// Reproduces the CPU golden in src/prep/stripe.rs over all nz slices at once.
// The per-column sorts use a cooperative bitonic network with a composite
// (value, original-row) key so ties break exactly like Rust's stable sort_by;
// the cross-column detection (polyfit/threshold/dilation) and the bilinear
// dead-column fill run one thread per slice. Correlation parity, not bit-exact
// (the per-column reductions reassociate sums vs the serial CPU).
// ===========================================================================

// scipy half-sample `reflect` index into [0, n) — matches stripe::reflect_index.
__device__ __forceinline__ int vo_reflect(long long i, long long n) {
  if (n == 1) return 0;
  long long period = 2 * n;
  long long j = i % period;
  if (j < 0) j += period;
  if (j >= n) j = period - 1 - j;
  return (int) j;
}

// IEEE-754 total-order key: a strict monotonic map of `f32::total_cmp` order
// into unsigned compare (handles -0.0 < +0.0 and NaN positions). Equal keys ⟺
// identical bit patterns ⟺ total_cmp == Equal.
__device__ __forceinline__ unsigned vo_okey(float f) {
  unsigned u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Total order for the bitonic sort, matching Rust's `total_cmp` plus stable
// tie-break. ascending: increasing value, ties by smaller original index.
// descending: decreasing value, ties by smaller index (Rust
// `sort_by(|a,b| b.total_cmp(a))` is stable → equal keys keep input order =
// ascending index). Returns true iff (av,ai) must precede (bv,bi).
__device__ __forceinline__ bool vo_before(float av, int ai, float bv, int bi, bool ascending) {
  unsigned ka = vo_okey(av), kb = vo_okey(bv);
  if (ka != kb) return ascending ? (ka < kb) : (ka > kb);
  return ai < bi;
}

// Cooperative bitonic sort of `n` (power of two) (value,index) pairs in shared
// memory into `vo_before` order. Block loops over the index space.
__device__ void vo_bitonic(float* v, int* idx, int n, bool ascending) {
  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      __syncthreads();
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int l = i ^ j;
        if (l > i) {
          bool up = ((i & k) == 0);
          bool i_first = vo_before(v[i], idx[i], v[l], idx[l], ascending);
          // up segment wants i before l; down segment wants l before i.
          bool swap = up ? !i_first : i_first;
          if (swap) {
            float tv = v[i];
            v[i] = v[l];
            v[l] = tv;
            int ti = idx[i];
            idx[i] = idx[l];
            idx[l] = ti;
          }
        }
      }
    }
  }
  __syncthreads();
}

// uniform_filter1d along axis 0 (the projection axis), mode='reflect', size.
__global__ void vo_uniform_axis0_ker(const float* sino, float* out, int nz, int nrow, int nc,
                                     int size) {
  long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nrow * nc;
  if (t >= total) return;
  int c = (int) (t % nc);
  long long zr = t / nc;
  int i = (int) (zr % nrow);
  int z = (int) (zr / nrow);
  int half = size / 2;
  const float* col = sino + (long long) z * nrow * nc + c;
  double sum = 0.0;
  for (int k = 0; k < size; k++) {
    int r = vo_reflect((long long) i - half + k, nrow);
    sum += (double) col[(long long) r * nc];
  }
  out[t] = (float) (sum / (double) size);
}

// listdiff[z,c] = sum_r |sino - smooth|  (per-column L1 difference).
__global__ void vo_absdiff_colsum_ker(const float* sino, const float* smooth, float* listdiff,
                                      int nz, int nrow, int nc) {
  long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nc;
  if (t >= total) return;
  int c = (int) (t % nc);
  int z = (int) (t / nc);
  const float* s = sino + (long long) z * nrow * nc + c;
  const float* m = smooth + (long long) z * nrow * nc + c;
  double sum = 0.0;
  for (int r = 0; r < nrow; r++) {
    sum += fabs((double) s[(long long) r * nc] - (double) m[(long long) r * nc]);
  }
  listdiff[t] = (float) sum;
}

// 1-D median filter along the last axis (columns) of [nz, R, nc], reflect,
// window `size`. Used for listdiffbck (R=1), rs_large sinosmooth and rs_sort
// smoothed (R=nrow). `size` must be <= 256.
__global__ void vo_median_axis1_ker(const float* in, float* out, int nz, int R, int nc, int size) {
  long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * R * nc;
  if (t >= total) return;
  int c = (int) (t % nc);
  long long zr = t / nc;
  int r = (int) (zr % R);
  int z = (int) (zr / R);
  int half = size / 2;
  const float* row = in + ((long long) z * R + r) * nc;
  float w[256];
  for (int k = 0; k < size; k++) {
    int cc = vo_reflect((long long) c - half + k, nc);
    w[k] = row[cc];
  }
  // insertion sort ascending (size is small).
  for (int a = 1; a < size; a++) {
    float key = w[a];
    int b = a - 1;
    while (b >= 0 && w[b] > key) {
      w[b + 1] = w[b];
      b--;
    }
    w[b + 1] = key;
  }
  out[t] = w[size / 2];
}

// listfact = num/den (or 1 where den == 0), elementwise.
__global__ void vo_ratio_ker(const float* num, const float* den, float* out, long long n) {
  long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  float d = den[t];
  out[t] = (d != 0.0f) ? (num[t] / d) : 1.0f;
}

// Sort each column of sino[nz,nrow,nc] into sorted[z,rank,c] (+ optional perm
// rows) using one block per (z,c). `pow2` >= nrow is the padded length.
__global__ void vo_colsort_ker(const float* sino, float* sorted, int* perm, int nz, int nrow,
                               int nc, int pow2, bool ascending) {
  extern __shared__ float smem[];
  float* v = smem;
  int* idx = (int*) (v + pow2);
  long long blk = blockIdx.x;  // z*nc + c
  int c = (int) (blk % nc);
  int z = (int) (blk / nc);
  const float* col = sino + (long long) z * nrow * nc + c;
  float sentinel = ascending ? INFINITY : -INFINITY;
  for (int i = threadIdx.x; i < pow2; i += blockDim.x) {
    if (i < nrow) {
      v[i] = col[(long long) i * nc];
      idx[i] = i;
    } else {
      v[i] = sentinel;
      idx[i] = INT_MAX;
    }
  }
  __syncthreads();
  vo_bitonic(v, idx, pow2, ascending);
  for (int i = threadIdx.x; i < nrow; i += blockDim.x) {
    long long o = (long long) z * nrow * nc + (long long) i * nc + c;
    sorted[o] = v[i];
    if (perm) perm[o] = idx[i];
  }
}

// Sort each slice row of in[nz,n] into sorted[z,rank], one block per slice.
__global__ void vo_slicesort_ker(const float* in, float* sorted, int nz, int n, int pow2,
                                 bool ascending) {
  extern __shared__ float smem[];
  float* v = smem;
  int* idx = (int*) (v + pow2);
  int z = blockIdx.x;
  const float* row = in + (long long) z * n;
  float sentinel = ascending ? INFINITY : -INFINITY;
  for (int i = threadIdx.x; i < pow2; i += blockDim.x) {
    if (i < n) {
      v[i] = row[i];
      idx[i] = i;
    } else {
      v[i] = sentinel;
      idx[i] = INT_MAX;
    }
  }
  __syncthreads();
  vo_bitonic(v, idx, pow2, ascending);
  for (int i = threadIdx.x; i < n; i += blockDim.x) sorted[(long long) z * n + i] = v[i];
}

// _detect_stripe (Vo algorithm 4): one thread per slice. `listsorted` is the
// descending-sorted copy of `listfact`. Writes the raw (un-dilated) mask.
__global__ void vo_detect_rawmask_ker(const float* listfact, const float* listsorted,
                                      float* rawmask, int nz, int nc, float snr) {
  int z = blockIdx.x * blockDim.x + threadIdx.x;
  if (z >= nz) return;
  const float* lf = listfact + (long long) z * nc;
  const float* ls = listsorted + (long long) z * nc;
  float* rm = rawmask + (long long) z * nc;
  for (int c = 0; c < nc; c++) rm[c] = 0.0f;
  int ndrop = (int) (short) (0.25 * (double) nc);  // np.int16(0.25*numdata)
  if (nc < 2 * ndrop + 2) return;
  int lo = ndrop, hi = nc - ndrop - 1;  // fit over xlist[ndrop : numdata-ndrop-1]
  double n = (double) (hi - lo);
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int v = lo; v < hi; v++) {
    double x = (double) v, y = (double) ls[v];
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  double denom = n * sxx - sx * sx;
  double slope = (denom != 0.0) ? (n * sxy - sx * sy) / denom : 0.0;
  double intercept = (sy - slope * sx) / n;
  double numt1 = intercept + slope * (double) (nc - 1);
  double noiselevel = fabs(numt1 - intercept);
  if (noiselevel < 1e-6) noiselevel = 1e-6;
  double snrd = (double) snr;
  double val1 = fabs((double) ls[0] - intercept) / noiselevel;
  double val2 = fabs((double) ls[nc - 1] - numt1) / noiselevel;
  if (val1 >= snrd) {
    double upper = intercept + noiselevel * snrd * 0.5;
    for (int c = 0; c < nc; c++)
      if ((double) lf[c] > upper) rm[c] = 1.0f;
  }
  if (val2 >= snrd) {
    double lower = numt1 - noiselevel * snrd * 0.5;
    for (int c = 0; c < nc; c++)
      if ((double) lf[c] <= lower) rm[c] = 1.0f;
  }
}

// binary_dilation (3-element SE, one iteration, border 0); optional border-zero
// of the two outer columns each side (the _rs_dead protection).
__global__ void vo_dilate_ker(const float* rawmask, float* mask, int nz, int nc, int border_zero) {
  long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nc;
  if (t >= total) return;
  int c = (int) (t % nc);
  int z = (int) (t / nc);
  const float* rm = rawmask + (long long) z * nc;
  float l = (c > 0) ? rm[c - 1] : 0.0f;
  float m = rm[c];
  float r = (c + 1 < nc) ? rm[c + 1] : 0.0f;
  float out = (l > 0.0f || m > 0.0f || r > 0.0f) ? 1.0f : 0.0f;
  if (border_zero && (c < 2 || c >= nc - 2)) out = 0.0f;
  mask[t] = out;
}

// Compact the good (mask < 1) columns of each slice, ascending, into goodx[z,:]
// and record the count. One thread per slice.
__global__ void vo_build_goodx_ker(const float* mask, int* goodx, int* goodcount, int nz, int nc) {
  int z = blockIdx.x * blockDim.x + threadIdx.x;
  if (z >= nz) return;
  const float* mk = mask + (long long) z * nc;
  int* gx = goodx + (long long) z * nc;
  int cnt = 0;
  for (int c = 0; c < nc; c++)
    if (mk[c] < 1.0f) gx[cnt++] = c;
  goodcount[z] = cnt;
}

// Bilinear (per-row linear) fill of dead columns from the bracketing good
// columns. `work` is pre-seeded with a copy of `sino`; good columns are left
// unchanged. One thread per (z,c).
__global__ void vo_interp_fill_ker(const float* sino, float* work, const float* mask,
                                   const int* goodx, const int* goodcount, int nz, int nrow,
                                   int nc) {
  long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nc;
  if (t >= total) return;
  int c = (int) (t % nc);
  int z = (int) (t / nc);
  if (mask[(long long) z * nc + c] <= 0.0f) return;  // good column
  int cnt = goodcount[z];
  if (cnt < 2) return;  // CPU: no fill when goodx.len() < 2
  const int* gx = goodx + (long long) z * nc;
  int i0 = 0;
  for (int k = 0; k < cnt - 1; k++) {
    if (gx[k] <= c && c <= gx[k + 1]) {
      i0 = k;
      break;
    }
  }
  int c0 = gx[i0], c1 = gx[i0 + 1];
  double tt = (c1 != c0) ? ((double) c - (double) c0) / ((double) c1 - (double) c0) : 0.0;
  const float* col0 = sino + (long long) z * nrow * nc + c0;
  const float* col1 = sino + (long long) z * nrow * nc + c1;
  float* wc = work + (long long) z * nrow * nc + c;
  for (int r = 0; r < nrow; r++) {
    double v0 = col0[(long long) r * nc];
    double v1 = col1[(long long) r * nc];
    wc[(long long) r * nc] = (float) ((1.0 - tt) * v0 + tt * v1);
  }
}

// _rs_large per-column intensity factor: mean(sinosort central) / mean(smooth
// central) over rows [ndrop, nrow-ndrop). Writes both f64 (for normalisation)
// and f32 (for detection) copies. One thread per (z,c).
__global__ void vo_rs_large_listfact_ker(const float* sinosort, const float* sinosmooth,
                                         double* lf64, float* lf32, int nz, int nrow, int nc,
                                         int ndrop) {
  long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nc;
  if (t >= total) return;
  int c = (int) (t % nc);
  int z = (int) (t / nc);
  const float* a = sinosort + (long long) z * nrow * nc + c;
  const float* b = sinosmooth + (long long) z * nrow * nc + c;
  double s1 = 0.0, s2 = 0.0;
  for (int r = ndrop; r < nrow - ndrop; r++) {
    s1 += (double) a[(long long) r * nc];
    s2 += (double) b[(long long) r * nc];
  }
  int cntr = nrow - 2 * ndrop;
  if (cntr < 1) cntr = 1;
  double cnt = (double) cntr;
  double m1 = s1 / cnt, m2 = s2 / cnt;
  double lf = (m2 != 0.0) ? (m1 / m2) : 1.0;
  lf64[t] = lf;
  lf32[t] = (float) lf;
}

// Normalise each column by 1/listfact (the _rs_large norm=True working copy).
__global__ void vo_normalize_ker(const float* s, const double* lf64, float* out, int nz, int nrow,
                                 int nc) {
  long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nrow * nc;
  if (t >= total) return;
  int c = (int) (t % nc);
  long long zr = t / nc;
  int z = (int) (zr / nrow);
  out[t] = (float) ((double) s[t] / lf64[(long long) z * nc + c]);
}

// Scatter the rank-smoothed profile back through the (normalised) sort order for
// masked columns only; `out` is pre-seeded with the normalised working copy.
__global__ void vo_scatter_masked_ker(const int* perm, const float* sinosmooth, const float* mask,
                                      float* out, int nz, int nrow, int nc) {
  long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nrow * nc;
  if (t >= total) return;
  int c = (int) (t % nc);
  long long zr = t / nc;
  int rank = (int) (zr % nrow);
  int z = (int) (zr / nrow);
  if (mask[(long long) z * nc + c] <= 0.0f) return;
  int row = perm[t];
  out[(long long) z * nrow * nc + (long long) row * nc + c] = sinosmooth[t];
}

// Scatter the smoothed sorted profile back to the original projection order for
// every column (the _rs_sort unsort). `out` is fully overwritten.
__global__ void vo_unsort_scatter_ker(const int* perm, const float* smoothed, float* out, int nz,
                                      int nrow, int nc) {
  long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long) nz * nrow * nc;
  if (t >= total) return;
  int c = (int) (t % nc);
  long long zr = t / nc;
  int z = (int) (zr / nrow);
  int row = perm[t];
  out[(long long) z * nrow * nc + (long long) row * nc + c] = smoothed[t];
}

// Smallest power of two >= n (n >= 1).
static inline int vo_pow2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

extern "C" {

int tomoxide_vo_uniform_axis0(const void* sino, void* out, size_t nz, size_t nrow, size_t nc,
                              size_t size, void* stream) {
  int block = 256;
  long long total = (long long) nz * nrow * nc;
  vo_uniform_axis0_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const float*) sino, (float*) out, (int) nz, (int) nrow, (int) nc, (int) size);
  return (int) cudaGetLastError();
}

int tomoxide_vo_absdiff_colsum(const void* sino, const void* smooth, void* listdiff, size_t nz,
                               size_t nrow, size_t nc, void* stream) {
  int block = 256;
  long long total = (long long) nz * nc;
  vo_absdiff_colsum_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const float*) sino, (const float*) smooth, (float*) listdiff, (int) nz, (int) nrow,
      (int) nc);
  return (int) cudaGetLastError();
}

int tomoxide_vo_median_axis1(const void* in, void* out, size_t nz, size_t R, size_t nc, size_t size,
                             void* stream) {
  if (size > 256) return -1;
  int block = 256;
  long long total = (long long) nz * R * nc;
  vo_median_axis1_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const float*) in, (float*) out, (int) nz, (int) R, (int) nc, (int) size);
  return (int) cudaGetLastError();
}

int tomoxide_vo_ratio(const void* num, const void* den, void* out, size_t n, void* stream) {
  int block = 256;
  vo_ratio_ker<<<fw_grid((long long) n, block), block, 0, (cudaStream_t) stream>>>(
      (const float*) num, (const float*) den, (float*) out, (long long) n);
  return (int) cudaGetLastError();
}

int tomoxide_vo_colsort(const void* sino, void* sorted, void* perm, size_t nz, size_t nrow,
                        size_t nc, int ascending, void* stream) {
  int p2 = vo_pow2((int) nrow);
  size_t shmem = (size_t) p2 * (sizeof(float) + sizeof(int));
  cudaFuncSetAttribute(vo_colsort_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem);
  unsigned blocks = (unsigned) (nz * nc);
  vo_colsort_ker<<<blocks, 256, shmem, (cudaStream_t) stream>>>(
      (const float*) sino, (float*) sorted, (int*) perm, (int) nz, (int) nrow, (int) nc, p2,
      ascending != 0);
  return (int) cudaGetLastError();
}

int tomoxide_vo_slicesort(const void* in, void* sorted, size_t nz, size_t n, int ascending,
                          void* stream) {
  int p2 = vo_pow2((int) n);
  size_t shmem = (size_t) p2 * (sizeof(float) + sizeof(int));
  cudaFuncSetAttribute(vo_slicesort_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem);
  vo_slicesort_ker<<<(unsigned) nz, 256, shmem, (cudaStream_t) stream>>>(
      (const float*) in, (float*) sorted, (int) nz, (int) n, p2, ascending != 0);
  return (int) cudaGetLastError();
}

int tomoxide_vo_detect_rawmask(const void* listfact, const void* listsorted, void* rawmask,
                               size_t nz, size_t nc, float snr, void* stream) {
  int block = 64;
  int blocks = (int) ((nz + block - 1) / block);
  vo_detect_rawmask_ker<<<blocks, block, 0, (cudaStream_t) stream>>>(
      (const float*) listfact, (const float*) listsorted, (float*) rawmask, (int) nz, (int) nc,
      snr);
  return (int) cudaGetLastError();
}

int tomoxide_vo_dilate(const void* rawmask, void* mask, size_t nz, size_t nc, int border_zero,
                       void* stream) {
  int block = 256;
  long long total = (long long) nz * nc;
  vo_dilate_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const float*) rawmask, (float*) mask, (int) nz, (int) nc, border_zero);
  return (int) cudaGetLastError();
}

int tomoxide_vo_build_goodx(const void* mask, void* goodx, void* goodcount, size_t nz, size_t nc,
                            void* stream) {
  int block = 64;
  int blocks = (int) ((nz + block - 1) / block);
  vo_build_goodx_ker<<<blocks, block, 0, (cudaStream_t) stream>>>(
      (const float*) mask, (int*) goodx, (int*) goodcount, (int) nz, (int) nc);
  return (int) cudaGetLastError();
}

int tomoxide_vo_interp_fill(const void* sino, void* work, const void* mask, const void* goodx,
                            const void* goodcount, size_t nz, size_t nrow, size_t nc,
                            void* stream) {
  int block = 256;
  long long total = (long long) nz * nc;
  vo_interp_fill_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const float*) sino, (float*) work, (const float*) mask, (const int*) goodx,
      (const int*) goodcount, (int) nz, (int) nrow, (int) nc);
  return (int) cudaGetLastError();
}

int tomoxide_vo_rs_large_listfact(const void* sinosort, const void* sinosmooth, void* lf64,
                                  void* lf32, size_t nz, size_t nrow, size_t nc, size_t ndrop,
                                  void* stream) {
  int block = 256;
  long long total = (long long) nz * nc;
  vo_rs_large_listfact_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const float*) sinosort, (const float*) sinosmooth, (double*) lf64, (float*) lf32, (int) nz,
      (int) nrow, (int) nc, (int) ndrop);
  return (int) cudaGetLastError();
}

int tomoxide_vo_normalize(const void* s, const void* lf64, void* out, size_t nz, size_t nrow,
                          size_t nc, void* stream) {
  int block = 256;
  long long total = (long long) nz * nrow * nc;
  vo_normalize_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const float*) s, (const double*) lf64, (float*) out, (int) nz, (int) nrow, (int) nc);
  return (int) cudaGetLastError();
}

int tomoxide_vo_scatter_masked(const void* perm, const void* sinosmooth, const void* mask,
                               void* out, size_t nz, size_t nrow, size_t nc, void* stream) {
  int block = 256;
  long long total = (long long) nz * nrow * nc;
  vo_scatter_masked_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const int*) perm, (const float*) sinosmooth, (const float*) mask, (float*) out, (int) nz,
      (int) nrow, (int) nc);
  return (int) cudaGetLastError();
}

int tomoxide_vo_unsort_scatter(const void* perm, const void* smoothed, void* out, size_t nz,
                               size_t nrow, size_t nc, void* stream) {
  int block = 256;
  long long total = (long long) nz * nrow * nc;
  vo_unsort_scatter_ker<<<fw_grid(total, block), block, 0, (cudaStream_t) stream>>>(
      (const int*) perm, (const float*) smoothed, (float*) out, (int) nz, (int) nrow, (int) nc);
  return (int) cudaGetLastError();
}

}  // extern "C"
