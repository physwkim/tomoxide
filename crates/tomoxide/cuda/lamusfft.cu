// Device-resident gridding + modulation kernels for Fourier/USFFT laminography
// (tomocupy `LamFourierRec`). These are a faithful, full-complex GPU port of the
// host loops in `crates/tomoxide/src/recon/lamino.rs` — the CPU golden that is
// itself validated against tomocupy (Pearson 0.99995). The FFTs themselves are
// NOT here: the Rust orchestrator drives the existing device-resident cuFFT
// (`tomoxide_fft_1d`/`tomoxide_fft_2d`, bound to `cudaStreamPerThread`) between
// these gridding stages, so the whole pipeline stays on the GPU with no host
// round-trips.
//
// Like the CPU port, this carries the FULL complex spectra (tomocupy's R2C
// half-spectrum optimization is dropped) so each operator is a clean transpose.
// The gather/wrap adjoints accumulate with `atomicAdd`, so the low bits are
// nondeterministic vs the serial CPU path — correlation-verified, not bit-exact
// (tomocupy's CUDA gather is likewise atomic).
//
// Layout & index conventions mirror lamino.rs exactly:
//   fdee1 (usfft1d): [ng, n1, n0]  index a*n0*n1 + ty*n0 + tx   (batch = n0*n1)
//   fdee2 (usfft2d): [nky, gy, gx] index ky*gx*gy + yg*gx + xg  (batch = nky)
// Sign modulation `sign(i) = -1 (i even) / +1 (i odd)` uses the GLOBAL grid index.
// Compiled by build.rs (nvcc, `--default-stream per-thread`); kernels launch on
// the passed stream (null == per-thread default == the FFT's stream).

#include <cuda_runtime.h>
#include <math.h>

#define LAM_PI 3.14159265358979323846f

// Centered-DFT modulation sign: -1 on even indices, +1 on odd (tomocupy fftshiftc).
__device__ __forceinline__ float lam_sign(long long i) { return (i & 1LL) ? 1.0f : -1.0f; }

__device__ __forceinline__ float2 lam_cscale(float2 a, float s) {
  return make_float2(a.x * s, a.y * s);
}

// atomicAdd on a complex cell (real & imag separately; float atomicAdd is native).
__device__ __forceinline__ void lam_catomic(float2* p, float2 v) {
  atomicAdd(&(p->x), v.x);
  atomicAdd(&(p->y), v.y);
}

static inline unsigned int lam_grid(long long total, int block) {
  return (unsigned int)((total + block - 1) / block);
}

extern "C" {

// ===========================================================================
// ramp_filter_detw  (lamino.rs:668) — plain |f| ramp on detector-width lines.
//   proj [ntheta, deth, detw] real, ne = 2*detw, pad = (ne-detw)/2, nlines = ntheta*deth.
// ===========================================================================

// Edge-replicate real proj into complex padded lines [nlines, ne].
__global__ void lam_ramp_pad_ker(const float* proj, float2* buf, long long nlines,
                                 int detw, int ne, int pad) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = nlines * (long long)ne;
  if (i >= total) return;
  long long l = i / ne;
  int k = (int)(i - l * ne);
  int src = k - pad;
  if (src < 0) src = 0;
  else if (src >= detw) src = detw - 1;
  float v = proj[l * detw + src];
  buf[i] = make_float2(v, 0.0f);
}

// Multiply each frequency by the centered |f| ramp: (k<=ne/2 ? k : ne-k)/ne,
// then by the rotation-center linear phase exp(-2*pi*i * t * shift) with
// t = fftfreq(ne) and shift = detw/2 - center (tomocupy `fbp_filter_center`).
// shift == 0 (center == detw/2) leaves the ramp unchanged.
__global__ void lam_ramp_mul_ker(float2* buf, long long nlines, int ne, float shift) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = nlines * (long long)ne;
  if (i >= total) return;
  long long l = i / ne;
  int k = (int)(i - l * ne);
  float f = (k <= ne / 2) ? (float)k : (float)(ne - k);
  float r = f / (float)ne;
  // signed frequency t (Hermitian) so the center-shift phase keeps the crop real.
  float t = (k <= ne / 2) ? (float)k / (float)ne : (float)(k - ne) / (float)ne;
  float ang = -2.0f * LAM_PI * t * shift;
  float c = cosf(ang), s = sinf(ang);
  float2 b = buf[i];
  buf[i] = make_float2(r * (b.x * c - b.y * s), r * (b.x * s + b.y * c));
}

// Crop the padded lines back to real proj [ntheta, deth, detw] (take .re at pad+k).
__global__ void lam_ramp_crop_ker(const float2* buf, float* proj, long long nlines,
                                  int detw, int ne, int pad) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = nlines * (long long)detw;
  if (i >= total) return;
  long long l = i / detw;
  int k = (int)(i - l * detw);
  proj[i] = buf[l * (long long)ne + pad + k].x;
}

int tomoxide_lam_ramp_pad(const void* proj, void* buf, long long nlines, int detw,
                          int ne, int pad, void* stream) {
  int block = 256;
  lam_ramp_pad_ker<<<lam_grid(nlines * (long long)ne, block), block, 0, (cudaStream_t)stream>>>(
      (const float*)proj, (float2*)buf, nlines, detw, ne, pad);
  return (int)cudaGetLastError();
}
int tomoxide_lam_ramp_mul(void* buf, long long nlines, int ne, float shift, void* stream) {
  int block = 256;
  lam_ramp_mul_ker<<<lam_grid(nlines * (long long)ne, block), block, 0, (cudaStream_t)stream>>>(
      (float2*)buf, nlines, ne, shift);
  return (int)cudaGetLastError();
}
int tomoxide_lam_ramp_crop(const void* buf, void* proj, long long nlines, int detw,
                           int ne, int pad, void* stream) {
  int block = 256;
  lam_ramp_crop_ker<<<lam_grid(nlines * (long long)detw, block), block, 0, (cudaStream_t)stream>>>(
      (const float2*)buf, (float*)proj, nlines, detw, ne, pad);
  return (int)cudaGetLastError();
}

// ===========================================================================
// fft2d_fwd  (lamino.rs:183) — centered 2-D FFT of each projection, scaled 1/(deth*detw).
//   pre:  real proj [ntheta,deth,detw] -> complex out * sign(tx)*sign(ty)
//   (cuFFT 2d forward, batch=ntheta, in place)
//   post: complex *= -sign(tx)*sign(ty)*scale
// ===========================================================================

__global__ void lam_fft2d_pre_ker(const float* proj, float2* out, long long ntheta,
                                  int deth, int detw) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long slice = (long long)deth * detw;
  long long total = ntheta * slice;
  if (i >= total) return;
  long long r = i % slice;
  int ty = (int)(r / detw);
  int tx = (int)(r - (long long)ty * detw);
  float s = lam_sign(tx) * lam_sign(ty);
  out[i] = make_float2(proj[i] * s, 0.0f);
}

__global__ void lam_fft2d_post_ker(float2* out, long long ntheta, int deth, int detw,
                                   float scale) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long slice = (long long)deth * detw;
  long long total = ntheta * slice;
  if (i >= total) return;
  long long r = i % slice;
  int ty = (int)(r / detw);
  int tx = (int)(r - (long long)ty * detw);
  float s = -lam_sign(tx) * lam_sign(ty) * scale;
  out[i] = lam_cscale(out[i], s);
}

int tomoxide_lam_fft2d_pre(const void* proj, void* out, long long ntheta, int deth,
                           int detw, void* stream) {
  int block = 256;
  long long total = ntheta * (long long)deth * detw;
  lam_fft2d_pre_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (const float*)proj, (float2*)out, ntheta, deth, detw);
  return (int)cudaGetLastError();
}
int tomoxide_lam_fft2d_post(void* out, long long ntheta, int deth, int detw, void* stream) {
  int block = 256;
  long long total = ntheta * (long long)deth * detw;
  float scale = 1.0f / (float)((long long)deth * detw);
  lam_fft2d_post_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (float2*)out, ntheta, deth, detw, scale);
  return (int)cudaGetLastError();
}

// ===========================================================================
// usfft2d_adj  (lamino.rs:479)
//   g [ntheta,nky,detw] complex -> f [n1,nky,n0] complex (summed over theta)
//   n0=detw(x-freq), n1(y-freq), nky depth-freq slices (batch). gx=2n0+2m0, gy=2n1+2m1.
// ===========================================================================

// take_x: in-plane frequency sample positions xs, ys [ntheta, nky, detw].
// `nky` is the number of depth-frequency slices stored in THIS chunk; `ky0` is
// the absolute index of the chunk's first slice and `deth` the full depth-freq
// count, so the sample coordinate uses the absolute (ky0+ky)/deth normalization
// while the buffer stays chunk-sized (whole-volume path passes ky0=0, nky=deth).
__global__ void lam_takexy2d_ker(const float* theta, float* xs, float* ys,
                                 long long ntheta, int nky, int detw, float cph,
                                 int ky0, int deth) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = ntheta * (long long)nky * detw;
  if (i >= total) return;
  long long per = (long long)nky * detw;
  int tz = (int)(i / per);
  long long r = i - (long long)tz * per;
  int ky = (int)(r / detw);
  int kx = (int)(r - (long long)ky * detw);
  float th = theta[tz];
  float st = sinf(th), ct = cosf(th);
  float lim = 0.5f - 1e-5f;
  float kv = ((float)(ky0 + ky) - (float)deth / 2.0f) / (float)deth;
  float ku = ((float)kx - (float)detw / 2.0f) / (float)detw;
  float x = ku * ct + kv * st * cph;
  float y = ku * st - kv * ct * cph;
  x = fminf(fmaxf(x, -lim), lim);
  y = fminf(fmaxf(y, -lim), lim);
  xs[i] = x;
  ys[i] = y;
}

// gather2d adj: scatter each (tz,ky,kx) sample into the (x,y) grid, atomicAdd.
// `xs`/`ys` are chunk-local `[ntheta, nky, detw]` (indexed by `i`); the input
// spectrum `g` may be a strided view of a larger `[ntheta, gdeth, detw]` buffer,
// read at absolute depth-freq `ky0+ky` (device-resident path); the host path
// passes `gdeth==nky, ky0==0` so `gidx==i`.
__global__ void lam_gather2d_adj_ker(const float2* g, const float* xs, const float* ys,
                                     float2* fdee, long long ntheta, int nky, int detw,
                                     int n0, int n1, int m0, int m1, float mu0, float mu1,
                                     int gdeth, int ky0) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = ntheta * (long long)nky * detw;
  if (i >= total) return;
  long long per = (long long)nky * detw;
  int tz = (int)(i / per);
  long long r = i - (long long)tz * per;
  int ky = (int)(r / detw);
  int kx = (int)(r - (long long)ky * detw);
  float x0 = xs[i], y0 = ys[i];
  long long gidx = (long long)tz * gdeth * detw + (long long)(ky0 + ky) * detw + kx;
  float2 g0 = g[gidx];
  int gx = 2 * n0 + 2 * m0;
  int gy = 2 * n1 + 2 * m1;
  long long sub = (long long)ky * gx * gy;
  float wpre = LAM_PI / sqrtf(mu0 * mu1 * (float)ntheta);
  long long base0 = (long long)floorf(2.0f * (float)n0 * x0) - m0;
  long long base1 = (long long)floorf(2.0f * (float)n1 * y0) - m1;
  for (int i1 = 0; i1 < 2 * m1 + 1; i1++) {
    long long ell1 = base1 + i1;
    float w1 = (float)ell1 / (2.0f * (float)n1) - y0;
    float ew1 = expf(-LAM_PI * LAM_PI / mu1 * w1 * w1);
    long long yg = (long long)n1 + m1 + ell1;
    for (int i0 = 0; i0 < 2 * m0 + 1; i0++) {
      long long ell0 = base0 + i0;
      float w0 = (float)ell0 / (2.0f * (float)n0) - x0;
      float w = wpre * expf(-LAM_PI * LAM_PI / mu0 * w0 * w0) * ew1;
      long long xg = (long long)n0 + m0 + ell0;
      lam_catomic(&fdee[sub + yg * gx + xg], lam_cscale(g0, w));
    }
  }
}

// wrap2d adj: fold the m-wide borders back into the interior in x and y, per ky slice.
__global__ void lam_wrap2d_adj_ker(float2* fdee, int nky, int n0, int n1, int m0, int m1) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int gx = 2 * n0 + 2 * m0;
  int gy = 2 * n1 + 2 * m1;
  long long total = (long long)nky * gy * gx;
  if (i >= total) return;
  long long per = (long long)gy * gx;
  int ky = (int)(i / per);
  long long r = i - (long long)ky * per;
  int ty = (int)(r / gx);
  int tx = (int)(r - (long long)ty * gx);
  if (tx < m0 || tx >= 2 * n0 + m0 || ty < m1 || ty >= 2 * n1 + m1) {
    int tx0 = (tx + 2 * n0 - m0) % (2 * n0);
    int ty0 = (ty + 2 * n1 - m1) % (2 * n1);
    long long sub = (long long)ky * gx * gy;
    long long id2 = sub + (long long)(ty0 + m1) * gx + (tx0 + m0);
    lam_catomic(&fdee[id2], fdee[i]);
  }
}

// centered inverse 2-D FFT window extract: fdee[nky,gy,gx] -> win[nky,wy,wx] * sign.
__global__ void lam_win2d_extract_ker(const float2* fdee, float2* win, int nky,
                                      int n0, int n1, int m0, int m1) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int gx = 2 * n0 + 2 * m0;
  int gy = 2 * n1 + 2 * m1;
  int wx = 2 * n0, wy = 2 * n1;
  long long perw = (long long)wy * wx;
  long long total = (long long)nky * perw;
  if (i >= total) return;
  int ky = (int)(i / perw);
  long long r = i - (long long)ky * perw;
  int iy = (int)(r / wx);
  int ix = (int)(r - (long long)iy * wx);
  float s = lam_sign(m0 + ix) * lam_sign(m1 + iy);
  long long fidx = (long long)ky * gx * gy + (long long)(m1 + iy) * gx + (m0 + ix);
  win[i] = lam_cscale(fdee[fidx], s);
}

// scatter window back into fdee with sign * renorm (renorm = wx*wy for inverse).
__global__ void lam_win2d_scatter_ker(float2* fdee, const float2* win, int nky,
                                      int n0, int n1, int m0, int m1) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int gx = 2 * n0 + 2 * m0;
  int gy = 2 * n1 + 2 * m1;
  int wx = 2 * n0, wy = 2 * n1;
  long long perw = (long long)wy * wx;
  long long total = (long long)nky * perw;
  if (i >= total) return;
  int ky = (int)(i / perw);
  long long r = i - (long long)ky * perw;
  int iy = (int)(r / wx);
  int ix = (int)(r - (long long)iy * wx);
  float renorm = (float)(wx * wy);
  float s = lam_sign(m0 + ix) * lam_sign(m1 + iy) * renorm;
  long long fidx = (long long)ky * gx * gy + (long long)(m1 + iy) * gx + (m0 + ix);
  fdee[fidx] = lam_cscale(win[i], s);
}

// divker2d adj: deapodize -> f[n1,nky,n0] complex (note n1-ty-1 y-flip).
// The grid `fdee` is chunk-local `[nky,gy,gx]`; the output `f` may be a strided
// view of a larger `[n1, fdeth, n0]` buffer, written at absolute depth-freq
// `ky0+ky` (device-resident path); the host path passes `fdeth==nky, ky0==0`.
__global__ void lam_divker2d_adj_ker(const float2* fdee, float2* f, int n1, int nky,
                                     int n0, int m0, int m1, float mu0, float mu1,
                                     int fdeth, int ky0) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long per = (long long)nky * n0;
  long long total = (long long)n1 * per;
  if (i >= total) return;
  int ty = (int)(i / per);
  long long r = i - (long long)ty * per;
  int ky = (int)(r / n0);
  int tx = (int)(r - (long long)ky * n0);
  int gx = 2 * n0 + 2 * m0;
  int gy = 2 * n1 + 2 * m1;
  long long yg = (long long)(n1 - ty - 1) + n1 / 2 + m1;
  long long xg = (long long)tx + n0 / 2 + m0;
  float ker = expf(-mu0 * ((float)tx - (float)n0 / 2.0f) * ((float)tx - (float)n0 / 2.0f)
                   - mu1 * ((float)ty - (float)n1 / 2.0f) * ((float)ty - (float)n1 / 2.0f));
  long long gidx = (long long)ky * gx * gy + yg * gx + xg;
  float d = ker * (float)((long long)n0 * n1);
  long long fidx = (long long)ty * fdeth * n0 + (long long)(ky0 + ky) * n0 + tx;
  f[fidx] = lam_cscale(fdee[gidx], 1.0f / d);
}

int tomoxide_lam_takexy2d(const void* theta, void* xs, void* ys, long long ntheta,
                          int nky, int detw, float phi, int ky0, int deth, void* stream) {
  int block = 256;
  long long total = ntheta * (long long)nky * detw;
  float cph = cosf(phi);
  lam_takexy2d_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (const float*)theta, (float*)xs, (float*)ys, ntheta, nky, detw, cph, ky0, deth);
  return (int)cudaGetLastError();
}
int tomoxide_lam_gather2d_adj(const void* g, const void* xs, const void* ys, void* fdee,
                              long long ntheta, int nky, int detw, int n0, int n1,
                              int m0, int m1, float mu0, float mu1, int gdeth, int ky0,
                              void* stream) {
  int block = 256;
  long long total = ntheta * (long long)nky * detw;
  lam_gather2d_adj_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (const float2*)g, (const float*)xs, (const float*)ys, (float2*)fdee, ntheta, nky,
      detw, n0, n1, m0, m1, mu0, mu1, gdeth, ky0);
  return (int)cudaGetLastError();
}
int tomoxide_lam_wrap2d_adj(void* fdee, int nky, int n0, int n1, int m0, int m1, void* stream) {
  int block = 256;
  long long total = (long long)nky * (2 * n1 + 2 * m1) * (2 * n0 + 2 * m0);
  lam_wrap2d_adj_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (float2*)fdee, nky, n0, n1, m0, m1);
  return (int)cudaGetLastError();
}
int tomoxide_lam_win2d_extract(const void* fdee, void* win, int nky, int n0, int n1,
                               int m0, int m1, void* stream) {
  int block = 256;
  long long total = (long long)nky * (2 * n1) * (2 * n0);
  lam_win2d_extract_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (const float2*)fdee, (float2*)win, nky, n0, n1, m0, m1);
  return (int)cudaGetLastError();
}
int tomoxide_lam_win2d_scatter(void* fdee, const void* win, int nky, int n0, int n1,
                               int m0, int m1, void* stream) {
  int block = 256;
  long long total = (long long)nky * (2 * n1) * (2 * n0);
  lam_win2d_scatter_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (float2*)fdee, (const float2*)win, nky, n0, n1, m0, m1);
  return (int)cudaGetLastError();
}
int tomoxide_lam_divker2d_adj(const void* fdee, void* f, int n1, int nky, int n0,
                              int m0, int m1, float mu0, float mu1, int fdeth, int ky0,
                              void* stream) {
  int block = 256;
  long long total = (long long)n1 * nky * n0;
  lam_divker2d_adj_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (const float2*)fdee, (float2*)f, n1, nky, n0, m0, m1, mu0, mu1, fdeth, ky0);
  return (int)cudaGetLastError();
}

// ===========================================================================
// usfft1d_adj  (lamino.rs:279)
//   g [n1,deth,n0] complex -> f [n1,n2,n0] real.  n0=detw, n2=rh, ng=2n2+2m2.
//   batch = n0*n1 for the centered axis-0 FFT.
// ===========================================================================

// take_x (z positions): z[t] = (t - deth/2)/deth * sin(phi).
__global__ void lam_takez1d_ker(float* z, int deth, float sph) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= deth) return;
  z[i] = ((float)i - (float)deth / 2.0f) / (float)deth * sph;
}

// gather1d adj: scatter each g sample into the oversampled depth grid, atomicAdd.
__global__ void lam_gather1d_adj_ker(const float2* g, const float* z, float2* fdee,
                                     int n1, int deth, int n0, int n2, int m2, float mu2) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long per = (long long)deth * n0;
  long long total = (long long)n1 * per;
  if (i >= total) return;
  int ty = (int)(i / per);
  long long r = i - (long long)ty * per;
  int tz = (int)(r / n0);
  int tx = (int)(r - (long long)tz * n0);
  // g index == i (tx + tz*n0 + ty*n0*deth).
  float2 g0 = g[i];
  float z0 = z[tz];
  float wscale = sqrtf(LAM_PI / (mu2 * (float)n0));
  long long base = (long long)floorf(2.0f * (float)n2 * z0) - m2;
  long long stride_a = (long long)n0 * n1;
  long long col = (long long)ty * n0 + tx;
  for (int i2 = 0; i2 < 2 * m2 + 1; i2++) {
    long long ell2 = base + i2;
    float w2 = (float)ell2 / (2.0f * (float)n2) - z0;
    float w = wscale * expf(-LAM_PI * LAM_PI / mu2 * w2 * w2);
    long long a = (long long)n2 + m2 + ell2;
    lam_catomic(&fdee[a * stride_a + col], lam_cscale(g0, w));
  }
}

// wrap1d adj: fold the m2-wide borders back into the interior along the depth grid.
__global__ void lam_wrap1d_adj_ker(float2* fdee, int n0, int n1, int n2, int m2) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int ng = 2 * n2 + 2 * m2;
  long long per = (long long)n1 * n0;
  long long total = (long long)ng * per;
  if (i >= total) return;
  int a = (int)(i / per);
  long long col = i - (long long)a * per;  // ty*n0 + tx
  if (a < m2 || a >= 2 * n2 + m2) {
    int a0 = (a + 2 * n2 - m2) % (2 * n2);
    long long dst = (long long)(a0 + m2) * per + col;
    lam_catomic(&fdee[dst], fdee[i]);
  }
}

// centered inverse axis-0 FFT window extract: lines[b*len + ii] = fdee[(m2+ii)*batch+b]*sign.
__global__ void lam_win1d_extract_ker(const float2* fdee, float2* lines, long long batch,
                                      int n2, int m2) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int len = 2 * n2;
  long long total = batch * len;
  if (i >= total) return;
  long long b = i / len;
  int ii = (int)(i - b * len);
  float s = lam_sign(m2 + ii);
  lines[i] = lam_cscale(fdee[(long long)(m2 + ii) * batch + b], s);
}

// scatter window back: fdee[(m2+ii)*batch+b] = lines[b*len+ii]*sign*renorm (renorm=len).
__global__ void lam_win1d_scatter_ker(float2* fdee, const float2* lines, long long batch,
                                      int n2, int m2) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int len = 2 * n2;
  long long total = batch * len;
  if (i >= total) return;
  long long b = i / len;
  int ii = (int)(i - b * len);
  float s = lam_sign(m2 + ii) * (float)len;
  fdee[(long long)(m2 + ii) * batch + b] = lam_cscale(lines[i], s);
}

// divker1d adj: deapodize, take real part -> f[n1,n2,n0] real.
__global__ void lam_divker1d_adj_ker(const float2* fdee, float* f, int n1, int n2,
                                     int n0, int m2, float mu2) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long per = (long long)n2 * n0;
  long long total = (long long)n1 * per;
  if (i >= total) return;
  int ty = (int)(i / per);
  long long r = i - (long long)ty * per;
  int tz = (int)(r / n0);
  int tx = (int)(r - (long long)tz * n0);
  float ker = expf(-mu2 * ((float)tz - (float)n2 / 2.0f) * ((float)tz - (float)n2 / 2.0f));
  long long a = (long long)tz + n2 / 2 + m2;
  long long gidx = a * (long long)n0 * n1 + (long long)ty * n0 + tx;
  f[i] = fdee[gidx].x / ker / (2.0f * (float)n2);
}

int tomoxide_lam_takez1d(void* z, int deth, float phi, void* stream) {
  int block = 256;
  float sph = sinf(phi);
  lam_takez1d_ker<<<lam_grid(deth, block), block, 0, (cudaStream_t)stream>>>(
      (float*)z, deth, sph);
  return (int)cudaGetLastError();
}
int tomoxide_lam_gather1d_adj(const void* g, const void* z, void* fdee, int n1, int deth,
                              int n0, int n2, int m2, float mu2, void* stream) {
  int block = 256;
  long long total = (long long)n1 * deth * n0;
  lam_gather1d_adj_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (const float2*)g, (const float*)z, (float2*)fdee, n1, deth, n0, n2, m2, mu2);
  return (int)cudaGetLastError();
}
int tomoxide_lam_wrap1d_adj(void* fdee, int n0, int n1, int n2, int m2, void* stream) {
  int block = 256;
  long long total = (long long)(2 * n2 + 2 * m2) * n1 * n0;
  lam_wrap1d_adj_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (float2*)fdee, n0, n1, n2, m2);
  return (int)cudaGetLastError();
}
int tomoxide_lam_win1d_extract(const void* fdee, void* lines, long long batch, int n2,
                               int m2, void* stream) {
  int block = 256;
  long long total = batch * (long long)(2 * n2);
  lam_win1d_extract_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (const float2*)fdee, (float2*)lines, batch, n2, m2);
  return (int)cudaGetLastError();
}
int tomoxide_lam_win1d_scatter(void* fdee, const void* lines, long long batch, int n2,
                               int m2, void* stream) {
  int block = 256;
  long long total = batch * (long long)(2 * n2);
  lam_win1d_scatter_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (float2*)fdee, (const float2*)lines, batch, n2, m2);
  return (int)cudaGetLastError();
}
int tomoxide_lam_divker1d_adj(const void* fdee, void* f, int n1, int n2, int n0, int m2,
                              float mu2, void* stream) {
  int block = 256;
  long long total = (long long)n1 * n2 * n0;
  lam_divker1d_adj_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (const float2*)fdee, (float*)f, n1, n2, n0, m2, mu2);
  return (int)cudaGetLastError();
}

// ===========================================================================
// copyTransposed  (lamino.rs:761) — p00 [n1,rh,n2] -> vol [rh,n1,n2].
// ===========================================================================
__global__ void lam_transpose_ker(const float* p00, float* vol, int n1, int rh, int n2) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long per = (long long)n1 * n2;
  long long total = (long long)rh * per;
  if (i >= total) return;
  int tz = (int)(i / per);
  long long r = i - (long long)tz * per;
  int ty = (int)(r / n2);
  int tx = (int)(r - (long long)ty * n2);
  vol[i] = p00[((long long)ty * rh + tz) * n2 + tx];
}

int tomoxide_lam_transpose(const void* p00, void* vol, int n1, int rh, int n2, void* stream) {
  int block = 256;
  long long total = (long long)rh * n1 * n2;
  lam_transpose_ker<<<lam_grid(total, block), block, 0, (cudaStream_t)stream>>>(
      (const float*)p00, (float*)vol, n1, rh, n2);
  return (int)cudaGetLastError();
}

}  // extern "C"
