// Batched C2C FFT via cuFFT, exposing the tomoxide `Fft` capability on the GPU.
//
// Implementing this one capability lets every Fft-composing method
// (`gridrec`, `fourierrec`, `lprec`, Paganin/GPaganin/Farago phase, the
// Fourier-wavelet stripe filter) run on CUDA through the existing backend-
// agnostic code — the same way they compose onto wgpu. cuFFT leaves the inverse
// unnormalized, so we divide by the transform size to match tomoxide's
// convention (`ifft(fft(x)) == x`), as the CPU/wgpu backends do.
//
// Data crosses as device `cufftComplex*` (== interleaved float2, layout-
// compatible with tomoxide's `Complex32`). Compiled by build.rs (nvcc, with
// `--default-stream per-thread`) and linked against cuFFT.
//
// Concurrency: `Fft::for_each_slice` fans the per-slice recon loop across host
// threads (one device-pinned pool per selected GPU). Two structural changes make
// that fast and safe:
//   * a THREAD-LOCAL cuFFT plan cache keyed by (rank, dims, batch) — plans are
//     created once per worker thread and reused, instead of `cufftPlanMany` +
//     `cufftDestroy` on every call (the dominant per-slice cost). Thread-local
//     ⇒ no locking and each plan lives on the device its worker is pinned to.
//   * every transform binds its plan and scale kernel to `cudaStreamPerThread`
//     and syncs only that stream, so concurrent workers overlap instead of
//     serializing on the legacy null stream / a device-wide sync.
// Plans are intentionally never `cufftDestroy`d: doing so at thread/static
// teardown can run after the CUDA context is gone. They are reclaimed when the
// process exits.

#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <unordered_map>

__global__ void cscale_ker(float* d, long long n, float f) {
  long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  d[i] *= f;
}

static int run_scale(void* data, long long nfloat, float f) {
  int block = 256;
  int grid = (int) ((nfloat + block - 1) / block);
  cscale_ker<<<grid, block, 0, cudaStreamPerThread>>>((float*) data, nfloat, f);
  return (int) cudaGetLastError();
}

namespace {

// `type`: 0 = C2C, 1 = R2C (in-place, padded), 2 = C2R (in-place, padded).
struct PlanKey {
  int rank, n0, n1, batch, type;
  bool operator==(const PlanKey& o) const {
    return rank == o.rank && n0 == o.n0 && n1 == o.n1 && batch == o.batch &&
           type == o.type;
  }
};

struct PlanKeyHash {
  size_t operator()(const PlanKey& k) const {
    size_t h = 1469598103934665603ull;  // FNV-1a over the five fields
    int vs[5] = {k.rank, k.n0, k.n1, k.batch, k.type};
    for (int v : vs) {
      h ^= (size_t) (uint32_t) v;
      h *= 1099511628211ull;
    }
    return h;
  }
};

// Per worker thread (each pinned to one GPU): its own plans, bound to its
// per-thread default stream. No locking; never destroyed (see file header).
thread_local std::unordered_map<PlanKey, cufftHandle, PlanKeyHash> g_plans;

// 0 on success (sets *out), -1 on plan creation failure.
int get_plan(const PlanKey& key, cufftHandle* out) {
  auto it = g_plans.find(key);
  if (it != g_plans.end()) {
    *out = it->second;
    return 0;
  }
  cufftHandle plan;
  cufftResult r;
  if (key.type == 1 || key.type == 2) {
    // In-place real transform of `batch` images [nrho=n0, ntheta=n1]. The real
    // data lives in a row-padded buffer [nrho, ntheta+2] so it overlays the
    // half-complex spectrum [nrho, ntheta/2+1] (cuFFT's in-place R2C/C2R
    // layout). idist (reals) == odist (complex) * 2, so both views are
    // contiguous over the batch.
    int dims[2] = {key.n0, key.n1};
    int nc = key.n1 / 2 + 1;        // complex half-width
    int npad = 2 * nc;             // padded real width
    int rembed[2] = {key.n0, npad};
    int cembed[2] = {key.n0, nc};
    int rdist = key.n0 * npad;     // reals per image
    int cdist = key.n0 * nc;       // complex per image
    if (key.type == 1) {
      r = cufftPlanMany(&plan, 2, dims, rembed, 1, rdist, cembed, 1, cdist,
                        CUFFT_R2C, key.batch);
    } else {
      r = cufftPlanMany(&plan, 2, dims, cembed, 1, cdist, rembed, 1, rdist,
                        CUFFT_C2R, key.batch);
    }
  } else if (key.rank == 1) {
    int dims[1] = {key.n0};
    r = cufftPlanMany(&plan, 1, dims, nullptr, 1, key.n0, nullptr, 1, key.n0, CUFFT_C2C, key.batch);
  } else {
    int dims[2] = {key.n0, key.n1};
    int stride = key.n0 * key.n1;
    r = cufftPlanMany(&plan, 2, dims, nullptr, 1, stride, nullptr, 1, stride, CUFFT_C2C, key.batch);
  }
  if (r != CUFFT_SUCCESS) return -1;
  cufftSetStream(plan, cudaStreamPerThread);
  g_plans.emplace(key, plan);
  *out = plan;
  return 0;
}

}  // namespace

extern "C" {

// In-place batched 1-D C2C FFT of `batch` transforms of length `n`.
// Returns 0 on success.
int tomoxide_fft_1d(void* data, size_t n, size_t batch, int inverse) {
  PlanKey key{1, (int) n, 0, (int) batch, 0};
  cufftHandle plan;
  if (get_plan(key, &plan) != 0) return -1;
  if (cufftExecC2C(plan, (cufftComplex*) data, (cufftComplex*) data,
                   inverse ? CUFFT_INVERSE : CUFFT_FORWARD) != CUFFT_SUCCESS)
    return -2;
  if (inverse) {
    long long cnt = 2ll * (long long) n * (long long) batch;
    run_scale(data, cnt, 1.0f / (float) n);
  }
  return (int) cudaStreamSynchronize(cudaStreamPerThread);
}

// In-place batched 2-D C2C FFT of `batch` images of size `rows × cols`.
int tomoxide_fft_2d(void* data, size_t rows, size_t cols, size_t batch, int inverse) {
  PlanKey key{2, (int) rows, (int) cols, (int) batch, 0};
  cufftHandle plan;
  if (get_plan(key, &plan) != 0) return -1;
  if (cufftExecC2C(plan, (cufftComplex*) data, (cufftComplex*) data,
                   inverse ? CUFFT_INVERSE : CUFFT_FORWARD) != CUFFT_SUCCESS)
    return -2;
  if (inverse) {
    long long cnt = 2ll * (long long) rows * (long long) cols * (long long) batch;
    run_scale(data, cnt, 1.0f / (float) (rows * cols));
  }
  return (int) cudaStreamSynchronize(cudaStreamPerThread);
}

// In-place batched 2-D **R2C** FFT (forward) of `batch` real images
// [rows × cols], laid out row-padded to [rows × (cols+2 rounded)] so the real
// input overlays the half-complex output [rows × (cols/2+1)]. `data` is the
// padded real buffer; on return it holds the half-complex spectrum. Forward is
// unnormalized (the C2R inverse carries the 1/(rows*cols) scale), matching the
// C2C convention. Returns 0 on success.
int tomoxide_fft_2d_r2c(void* data, size_t rows, size_t cols, size_t batch) {
  PlanKey key{2, (int) rows, (int) cols, (int) batch, 1};
  cufftHandle plan;
  if (get_plan(key, &plan) != 0) return -1;
  if (cufftExecR2C(plan, (cufftReal*) data, (cufftComplex*) data) != CUFFT_SUCCESS)
    return -2;
  return (int) cudaStreamSynchronize(cudaStreamPerThread);
}

// In-place batched 2-D **C2R** FFT (inverse) of `batch` half-complex spectra
// [rows × (cols/2+1)] back to row-padded real images [rows × cols]. Normalized
// by 1/(rows*cols) to match `ifft(fft(x)) == x` (cuFFT leaves it unnormalized),
// scaling only the `rows*(cols+2 rounded)` real floats per image in place.
int tomoxide_fft_2d_c2r(void* data, size_t rows, size_t cols, size_t batch) {
  PlanKey key{2, (int) rows, (int) cols, (int) batch, 2};
  cufftHandle plan;
  if (get_plan(key, &plan) != 0) return -1;
  if (cufftExecC2R(plan, (cufftComplex*) data, (cufftReal*) data) != CUFFT_SUCCESS)
    return -2;
  long long npad = 2ll * ((long long) cols / 2 + 1);  // padded real width
  long long cnt = (long long) rows * npad * (long long) batch;
  run_scale(data, cnt, 1.0f / (float) (rows * cols));
  return (int) cudaStreamSynchronize(cudaStreamPerThread);
}

// ---- async (non-syncing) C2C variants ----
// Identical to `tomoxide_fft_1d`/`tomoxide_fft_2d` but they only *enqueue* the
// transform (+ inverse scale) on `cudaStreamPerThread` and return without a host
// sync. The caller is then responsible for ordering: because the laminography
// stage kernels also run on the null (== per-thread) stream, the FFT stays
// correctly serialized with them on the device, while the host thread is free to
// run the CPU gather/scatter concurrently — the overlap the host-syncing variants
// above cannot give. Only C2C is exposed here (the lamino path uses 1-D and 2-D
// C2C exclusively); the R2C/C2R and every other `Fft` consumer keep the syncing
// variants, so their "result ready on return" contract is unchanged.
int tomoxide_fft_1d_async(void* data, size_t n, size_t batch, int inverse) {
  PlanKey key{1, (int) n, 0, (int) batch, 0};
  cufftHandle plan;
  if (get_plan(key, &plan) != 0) return -1;
  if (cufftExecC2C(plan, (cufftComplex*) data, (cufftComplex*) data,
                   inverse ? CUFFT_INVERSE : CUFFT_FORWARD) != CUFFT_SUCCESS)
    return -2;
  if (inverse) {
    long long cnt = 2ll * (long long) n * (long long) batch;
    run_scale(data, cnt, 1.0f / (float) n);
  }
  return (int) cudaGetLastError();
}

int tomoxide_fft_2d_async(void* data, size_t rows, size_t cols, size_t batch, int inverse) {
  PlanKey key{2, (int) rows, (int) cols, (int) batch, 0};
  cufftHandle plan;
  if (get_plan(key, &plan) != 0) return -1;
  if (cufftExecC2C(plan, (cufftComplex*) data, (cufftComplex*) data,
                   inverse ? CUFFT_INVERSE : CUFFT_FORWARD) != CUFFT_SUCCESS)
    return -2;
  if (inverse) {
    long long cnt = 2ll * (long long) rows * (long long) cols * (long long) batch;
    run_scale(data, cnt, 1.0f / (float) (rows * cols));
  }
  return (int) cudaGetLastError();
}

}  // extern "C"
