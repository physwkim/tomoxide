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
//
// Work areas are SHARED across a thread's plans, not owned per plan. A cached,
// never-destroyed plan that owns its scratch makes the cache's device-memory
// cost the SUM over plans, which is unaffordable once cuFFT falls back to
// Bluestein: any transform length with a large prime factor (laminography's
// `2*rh` routinely does — rh = ceil(nz/cos(tilt)/2)*2) needs scratch on the
// order of the data itself, so a single plan can want gigabytes while a
// power-of-two plan of the same batch wants none. Keying the cache on `batch`
// then makes a merely ragged tail chunk mint a second full-size plan and double
// that cost. So auto-allocation is disabled and every plan is pointed at one
// per-thread buffer grown to the largest plan seen: the cache costs
// max-over-plans instead of sum-over-plans, and duplicate-shaped plans cost only
// their handle. Sharing is safe because all of a thread's plans execute on that
// thread's `cudaStreamPerThread` and are therefore serialized, and the buffer is
// thread-local so no other thread can observe it.

#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
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

// The cuFFT call shape a PlanKey denotes, written down once so that estimating a
// plan's scratch, creating it, and stepping the data pointer between sub-batches
// cannot drift apart.
struct PlanDesc {
  int rank;
  int dims[2];
  int inembed[2], onembed[2];
  int *ip, *op;  // null ⇒ cuFFT's default (contiguous) layout
  int idist, odist;
  cufftType type;
  size_t item_bytes;  // device stride between consecutive transforms
};

PlanDesc describe(const PlanKey& k) {
  PlanDesc d{};
  if (k.type == 1 || k.type == 2) {
    // In-place real transform of `batch` images [nrho=n0, ntheta=n1]. The real
    // data lives in a row-padded buffer [nrho, ntheta+2] so it overlays the
    // half-complex spectrum [nrho, ntheta/2+1] (cuFFT's in-place R2C/C2R
    // layout). idist (reals) == odist (complex) * 2, so both views are
    // contiguous over the batch.
    int nc = k.n1 / 2 + 1;    // complex half-width
    int npad = 2 * nc;        // padded real width
    int rdist = k.n0 * npad;  // reals per image
    int cdist = k.n0 * nc;    // complex per image
    d.rank = 2;
    d.dims[0] = k.n0;
    d.dims[1] = k.n1;
    d.inembed[0] = d.onembed[0] = k.n0;
    d.ip = d.inembed;
    d.op = d.onembed;
    d.item_bytes = (size_t) rdist * sizeof(float);
    if (k.type == 1) {
      d.inembed[1] = npad;
      d.onembed[1] = nc;
      d.idist = rdist;
      d.odist = cdist;
      d.type = CUFFT_R2C;
    } else {
      d.inembed[1] = nc;
      d.onembed[1] = npad;
      d.idist = cdist;
      d.odist = rdist;
      d.type = CUFFT_C2R;
    }
  } else if (k.rank == 1) {
    d.rank = 1;
    d.dims[0] = k.n0;
    d.idist = d.odist = k.n0;
    d.type = CUFFT_C2C;
    d.item_bytes = (size_t) k.n0 * sizeof(cufftComplex);
  } else {
    d.rank = 2;
    d.dims[0] = k.n0;
    d.dims[1] = k.n1;
    d.idist = d.odist = k.n0 * k.n1;
    d.type = CUFFT_C2C;
    d.item_bytes = (size_t) k.n0 * k.n1 * sizeof(cufftComplex);
  }
  return d;
}

// Ceiling on the device scratch a single plan may hold. cuFFT needs no scratch
// for a 2/3/5/7-smooth length, but falls back to Bluestein for any other — and
// Bluestein's scratch scales with `batch`, reaching several times the data
// itself. Left uncapped, the scratch therefore scales with whatever chunk size
// the caller picked from its own VRAM budget, which is how a laminography
// `2*rh` of 5988 (= 2²·3·499) came to ask for 8.8 GiB. Capping is free of
// approximation: consecutive transforms are contiguous (idist == odist == the
// transform size), so running sub-batch k at `data + k*sub*item_bytes` performs
// exactly the transforms the one big call would have.
//
// `TOMOXIDE_CUFFT_MAX_WORK_MB` overrides the default, both to lower it on a
// card whose free VRAM is tighter than this assumes and so the split path can be
// exercised at test scale (a cap in MiB forces a split without allocating the
// gigabytes it would otherwise take to reach one). Read per call rather than
// cached: `split_batch` runs once per transform, against which a `getenv` is
// nothing, and a cached first-touch value would depend on which test ran first.
size_t max_plan_work_bytes() {
  if (const char* s = std::getenv("TOMOXIDE_CUFFT_MAX_WORK_MB")) {
    char* end = nullptr;
    unsigned long long mb = std::strtoull(s, &end, 10);
    if (end != s && mb > 0) return (size_t) mb << 20;
  }
  return 512ull << 20;
}

// Largest sub-batch of `batch` whose plan scratch fits the cap. Returns `batch`
// unchanged whenever it already fits — which is every smooth length, so the
// non-Bluestein consumers keep their single-call behaviour.
int split_batch(const PlanKey& key, int batch) {
  const size_t cap = max_plan_work_bytes();
  PlanDesc d = describe(key);
  int sub = batch;
  while (sub > 1) {
    size_t ws = 0;
    if (cufftEstimateMany(d.rank, d.dims, d.ip, 1, d.idist, d.op, 1, d.odist,
                          d.type, sub, &ws) != CUFFT_SUCCESS)
      break;  // no estimate ⇒ let plan creation report the real error
    if (ws <= cap) break;
    sub = (sub + 1) / 2;
  }
  return sub;
}

// Per worker thread (each pinned to one GPU): its own plans, bound to its
// per-thread default stream. No locking; never destroyed (see file header).
thread_local std::unordered_map<PlanKey, cufftHandle, PlanKeyHash> g_plans;

// The one work area every plan in `g_plans` points at, grown to the largest
// plan this thread has needed (see file header). Never freed, like the plans.
thread_local void* g_work = nullptr;
thread_local size_t g_work_bytes = 0;

// Grow the shared work area to `need` bytes and re-point every cached plan at
// the new buffer. Returns 0 on success, -1 if the allocation fails.
int grow_work_area(size_t need) {
  if (need <= g_work_bytes) return 0;
  void* p = nullptr;
  if (cudaMalloc(&p, need) != cudaSuccess) {
    cudaGetLastError();  // clear the sticky-free OOM so later calls see a clean slate
    return -1;
  }
  // Plans already enqueued on this thread's stream may still be reading the old
  // buffer; drain before releasing it. Growth is rare, so the sync is not hot.
  if (g_work) {
    cudaStreamSynchronize(cudaStreamPerThread);
    cudaFree(g_work);
  }
  g_work = p;
  g_work_bytes = need;
  for (auto& kv : g_plans) {
    if (cufftSetWorkArea(kv.second, g_work) != CUFFT_SUCCESS) return -1;
  }
  return 0;
}

// 0 on success (sets *out), -1 on plan creation failure.
int get_plan(const PlanKey& key, cufftHandle* out) {
  auto it = g_plans.find(key);
  if (it != g_plans.end()) {
    *out = it->second;
    return 0;
  }
  cufftHandle plan;
  if (cufftCreate(&plan) != CUFFT_SUCCESS) return -1;
  // Must precede cufftMakePlanMany: the plan's scratch is this thread's shared
  // buffer, bound below, so cuFFT must not allocate one of its own.
  if (cufftSetAutoAllocation(plan, 0) != CUFFT_SUCCESS) {
    cufftDestroy(plan);
    return -1;
  }
  PlanDesc d = describe(key);
  size_t need = 0;
  if (cufftMakePlanMany(plan, d.rank, d.dims, d.ip, 1, d.idist, d.op, 1, d.odist,
                        d.type, key.batch, &need) != CUFFT_SUCCESS) {
    cufftDestroy(plan);
    return -1;
  }
  // `need == 0` (the smooth-length Cooley-Tukey path) wants no scratch at all;
  // leave such a plan unbound rather than pointing it at a null buffer.
  if (need > 0 && (grow_work_area(need) != 0 ||
                   cufftSetWorkArea(plan, g_work) != CUFFT_SUCCESS)) {
    cufftDestroy(plan);
    return -1;
  }
  cufftSetStream(plan, cudaStreamPerThread);
  g_plans.emplace(key, plan);
  *out = plan;
  return 0;
}

// Enqueue `batch` in-place transforms of shape `key` over `data` on this
// thread's stream, in sub-batches small enough to keep each plan's scratch under
// the cap. `inverse` applies to C2C only (R2C/C2R carry their direction in the
// type). Returns 0, -1 if a plan cannot be made, -2 if a transform fails.
int exec_batched(PlanKey key, void* data, int batch, int inverse) {
  const size_t item_bytes = describe(key).item_bytes;
  int sub = split_batch(key, batch);
  for (int off = 0; off < batch;) {
    const int b = sub < batch - off ? sub : batch - off;
    key.batch = b;
    cufftHandle plan;
    if (get_plan(key, &plan) != 0) {
      // `split_batch` only consulted cufftEstimateMany, which cuFFT documents as
      // approximate; plan creation is the authority. Halve and retry so the cap
      // holds against what cuFFT actually wants, not what it predicted.
      if (b > 1) {
        sub = (b + 1) / 2;
        continue;
      }
      return -1;
    }
    char* p = (char*) data + (size_t) off * item_bytes;
    cufftResult r;
    if (key.type == 1) {
      r = cufftExecR2C(plan, (cufftReal*) p, (cufftComplex*) p);
    } else if (key.type == 2) {
      r = cufftExecC2R(plan, (cufftComplex*) p, (cufftReal*) p);
    } else {
      r = cufftExecC2C(plan, (cufftComplex*) p, (cufftComplex*) p,
                       inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
    }
    if (r != CUFFT_SUCCESS) return -2;
    off += b;
  }
  return 0;
}

}  // namespace

extern "C" {

// The sub-batch the `tomoxide_fft_*` entry points would run this shape in — an
// exact `batch` when the scratch already fits, smaller when it must be split.
// A pure query (creates nothing), exposed so the split policy can be asserted
// rather than inferred from results that would match either way.
int tomoxide_fft_plan_split(int rank, int n0, int n1, int batch, int type) {
  PlanKey key{rank, n0, n1, batch, type};
  return split_batch(key, batch);
}

// In-place batched 1-D C2C FFT of `batch` transforms of length `n`.
// Returns 0 on success.
int tomoxide_fft_1d(void* data, size_t n, size_t batch, int inverse) {
  PlanKey key{1, (int) n, 0, (int) batch, 0};
  int rc = exec_batched(key, data, (int) batch, inverse);
  if (rc != 0) return rc;
  if (inverse) {
    long long cnt = 2ll * (long long) n * (long long) batch;
    run_scale(data, cnt, 1.0f / (float) n);
  }
  return (int) cudaStreamSynchronize(cudaStreamPerThread);
}

// In-place batched 2-D C2C FFT of `batch` images of size `rows × cols`.
int tomoxide_fft_2d(void* data, size_t rows, size_t cols, size_t batch, int inverse) {
  PlanKey key{2, (int) rows, (int) cols, (int) batch, 0};
  int rc = exec_batched(key, data, (int) batch, inverse);
  if (rc != 0) return rc;
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
  int rc = exec_batched(key, data, (int) batch, 0);
  if (rc != 0) return rc;
  return (int) cudaStreamSynchronize(cudaStreamPerThread);
}

// In-place batched 2-D **C2R** FFT (inverse) of `batch` half-complex spectra
// [rows × (cols/2+1)] back to row-padded real images [rows × cols]. Normalized
// by 1/(rows*cols) to match `ifft(fft(x)) == x` (cuFFT leaves it unnormalized),
// scaling only the `rows*(cols+2 rounded)` real floats per image in place.
int tomoxide_fft_2d_c2r(void* data, size_t rows, size_t cols, size_t batch) {
  PlanKey key{2, (int) rows, (int) cols, (int) batch, 2};
  int rc = exec_batched(key, data, (int) batch, 0);
  if (rc != 0) return rc;
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
  int rc = exec_batched(key, data, (int) batch, inverse);
  if (rc != 0) return rc;
  if (inverse) {
    long long cnt = 2ll * (long long) n * (long long) batch;
    run_scale(data, cnt, 1.0f / (float) n);
  }
  return (int) cudaGetLastError();
}

int tomoxide_fft_2d_async(void* data, size_t rows, size_t cols, size_t batch, int inverse) {
  PlanKey key{2, (int) rows, (int) cols, (int) batch, 0};
  int rc = exec_batched(key, data, (int) batch, inverse);
  if (rc != 0) return rc;
  if (inverse) {
    long long cnt = 2ll * (long long) rows * (long long) cols * (long long) batch;
    run_scale(data, cnt, 1.0f / (float) (rows * cols));
  }
  return (int) cudaGetLastError();
}

}  // extern "C"
