// Tap geometry shared by the back-projector (kernels_linerec.cuh) and its
// transpose (forwardproj.cu). Both compute every tap through these helpers,
// with the roundings spelled out (__fmaf_rn), so a voxel's taps and weights
// are bit-identical in A and Aᵀ whatever the compiler would otherwise contract
// — the pair stays an exact adjoint.
#ifndef PROJGEOM_CUH
#define PROJGEOM_CUH

// The tilt every parallel-beam caller passes (Rust `FRAC_PI_2`). At this tilt
// the detector row of slice tz is the integer r = tz + sz - rh/2 + nz/2, so its
// row taps are r with weight 1 and r + 1 with weight 0: the parallel kernels
// read row r alone, and a slice contributes iff tap_in(r, nz).
static inline __host__ __device__ bool is_parallel_tilt(float phi)
{
    return phi == 1.57079632679489661923f;
}

// Slices one thread of a parallel kernel covers: the column tap depends on
// (x, y, θ) only, so it is computed once and reused for every slice.
#define PAR_G 8

// cos θ and sin θ of a projection angle from the accurate sincosf, as the CPU
// projectors' sin_cos: __cosf/__sinf are ~1e-6 off (more past ±π), a ~2e-4 px
// tap shift at a 1024² corner that SIRT's 1/A(1) amplifies in sparse edge bins.
static __device__ __forceinline__ void theta_cs(float t, float *c, float *s)
{
    sincosf(t, s, c);
}

// Detector column of voxel (x, y) at angle (c, s) = (cos θ, sin θ), h = n/2;
// axis is the column voxel column h lands on (the rotation centre).
static __device__ __forceinline__ float proj_u(float c, float s, int x, int y, int h, float axis)
{
    return __fmaf_rn(c, (float)(x - h), __fmaf_rn(s, (float)(y - h), axis));
}

// Whether sample i of n exists.
static __device__ __forceinline__ bool tap_in(int i, int n)
{
    return (unsigned)i < (unsigned)n;
}

// The linear taps of detector coordinate t over samples [0, n): sample i = ⌊t⌋
// with weight 1 - f and sample i + 1 with weight f. A sample off the detector is
// zero, so lo/hi say which of the two exist and an absent one is never read or
// written. The CPU and wgpu projectors (`backend::linear_taps`) use the same rule.
struct Tap
{
    int i;
    float f;
    bool lo, hi;
};
static __device__ __forceinline__ Tap proj_tap(float t, int n)
{
    int i = __float2int_rd(t);
    return {i, __fsub_rn(t, (float)i), tap_in(i, n), tap_in(i + 1, n)};
}
static __device__ __forceinline__ bool tap_any(Tap k)
{
    return k.lo | k.hi;
}

// Bilinear sample at column taps cu and row taps rv, q = &stack[rv.i][..][cu.i]
// with row stride zs; a tap off the detector reads 0 and is not loaded. The
// reconstruction, both probes and the frame-dot kernels sample through this one
// helper with the roundings spelled out, so a probed slice is bit-identical to
// the same slice of the reconstruction.
template <typename T>
static __device__ __forceinline__ float lam_bilerp(const T *q, size_t zs, Tap cu, Tap rv)
{
    float q00 = cu.lo & rv.lo ? (float)q[0] : 0.f;
    float q01 = cu.hi & rv.lo ? (float)q[1] : 0.f;
    float q10 = cu.lo & rv.hi ? (float)q[zs] : 0.f;
    float q11 = cu.hi & rv.hi ? (float)q[zs + 1] : 0.f;
    float a = __fmaf_rn(q01, cu.f, __fmul_rn(q00, 1.f - cu.f));
    float b = __fmaf_rn(q11, cu.f, __fmul_rn(q10, 1.f - cu.f));
    return __fmaf_rn(b, rv.f, __fmul_rn(a, 1.f - rv.f));
}

// Tilted detector row v of voxel (x, y) in slice zz = tz + sz - rh/2, split so
// the (x, y) part lam_vxy is computed once for every slice of a column. cphi and
// sphi are __cosf/__sinf(phi); at the parallel tilt __cosf gives exactly 0, so
// v is the integer zz + nz/2.
static __device__ __forceinline__ float lam_vxy(float c, float s, float cphi, int x, int y, int h)
{
    return __fmaf_rn(__fmul_rn(s, cphi), (float)(x - h), __fmul_rn(__fmul_rn(-c, cphi), (float)(y - h)));
}
static __device__ __forceinline__ float lam_v(float vxy, float sphi, int zz, int nz)
{
    return __fadd_rn(__fmaf_rn(sphi, (float)zz, vxy), (float)(nz / 2));
}

#endif
