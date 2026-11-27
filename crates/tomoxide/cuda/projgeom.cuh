// Tap geometry shared by the back-projector (kernels_linerec.cuh) and its
// transpose (forwardproj.cu). Both compute every tap through these helpers,
// with the roundings spelled out (__fmaf_rn), so a voxel's taps and weights
// are bit-identical in A and Aᵀ whatever the compiler would otherwise contract
// — the pair stays an exact adjoint.
#ifndef PROJGEOM_CUH
#define PROJGEOM_CUH

// The tilt every parallel-beam caller passes (Rust `FRAC_PI_2`). At this tilt
// the detector row of slice tz is the integer tz + sz - rh/2 + nz/2, so the
// parallel kernels read that one row instead of interpolating between two.
static inline __host__ __device__ bool is_parallel_tilt(float phi)
{
    return phi == 1.57079632679489661923f;
}

// Slices one thread of a parallel kernel covers: the column tap depends on
// (x, y, θ) only, so it is computed once and reused for every slice.
#define PAR_G 8

// Detector column of voxel (x, y) at angle (c, s) = (cos θ, sin θ), h = n/2;
// axis is the column voxel column h lands on (the rotation centre).
static __device__ __forceinline__ float proj_u(float c, float s, int x, int y, int h, float axis)
{
    return __fmaf_rn(c, (float)(x - h), __fmaf_rn(s, (float)(y - h), axis));
}

// tomocupy's tap: truncation toward zero, so u in (-1, 0] also taps column 0.
static __device__ __forceinline__ int proj_tap(float u)
{
    return (int)(u - 1e-5f);
}

// Whether a parallel-beam slice landing on detector row r contributes. This is
// the tilted kernel's row guard evaluated at v = r (`vr = (int)(v-1e-5f)`,
// `v > -1e-5f`, `vr < nz-1`), so the parallel kernels are the same operator.
static __device__ __forceinline__ bool par_row_ok(int r, int nz)
{
    int vr = (int)((float)r - 1e-5f);
    return ((float)r > -1e-5f) & (vr < nz - 1);
}

#endif
