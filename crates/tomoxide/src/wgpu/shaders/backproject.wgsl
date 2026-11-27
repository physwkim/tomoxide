// Parallel-beam voxel-driven back-projection — port of the CpuBackend
// FilteredBackproject impl (tomopy `libtomo/recon/fbp.c`).
//
// One thread per output voxel (flat index over [nz, ny, nx]) accumulates the
// already-filtered sinogram along all angles by linear interpolation, then
// scales by the caller's `scale` gain: π/nproj for the analytic FBP paths (the
// angular-quadrature dθ weight) and 1.0 for the iterative solvers (the pure
// adjoint of project.wgsl's unweighted scatter). `cossin` carries the per-angle
// (cosθ, sinθ) interleaved and `center` the per-row rotation-axis position —
// both computed host-side so the trig matches the CPU reference exactly; the
// GPU/CPU drift is only in the multiply-accumulate rounding, hence callers
// compare with a tolerance.

struct Params {
    nproj : u32,
    ncols : u32,
    ny    : u32,
    nx    : u32,
    scale : f32, // caller gain: π/nproj (analytic FBP) or 1.0 (iterative adjoint)
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(0) var<storage, read>       sino   : array<f32>; // [nz, nproj, ncols]
@group(0) @binding(1) var<storage, read>       cossin : array<f32>; // [nproj*2] (c, sn)
@group(0) @binding(2) var<storage, read>       center : array<f32>; // [nz]
@group(0) @binding(3) var<storage, read_write> vol    : array<f32>; // [nz, ny, nx]
@group(0) @binding(4) var<uniform>             params : Params;

@compute @workgroup_size(WG)
fn backproject(@builtin(global_invocation_id) gid : vec3<u32>,
               @builtin(num_workgroups) nwg : vec3<u32>) {
    let flat = gid.y * nwg.x * WG + gid.x;
    let plane = params.ny * params.nx;
    let nz = arrayLength(&center);
    if (flat >= nz * plane) { return; }

    let iz = flat / plane;
    let rem = flat % plane;
    let iy = rem / params.nx;
    let ix = rem % params.nx;

    let cx = f32(params.nx) * 0.5;
    let cy = f32(params.ny) * 0.5;
    let gx = f32(ix) - cx;
    let gy = f32(iy) - cy;
    let ctr = center[iz];
    let base = iz * params.nproj * params.ncols;

    var acc = 0.0;
    for (var ia = 0u; ia < params.nproj; ia = ia + 1u) {
        let c = cossin[ia * 2u];
        let sn = cossin[ia * 2u + 1u];
        let t = gx * c + gy * sn + ctr;
        let t0 = floor(t);
        let i0 = i32(t0);
        let frac = t - t0;
        // `linear_taps` (backend.rs): taps ⌊t⌋ and ⌊t⌋ + 1, zero off the
        // detector. u32(i32) keeps the bits, so a negative tap fails `< ncols`.
        let row = base + ia * params.ncols;
        if (u32(i0) < params.ncols) {
            acc = acc + sino[row + u32(i0)] * (1.0 - frac);
        }
        if (u32(i0 + 1) < params.ncols) {
            acc = acc + sino[row + u32(i0 + 1)] * frac;
        }
    }
    vol[flat] = acc * params.scale;
}
