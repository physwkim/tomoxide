// Parallel-beam pixel-driven forward projection (the Radon transform) — port of
// the CpuBackend ForwardProject impl (tomopy `libtomo/recon/project.c`), the
// exact linear-interp adjoint of backproject.wgsl.
//
// Forward projection is a SCATTER: each object pixel splats its value across the
// two nearest detector columns. To stay race-free on the GPU we map one thread
// per (row, angle): a thread owns the detector-column span [sbase, sbase+ncols)
// of exactly one (row, angle), which is disjoint from every other thread's, and
// iterates the pixels in the same (iy, ix) order as the CPU — so the per-column
// accumulation order matches the reference and the only GPU/CPU divergence is
// multiply-accumulate rounding (callers compare with a tolerance).

struct Params {
    nproj : u32,
    ncols : u32,
    ny    : u32,
    nx    : u32,
    scale : f32, // π/nproj — the adjoint gain matching the back-projector
    _pad1 : u32,
    _pad2 : u32,
    _pad3 : u32,
};

@group(0) @binding(0) var<storage, read>       vol    : array<f32>; // [nz, ny, nx]
@group(0) @binding(1) var<storage, read>       cossin : array<f32>; // [nproj*2] (c, sn)
@group(0) @binding(2) var<storage, read>       center : array<f32>; // [nz]
@group(0) @binding(3) var<storage, read_write> sino   : array<f32>; // [nz, nproj, ncols]
@group(0) @binding(4) var<uniform>             params : Params;

@compute @workgroup_size(WG)
fn project(@builtin(global_invocation_id) gid : vec3<u32>,
           @builtin(num_workgroups) nwg : vec3<u32>) {
    let lane = gid.y * nwg.x * WG + gid.x; // flat over (nz * nproj)
    let nz = arrayLength(&center);
    if (lane >= nz * params.nproj) { return; }

    let row = lane / params.nproj;
    let ia = lane % params.nproj;
    let c = cossin[ia * 2u];
    let sn = cossin[ia * 2u + 1u];
    let ctr = center[row];
    let cx = f32(params.nx) * 0.5;
    let cy = f32(params.ny) * 0.5;
    let vbase = row * params.ny * params.nx;
    let sbase = (row * params.nproj + ia) * params.ncols;

    for (var iy = 0u; iy < params.ny; iy = iy + 1u) {
        let gy = f32(iy) - cy;
        for (var ix = 0u; ix < params.nx; ix = ix + 1u) {
            let f = vol[vbase + iy * params.nx + ix];
            if (f == 0.0) { continue; }
            let gx = f32(ix) - cx;
            let t = gx * c + gy * sn + ctr;
            let t0 = floor(t);
            let i0 = i32(t0);
            // && short-circuits, so u32(i0) is only evaluated when i0 >= 0.
            if (i0 >= 0 && u32(i0) + 1u < params.ncols) {
                let frac = t - t0;
                let off = sbase + u32(i0);
                sino[off] = sino[off] + f * (1.0 - frac);
                sino[off + 1u] = sino[off + 1u] + f * frac;
            }
        }
    }

    // Scale by π/nproj so the forward projector is the true adjoint of the
    // back-projector (which carries the same gain). This thread uniquely owns the
    // detector-column span [sbase, sbase+ncols) of its (row, angle), so scaling it
    // here is race-free — mirroring how backproject.wgsl bakes its scale in-kernel.
    for (var j = 0u; j < params.ncols; j = j + 1u) {
        sino[sbase + j] = sino[sbase + j] * params.scale;
    }
}
