// Parallel-beam forward projection (the Radon transform) — port of the
// CpuBackend ForwardProject impl (tomopy `libtomo/recon/project.c`), the exact
// linear-interp adjoint of backproject.wgsl.
//
// Forward projection is a SCATTER: each object voxel splats its value across the
// two nearest detector columns of every angle. This kernel is the exact
// TRANSPOSE of the voxel-driven back-projector — one thread per object voxel
// (flat index over [nz, ny, nx]), the same trig/interp, read↔write swapped — so
// {A, Aᵀ} is a matched pair by construction (the iterative solvers rely on this).
// Voxels of different (iy, ix) splat onto the same detector column, so the
// accumulation uses an emulated f32 atomic add (WGSL has only integer atomics: a
// compare-exchange loop on the bit-cast lane). The atomic resolves collisions in
// a nondeterministic order, so the result matches the CPU reference only to a
// tolerance (accumulation-order rounding), not bit-for-bit — the same contract as
// the rest of the GPU recon path. This one-thread-per-voxel mapping gives
// nz·ny·nx-way parallelism, unlike the old one-thread-per-(row,angle) design
// (nz·nproj threads each looping the whole n² grid) it replaces.

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

@group(0) @binding(0) var<storage, read>       vol    : array<f32>;         // [nz, ny, nx]
@group(0) @binding(1) var<storage, read>       cossin : array<f32>;         // [nproj*2] (c, sn)
@group(0) @binding(2) var<storage, read>       center : array<f32>;         // [nz]
@group(0) @binding(3) var<storage, read_write> sino   : array<atomic<u32>>; // [nz, nproj, ncols]
@group(0) @binding(4) var<uniform>             params : Params;

// Emulated f32 atomic add on lane `idx` of `sino` (compare-exchange on the
// bit-cast integer). Inlined semantics of tomocupy's atomicAdd; naga forbids a
// storage pointer parameter, so the loop is written where it is called.
fn atomic_add_sino(idx : u32, add : f32) {
    var old = atomicLoad(&sino[idx]);
    loop {
        let r = atomicCompareExchangeWeak(&sino[idx], old, bitcast<u32>(bitcast<f32>(old) + add));
        if (r.exchanged) { break; }
        old = r.old_value;
    }
}

@compute @workgroup_size(WG)
fn project(@builtin(global_invocation_id) gid : vec3<u32>,
           @builtin(num_workgroups) nwg : vec3<u32>) {
    let flat = gid.y * nwg.x * WG + gid.x; // flat over (nz * ny * nx)
    let plane = params.ny * params.nx;
    let nz = arrayLength(&center);
    if (flat >= nz * plane) { return; }

    // Zero-valued voxels contribute nothing — skip the whole angle loop and its
    // atomics (a large fraction of a reconstruction's grid is background).
    let f = vol[flat];
    if (f == 0.0) { return; }
    let fs = f * params.scale;

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

    for (var ia = 0u; ia < params.nproj; ia = ia + 1u) {
        let c = cossin[ia * 2u];
        let sn = cossin[ia * 2u + 1u];
        let t = gx * c + gy * sn + ctr;
        let t0 = floor(t);
        let i0 = i32(t0);
        // && short-circuits, so u32(i0) is only evaluated when i0 >= 0.
        if (i0 >= 0 && u32(i0) + 1u < params.ncols) {
            let frac = t - t0;
            let off = base + ia * params.ncols + u32(i0);
            atomic_add_sino(off, fs * (1.0 - frac));
            atomic_add_sino(off + 1u, fs * frac);
        }
    }
}
