// Parallel-beam back-projection — port of tomocupy linerec / tomopy fbp back-projection.
// One thread per output pixel accumulates the filtered sinogram along all angles.
// Scaffold: filled in milestone M6.

struct Params {
    n      : u32,   // slice is n x n
    nproj  : u32,
    nz     : u32,
    center : f32,
};

@group(0) @binding(0) var<storage, read>       sino : array<f32>; // [nz, nproj, n]
@group(0) @binding(1) var<storage, read>       theta : array<f32>;
@group(0) @binding(2) var<storage, read_write> vol  : array<f32>; // [nz, n, n]
@group(0) @binding(3) var<uniform>             params : Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.n || y >= params.n) { return; }
    // TODO(M6): for each angle, t = (x-center)*cos(theta) + (y-center)*sin(theta);
    //           accumulate linear-interpolated sino sample into vol.
}
