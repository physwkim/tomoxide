// Elementwise preprocessing — port of tomocupy proc_functions.{darkflat_correction,minus_log}.
// Scaffold: filled in milestone M6.

struct Params {
    n_elems : u32,
    op      : u32, // 0 = darkflat, 1 = minus_log
};

@group(0) @binding(0) var<storage, read_write> data : array<f32>;
@group(0) @binding(1) var<storage, read>       flat : array<f32>;
@group(0) @binding(2) var<storage, read>       dark : array<f32>;
@group(0) @binding(3) var<uniform>             params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_elems) { return; }
    // TODO(M6): darkflat = (data[i] - dark[i]) / (flat[i] - dark[i]);
    //           minus_log = -log(max(data[i], 1e-6));
}
