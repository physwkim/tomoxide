// Per-image transpose of an interleaved complex buffer, used to turn the column
// pass of a 2-D FFT into a contiguous row pass. Reads a `rows × cols` image and
// writes its `cols × rows` transpose: dst[c·rows + r] = src[r·cols + c], offset
// per batch image.

struct TParams {
    rows  : u32,
    cols  : u32,
    _pad0 : u32,
    _pad1 : u32,
};

@group(0) @binding(0) var<storage, read>       tsrc    : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> tdst    : array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             tparams : TParams;

@compute @workgroup_size(WG)
fn transpose(@builtin(global_invocation_id) gid : vec3<u32>) {
    let tid = gid.x;
    let total = arrayLength(&tsrc);
    if (tid >= total) { return; }

    let img = tparams.rows * tparams.cols;
    let base = (tid / img) * img;
    let local = tid % img;
    let r = local / tparams.cols;
    let c = local % tparams.cols;
    tdst[base + c * tparams.rows + r] = tsrc[base + r * tparams.cols + c];
}
