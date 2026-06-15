// FBP filter application — port of tomocupy cfunc_filter / fbp_filter_center.
// Applies the frequency-domain filter `w` plus the sub-pixel rotation-center
// phase shift. Scaffold: filled in milestone M6 (needs a WGSL FFT or a
// vendored GPU FFT).

struct Params {
    n      : u32,   // padded transform length
    nproj  : u32,
    nz     : u32,
    center : f32,
};

@group(0) @binding(0) var<storage, read_write> g : array<f32>; // sinogram (interleaved re/im)
@group(0) @binding(1) var<storage, read>       w : array<f32>; // filter kernel
@group(0) @binding(2) var<uniform>             params : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // TODO(M6): forward FFT along detector axis, multiply by w * exp(-2*pi*i*(-center+n/2)*f),
    //           inverse FFT.
}
