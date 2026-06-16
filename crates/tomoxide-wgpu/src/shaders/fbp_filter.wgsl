// FBP filter application — the frequency-domain multiply between the forward
// and inverse FFT of FbpFilter::apply. Each lane's padded spectrum (length
// `pad`) is multiplied, component-wise, by the real apodized ramp filter `w`
// (broadcast across the batch of lanes). Ramp filtering is a shift-invariant
// 1-D convolution, so the rotation center is handled entirely by the
// back-projector — there is no phase shift here, matching CpuBackend::apply.

struct Params {
    pad   : u32,   // padded transform length (== w length)
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(0) var<storage, read_write> g : array<vec2<f32>>; // [batch·pad] spectra
@group(0) @binding(1) var<storage, read>       w : array<f32>;       // [pad] real filter
@group(0) @binding(2) var<uniform>             params : Params;

@compute @workgroup_size(WG)
fn apply_filter(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&g)) {
        return;
    }
    let k = idx % params.pad;   // frequency bin within this lane's transform
    g[idx] = g[idx] * w[k];     // complex × real (component-wise scale)
}
