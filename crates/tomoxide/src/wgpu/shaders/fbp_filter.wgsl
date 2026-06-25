// FBP filter application — the frequency-domain multiply between the forward
// and inverse FFT of FbpFilter::apply. Each lane's padded spectrum (length
// `pad`) is multiplied, component-wise, by the real apodized ramp filter `w`
// (broadcast across the batch of lanes) AND by a per-lane Fourier-shift phase
// that folds the rotation-centre correction into the filter (tomocupy
// `fbp_filter_center`). `deltas[lane] = ncols/2 − center` is the shift, in
// detector pixels, that lands the rotation axis on the midpoint; at δ=0 (the
// default centre) the phase is unity and only the ramp applies. Mirrors
// CpuBackend::apply, including the SIGNED-frequency convention.

struct Params {
    pad   : u32,   // padded transform length (== w length)
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(0) var<storage, read_write> g : array<vec2<f32>>; // [batch·pad] spectra
@group(0) @binding(1) var<storage, read>       w : array<f32>;       // [pad] real filter
@group(0) @binding(2) var<uniform>             params : Params;
@group(0) @binding(3) var<storage, read>       deltas : array<f32>;  // [batch] per-lane shift

@compute @workgroup_size(WG)
fn apply_filter(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&g)) {
        return;
    }
    let k    = idx % params.pad;   // frequency bin within this lane's transform
    let lane = idx / params.pad;   // which detector lane (→ which slice centre)
    var v = g[idx] * w[k];         // complex × real ramp (component-wise scale)
    let delta = deltas[lane];
    if (delta != 0.0) {
        // SIGNED frequency f_k = k (k ≤ pad/2) else k − pad: only this form is
        // Hermitian-symmetric, so the inverse transform of (real ramp × phase)
        // stays real. A raw index collapses the slice at a half-integer centre.
        var fk = f32(k);
        if (k > params.pad / 2u) {
            fk = fk - f32(params.pad);
        }
        let ang = -6.283185307179586 * fk * delta / f32(params.pad);
        let c = cos(ang);
        let s = sin(ang);
        v = vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);
    }
    g[idx] = v;
}
