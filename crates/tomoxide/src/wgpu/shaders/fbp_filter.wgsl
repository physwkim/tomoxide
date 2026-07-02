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
    pad      : u32,   // padded transform length (== w length)
    ncols    : u32,   // detector width (unpadded lane length)
    pad_side : u32,   // left pad = pad/2 − ncols/2 (centre offset)
    _pad0    : u32,
    scale    : f32,   // unpack normalisation, 1/pad (fft_passes leaves inverse raw)
    _pad1    : f32,
    _pad2    : f32,
    _pad3    : f32,
};

@group(0) @binding(0) var<storage, read_write> g : array<vec2<f32>>; // [batch·pad] spectra
@group(0) @binding(1) var<storage, read>       w : array<f32>;       // [pad] real filter
@group(0) @binding(2) var<uniform>             params : Params;
@group(0) @binding(3) var<storage, read>       deltas : array<f32>;  // [batch] per-lane shift

@compute @workgroup_size(WG)
fn apply_filter(@builtin(global_invocation_id) gid : vec3<u32>,
                @builtin(num_workgroups) nwg : vec3<u32>) {
    let idx = gid.y * nwg.x * WG + gid.x;
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

// --- device-resident pack: real sinogram → centred, edge-replicate-padded
// complex spectrum input. Replaces the host-side complex packing + upload of the
// full padded batch (which dominated FbpFilter::apply wall time): only the raw
// real sinogram is uploaded, and every output complex element is filled on-GPU.
// Lane `L` occupies `[L·pad, (L+1)·pad)`; within it, `[0, pad_side)` replicates
// the first column, `[pad_side, pad_side+ncols)` copies the lane, the tail
// replicates the last column — matching CpuBackend::apply / the old host loop.
@group(0) @binding(0) var<storage, read>       pk_sino : array<f32>;       // [batch·ncols]
@group(0) @binding(1) var<storage, read_write> pk_out  : array<vec2<f32>>; // [batch·pad]
@group(0) @binding(2) var<uniform>             pk_p    : Params;

@compute @workgroup_size(WG)
fn pack(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) nwg : vec3<u32>) {
    let idx = gid.y * nwg.x * WG + gid.x;
    if (idx >= arrayLength(&pk_out)) {
        return;
    }
    let lane = idx / pk_p.pad;
    let j = idx % pk_p.pad;
    let base = lane * pk_p.ncols;
    var val : f32;
    if (j < pk_p.pad_side) {
        val = pk_sino[base];
    } else if (j < pk_p.pad_side + pk_p.ncols) {
        val = pk_sino[base + (j - pk_p.pad_side)];
    } else {
        val = pk_sino[base + pk_p.ncols - 1u];
    }
    pk_out[idx] = vec2<f32>(val, 0.0);
}

// --- device-resident unpack: extract the real central `ncols` window of each
// filtered lane (× 1/pad normalisation), writing a compact real sinogram back.
// Replaces the host download of the full complex padded batch + host scatter.
@group(0) @binding(0) var<storage, read>       up_in  : array<vec2<f32>>; // [batch·pad]
@group(0) @binding(1) var<storage, read_write> up_out : array<f32>;       // [batch·ncols]
@group(0) @binding(2) var<uniform>             up_p   : Params;

@compute @workgroup_size(WG)
fn unpack(@builtin(global_invocation_id) gid : vec3<u32>,
          @builtin(num_workgroups) nwg : vec3<u32>) {
    let idx = gid.y * nwg.x * WG + gid.x;
    if (idx >= arrayLength(&up_out)) {
        return;
    }
    let lane = idx / up_p.ncols;
    let i = idx % up_p.ncols;
    up_out[idx] = up_in[lane * up_p.pad + up_p.pad_side + i].x * up_p.scale;
}
