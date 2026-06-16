// Bluestein (chirp-z) convolution multiply — the pointwise step between the
// forward and inverse FFT of the chirp-z transform that lets the radix-2 FFT
// evaluate an arbitrary (non-power-of-two) length-N DFT.
//
// A length-N DFT is rewritten as a linear convolution of the chirp-premultiplied
// input with a fixed chirp kernel, evaluated by a power-of-two circular
// convolution of length `m = next_power_of_two(2N-1)`: FFT both, multiply
// spectra, inverse FFT. This kernel does that spectral multiply, broadcasting
// the single kernel spectrum `h` (length `m`) across every lane of `a`
// (`batch·m`). Both are interleaved complex (vec2 = (re, im)); chirp generation,
// the input premultiply, and the output postmultiply/crop run host-side where
// the argument reduction matches the CPU reference.

struct Params {
    m     : u32, // convolution length (power of two), == kernel-spectrum length
    _pad0 : u32,
    _pad1 : u32,
    _pad2 : u32,
};

@group(0) @binding(0) var<storage, read_write> a : array<vec2<f32>>; // [batch·m] lane spectra
@group(0) @binding(1) var<storage, read>       h : array<vec2<f32>>; // [m] kernel spectrum
@group(0) @binding(2) var<uniform>             params : Params;

@compute @workgroup_size(WG)
fn cmul(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&a)) {
        return;
    }
    let j = idx % params.m; // bin within this lane's transform
    let x = a[idx];
    let y = h[j];
    a[idx] = vec2<f32>(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}
