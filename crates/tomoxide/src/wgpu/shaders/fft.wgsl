// Batched radix-2 FFT — iterative Cooley-Tukey (DIT) over power-of-two lengths.
// Two entry points share one layout (data read_write + params): `bitrev`
// permutes each transform into bit-reversed order in place, then `butterfly` is
// dispatched once per stage (log2(n) times) to combine. Stages run as separate
// queue submissions, which serialize in order, so each stage sees the previous
// stage's writes. `data` is an interleaved complex buffer (vec2 = (re, im)),
// laid out as `batch` contiguous transforms of length `n`.
//
// GPU cos/sin twiddles differ from a CPU library's by a few ULP, so callers
// compare with a tolerance (the inverse normalization 1/n is applied host-side).

struct FftParams {
    n    : u32, // transform length (power of two)
    logn : u32, // log2(n)
    m    : u32, // current stage size 2^(stage+1) (butterfly only)
    sign : f32, // -1 forward, +1 inverse (sign of the twiddle exponent)
};

@group(0) @binding(0) var<storage, read_write> data   : array<vec2<f32>>;
@group(0) @binding(1) var<uniform>             params : FftParams;

@compute @workgroup_size(WG)
fn bitrev(@builtin(global_invocation_id) gid : vec3<u32>,
          @builtin(num_workgroups) nwg : vec3<u32>) {
    let tid = gid.y * nwg.x * WG + gid.x;
    let total = arrayLength(&data);
    if (tid >= total) { return; }
    let lane = tid / params.n;
    let i = tid % params.n;

    var r = 0u;
    var x = i;
    for (var b = 0u; b < params.logn; b = b + 1u) {
        r = (r << 1u) | (x & 1u);
        x = x >> 1u;
    }
    // Each unordered pair is swapped once, by the lower-index thread; the
    // partner thread (i > r) skips, so the swaps never race.
    if (i < r) {
        let base = lane * params.n;
        let tmp = data[base + i];
        data[base + i] = data[base + r];
        data[base + r] = tmp;
    }
}

@compute @workgroup_size(WG)
fn butterfly(@builtin(global_invocation_id) gid : vec3<u32>,
             @builtin(num_workgroups) nwg : vec3<u32>) {
    let tid = gid.y * nwg.x * WG + gid.x;
    let total = arrayLength(&data) / 2u; // batch * (n/2) butterflies
    if (tid >= total) { return; }

    let half = params.m / 2u;
    let nbf = params.n / 2u;
    let lane = tid / nbf;
    let b = tid % nbf;
    let group = b / half;
    let j = b % half;
    let base = lane * params.n;
    let ilow = base + group * params.m + j;
    let ihigh = ilow + half;

    let theta = params.sign * 6.2831853071795862 * f32(j) / f32(params.m);
    let w = vec2<f32>(cos(theta), sin(theta));
    let lo = data[ilow];
    let hi = data[ihigh];
    let t = vec2<f32>(w.x * hi.x - w.y * hi.y, w.x * hi.y + w.y * hi.x);
    data[ilow] = lo + t;
    data[ihigh] = lo - t;
}
