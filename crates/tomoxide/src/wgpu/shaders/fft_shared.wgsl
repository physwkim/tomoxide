// Shared-memory batched radix-2 FFT — one workgroup per transform. The whole
// length-NN transform is loaded into workgroup memory once (in bit-reversed
// order), all LOGN Cooley-Tukey (DIT) stages run there with a `workgroupBarrier`
// between them, then it is written back once. This collapses the multi-pass
// global kernel's (1 + LOGN) separate submissions + global round-trips into a
// single dispatch with O(1) global traffic, so it is used whenever the transform
// fits shared memory (see `SHARED_FFT_MAX`); larger lengths fall back to the
// global `fft.wgsl` passes.
//
// The butterfly arithmetic, stage order, and twiddles are identical to
// `fft.wgsl`, so results match that path to the bit — only the memory the
// butterflies read/write differs. `WG` (threads per workgroup), `NN` (transform
// length), and `LOGN` (log2 NN) are injected as `const` from the host so the
// shared array is sized exactly and `@workgroup_size` is a single source of
// truth, matching the dispatch's workgroup count.

const TWO_PI : f32 = 6.2831853071795862;

struct FftParams {
    n    : u32,
    logn : u32,
    m    : u32,
    sign : f32, // -1 forward, +1 inverse (sign of the twiddle exponent)
};

@group(0) @binding(0) var<storage, read_write> data   : array<vec2<f32>>;
@group(0) @binding(1) var<uniform>             params : FftParams;

var<workgroup> sh : array<vec2<f32>, NN>;

@compute @workgroup_size(WG)
fn fft_shared(@builtin(workgroup_id) wg : vec3<u32>,
              @builtin(num_workgroups) nwg : vec3<u32>,
              @builtin(local_invocation_index) lid : u32) {
    let lanes = arrayLength(&data) / NN;
    let lane = wg.y * nwg.x + wg.x;
    if (lane >= lanes) { return; }
    let base = lane * NN;

    // Load into bit-reversed positions: sh[rev(i)] = data[base + i]. Each thread
    // strides over the transform so any WG ≤ NN covers every element.
    for (var i = lid; i < NN; i = i + WG) {
        var r = 0u;
        var x = i;
        for (var b = 0u; b < LOGN; b = b + 1u) {
            r = (r << 1u) | (x & 1u);
            x = x >> 1u;
        }
        sh[r] = data[base + i];
    }
    workgroupBarrier();

    // LOGN combine stages; the NN/2 butterflies of each stage are shared across
    // the WG threads by striding.
    var m = 2u;
    for (var s = 0u; s < LOGN; s = s + 1u) {
        let half = m / 2u;
        for (var k = lid; k < NN / 2u; k = k + WG) {
            let group = k / half;
            let j = k % half;
            let ilow = group * m + j;
            let ihigh = ilow + half;
            let theta = params.sign * TWO_PI * f32(j) / f32(m);
            let w = vec2<f32>(cos(theta), sin(theta));
            let lo = sh[ilow];
            let hi = sh[ihigh];
            let t = vec2<f32>(w.x * hi.x - w.y * hi.y, w.x * hi.y + w.y * hi.x);
            sh[ilow] = lo + t;
            sh[ihigh] = lo - t;
        }
        workgroupBarrier();
        m = m << 1u;
    }

    for (var i = lid; i < NN; i = i + WG) {
        data[base + i] = sh[i];
    }
}
