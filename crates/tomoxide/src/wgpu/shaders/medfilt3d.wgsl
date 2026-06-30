// 3-D median / dezinger rank filter — port of the CpuBackend RankFilter impl
// (tomopy `libtomo/misc/median_filt3d.c::medfilt3D_float`). One thread per voxel
// gathers the (2·radius+1)³ neighbourhood with clamp-to-center boundary (an
// out-of-range index on any axis reverts to that axis's *center* index), takes
// the value at total/2, and (dezinger) replaces the voxel only when it deviates
// from that median by at least `threshold`.
//
// This is a pure gather + order-statistic + one f32 subtraction — no
// transcendentals and no reduction — so the GPU result is BIT-EXACT with the CPU
// reference (Δ=0), unlike the multiply-accumulate kernels.
//
// WGSL private arrays need a compile-time size, so the window is capped at
// MAX_WIN (diameter 7 ⇒ size ≤ 7); the host validates `total ≤ MAX_WIN` and
// errors for larger windows rather than truncating silently.

const MAX_WIN : u32 = 343u; // 7³ — diameter up to 7 (size up to 7)

struct Params {
    dz        : u32,
    dy        : u32,
    dx        : u32,
    radius    : u32,
    threshold : f32,
    _pad0     : u32,
    _pad1     : u32,
    _pad2     : u32,
};

@group(0) @binding(0) var<storage, read>       inp    : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp   : array<f32>;
@group(0) @binding(2) var<uniform>             params : Params;

@compute @workgroup_size(WG)
fn medfilt3d(@builtin(global_invocation_id) gid : vec3<u32>,
             @builtin(num_workgroups) nwg : vec3<u32>) {
    let flat = gid.y * nwg.x * WG + gid.x;
    let n = params.dz * params.dy * params.dx;
    if (flat >= n) { return; }

    let plane = params.dy * params.dx;
    let z = i32(flat / plane);
    let rem = flat % plane;
    let y = i32(rem / params.dx);
    let x = i32(rem % params.dx);

    let r = i32(params.radius);
    let dzi = i32(params.dz);
    let dyi = i32(params.dy);
    let dxi = i32(params.dx);

    // Gather order x(di) → y(dj) → z(dk), matching the CPU reference; each axis
    // clamps independently to its own center index when out of range.
    var win : array<f32, MAX_WIN>;
    var count = 0u;
    for (var di = -r; di <= r; di = di + 1) {
        var xi = x + di;
        if (xi < 0 || xi >= dxi) { xi = x; }
        for (var dj = -r; dj <= r; dj = dj + 1) {
            var yj = y + dj;
            if (yj < 0 || yj >= dyi) { yj = y; }
            for (var dk = -r; dk <= r; dk = dk + 1) {
                var zk = z + dk;
                if (zk < 0 || zk >= dzi) { zk = z; }
                let idx = (u32(zk) * params.dy + u32(yj)) * params.dx + u32(xi);
                win[count] = inp[idx];
                count = count + 1u;
            }
        }
    }

    // Partial selection sort up to total/2: position midval ends up holding the
    // midval-th smallest, exactly the value a full sort would place there (ties
    // are interchangeable), so this matches the CPU's `sort; window[total/2]`.
    let midval = count / 2u;
    for (var i = 0u; i <= midval; i = i + 1u) {
        var m = i;
        for (var j = i + 1u; j < count; j = j + 1u) {
            // Finite-float ordering == `<` (CPU uses total_cmp; inputs finite).
            if (win[j] < win[m]) { m = j; }
        }
        let tmp = win[m];
        win[m] = win[i];
        win[i] = tmp;
    }

    let median = win[midval];
    let center = inp[flat];
    // threshold == 0 ⇒ |Δ| ≥ 0 always true ⇒ plain median (matches CPU).
    if (abs(center - median) >= params.threshold) {
        outp[flat] = median;
    } else {
        outp[flat] = center;
    }
}
