// Device-resident log-polar (lprec) runtime — WGSL port of the per-slice
// `recon::lprec::process_row` (and tomocupy cuda/lprec.cu), mirroring the CPU
// path step-for-step so the result is parity. The geometry grids (kfull kernel,
// per-span log-polar/Cartesian coords, index sets) are precomputed once on the
// host by `build_grids` and uploaded; these kernels consume them. The 2-D FFT
// convolution reuses the shared device-resident fft_passes/fft_transpose, driven
// from the Rust orchestrator between gather and scatter.
//
// C2C (not the CUDA R2C/C2R): the wgpu FFT is full-complex radix-2 only, so the
// log-polar buffer is a full [nrho, ntheta] complex grid (stride ntheta, no row
// padding), matching the CPU port. `WG` is injected by dispatch1d.
//
// Float atomics: `gather` accumulates through the injected `atom_add_ga_fl`
// helper (WgpuBackend::atomic_f32_decl) — native f32 atomicAdd on devices with
// SHADER_FLOAT32_ATOMIC, else the portable compare-exchange emulation on the
// bit-cast u32 lane (main and wrapping point sets can hit the same target).

const POLE : f32 = -0.2679492; // cubic-B-spline pole √3−2

struct LpParams {
    nz     : u32,
    nang   : u32,
    n      : u32,
    ntheta : u32,
    nrho   : u32,
    npts   : u32, // point count of the current gather/scatter set
    _p0    : u32,
    _p1    : u32,
    scale  : f32, // take_re scale = 2 / (nrho*ntheta)
    _p2    : f32,
    _p3    : f32,
    _p4    : f32,
};

// Cubic-B-spline basis weights at fractional offset f∈[0,1); taps floor−1..floor+2.
fn bspline_weights(f : f32) -> vec4<f32> {
    let one = 1.0 - f;
    let sq = f * f;
    let one_sq = one * one;
    return vec4<f32>(
        (1.0 / 6.0) * one_sq * one,
        2.0 / 3.0 - 0.5 * sq * (2.0 - f),
        2.0 / 3.0 - 0.5 * one_sq * (2.0 - one),
        (1.0 / 6.0) * sq * f,
    );
}

fn wrapi(i : i32, n : i32) -> i32 {
    return ((i % n) + n) % n;
}

// --- cubic-B-spline prefilter (samples → spline coefficients) ----------------
// Sequential causal/anticausal recursion; one thread owns one whole line, so it
// references the global buffer directly (both prefilter entries share pf_g).
@group(0) @binding(0) var<storage, read_write> pf_g : array<f32>;
@group(0) @binding(1) var<uniform>             pf_p : LpParams;

fn convert_coeffs(base : u32, len : u32, stride : u32) {
    if (len < 2u) { return; }
    let lambda = (1.0 - POLE) * (1.0 - 1.0 / POLE);
    var horizon = 12u;
    if (len < 12u) { horizon = len; }
    var zn = POLE;
    var sum = pf_g[base];
    for (var k = 0u; k < horizon; k = k + 1u) {
        sum = sum + zn * pf_g[base + k * stride];
        zn = zn * POLE;
    }
    pf_g[base] = lambda * sum;
    var prev = pf_g[base];
    for (var k = 1u; k < len; k = k + 1u) {
        let v = lambda * pf_g[base + k * stride] + POLE * prev;
        pf_g[base + k * stride] = v;
        prev = v;
    }
    let last = base + (len - 1u) * stride;
    pf_g[last] = pf_g[last] * (POLE / (POLE - 1.0));
    prev = pf_g[last];
    for (var kk = len - 1u; kk > 0u; kk = kk - 1u) {
        let idx = base + (kk - 1u) * stride;
        let v = POLE * (prev - pf_g[idx]);
        pf_g[idx] = v;
        prev = v;
    }
}

// Detector-axis prefilter: one thread per (slice, angle) line, stride 1.
@compute @workgroup_size(WG)
fn prefilter_rows(@builtin(global_invocation_id) gid : vec3<u32>,
                  @builtin(num_workgroups) nwg : vec3<u32>) {
    let t = gid.y * nwg.x * WG + gid.x;
    if (t >= pf_p.nz * pf_p.nang) { return; }
    convert_coeffs(t * pf_p.n, pf_p.n, 1u);
}

// Angle-axis prefilter: one thread per (slice, detector) column, stride n.
@compute @workgroup_size(WG)
fn prefilter_cols(@builtin(global_invocation_id) gid : vec3<u32>,
                  @builtin(num_workgroups) nwg : vec3<u32>) {
    let t = gid.y * nwg.x * WG + gid.x;
    if (t >= pf_p.nz * pf_p.n) { return; }
    let s = t / pf_p.n;
    let d = t % pf_p.n;
    convert_coeffs(s * pf_p.nang * pf_p.n + d, pf_p.nang, pf_p.n);
}

// --- gather: polar → log-polar cubic interpolation (atomic accumulate) --------
// @binding(1) `ga_fl` + `atom_add_ga_fl` — injected (atomic_f32_decl).
@group(0) @binding(0) var<storage, read>       ga_g       : array<f32>;
@group(0) @binding(2) var<storage, read>       ga_targets : array<u32>;
@group(0) @binding(3) var<storage, read>       ga_xs      : array<f32>; // detector coord (width n)
@group(0) @binding(4) var<storage, read>       ga_ys      : array<f32>; // angle coord (height nang)
@group(0) @binding(5) var<uniform>             ga_p       : LpParams;

@compute @workgroup_size(WG)
fn gather(@builtin(global_invocation_id) gid : vec3<u32>,
          @builtin(num_workgroups) nwg : vec3<u32>) {
    let t = gid.y * nwg.x * WG + gid.x;
    let npts = ga_p.npts;
    if (t >= ga_p.nz * npts) { return; }
    let s = t / npts;
    let idx = t % npts;

    let width = ga_p.n;
    let height = ga_p.nang;
    let gbase = s * ga_p.nang * ga_p.n;
    let xv = ga_xs[idx];
    let yv = ga_ys[idx];
    let ixf = floor(xv);
    let iyf = floor(yv);
    let wx = bspline_weights(xv - ixf);
    let wy = bspline_weights(yv - iyf);
    let ix = i32(ixf);
    let iy = i32(iyf);
    var val = 0.0;
    for (var jj = 0; jj < 4; jj = jj + 1) {
        let py = wrapi(iy - 1 + jj, i32(height));
        let rowb = gbase + u32(py) * width;
        var acc = 0.0;
        for (var ii = 0; ii < 4; ii = ii + 1) {
            let px = wrapi(ix - 1 + ii, i32(width));
            acc = acc + wx[ii] * ga_g[rowb + u32(px)];
        }
        val = val + wy[jj] * acc;
    }

    let cell = s * ga_p.nrho * ga_p.ntheta + ga_targets[idx];
    atom_add_ga_fl(cell, val);
}

// --- real → complex (load the atomic-accumulated real grid into the FFT buffer) ---
@group(0) @binding(0) var<storage, read>       rc_fl  : array<f32>; // fl bytes as f32
@group(0) @binding(1) var<storage, read_write> rc_flc : array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             rc_p   : LpParams;

@compute @workgroup_size(WG)
fn real_to_complex(@builtin(global_invocation_id) gid : vec3<u32>,
                   @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * WG + gid.x;
    if (i >= rc_p.nz * rc_p.nrho * rc_p.ntheta) { return; }
    rc_flc[i] = vec2<f32>(rc_fl[i], 0.0);
}

// --- broadcast complex multiply by the convolution kernel spectrum -----------
@group(0) @binding(0) var<storage, read_write> cm_flc   : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       cm_kfull : array<vec2<f32>>; // [nrho*ntheta]
@group(0) @binding(2) var<uniform>             cm_p     : LpParams;

@compute @workgroup_size(WG)
fn cmul(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * WG + gid.x;
    let ng = cm_p.nrho * cm_p.ntheta;
    if (i >= cm_p.nz * ng) { return; }
    let a = cm_flc[i];
    let b = cm_kfull[i % ng];
    cm_flc[i] = vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// --- take real part × scale (post inverse FFT), producing the scatter coeffs --
@group(0) @binding(0) var<storage, read>       tr_flc   : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> tr_flcre : array<f32>;
@group(0) @binding(2) var<uniform>             tr_p     : LpParams;

@compute @workgroup_size(WG)
fn take_re(@builtin(global_invocation_id) gid : vec3<u32>,
           @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * WG + gid.x;
    if (i >= tr_p.nz * tr_p.nrho * tr_p.ntheta) { return; }
    tr_flcre[i] = tr_flc[i].x * tr_p.scale;
}

// --- scatter: log-polar → Cartesian disk cubic interpolation (plain add) ------
// cids targets are distinct within a span, so each thread owns a unique f cell;
// spans accumulate across successive launches (no atomics needed).
@group(0) @binding(0) var<storage, read>       sc_flcre   : array<f32>;
@group(0) @binding(1) var<storage, read_write> sc_f       : array<f32>;
@group(0) @binding(2) var<storage, read>       sc_targets : array<u32>;
@group(0) @binding(3) var<storage, read>       sc_xs      : array<f32>; // theta coord (width ntheta)
@group(0) @binding(4) var<storage, read>       sc_ys      : array<f32>; // rho coord (height nrho)
@group(0) @binding(5) var<uniform>             sc_p       : LpParams;

@compute @workgroup_size(WG)
fn scatter(@builtin(global_invocation_id) gid : vec3<u32>,
           @builtin(num_workgroups) nwg : vec3<u32>) {
    let t = gid.y * nwg.x * WG + gid.x;
    let npts = sc_p.npts;
    if (t >= sc_p.nz * npts) { return; }
    let s = t / npts;
    let idx = t % npts;

    let width = sc_p.ntheta;
    let height = sc_p.nrho;
    let base = s * sc_p.nrho * sc_p.ntheta;
    let xv = sc_xs[idx];
    let yv = sc_ys[idx];
    let ixf = floor(xv);
    let iyf = floor(yv);
    let wx = bspline_weights(xv - ixf);
    let wy = bspline_weights(yv - iyf);
    let ix = i32(ixf);
    let iy = i32(iyf);
    var val = 0.0;
    for (var jj = 0; jj < 4; jj = jj + 1) {
        let py = wrapi(iy - 1 + jj, i32(height));
        let rowb = base + u32(py) * width;
        var acc = 0.0;
        for (var ii = 0; ii < 4; ii = ii + 1) {
            let px = wrapi(ix - 1 + ii, i32(width));
            acc = acc + wx[ii] * sc_flcre[rowb + u32(px)];
        }
        val = val + wy[jj] * acc;
    }
    let fcell = s * sc_p.n * sc_p.n + sc_targets[idx];
    sc_f[fcell] = sc_f[fcell] + val;
}
