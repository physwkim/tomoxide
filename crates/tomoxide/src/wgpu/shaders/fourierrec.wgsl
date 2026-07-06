// Device-resident Gaussian-USFFT gridding — WGSL port of tomocupy's
// `cfunc_fourierrec` (kernels_fourierrec.cuh) that mirrors the CPU
// `recon::fourierrec` step-for-step so the result is bit-parity (modulo GPU
// transcendental ULPs). The whole chain runs on-device: only the filtered
// sinogram is uploaded and the reconstructed volume downloaded.
//
// `WG` (workgroup size) is injected by `dispatch1d`; `M` (Gaussian half-width)
// is injected by the host so the separable-kernel array `array<f32, K>` is sized
// exactly. Every entry recovers its flat thread index as
// `gid.y * nwg.x * WG + gid.x` (the 2-D dispatch fold used by all wgpu kernels).
//
// Float atomics: `gather`/`wrap` accumulate into the grid (two lanes per
// complex cell) through the injected `atom_add_ga_grid`/`atom_add_wr_grid`
// helpers (WgpuBackend::atomic_f32_decl) — native f32 atomicAdd on devices with
// SHADER_FLOAT32_ATOMIC, else the portable compare-exchange emulation on
// bit-cast u32 lanes. Kernels that only read the grid bind the same buffer as
// `array<f32>` (identical bytes either way).

const PI : f32 = 3.14159265358979;
const K : u32 = 2u * M + 1u; // separable Gaussian kernel length (2m+1)

struct FrParams {
    nz       : u32,
    nang     : u32,
    nd       : u32,
    n        : u32, // output slice size
    ng       : u32, // oversampled grid width  = 2*nd + 2*m
    nf       : u32, // centred inverse-FFT size = 2*nd
    crop     : u32, // central-crop offset      = (nd - min(n,nd)) / 2
    _p0      : u32,
    mu       : f32,
    coeff0   : f32, // gather amplitude   = PI / (mu * 4 * nd^2)
    coeff1   : f32, // Gaussian exponent  = -PI^2 / mu
    gscale   : f32, // gather input scale = 4 / nd
    phi_sign : f32, // divphi global sign = 1 - nd%4
    norm     : f32, // pi/4 = (1/nf^2 inverse-FFT normalisation) x (pi*nd^2 unified amplitude)
    _p1      : f32,
    _p2      : f32,
};

// Centred-DFT modulation sign 1 - 2*((i+1)%2): -1 on even, +1 on odd.
fn shift_sign(i : u32) -> f32 {
    return select(1.0, -1.0, (i & 1u) == 0u);
}

// Emulated f32 atomic add is inlined at each call site: naga (without the
// unrestricted-pointer-parameters extension) forbids passing a storage pointer
// into a helper, so the compare-exchange loop on the bit-cast integer lane is
// written out where it is used (`gather`, `wrap`).

// --- 1. build the complex radial buffer with the pre-FFT shift modulation -----
@group(0) @binding(0) var<storage, read>       br_sino   : array<f32>;
@group(0) @binding(1) var<storage, read_write> br_radial : array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             br_p      : FrParams;

@compute @workgroup_size(WG)
fn build_radial(@builtin(global_invocation_id) gid : vec3<u32>,
                @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * WG + gid.x;
    if (i >= br_p.nz * br_p.nang * br_p.nd) { return; }
    let j = i % br_p.nd;
    br_radial[i] = vec2<f32>(br_sino[i] * shift_sign(j), 0.0);
}

// --- 2. post-FFT shift modulation (in place on the radial spectrum) ----------
@group(0) @binding(0) var<storage, read_write> pm_radial : array<vec2<f32>>;
@group(0) @binding(1) var<uniform>             pm_p      : FrParams;

@compute @workgroup_size(WG)
fn postmod(@builtin(global_invocation_id) gid : vec3<u32>,
           @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * WG + gid.x;
    if (i >= pm_p.nz * pm_p.nang * pm_p.nd) { return; }
    let j = i % pm_p.nd;
    pm_radial[i] = pm_radial[i] * shift_sign(j);
}

// --- 3. Gaussian gather (scatter each radial sample onto the Cartesian grid) --
// @binding(1) `ga_grid` + `atom_add_ga_grid` — injected (atomic_f32_decl).
@group(0) @binding(0) var<storage, read>       ga_radial : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       ga_trig   : array<vec2<f32>>; // (cos, sin)
@group(0) @binding(3) var<uniform>             ga_p      : FrParams;

@compute @workgroup_size(WG)
fn gather(@builtin(global_invocation_id) gid : vec3<u32>,
          @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * WG + gid.x;
    let nd = ga_p.nd;
    let nang = ga_p.nang;
    if (i >= ga_p.nz * nang * nd) { return; }
    let td = i % nd;
    let ia = (i / nd) % nang;
    let z  = i / (nd * nang);

    let cs = ga_trig[ia];
    let ndf = f32(nd);
    var x0 = (f32(td) - ndf * 0.5) / ndf * cs.x;
    var y0 = (f32(td) - ndf * 0.5) / ndf * cs.y;
    if (x0 >= 0.5) { x0 = 0.5 - 1e-5; }
    if (y0 >= 0.5) { y0 = 0.5 - 1e-5; }
    let g0 = ga_radial[i] * ga_p.gscale;

    let base0 = i32(floor(2.0 * ndf * x0)) - i32(M);
    let base1 = i32(floor(2.0 * ndf * y0)) - i32(M);
    var kern0 : array<f32, K>;
    var kern1 : array<f32, K>;
    for (var i0 = 0u; i0 < K; i0 = i0 + 1u) {
        let w0 = f32(base0 + i32(i0)) / (2.0 * ndf) - x0;
        kern0[i0] = exp(ga_p.coeff1 * w0 * w0);
    }
    for (var i1 = 0u; i1 < K; i1 = i1 + 1u) {
        let w1 = f32(base1 + i32(i1)) / (2.0 * ndf) - y0;
        kern1[i1] = exp(ga_p.coeff1 * w1 * w1);
    }

    let ng = ga_p.ng;
    let col0 = i32(nd) + i32(M) + base0;
    let row0 = i32(nd) + i32(M) + base1;
    let zbase = z * ng * ng;
    for (var i1 = 0u; i1 < K; i1 = i1 + 1u) {
        let rr = u32(row0 + i32(i1));
        for (var i0 = 0u; i0 < K; i0 = i0 + 1u) {
            let w = ga_p.coeff0 * kern0[i0] * kern1[i1];
            let cc = u32(col0 + i32(i0));
            let cell = zbase + rr * ng + cc;
            atom_add_ga_grid(2u * cell, g0.x * w);
            atom_add_ga_grid(2u * cell + 1u, g0.y * w);
        }
    }
}

// --- 4. wrap the m-wide borders back into the interior (periodic over 2*nd) ---
// @binding(0) `wr_grid` + `atom_add_wr_grid`/`atom_load_wr_grid` — injected.
@group(0) @binding(1) var<uniform>             wr_p    : FrParams;

@compute @workgroup_size(WG)
fn wrap(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * WG + gid.x;
    let ng = wr_p.ng;
    if (i >= wr_p.nz * ng * ng) { return; }
    let tx = i % ng;
    let ty = (i / ng) % ng;
    let z  = i / (ng * ng);
    let m = M;
    let twond = 2u * wr_p.nd;
    if (tx < m || tx >= twond + m || ty < m || ty >= twond + m) {
        let tx0 = (tx + twond - m) % twond;
        let ty0 = (ty + twond - m) % twond;
        let zbase = z * ng * ng;
        let src = zbase + ty * ng + tx;
        let dst = zbase + (ty0 + m) * ng + (tx0 + m);
        // Border cells receive no in-place accumulation after the gather, and
        // each border cell wraps to exactly one interior destination, so the
        // loads race with nothing; the adds still race with other border cells
        // mapping to the same interior cell.
        atom_add_wr_grid(2u * dst, atom_load_wr_grid(2u * src));
        atom_add_wr_grid(2u * dst + 1u, atom_load_wr_grid(2u * src + 1u));
    }
}

// --- 5. extract the 2*nd interior block + pre-inverse-FFT shift modulation ----
@group(0) @binding(0) var<storage, read>       ex_grid  : array<f32>; // grid bytes as f32
@group(0) @binding(1) var<storage, read_write> ex_inner : array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             ex_p     : FrParams;

@compute @workgroup_size(WG)
fn extract(@builtin(global_invocation_id) gid : vec3<u32>,
           @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * WG + gid.x;
    let nf = ex_p.nf;
    if (i >= ex_p.nz * nf * nf) { return; }
    let rx = i % nf;
    let ry = (i / nf) % nf;
    let z  = i / (nf * nf);
    let ng = ex_p.ng;
    let m = M;
    let gcell = z * ng * ng + (ry + m) * ng + (rx + m);
    let v = vec2<f32>(ex_grid[2u * gcell], ex_grid[2u * gcell + 1u]);
    ex_inner[i] = v * (shift_sign(rx) * shift_sign(ry));
}

// --- 6. post-inverse-FFT shift modulation (in place) -------------------------
@group(0) @binding(0) var<storage, read_write> fs_inner : array<vec2<f32>>;
@group(0) @binding(1) var<uniform>             fs_p     : FrParams;

@compute @workgroup_size(WG)
fn fftshift(@builtin(global_invocation_id) gid : vec3<u32>,
            @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * WG + gid.x;
    let nf = fs_p.nf;
    if (i >= fs_p.nz * nf * nf) { return; }
    let rx = i % nf;
    let ry = (i / nf) % nf;
    fs_inner[i] = fs_inner[i] * (shift_sign(rx) * shift_sign(ry));
}

// --- 7. Gaussian deapodize (divphi) + central crop + unit-disk mask (circ) ----
@group(0) @binding(0) var<storage, read>       de_inner : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> de_out   : array<f32>;
@group(0) @binding(2) var<uniform>             de_p     : FrParams;

@compute @workgroup_size(WG)
fn deapodize(@builtin(global_invocation_id) gid : vec3<u32>,
             @builtin(num_workgroups) nwg : vec3<u32>) {
    let i = gid.y * nwg.x * WG + gid.x;
    let n = de_p.n;
    if (i >= de_p.nz * n * n) { return; }
    let ox = i % n;
    let oy = (i / n) % n;
    let z  = i / (n * n);
    let nd = de_p.nd;
    let nf = de_p.nf;
    let ty = oy + de_p.crop;
    let tx = ox + de_p.crop;
    let ndf = f32(nd);
    let dy = f32(ty) / ndf - 0.5;
    let dx = f32(tx) / ndf - 0.5;
    let phi = exp(de_p.mu * ndf * ndf * (dx * dx + dy * dy)) / f32(de_p.nang) * de_p.phi_sign;
    let inner_row = ty + nd / 2u;
    let inner_col = tx + nd / 2u;
    let icell = z * nf * nf + inner_row * nf + inner_col;
    let v = de_inner[icell].x * phi * de_p.norm;
    let my = (f32(ty) - ndf * 0.5) / ndf;
    let mx = (f32(tx) - ndf * 0.5) / ndf;
    let masked = select(0.0, v, (4.0 * mx * mx + 4.0 * my * my) < 1.0);
    de_out[z * n * n + oy * n + ox] = masked;
}
