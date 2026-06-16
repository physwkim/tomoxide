// Elementwise preprocessing — ports of tomocupy proc_functions.{darkflat_correction, minus_log}
// and tomopy prep/normalize::minus_log. Two entry points share this module.

// --- minus_log: in-place -ln(max(x, 1e-6)), non-finite -> 0 -----------------
// Matches the CpuBackend definition; GPU `log` differs from libm by a few ULP,
// so callers compare with a tolerance, not bit-for-bit.
@group(0) @binding(0) var<storage, read_write> ml_data : array<f32>;

@compute @workgroup_size(256)
fn minus_log(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&ml_data)) { return; }
    let out = -log(max(ml_data[i], 1e-6));
    // abs(out) < f32::MAX is false for both NaN and ±inf (any NaN compare is
    // false), so this is the WGSL equivalent of Rust `is_finite`.
    ml_data[i] = select(0.0, out, abs(out) < 3.4028235e38);
}

// --- darkflat: (data - dark2d) / denom, broadcast over projections ----------
// `dark2d`/`denom` are the frame-averaged dark and (flat-dark) planes (computed
// host-side, with denom guarded away from zero); `data` is in projection layout
// so element i = proj*plane_size + (row*cols+col), and rc = i % plane_size.
struct DfParams {
    n_elems    : u32,
    plane_size : u32,
    _pad0      : u32,
    _pad1      : u32,
};

@group(0) @binding(0) var<storage, read_write> df_data  : array<f32>;
@group(0) @binding(1) var<storage, read>       df_dark  : array<f32>;
@group(0) @binding(2) var<storage, read>       df_denom : array<f32>;
@group(0) @binding(3) var<uniform>             df_params : DfParams;

@compute @workgroup_size(256)
fn darkflat(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= df_params.n_elems) { return; }
    let rc = i % df_params.plane_size;
    df_data[i] = (df_data[i] - df_dark[rc]) / df_denom[rc];
}
