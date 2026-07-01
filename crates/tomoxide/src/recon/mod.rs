//! # tomoxide-recon
//!
//! Backend-agnostic reconstruction. The single [`recon`] entry point dispatches
//! every [`Algorithm`] to backend capability traits, so the same code path runs
//! on CPU, CUDA, or wgpu. It depends on `tomoxide-core` only — never on a
//! concrete backend.
//!
//! Analytic methods (`fbp`, `gridrec`, `fourierrec`, `lprec`, `linerec`) are a
//! filter + back-projection pass; iterative methods compose forward projection
//! and back-projection in a loop (see `docs/ARCHITECTURE.md` §3).
//!
//! Laminography ([`lamino`]) is intrinsically 3-D — every tilted projection
//! contributes to every voxel — so it has its own entry point
//! ([`lamino::lamino`]) rather than the per-slice [`recon`] dispatch.
#![forbid(unsafe_code)]

pub mod center;
mod fourierrec;
mod gridrec;
pub mod lamino;
pub(crate) mod lprec;
pub mod ring;
pub mod vector;

use crate::backend::{Backend, FilteredBackproject, ForwardProject, RayProject, RayRow};
use crate::data::{Layout, Tomo, Volume};
use crate::error::{Error, Result};
use crate::geometry::{Angles, Center, Geometry};
use crate::params::{Algorithm, ReconParams};
use ndarray::{Array3, Axis};

fn missing(capability: &'static str, backend: &dyn Backend) -> Error {
    Error::MissingCapability {
        backend: backend.name(),
        capability,
    }
}

/// Reconstruct a volume from a sinogram stack with the given algorithm.
///
/// `sino` is consumed by neither path (it is cloned where filtering mutates a
/// copy). The output volume is `[n_rows, n_grid, n_grid]`.
pub fn recon(
    sino: &Tomo<f32>,
    geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    backend: &dyn Backend,
) -> Result<Volume<f32>> {
    if algorithm.is_analytic() {
        analytic(sino, geom, algorithm, params, backend)
    } else {
        iterative(sino, geom, algorithm, params, backend)
    }
}

/// Grid size for the output slice (defaults to the detector width).
fn grid_size(sino: &Tomo<f32>, params: &ReconParams) -> usize {
    params.num_gridx.unwrap_or_else(|| sino.n_cols())
}

fn analytic(
    sino: &Tomo<f32>,
    geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    backend: &dyn Backend,
) -> Result<Volume<f32>> {
    let n = grid_size(sino, params);
    let nz = sino.n_rows();

    // gridrec is a Fourier-grid method that applies its own ramp internally
    // (needs only the Fft capability). Every other analytic method is a shared
    // FBP-filter pass followed by either a Fourier-grid gridding (fourierrec) or
    // a back-projection (fbp/linerec/lprec) — tomocupy's
    // `fbp_filter_center(data)` → `cl_rec.backprojection`.
    if algorithm == Algorithm::Gridrec {
        let fft = backend.fft().ok_or_else(|| missing("Fft", backend))?;
        return Ok(Volume::new(gridrec::gridrec(sino, geom, n, fft)?));
    }
    // A backend with a fused on-device analytic path (CUDA) runs the whole
    // filter → back-projection / Fourier chain resident on the device — one
    // upload, one download — instead of host-roundtripping each capability.
    if matches!(
        algorithm,
        Algorithm::Fbp | Algorithm::Linerec | Algorithm::Fourierrec
    ) {
        if let Some(ar) = backend.analytic_reconstruct() {
            return ar.reconstruct(sino, geom, algorithm, params);
        }
    }
    let filt = backend
        .fbp_filter()
        .ok_or_else(|| missing("FbpFilter", backend))?;
    let kernel = filt.make_filter(params.filter_name, sino.n_cols())?;
    let mut filtered = sino.clone();
    filt.apply(&mut filtered, &kernel, geom)?;

    // fourierrec grids the filtered projections onto the Fourier plane with
    // tomocupy's Gaussian USFFT kernel. A backend with a monolithic Fourier
    // method (CUDA cfunc_fourierrec) handles it directly; otherwise it composes
    // from the generic Fft capability (CPU/wgpu).
    if algorithm == Algorithm::Fourierrec {
        if let Some(fr) = backend.fourier_reconstruct() {
            return fr.reconstruct(&filtered, geom, n);
        }
        let fft = backend.fft().ok_or_else(|| missing("Fft", backend))?;
        return Ok(Volume::new(fourierrec::fourierrec(
            &filtered, geom, n, fft,
        )?));
    }

    // lprec is the log-polar (Andersson–Carlsson–Nikitin) method: it maps the
    // filtered sinogram into log-polar coordinates where back-projection is a 2-D
    // FFT convolution, then resamples to the Cartesian grid (needs only the Fft
    // capability). Faithful port of tomocupy `lprec`.
    if algorithm == Algorithm::Lprec {
        // A backend with a device-resident lprec (CUDA cuda/lprec.cu) runs the
        // gather/scatter + spline prefilter on the GPU; otherwise it composes
        // from the generic Fft capability (CPU/wgpu) with host interpolation.
        if let Some(lr) = backend.lprec_reconstruct() {
            return lr.reconstruct(&filtered, geom, n);
        }
        let fft = backend.fft().ok_or_else(|| missing("Fft", backend))?;
        return Ok(Volume::new(lprec::lprec(&filtered, geom, n, fft)?));
    }

    // fbp / linerec: filtered back-projection. linerec is genuinely a line
    // back-projection (tomocupy `cfunc_linerec` reduces to parallel-beam BP with
    // linear interpolation), so it shares this FBP path. `FbpFilter::apply` has
    // already shifted the rotation axis onto the detector midpoint, so the
    // back-projector runs against a centre = ncols/2 geometry — matching
    // tomocupy, whose back-projection kernels assume n/2 after fbp_filter_center.
    // (The iterative path keeps `geom.center`; only this analytic path recenters.)
    let bp = backend
        .backprojector()
        .ok_or_else(|| missing("FilteredBackproject", backend))?;
    let centered = Geometry {
        center: Center::Scalar(sino.n_cols() as f32 / 2.0),
        ..geom.clone()
    };
    let mut vol = Volume::new(Array3::zeros((nz, n, n)));
    bp.backproject(&filtered, &centered, &mut vol)?;
    Ok(vol)
}

fn iterative(
    sino: &Tomo<f32>,
    geom: &Geometry,
    algorithm: Algorithm,
    params: &ReconParams,
    backend: &dyn Backend,
) -> Result<Volume<f32>> {
    // ART/BART are row-action (Kaczmarz) — they consume the single-ray rows, not
    // the whole-sinogram forward/back-projectors the other methods compose.
    if matches!(algorithm, Algorithm::Art | Algorithm::Bart) {
        let rp = backend
            .ray_projector()
            .ok_or_else(|| missing("RayProject", backend))?;
        return match algorithm {
            Algorithm::Art => art(sino, geom, params, rp),
            _ => bart(sino, geom, params, rp),
        };
    }

    // Device-resident fast path: a backend that keeps the volume/sinogram on the
    // device across all iterations (CUDA) runs the whole loop with no per-iteration
    // host↔device transfers. It returns `None` for algorithms it does not
    // device-implement, so those fall through to the generic host solvers below.
    if let Some(it) = backend.iterative_reconstruct() {
        if let Some(vol) = it.solve(sino, geom, algorithm, params)? {
            return Ok(vol);
        }
    }

    let proj = backend
        .projector()
        .ok_or_else(|| missing("ForwardProject", backend))?;
    let bp = backend
        .backprojector()
        .ok_or_else(|| missing("FilteredBackproject", backend))?;
    match algorithm {
        Algorithm::Sirt => sirt(sino, geom, params, proj, bp),
        Algorithm::Mlem => mlem(sino, geom, params, proj, bp),
        Algorithm::Osem => osem(sino, geom, params, proj, bp),
        // pml_* are ospml_* with a single block (tomopy extern/recon.py). The
        // quadratic prior is delta=None; the hybrid prior uses reg_par[1] as the
        // edge threshold (None if absent ⇒ degenerates to the quadratic prior).
        Algorithm::OspmlQuad => ospml(sino, geom, params, proj, bp, params.num_block, None),
        Algorithm::PmlQuad => ospml(sino, geom, params, proj, bp, 1, None),
        Algorithm::OspmlHybrid => ospml(
            sino,
            geom,
            params,
            proj,
            bp,
            params.num_block,
            params.reg_par.get(1).copied(),
        ),
        Algorithm::PmlHybrid => ospml(
            sino,
            geom,
            params,
            proj,
            bp,
            1,
            params.reg_par.get(1).copied(),
        ),
        Algorithm::Grad => grad(sino, geom, params, proj, bp),
        Algorithm::Tikh => tikh(sino, geom, params, proj, bp),
        Algorithm::Tv => tv(sino, geom, params, proj, bp),
        // Vector tomography reconstructs a vector field from one to three tilt
        // datasets, so it can't fit the scalar (one sinogram → one volume)
        // signature here. It lives in [`vector`] with its own multi-dataset API.
        Algorithm::Vector => Err(Error::InvalidParam(
            "vector tomography is not a scalar reconstruction; call \
             crate::recon::vector::{vector,vector2,vector3} directly"
                .into(),
        )),
        // Analytic methods are dispatched by `recon()` → `analytic()`, and
        // Art/Bart by the ray-projector path above, so they never reach here.
        _ => unreachable!("non-iterative algorithm reached iterative dispatch"),
    }
}

/// Simultaneous Iterative Reconstruction Technique.
///
/// R/C-weighted update `x ← x + C ∘ Aᵀ(R ∘ (b − A x))` with `R = 1/A(1)`
/// (per-ray length) and `C = 1/Aᵀ(1)` (per-pixel sensitivity). This is the
/// parameter-free, convergent form of tomopy's rotation-based SIRT (which
/// distributes the per-ray residual by `1/nx` and averages over angles —
/// exactly `R` and `C` here). `A` = forward projector, `Aᵀ` = back-projector.
fn sirt(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    proj: &dyn ForwardProject,
    bp: &dyn FilteredBackproject,
) -> Result<Volume<f32>> {
    let n = grid_size(sino, params);
    let nz = sino.n_rows();
    let b = sino.as_layout(Layout::Sinogram);
    let nang = b.n_angles();
    let ncols = b.n_cols();

    // Ray-length weights R = 1 / A(1).
    let ones_img = Volume::new(Array3::from_elem((nz, n, n), 1.0));
    let mut ray = Tomo::new(Array3::zeros((nz, nang, ncols)), Layout::Sinogram);
    proj.project(&ones_img, geom, &mut ray)?;
    let rw = ray
        .array
        .mapv(|v| if v.abs() > 1e-6 { 1.0 / v } else { 0.0 });

    // Sensitivity weights C = 1 / Aᵀ(1).
    let ones_sino = Tomo::new(Array3::from_elem((nz, nang, ncols), 1.0), Layout::Sinogram);
    let mut sens = Volume::new(Array3::zeros((nz, n, n)));
    bp.backproject(&ones_sino, geom, &mut sens)?;
    let cw = sens
        .array
        .mapv(|v| if v.abs() > 1e-6 { 1.0 / v } else { 0.0 });

    let mut vol = Volume::new(Array3::zeros((nz, n, n)));
    let mut ax = Tomo::new(Array3::zeros((nz, nang, ncols)), Layout::Sinogram);
    let mut corr = Volume::new(Array3::zeros((nz, n, n)));
    for _ in 0..params.num_iter.max(1) {
        proj.project(&vol, geom, &mut ax)?; // A x
        let mut resid = &b.array - &ax.array; // b − A x
        resid *= &rw; // R ∘ (b − A x)
        bp.backproject(&Tomo::new(resid, Layout::Sinogram), geom, &mut corr)?;
        vol.array += &(&cw * &corr.array); // x += C ∘ Aᵀ(…)
    }
    Ok(vol)
}

/// Maximum-Likelihood Expectation-Maximization.
///
/// Multiplicative update `x ← x ∘ Aᵀ(b ⊘ A x) ⊘ Aᵀ(1)`, positivity-preserving
/// from a positive initial guess; requires a non-negative sinogram. Ports the
/// EM update of tomopy `accel/cxx/mlem.cc`. `A` = forward projector,
/// `Aᵀ` = back-projector.
fn mlem(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    proj: &dyn ForwardProject,
    bp: &dyn FilteredBackproject,
) -> Result<Volume<f32>> {
    let n = grid_size(sino, params);
    let nz = sino.n_rows();
    let b = sino.as_layout(Layout::Sinogram);
    let nang = b.n_angles();
    let ncols = b.n_cols();

    // Sensitivity Aᵀ(1).
    let ones_sino = Tomo::new(Array3::from_elem((nz, nang, ncols), 1.0), Layout::Sinogram);
    let mut sens = Volume::new(Array3::zeros((nz, n, n)));
    bp.backproject(&ones_sino, geom, &mut sens)?;

    let mut vol = Volume::new(Array3::from_elem((nz, n, n), 1.0)); // positive init
    let mut ax = Tomo::new(Array3::zeros((nz, nang, ncols)), Layout::Sinogram);
    let mut corr = Volume::new(Array3::zeros((nz, n, n)));
    for _ in 0..params.num_iter.max(1) {
        proj.project(&vol, geom, &mut ax)?; // A x
        let mut ratio = b.array.clone();
        ndarray::Zip::from(&mut ratio)
            .and(&ax.array)
            .for_each(|r, &a| {
                *r = if a.abs() > 1e-6 { *r / a } else { 0.0 }; // b ⊘ A x
            });
        bp.backproject(&Tomo::new(ratio, Layout::Sinogram), geom, &mut corr)?;
        ndarray::Zip::from(&mut vol.array)
            .and(&corr.array)
            .and(&sens.array)
            .for_each(|x, &c, &s| {
                if s.abs() > 1e-6 {
                    *x = *x * c / s; // x ∘ Aᵀ(ratio) ⊘ sens
                }
            });
    }
    Ok(vol)
}

/// Partition the `nang` angle indices into ordered subsets, matching tomopy
/// `osem.c`: each subset is a contiguous slice of the angle *ordering* — the
/// caller's `ind_block` permutation when it has length `nang`, else the identity
/// `0..nang`. Subset `os` has `nang/num_block + (os < nang % num_block)` angles,
/// so the blocks tile `0..nang` exactly. `num_block` is clamped to `1..=nang`,
/// and `1` yields the single full-angle subset (i.e. plain MLEM).
pub(crate) fn ordered_subsets(nang: usize, params: &ReconParams) -> Vec<Vec<usize>> {
    let num_block = params.num_block.clamp(1, nang.max(1));
    let order: Vec<usize> = if params.ind_block.len() == nang {
        params.ind_block.iter().map(|&i| i as usize).collect()
    } else {
        (0..nang).collect()
    };
    let blocksize = nang / num_block;
    let remainder = nang % num_block;
    let mut subsets = Vec::with_capacity(num_block);
    let mut start = 0;
    for os in 0..num_block {
        let len = blocksize + usize::from(os < remainder);
        subsets.push(order[start..start + len].to_vec());
        start += len;
    }
    subsets
}

/// One ordered subset: its sub-geometry, measured sinogram slice `[nz, len,
/// ncols]`, and the iteration-invariant sensitivity `Aₛᵀ(1)` `[nz, n, n]`.
struct OsSubset {
    geom: Geometry,
    b: Array3<f32>,
    sens: Array3<f32>,
}

/// Build the ordered subsets for a sinogram: each carries its sub-geometry, the
/// gathered sinogram slice, and the precomputed (geometry-only) sensitivity
/// `Aₛᵀ(1)`. Single owner of subset construction for OSEM and the OS-penalized
/// methods. `block_params` supplies `num_block`/`ind_block` to [`ordered_subsets`].
fn build_subsets(
    b: &Tomo<f32>,
    geom: &Geometry,
    n: usize,
    block_params: &ReconParams,
    bp: &dyn FilteredBackproject,
) -> Result<Vec<OsSubset>> {
    let nz = b.n_rows();
    let ncols = b.n_cols();
    let nang = b.n_angles();
    let mut subsets = Vec::new();
    for idx in ordered_subsets(nang, block_params) {
        let len = idx.len();
        let mut sub_geom = geom.clone();
        sub_geom.angles = Angles(idx.iter().map(|&p| geom.angles.0[p]).collect());
        // `select` returns a non-standard-layout owned array; the recon backends
        // consume the subset sinogram via `as_slice()` (CPU back-projection errors
        // on a non-contiguous input), so make it C-contiguous once here — the
        // single owner of subset construction, so every consumer sees contiguous.
        let sub_b = b
            .array
            .select(Axis(1), &idx)
            .as_standard_layout()
            .into_owned();
        let ones = Tomo::new(Array3::from_elem((nz, len, ncols), 1.0), Layout::Sinogram);
        let mut sens = Volume::new(Array3::zeros((nz, n, n)));
        bp.backproject(&ones, &sub_geom, &mut sens)?;
        subsets.push(OsSubset {
            geom: sub_geom,
            b: sub_b,
            sens: sens.array,
        });
    }
    Ok(subsets)
}

/// `corr ← Aₛᵀ(b_s ⊘ Aₛ x)`, the EM correction backprojected over one subset's
/// rays. Shared by OSEM (multiplicative update) and the OS-penalized methods
/// (where it feeds the data term `E`).
fn subset_em_correction(
    vol: &Volume<f32>,
    sub: &OsSubset,
    proj: &dyn ForwardProject,
    bp: &dyn FilteredBackproject,
    corr: &mut Volume<f32>,
) -> Result<()> {
    let nz = vol.dims().0;
    let len = sub.geom.angles.0.len();
    let ncols = sub.b.shape()[2];
    let mut ax = Tomo::new(Array3::zeros((nz, len, ncols)), Layout::Sinogram);
    proj.project(vol, &sub.geom, &mut ax)?; // Aₛ x
    let mut ratio = sub.b.clone();
    ndarray::Zip::from(&mut ratio)
        .and(&ax.array)
        .for_each(|r, &a| {
            *r = if a.abs() > 1e-6 { *r / a } else { 0.0 }; // b ⊘ Aₛ x
        });
    bp.backproject(&Tomo::new(ratio, Layout::Sinogram), &sub.geom, corr)
}

/// Ordered-Subset Expectation-Maximization.
///
/// MLEM restricted to ordered angle-subsets: each subset `s` applies one
/// multiplicative `x ← x ∘ Aₛᵀ(b ⊘ Aₛ x) ⊘ Aₛᵀ(1)` update over only its own
/// angles, so a single outer iteration performs `num_block` updates (faster
/// early convergence than MLEM). With `num_block ≤ 1` it is exactly [`mlem`].
/// The per-subset sensitivity `Aₛᵀ(1)` is geometry-only, so it is precomputed
/// once. Ports tomopy `libtomo/recon/osem.c`. `Aₛ` = forward projector over
/// subset `s`'s angles, `Aₛᵀ` = its back-projector.
fn osem(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    proj: &dyn ForwardProject,
    bp: &dyn FilteredBackproject,
) -> Result<Volume<f32>> {
    let n = grid_size(sino, params);
    let b = sino.as_layout(Layout::Sinogram);
    let nz = b.n_rows();
    let subsets = build_subsets(&b, geom, n, params, bp)?;

    let mut vol = Volume::new(Array3::from_elem((nz, n, n), 1.0)); // positive init
    let mut corr = Volume::new(Array3::zeros((nz, n, n)));
    for _ in 0..params.num_iter.max(1) {
        for sub in &subsets {
            subset_em_correction(&vol, sub, proj, bp, &mut corr)?;
            ndarray::Zip::from(&mut vol.array)
                .and(&corr.array)
                .and(&sub.sens)
                .for_each(|x, &c, &s| {
                    if s.abs() > 1e-6 {
                        *x = *x * c / s; // x ∘ Aₛᵀ(ratio) ⊘ Aₛᵀ(1)
                    }
                });
        }
    }
    Ok(vol)
}

/// One penalized-ML, quadratic-prior pixel update over a slice (tomopy
/// `ospml_quad.c`). With `corr = Aₛᵀ(b ⊘ Aₛ x)` and `sens = Aₛᵀ(1)`, each pixel
/// solves the De Pierro quadratic `2F·xʹ² + G·xʹ + E = 0` (positive root), where
///
/// - `E = −x·corr`            (data term),
/// - `F = Σ_g 2·reg·w_g·γ_g`,
/// - `G = sens − Σ_g 2·reg·w_g·γ_g·(x + x_g)`,
///
/// over the in-grid 8-neighbours `g`, where `w_g` is `1` (cardinal) / `1/√2`
/// (diagonal) normalized by the present-weight sum — the uniform form of
/// tomopy's separate interior/edge/corner weight tables (each already sums to
/// one). The edge factor `γ_g` selects the prior:
///
/// - `delta = None` → `γ_g = 1`: the plain quadratic prior (`ospml_quad`), where
///   `F` collapses to `2·reg` and `G` to `sens − 2·reg·(x + ⟨neighbours⟩)`.
/// - `delta = Some(δ)` → `γ_g = 1/(1 + |x − x_g|/δ)`: the edge-preserving hybrid
///   prior (`ospml_hybrid`), which down-weights smoothing across large jumps.
///
/// At `reg = 0` (`F = 0`) the quadratic degenerates to the linear root
/// `x·corr/sens`, i.e. exactly the MLEM/OSEM step (tomopy instead leaves
/// `reg = 0` pixels untouched; taking the correct limit makes `reg = 0` reduce
/// to [`osem`]). All reads use the pre-update slice `old`, matching tomopy's
/// read-then-write ordering.
fn penalized_ml_update(
    x: &mut ndarray::ArrayViewMut2<f32>,
    corr: &ndarray::ArrayView2<f32>,
    sens: &ndarray::ArrayView2<f32>,
    reg: f32,
    delta: Option<f32>,
) {
    const S: f32 = std::f32::consts::FRAC_1_SQRT_2;
    const NEIGHBORS: [(isize, isize, f32); 8] = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, S),
        (-1, 1, S),
        (1, -1, S),
        (1, 1, S),
    ];
    let (h, w) = x.dim();
    let old = x.to_owned();
    for i in 0..h {
        for j in 0..w {
            let xij = old[[i, j]];
            let e = -xij * corr[[i, j]];
            let (mut f, mut g) = (0.0f32, sens[[i, j]]);
            if reg != 0.0 {
                // Normalize the neighbour weights by the present-weight sum.
                let mut wtot = 0.0f32;
                for (di, dj, raw) in NEIGHBORS {
                    let (ni, nj) = (i as isize + di, j as isize + dj);
                    if ni >= 0 && ni < h as isize && nj >= 0 && nj < w as isize {
                        wtot += raw;
                    }
                }
                for (di, dj, raw) in NEIGHBORS {
                    let (ni, nj) = (i as isize + di, j as isize + dj);
                    if ni >= 0 && ni < h as isize && nj >= 0 && nj < w as isize {
                        let xg = old[[ni as usize, nj as usize]];
                        let gamma = match delta {
                            Some(d) => 1.0 / (1.0 + ((xij - xg) / d).abs()),
                            None => 1.0,
                        };
                        let coef = 2.0 * reg * (raw / wtot) * gamma;
                        f += coef;
                        g -= coef * (xij + xg);
                    }
                }
            }
            x[[i, j]] = if f != 0.0 {
                (-g + (g * g - 8.0 * f * e).sqrt()) / (4.0 * f)
            } else if g.abs() > 1e-6 {
                -e / g // reg = 0 ⟹ MLEM/OSEM step
            } else {
                xij
            };
        }
    }
}

/// Ordered-Subset Penalized Maximum-Likelihood (quadratic or hybrid prior).
///
/// OSEM with a smoothness penalty applied at each subset update via
/// [`penalized_ml_update`] (the De Pierro one-step-late form). Ports tomopy
/// `libtomo/recon/ospml_quad.c` (`delta = None`) and `ospml_hybrid.c`
/// (`delta = Some`); tomopy's `pml_quad`/`pml_hybrid` are these with `num_block
/// = 1` (`extern/recon.py`), so the caller passes `block_count`. The penalty
/// strength is `reg_par[0]` (0 ⇒ reduces to [`osem`]); the hybrid edge threshold
/// is `reg_par[1]` (passed in as `delta`).
fn ospml(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    proj: &dyn ForwardProject,
    bp: &dyn FilteredBackproject,
    block_count: usize,
    delta: Option<f32>,
) -> Result<Volume<f32>> {
    let n = grid_size(sino, params);
    let b = sino.as_layout(Layout::Sinogram);
    let nz = b.n_rows();
    let reg = params.reg_par.first().copied().unwrap_or(0.0);

    // Reuse the ordering rule but force the requested block count (pml_* ⇒ 1).
    let block_params = ReconParams {
        num_block: block_count,
        ..params.clone()
    };
    let subsets = build_subsets(&b, geom, n, &block_params, bp)?;

    let mut vol = Volume::new(Array3::from_elem((nz, n, n), 1.0)); // positive init
    let mut corr = Volume::new(Array3::zeros((nz, n, n)));
    for _ in 0..params.num_iter.max(1) {
        for sub in &subsets {
            subset_em_correction(&vol, sub, proj, bp, &mut corr)?; // Aₛᵀ(b ⊘ Aₛ x)
            for z in 0..nz {
                penalized_ml_update(
                    &mut vol.array.index_axis_mut(Axis(0), z),
                    &corr.array.index_axis(Axis(0), z),
                    &sub.sens.index_axis(Axis(0), z),
                    reg,
                    delta,
                );
            }
        }
    }
    Ok(vol)
}

/// Least-squares gradient descent (tomopy `grad.c`) and its Tikhonov variant
/// (`tikh.c`).
///
/// Minimizes ‖r·R x − b‖² (plus the Tikhonov term below) by gradient descent: the
/// data gradient is `g = 2r·Rᵀ(r·R x − b)`, the step `x ← x − λ g`. The step `λ`
/// is either fixed (`reg_par[0] ≥ 0`) or Barzilai–Borwein adaptive
/// (`reg_par[0] < 0`, `λ = ⟨Δx, Δg⟩ / ⟨Δg, Δg⟩`, first step `1e-3`). `x` iterates
/// in the r-scaled domain (tomopy scales the initial guess by `1/r`; from a zero
/// start that is a no-op) and is multiplied by `r` on return. `R` = forward
/// projector, `Rᵀ` = its *raw* adjoint — the back-projector bakes in a `π/nang`
/// factor (for FBP), which is divided back out here so the `r` normalization
/// holds. Unlike the EM methods this imposes no positivity.
///
/// `tikhonov = Some((reg1, prior))` adds the Tikhonov gradient
/// `2·reg1·(x − prior)` (penalizing ‖x − prior‖², a ridge term pulling the
/// r-scaled iterate toward `prior`); `None` is plain `grad`. tomopy's `tikh`
/// likewise adds the term in the scaled domain and does *not* rescale `reg_data`,
/// so `prior` is compared to the scaled iterate as-is.
///
/// The operator normalization `r = 1/√(ncols·nang/2)` is tomopy's heuristic,
/// tuned so `2r²·λmax(RᵀR) ≈ 2` (step-1 stability boundary) for *its* Siddon
/// projector. This crate's linear-interp adjoint pair has a larger operator norm,
/// so a unit fixed step diverges; use a smaller fixed step or the projector-
/// agnostic Barzilai–Borwein path (`reg_par[0] < 0`). The numeric result differs
/// from tomopy for the same reason FBP does — the projector model, not a porting
/// error.
fn gradient_descent(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    proj: &dyn ForwardProject,
    bp: &dyn FilteredBackproject,
    tikhonov: Option<(f32, Array3<f32>)>,
) -> Result<Volume<f32>> {
    let n = grid_size(sino, params);
    let b = sino.as_layout(Layout::Sinogram);
    let nz = b.n_rows();
    let nang = b.n_angles();
    let ncols = b.n_cols();

    let r = 1.0 / ((ncols * nang) as f32 / 2.0).sqrt();
    let adj_scale = nang as f32 / std::f32::consts::PI; // undo the back-projector's π/nang
    let fixed_step = params.reg_par.first().copied().unwrap_or(1.0);

    let mut vol = Volume::new(Array3::zeros((nz, n, n))); // r-scaled domain, init 0
    let mut recon0 = vol.array.clone();
    let mut grad0 = Array3::<f32>::zeros((nz, n, n));
    let mut ax = Tomo::new(Array3::zeros((nz, nang, ncols)), Layout::Sinogram);
    let mut bpv = Volume::new(Array3::zeros((nz, n, n)));
    let mut lambda = vec![0.0f32; nz];

    for it in 0..params.num_iter.max(1) {
        proj.project(&vol, geom, &mut ax)?; // R x
        let mut prox1 = ax.array.clone();
        ndarray::Zip::from(&mut prox1)
            .and(&b.array)
            .for_each(|p, &d| *p = *p * r - d); // r·R x − b
        bp.backproject(&Tomo::new(prox1, Layout::Sinogram), geom, &mut bpv)?;
        let mut grad = bpv.array.mapv(|v| 2.0 * r * adj_scale * v); // 2r·Rᵀ(…)

        // Tikhonov gradient 2·reg1·(x − prior), added in the scaled domain.
        if let Some((reg1, ref prior)) = tikhonov {
            ndarray::Zip::from(&mut grad)
                .and(&vol.array)
                .and(prior)
                .for_each(|g, &x, &p0| *g += 2.0 * reg1 * (x - p0));
        }

        // Step size per slice (computed from the previous iterate before saving).
        for (z, lam) in lambda.iter_mut().enumerate() {
            *lam = if fixed_step >= 0.0 {
                fixed_step
            } else if it == 0 {
                1e-3
            } else {
                let (mut num, mut den) = (0.0f32, 0.0f32);
                ndarray::Zip::from(vol.array.index_axis(Axis(0), z))
                    .and(recon0.index_axis(Axis(0), z))
                    .and(grad.index_axis(Axis(0), z))
                    .and(grad0.index_axis(Axis(0), z))
                    .for_each(|&x, &x0, &g, &g0| {
                        let dg = g - g0;
                        num += (x - x0) * dg;
                        den += dg * dg;
                    });
                if den != 0.0 {
                    num / den
                } else {
                    1e-3
                }
            };
        }

        recon0 = vol.array.clone();
        grad0 = grad.clone();
        for (z, &l) in lambda.iter().enumerate() {
            ndarray::Zip::from(vol.array.index_axis_mut(Axis(0), z))
                .and(grad.index_axis(Axis(0), z))
                .for_each(|x, &g| *x -= l * g); // x ← x − λ g
        }
    }

    vol.array.mapv_inplace(|v| v * r); // back to the unscaled domain
    Ok(vol)
}

/// Least-squares gradient descent (tomopy `grad.c`): [`gradient_descent`] with no
/// Tikhonov term.
fn grad(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    proj: &dyn ForwardProject,
    bp: &dyn FilteredBackproject,
) -> Result<Volume<f32>> {
    gradient_descent(sino, geom, params, proj, bp, None)
}

/// Tikhonov-regularized gradient descent (tomopy `tikh.c`): [`gradient_descent`]
/// with the term `2·reg_par[1]·(x − reg_data)`. `reg_data` is the prior image
/// `[nz, n, n]` flattened (tomopy defaults it to zeros, a ridge term toward 0);
/// an empty `reg_data` is that zero prior. With `reg_par[1] = 0` (or absent) the
/// term vanishes and this is bit-identical to `grad`.
fn tikh(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    proj: &dyn ForwardProject,
    bp: &dyn FilteredBackproject,
) -> Result<Volume<f32>> {
    let n = grid_size(sino, params);
    let nz = sino.n_rows();
    let reg1 = params.reg_par.get(1).copied().unwrap_or(0.0);
    let prior = if params.reg_data.is_empty() {
        Array3::zeros((nz, n, n))
    } else {
        Array3::from_shape_vec((nz, n, n), params.reg_data.clone()).map_err(|_| {
            Error::ShapeMismatch {
                expected: format!("reg_data of {nz}·{n}·{n} = {} elements", nz * n * n),
                found: format!("{} elements", params.reg_data.len()),
            }
        })?
    };
    gradient_descent(sino, geom, params, proj, bp, Some((reg1, prior)))
}

/// Total-variation regularized least squares (tomopy `tv.c`), a Chambolle–Pock
/// primal–dual solver of `min_x ½‖r·R x − b‖² + λ·TV(x)`.
///
/// Per iteration: ascend the two dual variables from the extrapolated primal
/// point `x̄` — the isotropic TV dual `pᵀᵛ ← Π_{‖·‖≤λ}(pᵀᵛ + c·∇x̄)` (forward
/// differences, pointwise projection onto the λ-ball) and the data dual
/// `pᵈ ← (pᵈ + c(r·R x̄ − b))/(1+c)` — then a primal step
/// `xₙ ← x_old − c·r·Rᵀ(pᵈ) + c·div(pᵀᵛ)` and the θ=1 over-relaxation
/// `x̄ ← 2xₙ − x_old`. `λ = reg_par[0]` is the TV strength; `c = 0.35` is tomopy's
/// fixed primal–dual step. As in `grad`, `x` lives in the r-scaled domain
/// (`r = 1/√(ncols·nang/2)`) and is rescaled by `r` on return, the back-projector's
/// `π/nang` factor is divided back out for the raw adjoint, and the result is the
/// final extrapolated point `x̄` (matching tomopy, which returns `recon`). No
/// positivity constraint.
///
/// The projector-model caveat from `grad` applies: tomopy's `r` and the hardcoded
/// `c = 0.35` (the Chambolle–Pock step, where convergence wants `c²·‖K‖² ≤ 1` for
/// `K = [r·R; ∇]`) are tuned for tomopy's Siddon projector. Unlike `grad`'s unit
/// fixed step, this iteration stays stable (finite) for this crate's linear-interp
/// adjoint pair across the tested `λ`/iteration range, but the numeric result
/// still differs from tomopy by the projector model.
fn tv(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    proj: &dyn ForwardProject,
    bp: &dyn FilteredBackproject,
) -> Result<Volume<f32>> {
    let n = grid_size(sino, params);
    let b = sino.as_layout(Layout::Sinogram);
    let nz = b.n_rows();
    let nang = b.n_angles();
    let ncols = b.n_cols();

    let r = 1.0 / ((ncols * nang) as f32 / 2.0).sqrt();
    let adj_scale = nang as f32 / std::f32::consts::PI; // undo the back-projector's π/nang
    let lambda = params.reg_par.first().copied().unwrap_or(1.0); // TV strength
    const C: f32 = 0.35; // tomopy's fixed primal–dual step

    let mut xbar = Volume::new(Array3::zeros((nz, n, n))); // extrapolated primal (recon)
    let mut x = Array3::<f32>::zeros((nz, n, n)); // primal iterate (update)
    let mut p0x = Array3::<f32>::zeros((nz, n, n)); // TV dual, x-gradient
    let mut p0y = Array3::<f32>::zeros((nz, n, n)); // TV dual, y-gradient
    let mut pd = Tomo::new(Array3::zeros((nz, nang, ncols)), Layout::Sinogram); // data dual
    let mut ax = Tomo::new(Array3::zeros((nz, nang, ncols)), Layout::Sinogram);
    let mut bpv = Volume::new(Array3::zeros((nz, n, n)));

    for _ in 0..params.num_iter.max(1) {
        proj.project(&xbar, geom, &mut ax)?; // R x̄

        // Data dual: pᵈ ← (pᵈ + c·r·R x̄ − c·b)/(1+c).
        ndarray::Zip::from(&mut pd.array)
            .and(&ax.array)
            .and(&b.array)
            .for_each(|q, &a, &d| *q = (*q + C * r * a - C * d) / (1.0 + C));
        bp.backproject(&pd, geom, &mut bpv)?; // (π/nang)·Rᵀ(pᵈ)

        for z in 0..nz {
            // TV dual ascent on x̄, then project onto the λ-ball (interior stencil;
            // the last row/col of pᵀᵛ stay 0, matching tomopy's loop bounds).
            let xb = xbar.array.index_axis(Axis(0), z);
            for iy in 0..n.saturating_sub(1) {
                for ix in 0..n.saturating_sub(1) {
                    let px = p0x[[z, iy, ix]] + C * (xb[[iy, ix + 1]] - xb[[iy, ix]]);
                    let py = p0y[[z, iy, ix]] + C * (xb[[iy + 1, ix]] - xb[[iy, ix]]);
                    let upd = ((px * px + py * py).sqrt() / lambda).max(1.0);
                    p0x[[z, iy, ix]] = px / upd;
                    p0y[[z, iy, ix]] = py / upd;
                }
            }
            // Primal step xₙ = x_old − c·r·Rᵀ(pᵈ) + c·div(pᵀᵛ), then x̄ = 2xₙ − x_old.
            for iy in 0..n {
                for ix in 0..n {
                    let x_old = x[[z, iy, ix]];
                    let mut u = x_old - C * r * adj_scale * bpv.array[[z, iy, ix]];
                    u += if ix == 0 {
                        C * p0x[[z, iy, 0]]
                    } else {
                        C * (p0x[[z, iy, ix]] - p0x[[z, iy, ix - 1]])
                    };
                    u += if iy == 0 {
                        C * p0y[[z, 0, ix]]
                    } else {
                        C * (p0y[[z, iy, ix]] - p0y[[z, iy - 1, ix]])
                    };
                    x[[z, iy, ix]] = u;
                    xbar.array[[z, iy, ix]] = 2.0 * u - x_old;
                }
            }
        }
    }

    xbar.array.mapv_inplace(|v| v * r); // back to the unscaled domain
    Ok(xbar)
}

/// Squared row norms `‖a‖²` for every ray, matching the `rows` layout. Computed
/// once and reused across all ART/BART iterations (geometry-invariant).
fn ray_norms(rows: &[Vec<RayRow>]) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|ang| {
            ang.iter()
                .map(|r| r.weights.iter().map(|w| w * w).sum())
                .collect()
        })
        .collect()
}

/// Algebraic Reconstruction Technique (tomopy `art.c`), a row-action Kaczmarz
/// solver of `R x = b`.
///
/// For each ray (angle `p`, detector `d`) in natural order, project onto the
/// ray's hyperplane `⟨a, x⟩ = b`: `x ← x + (b − ⟨a, x⟩)/‖a‖² · a`, where `a` is
/// the ray's sparse row of `R`. The reconstruction is updated immediately, so the
/// next ray sees it — this sequential per-ray update is what distinguishes ART
/// from the simultaneous methods and is why it consumes the [`RayProject`]
/// single-ray rows. Rows (and their `‖a‖²`) are geometry-only, so they are built
/// once and reused across iterations. No positivity constraint.
fn art(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    rp: &dyn RayProject,
) -> Result<Volume<f32>> {
    let n = grid_size(sino, params);
    let b = sino.as_layout(Layout::Sinogram);
    let nz = b.n_rows();
    let nang = b.n_angles();
    let ncols = b.n_cols();

    let rows = rp.ray_rows(geom, n)?;
    let norms = ray_norms(&rows);
    let bslab = b
        .array
        .as_slice()
        .ok_or_else(|| Error::InvalidParam("non-contiguous sinogram".into()))?;

    let npix = n * n;
    let mut recon = vec![0.0f32; nz * npix];
    for _ in 0..params.num_iter.max(1) {
        for ia in 0..nang {
            for d in 0..ncols {
                let row = &rows[ia][d];
                let norm2 = norms[ia][d];
                if norm2 == 0.0 {
                    continue;
                }
                for s in 0..nz {
                    let sl = &mut recon[s * npix..(s + 1) * npix];
                    let mut sim = 0.0f32;
                    for (k, &pix) in row.pixels.iter().enumerate() {
                        sim += sl[pix as usize] * row.weights[k];
                    }
                    let upd = (bslab[(s * nang + ia) * ncols + d] - sim) / norm2;
                    for (k, &pix) in row.pixels.iter().enumerate() {
                        sl[pix as usize] += upd * row.weights[k];
                    }
                }
            }
        }
    }
    let array = Array3::from_shape_vec((nz, n, n), recon)
        .map_err(|e| Error::InvalidParam(format!("art shape: {e}")))?;
    Ok(Volume::new(array))
}

/// Block Algebraic Reconstruction Technique (tomopy `bart.c`), ordered-subset
/// SART.
///
/// Like [`art`] but block-simultaneous: within an ordered subset every ray reads
/// the same (subset-start) reconstruction, the per-ray corrections accumulate
/// into `update[pix]` and the per-pixel total weights into `sum_dist[pix]`, and
/// the block is applied once at the subset end as
/// `x[pix] += update[pix] / sum_dist[pix]`. The subsets use the same `num_block`/
/// `ind_block` tiling as OSEM ([`ordered_subsets`]); `num_block = 1` is one
/// full-angle simultaneous update. No positivity constraint.
fn bart(
    sino: &Tomo<f32>,
    geom: &Geometry,
    params: &ReconParams,
    rp: &dyn RayProject,
) -> Result<Volume<f32>> {
    let n = grid_size(sino, params);
    let b = sino.as_layout(Layout::Sinogram);
    let nz = b.n_rows();
    let nang = b.n_angles();
    let ncols = b.n_cols();

    let rows = rp.ray_rows(geom, n)?;
    let norms = ray_norms(&rows);
    let subsets = ordered_subsets(nang, params);
    let bslab = b
        .array
        .as_slice()
        .ok_or_else(|| Error::InvalidParam("non-contiguous sinogram".into()))?;

    let npix = n * n;
    let mut recon = vec![0.0f32; nz * npix];
    let mut update = vec![0.0f32; npix];
    let mut sum_dist = vec![0.0f32; npix];
    for _ in 0..params.num_iter.max(1) {
        for s in 0..nz {
            let sl = &mut recon[s * npix..(s + 1) * npix];
            for subset in &subsets {
                update.iter_mut().for_each(|u| *u = 0.0);
                sum_dist.iter_mut().for_each(|u| *u = 0.0);
                for &p in subset {
                    for d in 0..ncols {
                        let row = &rows[p][d];
                        let mut sim = 0.0f32;
                        for (k, &pix) in row.pixels.iter().enumerate() {
                            let pix = pix as usize;
                            sim += sl[pix] * row.weights[k];
                            sum_dist[pix] += row.weights[k]; // accumulated for every ray
                        }
                        let norm2 = norms[p][d];
                        if norm2 != 0.0 {
                            let upd = (bslab[(s * nang + p) * ncols + d] - sim) / norm2;
                            for (k, &pix) in row.pixels.iter().enumerate() {
                                update[pix as usize] += upd * row.weights[k];
                            }
                        }
                    }
                }
                for pix in 0..npix {
                    if sum_dist[pix] != 0.0 {
                        sl[pix] += update[pix] / sum_dist[pix];
                    }
                }
            }
        }
    }
    let array = Array3::from_shape_vec((nz, n, n), recon)
        .map_err(|e| Error::InvalidParam(format!("bart shape: {e}")))?;
    Ok(Volume::new(array))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, DeviceKind};
    use crate::data::Layout;
    use crate::dtype::Dtype;
    use crate::geometry::Angles;

    /// A backend that advertises no capabilities, to exercise dispatch.
    struct NullBackend;
    impl Backend for NullBackend {
        fn name(&self) -> &'static str {
            "null"
        }
        fn device(&self) -> DeviceKind {
            DeviceKind::Cpu
        }
        fn supports(&self, _dt: Dtype) -> bool {
            false
        }
    }

    fn tiny_sino() -> (Tomo<f32>, Geometry) {
        let s = Tomo::new(Array3::<f32>::zeros((2, 4, 8)), Layout::Sinogram); // [row,angle,col]
        let g = Geometry::parallel(Angles::uniform(4, 0.0, std::f32::consts::PI), 8, 2, 1.0);
        (s, g)
    }

    #[test]
    fn analytic_without_capabilities_reports_missing_capability() {
        // FBP is filter-then-back-project, and the dispatcher acquires the FBP
        // filter first (so fourierrec, which needs no back-projector, can branch
        // before it). A capability-less backend therefore surfaces the filter as
        // the first missing capability.
        let (s, g) = tiny_sino();
        let err = recon(
            &s,
            &g,
            Algorithm::Fbp,
            &ReconParams::default(),
            &NullBackend,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::MissingCapability {
                capability: "FbpFilter",
                ..
            }
        ));
    }

    #[test]
    fn iterative_without_projector_reports_missing_capability() {
        let (s, g) = tiny_sino();
        let err = recon(
            &s,
            &g,
            Algorithm::Sirt,
            &ReconParams::default(),
            &NullBackend,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::MissingCapability {
                capability: "ForwardProject",
                ..
            }
        ));
    }
}
