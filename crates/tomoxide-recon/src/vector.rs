//! Vector-field tomography — faithful port of tomopy
//! `libtomo/recon/vector.c` plus the shared ray-tracing helpers in
//! `libtomo/recon/utils.c`.
//!
//! Unlike the scalar [`crate::recon`] dispatch (one sinogram in, one scalar
//! volume out), vector tomography reconstructs a **2-D vector field per slice**
//! from one, two, or three tilt datasets. Each ray measures the line integral
//! of the field component *along the ray direction* `(vx, vy)` (tomopy
//! `calc_simdata2`/`calc_simdata3`), and a SART-style update distributes the
//! residual back onto the two component grids. Because this is a different
//! shape of problem (multi-dataset in, vector-field out) it has its own API
//! rather than going through the `Algorithm` enum.
//!
//! Parity: this is a line-for-line port of tomopy's C kernels, including the
//! mixed `float`/`double` arithmetic, the `calc_quadrant` integer trick, and
//! the truncate-then-correct index rule, so it reproduces tomopy bit-for-bit.
//! Inputs follow tomopy's public `vector()` API — projection order
//! `(dt, dy, dx)` = (angles, slices, detector); internally swapped to the
//! `(dy, dt, dx)` contiguous order the C kernels consume (tomopy `init_tomo`
//! with `sinogram_order=False`). Outputs are `(dy, dx, dx)`, matching tomopy's
//! `recon_shape`.
//!
//! The index-based loops and wide helper signatures mirror tomopy's C
//! one-for-one (clippy's `needless_range_loop`/`too_many_arguments` lints are
//! allowed module-wide to keep the port literal).
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use ndarray::{Array3, ArrayView3};
use std::f32::consts::PI as PI_F32;
use std::f64::consts::PI as PI_F64;
use tomoxide_core::error::{Error, Result};

// ---------------------------------------------------------------------------
// Shared ray-tracing helpers (tomopy libtomo/recon/utils.c)
// ---------------------------------------------------------------------------

/// tomopy `preprocessing`: build the Cartesian grid lines and the detector
/// shift `mov` for one slice. `gridx`/`gridy` have `ry+1`/`rz+1` entries.
fn preprocessing(ry: i32, rz: i32, num_pixels: i32, center: f32, gridx: &mut [f32], gridy: &mut [f32]) -> f32 {
    for i in 0..=ry as usize {
        gridx[i] = -ry as f32 * 0.5 + i as f32;
    }
    for i in 0..=rz as usize {
        gridy[i] = -rz as f32 * 0.5 + i as f32;
    }
    // mov is float; the `+= 0.5` is a double literal but folds back to f32.
    let mut mov = (num_pixels as f32 - 1.0) * 0.5 - center;
    if mov - mov.floor() < 0.01 {
        mov += 0.01;
    }
    mov += 0.5;
    mov
}

/// tomopy `calc_quadrant`: the int32 rescaling trick, replicated exactly
/// (including the float arithmetic in the threshold comparisons).
fn calc_quadrant(theta_p: f32) -> i32 {
    const IPI_C: i32 = 340870420;
    let mut theta_i = (theta_p * IPI_C as f32) as i32;
    if theta_i < 0 {
        // C: `theta_i += (2.0f * M_PI * ipi_c)` — int += float ⇒ done in float.
        theta_i = (theta_i as f32 + 2.0 * PI_F32 * IPI_C as f32) as i32;
    }
    let ti = theta_i as f32;
    let half = 0.5 * PI_F32 * IPI_C as f32;
    let one = 1.0 * PI_F32 * IPI_C as f32;
    let onehalf = 1.5 * PI_F32 * IPI_C as f32;
    if (theta_i >= 0 && ti < half) || (ti >= one && ti < onehalf) {
        1
    } else {
        0
    }
}

/// tomopy `calc_coords`: intersection of the ray with every grid line.
#[allow(clippy::too_many_arguments)]
fn calc_coords(
    ry: i32,
    rz: i32,
    xi: f32,
    yi: f32,
    sin_p: f32,
    cos_p: f32,
    gridx: &[f32],
    gridy: &[f32],
    coordx: &mut [f32],
    coordy: &mut [f32],
) {
    let srcx = xi * cos_p - yi * sin_p;
    let srcy = xi * sin_p + yi * cos_p;
    let detx = -xi * cos_p - yi * sin_p;
    let dety = -xi * sin_p + yi * cos_p;
    let slope = (srcy - dety) / (srcx - detx);
    let islope = (srcx - detx) / (srcy - dety);
    for n in 0..=rz as usize {
        coordx[n] = islope * (gridy[n] - srcy) + srcx;
    }
    for n in 0..=ry as usize {
        coordy[n] = slope * (gridx[n] - srcx) + srcy;
    }
}

/// tomopy `trim_coords`: keep only the in-bounds intersections.
#[allow(clippy::too_many_arguments)]
fn trim_coords(
    ry: i32,
    rz: i32,
    coordx: &[f32],
    coordy: &[f32],
    gridx: &[f32],
    gridy: &[f32],
    ax: &mut [f32],
    ay: &mut [f32],
    bx: &mut [f32],
    by: &mut [f32],
) -> (usize, usize) {
    let mut asize = 0usize;
    let mut bsize = 0usize;
    let gridx_gt = gridx[0] + 0.01;
    let gridx_le = gridx[ry as usize] - 0.01;
    for n in 0..=rz as usize {
        if coordx[n] >= gridx_gt && coordx[n] <= gridx_le {
            ax[asize] = coordx[n];
            ay[asize] = gridy[n];
            asize += 1;
        }
    }
    let gridy_gt = gridy[0] + 0.01;
    let gridy_le = gridy[rz as usize] - 0.01;
    for n in 0..=ry as usize {
        if coordy[n] >= gridy_gt && coordy[n] <= gridy_le {
            bx[bsize] = gridx[n];
            by[bsize] = coordy[n];
            bsize += 1;
        }
    }
    (asize, bsize)
}

/// tomopy `sort_intersections`: merge the two intersection lists into a single
/// ray-ordered list. `ind_condition` is the quadrant (controls the a-list
/// direction).
#[allow(clippy::too_many_arguments)]
fn sort_intersections(
    ind_condition: i32,
    asize: usize,
    ax: &[f32],
    ay: &[f32],
    bsize: usize,
    bx: &[f32],
    by: &[f32],
    coorx: &mut [f32],
    coory: &mut [f32],
) -> usize {
    let (mut i, mut j, mut k) = (0usize, 0usize, 0usize);
    if ind_condition == 0 {
        while i < asize && j < bsize {
            if ax[asize - 1 - i] < bx[j] {
                coorx[k] = ax[asize - 1 - i];
                coory[k] = ay[asize - 1 - i];
                i += 1;
            } else {
                coorx[k] = bx[j];
                coory[k] = by[j];
                j += 1;
            }
            k += 1;
        }
        while i < asize {
            coorx[k] = ax[asize - 1 - i];
            coory[k] = ay[asize - 1 - i];
            i += 1;
            k += 1;
        }
        while j < bsize {
            coorx[k] = bx[j];
            coory[k] = by[j];
            j += 1;
            k += 1;
        }
    } else {
        while i < asize && j < bsize {
            if ax[i] < bx[j] {
                coorx[k] = ax[i];
                coory[k] = ay[i];
                i += 1;
            } else {
                coorx[k] = bx[j];
                coory[k] = by[j];
                j += 1;
            }
            k += 1;
        }
        while i < asize {
            coorx[k] = ax[i];
            coory[k] = ay[i];
            i += 1;
            k += 1;
        }
        while j < bsize {
            coorx[k] = bx[j];
            coory[k] = by[j];
            j += 1;
            k += 1;
        }
    }
    let _ = k;
    asize + bsize
}

/// tomopy `calc_dist2`: segment lengths between consecutive intersections and
/// the grid index each segment falls in. `dist` uses double `sqrt` (the C
/// promotes the float sum), and the index is the truncate-then-correct rule.
fn calc_dist2(
    ry: i32,
    rz: i32,
    csize: usize,
    coorx: &[f32],
    coory: &[f32],
    indx: &mut [i32],
    indy: &mut [i32],
    dist: &mut [f32],
) {
    if csize == 0 {
        return;
    }
    for n in 0..csize - 1 {
        let diffx = coorx[n + 1] - coorx[n];
        let diffy = coory[n + 1] - coory[n];
        let s = diffx * diffx + diffy * diffy; // f32, then double sqrt
        dist[n] = (s as f64).sqrt() as f32;
    }
    for n in 0..csize - 1 {
        let midx = (coorx[n + 1] + coorx[n]) * 0.5;
        let midy = (coory[n + 1] + coory[n]) * 0.5;
        let x1 = midx + ry as f32 * 0.5;
        let x2 = midy + rz as f32 * 0.5;
        let i1 = (midx + ry as f32 * 0.5) as i32; // truncation toward zero
        let i2 = (midy + rz as f32 * 0.5) as i32;
        indx[n] = i1 - ((i1 as f32 > x1) as i32);
        indy[n] = i2 - ((i2 as f32 > x2) as i32);
    }
}

// ---------------------------------------------------------------------------
// Input layout helper
// ---------------------------------------------------------------------------

/// Reorder a projection-order `(dt, dy, dx)` view into the `(dy, dt, dx)`
/// contiguous flat buffer the C kernels index as `[d + p*dx + s*dt*dx]`
/// (tomopy `init_tomo(sinogram_order=False)` = `swapaxes(0, 1)` + `ascontiguous`).
fn to_kernel_order(tomo: ArrayView3<f32>) -> Vec<f32> {
    let (dt, dy, dx) = tomo.dim();
    let mut out = vec![0.0f32; dy * dt * dx];
    for p in 0..dt {
        for s in 0..dy {
            for d in 0..dx {
                out[d + p * dx + s * dt * dx] = tomo[[p, s, d]];
            }
        }
    }
    out
}

/// Default rotation centres (tomopy `get_center`: `dx / 2` per slice).
fn resolve_center(center: Option<&[f32]>, dy: usize, dx: usize) -> Result<Vec<f32>> {
    match center {
        None => Ok(vec![dx as f32 / 2.0; dy]),
        Some(c) if c.len() == 1 => Ok(vec![c[0]; dy]),
        Some(c) if c.len() == dy => Ok(c.to_vec()),
        Some(c) => Err(Error::InvalidParam(format!(
            "vector: center length {} must be 1 or dy={dy}",
            c.len()
        ))),
    }
}

/// Per-call scratch buffers (sized like tomopy's per-call `malloc`s).
struct Scratch {
    gridx: Vec<f32>,
    gridy: Vec<f32>,
    coordx: Vec<f32>,
    coordy: Vec<f32>,
    ax: Vec<f32>,
    ay: Vec<f32>,
    bx: Vec<f32>,
    by: Vec<f32>,
    coorx: Vec<f32>,
    coory: Vec<f32>,
    dist: Vec<f32>,
    indx: Vec<i32>,
    indy: Vec<i32>,
}

impl Scratch {
    fn new(ngridx: usize, ngridy: usize) -> Self {
        let nn = ngridx + ngridy;
        Scratch {
            gridx: vec![0.0; ngridx + 1],
            gridy: vec![0.0; ngridy + 1],
            coordx: vec![0.0; ngridy + 1],
            coordy: vec![0.0; ngridx + 1],
            ax: vec![0.0; nn],
            ay: vec![0.0; nn],
            bx: vec![0.0; nn],
            by: vec![0.0; nn],
            coorx: vec![0.0; nn],
            coory: vec![0.0; nn],
            dist: vec![0.0; nn],
            indx: vec![0; nn + 1],
            indy: vec![0; nn + 1],
        }
    }
}

/// `yi` for detector pixel `d` (tomopy: `(1 - dx)/2.0 + d + mov`, done in
/// double then stored to float).
#[inline]
fn calc_yi(dx: i32, d: i32, mov: f32) -> f32 {
    ((1 - dx) as f64 / 2.0 + d as f64 + mov as f64) as f32
}

/// `(vx, vy)` ray direction for one detector pixel (tomopy uses double `sqrt`
/// of `pow(..,2)`).
#[inline]
fn ray_dir(xi: f32, yi: f32, sin_p: f32, cos_p: f32) -> (f32, f32) {
    let srcx = xi * cos_p - yi * sin_p;
    let srcy = xi * sin_p + yi * cos_p;
    let detx = -xi * cos_p - yi * sin_p;
    let dety = -xi * sin_p + yi * cos_p;
    let dv = (((srcx - detx) as f64).powi(2) + ((srcy - dety) as f64).powi(2)).sqrt() as f32;
    let vx = (srcx - detx) / dv;
    let vy = (srcy - dety) / dv;
    (vx, vy)
}

#[inline]
fn theta_mod(theta: f32) -> f32 {
    // fmod(theta, 2*M_PI) in double, stored to float.
    (theta as f64 % (2.0 * PI_F64)) as f32
}

// ---------------------------------------------------------------------------
// vector (single dataset → 2-component field)
// ---------------------------------------------------------------------------

/// Single-dataset vector tomography (tomopy `vector`).
///
/// `tomo` is `(dt, dy, dx)` (angles, slices, detector). Returns the two field
/// components `(recon1, recon2)`, each `(dy, dx, dx)`.
pub fn vector(
    tomo: ArrayView3<f32>,
    theta: &[f32],
    center: Option<&[f32]>,
    num_iter: usize,
) -> Result<(Array3<f32>, Array3<f32>)> {
    let (dt, dy, dx) = tomo.dim();
    if theta.len() != dt {
        return Err(Error::InvalidParam(format!(
            "vector: theta length {} must equal dt={dt}",
            theta.len()
        )));
    }
    let center = resolve_center(center, dy, dx)?;
    let data = to_kernel_order(tomo);
    let (ngridx, ngridy) = (dx, dx); // num_gridx = num_gridy = tomo.shape[2]
    let mut recon1 = vec![0.0f32; dy * ngridx * ngridy];
    let mut recon2 = vec![0.0f32; dy * ngridx * ngridy];
    let mut sc = Scratch::new(ngridx, ngridy);

    let (rx, rz) = (ngridx as i32, ngridy as i32);
    let dxi = dx as i32;

    for _ in 0..num_iter {
        let mut simdata = vec![0.0f32; dt * dy * dx];
        for s in 0..dy {
            let mov = preprocessing(rx, rz, dxi, center[s], &mut sc.gridx, &mut sc.gridy);
            let mut sum_dist = vec![0.0f32; ngridx * ngridy];
            let mut update1 = vec![0.0f32; ngridx * ngridy];
            let mut update2 = vec![0.0f32; ngridx * ngridy];
            for p in 0..dt {
                let theta_p = theta_mod(theta[p]);
                let quadrant = calc_quadrant(theta_p);
                let sin_p = theta_p.sin();
                let cos_p = theta_p.cos();
                for d in 0..dx {
                    let xi = (-(ngridx as i32) - ngridy as i32) as f32;
                    let yi = calc_yi(dxi, d as i32, mov);
                    let (vx, vy) = ray_dir(xi, yi, sin_p, cos_p);
                    calc_coords(rx, rz, xi, yi, sin_p, cos_p, &sc.gridx, &sc.gridy, &mut sc.coordx, &mut sc.coordy);
                    let (asize, bsize) = trim_coords(
                        rx, rz, &sc.coordx, &sc.coordy, &sc.gridx, &sc.gridy, &mut sc.ax, &mut sc.ay,
                        &mut sc.bx, &mut sc.by,
                    );
                    let csize = sort_intersections(
                        quadrant, asize, &sc.ax, &sc.ay, bsize, &sc.bx, &sc.by, &mut sc.coorx,
                        &mut sc.coory,
                    );
                    calc_dist2(rx, rz, csize, &sc.coorx, &sc.coory, &mut sc.indx, &mut sc.indy, &mut sc.dist);

                    // calc_simdata2: project both components onto (vx, vy).
                    let ind_data = d + p * dx + s * dt * dx;
                    let base = s * ngridx * ngridy;
                    if csize >= 1 {
                        for n in 0..csize - 1 {
                            let idx = sc.indy[n] as usize + sc.indx[n] as usize * ngridy + base;
                            simdata[ind_data] += (recon1[idx] * vx + recon2[idx] * vy) * sc.dist[n];
                        }
                    }

                    let mut sum_dist2 = 0.0f32;
                    if csize >= 1 {
                        for n in 0..csize - 1 {
                            sum_dist2 += sc.dist[n] * sc.dist[n];
                            sum_dist[sc.indy[n] as usize + sc.indx[n] as usize * ngridy] += sc.dist[n];
                        }
                    }
                    if sum_dist2 != 0.0 {
                        let upd = (data[ind_data] - simdata[ind_data]) / sum_dist2;
                        for n in 0..csize - 1 {
                            let g = sc.indy[n] as usize + sc.indx[n] as usize * ngridy;
                            update1[g] += upd * sc.dist[n] * vx;
                            update2[g] += upd * sc.dist[n] * vy;
                        }
                    }
                }
            }
            for m in 0..ngridx {
                for n in 0..ngridy {
                    let g = n + m * ngridy;
                    let r = g + s * ngridx * ngridy;
                    recon1[r] += update1[g] / sum_dist[g];
                    recon2[r] += update2[g] / sum_dist[g];
                }
            }
        }
        drop(simdata);
    }

    Ok((
        Array3::from_shape_vec((dy, ngridx, ngridy), recon1)
            .map_err(|e| Error::InvalidParam(format!("vector recon1 shape: {e}")))?,
        Array3::from_shape_vec((dy, ngridx, ngridy), recon2)
            .map_err(|e| Error::InvalidParam(format!("vector recon2 shape: {e}")))?,
    ))
}

// ---------------------------------------------------------------------------
// vector2 / vector3 (multi-dataset → 3-component field)
// ---------------------------------------------------------------------------

/// Per-axis flat index into a 3-component grid (tomopy `calc_simdata3` and the
/// matching `vectorN` write loops use identical formulas):
/// axis 0 ⇒ `indy + indx·rz + s·ry·rz`, axis 1 ⇒ `s + indx·rz + indy·ry·rz`,
/// axis 2 ⇒ `indx + s·rz + indy·ry·rz`.
#[inline]
fn idx_axis(axis: i32, indx: usize, indy: usize, s: usize, ngridx: usize, ngridy: usize) -> usize {
    match axis {
        1 => s + indx * ngridy + indy * ngridx * ngridy,
        2 => indx + s * ngridy + indy * ngridx * ngridy,
        _ => indy + indx * ngridy + s * ngridx * ngridy,
    }
}

/// One SART pass over a single dataset, projecting onto the two field
/// components selected by `axis` (`comp_a` ← update·vx, `comp_b` ← update·vy).
/// Faithful port of one `for(s)` block of tomopy `vector2`/`vector3`: it reads
/// the current estimate of `(comp_a, comp_b)` for `simdata`, then writes back
/// the residual-weighted update with the `sum_dist != 0` guard. tomopy always
/// passes `theta1`/`center1` here (the 2nd/3rd theta/center are accepted but
/// unused upstream), so callers do the same.
#[allow(clippy::too_many_arguments)]
fn vector_pass(
    data: &[f32],
    theta: &[f32],
    center: &[f32],
    dt: usize,
    dy: usize,
    dx: usize,
    ngridx: usize,
    ngridy: usize,
    axis: i32,
    comp_a: &mut [f32],
    comp_b: &mut [f32],
    sc: &mut Scratch,
) {
    let (rx, rz) = (ngridx as i32, ngridy as i32);
    let dxi = dx as i32;
    let mut simdata = vec![0.0f32; dt * dy * dx];
    for s in 0..dy {
        let mov = preprocessing(rx, rz, dxi, center[s], &mut sc.gridx, &mut sc.gridy);
        let mut sum_dist = vec![0.0f32; ngridx * ngridy];
        let mut update1 = vec![0.0f32; ngridx * ngridy];
        let mut update2 = vec![0.0f32; ngridx * ngridy];
        for p in 0..dt {
            let theta_p = theta_mod(theta[p]);
            let quadrant = calc_quadrant(theta_p);
            let sin_p = theta_p.sin();
            let cos_p = theta_p.cos();
            for d in 0..dx {
                let xi = (-(ngridx as i32) - ngridy as i32) as f32;
                let yi = calc_yi(dxi, d as i32, mov);
                let (vx, vy) = ray_dir(xi, yi, sin_p, cos_p);
                calc_coords(rx, rz, xi, yi, sin_p, cos_p, &sc.gridx, &sc.gridy, &mut sc.coordx, &mut sc.coordy);
                let (asize, bsize) = trim_coords(
                    rx, rz, &sc.coordx, &sc.coordy, &sc.gridx, &sc.gridy, &mut sc.ax, &mut sc.ay,
                    &mut sc.bx, &mut sc.by,
                );
                let csize = sort_intersections(
                    quadrant, asize, &sc.ax, &sc.ay, bsize, &sc.bx, &sc.by, &mut sc.coorx, &mut sc.coory,
                );
                calc_dist2(rx, rz, csize, &sc.coorx, &sc.coory, &mut sc.indx, &mut sc.indy, &mut sc.dist);

                let ind_data = d + p * dx + s * dt * dx;
                if csize >= 1 {
                    for n in 0..csize - 1 {
                        let ri = idx_axis(axis, sc.indx[n] as usize, sc.indy[n] as usize, s, ngridx, ngridy);
                        simdata[ind_data] += (comp_a[ri] * vx + comp_b[ri] * vy) * sc.dist[n];
                    }
                }

                let mut sum_dist2 = 0.0f32;
                if csize >= 1 {
                    for n in 0..csize - 1 {
                        sum_dist2 += sc.dist[n] * sc.dist[n];
                        sum_dist[sc.indy[n] as usize + sc.indx[n] as usize * ngridy] += sc.dist[n];
                    }
                }
                if sum_dist2 != 0.0 {
                    let upd = (data[ind_data] - simdata[ind_data]) / sum_dist2;
                    for n in 0..csize - 1 {
                        let g = sc.indy[n] as usize + sc.indx[n] as usize * ngridy;
                        update1[g] += upd * sc.dist[n] * vx;
                        update2[g] += upd * sc.dist[n] * vy;
                    }
                }
            }
        }
        for m in 0..ngridx {
            for n in 0..ngridy {
                let g = n + m * ngridy;
                if sum_dist[g] != 0.0 {
                    let ri = idx_axis(axis, m, n, s, ngridx, ngridy);
                    comp_a[ri] += update1[g] / sum_dist[g];
                    comp_b[ri] += update2[g] / sum_dist[g];
                }
            }
        }
    }
}

/// The two field components a pass with the given `axis` reads and writes:
/// axis 0 ⇒ (1, 2), axis 1 ⇒ (2, 3), axis 2 ⇒ (1, 3) (tomopy `calc_simdata3`).
fn axis_components(axis: i32) -> (usize, usize) {
    match axis {
        1 => (1, 2),
        2 => (0, 2),
        _ => (0, 1),
    }
}

/// Borrow two distinct slots of `[r0, r1, r2]` mutably at once.
fn pick2<'a>(
    r0: &'a mut [f32],
    r1: &'a mut [f32],
    r2: &'a mut [f32],
    a: usize,
    b: usize,
) -> (&'a mut [f32], &'a mut [f32]) {
    match (a, b) {
        (0, 1) => (r0, r1),
        (0, 2) => (r0, r2),
        (1, 2) => (r1, r2),
        _ => unreachable!("axis component pairs are (0,1), (0,2), (1,2)"),
    }
}

/// vector2/vector3 reconstruct a full 3-D vector field (`dx**3` voxels); their
/// axis-1/axis-2 write indices only stay in bounds when the slice count equals
/// the detector width. tomopy silently corrupts memory when `dy != dx`, so we
/// reject it explicitly.
fn require_cube(dy: usize, dx: usize, who: &str) -> Result<()> {
    if dy != dx {
        return Err(Error::InvalidParam(format!(
            "{who}: needs a cube (slices dy={dy} must equal detector dx={dx}); \
             the 3-D vector field is dx**3 voxels"
        )));
    }
    Ok(())
}

fn check_dims(tomo: ArrayView3<f32>, theta: &[f32], who: &str) -> Result<(usize, usize, usize)> {
    let (dt, dy, dx) = tomo.dim();
    if theta.len() != dt {
        return Err(Error::InvalidParam(format!(
            "{who}: theta length {} must equal dt={dt}",
            theta.len()
        )));
    }
    Ok((dt, dy, dx))
}

/// Two-dataset vector tomography (tomopy `vector2`). `axis1`/`axis2` default to
/// `1`/`2`. Returns the three field components, each `(dy, dx, dx)`.
///
/// As in tomopy, only `theta1`/`center1` drive the geometry of *both* passes;
/// `theta2`/`center2` are accepted for API compatibility but unused.
#[allow(clippy::too_many_arguments)]
pub fn vector2(
    tomo1: ArrayView3<f32>,
    tomo2: ArrayView3<f32>,
    theta1: &[f32],
    _theta2: &[f32],
    center1: Option<&[f32]>,
    _center2: Option<&[f32]>,
    num_iter: usize,
    axis1: i32,
    axis2: i32,
) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
    let (dt, dy, dx) = check_dims(tomo1, theta1, "vector2")?;
    if tomo2.dim() != (dt, dy, dx) {
        return Err(Error::InvalidParam("vector2: tomo1/tomo2 shapes must match".into()));
    }
    require_cube(dy, dx, "vector2")?;
    let center = resolve_center(center1, dy, dx)?;
    let data1 = to_kernel_order(tomo1);
    let data2 = to_kernel_order(tomo2);
    let (ngridx, ngridy) = (dx, dx);
    let sz = dy * ngridx * ngridy;
    let mut r1 = vec![0.0f32; sz];
    let mut r2 = vec![0.0f32; sz];
    let mut r3 = vec![0.0f32; sz];
    let mut sc = Scratch::new(ngridx, ngridy);

    for _ in 0..num_iter {
        let (a, b) = axis_components(axis1);
        let (ca, cb) = pick2(&mut r1, &mut r2, &mut r3, a, b);
        vector_pass(&data1, theta1, &center, dt, dy, dx, ngridx, ngridy, axis1, ca, cb, &mut sc);
        let (a, b) = axis_components(axis2);
        let (ca, cb) = pick2(&mut r1, &mut r2, &mut r3, a, b);
        vector_pass(&data2, theta1, &center, dt, dy, dx, ngridx, ngridy, axis2, ca, cb, &mut sc);
    }

    finalize3(dy, ngridx, ngridy, r1, r2, r3)
}

/// Three-dataset vector tomography (tomopy `vector3`). `axis1`/`axis2`/`axis3`
/// default to `0`/`1`/`2`. Only `theta1`/`center1` drive all three passes
/// (tomopy behaviour). Returns the three field components, each `(dy, dx, dx)`.
#[allow(clippy::too_many_arguments)]
pub fn vector3(
    tomo1: ArrayView3<f32>,
    tomo2: ArrayView3<f32>,
    tomo3: ArrayView3<f32>,
    theta1: &[f32],
    _theta2: &[f32],
    _theta3: &[f32],
    center1: Option<&[f32]>,
    _center2: Option<&[f32]>,
    _center3: Option<&[f32]>,
    num_iter: usize,
    axis1: i32,
    axis2: i32,
    axis3: i32,
) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
    let (dt, dy, dx) = check_dims(tomo1, theta1, "vector3")?;
    if tomo2.dim() != (dt, dy, dx) || tomo3.dim() != (dt, dy, dx) {
        return Err(Error::InvalidParam("vector3: all tomo shapes must match".into()));
    }
    require_cube(dy, dx, "vector3")?;
    let center = resolve_center(center1, dy, dx)?;
    let data1 = to_kernel_order(tomo1);
    let data2 = to_kernel_order(tomo2);
    let data3 = to_kernel_order(tomo3);
    let (ngridx, ngridy) = (dx, dx);
    let sz = dy * ngridx * ngridy;
    let mut r1 = vec![0.0f32; sz];
    let mut r2 = vec![0.0f32; sz];
    let mut r3 = vec![0.0f32; sz];
    let mut sc = Scratch::new(ngridx, ngridy);

    for _ in 0..num_iter {
        for (data, axis) in [(&data1, axis1), (&data2, axis2), (&data3, axis3)] {
            let (a, b) = axis_components(axis);
            let (ca, cb) = pick2(&mut r1, &mut r2, &mut r3, a, b);
            vector_pass(data, theta1, &center, dt, dy, dx, ngridx, ngridy, axis, ca, cb, &mut sc);
        }
    }

    finalize3(dy, ngridx, ngridy, r1, r2, r3)
}

fn finalize3(
    dy: usize,
    ngridx: usize,
    ngridy: usize,
    r1: Vec<f32>,
    r2: Vec<f32>,
    r3: Vec<f32>,
) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
    let mk = |v: Vec<f32>, name: &str| {
        Array3::from_shape_vec((dy, ngridx, ngridy), v)
            .map_err(|e| Error::InvalidParam(format!("vector {name} shape: {e}")))
    };
    Ok((mk(r1, "recon1")?, mk(r2, "recon2")?, mk(r3, "recon3")?))
}
