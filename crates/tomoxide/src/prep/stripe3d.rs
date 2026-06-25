//! 3-D stripe detection and masking (ports tomopy `prep/stripe.py:984`
//! `stripes_detect3d` and `:1058` `stripes_mask3d`, both backed by libtomo
//! `prep/stripes_detect3d.c`). Daniil Kazantsev's algorithm (Kazantsev 2023):
//! works with full and partial stripes of constant or varying intensity. See
//! `docs/PORTING.md` §D.
//!
//! Unlike the [`stripe`](crate::stripe) removal methods, these do not modify the
//! data. [`stripes_detect3d`] returns a `[0, 1]` weights volume in which stripe
//! *edges* are highlighted with smaller (e.g. `< 0.5`) values, and
//! [`stripes_mask3d`] turns those weights into a binary `bool` mask of the
//! stripe regions. Both kernels are pure `f32`/`bool` arithmetic with no FFT, so
//! they reproduce tomopy bit-for-bit (Δ = 0).
//!
//! `stripes_detect3d` is four full-volume passes over the
//! `[angle, detY(depth), detX(horizontal)]` stack:
//!
//! 1. a gentle 6-stencil 3-D mean smoothing (used only as the fallback value in
//!    the zero-gradient corner of pass 3),
//! 2. a horizontal forward-difference gradient along detX with step 2,
//! 3. a per-voxel ratio between the mean `|gradient|` in a 2-D plate *parallel*
//!    to the stripe (the angle×depth plane at fixed detX) and the mean
//!    orthogonal to it (left/right along detX) — small/large or large/small at
//!    a stripe edge,
//! 4. a vertical (along-angle) median filter of that ratio map to drop
//!    inconsistent short features.
//!
//! `stripes_mask3d` thresholds the weights, then enforces stripe consistency in
//! depth and along-angle, drops short stripes, and iteratively merges nearby
//! ones. Each pass reads the previous pass's full result and writes a fresh
//! buffer (the C kernel's read/write `mask`/`Output` split), so neighbour reads
//! never see a half-updated mask.

use ndarray::Array3;

use crate::data::{Layout, Tomo};
use crate::error::{Error, Result};

/// Detect stripes in a 3-D projection stack and return per-voxel weights.
///
/// `tomo` is read in `[angle, detY(depth), detX(horizontal)]` orientation
/// (tomopy's projection order); stripes are features along the angle axis. The
/// returned weights are in `[0, 1]` and in the same `[angle, detY, detX]`
/// orientation, with stripe edges marked by smaller values.
///
/// * `size` — half-length of the 1-D vertical (along-angle) median filter that
///   suppresses false detections; must be `> 0` and `<= n_angles / 2`.
/// * `radius` — pixel radius of the 3-D stencil for the mean ratio between the
///   angular and detX orientations of the detX gradient (use 1, 2, or 3).
///
/// Bit-exact against tomopy 1.15.3 `stripes_detect3d`.
pub fn stripes_detect3d(tomo: &Tomo<f32>, size: usize, radius: usize) -> Result<Array3<f32>> {
    let proj = tomo.to_layout(Layout::Projection);
    let input = &proj.array; // [angle(dz), detY(dy), detX(dx)]
    let (dz, dy, dx) = input.dim();

    if dz == 0 || dy == 0 || dx == 0 {
        return Err(Error::InvalidParam(
            "stripes_detect3d: every dimension must be non-zero".into(),
        ));
    }
    if size == 0 || size > dz / 2 {
        return Err(Error::InvalidParam(format!(
            "stripes_detect3d: size ({size}) must be > 0 and <= n_angles/2 ({})",
            dz / 2
        )));
    }

    let radius = radius as isize;
    let window_half = size as isize; // window_halflength_vertical
    let window_full = 2 * size + 1; // window_fulllength
                                    // C: `(int)(0.5F * window_fulllength) - 1` — float multiply, then truncate.
    let midval_index = ((0.5f32 * window_full as f32) as isize - 1) as usize;

    // Pass 1: 6-stencil mean smoothing. Its result survives into the output only
    // in the zero-gradient corner of pass 3, but the C kernel computes it
    // unconditionally, so we do too.
    let mut temp = smooth_mean(input);
    // Pass 2: horizontal (detX) forward-difference gradient, step 2.
    let grad = gradient_x_step2(input);
    // Pass 3: parallel/orthogonal mean-ratio map, written over `temp` in place.
    ratio_map(&grad, &mut temp, radius);
    // Pass 4: vertical (along-angle) median filter of the ratio map.
    Ok(vertical_median(&temp, window_half, midval_index))
}

/// `mean_stride3d`: gentle 6-neighbour + self mean (factor `0.1428`), with
/// clamp-to-mirror boundaries chosen exactly as the C kernel.
fn smooth_mean(input: &Array3<f32>) -> Array3<f32> {
    let (dz, dy, dx) = input.dim();
    let (dzi, dyi, dxi) = (dz as isize, dy as isize, dx as isize);
    let mut out = Array3::<f32>::zeros((dz, dy, dx));
    for k in 0..dz {
        for j in 0..dy {
            for i in 0..dx {
                let (ki, ji, ii) = (k as isize, j as isize, i as isize);
                let mut i1 = ii - 1;
                let mut i2 = ii + 1;
                let mut j1 = ji - 1;
                let mut j2 = ji + 1;
                let mut k1 = ki - 1;
                let mut k2 = ki + 1;
                if i1 < 0 {
                    i1 = i2;
                }
                if i2 >= dxi {
                    i2 = i1;
                }
                if j1 < 0 {
                    j1 = j2;
                }
                if j2 >= dyi {
                    j2 = j1;
                }
                if k1 < 0 {
                    k1 = k2;
                }
                if k2 >= dzi {
                    k2 = k1;
                }
                let val1 = input[[k, j, i1 as usize]];
                let val2 = input[[k, j, i2 as usize]];
                let val3 = input[[k, j1 as usize, i]];
                let val4 = input[[k, j2 as usize, i]];
                let val5 = input[[k1 as usize, j, i]];
                let val6 = input[[k2 as usize, j, i]];
                out[[k, j, i]] =
                    0.1428f32 * (input[[k, j, i]] + val1 + val2 + val3 + val4 + val5 + val6);
            }
        }
    }
    out
}

/// `gradient3D_local(axis = 0, step = 2)`: forward difference along detX
/// (`input[.., .., i+2] - input[.., .., i]`), mirroring back at the far edge.
fn gradient_x_step2(input: &Array3<f32>) -> Array3<f32> {
    let (dz, dy, dx) = input.dim();
    let dxi = dx as isize;
    let mut out = Array3::<f32>::zeros((dz, dy, dx));
    for k in 0..dz {
        for j in 0..dy {
            for i in 0..dx {
                let mut i1 = i as isize + 2;
                if i1 >= dxi {
                    i1 = i as isize - 2;
                }
                out[[k, j, i]] = input[[k, j, i1 as usize]] - input[[k, j, i]];
            }
        }
    }
    out
}

/// `ratio_mean_stride3d`: per voxel, the ratio between the mean `|gradient|` in
/// the angle×depth plate parallel to the stripe and the means orthogonal to it
/// (right then left along detX), keeping the smaller of the two orientations.
/// Reads `grad`, read-modify-writes `temp` (whose incoming value — the smoothed
/// volume — is the fallback kept when both means are zero).
fn ratio_map(grad: &Array3<f32>, temp: &mut Array3<f32>, radius: isize) {
    let (dz, dy, dx) = grad.dim();
    let (dzi, dyi, dxi) = (dz as isize, dy as isize, dx as isize);
    let diameter = 2 * radius + 1;
    let all_pixels_window = (diameter * diameter) as f32;
    let horiz_norm = (radius * 3) as f32;
    for k in 0..dz {
        for j in 0..dy {
            for i in 0..dx {
                let (ki, ji) = (k as isize, j as isize);

                // mean of |gradient| in the 2-D plate parallel to stripes (i fixed).
                let mut mean_plate = 0.0f32;
                for j_m in -radius..=radius {
                    let mut j1 = ji + j_m;
                    if j1 < 0 || j1 >= dyi {
                        j1 = ji - j_m;
                    }
                    for k_m in -radius..=radius {
                        let mut k1 = ki + k_m;
                        if k1 < 0 || k1 >= dzi {
                            k1 = ki - k_m;
                        }
                        mean_plate += grad[[k1 as usize, j1 as usize, i]].abs();
                    }
                }
                mean_plate /= all_pixels_window;

                // mean orthogonal to stripes, to the right (i+1 .. i+radius).
                let mut mean_horiz = 0.0f32;
                for j_m in -1..=1 {
                    let mut j1 = ji + j_m;
                    if j1 < 0 || j1 >= dyi {
                        j1 = ji - j_m;
                    }
                    for i_m in 1..=radius {
                        let mut i1 = i as isize + i_m;
                        if i1 >= dxi {
                            i1 = i as isize - i_m;
                        }
                        mean_horiz += grad[[k, j1 as usize, i1 as usize]].abs();
                    }
                }
                mean_horiz /= horiz_norm;

                // and to the left (i-radius .. i-1), symmetrically.
                let mut mean_horiz2 = 0.0f32;
                for j_m in -1..=1 {
                    let mut j1 = ji + j_m;
                    if j1 < 0 || j1 >= dyi {
                        j1 = ji - j_m;
                    }
                    for i_m in -radius..=-1 {
                        let mut i1 = i as isize + i_m;
                        if i1 < 0 {
                            i1 = i as isize - i_m;
                        }
                        mean_horiz2 += grad[[k, j1 as usize, i1 as usize]].abs();
                    }
                }
                mean_horiz2 /= horiz_norm;

                // The ratio is small/large or large/small at a stripe edge; keep
                // the smaller of the right- and left-orthogonal ratios. The
                // incoming `temp` value (smoothed) is the fallback if unwritten.
                let mut out_val = temp[[k, j, i]];
                let mut min_val = 0.0f32;
                if mean_horiz >= mean_plate && mean_horiz != 0.0 {
                    out_val = mean_plate / mean_horiz;
                }
                if mean_horiz < mean_plate && mean_plate != 0.0 {
                    out_val = mean_horiz / mean_plate;
                }
                if mean_horiz2 >= mean_plate && mean_horiz2 != 0.0 {
                    min_val = mean_plate / mean_horiz2;
                }
                if mean_horiz2 < mean_plate && mean_plate != 0.0 {
                    min_val = mean_horiz2 / mean_plate;
                }
                if out_val > min_val {
                    out_val = min_val;
                }
                temp[[k, j, i]] = out_val;
            }
        }
    }
}

/// `vertical_median_stride3d`: 1-D median filter of the ratio map along the
/// angle axis, with mirror boundaries and the same off-by-one median index the
/// C kernel uses (`(int)(0.5 * window_fulllength) - 1`).
fn vertical_median(temp: &Array3<f32>, window_half: isize, midval_index: usize) -> Array3<f32> {
    let (dz, dy, dx) = temp.dim();
    let dzi = dz as isize;
    let mut out = Array3::<f32>::zeros((dz, dy, dx));
    let mut values: Vec<f32> = Vec::with_capacity((2 * window_half + 1) as usize);
    for k in 0..dz {
        for j in 0..dy {
            for i in 0..dx {
                values.clear();
                for k_m in -window_half..=window_half {
                    let mut k1 = k as isize + k_m;
                    if k1 < 0 || k1 >= dzi {
                        k1 = k as isize - k_m;
                    }
                    values.push(temp[[k1 as usize, j, i]]);
                }
                values.sort_by(|a, b| a.total_cmp(b));
                out[[k, j, i]] = values[midval_index];
            }
        }
    }
    out
}

/// Turn a [`stripes_detect3d`] weights volume into a binary stripe mask.
///
/// `weights` is the `[0, 1]` `[angle, detY(depth), detX(horizontal)]` volume
/// from [`stripes_detect3d`]. The returned `bool` mask is `true` where stripes
/// are present. The pipeline thresholds the weights, then prunes the candidate
/// mask by stripe consistency in depth and along-angle, drops short stripes, and
/// iteratively merges nearby ones.
///
/// * `threshold` — weights `<=` this are stripe candidates (good range 0.5–0.7).
/// * `min_stripe_length` — minimum along-angle length, `> 0` and `< n_angles`.
/// * `min_stripe_depth` — minimum detY depth, `>= 0` and `< n_detY`.
/// * `min_stripe_width` — minimum detX width and the merge-iteration count,
///   `> 0` and `< n_detX`.
/// * `sensitivity_perc` — strictness of the length/depth thresholds in `(0, 100]`.
///
/// Bit-exact against tomopy 1.15.3 `stripes_mask3d`.
pub fn stripes_mask3d(
    weights: &Array3<f32>,
    threshold: f32,
    min_stripe_length: usize,
    min_stripe_depth: usize,
    min_stripe_width: usize,
    sensitivity_perc: f32,
) -> Result<Array3<bool>> {
    let (dz, dy, dx) = weights.dim();
    if dz == 0 || dy == 0 || dx == 0 {
        return Err(Error::InvalidParam(
            "stripes_mask3d: every dimension must be non-zero".into(),
        ));
    }
    if min_stripe_length == 0 || min_stripe_length >= dz {
        return Err(Error::InvalidParam(format!(
            "stripes_mask3d: min_stripe_length ({min_stripe_length}) must be > 0 and < n_angles ({dz})"
        )));
    }
    if min_stripe_depth >= dy {
        return Err(Error::InvalidParam(format!(
            "stripes_mask3d: min_stripe_depth ({min_stripe_depth}) must be < n_detY ({dy})"
        )));
    }
    if min_stripe_width == 0 || min_stripe_width >= dx {
        return Err(Error::InvalidParam(format!(
            "stripes_mask3d: min_stripe_width ({min_stripe_width}) must be > 0 and < n_detX ({dx})"
        )));
    }
    if !(sensitivity_perc > 0.0 && sensitivity_perc <= 100.0) {
        return Err(Error::InvalidParam(format!(
            "stripes_mask3d: sensitivity_perc ({sensitivity_perc}) must be in (0, 100]"
        )));
    }

    // Threshold: stripe candidates are the weights at or below `threshold`.
    let mut mask = weights.mapv(|w| w <= threshold);
    // Depth consistency, then vertical (along-angle) consistency.
    mask = remove_inconsistent_stripes(
        &mask,
        min_stripe_length,
        min_stripe_depth,
        sensitivity_perc,
        true,
    );
    mask = remove_inconsistent_stripes(
        &mask,
        min_stripe_length,
        min_stripe_depth,
        sensitivity_perc,
        false,
    );
    // Drop stripes shorter than half the minimum length.
    mask = remove_short_stripes(&mask, min_stripe_length);
    // Iteratively merge nearby stripes (one iteration per unit of min width).
    for _ in 0..min_stripe_width {
        mask = merge_stripes(&mask, min_stripe_width);
    }
    Ok(mask)
}

/// `remove_inconsistent_stripes`: prune candidates by stripe consistency. With
/// `depth_pass`, drop voxels whose depth (detY) extent exceeds the depth
/// threshold (stripes are shallow); otherwise drop voxels whose along-angle run
/// is shorter than the vertical threshold. Reads `src`, writes a fresh mask.
fn remove_inconsistent_stripes(
    src: &Array3<bool>,
    min_len: usize,
    min_depth: usize,
    sensitivity: f32,
    depth_pass: bool,
) -> Array3<bool> {
    let (dz, dy, dx) = src.dim();
    let mut out = src.clone();
    if depth_pass {
        if min_depth == 0 {
            return out; // C: inner `if(stripe_depth_min != 0)` guard.
        }
        let half = (min_depth / 2) as isize;
        // C: `(int)((0.01F * sensitivity) * stripe_depth_min)` — f32, truncated.
        let threshold = ((0.01f32 * sensitivity) * min_depth as f32) as i32;
        let dyi = dy as isize;
        for k in 0..dz {
            for j in 0..dy {
                for i in 0..dx {
                    if !src[[k, j, i]] {
                        continue;
                    }
                    let mut counter = 0i32;
                    for y_m in -half..=half {
                        let mut y1 = j as isize + y_m;
                        if y1 < 0 || y1 >= dyi {
                            y1 = j as isize - y_m;
                        }
                        if src[[k, y1 as usize, i]] {
                            counter += 1;
                        }
                    }
                    if counter > threshold {
                        out[[k, j, i]] = false;
                    }
                }
            }
        }
    } else {
        let half = (min_len / 2) as isize;
        let threshold = ((0.01f32 * sensitivity) * min_len as f32) as i32;
        let dzi = dz as isize;
        for k in 0..dz {
            for j in 0..dy {
                for i in 0..dx {
                    if !src[[k, j, i]] {
                        continue;
                    }
                    let mut counter = 0i32;
                    for k_m in -half..=half {
                        let mut k1 = k as isize + k_m;
                        if k1 < 0 || k1 >= dzi {
                            k1 = k as isize - k_m;
                        }
                        if src[[k1 as usize, j, i]] {
                            counter += 1;
                        }
                    }
                    if counter < threshold {
                        out[[k, j, i]] = false;
                    }
                }
            }
        }
    }
    out
}

/// `remove_short_stripes`: drop a candidate whose along-angle run is shorter
/// than `min_stripe_length / 2`.
fn remove_short_stripes(src: &Array3<bool>, min_len: usize) -> Array3<bool> {
    let (dz, dy, dx) = src.dim();
    let mut out = src.clone();
    let half = (min_len / 2) as isize;
    let dzi = dz as isize;
    for k in 0..dz {
        for j in 0..dy {
            for i in 0..dx {
                if !src[[k, j, i]] {
                    continue;
                }
                let mut counter = 0i32;
                for k_m in -half..=half {
                    let mut k1 = k as isize + k_m;
                    if k1 < 0 || k1 >= dzi {
                        k1 = k as isize - k_m;
                    }
                    if src[[k1 as usize, j, i]] {
                        counter += 1;
                    }
                }
                if counter < half as i32 {
                    out[[k, j, i]] = false;
                }
            }
        }
    }
    out
}

/// `merge_stripes`: fill a non-stripe voxel that is bracketed by stripes on both
/// the left/right (within `min_stripe_width / 2`) or above/below (within
/// `2 * min_stripe_width`). One iteration; the caller loops `min_stripe_width`
/// times.
fn merge_stripes(src: &Array3<bool>, min_width: usize) -> Array3<bool> {
    let (dz, dy, dx) = src.dim();
    let mut out = src.clone();
    let half_w = (min_width / 2) as isize;
    let vert_len = (2 * min_width) as isize;
    let (dzi, dxi) = (dz as isize, dx as isize);
    for k in 0..dz {
        for j in 0..dy {
            for i in 0..dx {
                if src[[k, j, i]] {
                    continue;
                }
                let ii = i as isize;
                // Is there a stripe to the left within half the min width?
                let mut mask_left = false;
                for x in -half_w..=0 {
                    let mut x_l = ii + x;
                    if x_l < 0 {
                        x_l = ii - x;
                    }
                    if src[[k, j, x_l as usize]] {
                        mask_left = true;
                        break;
                    }
                }
                // ... and to the right?
                let mut mask_right = false;
                for x in 0..=half_w {
                    let mut x_r = ii + x;
                    if x_r >= dxi {
                        x_r = ii - x;
                    }
                    if src[[k, j, x_r as usize]] {
                        mask_right = true;
                        break;
                    }
                }
                let mut out_val = false;
                if mask_left && mask_right {
                    out_val = true;
                }
                // Otherwise try the vertical (along-angle) bracketing.
                if !out_val {
                    let ki = k as isize;
                    let mut mask_up = false;
                    for x in -vert_len..=0 {
                        let mut k_u = ki + x;
                        if k_u < 0 {
                            k_u = ki - x;
                        }
                        if src[[k_u as usize, j, i]] {
                            mask_up = true;
                            break;
                        }
                    }
                    let mut mask_down = false;
                    for x in 0..=vert_len {
                        let mut k_d = ki + x;
                        if k_d >= dzi {
                            k_d = ki - x;
                        }
                        if src[[k_d as usize, j, i]] {
                            mask_down = true;
                            break;
                        }
                    }
                    if mask_up && mask_down {
                        out_val = true;
                    }
                }
                out[[k, j, i]] = out_val;
            }
        }
    }
    out
}
