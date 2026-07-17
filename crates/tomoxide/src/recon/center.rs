//! Rotation-center finding (ports tomopy `recon/rotation.py` + tomocupy
//! `find_center.py`). `find_center` (entropy), `find_center_vo` (Nghia Vo's
//! method), `find_center_pc` (phase correlation), and `write_center`
//! (reconstruct a slice across a range of centers) are implemented;
//! `find_center_sift` is implemented behind the `sift-center` feature (it links
//! OpenCV). See `docs/PORTING.md` §C.
//!
//! Laminography alignment lives here too, since it is the same question asked of
//! a second axis: `find_center_rings` reads the axis off the raw projections (and
//! says whether the scan was aligned at acquisition at all), and
//! `lamino_tilt_scan` scores the tilt. `docs/LAMINOGRAPHY_ALIGNMENT.md` is the
//! method both implement.

use crate::backend::{Backend, Fft};
use crate::data::{Layout, Tomo};
use crate::dtype::Complex32;
use crate::error::{Error, Result};
use crate::geometry::{Angles, Beam, Center, Detector, Geometry};
use ndarray::{Array2, Array3, ArrayViewMut2, Axis, Slice};

/// What [`find_center_rings`] found, and whether to believe it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RingCenter {
    /// Estimated rotation-axis column, sub-pixel.
    pub center: f32,
    /// Robust prominence of the registration peak, `(peak − median) / MAD` of the
    /// correlation profile. This is the number that says whether `center` means
    /// anything: a bullseye registers as a sharp peak, arcs that never close do
    /// not peak at all. See [`find_center_rings`] for the two measured
    /// calibration points.
    pub prominence: f32,
    /// `prominence >= 8.0`. False ⇒ **do not use `center`**: the scan is very
    /// likely mis-aligned at acquisition, which no reconstruction geometry
    /// repairs.
    pub trustworthy: bool,
}

/// Rotation-axis column from the ring pattern of a 360° scan — the cheapest
/// estimator here (one pass, no reconstruction) and the only one validated for
/// **laminography**, where `find_center_vo`/`find_center_pc` do not apply.
///
/// Over a full turn every object point at radius `r` from the axis traces
/// `x(θ) = c + r·cos(θ + φ)`, so the mean projection smears each point into a
/// ring centred on the axis: a bullseye whose centre column is `c`. Rings are
/// mirror-symmetric about that column, so registering the mean projection
/// against its own left–right flip puts the axis at `dx = 2·(c − nx/2)`. `tomo`
/// is the flat-corrected, minus-log projection stack; every `step`-th projection
/// is averaged (`step = 10` over 1800 projections is plenty — the rings are a
/// bulk feature).
///
/// **Why this exists when three other `find_center_*` do.** The laminographic
/// tilt gives the axis a component along the beam, so a 180° rotation is not a
/// mirror of the object and the 0°/180° symmetry every other estimator assumes is
/// gone: on the scan below `find_center_pc`-style mirror registration scattered
/// 395…607 against a known 396. The 2-D ring pattern does not depend on that
/// symmetry. (A 1-D column profile of the same mean projection does not work
/// either — it gave 847 — because the sample is wider than the field and
/// truncation destroys the profile's symmetry while leaving the rings intact.)
///
/// **The estimate carries its own alibi.** `docs/LAMINOGRAPHY_ALIGNMENT.md`
/// prescribes reading the bullseye by eye and treating eye-vs-correlation
/// disagreement as the misalignment flag, because on a mis-aligned scan the
/// lopsided arcs drag the correlation peak. That disagreement is measurable
/// without the eye: the arcs do not merely move the peak, they leave no peak.
/// Measured on the two scans that doc is written from (2048² binned 2×, 1800
/// projections over 360°):
///
/// | scan | `center` | truth | `prominence` |
/// |---|---|---|---|
/// | 0.55 s, closed concentric rings | 397.4 | 396 | **16.2** |
/// | 0.6 s, arcs that never close | 281.7 | 138 | **2.2** |
///
/// The mis-aligned scan's profile sits at 80 % of its own peak height even at the
/// array edges — it is flat, and its `argmax` is noise. Hence
/// [`RingCenter::trustworthy`], thresholded at 8.0: comfortably between the two,
/// but **calibrated on exactly those two scans**. Treat a `prominence` near the
/// threshold as "look at the mean projection yourself", not as a verdict.
///
/// (The doc quotes 397.5/281.1 and prominence 21.0/2.4 for its NumPy reference,
/// which clips at `1e-4` before the log where this goes through
/// [`prep::normalize`](crate::prep::normalize). The centre lands within 0.13 px
/// either way; the prominence scale differs, so the numbers above — not the
/// doc's — are the ones this function returns.)
pub fn find_center_rings(
    tomo: &Tomo<f32>,
    backend: &dyn Backend,
    step: usize,
) -> Result<RingCenter> {
    let fft = backend.fft().ok_or_else(|| Error::MissingCapability {
        backend: backend.name(),
        capability: "Fft",
    })?;
    if step == 0 {
        return Err(Error::InvalidParam(
            "find_center_rings: step must be ≥1".into(),
        ));
    }
    let p = tomo.as_layout(Layout::Projection); // [nproj, ny, nx]
    let (nproj, ny, nx) = p.array.dim();
    if nproj == 0 || ny < 5 || nx < 4 {
        return Err(Error::InvalidParam(format!(
            "find_center_rings needs ≥1 projection and ≥5×4 pixels; got {nproj}×{ny}×{nx}"
        )));
    }

    let prof = flip_registration_profile(&mean_projection(tomo, step)?, fft)?;
    Ok(ring_center_from_profile(&prof, nx))
}

/// The bullseye itself: the mean of every `step`-th projection, `[ny, nx]`.
///
/// [`find_center_rings`] registers this image against its own flip to get a
/// number, but `docs/LAMINOGRAPHY_ALIGNMENT.md` §1 makes *looking* at it step
/// one, and treats eye-vs-correlation disagreement as the misalignment flag —
/// which needs the image, not only the number. Same function both ways, so what
/// a viewer shows is what the estimate was computed from.
pub fn mean_projection(tomo: &Tomo<f32>, step: usize) -> Result<Array2<f32>> {
    if step == 0 {
        return Err(Error::InvalidParam(
            "mean_projection: step must be ≥1".into(),
        ));
    }
    let p = tomo.as_layout(Layout::Projection); // [nproj, ny, nx]
    let (nproj, ny, nx) = p.array.dim();
    if nproj == 0 {
        return Err(Error::InvalidParam(
            "mean_projection: the stack has no projections".into(),
        ));
    }
    let mut mean = Array2::<f32>::zeros((ny, nx));
    let mut cnt = 0usize;
    for i in (0..nproj).step_by(step) {
        mean += &p.array.index_axis(Axis(0), i);
        cnt += 1;
    }
    mean /= cnt as f32;
    Ok(mean)
}

/// Row-summed cross-correlation of `m` against its own left–right flip.
/// Rings put their signature on the middle rows, so only those are summed —
/// following `docs/LAMINOGRAPHY_ALIGNMENT.md`.
fn flip_registration_profile(m: &Array2<f32>, fft: &dyn Fft) -> Result<Vec<f32>> {
    let (ny, nx) = m.dim();
    let mean = m.sum() / (ny * nx) as f32;

    // a = m − mean, b = mirror(a). Correlate: irfft2(rfft2(a) · conj(rfft2(b))).
    // Done as a full C2C 2-D transform via the backend's 1-D FFT over each axis.
    let mut a: Vec<Complex32> = m.iter().map(|&v| Complex32::new(v - mean, 0.0)).collect();
    let mut b: Vec<Complex32> = Vec::with_capacity(ny * nx);
    for y in 0..ny {
        for x in 0..nx {
            b.push(Complex32::new(m[[y, nx - 1 - x]] - mean, 0.0));
        }
    }
    fft.fft_2d(&mut a, ny, nx, 1, false)?;
    fft.fft_2d(&mut b, ny, nx, 1, false)?;
    for (av, bv) in a.iter_mut().zip(b.iter()) {
        *av *= bv.conj();
    }
    fft.fft_2d(&mut a, ny, nx, 1, true)?;

    // fftshift along x, then sum the 5 middle rows (fftshifted along y too).
    let mut prof = vec![0.0f32; nx];
    for dy in 0..5usize {
        let y_shifted = ny / 2 - 2 + dy;
        let y = (y_shifted + ny.div_ceil(2)) % ny; // undo the y fftshift
        for (x, p) in prof.iter_mut().enumerate() {
            let xs = (x + nx.div_ceil(2)) % nx; // undo the x fftshift
            *p += a[y * nx + xs].re;
        }
    }
    Ok(prof)
}

/// Peak of the flip-registration profile → centre column + how much to trust it.
fn ring_center_from_profile(prof: &[f32], nx: usize) -> RingCenter {
    let i = prof
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (k, &v)| {
            if v > bv {
                (k, v)
            } else {
                (bi, bv)
            }
        })
        .0;
    // Parabolic sub-pixel refinement, guarded at the ends and against a flat top
    // (a mis-aligned scan's profile has no curvature to fit).
    let d = if i == 0 || i + 1 >= prof.len() {
        0.0
    } else {
        let (y0, y1, y2) = (prof[i - 1], prof[i], prof[i + 1]);
        let den = 2.0 * (y0 - 2.0 * y1 + y2);
        if den.abs() < f32::EPSILON {
            0.0
        } else {
            (y0 - y2) / den
        }
    };
    let center = nx as f32 / 2.0 + ((i as f32 + d) - (nx / 2) as f32) / 2.0;

    // Robust prominence: (peak − median) / MAD.
    let mut sorted: Vec<f32> = prof.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let mut dev: Vec<f32> = prof.iter().map(|v| (v - median).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = dev[dev.len() / 2];
    let prominence = if mad > 0.0 {
        (prof[i] - median) / mad
    } else {
        0.0
    };
    RingCenter {
        center,
        prominence,
        trustworthy: prominence >= 8.0,
    }
}

/// Focus of one reconstructed slice: mean |∇|² inside a 0.92-FOV disk. This is
/// the score `docs/LAMINOGRAPHY_ALIGNMENT.md` §2 prescribes, and what
/// [`lamino_tilt_scan`] maximises.
///
/// The disk is not cosmetic. Reconstruction geometry decides where the
/// back-projection sampling leaves the detector, so the ring outside the
/// reconstructed disk holds artifacts that move with the very parameter being
/// scored. Scoring the full frame scores that boundary, and it is enough to flip
/// the ranking.
pub fn slice_focus(img: &ndarray::ArrayView2<f32>) -> f64 {
    let (ny, nx) = img.dim();
    if ny < 3 || nx < 3 {
        return 0.0;
    }
    let (cy, cx) = (ny as f32 / 2.0, nx as f32 / 2.0);
    let r2 = (0.46 * nx.min(ny) as f32).powi(2); // 0.92 FOV across
    let (mut acc, mut cnt) = (0.0f64, 0usize);
    for y in 1..ny - 1 {
        for x in 1..nx - 1 {
            if (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2) >= r2 {
                continue;
            }
            let gx = (img[[y, x + 1]] - img[[y, x - 1]]) as f64;
            let gy = (img[[y + 1, x]] - img[[y - 1, x]]) as f64;
            acc += gx * gx + gy * gy;
            cnt += 1;
        }
    }
    acc / cnt.max(1) as f64
}

/// Another local maximum of a sweep that the metric did not rule out — see
/// [`SweepVerdict::Ambiguous`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rival {
    /// Index into the candidate/score arrays.
    pub index: usize,
    /// The candidate itself.
    pub value: f32,
    /// How far this lobe rises above the highest saddle joining it to higher
    /// ground — its topographic prominence. This, not its height, is what makes
    /// a lobe a competitor: a wiggle on the winner's flank has none.
    pub prominence: f64,
    /// How much lower it scored than the winner, as a fraction of the winner's
    /// score.
    pub deficit: f64,
}

/// What a sweep's focus curve actually established — see [`judge_sweep`].
///
/// A sum type rather than a value plus a `railed` flag, because a flag is
/// ignorable and these sweeps fail in more than one way: **a maximum can be
/// interior and still be no answer.** Only [`SweepVerdict::resolved`] hands back
/// a value to adopt, so a caller cannot reach an answer without passing the
/// verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum SweepVerdict {
    /// One lobe rises clear of every other: `value` is the sweep's answer.
    Resolved {
        /// Index into the candidate/score arrays.
        index: usize,
        /// The candidate itself.
        value: f32,
    },
    /// The curve rises to an edge and never comes back down, so the search
    /// stopped before the metric did. **The true peak is at that edge or beyond
    /// it and there is no way to tell which from inside the range**, so this is
    /// not an optimum, it is the range running out. Widen it or recentre it.
    ///
    /// Landing exactly on the first or last candidate is the obvious shape. It is
    /// not the only one: a curve that climbs across the whole window and turns
    /// over one sample short of the edge has an *interior* maximum and no second
    /// lobe to contradict it, and it is still a range that ran out. Both are the
    /// same fact — the winner owns no lobe of its own — and both land here.
    Railed {
        /// Index into the candidate/score arrays.
        index: usize,
        /// The candidate itself.
        value: f32,
    },
    /// Two or more lobes compete and the metric does not separate them, so the
    /// winner is not an answer.
    ///
    /// Measured, on the pouch-cell scan this check was written for: a ±40 px
    /// centre sweep put its maximum at 417 while the known-correct 396 scored
    /// only 0.34 % lower, 21 px away, and *neither* was at an edge. A rail check
    /// cannot see that, which is why this variant exists. Narrow the range onto a
    /// prior you trust rather than believing the winner.
    Ambiguous {
        /// Index into the candidate/score arrays.
        index: usize,
        /// The candidate itself — the winner, which is **not** an answer.
        value: f32,
        /// The competitors, most prominent first. Never empty.
        rivals: Vec<Rival>,
    },
    /// Every candidate scored the same, so the metric never responded to this
    /// axis and the argmax is just the curve's first entry. Reconstruction output
    /// that comes out uniform scores exactly flat, and uniform output has shipped
    /// from this crate before, so this is an observed state rather than a
    /// defensive one. Nothing about the range will help; the recon or the metric
    /// is what is wrong.
    Flat {
        /// Index into the candidate/score arrays.
        index: usize,
        /// The candidate itself.
        value: f32,
    },
}

impl SweepVerdict {
    /// The candidate to adopt, or `None` when the sweep established none. This is
    /// the only accessor that answers the sweep; the others only describe it.
    pub fn resolved(&self) -> Option<f32> {
        match *self {
            SweepVerdict::Resolved { value, .. } => Some(value),
            _ => None,
        }
    }

    /// The highest-scoring candidate whatever the verdict — to *show* the sweep
    /// (mark the montage, label the plot), never to adopt. Use [`Self::resolved`]
    /// for that.
    pub fn best(&self) -> (usize, f32) {
        match *self {
            SweepVerdict::Resolved { index, value }
            | SweepVerdict::Railed { index, value }
            | SweepVerdict::Ambiguous { index, value, .. }
            | SweepVerdict::Flat { index, value } => (index, value),
        }
    }
}

/// The share of a curve's full height that a lobe must own to be taken
/// seriously. One bar, applied to every lobe including the winner, which is what
/// makes the two failures below the same test rather than two special cases:
///
/// * the **winner** below it owns no lobe of its own — it is the top of a ramp,
///   and a curve that only rises is a range that ran out;
/// * any **other** lobe above it is a competitor the metric has not ruled out.
///
/// Calibrated on measured pouch-cell centre sweeps at tilt 44°, in units of the
/// curve's height. The ±40 px sweep on the rh/2 plane: winner (417) **35 %**,
/// the known-correct lobe at 396 **55 %**, a third lobe at 374 **25 %**, a
/// shoulder at the range edge **0 %** — three competitors, no answer. The same
/// ±40 px sweep on the sample plane: winner (435) **0.29 %**, a ramp. The ±8 px
/// sweep around a good prior, which does resolve: winner **83 %**, no second lobe
/// at all.
const SIGNIFICANT_PROMINENCE: f64 = 0.2;

/// Indices of the local maxima of `score`, one per lobe.
///
/// Plateau-aware: a run of equal values is one lobe, reported at its first index,
/// when the curve falls off both ends of the run. Requiring a strict rise on both
/// sides instead finds **nothing** on the real ±8 px centre sweep, whose peak is a
/// two-sample tie at 395.75/396.00.
fn local_maxima(score: &[f64]) -> Vec<usize> {
    let n = score.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && score[j + 1] == score[i] {
            j += 1;
        }
        let rises_before = i > 0 && score[i - 1] > score[i];
        let rises_after = j + 1 < n && score[j + 1] > score[j];
        if !rises_before && !rises_after {
            out.push(i);
        }
        i = j + 1;
    }
    out
}

/// How far the lobe at `i` stands above the higher of the two troughs flanking
/// it, where each side is walked until **either** higher ground **or** the
/// window's edge, whichever comes first.
///
/// Stopping at the edge is what makes this uniform, and the uniformity is the
/// point. Textbook topographic prominence has no higher ground to find for the
/// global maximum and so measures it from the floor of the window — which hands
/// the top of a *ramp* the window's entire height and makes a curve that only
/// rises look like the most prominent lobe there is. That is backwards: a curve
/// that never comes back down is a range that ran out. Here it scores ~0, and the
/// same arithmetic serves the winner and its rivals alike.
fn prominence(score: &[f64], i: usize) -> f64 {
    let mut col = f64::NEG_INFINITY;
    for dir in [-1i64, 1] {
        let mut j = i as i64 + dir;
        let mut low = score[i];
        while j >= 0 && (j as usize) < score.len() {
            let v = score[j as usize];
            if v > score[i] {
                break;
            }
            low = low.min(v);
            j += dir;
        }
        col = col.max(low);
    }
    score[i] - col
}

/// Judge what a sweep's focus curve established: one answer, or one of the three
/// ways it can fail to have one.
///
/// The winner alone is not the question. A railed maximum reported as an answer
/// is how one bad axis poisons the next — a tilt railed at the top of its range
/// moved a centre re-sweep 1.7 px off the value the un-railed sweep found — and
/// an *interior* maximum can be just as empty: widening the pouch-cell centre
/// sweep from ±8 px to ±40 px moves its argmax from the correct 396 to a rival at
/// 417 that outscores it by 0.34 %, with no edge involved. So the caller gets a
/// [`SweepVerdict`], and only [`SweepVerdict::resolved`] yields a value.
///
/// Returns `None` for an empty sweep or a length mismatch. A sweep of one or two
/// candidates is all boundary, so it is never called railed: there is no interior
/// for a peak to be in and nothing to widen toward.
pub fn judge_sweep(cands: &[f32], score: &[f64]) -> Option<SweepVerdict> {
    if cands.is_empty() || score.len() != cands.len() {
        return None;
    }
    let index = score
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0;
    let value = cands[index];
    // Fewer than three candidates is all boundary: there is no interior for a
    // lobe to sit in, so there is no curve to judge. One candidate is a pin the
    // caller already chose; two is a comparison, not a sweep.
    if cands.len() < 3 {
        return Some(SweepVerdict::Resolved { index, value });
    }
    // Before the rail check, not after: the argmax of a flat curve is its first
    // entry, which sits on the boundary, so a flat sweep would otherwise be
    // reported as a range that ran out. Nothing ran out — nothing responded.
    // `is_nan()` as well as `<= 0.0`: a reconstruction carrying NaN scores NaN,
    // which is no more a curve than a flat one is, and `<= 0.0` alone is false
    // for NaN.
    let height = score[index] - score.iter().copied().fold(f64::INFINITY, f64::min);
    if height.is_nan() || height <= 0.0 {
        return Some(SweepVerdict::Flat { index, value });
    }
    let winner = prominence(score, index);
    // Two ways the range can run out, and only the first is visible as an index.
    // The second: the curve rises across the whole window and turns over just shy
    // of the edge, so the maximum is interior and there is no second lobe to
    // contradict it. Measured on the sample plane of the aligned reference scan —
    // a ±40 px sweep rising to index 79 of 80, naming 435 against a known 396 and
    // owning 0.29 % of the curve's height. That share is what tells it from a peak.
    if index == 0 || index == cands.len() - 1 || winner < SIGNIFICANT_PROMINENCE * height {
        return Some(SweepVerdict::Railed { index, value });
    }
    let mut rivals: Vec<Rival> = local_maxima(score)
        .into_iter()
        .filter(|&i| i != index)
        .filter_map(|i| {
            let p = prominence(score, i);
            (p >= SIGNIFICANT_PROMINENCE * height).then_some(Rival {
                index: i,
                value: cands[i],
                prominence: p,
                deficit: (score[index] - score[i]) / score[index],
            })
        })
        .collect();
    if rivals.is_empty() {
        return Some(SweepVerdict::Resolved { index, value });
    }
    rivals.sort_by(|a, b| {
        b.prominence
            .partial_cmp(&a.prominence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Some(SweepVerdict::Ambiguous {
        index,
        value,
        rivals,
    })
}

/// What one laminography tilt candidate scored, from [`lamino_tilt_scan`].
#[derive(Debug, Clone, PartialEq)]
pub struct TiltFocus {
    /// The tilt, in degrees, that was reconstructed.
    pub tilt_deg: f32,
    /// The best [`slice_focus`] any slice of that reconstruction reached.
    pub focus: f64,
    /// The slice that reached it — the in-focus layer. It moves with the tilt
    /// (measured: 800 → 1120 as tilt went 40° → 58°), which is the whole reason
    /// this is a full reconstruction and not a probe.
    pub z_peak: usize,
    /// Depth of the reconstruction `z_peak` indexes, `rh` for this tilt. It grows
    /// with the tilt (`ceil(nz / cos(tilt) / 2) * 2`), so `z_peak` values are only
    /// comparable across tilts as fractions of their own `depth`.
    pub depth: usize,
    /// [`slice_focus`] of every slice, indexed by z — `focus` is its maximum and
    /// `z_peak` its argmax. The scan computes all of it anyway, and keeping it is
    /// what separates the two things a bare `z_peak` cannot: a broad hump centred
    /// on the sample, versus a spike on a z-edge slice where few projections
    /// contribute and the streaks are the gradient. Believing an unexamined argmax
    /// is the failure this whole method exists to avoid — `docs/
    /// LAMINOGRAPHY_ALIGNMENT.md` §2 names its symptom, "a MONOTONE focus surface
    /// with the argmax pinned to a grid corner".
    pub focus_by_z: Vec<f64>,
}

/// Score each tilt in `tilts_deg` by fully reconstructing at it and taking the
/// **maximum [`slice_focus`] over the whole z range** — the only method
/// `docs/LAMINOGRAPHY_ALIGNMENT.md` §2 validates for the tilt.
///
/// It is expensive on purpose. The cheap alternative — score one fixed slice per
/// tilt, which is what [`crate::cuda::lamino_tilt_probe`] makes a single launch —
/// is ill-posed, and measurement says so: the in-focus layer moves in z with the
/// tilt, so a fixed row scores a plane whose error grows with the distance from
/// its own optimum, while the tilt's own signal is only ~2 % per degree. On the
/// pouch scans that sweep returns 48° or rails to the range edge depending only
/// on which row was picked. Taking the max over z needs every (tilt, z) pair,
/// which is exactly the work the probe skips, so the probe buys nothing here.
/// The centre is the opposite case — an in-plane shift does not move the layer,
/// and its response is ~40 % per 50 px — and that is what the probe is for.
///
/// `geom`'s beam is ignored; `tilts_deg` supplies it. `params.lamino_rh` is
/// honoured if set, but leaving it `None` lets each tilt take its own natural
/// depth, which is what the scan wants. The volume is never assembled: focus is
/// scored per streamed tile, so the memory cost is one tile regardless of `rh`.
///
/// `sino` must already be prepped (flat/dark, minus-log, stripe removal).
///
/// `on_tilt(result, slice)` is called with each result as it lands — one full
/// reconstruction apart, so a caller with a user attached should report it — and
/// returning `Err` from it aborts the scan. `slice` is the `z_peak` slice, the
/// in-focus layer itself: `docs/LAMINOGRAPHY_ALIGNMENT.md` §3 makes confirming
/// the winner by eye part of the method, and this is the one reconstruction that
/// already found it. Callers that only want the numbers ignore it.
///
/// CUDA-only: the streaming laminography reconstruction this rides on exists only
/// there.
pub fn lamino_tilt_scan(
    sino: &Tomo<f32>,
    geom: &Geometry,
    algorithm: crate::params::Algorithm,
    params: &crate::params::ReconParams,
    tilts_deg: &[f32],
    on_tilt: &mut dyn FnMut(&TiltFocus, ndarray::ArrayView2<f32>) -> Result<()>,
) -> Result<Vec<TiltFocus>> {
    let mut out = Vec::with_capacity(tilts_deg.len());
    for &tilt_deg in tilts_deg {
        let geom = Geometry {
            beam: Beam::Laminography {
                phi: std::f32::consts::FRAC_PI_2 + tilt_deg * std::f32::consts::PI / 180.0,
            },
            ..geom.clone()
        };
        // The volume is streamed and dropped tile by tile, so the winning slice
        // has to be kept as it goes past — it is gone by the time the scan knows
        // it won. Tiles arrive in ascending row order, so `focus_by_z` is built by
        // appending.
        let mut best: Option<(f64, usize, Array2<f32>)> = None;
        let mut focus_by_z: Vec<f64> = Vec::new();
        let (depth, _, _) = {
            let mut on_tile = |rh0: usize, tile: &crate::data::Volume<f32>| -> Result<()> {
                for (k, img) in tile.array.outer_iter().enumerate() {
                    let f = slice_focus(&img);
                    focus_by_z.push(f);
                    if best.as_ref().is_none_or(|b| f > b.0) {
                        best = Some((f, rh0 + k, img.to_owned()));
                    }
                }
                Ok(())
            };
            crate::cuda::reconstruct_lamino_streaming(
                sino,
                &geom,
                algorithm,
                params,
                None,
                &mut on_tile,
            )?
        };
        let (focus, z_peak, img) = best.ok_or_else(|| {
            Error::Backend(format!(
                "the reconstruction at tilt {tilt_deg}° produced no slices to score"
            ))
        })?;
        // The profile is only indexable by z if every row arrived exactly once.
        if focus_by_z.len() != depth {
            return Err(Error::Backend(format!(
                "tilt {tilt_deg}°: scored {} slices but the reconstruction is {depth} deep",
                focus_by_z.len()
            )));
        }
        let r = TiltFocus {
            tilt_deg,
            focus,
            z_peak,
            depth,
            focus_by_z,
        };
        on_tilt(&r, img.view())?;
        out.push(r);
    }
    Ok(out)
}

/// Entropy-based center finding (tomopy `rotation.py:82`).
///
/// Reconstructs a single slice with gridrec at candidate centers and minimises
/// the Shannon entropy of the masked reconstruction's 64-bin histogram with a
/// Nelder-Mead simplex search (Donath et al. 2006, doi:10.1364/JOSAA.23.001048),
/// exactly as tomopy does — the systematic artifact a wrong center introduces
/// raises the reconstruction's entropy, so the entropy minimum near a good
/// initial guess marks the axis.
///
/// Unlike `find_center_vo`/`find_center_pc` this goes *through* the projector
/// (gridrec), so it inherits the linear-interp-vs-Siddon gridrec gap (see
/// PORTING): the entropy surface is a near-replica of tomopy's but not bit-exact,
/// and the result is the local basin Nelder-Mead reaches from `init`. The center
/// is therefore held to ±~1 px of the injected/tomopy value, not bit parity.
///
/// `ind` selects the slice (default `n_rows/2`); `init` is the optimiser start
/// (default `n_cols/2`, tomopy's `dx//2`); `tol` sets both the x- and
/// f-tolerance of the simplex termination (tomopy `tol=0.5`).
pub fn find_center(
    tomo: &Tomo<f32>,
    theta: &[f32],
    backend: &dyn Backend,
    ind: Option<usize>,
    init: Option<f32>,
    tol: f32,
) -> Result<f32> {
    let fft = backend.fft().ok_or_else(|| Error::MissingCapability {
        backend: backend.name(),
        capability: "Fft",
    })?;
    let sino = tomo.as_layout(Layout::Sinogram); // [row, angle, col]
    let (nrows, nang, ncol) = sino.array.dim();
    if theta.len() != nang {
        return Err(Error::ShapeMismatch {
            expected: format!("{nang} angles"),
            found: format!("{} theta", theta.len()),
        });
    }
    let ind = ind.unwrap_or(nrows / 2).min(nrows.saturating_sub(1));

    // Single-slice sinogram [1, nang, ncol] for the per-center reconstructions.
    let slc = Tomo::new(
        sino.array
            .index_axis(Axis(0), ind)
            .to_owned()
            .insert_axis(Axis(0)),
        Layout::Sinogram,
    );
    let n = ncol; // reconstruction grid = detector width (tomopy num_gridx = dx)
    let init = init.unwrap_or((ncol / 2) as f32) as f64; // dx // 2

    // Histogram limits from a default-center (`dx/2`) reconstruction, masked
    // (tomopy `_adjust_hist_limits`).
    let mut rec0 = recon_at(&slc, theta, ncol as f32 / 2.0, n, fft)?;
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    {
        let mut img = rec0.index_axis_mut(Axis(0), 0);
        circ_mask_inplace(&mut img);
        for &v in img.iter() {
            mn = mn.min(v);
            mx = mx.max(v);
        }
    }
    let hmin = adjust_hist_min(mn) as f64;
    let hmax = adjust_hist_max(mx) as f64;

    // Entropy of the masked reconstruction at a candidate center (tomopy casts
    // the center to f32 before reconstructing).
    let cost = |center: f64| -> Result<f64> {
        let mut rec = recon_at(&slc, theta, center as f32, n, fft)?;
        let mut img = rec.index_axis_mut(Axis(0), 0);
        circ_mask_inplace(&mut img);
        Ok(entropy64(&img.view(), hmin, hmax))
    };
    let center = nelder_mead_1d(&cost, init, tol as f64, tol as f64)?;
    Ok(center as f32)
}

/// Nghia Vo's coarse+fine center search — the workhorse (tomopy `rotation.py:205`).
///
/// Sinogram-domain Fourier method (Vo et al. 2014, doi:10.1364/OE.22.019078):
/// pick one sinogram slice, denoise it, then minimise a double-wedge-masked
/// `mean(|fftshift(fft2(·))|)` metric over candidate centers — a coarse
/// integer-grid pass (step 0.5) followed by a fine subpixel pass (step `step`).
/// Because the metric lives entirely in the sinogram's Fourier domain it is
/// projector-independent, so it matches tomopy numerically (unlike the
/// projector-coupled analytic/iterative methods documented in PORTING).
///
/// `ind` selects the slice (default `n_rows/2`, averaging ±5 rows for SNR when
/// there are more than 10); `smin`/`smax` bound the coarse search around the
/// detector midline; `srad`/`step` set the fine radius/step; `ratio` sizes the
/// double-wedge mask (FOV/object) and `drop` blanks rows around its DC line.
#[allow(clippy::too_many_arguments)]
pub fn find_center_vo(
    tomo: &Tomo<f32>,
    backend: &dyn Backend,
    ind: Option<usize>,
    smin: f32,
    smax: f32,
    srad: f32,
    step: f32,
    ratio: f32,
    drop: i32,
) -> Result<f32> {
    let fft = backend.fft().ok_or_else(|| Error::MissingCapability {
        backend: backend.name(),
        capability: "Fft",
    })?;

    let sino = extract_slice(tomo, ind); // (nang, ncol)
    let (nrow, ncol) = sino.dim();

    // Denoise. The coarse and fine passes deliberately use different window
    // sizes (tomopy rotation.py:251-255).
    let tomo_cs = gaussian_filter2d(&sino, 3.0, 1.0);
    let tomo_fs = gaussian_filter2d(&sino, 2.0, 2.0);

    // For large data (>2k×2k) coarse-search a 4× column-downsampled copy.
    let fine_cen = if nrow * ncol > 4_000_000 {
        let coarse = downsample_cols(&tomo_cs, 2);
        let init_cen = search_coarse(&coarse, smin / 4.0, smax / 4.0, ratio, drop, fft)? * 4.0;
        search_fine(&tomo_fs, srad, step, init_cen, ratio, drop, fft)?
    } else {
        let init_cen = search_coarse(&tomo_cs, smin, smax, ratio, drop, fft)?;
        search_fine(&tomo_fs, srad, step, init_cen, ratio, drop, fft)?
    };
    Ok(fine_cen)
}

/// Phase-correlation center from a 0°/180° projection pair (tomopy
/// `rotation.py:391`).
///
/// Registers `proj0` against the mirrored `proj180` by subpixel phase
/// cross-correlation (a port of skimage `phase_cross_correlation` with
/// `normalization="phase"` and `upsample_factor = 1/tol`) and maps the recovered
/// column shift to a rotation center `(ncol + shift_col − 1)/2`. Because it is
/// pure Fourier-domain image registration it never touches a projector, so it
/// matches tomopy numerically. With the default `tol = 0.5` (upsample 2) the
/// shift is quantized to half a pixel, so the center lands on a quarter-pixel
/// grid exactly as tomopy's does.
///
/// `rotc_guess` (tomopy's pre-alignment shift) pre-shifts both projections by
/// `[0, −imgshift]` with `imgshift = rotc_guess − (ncol−1)/2` through a faithful
/// `scipy.ndimage.shift` (order-3 cubic spline, `mode='constant'`, `cval=0`; see
/// `ndimage_shift_spline3_constant`) and adds `imgshift` back to the recovered
/// center, exactly as tomopy `rotation.py:419-435`. With the default `None`,
/// `imgshift == 0`, where scipy's spline shift is the identity to f32 precision,
/// so the round-trip is skipped (bit-for-bit unchanged from before).
pub fn find_center_pc(
    proj0: &Array2<f32>,
    proj180: &Array2<f32>,
    backend: &dyn Backend,
    tol: f32,
    rotc_guess: Option<f32>,
) -> Result<f32> {
    let fft = backend.fft().ok_or_else(|| Error::MissingCapability {
        backend: backend.name(),
        capability: "Fft",
    })?;
    let (nrow, ncol) = proj0.dim();
    if proj180.dim() != (nrow, ncol) {
        return Err(Error::ShapeMismatch {
            expected: format!("{nrow}x{ncol}"),
            found: format!("{}x{}", proj180.dim().0, proj180.dim().1),
        });
    }
    // Pre-alignment shift about the detector midline (tomopy rotation.py:419).
    let imgshift = match rotc_guess {
        Some(g) => g as f64 - (ncol as f64 - 1.0) / 2.0,
        None => 0.0,
    };
    let upsample = 1.0f64 / tol as f64;
    // reference = proj0, moving = fliplr(proj180). When imgshift != 0 both
    // projections are spline-shifted by [0, -imgshift] first (rotation.py:422-423);
    // at imgshift == 0 the spline shift is f32-identity, so use the inputs as-is.
    let shift_col = if imgshift != 0.0 {
        let p0 = ndimage_shift_spline3_constant(proj0, 0.0, -imgshift);
        let p180 = ndimage_shift_spline3_constant(proj180, 0.0, -imgshift);
        let mov = fliplr(&p180);
        phase_cross_correlation(&p0, &mov, upsample, fft)?.1
    } else {
        let mov = fliplr(proj180);
        phase_cross_correlation(proj0, &mov, upsample, fft)?.1
    };
    let center = (ncol as f64 + shift_col - 1.0) / 2.0;
    Ok((center + imgshift) as f32)
}

/// SIFT-feature center detection (tomocupy `find_center.py:99`).
///
/// Available only with the **`sift-center`** feature (pure-Rust `lowe-sift`).
/// See [`sift`] for the implementation.
#[cfg(not(feature = "sift-center"))]
pub fn find_center_sift(
    _proj0: &Array2<f32>,
    _proj180: &Array2<f32>,
    _threshold: f32,
) -> Result<f32> {
    Err(Error::todo(
        "center::find_center_sift (build with the `sift-center` feature)",
        "tomocupy find_center.py:99",
    ))
}

#[cfg(feature = "sift-center")]
pub use sift::{find_center_sift, register_shift_sift};

/// SIFT-feature rotation-center finding (tomocupy `find_center.py`
/// `_register_shift_sift` + the `n//2 - shift_x/2` center formula). Requires the
/// `sift-center` feature.
#[cfg(feature = "sift-center")]
pub mod sift {
    use super::{fliplr, Error, Result};
    use lowe_sift::{match_descriptors, GrayImage, Sift};
    use ndarray::Array2;

    /// numpy `np.histogram(data, 1000)` peak-thresholded robust min/max
    /// (`_find_min_max`): 1000 uniform bins over `[min, max]`, keep bins whose
    /// count exceeds 0.5 % of the peak, return the outermost surviving edges.
    /// Replicates numpy's uniform-bin indexing (including the floating-point
    /// decrement/increment corrections) so the 0–255 normalization is bit-exact.
    fn find_min_max(data: &[f32]) -> (f32, f32) {
        const NBINS: usize = 1000;
        let (mut dmin, mut dmax) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in data {
            if v < dmin {
                dmin = v;
            }
            if v > dmax {
                dmax = v;
            }
        }
        let (first, last) = (dmin as f64, dmax as f64);
        if last <= first {
            return (dmin, dmax);
        }
        // bin_edges = linspace(first, last, NBINS+1) (last forced exact).
        let edges: Vec<f64> = (0..=NBINS)
            .map(|i| {
                if i == NBINS {
                    last
                } else {
                    first + (last - first) * (i as f64) / (NBINS as f64)
                }
            })
            .collect();
        let norm = NBINS as f64 / (last - first);
        let mut h = vec![0u64; NBINS];
        for &v in data {
            let x = v as f64;
            let mut idx = ((x - first) * norm) as i64;
            if idx == NBINS as i64 {
                idx -= 1;
            }
            let mut idx = idx.clamp(0, NBINS as i64 - 1) as usize;
            // numpy corrections (both applied, sequentially — not else-if).
            if x < edges[idx] && idx > 0 {
                idx -= 1;
            }
            if idx != NBINS - 1 && x >= edges[idx + 1] {
                idx += 1;
            }
            h[idx] += 1;
        }
        let hmax = *h.iter().max().unwrap_or(&0);
        let thr = hmax as f64 * 0.005;
        let st = h.iter().position(|&c| c as f64 > thr).unwrap_or(0);
        let end = h.iter().rposition(|&c| c as f64 > thr).unwrap_or(NBINS - 1);
        (edges[st] as f32, edges[end + 1] as f32)
    }

    /// `(img - mmin)/(mmax-mmin)*255`, clipped to 0–255 and `astype(uint8)`-
    /// truncated (tomocupy's numpy pipeline), then rescaled to the `[0, 1]`
    /// f32 pixels [`GrayImage`] expects. The u8 quantization is deliberately
    /// kept so the detector sees the same image content the cv2 golden did.
    fn to_gray(img: &Array2<f32>, mmin: f32, mmax: f32) -> Result<GrayImage> {
        let (rows, cols) = img.dim();
        let scale = mmax - mmin;
        let data = img
            .iter()
            // numpy clips >255→255 then <0→0; clamp is equivalent for finite v.
            .map(|&v| (((v - mmin) / scale * 255.0).clamp(0.0, 255.0) as u8) as f32 / 255.0)
            .collect();
        GrayImage::new(cols, rows, data).map_err(|e| Error::Backend(format!("lowe-sift: {e}")))
    }

    /// `(img normalized by its own robust min/max) → uint8` (tomocupy's
    /// per-image 0–255 mapping). Exposed for parity testing of the
    /// histogram-based normalization in isolation.
    pub fn normalize_to_u8(img: &Array2<f32>) -> Vec<u8> {
        let (mmin, mmax) = find_min_max(img.as_slice().expect("contiguous image"));
        let scale = mmax - mmin;
        img.iter()
            .map(|&v| ((v - mmin) / scale * 255.0).clamp(0.0, 255.0) as u8)
            .collect()
    }

    /// Per-pair SIFT shift estimate (tomocupy `_register_shift_sift`): SIFT
    /// detect+describe on the normalized `datap2`/`datap1` images (pure-Rust
    /// `lowe-sift`), exact nearest-neighbor matching with Lowe's ratio test
    /// (`threshold` — `distance < threshold·second_distance`, the same test
    /// tomocupy runs on the cv2 knn(k=2) pairs), and the mean keypoint
    /// displacement, returned as `[dy, dx]` per pair. min/max for
    /// normalization come from `datap1` (matching upstream). Also returns the
    /// number of good matches in the last pair.
    pub fn register_shift_sift(
        datap1: &[Array2<f32>],
        datap2: &[Array2<f32>],
        threshold: f32,
    ) -> Result<(Array2<f32>, usize)> {
        if datap1.len() != datap2.len() || datap1.is_empty() {
            return Err(Error::InvalidParam(
                "register_shift_sift: need equal, non-empty datap1/datap2".into(),
            ));
        }
        let sift = Sift::default();
        let mut shifts = Array2::<f32>::zeros((datap1.len(), 2));
        let mut ngood = 0usize;
        for (id, (p1, p2)) in datap1.iter().zip(datap2).enumerate() {
            let (mmin, mmax) = find_min_max(p1.as_slice().expect("contiguous datap1"));
            let f1 = sift.detect_and_compute(&to_gray(p2, mmin, mmax)?); // datap2 → query
            let f2 = sift.detect_and_compute(&to_gray(p1, mmin, mmax)?); // datap1 → train
            let d1: Vec<_> = f1.iter().map(|f| f.descriptor.clone()).collect();
            let d2: Vec<_> = f2.iter().map(|f| f.descriptor.clone()).collect();

            let (mut sum_x, mut sum_y, mut n) = (0.0f64, 0.0f64, 0usize);
            for m in match_descriptors(&d1, &d2, threshold) {
                let src = &f1[m.query_index].keypoint;
                let dst = &f2[m.train_index].keypoint;
                sum_x += (src.x - dst.x) as f64;
                sum_y += (src.y - dst.y) as f64;
                n += 1;
            }
            if n == 0 {
                return Err(Error::InvalidParam(format!(
                    "register_shift_sift: no good SIFT matches for pair {id}"
                )));
            }
            ngood = n;
            // np.mean(shift)[::-1] = [mean_y, mean_x].
            shifts[[id, 0]] = (sum_y / n as f64) as f32;
            shifts[[id, 1]] = (sum_x / n as f64) as f32;
        }
        Ok((shifts, ngood))
    }

    /// Rotation center from a 0°/180° projection pair via SIFT feature matching
    /// (tomocupy `find_center_sift`). The 180° projection is flipped left-right
    /// (`data[..., ::-1]`), matched against the 0° projection, and the center is
    /// `ncol/2 - mean_horizontal_shift/2`. `threshold` is the Lowe ratio (0.5
    /// upstream).
    pub fn find_center_sift(
        proj0: &Array2<f32>,
        proj180: &Array2<f32>,
        threshold: f32,
    ) -> Result<f32> {
        if proj0.dim() != proj180.dim() {
            return Err(Error::ShapeMismatch {
                expected: format!("{:?}", proj0.dim()),
                found: format!("{:?}", proj180.dim()),
            });
        }
        let ncol = proj0.dim().1;
        let datap1 = [proj0.clone()];
        let datap2 = [fliplr(proj180)];
        let (shifts, _) = register_shift_sift(&datap1, &datap2, threshold)?;
        Ok(ncol as f32 / 2.0 - shifts[[0, 1]] / 2.0)
    }
}

/// Reconstruct one slice across a range of rotation centers (tomopy
/// `rotation.py:438` `write_center`).
///
/// Helps pick the rotation axis by eye: the `ind`-th sinogram is reconstructed
/// with gridrec at every center in `cen_range = (start, stop, step)` (numpy
/// `arange` semantics — values `start, start+step, …` while `< stop`; default
/// `(ncol/2 − 5, ncol/2 + 5, 0.5)`), optionally circular-masked, and returned as a
/// `[len(centers), n, n]` stack (`n = ncol`) alongside the center values. tomopy
/// writes each slice to `{center:.2f}.tiff`; persist the returned stack the same
/// way (e.g. via `tomoxide-io`) if those files are wanted — this is the
/// I/O-free core so `tomoxide-recon` stays backend/`tomoxide-core`-only.
///
/// Parity scope: only the **center enumeration** is held to tomopy (Δ = 0 — it is
/// pure `np.arange`). The reconstruction *content* goes through tomoxide's
/// gridrec, a gridrec-*family* method (Kaiser–Bessel kernel, ramp weight; see
/// `gridrec.rs`), so the slice pixels are self-consistent gridrec
/// reconstructions, **not** bit-identical to tomopy's PSWF + `parzen` `gridrec`.
///
/// `ind` selects the slice (default `n_rows/2`, tomopy `dy//2`); `mask` applies a
/// `ratio`-scaled circular mask (tomopy `mask`/`ratio`, default `ratio = 1`).
pub fn write_center(
    tomo: &Tomo<f32>,
    theta: &[f32],
    backend: &dyn Backend,
    cen_range: Option<(f32, f32, f32)>,
    ind: Option<usize>,
    mask: bool,
    ratio: f32,
) -> Result<(Vec<f32>, Array3<f32>)> {
    let fft = backend.fft().ok_or_else(|| Error::MissingCapability {
        backend: backend.name(),
        capability: "Fft",
    })?;
    let sino = tomo.as_layout(Layout::Sinogram); // [row, angle, col]
    let (nrows, nang, ncol) = sino.array.dim();
    if theta.len() != nang {
        return Err(Error::ShapeMismatch {
            expected: format!("{nang} angles"),
            found: format!("{} theta", theta.len()),
        });
    }
    if nrows == 0 || nang == 0 || ncol == 0 {
        return Ok((Vec::new(), Array3::zeros((0, ncol, ncol))));
    }
    let ind = ind.unwrap_or(nrows / 2).min(nrows - 1);

    // Center range (numpy `arange`). Default: `arange(ncol/2 − 5, ncol/2 + 5, 0.5)`
    // (tomopy `rotation.py:548`).
    let (start, stop, step) = match cen_range {
        Some((a, b, s)) => (a as f64, b as f64, s as f64),
        None => {
            let half = ncol as f64 / 2.0;
            (half - 5.0, half + 5.0, 0.5)
        }
    };
    let centers = arange(start, stop, step);

    // Reconstruct the same slice at each center (tomopy replicates `tomo[:, ind, :]`
    // into a stack and reconstructs with per-slice centers).
    let slc = Tomo::new(
        sino.array
            .index_axis(Axis(0), ind)
            .to_owned()
            .insert_axis(Axis(0)),
        Layout::Sinogram,
    );
    let n = ncol; // tomopy num_gridx = dx
    let mut stack = Array3::<f32>::zeros((centers.len(), n, n));
    for (m, &c) in centers.iter().enumerate() {
        let mut rec = recon_at(&slc, theta, c as f32, n, fft)?; // [1, n, n]
        if mask {
            let mut img = rec.index_axis_mut(Axis(0), 0);
            circ_mask_inplace_ratio(&mut img, ratio as f64);
        }
        stack
            .index_axis_mut(Axis(0), m)
            .assign(&rec.index_axis(Axis(0), 0));
    }
    let centers: Vec<f32> = centers.iter().map(|&c| c as f32).collect();
    Ok((centers, stack))
}

/// numpy `arange(start, stop, step)` for a positive `step`: length
/// `⌈(stop − start)/step⌉`, value `start + i·step`.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    if step <= 0.0 {
        return Vec::new();
    }
    let len = ((stop - start) / step).ceil();
    let len = if len.is_finite() && len > 0.0 {
        len as usize
    } else {
        0
    };
    (0..len).map(|i| start + i as f64 * step).collect()
}

// ----------------------------------------------------------------------------
// Entropy center-finding internals (private). Mirrors tomopy
// rotation.py:82-202 (`find_center`, `_adjust_hist_limits`, `_find_center_cost`)
// plus scipy's scalar Nelder-Mead simplex (`_minimize_neldermead`).
// ----------------------------------------------------------------------------

/// gridrec a single-slice sinogram at rotation `center` (tomopy default grid =
/// detector width, parallel beam, unit pixel).
fn recon_at(
    sino: &Tomo<f32>,
    theta: &[f32],
    center: f32,
    n: usize,
    fft: &dyn Fft,
) -> Result<Array3<f32>> {
    let geom = Geometry {
        angles: Angles(theta.to_vec()),
        center: Center::Scalar(center),
        beam: Beam::Parallel,
        detector: Detector {
            width: sino.n_cols(),
            height: 1,
            pixel_size: 1.0,
        },
    };
    crate::recon::gridrec::gridrec(sino, &geom, n, fft)
}

/// Apply tomopy's `circ_mask(rec, axis=0, ratio, val=0)` to one reconstruction
/// slice: keep where `x²+y² < (ratio·min(ny,nx)/2)²`, zero outside (tomopy
/// `_get_mask`: `x²+y² < ratio²·r²`, `r = min/2`). Coordinates match
/// `np.ogrid[0.5−n/2 : 0.5+n/2]`, i.e. cell `(i, j)` sits at
/// `(i + 0.5 − ny/2, j + 0.5 − nx/2)`.
fn circ_mask_inplace_ratio(img: &mut ArrayViewMut2<f32>, ratio: f64) {
    let (ny, nx) = img.dim();
    let half_y = ny as f64 / 2.0;
    let half_x = nx as f64 / 2.0;
    let r2 = (ratio * (ny.min(nx) as f64 / 2.0)).powi(2);
    for i in 0..ny {
        let y = i as f64 + 0.5 - half_y;
        for j in 0..nx {
            let x = j as f64 + 0.5 - half_x;
            if x * x + y * y >= r2 {
                img[[i, j]] = 0.0;
            }
        }
    }
}

/// `circ_mask` with tomopy's default `ratio = 1`.
fn circ_mask_inplace(img: &mut ArrayViewMut2<f32>) {
    circ_mask_inplace_ratio(img, 1.0);
}

/// tomopy `_adjust_hist_min`: stretch the lower histogram bound away from zero.
fn adjust_hist_min(v: f32) -> f32 {
    if v < 0.0 {
        2.0 * v
    } else {
        0.5 * v
    }
}

/// tomopy `_adjust_hist_max`: stretch the upper histogram bound away from zero.
fn adjust_hist_max(v: f32) -> f32 {
    if v < 0.0 {
        0.5 * v
    } else {
        2.0 * v
    }
}

/// Shannon entropy of the 64-bin histogram of `img` over `[hmin, hmax]`,
/// `−Σ p·log2(p)` with `p = count/size + 1e-12` (tomopy `_find_center_cost`).
/// `size` counts every pixel (including masked and out-of-range ones), and the
/// `1e-12` floor is added to all 64 bins, exactly as numpy does.
fn entropy64(img: &ndarray::ArrayView2<f32>, hmin: f64, hmax: f64) -> f64 {
    const BINS: usize = 64;
    let size = img.len() as f64;
    let mut counts = [0.0f64; BINS];
    let span = hmax - hmin;
    for &v in img.iter() {
        let v = v as f64;
        if v < hmin || v > hmax {
            continue; // numpy histogram drops out-of-range values
        }
        let mut idx = ((v - hmin) / span * BINS as f64) as usize;
        if idx >= BINS {
            idx = BINS - 1; // v == hmax lands in the last bin
        }
        counts[idx] += 1.0;
    }
    let mut val = 0.0f64;
    for &c in counts.iter() {
        let p = c / size + 1e-12;
        val -= p * p.log2();
    }
    val
}

/// scipy's `_minimize_neldermead` specialised to one variable (`tol` sets both
/// `xatol` and `fatol`). The simplex is `{x0, 1.05·x0}`; reflect/expand/
/// contract/shrink use scipy's `ρ=1, χ=2, ψ=σ=0.5` coefficients. Returns the
/// best vertex when the simplex shrinks within tolerance (default 200 evals).
fn nelder_mead_1d<F>(f: &F, x0: f64, xatol: f64, fatol: f64) -> Result<f64>
where
    F: Fn(f64) -> Result<f64>,
{
    let (rho, chi, psi, sigma) = (1.0, 2.0, 0.5, 0.5);
    let (nonzdelt, zdelt) = (0.05, 0.00025);
    let mut s0 = x0;
    let mut s1 = if x0 != 0.0 {
        (1.0 + nonzdelt) * x0
    } else {
        zdelt
    };
    let mut f0 = f(s0)?;
    let mut f1 = f(s1)?;
    if f1 < f0 {
        std::mem::swap(&mut s0, &mut s1);
        std::mem::swap(&mut f0, &mut f1);
    }
    let (maxiter, maxfun) = (200usize, 200usize);
    let mut fcalls = 2usize;
    let mut iters = 1usize;
    while fcalls < maxfun && iters < maxiter {
        if (s1 - s0).abs() <= xatol && (f0 - f1).abs() <= fatol {
            break;
        }
        let xbar = s0; // centroid of the best vertex (N = 1)
        let xr = (1.0 + rho) * xbar - rho * s1;
        let fxr = f(xr)?;
        fcalls += 1;
        let mut doshrink = false;
        if fxr < f0 {
            // Reflection improved on the best — try to expand.
            let xe = (1.0 + rho * chi) * xbar - rho * chi * s1;
            let fxe = f(xe)?;
            fcalls += 1;
            if fxe < fxr {
                s1 = xe;
                f1 = fxe;
            } else {
                s1 = xr;
                f1 = fxr;
            }
        } else if fxr < f1 {
            // Reflection between best and worst — outside contraction.
            let xc = (1.0 + psi * rho) * xbar - psi * rho * s1;
            let fxc = f(xc)?;
            fcalls += 1;
            if fxc <= fxr {
                s1 = xc;
                f1 = fxc;
            } else {
                doshrink = true;
            }
        } else {
            // Reflection worse than the worst — inside contraction.
            let xcc = (1.0 - psi) * xbar + psi * s1;
            let fxcc = f(xcc)?;
            fcalls += 1;
            if fxcc < f1 {
                s1 = xcc;
                f1 = fxcc;
            } else {
                doshrink = true;
            }
        }
        if doshrink {
            s1 = s0 + sigma * (s1 - s0);
            f1 = f(s1)?;
            fcalls += 1;
        }
        if f1 < f0 {
            std::mem::swap(&mut s0, &mut s1);
            std::mem::swap(&mut f0, &mut f1);
        }
        iters += 1;
    }
    Ok(s0)
}

// ----------------------------------------------------------------------------
// Phase-correlation internals (private). Port of skimage
// `registration/_phase_cross_correlation.py` for the 2-D, real-input,
// `normalization="phase"`, `disambiguate=False` path that tomopy's
// `find_center_pc` uses. Only the registration shift is needed (tomopy discards
// the error/phasediff), so `CCmax`/amplitudes are not computed.
// ----------------------------------------------------------------------------

/// `numpy.fft.fftfreq(n, d)` — sample frequencies for an `n`-point DFT.
fn fftfreq(n: usize, d: f64) -> Vec<f64> {
    let nn = n as f64;
    let cut = (n - 1) / 2; // (n-1)//2 — last positive-frequency index
    (0..n)
        .map(|k| {
            let idx = if k <= cut { k as f64 } else { k as f64 - nn };
            idx / (nn * d)
        })
        .collect()
}

/// Register `moving` to `reference` by phase cross-correlation. Returns the
/// `(row, col)` shift; with `upsample_factor > 1` the half-pixel refinement is a
/// direct matrix-multiply upsampled DFT (skimage `_upsampled_dft`).
fn phase_cross_correlation(
    reference: &Array2<f32>,
    moving: &Array2<f32>,
    upsample_factor: f64,
    fft: &dyn Fft,
) -> Result<(f64, f64)> {
    use std::f64::consts::PI;
    let (nrow, ncol) = reference.dim();
    let n = nrow * ncol;

    // Forward 2-D FFTs (unnormalized — matches scipy `fftn`).
    let mut rf = vec![Complex32::new(0.0, 0.0); n];
    let mut mf = vec![Complex32::new(0.0, 0.0); n];
    for i in 0..nrow {
        for j in 0..ncol {
            rf[i * ncol + j] = Complex32::new(reference[[i, j]], 0.0);
            mf[i * ncol + j] = Complex32::new(moving[[i, j]], 0.0);
        }
    }
    fft.fft_2d(&mut rf, nrow, ncol, 1, false)?;
    fft.fft_2d(&mut mf, nrow, ncol, 1, false)?;

    // image_product = rf · conj(mf), phase-normalized: divide by
    // max(|·|, 100·eps) with f32 eps (scipy uses the complex64 real dtype).
    let eps100 = 100.0 * f32::EPSILON as f64;
    let mut pr = vec![0.0f64; n];
    let mut pi = vec![0.0f64; n];
    for k in 0..n {
        let (ar, ai) = (rf[k].re as f64, rf[k].im as f64);
        let (br, bi) = (mf[k].re as f64, mf[k].im as f64);
        let re = ar * br + ai * bi; // a · conj(b)
        let im = ai * br - ar * bi;
        let denom = (re * re + im * im).sqrt().max(eps100);
        pr[k] = re / denom;
        pi[k] = im / denom;
    }

    // Whole-pixel peak: argmax|ifft2(image_product)| (C-order, first max wins).
    let mut cc = vec![Complex32::new(0.0, 0.0); n];
    for k in 0..n {
        cc[k] = Complex32::new(pr[k] as f32, pi[k] as f32);
    }
    fft.fft_2d(&mut cc, nrow, ncol, 1, true)?;
    let (mut best, mut peak) = (-1.0f64, 0usize);
    for (k, c) in cc.iter().enumerate() {
        let m = (c.re as f64).hypot(c.im as f64);
        if m > best {
            best = m;
            peak = k;
        }
    }
    let (r0, c0) = ((peak / ncol) as f64, (peak % ncol) as f64);

    // Wrap to a signed shift about the midpoint `fix(axis/2)`.
    let mut shift_r = if r0 > (nrow as f64 / 2.0).trunc() {
        r0 - nrow as f64
    } else {
        r0
    };
    let mut shift_c = if c0 > (ncol as f64 / 2.0).trunc() {
        c0 - ncol as f64
    } else {
        c0
    };

    if upsample_factor > 1.0 {
        // Refine on an upsampled grid via the matrix-multiply DFT of
        // conj(image_product) — data = (pr, −pi).
        shift_r = (shift_r * upsample_factor).round() / upsample_factor;
        shift_c = (shift_c * upsample_factor).round() / upsample_factor;
        let region_f = (upsample_factor * 1.5).ceil();
        let region = region_f as usize;
        let dftshift = (region_f / 2.0).trunc();
        let off_r = dftshift - shift_r * upsample_factor;
        let off_c = dftshift - shift_c * upsample_factor;
        let freq_c = fftfreq(ncol, upsample_factor);
        let freq_r = fftfreq(nrow, upsample_factor);

        // D1[a][r] = Σ_c exp(−2πi (a−off_c) freq_c[c]) · conj(prod)[r][c].
        let mut d1r = vec![0.0f64; region * nrow];
        let mut d1i = vec![0.0f64; region * nrow];
        for a in 0..region {
            for r in 0..nrow {
                let (mut sre, mut sim) = (0.0f64, 0.0f64);
                for (c, &fc) in freq_c.iter().enumerate() {
                    let ang = -2.0 * PI * (a as f64 - off_c) * fc;
                    let (ks, kc) = ang.sin_cos();
                    let idx = r * ncol + c;
                    let (dre, dim) = (pr[idx], -pi[idx]); // conj(image_product)
                    sre += kc * dre - ks * dim;
                    sim += kc * dim + ks * dre;
                }
                d1r[a * nrow + r] = sre;
                d1i[a * nrow + r] = sim;
            }
        }
        // D2[b][a] = Σ_r exp(−2πi (b−off_r) freq_r[r]) · D1[a][r]; argmax|D2|.
        let (mut bestup, mut mb, mut ma) = (-1.0f64, 0usize, 0usize);
        for b in 0..region {
            for a in 0..region {
                let (mut sre, mut sim) = (0.0f64, 0.0f64);
                for (r, &fr) in freq_r.iter().enumerate() {
                    let ang = -2.0 * PI * (b as f64 - off_r) * fr;
                    let (ks, kc) = ang.sin_cos();
                    let (dre, dim) = (d1r[a * nrow + r], d1i[a * nrow + r]);
                    sre += kc * dre - ks * dim;
                    sim += kc * dim + ks * dre;
                }
                let m = (sre * sre + sim * sim).sqrt();
                if m > bestup {
                    bestup = m;
                    mb = b;
                    ma = a;
                }
            }
        }
        shift_r += (mb as f64 - dftshift) / upsample_factor;
        shift_c += (ma as f64 - dftshift) / upsample_factor;
    }

    // A unit-length axis carries no shift information.
    if nrow == 1 {
        shift_r = 0.0;
    }
    if ncol == 1 {
        shift_c = 0.0;
    }
    Ok((shift_r, shift_c))
}

// ----------------------------------------------------------------------------
// scipy.ndimage.shift (order=3 cubic spline, mode='constant', cval=0) — the
// exact port used by `find_center_pc`'s `rotc_guess` pre-alignment. tomopy
// shifts both projections by `[0, -imgshift]` (rotation.py:422-423) before phase
// correlation, so the spline output feeds straight into the FFT registration and
// must match scipy to the f64 floor — unlike the deliberately-approximate
// `cubic_shift_cols` above (truncated-horizon prefilter, taps dropped at the
// boundary), whose zero-filled boundary columns the Vo metric overwrites.
//
// Mirrors scipy 1.17.1: `_interpolation.py::shift` (prefilter both axes to
// float64 with `spline_filter`, negate the shift, call `zoom_shift`),
// `ni_splines.c::apply_filter`/`_init_causal_mirror`/`_init_anticausal_mirror`/
// `get_spline_interpolation_weights`, and `ni_interpolation.c::NI_ZoomShift`
// (mode='constant' ⇒ an out-of-bounds output centre collapses to `cval`, while
// in-bounds taps are whole-sample mirror-reflected) + `map_coordinate`.
// ----------------------------------------------------------------------------

/// `scipy.ndimage.shift(img, [shift_row, shift_col], order=3, mode='constant',
/// cval=0)` for a 2-D float32 image. The public scipy API negates the shift
/// internally, so output pixel `(i, j)` samples input coordinate
/// `(i − shift_row, j − shift_col)`; out-of-bounds centres yield 0.
fn ndimage_shift_spline3_constant(
    img: &Array2<f32>,
    shift_row: f64,
    shift_col: f64,
) -> Array2<f32> {
    let (nr, nc) = img.dim();
    let mut out = Array2::<f32>::zeros((nr, nc));
    if nr == 0 || nc == 0 {
        return out;
    }
    let coeff = spline_prefilter_2d(img); // float64 B-spline coefficients
                                          // scipy negates the user shift (`shift = [-ii for ii in shift]`).
    let (s_row, s_col) = (-shift_row, -shift_col);
    let (nri, nci) = (nr as isize, nc as isize);
    let (row_hi, col_hi) = (nr as f64 - 1.0, nc as f64 - 1.0);
    for i in 0..nr {
        let cc_row = i as f64 + s_row;
        // map_coordinate(.., CONSTANT): out-of-bounds centre ⇒ cval (0).
        if cc_row < 0.0 || cc_row > row_hi {
            continue; // whole row is cval
        }
        let w_row = spline_weights3(cc_row);
        let start_row = (cc_row.floor() as isize) - 1; // order odd ⇒ floor(cc) − order/2
        let row_taps = [
            mirror_index(start_row, nri),
            mirror_index(start_row + 1, nri),
            mirror_index(start_row + 2, nri),
            mirror_index(start_row + 3, nri),
        ];
        for j in 0..nc {
            let cc_col = j as f64 + s_col;
            if cc_col < 0.0 || cc_col > col_hi {
                continue; // out-of-bounds centre ⇒ cval (0)
            }
            let w_col = spline_weights3(cc_col);
            let start_col = (cc_col.floor() as isize) - 1;
            let col_taps = [
                mirror_index(start_col, nci),
                mirror_index(start_col + 1, nci),
                mirror_index(start_col + 2, nci),
                mirror_index(start_col + 3, nci),
            ];
            // 16-tap separable sum, axis-0 (rows) outer and axis-1 (cols) inner
            // with the `coeff * w_row * w_col` multiply order, matching
            // NI_ZoomShift's fcoordinates enumeration and accumulation.
            let mut t = 0.0f64;
            for a in 0..4 {
                let cr = coeff.row(row_taps[a]);
                let wr = w_row[a];
                for b in 0..4 {
                    t += cr[col_taps[b]] * wr * w_col[b];
                }
            }
            out[[i, j]] = t as f32;
        }
    }
    out
}

/// Prefilter `img` (float32) to cubic B-spline coefficients (float64) over both
/// axes with mirror-boundary initialisation — scipy `spline_filter` runs
/// `spline_filter1d` along axis 0 then axis 1, and for `mode='constant'` the
/// order-3 `apply_filter` uses the mirror init (`ni_splines.c:295-300`).
fn spline_prefilter_2d(img: &Array2<f32>) -> Array2<f64> {
    let (nr, nc) = img.dim();
    let z = 3.0f64.sqrt() - 2.0; // the single order-3 pole √3 − 2
    let mut c = Array2::<f64>::from_shape_fn((nr, nc), |(i, j)| img[[i, j]] as f64);
    // axis 0: filter each column (a line along axis 0). `len == 1` ⇒ no filter.
    if nr > 1 {
        let mut line = vec![0.0f64; nr];
        for j in 0..nc {
            for (i, v) in line.iter_mut().enumerate() {
                *v = c[[i, j]];
            }
            apply_spline_filter_mirror(&mut line, z);
            for (i, &v) in line.iter().enumerate() {
                c[[i, j]] = v;
            }
        }
    }
    // axis 1: filter each row (a line along axis 1).
    if nc > 1 {
        let mut line = vec![0.0f64; nc];
        for i in 0..nr {
            for (j, v) in line.iter_mut().enumerate() {
                *v = c[[i, j]];
            }
            apply_spline_filter_mirror(&mut line, z);
            for (j, &v) in line.iter().enumerate() {
                c[[i, j]] = v;
            }
        }
    }
    c
}

/// scipy `ni_splines.c::apply_filter` for a single order-3 pole with mirror
/// initialisation: gain `(1−z)(1−1/z)` applied first, then one causal/anticausal
/// pass — `_init_causal_mirror`, forward recursion, `_init_anticausal_mirror`,
/// backward recursion. Caller guarantees `len ≥ 2`.
fn apply_spline_filter_mirror(c: &mut [f64], z: f64) {
    let n = c.len();
    // _apply_filter_gain.
    let gain = (1.0 - z) * (1.0 - 1.0 / z);
    for v in c.iter_mut() {
        *v *= gain;
    }
    // _init_causal_mirror: c[0] = (c[0] + Σ_{i} z^i·(c[i] + z^{n-1}·c[n-1-i])) / (1 − z^{2(n-1)}).
    // `powf` (libm `pow`) matches scipy's `pow(z, n-1)` to the last bit; `powi`
    // (exponentiation by squaring) can differ in the final ULP.
    let z_n_1 = z.powf((n - 1) as f64);
    let mut acc = c[0] + z_n_1 * c[n - 1];
    let mut z_i = z;
    for i in 1..n - 1 {
        acc += z_i * (c[i] + z_n_1 * c[n - 1 - i]);
        z_i *= z;
    }
    c[0] = acc / (1.0 - z_n_1 * z_n_1);
    // Forward (causal) recursion.
    for i in 1..n {
        c[i] += z * c[i - 1];
    }
    // _init_anticausal_mirror.
    c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1.0);
    // Backward (anticausal) recursion.
    for i in (0..n - 1).rev() {
        c[i] = z * (c[i + 1] - c[i]);
    }
}

/// scipy `ni_splines.c::get_spline_interpolation_weights` for order 3: the cubic
/// B-spline weights at fractional offset `frac(cc)`, for the four taps
/// `floor(cc)−1 … floor(cc)+2`. The last weight is `1 − Σ` (in scipy's order).
fn spline_weights3(cc: f64) -> [f64; 4] {
    let x = cc - cc.floor(); // order is odd ⇒ x -= floor(x)
    let y = x;
    let z = 1.0 - x;
    let mut w = [0.0f64; 4];
    w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
    w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
    w[0] = z * z * z / 6.0;
    w[3] = 1.0 - w[0] - w[1] - w[2];
    w
}

/// scipy `map_coordinate(idx, len, NI_EXTEND_MIRROR)` for an integer tap index:
/// whole-sample reflection (period `2·len−2`, edges not repeated).
fn mirror_index(idx: isize, len: isize) -> usize {
    if len == 1 {
        return 0;
    }
    let mut v = idx;
    if v < 0 {
        let sz2 = 2 * len - 2;
        v = sz2 * ((-v) / sz2) + v;
        v = if v <= 1 - len { v + sz2 } else { -v };
    } else if v > len - 1 {
        let sz2 = 2 * len - 2;
        v -= sz2 * (v / sz2);
        if v >= len {
            v = sz2 - v;
        }
    }
    v as usize
}

// ----------------------------------------------------------------------------
// Vo's method internals (private). Mirrors tomopy rotation.py:236-388.
// ----------------------------------------------------------------------------

/// Pull a single 2-D sinogram slice `(angle, col)`, averaging ±5 rows around
/// `ind` for SNR when there are more than 10 slices (tomopy rotation.py:240-249).
fn extract_slice(tomo: &Tomo<f32>, ind: Option<usize>) -> Array2<f32> {
    let sino = tomo.as_layout(Layout::Sinogram); // [row, angle, col]
    let nrows = sino.array.dim().0;
    let ind = ind.unwrap_or(nrows / 2).min(nrows.saturating_sub(1));
    if nrows > 10 {
        let lo = ind.saturating_sub(5);
        let hi = (ind + 5).min(nrows);
        sino.array
            .slice_axis(Axis(0), Slice::from(lo..hi))
            .mean_axis(Axis(0))
            .unwrap()
    } else {
        sino.array.index_axis(Axis(0), ind).to_owned()
    }
}

/// scipy `gaussian_filter` weights for one axis: a normalised Gaussian truncated
/// at `truncate=4.0` standard deviations (`radius = int(4·σ + 0.5)`).
fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (4.0 * sigma + 0.5) as isize;
    let mut k = Vec::with_capacity((2 * radius + 1) as usize);
    let mut sum = 0.0f32;
    for i in -radius..=radius {
        let w = (-0.5 * (i as f32 / sigma).powi(2)).exp();
        k.push(w);
        sum += w;
    }
    for w in &mut k {
        *w /= sum;
    }
    k
}

/// scipy `'reflect'` boundary (half-sample symmetric: `d c b a | a b c d`).
fn reflect(p: isize, n: isize) -> usize {
    if n == 1 {
        return 0;
    }
    let m = 2 * n;
    let mut q = p % m;
    if q < 0 {
        q += m;
    }
    if q >= n {
        q = m - 1 - q;
    }
    q as usize
}

/// Separable 2-D Gaussian blur, `sigma0` along the angle axis and `sigma1`
/// along the column axis, `'reflect'` boundary (tomopy uses `mode='reflect'`).
fn gaussian_filter2d(img: &Array2<f32>, sigma0: f32, sigma1: f32) -> Array2<f32> {
    let (nr, nc) = img.dim();

    // Pass 1: columns (axis 1).
    let k1 = gaussian_kernel(sigma1);
    let r1 = (k1.len() / 2) as isize;
    let mut tmp = Array2::<f32>::zeros((nr, nc));
    for i in 0..nr {
        for j in 0..nc {
            let mut acc = 0.0f32;
            for (t, &w) in k1.iter().enumerate() {
                let jj = reflect(j as isize + t as isize - r1, nc as isize);
                acc += w * img[[i, jj]];
            }
            tmp[[i, j]] = acc;
        }
    }

    // Pass 2: rows (axis 0).
    let k0 = gaussian_kernel(sigma0);
    let r0 = (k0.len() / 2) as isize;
    let mut out = Array2::<f32>::zeros((nr, nc));
    for j in 0..nc {
        for i in 0..nr {
            let mut acc = 0.0f32;
            for (t, &w) in k0.iter().enumerate() {
                let ii = reflect(i as isize + t as isize - r0, nr as isize);
                acc += w * tmp[[ii, j]];
            }
            out[[i, j]] = acc;
        }
    }
    out
}

/// Column block-mean downsample by `2^level` (tomopy `downsample(level, axis=2)`).
/// argmin of the metric is scale-invariant, so mean vs sum is immaterial here.
fn downsample_cols(img: &Array2<f32>, level: u32) -> Array2<f32> {
    let f = 1usize << level;
    let (nr, nc) = img.dim();
    let nc2 = nc / f;
    let mut out = Array2::<f32>::zeros((nr, nc2));
    for i in 0..nr {
        for k in 0..nc2 {
            let mut s = 0.0f32;
            for d in 0..f {
                s += img[[i, k * f + d]];
            }
            out[[i, k]] = s / f as f32;
        }
    }
    out
}

/// Reverse columns (`np.fliplr`).
fn fliplr(a: &Array2<f32>) -> Array2<f32> {
    let (nr, nc) = a.dim();
    Array2::from_shape_fn((nr, nc), |(i, j)| a[[i, nc - 1 - j]])
}

/// Reverse rows (`np.flipud`).
fn flipud(a: &Array2<f32>) -> Array2<f32> {
    let (nr, nc) = a.dim();
    Array2::from_shape_fn((nr, nc), |(i, j)| a[[nr - 1 - i, j]])
}

fn clipf(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

/// Coarse search on the 0.5-pixel grid (tomopy `_search_coarse`).
fn search_coarse(
    sino: &Array2<f32>,
    smin: f32,
    smax: f32,
    ratio: f32,
    drop: i32,
    fft: &dyn Fft,
) -> Result<f32> {
    let (nrow, ncol) = sino.dim();
    let ncolf = ncol as f32;
    let cen_fliplr = (ncolf - 1.0) / 2.0;
    // np.int16(np.clip(s + cen, 0, ncol-1) - cen): truncates toward zero.
    let smin_i = (clipf(smin + cen_fliplr, 0.0, ncolf - 1.0) - cen_fliplr) as i32;
    let smax_i = (clipf(smax + cen_fliplr, 0.0, ncolf - 1.0) - cen_fliplr) as i32;
    let start_cor = (ncol / 2) as i32 + smin_i;
    let stop_cor = (ncol / 2) as i32 + smax_i;

    let flip = fliplr(sino);
    let comp = flipud(sino);
    let mask = create_mask(2 * nrow, ncol, 0.5 * ratio * ncolf, drop);

    // list_cor = np.arange(start_cor, stop_cor + 0.5, 0.5)
    let stop = stop_cor as f64 + 0.5;
    let (mut best, mut best_cor) = (f32::INFINITY, start_cor as f32);
    let mut k = 0i64;
    loop {
        let cor = start_cor as f64 + 0.5 * k as f64;
        if cor >= stop {
            break;
        }
        let shift = 2.0 * (cor - cen_fliplr as f64);
        let m = calculate_metric(shift, sino, &flip, &comp, &mask, fft)?;
        if m < best {
            best = m;
            best_cor = cor as f32;
        }
        k += 1;
    }
    Ok(best_cor)
}

/// Fine subpixel search around `init_cen` (tomopy `_search_fine`).
fn search_fine(
    sino: &Array2<f32>,
    srad: f32,
    step: f32,
    init_cen: f32,
    ratio: f32,
    drop: i32,
    fft: &dyn Fft,
) -> Result<f32> {
    let (nrow, ncol) = sino.dim();
    let ncolf = ncol as f32;
    let cen_fliplr = (ncolf - 1.0) / 2.0;
    let srad = clipf(srad.abs(), 1.0, ncolf / 4.0);
    let step = clipf(step.abs(), 0.1, srad);
    let init_cen = clipf(init_cen, srad, ncolf - srad - 1.0);

    let flip = fliplr(sino);
    let comp = flipud(sino);
    let mask = create_mask(2 * nrow, ncol, 0.5 * ratio * ncolf, drop);

    // list_cor = init_cen + np.arange(-srad, srad + step, step)
    let stop = (srad + step) as f64;
    let (mut best, mut best_cor) = (f32::INFINITY, init_cen - srad);
    let mut k = 0i64;
    loop {
        let off = -(srad as f64) + step as f64 * k as f64;
        if off >= stop {
            break;
        }
        let cor = init_cen as f64 + off;
        let shift = 2.0 * (cor - cen_fliplr as f64);
        let m = calculate_metric(shift, sino, &flip, &comp, &mask, fft)?;
        if m < best {
            best = m;
            best_cor = cor as f32;
        }
        k += 1;
    }
    Ok(best_cor)
}

/// The double-wedge binary mask, Eq.(3) of doi:10.1364/OE.22.019078
/// (tomopy `_create_mask`). `nrow` here is `2 ·` the sinogram angle count.
fn create_mask(nrow: usize, ncol: usize, radius: f32, drop: i32) -> Array2<f32> {
    let nrowf = nrow as f32;
    let ncolf = ncol as f32;
    let du = 1.0 / ncolf;
    let dv = (nrowf - 1.0) / (nrowf * 2.0 * std::f32::consts::PI);
    let cen_row = (nrowf / 2.0).ceil() as i32 - 1;
    let cen_col = (ncolf / 2.0).ceil() as i32 - 1;
    let drop = drop.min((0.05 * nrowf).ceil() as i32);

    let mut mask = Array2::<f32>::zeros((nrow, ncol));
    let ncol_i = ncol as i32;
    for i in 0..nrow as i32 {
        let val = (((i - cen_row) as f32) * dv / radius) / du;
        let pos = val.ceil() as i32; // np.int16(np.ceil(..)): toward zero.
        let (a, b) = (-pos + cen_col, pos + cen_col);
        let (p1, p2) = if a <= b { (a, b) } else { (b, a) };
        let p1 = p1.clamp(0, ncol_i - 1);
        let p2 = p2.clamp(0, ncol_i - 1);
        for j in p1..=p2 {
            mask[[i as usize, j as usize]] = 1.0;
        }
    }
    // Blank ±drop rows around the DC line and the three central columns.
    let r0 = (cen_row - drop).max(0);
    let r1 = (cen_row + drop).min(nrow as i32 - 1);
    for i in r0..=r1 {
        for j in 0..ncol {
            mask[[i as usize, j]] = 0.0;
        }
    }
    let c0 = (cen_col - 1).max(0);
    let c1 = (cen_col + 1).min(ncol_i - 1);
    for mut row in mask.rows_mut() {
        for j in c0..=c1 {
            row[j as usize] = 0.0;
        }
    }
    mask
}

/// The metric for one candidate `shift` (tomopy `_calculate_metric`): stack the
/// sinogram on its flipped-and-shifted copy, FFT, and average the
/// double-wedge-masked spectrum. Integer shifts use a circular roll; fractional
/// shifts use a cubic B-spline column shift. The wrapped/zero-filled boundary
/// columns are overwritten with the row-flipped sinogram (`comp`).
fn calculate_metric(
    shift: f64,
    sino: &Array2<f32>,
    flip: &Array2<f32>,
    comp: &Array2<f32>,
    mask: &Array2<f32>,
    fft: &dyn Fft,
) -> Result<f32> {
    let (nrow, ncol) = sino.dim();
    let nc = ncol as i32;

    let ss = if (shift - shift.round()).abs() < 1e-9 {
        let s = shift.round() as i32;
        let mut ss = Array2::<f32>::zeros((nrow, ncol));
        for i in 0..nrow {
            for j in 0..ncol {
                let src = ((j as i32 - s) % nc + nc) % nc;
                ss[[i, j]] = flip[[i, src as usize]];
            }
        }
        if s >= 0 {
            for i in 0..nrow {
                for j in 0..(s as usize).min(ncol) {
                    ss[[i, j]] = comp[[i, j]];
                }
            }
        } else {
            let from = (nc + s).max(0) as usize;
            for i in 0..nrow {
                for j in from..ncol {
                    ss[[i, j]] = comp[[i, j]];
                }
            }
        }
        ss
    } else {
        let mut ss = cubic_shift_cols(flip, shift);
        if shift >= 0.0 {
            let si = (shift.ceil() as i32).max(0) as usize;
            for i in 0..nrow {
                for j in 0..si.min(ncol) {
                    ss[[i, j]] = comp[[i, j]];
                }
            }
        } else {
            let from = (nc + shift.floor() as i32).max(0) as usize;
            for i in 0..nrow {
                for j in from..ncol {
                    ss[[i, j]] = comp[[i, j]];
                }
            }
        }
        ss
    };

    // mat = vstack((sino, ss)) -> (2·nrow, ncol), then fft2.
    let rr = 2 * nrow;
    let mut buf = vec![Complex32::new(0.0, 0.0); rr * ncol];
    for i in 0..nrow {
        for j in 0..ncol {
            buf[i * ncol + j] = Complex32::new(sino[[i, j]], 0.0);
            buf[(nrow + i) * ncol + j] = Complex32::new(ss[[i, j]], 0.0);
        }
    }
    fft.fft_2d(&mut buf, rr, ncol, 1, false)?;

    // metric = mean(|fftshift(fft2(mat))| * mask). Fold the fftshift into the
    // mask index so no shifted copy is materialised.
    let mut acc = 0.0f64;
    for i in 0..rr {
        let mi = (i + rr / 2) % rr;
        for j in 0..ncol {
            let mj = (j + ncol / 2) % ncol;
            acc += buf[i * ncol + j].norm() as f64 * mask[[mi, mj]] as f64;
        }
    }
    Ok((acc / (rr * ncol) as f64) as f32)
}

/// Cubic B-spline column shift, content moved right by `shift`, zero outside
/// (scipy `ndimage.shift(order=3, mode='constant')`). Each row is prefiltered
/// to B-spline coefficients (Unser/Thévenaz, mirror boundary) then resampled.
/// The boundary columns this routine zero-fills are overwritten by the caller,
/// so the prefilter's boundary convention has no effect on the metric.
fn cubic_shift_cols(img: &Array2<f32>, shift: f64) -> Array2<f32> {
    let (nr, nc) = img.dim();
    let mut out = Array2::<f32>::zeros((nr, nc));
    let mut coeff = vec![0.0f64; nc];
    for i in 0..nr {
        for j in 0..nc {
            coeff[j] = img[[i, j]] as f64;
        }
        prefilter_cubic(&mut coeff);
        for j in 0..nc {
            out[[i, j]] = eval_cubic(&coeff, j as f64 - shift) as f32;
        }
    }
    out
}

/// In-place cubic B-spline prefilter (single pole `√3 − 2`, gain 6),
/// Thévenaz "Interpolation Revisited" recursion with a truncated mirror init.
fn prefilter_cubic(c: &mut [f64]) {
    let n = c.len();
    if n < 2 {
        return;
    }
    let z = 3.0f64.sqrt() - 2.0;
    let lambda = (1.0 - z) * (1.0 - 1.0 / z); // = 6
    for v in c.iter_mut() {
        *v *= lambda;
    }
    // Causal initialisation, horizon truncated at tol 1e-9.
    let horizon = ((1e-9f64.ln() / z.abs().ln()).ceil() as usize).min(n);
    let mut zn = z;
    let mut sum = c[0];
    for v in c.iter().take(horizon).skip(1) {
        sum += zn * v;
        zn *= z;
    }
    c[0] = sum;
    for k in 1..n {
        c[k] += z * c[k - 1];
    }
    // Anticausal initialisation and recursion.
    c[n - 1] = (z / (z * z - 1.0)) * (z * c[n - 2] + c[n - 1]);
    for k in (0..n - 1).rev() {
        c[k] = z * (c[k + 1] - c[k]);
    }
}

/// Evaluate the cubic B-spline with coefficients `c` at position `x`,
/// zero outside `[0, n)` (mode `'constant'`).
fn eval_cubic(c: &[f64], x: f64) -> f64 {
    let n = c.len() as i64;
    let xf = x.floor();
    let base = xf as i64;
    let mut val = 0.0;
    for k in -1..=2 {
        let idx = base + k;
        if idx >= 0 && idx < n {
            val += beta3((x - (xf + k as f64)).abs()) * c[idx as usize];
        }
    }
    val
}

/// Cubic B-spline kernel β₃.
fn beta3(t: f64) -> f64 {
    if t < 1.0 {
        2.0 / 3.0 - t * t + 0.5 * t * t * t
    } else if t < 2.0 {
        let a = 2.0 - t;
        a * a * a / 6.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::{judge_sweep, ndimage_shift_spline3_constant, SweepVerdict};
    use ndarray::{Array2, Array3};
    use ndarray_npy::read_npy;

    /// The pouch-cell centre sweep at tilt 44°, scored on output slice rh/2 by
    /// `slice_focus`, in units of 1e-6. Candidates are 356..=436 px, step 1.
    ///
    /// This is the curve the whole [`SweepVerdict`] type exists for. Its argmax is
    /// 417 and its correct answer is 396 — the axis is known independently, from
    /// the rings (docs/LAMINOGRAPHY_ALIGNMENT.md §1) — and 417 outscores 396 by
    /// 0.34 % with neither at an edge.
    const WIDE: [f64; 81] = [
        1.9491, 1.9521, 1.9564, 1.9635, 1.9709, 1.9765, 1.9824, 1.9879, 1.9892, 1.9892, 1.9910,
        1.9971, 2.0092, 2.0228, 2.0306, 2.0338, 2.0348, 2.0357, 2.0372, 2.0352, 2.0272, 2.0167,
        2.0056, 1.9913, 1.9757, 1.9649, 1.9585, 1.9581, 1.9594, 1.9614, 1.9694, 1.9845, 2.0095,
        2.0437, 2.0865, 2.1321, 2.1749, 2.2101, 2.2364, 2.2557, 2.2625, 2.2521, 2.2318, 2.2077,
        2.1832, 2.1593, 2.1369, 2.1182, 2.1022, 2.0909, 2.0855, 2.0880, 2.0985, 2.1178, 2.1412,
        2.1663, 2.1944, 2.2219, 2.2412, 2.2536, 2.2645, 2.2701, 2.2694, 2.2666, 2.2618, 2.2572,
        2.2502, 2.2407, 2.2312, 2.2219, 2.2075, 2.1902, 2.1745, 2.1638, 2.1576, 2.1581, 2.1680,
        2.1814, 2.1933, 2.2041, 2.2153,
    ];

    fn wide_cands() -> Vec<f32> {
        (0..81).map(|k| 356.0 + k as f32).collect()
    }

    /// The defect this type was introduced for: a sweep whose maximum is interior
    /// — so no rail check fires — and yet is the wrong answer, because a rival
    /// lobe 21 px away is within 0.34 % of it. The old `Pick { railed }` reported
    /// 417 here as an answer. Nothing may hand back a value for this curve.
    #[test]
    fn judge_sweep_refuses_a_curve_whose_rival_lobe_it_cannot_separate() {
        let v = judge_sweep(&wide_cands(), &WIDE).unwrap();
        assert_eq!(
            v.best(),
            (61, 417.0),
            "the argmax is unchanged — the verdict is what changed"
        );
        assert_eq!(
            v.resolved(),
            None,
            "an interior argmax is not automatically an answer"
        );

        let SweepVerdict::Ambiguous { rivals, .. } = &v else {
            panic!("expected Ambiguous, got {v:?}");
        };
        // Most prominent first, and the most prominent rival is the correct axis.
        assert_eq!(rivals[0].value, 396.0);
        assert!(
            (rivals[0].deficit - 0.0033).abs() < 0.0005,
            "the correct lobe trails the winner by {:.4}, not the measured 0.34 %",
            rivals[0].deficit
        );
        // 374 competes too (25 % of the winner's prominence); the 436 shoulder at
        // 18 % is below the bar and must not be listed.
        assert_eq!(
            rivals.iter().map(|r| r.value).collect::<Vec<_>>(),
            vec![396.0, 374.0]
        );
    }

    /// The same metric, same slice, same scan — narrowed onto a prior from the
    /// rings. One lobe, so it resolves, and it lands on the known 396. This is the
    /// pair that says `Ambiguous` is about the *window*, not a metric that never
    /// works: the sweep is a refiner of a prior, not a finder.
    ///
    /// Its peak is a two-sample exact tie (395.75 and 396.00 both 2.2623e-6),
    /// which is why lobe detection has to be plateau-aware — a strict rise on both
    /// sides finds no lobe at all here and the curve would judge `Flat`.
    #[test]
    fn judge_sweep_resolves_the_same_sweep_narrowed_onto_a_prior() {
        let narrow: [f64; 17] = [
            2.2364, 2.2419, 2.2470, 2.2517, 2.2557, 2.2590, 2.2612, 2.2623, 2.2623, 2.2610, 2.2587,
            2.2554, 2.2514, 2.2468, 2.2419, 2.2366, 2.2310,
        ];
        let cands: Vec<f32> = (0..17).map(|k| 394.0 + k as f32 * 0.25).collect();
        let v = judge_sweep(&cands, &narrow).unwrap();
        assert_eq!(v.resolved(), Some(395.75), "got {v:?}");
    }

    /// An interior maximum with nothing else near it is an optimum; a boundary one
    /// is the range running out. A railed pick adopted as an answer is how one bad
    /// axis poisons the next.
    #[test]
    fn judge_sweep_separates_an_optimum_from_a_range_running_out() {
        let c = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let v = judge_sweep(&c, &[1.0, 5.0, 9.0, 4.0, 2.0]).unwrap();
        assert_eq!(v.resolved(), Some(3.0));

        for (score, want) in [
            ([9.0, 5.0, 3.0, 2.0, 1.0], 0usize),
            ([1.0, 2.0, 3.0, 5.0, 9.0], 4usize),
        ] {
            let v = judge_sweep(&c, &score).unwrap();
            assert!(
                matches!(v, SweepVerdict::Railed { index, .. } if index == want),
                "a peak at candidate {want} of 5 is a boundary, got {v:?}"
            );
            assert_eq!(v.resolved(), None);
        }
    }

    /// The measured ±40 px centre sweep on the **sample plane** (z=899, the round
    /// particles of the eye check) of the aligned pouch scan, tilt 44°, known axis
    /// 396. In units of 1e-7.
    ///
    /// A range that ran out without landing on a candidate index: the curve climbs
    /// across the whole window and turns over one sample short of the far edge.
    const RAMP: [f64; 81] = [
        4.1183, 4.1169, 4.1190, 4.1315, 4.1298, 4.1197, 4.1210, 4.1252, 4.1388, 4.1493, 4.1455,
        4.1588, 4.1625, 4.1693, 4.1633, 4.1294, 4.1122, 4.1089, 4.0989, 4.0949, 4.1065, 4.1236,
        4.1347, 4.1329, 4.1241, 4.1348, 4.1513, 4.1504, 4.1573, 4.1552, 4.1551, 4.1665, 4.1770,
        4.1792, 4.1778, 4.1795, 4.1852, 4.1893, 4.1866, 4.1959, 4.2149, 4.2293, 4.2380, 4.2457,
        4.2510, 4.2529, 4.2503, 4.2527, 4.2580, 4.2700, 4.2708, 4.2754, 4.2911, 4.3181, 4.3312,
        4.3420, 4.3433, 4.3342, 4.3294, 4.3467, 4.3753, 4.3886, 4.4000, 4.4071, 4.4062, 4.3896,
        4.3878, 4.3984, 4.4147, 4.4216, 4.4220, 4.4096, 4.4092, 4.4127, 4.4103, 4.4205, 4.4423,
        4.4492, 4.4645, 4.4749, 4.4738,
    ];

    /// The edge check alone cannot see a range that ran out, and this is the curve
    /// that proves it: the maximum is **interior** (index 79 of 80), so no index
    /// test fires, and the curve carries no second lobe, so no rival contradicts
    /// it. It is still nothing but a climb to the edge, and the answer it names —
    /// 435 against a known 396 — is 39 px wrong.
    ///
    /// What separates it from a peak is the one bar every lobe is held to: the
    /// winner stands 0.29 % of the curve's height above the trough that flanks it
    /// on the right, because there is no trough on the right — only one more
    /// sample, and then the range ends.
    #[test]
    fn judge_sweep_does_not_answer_from_a_curve_that_only_climbs() {
        let cands: Vec<f32> = (0..81).map(|k| 356.0 + k as f32).collect();
        let v = judge_sweep(&cands, &RAMP).unwrap();
        assert!(
            matches!(v, SweepVerdict::Railed { index: 79, .. }),
            "an interior maximum one sample short of the edge is still the range \
             running out, got {v:?}"
        );
        assert_eq!(v.resolved(), None, "435 is 39 px from the known 396");

        // Not by luck of the grid: drop the one sample past the peak and the same
        // curve rails on the index instead. Both shapes, one verdict.
        let v = judge_sweep(&cands[..80], &RAMP[..80]).unwrap();
        assert!(
            matches!(v, SweepVerdict::Railed { index: 79, .. }),
            "got {v:?}"
        );
    }

    /// A curve that never moved says nothing, and its argmax is just its first
    /// entry. Uniform reconstruction output — which has shipped from this crate —
    /// scores exactly this.
    #[test]
    fn judge_sweep_does_not_answer_from_a_curve_that_never_responded() {
        let v = judge_sweep(&[1.0f32, 2.0, 3.0, 4.0], &[0.0; 4]).unwrap();
        assert!(matches!(v, SweepVerdict::Flat { .. }), "got {v:?}");
        assert_eq!(v.resolved(), None);
    }

    /// A sweep of one or two candidates is all boundary: there is no interior for
    /// a peak to sit in and nothing to widen toward, so calling it railed would
    /// make every degenerate sweep an error.
    #[test]
    fn judge_sweep_does_not_rail_check_a_degenerate_sweep() {
        assert_eq!(judge_sweep(&[7.0], &[1.0]).unwrap().resolved(), Some(7.0));
        assert_eq!(
            judge_sweep(&[7.0, 8.0], &[9.0, 1.0]).unwrap().resolved(),
            Some(7.0)
        );
        assert!(judge_sweep(&[], &[]).is_none());
        // Mismatched lengths are a caller bug, not a pick.
        assert!(judge_sweep(&[1.0, 2.0], &[1.0]).is_none());
    }

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

    /// The cubic-spline `ndimage.shift` port reproduces scipy 1.17.1 bit-for-bit
    /// (order=3, mode='constant', cval=0) across fractional/integer shifts of
    /// both signs, including out-of-bounds (cval) and mirror-tap edge cases.
    #[test]
    fn ndimage_shift_matches_scipy() {
        let input: Array2<f32> = read_npy(format!("{FIXTURES}/ndimage_shift_input.npy")).unwrap();
        let outputs: Array3<f32> =
            read_npy(format!("{FIXTURES}/ndimage_shift_outputs.npy")).unwrap();
        let params: Array2<f64> = read_npy(format!("{FIXTURES}/ndimage_shift_params.npy")).unwrap();
        let ncases = params.dim().0;
        assert_eq!(outputs.dim().0, ncases);

        for k in 0..ncases {
            let (sr, sc) = (params[[k, 0]], params[[k, 1]]);
            let got = ndimage_shift_spline3_constant(&input, sr, sc);
            let want = outputs.index_axis(ndarray::Axis(0), k);
            let mut max_abs = 0.0f32;
            let mut n_mismatch = 0usize;
            for (g, w) in got.iter().zip(want.iter()) {
                let d = (g - w).abs();
                if d > max_abs {
                    max_abs = d;
                }
                if g.to_bits() != w.to_bits() {
                    n_mismatch += 1;
                }
            }
            assert_eq!(
                n_mismatch, 0,
                "case {k} shift=({sr},{sc}): {n_mismatch} f32 bit-mismatches vs scipy, max|Δ|={max_abs:e}"
            );
        }
    }
}
