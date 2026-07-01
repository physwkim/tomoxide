# tomoxide — Empirical Reconstruction Benchmarks

Companion to [`ALGORITHMS.md`](ALGORITHMS.md), which lists each method's design
properties. That guide notes a per-sample *quality* comparison had not been run;
**this document is that comparison** — measured reconstruction quality, speed, and
iteration behaviour of the analytic and iterative methods on real data.

> **Scope & honesty of the claims.** All numbers below are from **one real
> dataset** (a low-contrast, smooth-object micro-CT scan) on **one GPU**
> (RTX 5000 Ada, CUDA, release build). They are a concrete data point, **not a
> universal ranking** — a high-contrast, piecewise-constant object (metal/pore
> samples) is exactly where total-variation shines more and the trade-offs shift.
> Regularisation strength `λ` (`--reg_par`) is held at a common value except where
> a sweep is stated, so cross-method quality reflects *behaviour*, not each
> method's individually-tuned best. Reproduce on your own data before relying on a
> specific ordering.

---

## 1. Method & metric

- **Data.** A real micro-CT scan (`450` projections over 180°, central 64-row
  band, 800 px wide). The native file is a beamline absorbance stack; it was
  converted to DXchange (`data = exp(−absorbance)`, flat = 1, dark = 0,
  `theta = linspace(0,180,nproj)`), so tomoxide's minus-log recovers the measured
  absorbance. Three variants:
  - **dense** — all 450 views (well-conditioned);
  - **sparse** — every 7th view = 65 views (ill-posed);
  - **noisy** — sparse + simulated Poisson photon noise (`I₀ = 100`, ≈ 10 %).
- **Reference (quality proxy).** Real data has no ground truth, so the yardstick
  is the **dense 450-view SIRT@200** reconstruction of the same rows — the best
  available estimate of the true object. Sparse/noisy reconstructions are scored
  against it.
- **Metric.** **Pearson correlation** (scale- and offset-invariant, so it is not
  fooled by the amplitude-convention differences between methods) is the primary
  discriminator; relative L2 is reported where informative.
- **Semi-convergence caveat.** Because the yardstick is the *object* (dense ref),
  quality-vs-iteration curves **peak then decline**: an iterative method first
  approaches the object, then converges onto the *sparse* data's own solution
  (streaks/noise absent from the dense ref) and moves away. This "distance to the
  true object" view is deliberate — it is what shows the practical optimal
  iteration count. Measured against the sparse method's own limit the curves would
  be monotone instead.

---

## 2. Analytic: FBP vs gridrec (CUDA)

Dense 450-view, 64 slices, 800².

| method | path | wall | quality (Pearson) |
|--------|------|-----:|------------------:|
| `fbp` (parzen) | device-resident fused filter→backproject | **0.98 s** | **0.870** |
| `fbp` (ramp)   | same | 0.94 s | 0.839 |
| `gridrec`      | cuFFT + **host** gridding (not device-resident) | 3.09 s | 0.731 |

- **Kernels (nsys).** FBP's GPU time is 88.5 % `backprojection_ker` (the real
  O(N²·Nθ) work), with a small FFT-based filter pass. gridrec's GPU time is 82 %
  cuFFT (`regular_fft`/`vector_fft`) — **no back-projection kernel**, because the
  polar→Cartesian gridding runs on the host. gridrec's *GPU* kernel time is only
  ~11 ms (vs FBP 49 ms) yet its wall is 3.2× longer → it is **host-bound**, not a
  true GPU reconstruction.
- **Filter.** FBP honours `--filter` (apodisation); **gridrec applies a pure
  ramp only** (`filter_name` is ignored — see `recon/gridrec.rs`).
- **Amplitude.** gridrec output ≈ `0.0011 ×` FBP — a different normalisation
  convention (gridrec is **excluded** from the CUDA convention unification).
  Orientation matches FBP (no flip).
- **Verdict.** On CUDA, **FBP wins on this box**: faster (fused GPU
  back-projection) *and* higher fidelity (apodisation). gridrec's O(N² log N)
  advantage is unrealised while its gridding is host-side.

---

## 3. FBP apodisation filter

Relative high-frequency energy (clean, ramp = 1.0):
`ramp 1.00 > shepp 0.56 > cosine 0.12 > hamming 0.057 > cosine2 = hann 0.039 > parzen 0.003`.
(`none` = no filter = 1/r blur, not a valid reconstruction.)

| filter | clean Pearson | noisy Pearson |
|--------|--------------:|--------------:|
| ramp   | 0.839 | 0.068 |
| shepp  | 0.883 | 0.078 |
| cosine | 0.917 | 0.105 |
| **hamming** | **0.920** | 0.125 |
| cosine2 / hann | 0.908 | 0.131 |
| **parzen** | 0.870 | **0.262** |

- **Sharpness ↔ noise trade-off.** On **clean** data quality is **U-shaped** —
  mid apodisation (`hamming`/`cosine`) is best; `ramp` is too noisy, `parzen`
  over-smooths real detail. On **noisy** data the order **flips**: strong
  apodisation (`parzen`) wins, `ramp` is worst.
- The default `parzen` is a **safe choice for noisy real data** but slightly
  over-smooths clean, well-sampled data (0.870 vs hamming 0.920).
- `hann` and `cosine2` are **numerically identical** in this implementation
  (Hann window = cos²).
- `none` is not a reconstruction: it looks high on the noisy metric (0.648) only
  because a smooth blur happens to correlate with a smooth reference — no real
  detail is recovered.

---

## 4. Iterative methods at equal iterations (40 iters, 65-view sparse)

### 4.1 Speed & feasibility

- Device-resident methods (`sirt`, `cgls`, `mlem`, `osem`, `tv`, `grad`, `tikh`,
  `pml_*`, `ospml_*`) are all comparable, ~2.6–5.9 s at 40 iters (`cgls` and
  `sirt` are near-identical per iteration — each is one forward + one
  back-projection; `cgls` 40 iters 5.44 s vs `sirt` 5.87 s).
- **`art`/`bart` are host row-action code** (not device-resident): a *single*
  800² slice at 40 iters did **not finish in 115 s** (full volume > 2 h).
  Impractical at this resolution — reserve for small images.

### 4.2 Data-model applicability — negative projections

The real low-contrast sinogram is **29.5 % negative** (weakly-absorbing sample →
baseline near zero, noise dips below it). The emission/statistical family
**requires non-negative projections and blows up on it**:

| family | on raw (has negatives) |
|--------|------------------------|
| `sirt`, `cgls`, `tv`, `grad`, `tikh` (algebraic / least-squares) | **robust** (bounded) |
| `mlem`, `osem` (multiplicative Poisson) | **diverge** (output ±thousands, relL2 44–50) |
| `pml_*`, `ospml_*` (penalised-ML) | **NaN** |

This is a **data-model mismatch, not a bug**: MLEM's `b/(Ax)` update goes wrong
when `b < 0`. Clamping to non-negative makes MLEM reconstruct fine (0.787).

### 4.3 Handling the negatives

Negative attenuation is physically impossible, so **floor projections at 0**
(clip transmission ≤ 1). Then all methods are stable:

| method | clean | noisy | family |
|--------|------:|------:|--------|
| **sirt** | **0.836** | 0.535 | algebraic |
| tv (λ=0.001) | 0.818 | 0.628 | total-variation |
| pml_hybrid / ospml_hybrid | 0.788 | 0.320 | penalised-ML |
| mlem / osem | 0.787 | 0.319 | Poisson EM |
| cgls † | 0.770 | 0.266 | Krylov least-squares |
| grad / tikh | 0.670 | 0.647 | least-squares (BB) |

† **CGLS is past its peak at 40 iters** — as the fastest-converging method it
reaches its best (clean **0.839 at 10 iters**) far earlier and then semi-converges
*hard*, so a 40-iter snapshot understates it on clean data and buries it on noisy.
See [§4.5](#45-cgls--fastest-convergence-but-no-regularization) for its curve; do
not read its row here as its ceiling.

- Clean: **SIRT > TV > emission family > GRAD/TIKH** (the last under-converged at
  40 iters). Noisy: **GRAD/TIKH ≈ TV > SIRT ≫ emission** (multiplicative methods
  are noise-sensitive, and flooring 38 % of noisy rays to zero adds bias).
- Flooring is **not free** for the least-squares family — it discards ~30 % of
  rays, nudging SIRT clean 0.843 → 0.836. It *enables* the emission methods at a
  small cost to the algebraic ones.
- **Do NOT clip to a small positive ε instead of 0.** Mapping negatives to +ε
  (leaving positives) *destabilises* the emission methods — clean MLEM 0.787 →
  0.688, PML back to **NaN**, and ε-sensitivity is chaotic (MLEM: ε1e-4 → 0.787,
  1e-3 → 0.688, 1e-2 → 0.802). Cause: `b/(Ax)` blows up where the model predicts
  `Ax ≈ 0` but `b = ε > 0`. `sirt`/`tv`/`grad` are identical under ε vs 0
  (additive updates). **Floor to exactly 0.**
- `mlem == osem == pml == ospml` pairwise on this data (Poisson family, default
  subsets); `grad == tikh` (Tikhonov's L2 term is negligible at λ=0.001, reducing
  it to plain BB gradient descent).

### 4.4 Does any method "catch up" to SIRT with more iterations?

Sweep on floored clean data, Pearson vs dense ref:

| iters | SIRT | MLEM | GRAD |
|------:|-----:|-----:|-----:|
| 40 | **0.836** | 0.787 | 0.670 |
| 80 | 0.838 | 0.751 | 0.675 |
| 160 | 0.817 | 0.716 | 0.683 |
| 320 | 0.791 | 0.683 | 0.697 |
| 640 | 0.776 | 0.656 | 0.717 |

- **MLEM never reaches SIRT.** Its whole curve sits below SIRT's, and — like SIRT
  — it semi-converges, so *more iterations widen the gap*. Different objective
  (Poisson likelihood), lower/earlier plateau.
- **GRAD/TIKH climb very slowly.** At 640 iters (0.717) they are still below even
  SIRT@640. They minimise the same `‖Ax−b‖²` as SIRT, so they meet SIRT only at
  full least-squares convergence (thousands of iters) — by which point
  semi-convergence has pulled that shared quality *below* SIRT's early peak.
- **Takeaway:** SIRT@~40–80 is near-optimal here and cannot be beaten by throwing
  iterations at MLEM or GRAD.

### 4.5 CGLS — fastest convergence, but no regularization

CGLS solves the *same* `‖Ax−b‖²` normal equations as SIRT/GRAD, but as a Krylov
method (optimal step + conjugate directions) it converges in **far fewer
iterations**. That speed is a double-edged sword: with no regularizer it races to
the least-squares solution of the *sparse/noisy* data, so it peaks early then
semi-converges harder than SIRT.

Pearson vs the dense ref, per-slice mean, floored data:

**Clean:**

| iters | CGLS | SIRT |
|------:|-----:|-----:|
| 5 | 0.833 | 0.742 |
| 10 | **0.839** (CGLS peak) | 0.777 |
| 20 | 0.790 | 0.812 |
| 40 | 0.770 | **0.836** (SIRT peak) |

**Noisy:**

| iters | CGLS | SIRT |
|------:|-----:|-----:|
| 5 | **0.433** (CGLS peak) | 0.705 |
| 10 | 0.285 | 0.696 |
| 20 | 0.267 | 0.643 |
| 40 | 0.266 | 0.535 |

- **On clean data CGLS is the convergence-speed winner.** Its peak (0.839 @10)
  matches SIRT's eventual best (0.836 @40) in **4× fewer iterations** — and since
  a CGLS iteration costs the same as a SIRT one, that is a real **wall-time win**:
  `cgls --num_iter 10` = **2.01 s** for 0.839 vs `sirt --num_iter 40` = **5.87 s**
  for 0.836 (**≈2.9× faster** at equal-or-better quality). This is the payoff over
  FBP-seed warm-start ([§6](#6-warm-start-chaining---algorithm-ab)), which saved
  iterations but not wall-time on this GPU.
- **Early stopping is mandatory.** CGLS has *no* prior to resist semi-convergence,
  so past its peak it falls off faster than SIRT (clean 0.839→0.770 by 40).
- **On noisy data CGLS is the worst of the algebraic family.** It peaks at just
  5 iters (0.433 — already below SIRT@5's 0.705) and collapses to 0.266, because
  it fits the photon noise almost immediately. **For noisy/low-dose data use TV**
  (§5), whose prior is exactly what CGLS lacks.
- **Use CGLS for** well-posed / densely-sampled data where you want the
  least-squares answer fast, with a small iteration count and early stopping.

---

## 5. SIRT vs TV by iteration count

The two best methods on this data behave *fundamentally differently* with
iterations.

**Clean** (floored, λ=0.001):

| iters | SIRT | TV |
|------:|-----:|---:|
| 40 | **0.836** | 0.818 |
| 80 | 0.838 (SIRT peak) | 0.842 |
| 160 | 0.817 | **0.850** (TV peak) |
| 320 | 0.791 | 0.843 |
| 640 | 0.776 | 0.827 |

**Noisy** (floored):

| iters | SIRT | TV (λ=0.001) | TV (λ=0.1) |
|------:|-----:|-------------:|-----------:|
| 40 | 0.535 | 0.628 | **0.772** |
| 80 | 0.415 | 0.502 | 0.762 |
| 160 | 0.329 | 0.387 | 0.733 |
| 320 | 0.283 | 0.317 | 0.702 |
| 640 | 0.266 | 0.285 | 0.679 |

- **SIRT converges fast but has a delicate optimal stopping time.** It peaks early
  (~40–80 iters) then declines via semi-convergence — gently on clean
  (−0.062 by 640), **catastrophically on noisy** (0.535 → 0.266, −0.269). Early
  stopping is mandatory.
- **TV reaches a higher, later, flatter plateau** — its total-variation prior
  resists semi-convergence. Clean: peak 0.850 @160 (drop only −0.024). Crossover:
  SIRT wins ≤ ~60 iters (faster early), TV wins ≥ 80.
- **But TV's robustness depends on λ.** At too-low λ (0.001) TV crashes on noise
  like SIRT; at an adequate λ (0.1) it stays high (0.772 → 0.679). TV converts the
  brittle *stopping-time* problem into a more forgiving *λ-tuning* problem — once
  λ fits the noise level, iteration count is forgiving.

---

## 6. Warm-start chaining (`--algorithm a,b,…`)

- **FBP→SIRT** warm-start benefit is **view-count / conditioning dependent**:
  ~0.7 % at 450 views (well-sampled), ~5 % and roughly *halved iterations* at 65
  views (ill-posed). On this fast GPU at 800² it is **not a wall-time win** — the
  extra FBP stage costs more than the iterations it saves.
- **More stages is not automatically better.** A 3-stage `fbp,sirt,tv` does *not*
  reliably beat 2-stage on this data: clean → `fbp,sirt`; **noisy → `fbp,tv` or
  `tv`** (inserting a SIRT stage between FBP and TV *hurts* under noise — SIRT fits
  noise the later TV must undo). FBP-seeding helps SIRT but **not** TV (TV's prior
  dominates its initial guess).
- **Per-stage iterations:** `--algorithm fbp,sirt:30,tv:10` gives each iterative
  stage its own budget (see [`ALGORITHMS.md`](ALGORITHMS.md) and the README).

---

## 7. Practical selection guide

- **Analytic, quick look:** `fbp` (device-resident, fast). Pick the filter for the
  data — `hamming`/`cosine` for clean/well-sampled, `parzen` (default) for noisy.
  Prefer `fbp` over `gridrec` on CUDA (faster, higher fidelity, correct scale).
- **Iterative, attenuation data that can go negative:** `sirt` (fast, accurate,
  robust; stop at ~40–80 iters) or `tv` (higher ceiling and far more
  iteration-robust once `λ` fits the noise). Avoid `mlem`/`osem`/`pml`/`ospml`
  unless projections are guaranteed ≥ 0 (emission tomography, or floor to 0
  first). Never `art`/`bart` at high resolution.
- **Well-posed / dense data, want the least-squares answer fastest:** `cgls` with
  a **small** iteration count and early stopping (peak ~10 iters here, ≈2.9× less
  wall-time than SIRT for equal quality). No regularizer, so **not** for noisy or
  very sparse data — there it fits noise almost immediately (§4.5); use `tv` instead.
- **Noisy / low-dose:** `tv` with an adequately strong `λ` (do a short λ sweep);
  it resists the semi-convergence that collapses SIRT (and, worse, `cgls`).
- **Warm-start:** worth it for sparse/limited-angle data (`fbp,sirt`); match the
  chain to the data (`fbp,tv` for noise), and remember iteration savings ≠
  wall-time savings on a fast GPU.

> All results reproduced with the `cuda` feature build on real data; scripts and
> intermediate reconstructions are not committed. Re-run on your own sample before
> committing to a method — see the scope note at the top.
