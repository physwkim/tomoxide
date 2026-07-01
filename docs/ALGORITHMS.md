# tomoxide — Iterative Algorithm Guide

Practical companion to [`ARCHITECTURE.md` §3.2](ARCHITECTURE.md#32-iterative-project--backproject-loop),
which lists each iterative algorithm's parameters and tomopy upstream. This
document covers **what each method is good at and when to reach for it**,
**how to chain methods** (warm-start), plus the measured GPU behaviour of the
device-resident suite.

> **Scope of the claims.** The "strengths / use cases" below are the *design
> properties* of each algorithm (as defined by tomopy and the reconstruction
> literature). What tomoxide has *verified empirically* is: (a) every CUDA
> device-resident method reproduces the per-iteration CUDA path at Pearson
> r = 1.000000, and (b) the wall-clock numbers in [§5](#5-benchmark). A
> per-sample reconstruction-quality comparison across methods has **not** been
> run here — pick by the guidance below, then validate on your data.

---

## 1. Families at a glance

| Family | Members | Core update | Positivity |
|--------|---------|-------------|------------|
| Row-action (Kaczmarz) | `Art`, `Bart` | one ray (ART) / one block (BART) at a time | no |
| Least-squares / SIRT | `Sirt`, `Grad`, `Tikh`, `Cgls` | gradient / conjugate-gradient of `‖Ax − b‖²` (+ regularizer) | no |
| Statistical EM (Poisson) | `Mlem`, `Osem` | multiplicative `x ∘ Aᵀ(b ⊘ Ax) ⊘ Aᵀ(1)` | yes (by construction) |
| Penalized-ML (De Pierro) | `PmlQuad`, `PmlHybrid`, `OspmlQuad`, `OspmlHybrid` | EM step + 8-neighbour smoothness prior | yes |
| Total variation | `Tv` | Chambolle–Pock primal–dual on `‖Ax−b‖² + λ‖∇x‖₁` | no |

`A` = forward projector, `Aᵀ` = back-projector, `b` = measured sinogram.

---

## 2. Per-algorithm reference

### Row-action — `Art`, `Bart`

- **ART** projects the reconstruction onto each ray's hyperplane sequentially,
  updating immediately so the next ray sees the change.
  - *Strengths:* fast early convergence in few iterations; simple; needs no
    parameter; works when angles are few or unevenly sampled.
  - *Use for:* sparse / limited-angle data, quick previews.
- **BART** is block (ordered-subset) ART: within a subset every ray reads the
  same reconstruction, corrections accumulate, and the block is applied once.
  - *Strengths:* averages out per-ray noise sensitivity that plain ART has;
    more parallel.
  - *Use for:* the ART situations above when data is noisier.
- *Cost note:* in tomoxide ART/BART are **pure host** computation (geometry-only
  `RayProject` rows; the Kaczmarz sweep is inherently sequential). CUDA output is
  bit-identical to CPU, and there is essentially no GPU speed-up (~1×).

### Least-squares & SIRT — `Sirt`, `Grad`, `Tikh`, `Cgls`

- **SIRT** — parameter-free, convergent `x ← x + C∘Aᵀ(R∘(b − Ax))` with
  `R = 1/A(1)`, `C = 1/Aᵀ(1)`.
  - *Strengths:* very stable and robust, no tuning, gentle on noise.
  - *Use for:* the safe default baseline; noisy data; "just give me something
    reliable".
- **GRAD** — explicit least-squares gradient descent (with a Barzilai–Borwein
  step by default).
  - *Strengths:* flexible step control; explicit `‖Ax − b‖²` minimization.
  - *Caveat:* **unregularized GRAD + BB is run-to-run unstable** — there is no
    contraction, so the atomic-add nondeterminism in forward/back-projection
    drives divergent BB step sizes. Use a fixed step, or prefer TIKH, if you need
    reproducibility. (This is why the parity test fixes the GRAD step and uses BB
    only for TIKH.)
- **TIKH** — GRAD plus a Tikhonov ridge `‖x − prior‖²` (`reg_data` supplies the
  optional prior volume).
  - *Strengths:* the ridge contracts → **stable** (BB step is safe); can fold in
    a prior/reference volume.
  - *Use for:* when you have a prior volume, or when GRAD is too unstable.
- **CGLS** — conjugate-gradient least squares (the standard algorithm, Björck;
  its recurrence cross-checked against ASTRA's `CglsAlgorithm`, implemented
  independently — ASTRA is GPL-3.0, no ASTRA code is used): a Krylov solver of
  the same `‖Ax − b‖²` normal equations as GRAD/SIRT, but with the optimal step
  and conjugate search directions, so it reaches a given residual in **far fewer
  iterations** (≈4–30× vs SIRT here). Parameter-free; per-slice scalars; CUDA
  device-resident.
  - *Strengths:* fastest convergence of the least-squares family; no tuning.
  - *Caveat:* **no built-in regularization** — on ill-posed (sparse/noisy) data
    it semi-converges quickly, so it needs **early stopping** (peak quality often
    at ~10 iterations, then it fits sparse-view/noise artifacts). Being a Krylov
    method it is also more float-order sensitive than the contractive solvers
    (device vs host agree to ~1e-2 relative, Pearson ≈ 1, not the ~1e-8 of SIRT).
  - *Use for:* well-posed / densely-sampled data, or a fast few-iteration solve;
    pair with early stopping on sparse/noisy data. For noisy data prefer TV.

### Statistical EM — `Mlem`, `Osem`

- **MLEM** — multiplicative EM update; the maximum-likelihood estimator for
  Poisson-distributed counts.
  - *Strengths:* optimal noise model for low-dose / low-count data; output is
    **always non-negative** by construction.
  - *Use for:* low-dose CT, low photon counts, emission-style data.
- **OSEM** — MLEM restricted to ordered angle subsets (`num_block`); one outer
  iteration does `num_block` sub-updates.
  - *Strengths:* **~`num_block`× faster early convergence** — reaches MLEM
    quality in far fewer outer iterations. It is the practical replacement for
    MLEM.
  - *Use for:* whenever MLEM is appropriate but you want speed. Also the biggest
    GPU device-residency win (many subset back-projections per iteration).

### Penalized-ML — `PmlQuad`, `PmlHybrid`, `OspmlQuad`, `OspmlHybrid`

EM data term plus a De Pierro 8-neighbour smoothness prior (`reg_par[0]` =
strength; `reg = 0` reduces exactly to MLEM/OSEM). `Ospml*` add ordered subsets
on top (like OSEM); `Pml*` are the single-block case.

- **quad** (`PmlQuad`, `OspmlQuad`) — quadratic prior (`γ = 1`).
  - *Strengths:* smooths noise strongly and uniformly.
  - *Use for:* low-dose data where MLEM/OSEM is too noisy.
- **hybrid** (`PmlHybrid`, `OspmlHybrid`) — edge-preserving prior
  `γ = 1/(1 + |Δ|/δ)`, `δ = reg_par[1]`.
  - *Strengths:* smooths flat regions while **preserving edges** across large
    jumps.
  - *Use for:* low-dose + samples where sharp boundaries matter.
- *Cost note:* `Ospml*` show the **largest** device-residency speed-up in
  tomoxide (per-iteration stencil + subset back-projections are all kept on the
  GPU) — see [§5](#5-benchmark).

### Total variation — `Tv`

Chambolle–Pock primal–dual on `‖Ax − b‖² + λ‖∇x‖₁` (`λ = reg_par[0]`).

- *Strengths:* piecewise-smooth reconstruction with **sharp edges**; strong on
  sparse-view / limited-angle data; suppresses streak/staircase artefacts in
  homogeneous regions.
- *Use for:* few-angle acquisitions, samples with large uniform regions and
  distinct boundaries (materials, contrast-enhanced).

---

## 3. Selection guide

| If you… | Reach for |
|---------|-----------|
| are unsure / want a safe, tuning-free baseline | **SIRT** |
| have low-dose / low-count (Poisson) data | **OSEM** (MLEM quality, faster) |
| have low-dose data that comes out too noisy | **OSPML-quad**, or **OSPML-hybrid** to keep edges |
| have few / limited angles | **TV** (or ART/BART) |
| have a prior volume, or GRAD is unstable | **TIKH** |
| want maximum GPU throughput | **OSEM / OSPML** (largest device-residency gain) |
| need reproducible least-squares | **SIRT** or **TIKH** (avoid unregularized GRAD+BB) |
| want faster convergence, or a rough pass then a polish | **chain** an analytic seed into an iterative method (see [§4](#4-chaining-warm-start)) |

---

## 4. Chaining (warm-start)

An iterative reconstruction does not have to start from a blank (or flat)
volume — it can be **warm-started** from another reconstruction's output through
`ReconParams.init`. This is the tomography analogue of ptychography's engine
chaining (DM/RAAR → ML): run a fast method to get a good global estimate, then
hand it to a second method to refine.

Two motivations, both about *quality per iteration*, not escaping local minima
(tomographic least-squares is largely convex):

- **Convergence speed.** An analytic seed already carries the low-frequency
  structure that an iterative-from-zero solver spends many iterations building up,
  so the iterative pass starts far closer to the answer.
- **Regularizer polish.** A cheap rough reconstruction followed by a
  regularizing method (TV, PML) cleans up noise/streaks the first pass left.

### Producers vs. consumers

- **Seed producers** — *any* full-volume reconstruction. The analytic / direct
  methods **`Fbp`, `Gridrec`, `Fourierrec`, `Lprec`, `Linerec`** (whichever your
  backend provides) are the natural first stage: one fast pass, no tuning. The
  output of *any* iterative method can equally seed the next.
- **Seed consumers** (`ReconParams.init`) — **every iterative method except the
  row-action pair `Art`/`Bart`**, which sweep single rays and have no
  whole-volume iterate to seed (they *reject* a non-`None` init rather than
  silently dropping it). The analytic methods above **ignore** `init` — they
  produce, they do not consume.

So the "FBP →" in the recipes below is shorthand for *any* analytic seed:
**`Linerec`, `Fourierrec`, `Lprec`, and `Gridrec` are equally valid first
stages** — FBP is simply the one measured here.

### Recipes and measured effect

Under a **fixed total iteration budget** (so a chain is a fair comparison against
a single method run for the same number of iterations), on a Shepp–Logan phantom
at 64 sparse views with 3 % Gaussian noise, CPU backend, NRMSE over a centred
disk (`tests/warmstart_experiment.rs`):

| Chain | NRMSE | vs. the single method (same budget) |
|-------|------:|-------------------------------------|
| `Fbp → Sirt(40)`        | **0.249** | SIRT(40) from scratch 0.356 |
| `Fbp → Tv(40)`          | **0.268** | TV(40) from scratch 0.443 |
| `Osem(10) → Mlem(30)`   | **0.189** | MLEM(40) 0.214, OSEM(40) 0.215 — best overall |
| `Sirt(20) → Tv(20)`     | 0.393 | SIRT(40) 0.356 — **worse**: chaining into a regularizer is λ-sensitive |

Convergence: an FBP-warm-started SIRT reaches the quality of a 40-iteration
scratch SIRT in **≈10 iterations** (≈4× fewer). Full tables in the experiment.

**Reading it:** analytic → iterative and OSEM → MLEM are clear wins; chaining into
a regularizer (SIRT → TV) only helps if that regularizer's parameters are tuned —
a poorly-chosen `λ` can undo the head start. This is a single phantom / noise /
geometry; treat the numbers as indicative and validate on your data.

### Backend note (convention)

The experiment runs on the **CPU backend**, but since the cross-backend
convention unification the analytic and iterative solvers share **one
orientation and scale convention on every backend** (CPU/wgpu/CUDA all follow
tomopy — see `docs/ARCHITECTURE.md §4.1`). So an analytic seed drops straight
into an iterative solver on CUDA too: a CUDA `Fbp → Sirt` warm-start is
convention-matched out of the box, up to the deterministic ~1.6% `_wint`-ramp
shape residual (not a scale). The only exception is **laminography**, whose CUDA
and CPU paths are different algorithms and are not convention-comparable (do not
warm-start one from the other). Warm-start is honoured on both the host solvers
and the CUDA device-resident path (the seed is uploaded once), verified by the
device-resident split-SIRT test in `tests/cuda_forward_project.rs`.

---

## 5. Benchmark

Device-resident CUDA (volume/sinogram/weights stay on the GPU across all
iterations) vs. the per-iteration CUDA path (host round-trip every iteration)
vs. CPU. RTX 5000 Ada, 720 projections, 30 iterations, Shepp–Logan (non-negative).
Measured by `tests/cuda_iterative_bench.rs`.

**512² × 8 slices** — *device-residency gain* (per-iter ÷ device-resident) and
*CPU speed-up* (CPU ÷ device-resident):

| Method | device-residency gain | vs CPU |
|--------|----------------------:|-------:|
| SIRT              | 1.60× | 51× |
| MLEM              | 1.58× | 52× |
| OSEM (num_block 8) | 3.77× | 67× |
| OSPML-quad (num_block 8) | **11.39×** | **95×** |
| PML-hybrid        | 3.12× | 51× |
| GRAD              | 1.29× | 53× |
| TIKH              | 1.73× | 52× |
| TV                | 1.79× | 58× |

**1024² × 4 slices** — device-residency gain (CPU omitted; a full CPU pass over
all eight methods at this size takes hours):

| Method | device-residency gain |
|--------|----------------------:|
| SIRT | 1.43× · MLEM 1.17× · OSEM 2.50× · **OSPML-quad 10.83×** · PML-hybrid 2.25× · GRAD 1.26× · TIKH 1.29× · TV 1.41× |

**Reading it:** the device-residency gain scales with how many host↔device
transfers the per-iteration path performs each iteration. OSPML/OSEM do
`num_block` subset back-projections per iteration, so keeping them resident wins
most; single-operation GRAD wins least. ART/BART are pure host (CUDA == CPU,
~1×) and are not in this table.

All device-resident methods reproduce the per-iteration CUDA output at
Pearson r = 1.000000 (NRMSE: EM ~1e-3…3e-6, OSPML/PML 3e-6…4e-5,
GRAD/TIKH/TV ~6–7e-8), verified in `tests/cuda_forward_project.rs`.
