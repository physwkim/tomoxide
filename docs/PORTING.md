# tomoxide — Porting map (tomopy + tomocupy → tomoxide)

Provenance and status for every ported subsystem. Each module stub in the
source tree carries a doc comment pointing back at the upstream `file:line`
listed here. Status legend:

- `stub` — signature/dispatch exists, body returns `Error::NotImplemented`.
- `partial` — a real but incomplete CPU implementation exists.
- `done` — ported and tested against the reference.

Upstream roots (this machine):
- tomopy:   `/Users/stevek/codes/tomopy`
- tomocupy: `/Users/stevek/codes/tomocupy`

---

## A. Reconstruction — analytic (`tomoxide-recon`)

| tomoxide                     | Upstream                                                    | Backend  | Status |
|------------------------------|------------------------------------------------------------|----------|--------|
| `recon::fbp`                 | tomopy `libtomo/recon/fbp.c`; `extern/recon.py:238`         | CPU, wgpu | CPU done (golden: recovers phantom from tomopy sinogram r=0.87; tomopy's own `fbp` is a weak reference). wgpu back-projection step done — `FilteredBackproject` WGSL (backproject.wgsl): one thread/voxel, linear-interp sum over angles; the FBP π/nproj dθ weight is a caller-passed gain (analytic paths π/nproj, iterative pure adjoint 1.0). Per-angle (cosθ,sinθ) + per-row center host-computed so the inclusion boundary is bit-identical to CPU; tolerance parity (rtol 1e-4). GPU FBP filter apply now done too (see `recon::filter`), so the full filter→back-project FBP path runs on-device |
| `recon::gridrec`             | tomopy `libtomo/gridrec/gridrec.c:195`; `extern/gridrec.py:64` | CPU, wgpu | CPU done (DFI/Kaiser-Bessel not PSWF; golden: vs tomopy gridrec r=0.98; Fourier recenter shift uses signed freq — correct at sub-pixel centers). Runs on wgpu for free: gridrec needs only the `Fft` capability and every transform length is power-of-two (`pad=(2·ncols).next_power_of_two()`, grid `m=pad`), so `recon::recon(Gridrec, &WgpuBackend)` dispatches the radial 1-D + 2-D FFTs to the GPU (Kaiser-Bessel gridding/deapodization stays host-side, shared with CPU). Verified GPU↔CPU NRMSE 3.4e-7 (analytic_gpu_parity test) |
| `recon::fourierrec`          | tomocupy `reconstruction/fourierrec.py:46`; `cuda/cfunc_fourierrec.cu`, `include/cfunc_fourierrec.cuh:10` | CPU, wgpu, CUDA | CPU + wgpu done. Faithful port of the Gaussian-USFFT gridding (`kernels_fourierrec.cuh`): centred 1-D FFT (ifftshiftc modulation) → gather onto a `(2n+2m)²` grid with a separable Gaussian kernel (`m=4`, `mu` from `eps=1e-3`) → periodic border `wrap` → centred 2-D inverse FFT → Gaussian deapodization (`divphi`) + central crop + unit-disk mask. Takes pre-filtered data (tomocupy `fbp_filter_center` → `backprojection`), so the dispatcher runs `FbpFilter::apply` first; needs only the `Fft` capability, so it composes onto wgpu for free (n=128 pow2 path; n=96 exercises Bluestein 1-D + separable-Bluestein 2-D). Two tomocupy geometry conventions adapted to tomoxide's: gather y uses `+sin` (not `-sin`) and the crop is symmetric (no `+1` row bias) — each verified against the phantom. No CUDA golden offline; verified via fourierrec↔phantom r=0.9713 (== gridrec 0.9712) and fourierrec↔gridrec r=1.0000, GPU↔CPU NRMSE 3.5e-7. **CUDA (M4, `cuda` feature):** the vendored `cfunc_fourierrec.cu` (cuFFT) runs on the GPU via the new `FourierReconstruct` capability — slice-pairs packed into complex, kernel, de-interleave; FBP filter on the CPU; even slice count required. CUDA↔CPU fourierrec r=0.9997, ↔phantom 0.971 (`cuda_fbp_parity.rs`) |
| `recon::lprec`               | tomocupy `reconstruction/lprec.py:292`; `cuda/cfunc_lprec.cu`, `include/cfunc_lprec.cuh:9` | CPU, wgpu | CPU + wgpu done. Faithful port of the log-polar (Andersson–Carlsson–Nikitin) method — replaces the prior silent FBP aliasing. Precompute (`create_gl`/`create_adj`/`fzeta_loop_weights_adj`/`splineB3`/`osg`/`getparameters`): the `Nspan=3` overlapping `2β` spans (`β=π/3`), the log-polar grids (`ntheta=2^round(log2(nproj))`, `nrho=2·2^round(log2(n))`), the zeta convolution kernel `fZ` (discretization weights + the magic const, brought to standard FFT order by the 2-D fftshift of `create_adj`, B3-spline-deapodized, Hermitian-extended for C2C emulation of cuFFT's R2C/C2R), and the `C2lp`/`lp2p`/`lp2p_w` coordinate index sets. Runtime (`cfunc_lprec.cu`): cubic-B-spline prefilter (causal/anticausal recursion, pole `√3−2`) → per-span 4-tap cubic-B-spline gather polar→log-polar (the CPU equivalent of the GPU "cubic via 2 linear texture fetches", wrap addressing) → 2-D FFT × `fZ` × inverse → cubic gather log-polar→Cartesian. Takes pre-filtered data (`fbp_filter_center` → `LpRec.backprojection`), so the dispatcher runs `FbpFilter::apply` first; needs only the `Fft` capability (all pow2 at n=128), so it composes onto wgpu for free. One tomocupy geometry convention adapted: the output row coordinate uses the value-negation of tomocupy's `x2*(-1)` (`x2 = lin[i]+1/N`), the true y-flip that co-registers with tomoxide's `+sin` projector (index-mirroring instead mis-registers by ≈1 px) — verified against the phantom. No CUDA golden offline; verified via lprec↔phantom r=0.9348 and lprec↔gridrec r=0.9676 (genuinely different inversions, so ~0.97 not ~1.0), GPU↔CPU NRMSE 3.0e-7 |
| `recon::linerec`             | tomocupy `reconstruction/linerec.py:47`; `cuda/cfunc_linerec.cu`, `include/cfunc_linerec.cuh:9` | CPU, **CUDA** | Routed through the FBP filter+back-project path, which is faithful: `cfunc_linerec`'s `backprojection_ker` with the standard tomography tilt (φ=π/2) reduces to parallel-beam back-projection with linear interpolation (upstream scales it `4/nproj`; the vendored copy takes the gain from the caller, which passes the `π/nproj` dθ weight), i.e. a filtered back-projection. Laminographic φ≠π/2 (the `_lamino` kernel) is a separate `recon::lamino` concern. **CUDA (M4, `cuda` feature):** the vendored `cfunc_linerec.cu` runs on the GPU as `CudaBackend`'s `FilteredBackproject` (filter reused from the CPU definition); CUDA↔CPU back-projection Pearson = 1.0, and since the Phase 1/2 convention unification the CUDA output matches the CPU handedness and scale as well — upstream's `(n−1−ty)` y-flip and baked-in `4/nproj` are both gone (`docs/ARCHITECTURE.md` §4.1). Needs ≥2 slices (vertical interpolation) |
| `recon::filter` (FBP filter) | tomocupy `reconstruction/fbp_filter.py:46`; `cuda/cfunc_filter.cu`; tomopy filter in `fbp.c` | all | CPU + wgpu done. wgpu `Fft` primitive: 1-D handles **any length** — power-of-two runs the direct radix-2 kernel (fft.wgsl bit-reversal + per-stage butterflies), other lengths run the Bluestein chirp-z transform (bluestein.wgsl spectral multiply + 3 radix-2 FFTs of `m=next_pow2(2n−1)`; host chirp with `j² mod 2n` reduction, rel error ≈1e-7 vs rustfft); 2-D also handles **any dims** — power-of-two runs the on-device transpose fast path, other dims a separable pair of 1-D passes (radix-2 or Bluestein per axis) with a host transpose between (per-axis 1/cols·1/rows compose to 1/(rows·cols)), forward rel ≈1e-7 vs rustfft. Tolerance parity vs rustfft. `FbpFilter::apply` on GPU: each detector lane centred in `pad=filter.len()` (`= (4·n).next_power_of_two()`, tomocupy's `ne`) and edge-replicate-padded on both borders, then forward FFT → ×filter multiply (fbp_filter.wgsl `apply_filter`, complex×real broadcast) → inverse FFT, all one serialized submission chain, crop the centred n_cols window ×(1/pad); power-of-two pad only (else error→CPU fallback); tolerance parity (2e-3). `make_filter` is shared host arithmetic (`tomoxide_core::backend::make_fbp_filter`) so CPU/GPU build the identical kernel. **Rotation-center handling — faithful tomocupy `fbp_filter_center`.** `apply` folds the center into the shared filter as a per-row Fourier phase `exp(-2πi·f_k·(n/2−center)/pad)` with the **signed** frequency `f_k` (band-limited sub-pixel shift that lands the rotation axis on the detector midpoint `n/2`), exactly as tomocupy's `fbp_filter_center` (backproj_functions.py:103). After this pass every analytic back-projector and Fourier grid reconstructs against a center=`n/2` geometry: fbp/linerec back-project at `n/2` (a `center=ncols/2` geometry), and fourierrec/lprec are center-agnostic (their own in-grid recenter removed). The rotation center is now owned in this **one** place. Out of scope: gridrec keeps its own integrated filter+recenter (it never calls `apply`), and the iterative back-projectors keep `geom.center` in the matched projector/adjoint pair (their data is not filtered). The signed frequency is mandatory — a raw index is not Hermitian-symmetric and collapses a half-integer center (the trap `gridrec` documents). δ=0 at the default center `n/2` makes the phase unity, so the center-aligned goldens are unchanged; the off-center path is verified by `center_in_filter_parity.rs` (fbp/fourierrec/lprec recover an axis-at-66.5 sub-pixel phantom and match the centered recon) plus the e2e pipeline (find_center 63.5 → FBP r=0.87). Padding now matches tomocupy fully: the lane is centred in an `ne = (4·n).next_power_of_two()` buffer and **edge-replicated** on both borders (`tmp[:pad]=data[:1]`, `tmp[pad+n:]=data[-1:]`) so the long-tailed ramp does not ring against a hard zero step, then cropped back to the centred `[pad:pad+n]` window — `pad = ne//2 − n//2` (`(4·n).next_power_of_two()` equals tomocupy's float32 `ne=4·n` for every power-of-two width, and matches its float16 pow2-rounding otherwise, keeping the wgpu radix-2 FFT usable at any width) |
| `recon::lamino` (USFFT)      | tomocupy `reconstruction/lamfourierrec.py`, `backproj_lamfourier_parallel.py`; `cuda/cfunc_usfft1d.cu`, `cfunc_usfft2d.cu`, `cfunc_fft2d.cu` + `include/kernels_*.cu` | CPU, wgpu | CPU + wgpu done. Faithful port of `LamFourierRec` (arxiv 2401.11101) — the Fourier-based adjoint Radon transform for a tilted axis. Intrinsically 3-D (every tilted projection touches every voxel), so it has its own entry point `recon::lamino::lamino` (`[nproj,nz,n]` proj → `[rh,n,n]` volume), not the per-slice `recon` dispatch. Three chained Gaussian-gridding USFFT operators: `fft2d_fwd` (centred 2-D FFT of each ramp-filtered projection) → `usfft2d_adj` (scatter the `(θ,kx,ky)` slices into the in-plane `(x,y)` frequency volume along the laminographic map `take_x`: `x=ku·cosθ+kv·sinθ·cosφ`, `y=ku·sinθ−kv·cosθ·cosφ`) → `usfft1d_adj` (transform along the tilt-z axis to the real volume), with `φ=π/2+lamino_angle·π/180`. Each op (take_x positions, `exp(−π²/μ·w²)` Gaussian gather, `wrap` periodic borders, `exp(−μ(i−n/2)²)` deapodize) is ported from the kernels; a matched forward projector `lamino_project` (exact transpose chain `usfft1d_fwd→usfft2d_fwd→fft2d_inv`) supports self-consistent testing. Two pure-optimization deviations: drop the deth/ntheta/n1 chunking + double-buffering, and drop the R2C Hermitian-half (detw/2+1 at fft2d, deth/2+1 at the usfft2d→usfft1d interface, flip-block bookkeeping in the gathers) — carry full complex spectra instead. Gridding geometry and per-op FFT normalization (cuFFT-style unnormalized inverse, recovered by multiplying back the `1/N` that `Fft` divides out) are identical, so the result is numerically equivalent for real input while every operator is a clean transpose. Needs only the `Fft` capability, so it composes onto wgpu for free (n,rh,nz pow2 → radix-2). No CUDA golden offline; verified via `fft2d_inv(fft2d_fwd)==id` (max\|Δ\|<1e-4), usfft1d/usfft2d fwd/adj adjoint dot-product (rel<1e-4), 3-D two-sphere phantom forward-project→reconstruct Pearson 0.9520 (confirms the take_x geometry end to end), GPU↔CPU NRMSE 1.5e-7 |

### C-extern signatures to match (tomopy `include/libtomo/recon.h`)

```c
void fbp     (const float* data,int dy,int dt,int dx,const float* center,const float* theta,
              float* recon,int ngridx,int ngridy,const char* name,const float* filter_par);
void gridrec (const float* data,int dy,int dt,int dx,const float* center,const float* theta,
              float* recon,int ngridx,int ngridy,const char* fname,const float* filter_par);
```

### CUDA class boundary to match (tomocupy SWIG `cfunc_*`)

```c
cfunc_fourierrec(size_t nproj,size_t nz,size_t n,size_t theta_ptr);
  void backprojection(size_t f_ptr,size_t g_ptr,size_t stream_ptr);
cfunc_lprec(int nproj,int nz,int n,int ntheta,int nrho);
  void setgrids(...); void backprojection(size_t f,size_t g,size_t stream);
cfunc_linerec(size_t nproj,size_t nz,size_t n,size_t ncproj,size_t ncz);
  void backprojection(size_t f,size_t g,size_t theta,float phi,int sz,size_t stream);
  void backprojection_try(...); void backprojection_try_lamino(...);
cfunc_filter(size_t nproj,size_t nz,size_t n);
  void filter(size_t g_ptr,size_t w_ptr,size_t stream_ptr);
```

---

## B. Reconstruction — iterative (`tomoxide-recon::iterative`)

Every entry is `void NAME(const float* data,int dy,int dt,int dx,const float* center,
const float* theta,float* recon,int ngridx,int ngridy, …)` in tomopy
`include/libtomo/recon.h` / `libtomo/recon/NAME.c`.

| tomoxide `Algorithm::` | extra params                        | Upstream `.c`        | Status |
|------------------------|-------------------------------------|----------------------|--------|
| `Art`                  | `num_iter`                          | `recon/art.c`        | CPU done (row-action Kaczmarz via `RayProject`; r=0.99) |
| `Bart`                 | `num_iter,num_block,ind_block`      | `recon/bart.c`       | CPU done (ordered-subset SART via `RayProject`; r=0.98) |
| `Sirt`                 | `num_iter`                          | `recon/sirt.c` (+`accel/cxx/sirt.cc`) | partial (R/C-weighted) |
| `Mlem`                 | `num_iter`                          | `accel/cxx/mlem.cc`  | CPU done (r=0.99) |
| `Osem`                 | `num_iter,num_block,ind_block`      | `recon/osem.c`       | CPU done (MLEM over ordered subsets; r=0.99, num_block=1 ≡ MLEM) |
| `OspmlHybrid`          | `num_iter,reg_par,num_block,ind_block` | `recon/ospml_hybrid.c` | CPU done (edge-preserving prior; no reg_par[1] ≡ ospml_quad) |
| `OspmlQuad`            | `num_iter,reg_par,num_block,ind_block` | `recon/ospml_quad.c`   | CPU done (De Pierro quad prior; reg=0 ≡ OSEM) |
| `PmlHybrid`            | `num_iter,reg_par`                  | `recon/ospml_hybrid.c` (num_block=1) | CPU done (reg=0 ≡ MLEM) |
| `PmlQuad`              | `num_iter,reg_par`                  | `recon/ospml_quad.c` (num_block=1) | CPU done (reg=0 ≡ MLEM) |
| `Tv`                   | `num_iter,reg_par`                  | `recon/tv.c`         | CPU done (Chambolle–Pock TV; reg_par[0]=strength, c=0.35; r=0.95, larger λ smooths) |
| `Grad`                 | `num_iter,reg_par`                  | `recon/grad.c`       | CPU done (LS gradient descent; BB step reg_par[0]<0 → r=0.99; unit fixed step diverges for this projector, see note) |
| `Tikh`                 | `num_iter,reg_data,reg_par`         | `recon/tikh.c`       | CPU done (grad + ridge term 2·reg_par[1]·(x−reg_data); no reg_par[1] ≡ grad; shares grad's core) |
| `Vector{,2,3}`         | `num_iter,axis…`                    | `recon/vector.c`     | done (separate `recon::vector` API, not the scalar dispatch; bit-exact Δ=0 vs tomopy 1.15 — line-for-line C port incl. `calc_quadrant` int trick & mixed f32/f64; vector2/3 need a cube `dy==dx`, else tomopy corrupts memory) |

Forward model shared by all: tomopy `libtomo/recon/project.c`
(`void project(const float* obj,…,float* data,…,const float* center,const float* theta)`)
→ `tomoxide-recon::project` / `ForwardProject` capability. CPU done; wgpu
done — WGSL `project.wgsl` (one thread per `(row, angle)`, race-free scatter:
each owns a disjoint detector-column span, visits pixels in CPU order so the
per-column accumulation matches; tolerance parity rtol 1e-4). Exact linear-
interp adjoint of the wgpu back-projector, sharing the host-side
`(cosθ,sinθ)`/per-row-center build.

Because the whole `ForwardProject`+`FilteredBackproject` family dispatches
through `&dyn Backend`, every iterative method built on those projectors
(SIRT, MLEM, OSEM, PML/OSPML quad & hybrid, grad, tikh, tv) runs on the GPU
unchanged when a `WgpuBackend` is passed — the solver, its R/C weight maps,
and the residual updates all project on the GPU. Verified by the SIRT GPU↔CPU
parity test (`iterative_gpu_parity`: 100-iter loop, GPU↔CPU correlation
1.00000, NRMSE 1.8e-4, GPU recon as accurate as CPU). **Exception:** `Art`/
`Bart` use the host-side sparse `RayProject` (single-ray Kaczmarz rows), which
is not a GPU kernel, so they stay CPU-only.

---

## C. Center finding (`tomoxide-recon::center`)

| tomoxide                | Upstream                                  | Status |
|-------------------------|-------------------------------------------|--------|
| `center::find_center`   | tomopy `recon/rotation.py:82` (entropy)   | CPU ✓ — recovery (projector-coupled via gridrec): true axis ±0.5 px, tomopy `find_center` ±1 px |
| `center::find_center_vo`| tomopy `recon/rotation.py:205` (Vo coarse+fine) | CPU ✓ — tomopy parity Δ=0 |
| `center::find_center_pc`| tomopy `recon/rotation.py:391` (phase corr; skimage `phase_cross_correlation`) | CPU ✓ — tomopy parity Δ=0; `rotc_guess` pre-alignment ported via line-faithful `scipy.ndimage.shift` (order-3 spline, `mode='constant'`), isolated shift Δ=0 vs scipy 1.17.1 |
| `center::write_center`  | tomopy `recon/rotation.py:438`            | CPU ✓ — center enumeration Δ=0 vs numpy `arange`; recon content gridrec-family (KB kernel + ramp, not PSWF+parzen), self-consistent vs `recon(Gridrec)` (Δ=0). Returns `(centers, stack)`; I/O-free core (persist `{center:.2f}.tiff` via tomoxide-io) |
| `center::find_center_sift` | tomocupy `find_center.py:99`           | done (behind `sift-center` feature; pure-Rust SIFT via the `lowe-sift` crate, exact-NN + Lowe ratio, `n/2 − shift_x/2`; uint8 normalization bit-exact vs numpy; independent SIFT implementation → shifts ≈ 0.034 px / center ≈ 0.008 px vs the cv2 golden) |
| `center::find_center_ai`| tomocupy `find_center.py:86` (+`ai/inference.py`) | stub |

---

## D. Preprocessing (`tomoxide-prep`)

| tomoxide                          | Upstream                                          | Backend | Status  |
|-----------------------------------|---------------------------------------------------|---------|---------|
| `normalize::normalize`            | tomopy `prep/normalize.py:98`                     | CPU     | partial |
| `normalize::normalize_bg`         | tomopy `prep/normalize.py:207`; `libtomo/prep/prep.c` (`normalize_bg`) | CPU | CPU ✓ — tomopy parity (bit-exact, Δ=0) for `air=1` & `air=4`. Per-row air baseline (left/right boundary means) lerp'd across the column axis, divide; f32 in upstream order, `f32::mul_add` for the clang-contracted `air_left + air_slope·j` |
| `normalize::normalize_nf`         | tomopy `prep/normalize.py:245`                    | CPU     | CPU ✓ — tomopy parity (bit-exact, Δ=0), `averaging='mean'`. Per-group flat median, dark mean, `(proj−dark)/max(flat−dark,1e-6)` + cutoff; half-to-even group boundaries. `averaging='median'` done too (per-pixel dark median; upstream's bogus `np.median(dtype=)` monkeypatched away in the golden, Δ=0 for odd dark counts) |
| `normalize::normalize_roi`        | tomopy `prep/normalize.py:168`                    | CPU     | CPU ✓ — tomopy parity (bit-exact, Δ=0). Per-projection divide by the mean `bg` of `proj[r0:r2, r1:r3]` (skip when `bg=0`). ROI mean reproduces numpy's f32 pairwise summation (new `pairwise_sum_f32`: 8-acc base ≤128, recursive split otherwise) → exact divisor; a sequential sum diverges ~1 ULP. `prep::normalize_roi(data, [r0,r1,r2,r3])` |
| `normalize::minus_log`            | tomopy `prep/normalize.py:72`; tomocupy `proc_functions.minus_log` | all | CPU ✓ (Δ=0) + wgpu ✓ — `-ln(max(x,1e-6))` with non-finite scrub. WGSL `minus_log` (elementwise.wgsl), layout-independent. GPU `log` ≠ libm `ln` by a few ULP → parity is tolerance (CpuBackend ref, rtol 1e-5), not bit-exact |
| `normalize::darkflat`             | tomocupy `proc_functions.darkflat_correction:55`  | all     | CPU ✓ (Δ=0) + wgpu ✓ — `(data−dark)/denom`, `denom=max(flat−dark,1e-6)`. Frame-averaging + zero-guard host-side (order-sensitive); per-element broadcast on the GPU in projection layout (i=proj·plane+row·cols+col), restores caller layout. WGSL `darkflat`. Tolerance parity (rtol 1e-5) in projection & sinogram layout |
| `stripe::remove_stripe_fw`        | tomopy `prep/stripe.py:88`; tomocupy `remove_stripe.remove_stripe_fw` | CPU | CPU ✓ — tomopy parity (≈f32 floor, max rel Δ≈1.2e-6). db5 dwt2/idwt2 hand-ported (no new dep) in `wavelet.rs`, pywt-validated; float32-forward/f64-damp+inverse dtype flow. Damping uses a self-contained `O(n log n)` FFT (radix-2 + Bluestein, arbitrary length, no FFT dep) in `fft.rs` |
| `stripe::remove_stripe_ti`        | tomopy `prep/stripe.py:179` (Titarenko/Miqueles) | CPU | CPU ✓ — tomopy parity (≈f32 floor, max rel Δ≈5.2e-7); both `nblock=0` and the `_ringb` block path (`nblock>0`, incl. its `np.ones` tail fill — the NaN guard is a harmless no-op on modern numpy, not an error) |
| `stripe::remove_stripe_sf`        | tomopy `prep/stripe.py:333`; `libtomo/prep/stripe.c` (`remove_stripe_sf`) | CPU | CPU ✓ — tomopy parity (bit-exact) |
| `stripe::remove_stripe_based_sorting` | tomopy `prep/stripe.py:363` (Vo alg. 3)       | CPU     | CPU ✓ — tomopy parity (bit-exact, Δ=0) for `dim=1` & `dim=2`. `_rs_sort` is a pure rank-filter selection on f32; `StripeMethod::VoSort { size, dim }`, reuses the `rs_sort` scaffold (now smoother-pluggable) shared with `VoAll` |
| `stripe::remove_stripe_based_filtering` | tomopy `prep/stripe.py:437` (Vo alg. 2)     | CPU     | CPU ✓ — tomopy parity to the f32 floor (measured Δ=0) for `dim=1` & `dim=2`. `_rs_filter` Gaussian-Fourier low-pass + `_rs_sort` + high-pass residual; `StripeMethod::VoFilter { sigma, size, dim }`, reuses the f64 column FFT (`fft.rs`) and the `rs_sort`/`median_filter_2d` scaffolds from `VoSort` |
| `stripe::remove_stripe_based_fitting` | tomopy `prep/stripe.py:520` (Vo alg. 1)       | CPU     | CPU ✓ — tomopy parity ≈f32 floor (max rel ≈2e-7). `StripeMethod::VoFit { order, sigma }`. Savitzky-Golay weights via scaled normal equations (reproduce scipy SVD `lstsq` to f64 floor, no linalg dep); 2-D Fourier smoothing via new `fft::filter_real_2d` (separable, unit-tested vs naive 2-D DFT) |
| `stripe::remove_large_stripe`     | tomopy `prep/stripe.py:653` (Vo alg. 5)       | CPU     | CPU ✓ — tomopy parity: `norm=false` bit-exact (Δ=0, pure selection/copy), `norm=true` ≈f32 floor (max rel ≤1e-5, unmasked-column factor division). `StripeMethod::VoLarge { snr, size, drop_ratio, norm }`, reuses the `rs_large` helper shared with `VoAll` (unchanged) |
| `stripe::remove_dead_stripe`      | tomopy `prep/stripe.py:762` (Vo alg. 6)       | CPU     | CPU ✓ — tomopy parity ≈f32 floor (max rel ≤1e-5; bilinear dead-column fill is arithmetic). `StripeMethod::VoDead { snr, size, norm }`; `norm` gates the residual `rs_large` pass (threaded through the `rs_dead` helper shared with `VoAll`, which now passes `norm=true` and stays bit-identical) |
| `stripe::remove_all_stripe`       | tomopy `prep/stripe.py:843` (Vo alg. 3+5+6); tomocupy `remove_stripe.remove_all_stripe` | CPU | CPU ✓ — tomopy parity (≈f32 floor, max rel Δ≈5.8e-7) |
| `stripe3d::stripes_detect3d`      | tomopy `prep/stripe.py:984`; `libtomo/prep/stripes_detect3d.c::stripesdetect3d_main_float` | CPU | CPU ✓ — tomopy parity **bit-exact (Δ=0)** for size=10/radius=3 & size=5/radius=2 (pure f32, no FFT). New `stripe3d` module; `prep::stripes_detect3d(&Tomo, size, radius) -> Array3<f32>` weights, not a `StripeMethod` (returns weights, does not mutate). 6-stencil mean smoothing → detX forward gradient (step 2) → parallel/orthogonal mean-ratio map → vertical median filter |
| `stripe3d::stripes_mask3d`        | tomopy `prep/stripe.py:1058`; `libtomo/prep/stripes_detect3d.c::stripesmask3d_main_float` | CPU | CPU ✓ — tomopy parity **bit-exact (Δ=0, bool)** for the defaults and a looser param set (pure int/bool, one f32 threshold compare). `prep::stripes_mask3d(&Array3<f32> weights, threshold, min_stripe_length, min_stripe_depth, min_stripe_width, sensitivity_perc) -> Array3<bool>`. Threshold → depth then along-angle consistency prune → drop short stripes → iterative merge; each pass reads the prior full result and writes a fresh buffer (the C `mask`/`Output` split) |
| `phase::retrieve_phase` (Paganin) | tomopy `prep/phase.py:80`; tomocupy `retrieve_phase.paganin_filter:59` | CPU | ✓ — tomopy parity (max rel Δ≈2.4e-7) |
| `phase::retrieve_phase` (Gpaganin) | tomocupy `retrieve_phase.paganin_filter:59` (`method='Gpaganin'`, `_paganin_filter_factorG:215`) | CPU | ✓ — tomocupy parity (max rel Δ≈4.9e-7). Grid/filter in f32 to mirror cupy single precision (ill-conditioned, `scale≈1.2e3`); golden via scipy.fft single-precision transcription |
| `phase::retrieve_phase` (farago) | tomocupy `retrieve_phase.farago_filter:110` (`_farago_filter_factor:212`) | CPU | ✓ — tomocupy parity (max rel Δ≈4.6e-7). Filter `1/(cos θ + db·sin θ)` over the squared reciprocal grid, built in f32 to mirror cupy (f32-sensitive: `db≈1e3` amplifies θ rounding ~1e3×); exact-f32 reciprocal coord (`reciprocal_coord_f32`) makes the filter bit-identical to numpy's; golden via scipy.fft single-precision transcription |
| `hardening::BeamCorrector`        | tomocupy `processing/external/hardening.py` + `beamhardening` pkg | CPU | done (behind `beam-hardening` feature; LUTs via Simpson over the detected spectrum + 2 `np.interp` passes; xraylib cross sections vs upstream xraydb — f64-floor parity vs an xraylib reference) |
| `alignment::scale`                | tomopy `prep/alignment.py:460`                    | CPU     | CPU ✓ — tomopy parity (bit-exact, Δ=0, incl. the returned `scl`). Divide a projection stack in place by `scl = max(\|max\|, \|min\|)` into `[−1, 1]`. Pure order statistics + elementwise f32 divide → exact. `prep::scale(data) -> scl`. Matches tomopy's no-zero-guard (all-zero → `scl=0`, NaN) |
| `alignment::blur_edges`           | tomopy `prep/alignment.py:482`                    | CPU     | CPU ✓ — tomopy parity (bit-exact, Δ=0). Multiply every projection by a radial feather mask: `rad = √((row−dy/2)²+(col−dz/2)²)`, mask `1` for `rad<low·rmax`, `0` for `rad>high·rmax`, ramp `(rmax−rad)/(rmax−rmin)` between. `√` is IEEE-correctly-rounded (no round-off floor); `arr**2 == x·x` and the in-place `f32 *= f64` is f64-then-cast → exact. Sequential mask assignment matches even `low>high`. `prep::blur_edges(data, low, high)` |
| `align::align_seq/align_joint`    | tomopy `prep/alignment.py:89,216`                 | CPU     | stub    |

Paganin params (shared): `pixel_size` [cm], `dist` [cm], `energy` [keV],
`alpha` (regularization), `pad`.

---

## E. Misc / filters (`tomoxide-prep::filters`, `tomoxide-recon::ring`)

| tomoxide                    | Upstream                                              | Status |
|-----------------------------|-------------------------------------------------------|--------|
| `ring::remove_ring`         | tomopy `misc/corr.py:751`; `libtomo/misc/remove_ring.c` | CPU ✓ — tomopy parity Δ=0 (bit-exact, `int_mode` WRAP + REFLECT) |
| `filters::median_filter3d`  | tomopy `misc/corr.py:355`; `libtomo/misc/median_filt3d.c` | CPU ✓ + wgpu ✓ — tomopy parity (bit-exact). wgpu `RankFilter` (medfilt3d.wgsl): one thread/voxel, clamp-to-center gather + partial-selection order statistic. **Bit-exact (Δ=0)** with CPU — pure gather + one subtraction, no transcendentals. Window capped at 7³ (size ≤ 7; WGSL private-array size); larger errors out (use CPU) |
| `filters::remove_outlier3d` | tomopy `misc/corr.py:413` (dezinger); tomocupy `remove_outliers` | CPU ✓ + wgpu ✓ — tomopy parity (bit-exact). Same wgpu medfilt3d kernel as `median_filter3d` with the dezinger threshold `diff`; Δ=0 vs CPU |
| `filters::gaussian_filter`  | tomopy `misc/corr.py:118`                             | CPU ✓ — tomopy parity (f32 round-off floor; realised Δ=0, 0/12852). Per-slice separable Gaussian along `axis`; faithful port of scipy.ndimage `gaussian_filter1d`+`NI_Correlate1D`: kernel `exp(−x²/2σ²)` normalised by numpy f64 pairwise sum (`pairwise_sum_f64`) then reversed; f64 convolution with scipy's exact symmetric/anti-symmetric/general summation branch, `mode='reflect'`, f32 intermediate between the two passes; derivative `order` via scipy's `q'+q·p'` recurrence. Only the kernel `exp` (numpy f64 exp vs libm) can diverge ≤1 ULP. `correlate1d_2d` is the shared primitive for `sobel_filter`. `prep::filters::gaussian_filter(data, sigma, order, axis)` |
| `filters::sobel_filter`     | tomopy `misc/corr.py:474`                             | CPU ✓ — tomopy parity (bit-exact, Δ=0). Per-slice scipy.ndimage Sobel along `axis`: `[−1,0,1]` central-difference (anti-symmetric branch) along the slice's last axis then `[1,2,1]` smoothing (symmetric branch) along the other, reusing `gaussian_filter`'s f64 `correlate1d_2d`; integer weights + exact f64 accumulation → bit-exact. tomopy's published wrapper can't run (bare `filters.sobel` NameError + numpy-2.x `arr[slc]` list-index); golden inlines the verbatim body with those two one-token compat fixes. `prep::filters::sobel_filter(data, axis)` |
| `filters::circ_mask`        | tomopy `misc/corr.py:852`                             | partial|
| `filters::remove_nan/neg`   | tomopy `misc/corr.py:506,533`                         | partial|
| `filters::median_filter_nonfinite` | tomopy `misc/corr.py:281`                      | CPU ✓ — tomopy parity (bit-exact, Δ=0). Per-projection snapshot read + size×size finite-median replace of NaN/±inf; even-count median = f32 mean of the two middles; errors on an all-non-finite kernel. `prep::filters::median_filter_nonfinite(data, size)` |
| `filters::adjust_range`     | tomopy `misc/corr.py:90`                              | CPU ✓ — tomopy parity (bit-exact, Δ=0). Clip to `[dmin, dmax]`; `None` → data min/max, bound applied only when strictly tighter (no-op otherwise). `prep::filters::adjust_range(data, dmin, dmax)` |
| `filters::median_filter`    | tomopy `misc/corr.py:167`                            | CPU ✓ — tomopy parity (bit-exact, Δ=0). Per-slice `size×size` 2-D median along `axis` (scipy.ndimage default `mode='reflect'`, half-sample reflection); every pixel replaced (no threshold). Single order statistic (rank `size·size/2`, no average) → exact. `prep::filters::median_filter(data, size, axis)`. Distinct from `median_filter3d` (3-D cube) and `median_filter_nonfinite` |
| `filters::remove_outlier`   | tomopy `misc/corr.py:559`                            | CPU ✓ — tomopy parity (bit-exact, Δ=0). Axis-chunked 2-D dezinger: per-slice `size×size` median along `axis` (scipy.ndimage default `mode='reflect'`) then replace pixel by median where `arr−median ≥ diff`. Shares the `median2d_reflect` primitive with `median_filter`. Single order statistic + f32 `where` → exact. `prep::filters::remove_outlier(data, diff, size, axis)`. Distinct from `remove_outlier3d` (3-D cube) and `remove_outlier1d` (1-D mirror) |
| `filters::remove_outlier1d` | tomopy `misc/corr.py:615`                            | CPU ✓ — tomopy parity (bit-exact, Δ=0). 1-D `size`-tap median along `axis` (scipy.ndimage `mode='mirror'`, whole-sample reflection), replace pixel by median where `arr−median ≥ diff`. Single order statistic (rank `size/2`, no average) → exact. Golden inlines tomopy's verbatim body with the numpy-2.x `arr[tuple(slc)]` compat fix (its published 1.15.3 wrapper raises on numpy 2.x). `prep::filters::remove_outlier1d(data, diff, size, axis)` |
| `filters::inpainter_morph`  | tomopy `misc/corr.py:996`; `libtomo/misc/inpainter.c` | CPU ✓ — tomopy parity (bit-exact, Δ=0) for `Mean`/`Median`; `Random` has no parity reference. Morphological inpainter (Kazantsev 2023): zero the mask, grow inward from the non-empty boundary until filled, then `iterations` smoothing passes. `axis=None` = symmetric 3-D kernel, `axis=Some(a)` = per-2-D-slice. **Mean** (`eucl_weighting`): Gaussian-distance-weighted mean; `exp`/`powf` match macOS libm bit-for-bit and the f32 multiply-accumulate is fused via `mul_add` to match libtomo's FMA-contracted build (split `*`,`+` drifts ≤2 ULP). **Median**: reproduces the C buffer-sort quirks exactly — 2-D sorts `counter_local−1` entries, 3-D sorts the whole `window_fullength` zero-padded buffer, both pick `_values[counter_local/2]` (a boundary cell can land on the padding and stay `0.0`). **Random** (rand-pair + final mean smoothing): faithfully ported but C `rand()` under OpenMP is not reproducible run-to-run (tomopy's own output varies), so it uses an internal deterministic LCG and is covered structurally by unit tests only. `prep::filters::{inpainter_morph, InpaintingType}`. Distinct from the rank/median filters |
| `morph::downsample/upsample`| tomopy `misc/morph.py:191,212`; `libtomo/misc/morph.c` (`c_sample`) | CPU ✓ — tomopy parity (bit-exact, Δ=0) across axes 0/1/2. downsample = bin mean Σ(data/binsize) f32 (flat counter, divisibility-faithful); upsample = replicate ×binsize; `morph::{downsample, upsample}(arr, level, axis)` |
| `morph::pad/trim_sinogram`  | tomopy `misc/morph.py:73,255`                         | `pad` CPU ✓ — tomopy parity (bit-exact, Δ=0), constant/edge modes, `_get_npad` default, `morph::pad(arr, axis, npad, PadMode)`. `trim_sinogram` deferred — unrunnable on modern numpy (float slice bounds / float array shape raise) |
| `morph::sino_360_to_180`    | tomopy `misc/morph.py` (`sino_360_to_180`)            | CPU ✓ — tomopy parity (bit-exact, Δ=0) for `Left`/`Right` & `overlap=0/4`. Reversed-half stitch + f64 linspace seam cross-fade cast to f32; `prep::morph::{sino_360_to_180, Rotation}` |

C-extern signatures (tomopy `extern/misc.py`):

```c
void remove_ring(float* rec,float center_x,float center_y,int dx,int dy,int dz,
                 float thresh_max,float thresh_min,float thresh,int theta_min,
                 int rwidth,int int_mode,int istart,int iend);
void medianfilter_main_float(float* in,float* out,int kernel_half_size,float absdif,
                 int ncore,int dx,int dy,int dz);
```

---

## F. Simulation (`tomoxide-sim`)

| tomoxide                  | Upstream                                | Status  |
|---------------------------|-----------------------------------------|---------|
| `phantom::shepp2d`        | tomopy `misc/phantom.py:246`            | partial |
| `phantom::shepp3d`        | tomopy `misc/phantom.py:284`            | done — tomopy parity (bit-exact, Δ=0). Faithful f64 ellipsoid rasterizer: 10 ellipsoids on `mgrid[-1:1:n·j]`, Euler-rotated (libm sin/cos of `to_radians`, bit-exact vs numpy scalar trig), inclusion `Σ((R·r−c)/s)²≤1` in f64, amplitudes accumulated in f32 like numpy `obj[mask]+=A`, then `clip(0,∞)`. The only unreproduced step (BLAS `tensordot` dot order, ≤1 ULP) flips no voxel. Sizes 16/17/32. `sim::shepp3d(size)` |
| `phantom::{baboon,…}`     | tomopy `misc/phantom.py:89…`            | stub    |
| `sim::angles`             | tomopy `sim/project.py:241`             | done    |
| `sim::project`            | tomopy `sim/project.py:268`; `libtomo/recon/project.c` | partial (CPU parallel-beam) |
| `sim::add_drift`          | tomopy `sim/project.py:80`              | done (deterministic; f32 round-off floor ≤1 ULP — numpy's vectorized f64 `np.sin` vs libm `f64::sin` survives the f32 cast; drift `i` = amp·sin(2π·i/period)+mean+linspace(0,1)[i], const across detector) |
| `sim::add_{gaussian,poisson}` | tomopy `sim/project.py:110,136` | done (distribution parity: matched moments — numpy's MT19937 stream is not reproducible; Poisson ports Knuth-mult / Hörmann PTRS) |
| `sim::add_{rings,salt_pepper,zingers}` | tomopy `sim/project.py:153,183,211` | done (distribution parity: seeded SplitMix64; add_rings = fixed per-pixel sensitivity N(1,std) broadcast over angle, add_zingers = saturate fraction f to sat, add_salt_pepper = corrupt fraction prob to val (`<`), val=None → data max). Sim-noise family complete |

---

## G. Data I/O (`tomoxide-io`)

| tomoxide              | Upstream                              | Status |
|-----------------------|---------------------------------------|--------|
| `dxchange::Reader`    | tomocupy `dataio/reader.py:59`        | done (`open_dxchange`; pure-Rust `rust-hdf5`, no libhdf5; bit-exact, gzip/uint16 fixture) |
| `tiff::write`         | tomocupy `dataio/writer.py:281` (`--save-format tiff`) | done (`create_writer`; pure-Rust `tiff`, per-slice f32, bit-exact round-trip + tifffile-verified) |
| `h5::write`           | tomocupy `dataio/writer.py` (`h5nolinks`, `--save-format h5`) | done (`create_writer`; pure-Rust `rust-hdf5`, `/exchange/data` f32 `[nz,ny,nx]` + axes/description/units attrs, bit-exact round-trip + h5py-verified) |
| `zarr::write`         | tomocupy `--save-format zarr` (`dataio/writer.py` `initialize_zarr`) | done (`create_writer`; pure-Rust, **no new dependency**). Spec-compliant Zarr v2 `DirectoryStore`: `{path}.zarr/exchange/data` with hand-written `.zgroup`/`.zarray`/`.zattrs` and one raw little-endian-f32 chunk file per z-slice (`chunks=[1,ny,nx]`, like the h5 variant's `(1,n,n)`); readable by the Python `zarr` library. Verified by round-trip reassembly + structural metadata test. **Deviates** from tomocupy's Blosc-compressed multiscale NGFF pyramid (`Blosc(blosclz, clevel=5, shuffle=2)`, default chunk `8,64,64`, `downsampleZarr`): byte-faithful reproduction would need a zarr crate + a Blosc C-binding, which conflicts with the pure-Rust no-C-dep I/O stack (`rust-hdf5`). Uncompressed single-scale stores the identical sample values; Blosc + multiscale are a documented deferral |

DXchange HDF5 layout (constants in `tomoxide-io::dxchange`):

```
/exchange/data         projections  [nproj, nz, nx]
/exchange/data_white   flat fields  [nflat, nz, nx]
/exchange/data_dark    dark fields  [ndark, nz, nx]
/exchange/theta        angles       [nproj]   (optional → generate uniformly)
```

---

## H. Pipeline & config (`tomoxide`, `tomoxide-cli`)

| tomoxide                       | Upstream                              | Status |
|--------------------------------|---------------------------------------|--------|
| `pipeline::ReconFull`          | tomocupy `rec.py:GPURec`              | stub   |
| `pipeline::ReconSteps`         | tomocupy `rec_steps.py:GPURecSteps`  | stub   |
| `pipeline::ReconTry`           | tomocupy `rec.py:recon_try`          | stub   |
| `cli` `init/recon/recon_steps/status` | tomocupy `__main__.py:162`     | partial |
| `Config`                       | tomocupy `config.py` (param groups)  | partial |

tomocupy config groups mapped to `Config` fields: file-reading, remove-stripe,
retrieve-phase, reconstruction, lamino, rotate-proj, beam-hardening, output,
inference. Reconstruction algorithms exposed: `fourierrec`, `lprec`, `linerec`
(GPU) plus tomopy's analytic+iterative set on CPU; modes `full`, `try`,
`try_lamino`.

The M3 CPU chain (`open_dxchange → normalize/minus_log → remove_stripe →
find_center_vo → fbp → TIFF out`) is verified end-to-end by
`crates/tomoxide/tests/pipeline_e2e.rs` (`m3_pipeline_hdf_to_tiff`) on one
DXchange fixture: `find_center_vo = 63.500` (Δ=0 vs the Vo golden), FBP recovery
vs the phantom Pearson `r = 0.8727`, TIFF round-trip bit-exact. The streaming
orchestration above (`ReconFull/Steps/Try`) is M5, still stubbed.

---

## Notes on faithful porting

- **Array order.** tomopy passes `(dy,dt,dx)` (sinogram order) or `(dt,dy,dx)`
  (projection order) via `sinogram_order`; tomocupy is sinogram-chunked. Keep
  `Layout` explicit (ARCHITECTURE §1) and never transpose silently.
- **Center convention.** All treat `center` as the detector-column coordinate
  of the rotation axis. Like tomocupy's `fbp_filter_center`, every backend folds
  the sub-pixel shift `exp(-2πi·f_k·(n/2−center)/pad)` (signed frequency `f_k`)
  into the FBP filter, so the analytic back-projectors and Fourier grids all
  reconstruct against a `center = n/2` geometry — the center is owned in
  `FbpFilter::apply` (CPU and wgpu share the same signed-frequency convention).
  Exceptions: gridrec runs its own integrated filter+recenter, and the iterative
  back-projectors keep `geom.center` in the matched projector/adjoint pair. The
  CUDA port mirrors the same in-filter shift; results match within tolerance.
- **f16.** tomocupy compiles `*fp16` kernel variants; tomoxide selects them by
  `Dtype::F16` on the CUDA/wgpu backends only (CPU stays f32).
- **Filters.** Same named set on both sides: `ramp/shepp/cosine/cosine2/
  hamming/hann/parzen/none` — define once in `tomoxide-recon::filter`.
- **Projector model (linear-interp vs Siddon).** tomopy's C projectors use Siddon
  ray–pixel intersection lengths (`calc_dist`); the CPU backend uses a
  pixel-driven splat / voxel-driven gather pair with linear interpolation (an
  exact adjoint pair, but a *different* `A`). Numeric reconstructions therefore
  differ from tomopy by the projector model, not a porting error — this is why
  `fbp` is gated against the phantom rather than tomopy's `fbp`, and why `grad`'s
  step-normalization `r = 1/√(ncols·nang/2)` (tuned so a unit step sits at the
  Siddon stability boundary) overshoots here: a unit fixed step diverges, so a
  smaller fixed step or the Barzilai–Borwein self-tuning step (`reg_par[0] < 0`)
  is required. `gridrec` is model-independent (Fourier), so it matches tomopy to
  r=0.98.
