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
| `recon::fbp`                 | tomopy `libtomo/recon/fbp.c`; `extern/recon.py:238`         | CPU      | CPU done (golden: recovers phantom from tomopy sinogram r=0.87; tomopy's own `fbp` is a weak reference) |
| `recon::gridrec`             | tomopy `libtomo/gridrec/gridrec.c:195`; `extern/gridrec.py:64` | CPU   | CPU done (DFI/Kaiser-Bessel not PSWF; golden: vs tomopy gridrec r=0.98; Fourier recenter shift uses signed freq — correct at sub-pixel centers) |
| `recon::fourierrec`          | tomocupy `reconstruction/fourierrec.py:46`; `cuda/cfunc_fourierrec.cu`, `include/cfunc_fourierrec.cuh:10` | CUDA, wgpu | stub |
| `recon::lprec`               | tomocupy `reconstruction/lprec.py:292`; `cuda/cfunc_lprec.cu`, `include/cfunc_lprec.cuh:9` | CUDA, wgpu | stub |
| `recon::linerec`             | tomocupy `reconstruction/linerec.py:47`; `cuda/cfunc_linerec.cu`, `include/cfunc_linerec.cuh:9` | CUDA, wgpu | stub |
| `recon::filter` (FBP filter) | tomocupy `reconstruction/fbp_filter.py:46`; `cuda/cfunc_filter.cu`; tomopy filter in `fbp.c` | all | partial (CPU done) |
| `recon::lamino` (USFFT)      | tomocupy `reconstruction/lamfourierrec.py`; `cuda/cfunc_usfft1d.cu`, `cfunc_usfft2d.cu`, `cfunc_fft2d.cu` | CUDA | stub |

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
| `Vector{,2,3}`         | `num_iter,axis…`                    | `recon/vector.c`     | stub   |

Forward model shared by all: tomopy `libtomo/recon/project.c`
(`void project(const float* obj,…,float* data,…,const float* center,const float* theta)`)
→ `tomoxide-recon::project` / `ForwardProject` capability.

---

## C. Center finding (`tomoxide-recon::center`)

| tomoxide                | Upstream                                  | Status |
|-------------------------|-------------------------------------------|--------|
| `center::find_center`   | tomopy `recon/rotation.py:82` (entropy)   | CPU ✓ — recovery (projector-coupled via gridrec): true axis ±0.5 px, tomopy `find_center` ±1 px |
| `center::find_center_vo`| tomopy `recon/rotation.py:205` (Vo coarse+fine) | CPU ✓ — tomopy parity Δ=0 |
| `center::find_center_pc`| tomopy `recon/rotation.py:391` (phase corr; skimage `phase_cross_correlation`) | CPU ✓ — tomopy parity Δ=0 (`rotc_guess` path not ported) |
| `center::write_center`  | tomopy `recon/rotation.py:438`            | CPU ✓ — center enumeration Δ=0 vs numpy `arange`; recon content gridrec-family (KB kernel + ramp, not PSWF+parzen), self-consistent vs `recon(Gridrec)` (Δ=0). Returns `(centers, stack)`; I/O-free core (persist `{center:.2f}.tiff` via tomoxide-io) |
| `center::find_center_sift` | tomocupy `find_center.py:99`           | stub   |
| `center::find_center_ai`| tomocupy `find_center.py:86` (+`ai/inference.py`) | stub |

---

## D. Preprocessing (`tomoxide-prep`)

| tomoxide                          | Upstream                                          | Backend | Status  |
|-----------------------------------|---------------------------------------------------|---------|---------|
| `normalize::normalize`            | tomopy `prep/normalize.py:98`                     | CPU     | partial |
| `normalize::normalize_bg`         | tomopy `prep/normalize.py:207`; `libtomo/prep/prep.c` (`normalize_bg`) | CPU | CPU ✓ — tomopy parity (bit-exact, Δ=0) for `air=1` & `air=4`. Per-row air baseline (left/right boundary means) lerp'd across the column axis, divide; f32 in upstream order, `f32::mul_add` for the clang-contracted `air_left + air_slope·j` |
| `normalize::normalize_nf`         | tomopy `prep/normalize.py:245`                    | CPU     | CPU ✓ — tomopy parity (bit-exact, Δ=0), `averaging='mean'`. Per-group flat median, dark mean, `(proj−dark)/max(flat−dark,1e-6)` + cutoff; half-to-even group boundaries. `averaging='median'` returns TODO (upstream `np.median(dtype=)` errors on modern numpy) |
| `normalize::minus_log`            | tomopy `prep/normalize.py:72`; tomocupy `proc_functions.minus_log` | all | partial |
| `normalize::darkflat`             | tomocupy `proc_functions.darkflat_correction:55`  | all     | partial |
| `stripe::remove_stripe_fw`        | tomopy `prep/stripe.py:88`; tomocupy `remove_stripe.remove_stripe_fw` | CPU | CPU ✓ — tomopy parity (≈f32 floor, max rel Δ≈1.2e-6). db5 dwt2/idwt2 hand-ported (no new dep) in `wavelet.rs`, pywt-validated; float32-forward/f64-damp+inverse dtype flow. Damping uses a self-contained `O(n log n)` FFT (radix-2 + Bluestein, arbitrary length, no FFT dep) in `fft.rs` |
| `stripe::remove_stripe_ti`        | tomopy `prep/stripe.py:179` (Titarenko/Miqueles) | CPU | CPU ✓ — tomopy parity (≈f32 floor, max rel Δ≈5.2e-7); default `nblock=0` only (`_ringb` block path unrunnable on modern numpy) |
| `stripe::remove_stripe_sf`        | tomopy `prep/stripe.py:333`; `libtomo/prep/stripe.c` (`remove_stripe_sf`) | CPU | CPU ✓ — tomopy parity (bit-exact) |
| `stripe::remove_stripe_based_sorting` | tomopy `prep/stripe.py:363` (Vo alg. 3)       | CPU     | CPU ✓ — tomopy parity (bit-exact, Δ=0) for `dim=1` & `dim=2`. `_rs_sort` is a pure rank-filter selection on f32; `StripeMethod::VoSort { size, dim }`, reuses the `rs_sort` scaffold (now smoother-pluggable) shared with `VoAll` |
| `stripe::remove_stripe_based_filtering` | tomopy `prep/stripe.py:437` (Vo alg. 2)     | CPU     | CPU ✓ — tomopy parity to the f32 floor (measured Δ=0) for `dim=1` & `dim=2`. `_rs_filter` Gaussian-Fourier low-pass + `_rs_sort` + high-pass residual; `StripeMethod::VoFilter { sigma, size, dim }`, reuses the f64 column FFT (`fft.rs`) and the `rs_sort`/`median_filter_2d` scaffolds from `VoSort` |
| `stripe::remove_stripe_based_fitting` | tomopy `prep/stripe.py:520` (Vo alg. 1)       | CPU     | stub    |
| `stripe::remove_all_stripe`       | tomopy `prep/stripe.py:843` (Vo alg. 3+5+6); tomocupy `remove_stripe.remove_all_stripe` | CPU | CPU ✓ — tomopy parity (≈f32 floor, max rel Δ≈5.8e-7) |
| `stripe::stripes_detect3d`        | tomopy `prep/stripe.py:984`; `libtomo/prep/stripes_detect3d.c` | CPU | stub |
| `phase::retrieve_phase` (Paganin) | tomopy `prep/phase.py:80`; tomocupy `retrieve_phase.paganin_filter:59` | CPU | ✓ — tomopy parity (max rel Δ≈2.4e-7) |
| `phase::retrieve_phase` (Gpaganin) | tomocupy `retrieve_phase.paganin_filter:59` (`method='Gpaganin'`, `_paganin_filter_factorG:215`) | CPU | ✓ — tomocupy parity (max rel Δ≈4.9e-7). Grid/filter in f32 to mirror cupy single precision (ill-conditioned, `scale≈1.2e3`); golden via scipy.fft single-precision transcription |
| `phase::retrieve_phase` (farago) | tomocupy `retrieve_phase.farago_filter:110` (`_farago_filter_factor:212`) | CPU | ✓ — tomocupy parity (max rel Δ≈4.6e-7). Filter `1/(cos θ + db·sin θ)` over the squared reciprocal grid, built in f32 to mirror cupy (f32-sensitive: `db≈1e3` amplifies θ rounding ~1e3×); exact-f32 reciprocal coord (`reciprocal_coord_f32`) makes the filter bit-identical to numpy's; golden via scipy.fft single-precision transcription |
| `hardening::beam_correct`         | tomocupy `processing/external/hardening.py:50`    | GPU     | stub    |
| `align::align_seq/align_joint`    | tomopy `prep/alignment.py:89,216`                 | CPU     | stub    |

Paganin params (shared): `pixel_size` [cm], `dist` [cm], `energy` [keV],
`alpha` (regularization), `pad`.

---

## E. Misc / filters (`tomoxide-prep::filters`, `tomoxide-recon::ring`)

| tomoxide                    | Upstream                                              | Status |
|-----------------------------|-------------------------------------------------------|--------|
| `ring::remove_ring`         | tomopy `misc/corr.py:751`; `libtomo/misc/remove_ring.c` | CPU ✓ — tomopy parity Δ=0 (bit-exact, `int_mode` WRAP + REFLECT) |
| `filters::median_filter3d`  | tomopy `misc/corr.py:355`; `libtomo/misc/median_filt3d.c` | CPU ✓ — tomopy parity (bit-exact) |
| `filters::remove_outlier3d` | tomopy `misc/corr.py:413` (dezinger); tomocupy `remove_outliers` | CPU ✓ — tomopy parity (bit-exact) |
| `filters::gaussian_filter`  | tomopy `misc/corr.py:118`                             | stub   |
| `filters::circ_mask`        | tomopy `misc/corr.py:852`                             | partial|
| `filters::remove_nan/neg`   | tomopy `misc/corr.py:506,533`                         | partial|
| `filters::inpainter_morph`  | tomopy `misc/corr.py`; `libtomo/misc/inpainter.c`     | stub   |
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
| `phantom::shepp3d`        | tomopy `misc/phantom.py`                | stub    |
| `phantom::{baboon,…}`     | tomopy `misc/phantom.py:89…`            | stub    |
| `sim::angles`             | tomopy `sim/project.py:241`             | done    |
| `sim::project`            | tomopy `sim/project.py:268`; `libtomo/recon/project.c` | partial (CPU parallel-beam) |
| `sim::add_{gaussian,poisson}` | tomopy `sim/project.py:110,136` | done (distribution parity: matched moments — numpy's MT19937 stream is not reproducible; Poisson ports Knuth-mult / Hörmann PTRS) |
| `sim::add_{rings,zingers}` | tomopy `sim/project.py:153,211` | stub |

---

## G. Data I/O (`tomoxide-io`)

| tomoxide              | Upstream                              | Status |
|-----------------------|---------------------------------------|--------|
| `dxchange::Reader`    | tomocupy `dataio/reader.py:59`        | done (`open_dxchange`; pure-Rust `rust-hdf5`, no libhdf5; bit-exact, gzip/uint16 fixture) |
| `tiff::write`         | tomocupy `dataio/writer.py:281` (`--save-format tiff`) | done (`create_writer`; pure-Rust `tiff`, per-slice f32, bit-exact round-trip + tifffile-verified) |
| `h5::write`           | tomocupy `dataio/writer.py` (`h5nolinks`, `--save-format h5`) | done (`create_writer`; pure-Rust `rust-hdf5`, `/exchange/data` f32 `[nz,ny,nx]` + axes/description/units attrs, bit-exact round-trip + h5py-verified) |
| `zarr::write`         | tomocupy `--save-format zarr`         | stub   |

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
  of the rotation axis. tomocupy folds the sub-pixel shift
  `exp(-2πi·(-center+n/2)·freq)` into the FBP filter (`fbp_filter_center`). The
  CPU backend instead keeps the filter a pure shift-invariant ramp and applies
  `center` directly in the back-projector's `t = …+center` sampling — the
  filter's `geom` argument is unused on CPU. The CUDA port can keep tomocupy's
  in-filter shift; results must match within tolerance regardless of where the
  shift lives.
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
