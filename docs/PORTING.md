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
| `recon::fbp`                 | tomopy `libtomo/recon/fbp.c`; `extern/recon.py:238`         | CPU      | stub   |
| `recon::gridrec`             | tomopy `libtomo/gridrec/gridrec.c:195`; `extern/gridrec.py:64` | CPU   | stub   |
| `recon::fourierrec`          | tomocupy `reconstruction/fourierrec.py:46`; `cuda/cfunc_fourierrec.cu`, `include/cfunc_fourierrec.cuh:10` | CUDA, wgpu | stub |
| `recon::lprec`               | tomocupy `reconstruction/lprec.py:292`; `cuda/cfunc_lprec.cu`, `include/cfunc_lprec.cuh:9` | CUDA, wgpu | stub |
| `recon::linerec`             | tomocupy `reconstruction/linerec.py:47`; `cuda/cfunc_linerec.cu`, `include/cfunc_linerec.cuh:9` | CUDA, wgpu | stub |
| `recon::filter` (FBP filter) | tomocupy `reconstruction/fbp_filter.py:46`; `cuda/cfunc_filter.cu`; tomopy filter in `fbp.c` | all | stub |
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
| `Art`                  | `num_iter`                          | `recon/art.c`        | stub   |
| `Bart`                 | `num_iter,num_block,ind_block`      | `recon/bart.c`       | stub   |
| `Sirt`                 | `num_iter`                          | `recon/sirt.c` (+`accel/cxx/sirt.cc`) | stub |
| `Mlem`                 | `num_iter`                          | `accel/cxx/mlem.cc`  | stub   |
| `Osem`                 | `num_iter,num_block,ind_block`      | `recon/osem.c`       | stub   |
| `OspmlHybrid`          | `num_iter,reg_par,num_block,ind_block` | `recon/ospml_hybrid.c` | stub |
| `OspmlQuad`            | `num_iter,reg_par,num_block,ind_block` | `recon/ospml_quad.c`   | stub |
| `PmlHybrid`            | `num_iter,reg_par`                  | `recon/pml_hybrid.c` | stub   |
| `PmlQuad`              | `num_iter,reg_par`                  | `recon/pml_quad.c`   | stub   |
| `Tv`                   | `num_iter,reg_par`                  | `recon/tv.c`         | stub   |
| `Grad`                 | `num_iter,reg_par`                  | `recon/grad.c`       | stub   |
| `Tikh`                 | `num_iter,reg_data,reg_par`         | `recon/tikh.c`       | stub   |
| `Vector{,2,3}`         | `num_iter,axis…`                    | `recon/vector.c`     | stub   |

Forward model shared by all: tomopy `libtomo/recon/project.c`
(`void project(const float* obj,…,float* data,…,const float* center,const float* theta)`)
→ `tomoxide-recon::project` / `ForwardProject` capability.

---

## C. Center finding (`tomoxide-recon::center`)

| tomoxide                | Upstream                                  | Status |
|-------------------------|-------------------------------------------|--------|
| `center::find_center`   | tomopy `recon/rotation.py:82` (entropy)   | stub   |
| `center::find_center_vo`| tomopy `recon/rotation.py:205` (Vo coarse+fine) | stub |
| `center::find_center_pc`| tomopy `recon/rotation.py:391` (phase corr) | stub |
| `center::write_center`  | tomopy `recon/rotation.py:438`            | stub   |
| `center::find_center_sift` | tomocupy `find_center.py:99`           | stub   |
| `center::find_center_ai`| tomocupy `find_center.py:86` (+`ai/inference.py`) | stub |

---

## D. Preprocessing (`tomoxide-prep`)

| tomoxide                          | Upstream                                          | Backend | Status  |
|-----------------------------------|---------------------------------------------------|---------|---------|
| `normalize::normalize`            | tomopy `prep/normalize.py:98`                     | CPU     | partial |
| `normalize::normalize_bg`         | tomopy `prep/normalize.py:207`; `libtomo/prep/prep.c` (`normalize_bg`) | CPU | stub |
| `normalize::normalize_nf`         | tomopy `prep/normalize.py:245`                    | CPU     | stub    |
| `normalize::minus_log`            | tomopy `prep/normalize.py:72`; tomocupy `proc_functions.minus_log` | all | partial |
| `normalize::darkflat`             | tomocupy `proc_functions.darkflat_correction:55`  | all     | partial |
| `stripe::remove_stripe_fw`        | tomopy `prep/stripe.py:88`; tomocupy `remove_stripe.remove_stripe_fw` | CPU/GPU | stub |
| `stripe::remove_stripe_ti`        | tomopy `prep/stripe.py:179`; tomocupy `remove_stripe_ti` | CPU/GPU | stub |
| `stripe::remove_stripe_sf`        | tomopy `prep/stripe.py:333`; `libtomo/prep/stripe.c` (`remove_stripe_sf`) | CPU | stub |
| `stripe::remove_stripe_based_sorting` | tomopy `prep/stripe.py:363` (Vo alg. 3)       | CPU     | stub    |
| `stripe::remove_stripe_based_filtering` | tomopy `prep/stripe.py:437` (Vo alg. 2)     | CPU     | stub    |
| `stripe::remove_stripe_based_fitting` | tomopy `prep/stripe.py:520` (Vo alg. 1)       | CPU     | stub    |
| `stripe::remove_all_stripe`       | tomocupy `remove_stripe.remove_all_stripe` (vo-all) | GPU   | stub    |
| `stripe::stripes_detect3d`        | tomopy `prep/stripe.py:984`; `libtomo/prep/stripes_detect3d.c` | CPU | stub |
| `phase::retrieve_phase` (Paganin) | tomopy `prep/phase.py:80`; tomocupy `retrieve_phase.paganin_filter:59` | all | stub |
| `phase::retrieve_phase_g` (Gpaganin/farago) | tomocupy `retrieve_phase.farago_filter:110`  | GPU | stub |
| `hardening::beam_correct`         | tomocupy `processing/external/hardening.py:50`    | GPU     | stub    |
| `align::align_seq/align_joint`    | tomopy `prep/alignment.py:89,216`                 | CPU     | stub    |

Paganin params (shared): `pixel_size` [cm], `dist` [cm], `energy` [keV],
`alpha` (regularization), `pad`.

---

## E. Misc / filters (`tomoxide-prep::filters`, `tomoxide-recon::ring`)

| tomoxide                    | Upstream                                              | Status |
|-----------------------------|-------------------------------------------------------|--------|
| `ring::remove_ring`         | tomopy `misc/corr.py:751`; `libtomo/misc/remove_ring.c` | stub |
| `filters::median_filter3d`  | tomopy `misc/corr.py:355`; `libtomo/misc/median_filt3d.c` | stub |
| `filters::remove_outlier3d` | tomopy `misc/corr.py:413` (dezinger); tomocupy `remove_outliers` | stub |
| `filters::gaussian_filter`  | tomopy `misc/corr.py:118`                             | stub   |
| `filters::circ_mask`        | tomopy `misc/corr.py:852`                             | partial|
| `filters::remove_nan/neg`   | tomopy `misc/corr.py:506,533`                         | partial|
| `filters::inpainter_morph`  | tomopy `misc/corr.py`; `libtomo/misc/inpainter.c`     | stub   |
| `morph::downsample/upsample`| tomopy `misc/morph.py:191,212`                        | stub   |
| `morph::pad/trim_sinogram`  | tomopy `misc/morph.py:73,255`                         | stub   |
| `morph::sino_360_to_180`    | tomopy `misc/morph.py` (`sino_360_to_180`)            | stub   |

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
| `sim::project`            | tomopy `sim/project.py:268`; `libtomo/recon/project.c` | stub |
| `sim::add_{gaussian,poisson,rings,zingers}` | tomopy `sim/project.py:110,136,153,211` | stub |

---

## G. Data I/O (`tomoxide-io`)

| tomoxide              | Upstream                              | Status |
|-----------------------|---------------------------------------|--------|
| `dxchange::Reader`    | tomocupy `dataio/reader.py:59`        | stub   |
| `dxchange::Writer`    | tomocupy `dataio/writer.py:73`        | stub   |
| `tiff::{read,write}`  | tomocupy `--save-format tiff`         | stub   |
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

---

## Notes on faithful porting

- **Array order.** tomopy passes `(dy,dt,dx)` (sinogram order) or `(dt,dy,dx)`
  (projection order) via `sinogram_order`; tomocupy is sinogram-chunked. Keep
  `Layout` explicit (ARCHITECTURE §1) and never transpose silently.
- **Center convention.** Both treat `center` as the detector-column coordinate
  of the rotation axis; FBP filtering applies the sub-pixel shift
  `exp(-2πi·(-center+n/2)·freq)` (tomocupy `fbp_filter_center`).
- **f16.** tomocupy compiles `*fp16` kernel variants; tomoxide selects them by
  `Dtype::F16` on the CUDA/wgpu backends only (CPU stays f32).
- **Filters.** Same named set on both sides: `ramp/shepp/cosine/cosine2/
  hamming/hann/parzen/none` — define once in `tomoxide-recon::filter`.
