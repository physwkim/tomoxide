# tomoxide v0.3.0

tomoxide is a Rust tomographic reconstruction toolkit combining the algorithmic
breadth of tomopy (CPU `libtomo`) with the GPU-accelerated streaming
reconstruction of tomocupy (CUDA), behind a single tri-backend abstraction:
CPU / CUDA / wgpu.

This is a **filter correctness / convention** release: the CUDA analytic path
now matches tomocupy in absolute amplitude and ramp shape, and the default
filter follows tomocupy. It also adds GPU laminography.

## Behaviour changes — read before upgrading

- **CUDA analytic amplitude halved.** In 0.2.0 every CUDA analytic method
  (fbp / linerec / fourierrec / lprec / laminography, f32 + fp16) emitted output
  **2× larger** than tomocupy. 0.3.0 fixes the filter normalization so CUDA
  matches tomocupy in absolute amplitude. The CPU/wgpu path still matches
  tomopy. If you compared CUDA output against absolute units or against 0.2.0
  CUDA results, expect a factor-of-2 change.
- **Default FBP filter is now `parzen`** (was `ramp`), matching tomocupy's
  default. Default reconstructions are smoother; set `filter_name =
  FilterName::Ramp` to restore the sharp ramp.

## Highlights

- **CUDA analytic output matches tomocupy bit-for-bit.** Beyond the amplitude
  fix, the CUDA filter ramp is ported to tomocupy's exact degree-12 `_wint`
  quadrature shape, closing the residual ~1% straight-line-ramp gap. On a
  synthetic 180° dataset, CUDA fbp/linerec/fourierrec now match tomocupy at
  scale 1.00000, Pearson 1.000000.
- **Per-backend ramp by reference.** Each backend now builds the ramp of the
  library it ports: CPU/wgpu use tomopy's linear ramp, CUDA uses tomocupy's
  `_wint` quadrature ramp (`backend::RampShape`). Apodization, padding, clamp,
  DC doubling and FFT layout stay shared in `make_fbp_filter`.
- **GPU laminography.** `recon --lamino_angle` runs analytic linerec with a
  tilted rotation axis on CUDA, verified against tomocupy on real leaf data
  (Pearson 0.99997).

## Cross-backend convention note

The CUDA analytic streaming kernels emit each slice with tomocupy's handedness:
a vertical (y-axis) flip plus a per-algorithm scale. As of 0.3.0 the scales are
`2/π` for fbp/linerec and `≈2·n²` for fourierrec (half their 0.2.0 values, from
the amplitude fix), `½` for lprec, and `1` for gridrec (which keeps the
CPU/tomopy orientation and scale). After undoing the flip and scale, CUDA
matches the CPU/wgpu path very closely; a deterministic ~0.6% residual remains
from the tomocupy-`_wint` vs tomopy-linear ramp shapes (Pearson ≈1.0). This is
documented in `docs/ARCHITECTURE.md` §4.1 and pinned by
`tests/cuda_cpu_convention_parity.rs`.

## Install

```toml
[dependencies]
tomoxide = "0.3.0"
```

CLI:

```sh
cargo install tomoxide-cli
```

GPU features are opt-in (`cuda`, `gpu-wgpu`); see the README for the feature
matrix and benchmark tables.

See [CHANGELOG.md](CHANGELOG.md) for the complete list of changes.
