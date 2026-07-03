//! C-ABI bindings to the CUDA shim (`cuda/shim.cpp`).
//!
//! Minimal CUDA-runtime helpers (device probe + linear device memory) plus
//! tomocupy's `cfunc_linerec` parallel-beam back-projection, wrapped as
//! `extern "C"` so Rust links them without SWIG. Only compiled under the `cuda`
//! feature; the symbols resolve against the static library `build.rs` produces
//! with `nvcc`.
//!
//! Device pointers are opaque `*mut c_void`; `stream` is a `cudaStream_t`
//! (null = default stream). `theta`/`f`/`g` are device addresses.

use std::os::raw::c_void;

unsafe extern "C" {
    // --- device runtime helpers ---
    /// Number of CUDA devices (0 if none / driver missing).
    pub fn tomoxide_cuda_device_count() -> i32;
    /// Bind the calling host thread to a device (multi-GPU pools); 0 on success.
    pub fn tomoxide_cuda_set_device(dev: i32) -> i32;
    /// Name of the current device, written NUL-terminated into `buf` (capacity
    /// `len` bytes); 0 on success, non-zero `cudaError_t` otherwise.
    pub fn tomoxide_cuda_device_name(buf: *mut std::os::raw::c_char, len: usize) -> i32;
    /// `cudaMalloc` — returns a device pointer or null on failure.
    pub fn tomoxide_cuda_malloc(bytes: usize) -> *mut c_void;
    /// `cudaFree`.
    pub fn tomoxide_cuda_free(p: *mut c_void);
    /// `cudaMemcpy` host→device; returns 0 on success.
    pub fn tomoxide_cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    /// `cudaMemcpy` device→host; returns 0 on success.
    pub fn tomoxide_cuda_memcpy_d2h(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    /// `cudaMemcpyAsync` device→device on `stream` (null = per-thread default); 0 ok.
    pub fn tomoxide_cuda_memcpy_d2d_async(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `cudaMemset`; returns 0 on success.
    pub fn tomoxide_cuda_memset(p: *mut c_void, value: i32, bytes: usize) -> i32;
    /// `cudaDeviceSynchronize`; returns 0 on success.
    pub fn tomoxide_cuda_sync() -> i32;
    /// `cudaMemGetInfo` — free / total bytes on the current device; 0 on success.
    pub fn tomoxide_cuda_mem_info(free_bytes: *mut usize, total_bytes: *mut usize) -> i32;

    // --- async pipeline: streams, pinned host memory, async copies ---
    /// `cudaStreamCreate` — returns an opaque `cudaStream_t` or null on failure.
    pub fn tomoxide_cuda_stream_create() -> *mut c_void;
    /// `cudaStreamDestroy`.
    pub fn tomoxide_cuda_stream_destroy(stream: *mut c_void);
    /// `cudaStreamSynchronize` — block until the stream's work completes; 0 on ok.
    pub fn tomoxide_cuda_stream_sync(stream: *mut c_void) -> i32;
    /// `cudaHostAlloc` (page-locked) — returns a pinned host pointer or null.
    pub fn tomoxide_cuda_host_alloc(bytes: usize) -> *mut c_void;
    /// `cudaFreeHost`.
    pub fn tomoxide_cuda_host_free(p: *mut c_void);
    /// `cudaMemcpyAsync` host→device on `stream`; 0 on success.
    pub fn tomoxide_cuda_memcpy_h2d_async(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `cudaMemcpyAsync` device→host on `stream`; 0 on success.
    pub fn tomoxide_cuda_memcpy_d2h_async(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `cudaMemsetAsync` on `stream`; 0 on success.
    pub fn tomoxide_cuda_memset_async(
        p: *mut c_void,
        value: i32,
        bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // --- linerec (cfunc_linerec) ---
    /// `cfunc_linerec(nproj, nz, n, ncproj, ncz)`.
    pub fn tomoxide_linerec_new(
        nproj: usize,
        nz: usize,
        n: usize,
        ncproj: usize,
        ncz: usize,
    ) -> *mut c_void;
    /// `backprojection(f, g, theta, phi, gain, sz, stream)` (phi = π/2 for
    /// parallel beam). `gain` is the caller's angular quadrature weight:
    /// π/nproj for the analytic FBP paths (the dθ weight), 1.0 for the
    /// iterative solvers (pure adjoint of `tomoxide_forwardproject`).
    pub fn tomoxide_linerec_backproject(
        handle: *mut c_void,
        f: *mut c_void,
        g: *const c_void,
        theta: *const f32,
        phi: f32,
        gain: f32,
        sz: i32,
        stream: *mut c_void,
    );
    pub fn tomoxide_linerec_free(handle: *mut c_void);

    // --- forward projection (adjoint of cfunc_linerec back-projection) ---
    /// Parallel-beam forward projection (Radon), the exact discrete transpose of
    /// `tomoxide_linerec_backproject`. `g` (output sinogram `[nz, nproj, n]`)
    /// must be pre-zeroed; the kernel only `atomicAdd`s into it. `f` is the input
    /// volume `[nz, n, n]`, `phi = π/2` for parallel beam.
    pub fn tomoxide_forwardproject(
        g: *mut c_void,
        f: *const c_void,
        theta: *const f32,
        phi: f32,
        nz: i32,
        n: i32,
        nproj: i32,
        stream: *mut c_void,
    );

    // --- device-resident iterative solver elementwise ops (iterative.cu) ---
    /// `ax[i] = (b[i] - ax[i]) * rw[i]` over `n` elements (in-place into `ax`).
    pub fn tomoxide_iter_residual(
        ax: *mut c_void,
        b: *const c_void,
        rw: *const c_void,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `vol[i] += cw[i] * corr[i]` over `n` elements.
    pub fn tomoxide_iter_update(
        vol: *mut c_void,
        cw: *const c_void,
        corr: *const c_void,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `out[i] = |x[i]| > thr ? 1/x[i] : 0` over `n` elements (in-place ok).
    pub fn tomoxide_iter_recip_thresh(
        out: *mut c_void,
        x: *const c_void,
        thr: f32,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `ax[i] = |ax[i]| > 1e-6 ? b[i]/ax[i] : 0` (EM ratio, in-place into `ax`).
    pub fn tomoxide_iter_em_ratio(
        ax: *mut c_void,
        b: *const c_void,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `vol[i] = |sens[i]| > 1e-6 ? vol[i]*corr[i]/sens[i] : vol[i]` (EM update).
    pub fn tomoxide_iter_em_update(
        vol: *mut c_void,
        corr: *const c_void,
        sens: *const c_void,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// De Pierro penalized-ML pixel update over the snapshot `old` → `vol`
    /// (3-D grid `[nz,n,n]`). `delta` used only when `has_delta != 0` (hybrid
    /// prior); `reg = 0` reduces to the OSEM step.
    pub fn tomoxide_iter_pml_update(
        vol: *mut c_void,
        old: *const c_void,
        corr: *const c_void,
        sens: *const c_void,
        reg: f32,
        delta: f32,
        has_delta: i32,
        n: usize,
        nz: usize,
        stream: *mut c_void,
    ) -> i32;
    /// TV data dual proximal `pd = (pd + c·r·ax − c·b)/(1+c)` (in-place into `pd`).
    pub fn tomoxide_iter_tv_datadual(
        pd: *mut c_void,
        ax: *const c_void,
        b: *const c_void,
        c: f32,
        r: f32,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// TV dual ascent on `xbar` + λ-ball projection (interior stencil; last
    /// row/col of `p0x`/`p0y` untouched). 3-D grid `[nz,n,n]`.
    pub fn tomoxide_iter_tv_dual(
        p0x: *mut c_void,
        p0y: *mut c_void,
        xbar: *const c_void,
        c: f32,
        lambda: f32,
        n: usize,
        nz: usize,
        stream: *mut c_void,
    ) -> i32;
    /// TV primal step + over-relaxation: `x ← x − c·r·Rᵀ(pd) + c·div(pᵀᵛ)`,
    /// `xbar ← 2x − x_old`. 3-D grid `[nz,n,n]`.
    pub fn tomoxide_iter_tv_primal(
        x: *mut c_void,
        xbar: *mut c_void,
        bpv: *const c_void,
        p0x: *const c_void,
        p0y: *const c_void,
        c: f32,
        r: f32,
        n: usize,
        nz: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `ax[i] = ax[i]*r - b[i]` (GD data proximal, in-place into `ax`).
    pub fn tomoxide_iter_grad_prox(
        ax: *mut c_void,
        b: *const c_void,
        r: f32,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `grad[i] = coef * bpv[i]` (GD data gradient, fresh write).
    pub fn tomoxide_iter_grad_assemble(
        grad: *mut c_void,
        bpv: *const c_void,
        coef: f32,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `grad[i] += two_reg1 * (vol[i] - prior[i])` (Tikhonov gradient).
    pub fn tomoxide_iter_grad_tikh(
        grad: *mut c_void,
        vol: *const c_void,
        prior: *const c_void,
        two_reg1: f32,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `x[i] *= s` over `n` elements.
    pub fn tomoxide_iter_scale_inplace(
        x: *mut c_void,
        s: f32,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `vol[i] -= lambda[i/slice_len] * grad[i]` (per-slice gradient step).
    pub fn tomoxide_iter_axpy_neg_slice(
        vol: *mut c_void,
        grad: *const c_void,
        lambda: *const c_void,
        slice_len: usize,
        total_n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Per-slice BB reductions `num[z]=Σ(x−x0)(g−g0)`, `den[z]=Σ(g−g0)²` (one block
    /// per slice). `num`/`den` are device `[nz]`.
    pub fn tomoxide_iter_bb_reduce(
        num: *mut c_void,
        den: *mut c_void,
        x: *const c_void,
        x0: *const c_void,
        g: *const c_void,
        g0: *const c_void,
        slice_len: usize,
        nz: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `lambda[z] = fixed_step≥0 ? fixed_step : (is_first ? 1e-3 : den≠0 ? num/den : 1e-3)`.
    pub fn tomoxide_iter_bb_lambda(
        lambda: *mut c_void,
        num: *const c_void,
        den: *const c_void,
        fixed_step: f32,
        is_first: i32,
        nz: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Per-slice dot product `out[z] = Σ_i a[z,i]·b[z,i]` (one block per slice).
    /// `out` is device `[nz]`; `slice_len` is the per-slice element count.
    pub fn tomoxide_iter_slice_dot(
        out: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        slice_len: usize,
        nz: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CGLS direction recurrence `p[i] = z[i] + beta[s]·p[i]`, `s = i/slice_len`.
    pub fn tomoxide_iter_xpby_slice(
        p: *mut c_void,
        zv: *const c_void,
        beta: *const c_void,
        slice_len: usize,
        total_n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CGLS step `alpha[z] = wdot>0 ? gamma/wdot : 0`, `neg_alpha[z] = −alpha[z]`.
    pub fn tomoxide_iter_cgls_alpha(
        alpha: *mut c_void,
        neg_alpha: *mut c_void,
        gamma: *const c_void,
        wdot: *const c_void,
        nz: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CGLS `beta[z] = gamma>0 ? gnew/gamma : 0`, then advance `gamma[z] = gnew[z]`.
    pub fn tomoxide_iter_cgls_beta(
        beta: *mut c_void,
        gamma: *mut c_void,
        gnew: *const c_void,
        nz: usize,
        stream: *mut c_void,
    ) -> i32;

    // --- fourierrec (cfunc_fourierrec) ---
    /// `cfunc_fourierrec(nproj, nz, n, theta_ptr)` — `nz` is the number of
    /// complex slice-pairs (real input slices / 2); `theta` is a device pointer.
    pub fn tomoxide_fourierrec_new(
        nproj: usize,
        nz: usize,
        n: usize,
        theta: *const f32,
    ) -> *mut c_void;
    /// `backprojection(f, g, stream)` — `g` = packed complex filtered sinogram,
    /// `f` = packed complex output volume (both device pointers).
    pub fn tomoxide_fourierrec_backproject(
        handle: *mut c_void,
        f: *mut c_void,
        g: *const c_void,
        stream: *mut c_void,
    );
    pub fn tomoxide_fourierrec_free(handle: *mut c_void);

    // --- FBP filter (cfunc_filter) ---
    /// `cfunc_filter(nproj, nz, n)` — `n` is the padded width `ne`.
    pub fn tomoxide_filter_new(nproj: usize, nz: usize, n: usize) -> *mut c_void;
    /// `filter(g, w, stream)` — in-place R2C → ×w → C2R on the padded real
    /// sinogram `g` `[nz, nproj, ne]`; `w` is complex `[nz, ne/2+1]` (device).
    pub fn tomoxide_filter_apply(
        handle: *mut c_void,
        g: *mut c_void,
        w: *const c_void,
        stream: *mut c_void,
    );
    pub fn tomoxide_filter_free(handle: *mut c_void);

    // --- elementwise preprocessing ---
    /// `(data − dark2d) / denom` over a `[nproj, nz, nx]` projection volume;
    /// `dark2d`/`denom` are device `[nz, nx]`. Returns 0 on success.
    pub fn tomoxide_darkflat(
        data: *mut c_void,
        dark2d: *const c_void,
        denom: *const c_void,
        nproj: usize,
        nz: usize,
        nx: usize,
        stream: *mut c_void,
    ) -> i32;
    /// In-place `−ln(max(x, 1e-6))` (non-finite → 0) over `n` elements.
    pub fn tomoxide_minuslog(data: *mut c_void, n: usize, stream: *mut c_void) -> i32;
    /// Projection→sinogram transpose: `[nproj, nz, ncols]` → `[nz, nproj, ncols]`
    /// (swap the two outer axes). Pure reorder; matches the host `to_layout`
    /// permute bit-for-bit. Returns 0 on success.
    pub fn tomoxide_transpose(
        src: *const c_void,
        dst: *mut c_void,
        nproj: usize,
        nz: usize,
        ncols: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Titarenko (`remove_stripe_ti`, nblock=0) on the f32 sinogram
    /// `[nz, nproj, ncol]`, in place. `scratch` is `nz * 7 * ncol` doubles.
    /// One block per slice runs the conjugate-gradient solve in f64; held to
    /// correlation parity with the CPU golden, not bit-exactness. Returns 0 ok.
    pub fn tomoxide_stripe_ti(
        sino: *mut c_void,
        nz: usize,
        nproj: usize,
        ncol: usize,
        beta: f32,
        scratch: *mut c_void,
        stream: *mut c_void,
    ) -> i32;

    // --- Fourier-Wavelet (`remove_stripe_fw`) building blocks, orchestrated
    // from `CudaFbpStream::stripe_on_device`. All f64 except the damping FFT
    // (f32 cuFFT via `tomoxide_fft_1d`); batched over the leading `nz`. ---
    /// Pad f32 sino `[nz,nproj,ncol]` → f64 approx `[nz,nx,ncol]` at row `xshift`.
    pub fn tomoxide_fw_pad(
        in_: *const c_void,
        approx: *mut c_void,
        nz: usize,
        nproj: usize,
        ncol: usize,
        nx: usize,
        xshift: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Crop f64 `sli [nz,sliR,sliC]` rows `[xshift, xshift+nproj)` → f32 sino.
    pub fn tomoxide_fw_final(
        sli: *const c_void,
        out: *mut c_void,
        nz: usize,
        nproj: usize,
        ncol: usize,
        sli_r: usize,
        sli_c: usize,
        xshift: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Crop the top-left `[oR,oC]` block of `[nz,inR,inC]` (f64).
    pub fn tomoxide_fw_crop(
        in_: *const c_void,
        out: *mut c_void,
        nz: usize,
        in_r: usize,
        in_c: usize,
        o_r: usize,
        o_c: usize,
        stream: *mut c_void,
    ) -> i32;
    /// In-place f32 round-trip of `n` f64 elements (tomopy band quantization).
    pub fn tomoxide_fw_round(a: *mut c_void, n: usize, stream: *mut c_void) -> i32;
    /// Forward db5 DWT along the last axis: `[nz,R,C]` → `lo,hi [nz,R,(C+9)/2]`.
    pub fn tomoxide_fw_dwt_rows(
        in_: *const c_void,
        lo: *mut c_void,
        hi: *mut c_void,
        nz: usize,
        r: usize,
        c: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Forward db5 DWT along the middle axis: `[nz,R,C]` → `lo,hi [nz,(R+9)/2,C]`.
    pub fn tomoxide_fw_dwt_cols(
        in_: *const c_void,
        lo: *mut c_void,
        hi: *mut c_void,
        nz: usize,
        r: usize,
        c: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Inverse db5 DWT along the middle axis: `lo,hi [nz,L0,C]` → `[nz,2L0+2-10,C]`.
    pub fn tomoxide_fw_idwt_cols(
        lo: *const c_void,
        hi: *const c_void,
        out: *mut c_void,
        nz: usize,
        l0: usize,
        c: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Inverse db5 DWT along the last axis: `lo,hi [nz,R,L1]` → `[nz,R,2L1+2-10]`.
    pub fn tomoxide_fw_idwt_rows(
        lo: *const c_void,
        hi: *const c_void,
        out: *mut c_void,
        nz: usize,
        r: usize,
        l1: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Gather cv `[nz,my,mx]` (f64) → interleaved f32 complex `[nz*mx][my]`.
    pub fn tomoxide_fw_damp_gather(
        cv: *const c_void,
        cplx: *mut c_void,
        nz: usize,
        my: usize,
        mx: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Multiply each length-`my` spectrum by the damping vector `d[my]`.
    pub fn tomoxide_fw_damp_apply(
        cplx: *mut c_void,
        d: *const c_void,
        nz: usize,
        my: usize,
        mx: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Scatter the real part of `[nz*mx][my]` complex back into cv `[nz,my,mx]`.
    pub fn tomoxide_fw_damp_scatter(
        cplx: *const c_void,
        cv: *mut c_void,
        nz: usize,
        my: usize,
        mx: usize,
        stream: *mut c_void,
    ) -> i32;

    // --- Vo all-stripe (`remove_all_stripe`) building blocks, orchestrated
    // from `CudaFbpStream::vo_on_device`. Operate on the f32 sinogram
    // [nz, nproj, ncol] and per-column [nz, ncol] vectors; batched over nz. ---
    /// uniform_filter1d along axis 0 (projection), mode='reflect'.
    pub fn tomoxide_vo_uniform_axis0(
        sino: *const c_void,
        out: *mut c_void,
        nz: usize,
        nrow: usize,
        nc: usize,
        size: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `listdiff[z,c] = sum_r |sino - smooth|`.
    pub fn tomoxide_vo_absdiff_colsum(
        sino: *const c_void,
        smooth: *const c_void,
        listdiff: *mut c_void,
        nz: usize,
        nrow: usize,
        nc: usize,
        stream: *mut c_void,
    ) -> i32;
    /// 1-D median filter along the last axis of `[nz,R,nc]`, reflect, `size<=256`.
    pub fn tomoxide_vo_median_axis1(
        in_: *const c_void,
        out: *mut c_void,
        nz: usize,
        r: usize,
        nc: usize,
        size: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Elementwise `out = num/den` (or 1 where `den==0`).
    pub fn tomoxide_vo_ratio(
        num: *const c_void,
        den: *const c_void,
        out: *mut c_void,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Sort each column of `sino[nz,nrow,nc]` → `sorted[z,rank,c]` (+ optional
    /// `perm` rows). Composite (value,row) key = Rust stable order.
    pub fn tomoxide_vo_colsort(
        sino: *const c_void,
        sorted: *mut c_void,
        perm: *mut c_void,
        nz: usize,
        nrow: usize,
        nc: usize,
        ascending: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Sort each slice row of `in[nz,n]` → `sorted[z,rank]`.
    pub fn tomoxide_vo_slicesort(
        in_: *const c_void,
        sorted: *mut c_void,
        nz: usize,
        n: usize,
        ascending: i32,
        stream: *mut c_void,
    ) -> i32;
    /// `_detect_stripe` raw mask from `listfact` and its descending sort.
    pub fn tomoxide_vo_detect_rawmask(
        listfact: *const c_void,
        listsorted: *const c_void,
        rawmask: *mut c_void,
        nz: usize,
        nc: usize,
        snr: f32,
        stream: *mut c_void,
    ) -> i32;
    /// `binary_dilation` (3-element SE); `border_zero != 0` zeroes 2 cols/side.
    pub fn tomoxide_vo_dilate(
        rawmask: *const c_void,
        mask: *mut c_void,
        nz: usize,
        nc: usize,
        border_zero: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Compact good (`mask<1`) columns per slice → `goodx[z,:]` + `goodcount[z]`.
    pub fn tomoxide_vo_build_goodx(
        mask: *const c_void,
        goodx: *mut c_void,
        goodcount: *mut c_void,
        nz: usize,
        nc: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Per-row linear fill of dead columns from bracketing good columns.
    pub fn tomoxide_vo_interp_fill(
        sino: *const c_void,
        work: *mut c_void,
        mask: *const c_void,
        goodx: *const c_void,
        goodcount: *const c_void,
        nz: usize,
        nrow: usize,
        nc: usize,
        stream: *mut c_void,
    ) -> i32;
    /// `_rs_large` per-column intensity factor (f64 + f32 copies).
    pub fn tomoxide_vo_rs_large_listfact(
        sinosort: *const c_void,
        sinosmooth: *const c_void,
        lf64: *mut c_void,
        lf32: *mut c_void,
        nz: usize,
        nrow: usize,
        nc: usize,
        ndrop: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Normalise each column by `1/listfact` (f64 divide).
    pub fn tomoxide_vo_normalize(
        s: *const c_void,
        lf64: *const c_void,
        out: *mut c_void,
        nz: usize,
        nrow: usize,
        nc: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Scatter `sinosmooth` back through `perm` for masked columns (`out`
    /// pre-seeded with the normalised working copy).
    pub fn tomoxide_vo_scatter_masked(
        perm: *const c_void,
        sinosmooth: *const c_void,
        mask: *const c_void,
        out: *mut c_void,
        nz: usize,
        nrow: usize,
        nc: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Scatter `smoothed` back through `perm` for every column (`out` fully
    /// overwritten) — the `_rs_sort` unsort.
    pub fn tomoxide_vo_unsort_scatter(
        perm: *const c_void,
        smoothed: *const c_void,
        out: *mut c_void,
        nz: usize,
        nrow: usize,
        nc: usize,
        stream: *mut c_void,
    ) -> i32;

    // --- device-resident analytic pipeline helpers ---
    /// Edge-replicate pad `[nz,nproj,ncols]` → `[nz,nproj,ne]` (centred at `pad_side`).
    pub fn tomoxide_pad(
        src: *const c_void,
        dst: *mut c_void,
        nz: usize,
        nproj: usize,
        ncols: usize,
        ne: usize,
        pad_side: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Crop centred window `[nz,nproj,ne]` → `[nz,nproj,ncols]`.
    pub fn tomoxide_crop(
        src: *const c_void,
        dst: *mut c_void,
        nz: usize,
        nproj: usize,
        ncols: usize,
        ne: usize,
        pad_side: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Pack slice pairs `(s, s+nz/2)` → complex `[nz/2,nproj,ncols]` (interleaved).
    pub fn tomoxide_pack_pairs(
        src: *const c_void,
        dst: *mut c_void,
        nz: usize,
        nproj: usize,
        ncols: usize,
        stream: *mut c_void,
    ) -> i32;
    /// De-interleave complex `[nz/2,n,n]` → `[nz,n,n]`.
    pub fn tomoxide_unpack_pairs(
        src: *const c_void,
        dst: *mut c_void,
        nz: usize,
        n: usize,
        stream: *mut c_void,
    ) -> i32;

    // --- Log-polar (lprec) runtime kernels (cuda/lprec.cu) ---
    // Device-resident port of `recon/lprec.rs::process_row`. Buffers are batched
    // over `nz` slices: `g` [nz,nproj,n] real (also the in-place spline-coeff
    // buffer), `flc` [nz,nrho,ntheta] complex (float2), `f` [nz,n,n] real. The
    // geometry grids (`kfull`, per-span coords, target index sets) are uploaded
    // from the host `build_grids` output.
    /// Cubic-B-spline prefilter along the detector axis (one line per angle).
    pub fn tomoxide_lprec_prefilter_rows(
        g: *mut c_void,
        nz: i32,
        nproj: i32,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Cubic-B-spline prefilter along the angle axis (one strided column per detector).
    pub fn tomoxide_lprec_prefilter_cols(
        g: *mut c_void,
        nz: i32,
        nproj: i32,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Gather polar → log-polar: cubic interp of `g` accumulated into the padded
    /// real work buffer `flc` at `targets`. `xs` = detector coord (width n),
    /// `ys` = angle coord (height nproj). `ntheta_pad = 2*(ntheta/2+1)` is the
    /// padded row width (per-slice stride `nrho*ntheta_pad`); `targets` are the
    /// padded within-slice indices.
    #[allow(clippy::too_many_arguments)]
    pub fn tomoxide_lprec_gather(
        g: *const c_void,
        flc: *mut c_void,
        targets: *const c_void,
        xs: *const c_void,
        ys: *const c_void,
        npts: i32,
        nz: i32,
        nproj: i32,
        n: i32,
        nrho: i32,
        ntheta_pad: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Broadcast complex multiply `flc[s,i] *= kfull[i]` over the half-complex
    /// `[nrho, ntheta_c]` grid (`ntheta_c = ntheta/2+1`).
    pub fn tomoxide_lprec_cmul(
        flc: *mut c_void,
        kfull: *const c_void,
        nz: i32,
        nrho: i32,
        ntheta_c: i32,
        stream: *mut c_void,
    ) -> i32;
    /// Scatter log-polar → Cartesian disk: cubic interp of the padded real `flc`
    /// (×2) summed into `f` at `targets`. `xs` = theta coord (logical width
    /// ntheta), `ys` = rho coord (height nrho); `ntheta_pad` is the padded row
    /// stride.
    #[allow(clippy::too_many_arguments)]
    pub fn tomoxide_lprec_scatter(
        flc: *const c_void,
        f: *mut c_void,
        targets: *const c_void,
        xs: *const c_void,
        ys: *const c_void,
        npts: i32,
        nz: i32,
        n: i32,
        nrho: i32,
        ntheta: i32,
        ntheta_pad: i32,
        stream: *mut c_void,
    ) -> i32;

    // --- FP16 (half-precision) variants (cuda/shim_fp16.cu) ---
    // Same semantics as the f32 entry points above, but the device buffers hold
    // `half` (`real = half`) — 2 bytes/element — and the filter runs a
    // half-precision cuFFT (requires power-of-two `ne`). theta/shift stay f32.
    /// `cfunc_filter` compiled with `-DHALF`.
    pub fn tomoxide_filter_fp16_new(nproj: usize, nz: usize, n: usize) -> *mut c_void;
    pub fn tomoxide_filter_fp16_apply(
        handle: *mut c_void,
        g: *mut c_void,
        w: *const c_void,
        stream: *mut c_void,
    );
    pub fn tomoxide_filter_fp16_free(handle: *mut c_void);
    /// `cfunc_linerec` compiled with `-DHALF`.
    pub fn tomoxide_linerec_fp16_new(
        nproj: usize,
        nz: usize,
        n: usize,
        ncproj: usize,
        ncz: usize,
    ) -> *mut c_void;
    pub fn tomoxide_linerec_fp16_backproject(
        handle: *mut c_void,
        f: *mut c_void,
        g: *const c_void,
        theta: *const f32,
        phi: f32,
        gain: f32,
        sz: i32,
        stream: *mut c_void,
    );
    pub fn tomoxide_linerec_fp16_free(handle: *mut c_void);
    /// `cfunc_fourierrec` compiled with `-DHALF`.
    pub fn tomoxide_fourierrec_fp16_new(
        nproj: usize,
        nz: usize,
        n: usize,
        theta: *const f32,
    ) -> *mut c_void;
    pub fn tomoxide_fourierrec_fp16_backproject(
        handle: *mut c_void,
        f: *mut c_void,
        g: *const c_void,
        stream: *mut c_void,
    );
    pub fn tomoxide_fourierrec_fp16_free(handle: *mut c_void);
    /// Half-precision pad/crop/pack/unpack (pure `__half` data moves).
    pub fn tomoxide_pad_fp16(
        src: *const c_void,
        dst: *mut c_void,
        nz: usize,
        nproj: usize,
        ncols: usize,
        ne: usize,
        pad_side: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn tomoxide_crop_fp16(
        src: *const c_void,
        dst: *mut c_void,
        nz: usize,
        nproj: usize,
        ncols: usize,
        ne: usize,
        pad_side: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn tomoxide_pack_pairs_fp16(
        src: *const c_void,
        dst: *mut c_void,
        nz: usize,
        nproj: usize,
        ncols: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn tomoxide_unpack_pairs_fp16(
        src: *const c_void,
        dst: *mut c_void,
        nz: usize,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// On-device cast of `n` contiguous `f32` elements to `f16`, on `stream`.
    pub fn tomoxide_cast_f32_to_f16(
        src: *const c_void,
        dst: *mut c_void,
        n: usize,
        stream: *mut c_void,
    ) -> i32;
    /// On-device cast of `n` contiguous `f16` elements to `f32`, on `stream`.
    pub fn tomoxide_cast_f16_to_f32(
        src: *const c_void,
        dst: *mut c_void,
        n: usize,
        stream: *mut c_void,
    ) -> i32;

    // --- batched C2C FFT (cuFFT) ---
    /// In-place batched 1-D C2C FFT (`data` = device interleaved float2, length
    /// `n*batch`); inverse is normalized by `1/n`. Returns 0 on success.
    pub fn tomoxide_fft_1d(data: *mut c_void, n: usize, batch: usize, inverse: i32) -> i32;
    /// In-place batched 2-D C2C FFT (`rows*cols*batch`); inverse normalized by
    /// `1/(rows*cols)`.
    pub fn tomoxide_fft_2d(
        data: *mut c_void,
        rows: usize,
        cols: usize,
        batch: usize,
        inverse: i32,
    ) -> i32;
    /// In-place batched 2-D **R2C** FFT (forward, unnormalized). `data` is a
    /// row-padded real buffer `[rows, 2*(cols/2+1)]` per image (so it overlays
    /// the half-complex output `[rows, cols/2+1]`); on return it holds the
    /// spectrum. Returns 0 on success.
    pub fn tomoxide_fft_2d_r2c(data: *mut c_void, rows: usize, cols: usize, batch: usize) -> i32;
    /// In-place batched 2-D **C2R** FFT (inverse, normalized by `1/(rows*cols)`).
    /// Consumes the half-complex spectrum `[rows, cols/2+1]` and writes the
    /// row-padded real image `[rows, 2*(cols/2+1)]` per image in place.
    pub fn tomoxide_fft_2d_c2r(data: *mut c_void, rows: usize, cols: usize, batch: usize) -> i32;
}
