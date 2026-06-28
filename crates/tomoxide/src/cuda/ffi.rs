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
    /// `cudaMalloc` — returns a device pointer or null on failure.
    pub fn tomoxide_cuda_malloc(bytes: usize) -> *mut c_void;
    /// `cudaFree`.
    pub fn tomoxide_cuda_free(p: *mut c_void);
    /// `cudaMemcpy` host→device; returns 0 on success.
    pub fn tomoxide_cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    /// `cudaMemcpy` device→host; returns 0 on success.
    pub fn tomoxide_cuda_memcpy_d2h(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
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
    /// `backprojection(f, g, theta, phi, sz, stream)` (phi = π/2 for parallel beam).
    pub fn tomoxide_linerec_backproject(
        handle: *mut c_void,
        f: *mut c_void,
        g: *const c_void,
        theta: *const f32,
        phi: f32,
        sz: i32,
        stream: *mut c_void,
    );
    pub fn tomoxide_linerec_free(handle: *mut c_void);

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
}
