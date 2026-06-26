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
