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
}
