//! C-ABI bindings to the CUDA kernel shim (`cuda/shim.cpp`).
//!
//! These mirror tomocupy's SWIG-exposed `cfunc_*` classes, wrapped as
//! `extern "C"` create/call/free triples so Rust can link them without SWIG.
//! Only compiled under the `cuda` feature; the symbols resolve against the
//! static library `build.rs` produces with `nvcc`.
//!
//! Pointers (`f`, `g`) are device addresses and `stream` is a `cudaStream_t`;
//! tomocupy passes these as `size_t`, here they are opaque `*mut c_void`.
#![allow(dead_code)]

use std::os::raw::c_void;

unsafe extern "C" {
    // --- fourierrec (cfunc_fourierrec) ---
    /// `cfunc_fourierrec(nproj, nz, n, theta_ptr)`
    pub fn tomoxide_fourierrec_new(
        nproj: usize,
        nz: usize,
        n: usize,
        theta: *const f32,
    ) -> *mut c_void;
    /// `backprojection(f_ptr, g_ptr, stream_ptr)`
    pub fn tomoxide_fourierrec_backproject(
        handle: *mut c_void,
        f: *mut c_void,
        g: *const c_void,
        stream: *mut c_void,
    );
    pub fn tomoxide_fourierrec_free(handle: *mut c_void);

    // --- lprec (cfunc_lprec) ---
    /// `cfunc_lprec(nproj, nz, n, ntheta, nrho)`
    pub fn tomoxide_lprec_new(
        nproj: usize,
        nz: usize,
        n: usize,
        ntheta: usize,
        nrho: usize,
    ) -> *mut c_void;
    /// `backprojection(f_ptr, g_ptr, stream_ptr)` (grids set internally)
    pub fn tomoxide_lprec_backproject(
        handle: *mut c_void,
        f: *mut c_void,
        g: *const c_void,
        stream: *mut c_void,
    );
    pub fn tomoxide_lprec_free(handle: *mut c_void);

    // --- linerec (cfunc_linerec) ---
    /// `cfunc_linerec(nproj, nz, n, ncproj, ncz)`
    pub fn tomoxide_linerec_new(
        nproj: usize,
        nz: usize,
        n: usize,
        ncproj: usize,
        ncz: usize,
    ) -> *mut c_void;
    /// `backprojection(f, g, theta, phi, sz, stream)` (phi = laminography angle)
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

    // --- FBP filter (cfunc_filter) ---
    /// `cfunc_filter(nproj, nz, n)`
    pub fn tomoxide_filter_new(nproj: usize, nz: usize, n: usize) -> *mut c_void;
    /// `filter(g_ptr, w_ptr, stream_ptr)`
    pub fn tomoxide_filter_apply(
        handle: *mut c_void,
        g: *mut c_void,
        w: *const c_void,
        stream: *mut c_void,
    );
    pub fn tomoxide_filter_free(handle: *mut c_void);
}
