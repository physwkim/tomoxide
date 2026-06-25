//! Build script for the CUDA backend.
//!
//! It is a **no-op unless the `cuda` feature is enabled**, so the crate (and
//! the whole workspace) builds on machines without a CUDA toolkit — including
//! the Apple-Silicon dev box this project is scaffolded on.
//!
//! When `cuda` is enabled it compiles tomocupy's `cfunc_*.cu` kernels plus the
//! C-ABI `shim.cpp` into a static library via `nvcc`, for the gencode arches in
//! `$TOMOXIDE_CUDA_ARCH` (default `75;80;86;89;90`). Kernel sources are taken
//! from `$TOMOXIDE_CUDA_KERNELS` (default: the vendored `cuda/` dir).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Cargo sets CARGO_FEATURE_<NAME> for each enabled feature.
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        println!("cargo:warning=tomoxide-cuda: `cuda` feature off — skipping nvcc build");
        return;
    }

    // Escape hatch: `TOMOXIDE_CUDA_SKIP_BUILD=1 cargo check --features cuda`
    // type-checks the FFI bindings on a machine without an nvcc toolkit (CI,
    // this Apple-Silicon box). It skips the nvcc compile AND the link
    // directives, so it is only valid with `cargo check`, never `build`/`run`.
    if env::var_os("TOMOXIDE_CUDA_SKIP_BUILD").is_some() {
        println!("cargo:warning=tomoxide-cuda: TOMOXIDE_CUDA_SKIP_BUILD set — type-check only (no nvcc, no link)");
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = env::var("TOMOXIDE_CUDA_KERNELS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("cuda"));
    let arches = env::var("TOMOXIDE_CUDA_ARCH").unwrap_or_else(|_| "75;80;86;89;90".into());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TOMOXIDE_CUDA_KERNELS");
    println!("cargo:rerun-if-env-changed=TOMOXIDE_CUDA_ARCH");
    println!("cargo:rerun-if-changed={}", kernel_dir.display());

    let shim = manifest_dir.join("cuda").join("shim.cpp");
    assert!(
        shim.exists(),
        "tomoxide-cuda: missing C-ABI shim at {}",
        shim.display()
    );

    // Collect kernel sources. With the placeholder vendored tree there are none
    // yet; the real port copies tomocupy/src/cuda/*.cu next to shim.cpp (or
    // points TOMOXIDE_CUDA_KERNELS at them). See cuda/README.md.
    let mut sources: Vec<PathBuf> = vec![shim.clone()];
    if kernel_dir.exists() {
        for entry in std::fs::read_dir(&kernel_dir).expect("read kernel dir") {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) == Some("cu") {
                println!("cargo:rerun-if-changed={}", path.display());
                sources.push(path);
            }
        }
    }

    let lib = out_dir.join("libtomoxide_cuda_kernels.a");
    let mut cmd = Command::new("nvcc");
    cmd.arg("-lib").arg("-o").arg(&lib);
    cmd.arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-O3")
        .arg("--std=c++17");
    // Per-thread default stream: each host thread gets its own implicit stream,
    // so concurrent `for_each_slice` workers (one per GPU, fanned across host
    // cores) overlap their FFTs/copies instead of serializing on the legacy
    // null stream. The thread-local cuFFT plan cache in fft.cu relies on this.
    cmd.arg("--default-stream").arg("per-thread");
    cmd.arg(format!("-I{}", manifest_dir.join("cuda").display()));
    let include = tomocupy_include(&kernel_dir);
    if let Some(inc) = &include {
        cmd.arg(format!("-I{}", inc.display()));
    }
    for arch in arches.split(';').filter(|a| !a.is_empty()) {
        cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }
    cmd.args(&sources);

    let status = cmd.status().unwrap_or_else(|e| {
        panic!(
            "tomoxide-cuda: failed to launch nvcc: {e}. Is the CUDA toolkit installed and on PATH?"
        )
    });
    assert!(status.success(), "tomoxide-cuda: nvcc failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=tomoxide_cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
    // cfunc_fourierrec uses cuFFT (plan1d/plan2d); cfunc_linerec does not.
    println!("cargo:rustc-link-lib=dylib=cufft");
    // The shim is C++ (new/delete on the cfunc classes) → needs the C++ runtime.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    if let Some(libdir) = cuda_lib_dir() {
        println!("cargo:rustc-link-search=native={}", libdir.display());
    }
}

/// tomocupy keeps its `.cuh` headers in a sibling `include/` directory.
fn tomocupy_include(kernel_dir: &Path) -> Option<PathBuf> {
    let candidate = kernel_dir.parent()?.join("include");
    candidate.exists().then_some(candidate)
}

/// Best-effort discovery of the CUDA runtime library directory.
fn cuda_lib_dir() -> Option<PathBuf> {
    if let Ok(root) = env::var("CUDA_PATH").or_else(|_| env::var("CUDA_HOME")) {
        let p = PathBuf::from(root).join("lib64");
        if p.exists() {
            return Some(p);
        }
    }
    let default = PathBuf::from("/usr/local/cuda/lib64");
    default.exists().then_some(default)
}
