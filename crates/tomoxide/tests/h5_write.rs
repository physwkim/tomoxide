//! Round-trip test for the single-file HDF5 reconstruction writer.
//!
//! `create_writer(.., SaveFormat::H5)` writes one contiguous `/exchange/data`
//! dataset (float32, tomocupy `h5nolinks` shape `[nz, ny, nx]`) with the
//! `axes`/`description`/`units` attributes. There is no numeric transform, so
//! the round-trip is bit-exact (Δ=0): every voxel read back equals the source
//! `Volume`. The data is read straight from the file via the same pure-Rust
//! `rust-hdf5` the writer uses, so this also exercises its write→read path.

use ndarray::{Array3, Axis, Slice};
use rust_hdf5::H5File;
use tomoxide::data::Volume;
use tomoxide::io::{create_writer, SaveFormat};

/// A unique scratch directory for this test process (no tempfile dependency).
fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tomoxide_h5_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn write_then_read_back_bit_exact() {
    let (nz, ny, nx) = (3usize, 4usize, 5usize);
    // Distinct, f32-exact, non-integer values per voxel.
    let vol = Volume::new(Array3::from_shape_fn((nz, ny, nx), |(z, y, x)| {
        z as f32 * 100.0 + y as f32 * 10.0 + x as f32 + 0.5
    }));

    let dir = scratch("roundtrip");
    let base = dir.join("recon");
    {
        let mut w = create_writer(base.to_str().unwrap(), SaveFormat::H5).unwrap();
        w.reserve(nz).unwrap();
        w.write_chunk(&vol, 0, nz).unwrap();
    } // drop the writer; the file was flushed after the chunk write.

    // The writer appends `.h5` to the base.
    let path = dir.join("recon.h5");
    assert!(path.exists(), "missing {}", path.display());

    let file = H5File::open(path.to_str().unwrap()).unwrap();
    let ds = file.dataset("exchange/data").unwrap();
    assert_eq!(ds.shape(), vec![nz, ny, nx], "dataset shape");

    // Row-major [z, y, x] must match the source volume exactly.
    let got: Vec<f32> = ds.read_raw::<f32>().unwrap();
    let expect: Vec<f32> = vol.array.iter().copied().collect();
    assert_eq!(got, expect, "/exchange/data not bit-exact");

    // tomocupy metadata attributes.
    assert_eq!(ds.attr("axes").unwrap().read_string().unwrap(), "z:y:x");
    assert_eq!(
        ds.attr("description").unwrap().read_string().unwrap(),
        "ReconData"
    );
    assert_eq!(ds.attr("units").unwrap().read_string().unwrap(), "counts");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_chunk_range_is_validated() {
    let vol = Volume::new(Array3::<f32>::zeros((2, 3, 3)));
    let dir = scratch("oob");
    let base = dir.join("r");
    let mut w = create_writer(base.to_str().unwrap(), SaveFormat::H5).unwrap();
    // Both invalid ranges are rejected before the file/dataset is created.
    assert!(w.write_chunk(&vol, 0, 5).is_err()); // end > nz
    assert!(w.write_chunk(&vol, 2, 1).is_err()); // start > end
    assert!(!dir.join("r.h5").exists(), "no file on a rejected range");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_chunks_fill_disjoint_ranges() {
    // reserve(nz) sizes the dataset, then two per-chunk volumes — local slices
    // [0,1) and [1,3) — land in their global ranges and cover it completely.
    let (nz, ny, nx) = (3usize, 2usize, 2usize);
    let vol = Volume::new(Array3::from_shape_fn((nz, ny, nx), |(z, y, x)| {
        (z * 100 + y * 10 + x) as f32 + 0.25
    }));
    let dir = scratch("chunks");
    let base = dir.join("recon");
    {
        let mut w = create_writer(base.to_str().unwrap(), SaveFormat::H5).unwrap();
        w.reserve(nz).unwrap();
        let c0 = Volume::new(vol.array.slice_axis(Axis(0), Slice::from(0..1)).to_owned());
        let c1 = Volume::new(vol.array.slice_axis(Axis(0), Slice::from(1..nz)).to_owned());
        w.write_chunk(&c0, 0, 1).unwrap();
        w.write_chunk(&c1, 1, nz).unwrap();
    }

    let file = H5File::open(dir.join("recon.h5").to_str().unwrap()).unwrap();
    let ds = file.dataset("exchange/data").unwrap();
    assert_eq!(ds.shape(), vec![nz, ny, nx], "dataset shape");
    let got: Vec<f32> = ds.read_raw::<f32>().unwrap();
    let expect: Vec<f32> = vol.array.iter().copied().collect();
    assert_eq!(got, expect, "disjoint-range fill not bit-exact");

    std::fs::remove_dir_all(&dir).ok();
}
