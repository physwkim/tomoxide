//! Round-trip test for the per-slice TIFF writer.
//!
//! The writer emits one 32-bit-float TIFF per reconstruction slice
//! (`{prefix}_{i:05}.tiff`), matching tomocupy `dataio/writer.py:281`. There is
//! no numeric transform, so the round-trip is bit-exact (Δ=0): every pixel read
//! back equals the source `Volume`.

use std::fs::File;
use std::io::BufReader;

use ndarray::Array3;
use tiff::decoder::{Decoder, DecodingResult};
use tomoxide::data::Volume;
use tomoxide::io::{create_writer, SaveFormat};

/// A unique scratch directory for this test process (no tempfile dependency).
fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("tomoxide_tiff_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn read_tiff_f32(path: &std::path::Path) -> ((u32, u32), Vec<f32>) {
    let mut dec = Decoder::new(BufReader::new(File::open(path).unwrap())).unwrap();
    let dims = dec.dimensions().unwrap();
    match dec.read_image().unwrap() {
        DecodingResult::F32(v) => (dims, v),
        other => panic!("expected F32 image, got {other:?}"),
    }
}

#[test]
fn write_then_read_back_bit_exact() {
    let (nz, ny, nx) = (3usize, 4usize, 5usize);
    // Distinct, f32-exact, non-integer values per voxel.
    let vol = Volume::new(Array3::from_shape_fn((nz, ny, nx), |(z, y, x)| {
        z as f32 * 100.0 + y as f32 * 10.0 + x as f32 + 0.5
    }));

    let dir = scratch("roundtrip");
    let prefix = dir.join("recon");
    let prefix_str = prefix.to_str().unwrap();

    let mut w = create_writer(prefix_str, SaveFormat::Tiff).unwrap();
    w.write_chunk(&vol, 0, nz).unwrap();

    for z in 0..nz {
        let fname = dir.join(format!("recon_{z:05}.tiff"));
        assert!(fname.exists(), "missing {}", fname.display());

        let ((w_px, h_px), buf) = read_tiff_f32(&fname);
        // TIFF width = nx (columns/x), height = ny (rows/y).
        assert_eq!((w_px, h_px), (nx as u32, ny as u32), "slice {z} dimensions");

        // Row-major [y, x] must match the source slice exactly.
        let expect: Vec<f32> = (0..ny)
            .flat_map(|y| (0..nx).map(move |x| z as f32 * 100.0 + y as f32 * 10.0 + x as f32 + 0.5))
            .collect();
        assert_eq!(buf, expect, "slice {z} pixels");
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_chunk_range_is_validated() {
    let vol = Volume::new(Array3::<f32>::zeros((2, 3, 3)));
    let dir = scratch("oob");
    let mut w = create_writer(dir.join("r").to_str().unwrap(), SaveFormat::Tiff).unwrap();
    // end > nz is rejected, and nothing is written.
    assert!(w.write_chunk(&vol, 0, 5).is_err());
    assert!(w.write_chunk(&vol, 2, 1).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn write_chunk_writes_only_the_requested_range() {
    let vol = Volume::new(Array3::<f32>::zeros((4, 2, 2)));
    let dir = scratch("range");
    let prefix = dir.join("recon");
    let mut w = create_writer(prefix.to_str().unwrap(), SaveFormat::Tiff).unwrap();
    w.write_chunk(&vol, 1, 3).unwrap();

    // Only slices 1 and 2 exist; the filename index is the global slice index.
    assert!(!dir.join("recon_00000.tiff").exists());
    assert!(dir.join("recon_00001.tiff").exists());
    assert!(dir.join("recon_00002.tiff").exists());
    assert!(!dir.join("recon_00003.tiff").exists());
    std::fs::remove_dir_all(&dir).ok();
}
