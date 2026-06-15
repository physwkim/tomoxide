//! # tomoxide-io
//!
//! Dataset readers and writers. The primary format is **DXchange** HDF5 (the
//! APS/synchrotron convention used by both tomopy and tomocupy); TIFF and zarr
//! outputs mirror tomocupy's `--save-format`. All back-ends are stubs in this
//! scaffold; see `docs/PORTING.md` §G.
#![forbid(unsafe_code)]

use tomoxide_core::data::{Dataset, Volume};
use tomoxide_core::error::{Error, Result};

/// DXchange HDF5 dataset paths (tomocupy `dataio/reader.py`).
pub mod dxchange {
    /// Projections, shape `[nproj, nz, nx]`.
    pub const DATA: &str = "/exchange/data";
    /// Flat (white) fields, shape `[nflat, nz, nx]`.
    pub const DATA_WHITE: &str = "/exchange/data_white";
    /// Dark fields, shape `[ndark, nz, nx]`.
    pub const DATA_DARK: &str = "/exchange/data_dark";
    /// Projection angles `[nproj]` (optional — generated uniformly if absent).
    pub const THETA: &str = "/exchange/theta";
}

/// A chunked dataset reader (port of tomocupy `dataio/reader.py:59`).
///
/// The streaming pipeline pulls projection/sinogram chunks through this trait;
/// an implementation owns the open file handle and chunking metadata.
pub trait DatasetReader {
    /// Read sizes/metadata: `(nproj, nz, nx, nflat, ndark)`.
    fn read_sizes(&mut self) -> Result<(usize, usize, usize, usize, usize)>;
    /// Read the projection angles (radians), generating uniform ones if absent.
    fn read_theta(&mut self) -> Result<Vec<f32>>;
    /// Read the whole dataset into memory (the "full"/"steps" entry point).
    fn read_all(&mut self) -> Result<Dataset<f32>>;
}

/// A reconstruction writer (port of tomocupy `dataio/writer.py:73`).
pub trait VolumeWriter {
    /// Write a contiguous chunk of slices `[start, end)` of the volume.
    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, end: usize) -> Result<()>;
}

/// Output container format (tomocupy `--save-format`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SaveFormat {
    /// One TIFF per slice.
    #[default]
    Tiff,
    /// Single HDF5 file.
    H5,
    /// Zarr store.
    Zarr,
}

/// Open a DXchange HDF5 file for reading (stub).
pub fn open_dxchange(_path: &str) -> Result<Box<dyn DatasetReader>> {
    Err(Error::todo(
        "io::open_dxchange (HDF5 reader)",
        "tomocupy dataio/reader.py:59",
    ))
}

/// Create a writer for the given output format (stub).
pub fn create_writer(_path: &str, _format: SaveFormat) -> Result<Box<dyn VolumeWriter>> {
    Err(Error::todo(
        "io::create_writer",
        "tomocupy dataio/writer.py:103 (tiff/h5/zarr)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dxchange_paths_are_stable() {
        assert_eq!(dxchange::DATA, "/exchange/data");
        assert!(matches!(
            open_dxchange("x.h5"),
            Err(Error::NotImplemented { .. })
        ));
    }
}
