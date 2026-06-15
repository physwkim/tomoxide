//! # tomoxide-io
//!
//! Dataset readers and writers. The primary format is **DXchange** HDF5 (the
//! APS/synchrotron convention used by both tomopy and tomocupy); TIFF and zarr
//! outputs mirror tomocupy's `--save-format`. All back-ends are stubs in this
//! scaffold; see `docs/PORTING.md` §G.
#![forbid(unsafe_code)]

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use ndarray::{Array3, Axis, Slice};
use rust_hdf5::{ByteOrder, DatatypeMessage, H5Dataset, H5File, Hdf5Error, VarLenUnicode};
use tiff::encoder::{colortype::Gray32Float, TiffEncoder};
use tomoxide_core::data::{Dataset, Frames, Layout, Tomo, Volume};
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

/// Open a DXchange HDF5 file for reading.
///
/// Backed by the pure-Rust `rust-hdf5` crate (no libhdf5). Reads the standard
/// DXchange layout (`/exchange/{data,data_white,data_dark,theta}`); flat, dark,
/// and theta are optional. Ports the read semantics of tomocupy
/// `dataio/reader.py:59` `Reader`.
pub fn open_dxchange(path: &str) -> Result<Box<dyn DatasetReader>> {
    Ok(Box::new(H5DxchangeReader::open(path)?))
}

/// DXchange HDF5 reader over a pure-Rust [`H5File`].
struct H5DxchangeReader {
    file: H5File,
}

impl H5DxchangeReader {
    fn open(path: &str) -> Result<Self> {
        let file = H5File::open(path).map_err(|e| Error::Io(format!("open {path}: {e}")))?;
        Ok(Self { file })
    }

    /// A dataset that must be present.
    fn required(&self, path: &str) -> Result<H5Dataset> {
        self.optional(path)?
            .ok_or_else(|| Error::Io(format!("dataset {path}: not found")))
    }

    /// A dataset that may be absent (`NotFound` → `None`); other errors bubble.
    fn optional(&self, path: &str) -> Result<Option<H5Dataset>> {
        // rust-hdf5 keys nested datasets without a leading slash
        // ("exchange/data"); the DXchange constants are absolute paths.
        let key = path.strip_prefix('/').unwrap_or(path);
        match self.file.dataset(key) {
            Ok(ds) => Ok(Some(ds)),
            Err(Hdf5Error::NotFound(_)) => Ok(None),
            Err(e) => Err(Error::Io(format!("dataset {path}: {e}"))),
        }
    }

    /// `/exchange/data` shape `[nproj, nz, nx]`, validated as 3-D.
    fn data_shape(&self) -> Result<[usize; 3]> {
        let shape = self.required(dxchange::DATA)?.shape();
        match shape.as_slice() {
            &[nproj, nz, nx] => Ok([nproj, nz, nx]),
            other => Err(Error::ShapeMismatch {
                expected: "3-D [nproj, nz, nx]".into(),
                found: format!("{other:?}"),
            }),
        }
    }
}

impl DatasetReader for H5DxchangeReader {
    fn read_sizes(&mut self) -> Result<(usize, usize, usize, usize, usize)> {
        let [nproj, nz, nx] = self.data_shape()?;
        let nflat = match self.optional(dxchange::DATA_WHITE)? {
            Some(ds) => ds.shape().first().copied().unwrap_or(0),
            None => 0,
        };
        let ndark = match self.optional(dxchange::DATA_DARK)? {
            Some(ds) => ds.shape().first().copied().unwrap_or(0),
            None => 0,
        };
        Ok((nproj, nz, nx, nflat, ndark))
    }

    fn read_theta(&mut self) -> Result<Vec<f32>> {
        // tomocupy reader.py:313 — /exchange/theta is in DEGREES; convert to
        // radians as `deg.astype(f32) / 180 * pi`. If absent, linspace(0, pi,
        // nproj) over the projection axis (endpoint-inclusive, like numpy).
        if let Some(ds) = self.optional(dxchange::THETA)? {
            let deg = read_f32_vec(&ds)?;
            Ok(deg
                .into_iter()
                .map(|d| d / 180.0 * std::f32::consts::PI)
                .collect())
        } else {
            let nproj = self.data_shape()?[0];
            Ok(linspace_inclusive(nproj))
        }
    }

    fn read_all(&mut self) -> Result<Dataset<f32>> {
        let [nproj, nz, nx] = self.data_shape()?;
        let data = read_f32_array(&self.required(dxchange::DATA)?, (nproj, nz, nx))?;

        let flat = match self.optional(dxchange::DATA_WHITE)? {
            Some(ds) => Some(Frames::new(read_frames(&ds, nz, nx)?)),
            None => None,
        };
        let dark = match self.optional(dxchange::DATA_DARK)? {
            Some(ds) => Some(Frames::new(read_frames(&ds, nz, nx)?)),
            None => None,
        };
        let theta = self.read_theta()?;

        Ok(Dataset {
            data: Tomo::new(data, Layout::Projection),
            flat,
            dark,
            theta,
        })
    }
}

/// `np.linspace(0, pi, n)` in f32: endpoint-inclusive, matching tomocupy's
/// absent-theta fallback. `n == 1` is the single start point `[0.0]`.
fn linspace_inclusive(n: usize) -> Vec<f32> {
    match n {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => {
            let step = std::f32::consts::PI / (n - 1) as f32;
            (0..n).map(|i| i as f32 * step).collect()
        }
    }
}

/// Read a numeric dataset and reshape to a `[frame, row, col]` array.
fn read_frames(ds: &H5Dataset, nz: usize, nx: usize) -> Result<Array3<f32>> {
    let nframe = ds.shape().first().copied().unwrap_or(0);
    read_f32_array(ds, (nframe, nz, nx))
}

/// Read a numeric dataset into a flat `f32` vector, casting from its on-disk
/// dtype, then reshape (C-order) into `dims`.
fn read_f32_array(ds: &H5Dataset, dims: (usize, usize, usize)) -> Result<Array3<f32>> {
    let v = read_f32_vec(ds)?;
    Array3::from_shape_vec(dims, v).map_err(|e| Error::ShapeMismatch {
        expected: format!("{dims:?}"),
        found: e.to_string(),
    })
}

/// Read a numeric HDF5 dataset as `Vec<f32>`, converting from its on-disk
/// integer/float dtype. `rust-hdf5`'s `read_raw::<T>` byte-copies (no numeric
/// conversion), so the read type must match the on-disk element exactly; we
/// dispatch on the datatype and cast each element to `f32`.
fn read_f32_vec(ds: &H5Dataset) -> Result<Vec<f32>> {
    let dt = ds
        .datatype()
        .map_err(|e| Error::Io(format!("datatype: {e}")))?;
    let raw = |e: Hdf5Error| Error::Io(format!("read: {e}"));
    let v: Vec<f32> = match dt {
        DatatypeMessage::FloatingPoint {
            size: 4,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            ds.read_raw::<f32>().map_err(raw)?
        }
        DatatypeMessage::FloatingPoint {
            size: 8,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            ds.read_raw::<f64>()
                .map_err(raw)?
                .into_iter()
                .map(|x| x as f32)
                .collect()
        }
        DatatypeMessage::FixedPoint {
            size: 1,
            signed: false,
            ..
        } => ds
            .read_raw::<u8>()
            .map_err(raw)?
            .into_iter()
            .map(|x| x as f32)
            .collect(),
        DatatypeMessage::FixedPoint {
            size: 1,
            signed: true,
            ..
        } => ds
            .read_raw::<i8>()
            .map_err(raw)?
            .into_iter()
            .map(|x| x as f32)
            .collect(),
        DatatypeMessage::FixedPoint {
            size: 2,
            signed: false,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            ds.read_raw::<u16>()
                .map_err(raw)?
                .into_iter()
                .map(|x| x as f32)
                .collect()
        }
        DatatypeMessage::FixedPoint {
            size: 2,
            signed: true,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            ds.read_raw::<i16>()
                .map_err(raw)?
                .into_iter()
                .map(|x| x as f32)
                .collect()
        }
        DatatypeMessage::FixedPoint {
            size: 4,
            signed: false,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            ds.read_raw::<u32>()
                .map_err(raw)?
                .into_iter()
                .map(|x| x as f32)
                .collect()
        }
        DatatypeMessage::FixedPoint {
            size: 4,
            signed: true,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            ds.read_raw::<i32>()
                .map_err(raw)?
                .into_iter()
                .map(|x| x as f32)
                .collect()
        }
        DatatypeMessage::FixedPoint {
            size: 8,
            signed: false,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            ds.read_raw::<u64>()
                .map_err(raw)?
                .into_iter()
                .map(|x| x as f32)
                .collect()
        }
        DatatypeMessage::FixedPoint {
            size: 8,
            signed: true,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            ds.read_raw::<i64>()
                .map_err(raw)?
                .into_iter()
                .map(|x| x as f32)
                .collect()
        }
        other => {
            return Err(Error::Io(format!(
                "unsupported HDF5 datatype for numeric read: {other:?}"
            )))
        }
    };
    Ok(v)
}

/// Guard against silently mis-reading a big-endian dataset: `read_raw`
/// byte-copies without swapping, so a BE file on this LE host would be garbage.
/// DXchange data is little-endian in practice; reject BE explicitly.
fn ensure_le(byte_order: ByteOrder) -> Result<()> {
    match byte_order {
        ByteOrder::LittleEndian => Ok(()),
        ByteOrder::BigEndian => Err(Error::Io(
            "big-endian HDF5 datasets are not supported (read_raw does not byte-swap)".into(),
        )),
    }
}

/// Create a writer for the given output format.
///
/// [`SaveFormat::Tiff`] and [`SaveFormat::H5`] are implemented; `Zarr` remains a
/// stub. `path` is the output **base**, and each writer appends its own suffix:
/// - TIFF — slice `i` → `{path}_{i:05}.tiff` (tomocupy `dataio/writer.py:281`,
///   `{fnameout}_{fid:05}.tiff`).
/// - H5 — a single `{path}.h5` file holding one `/exchange/data` dataset
///   (tomocupy `dataio/writer.py` `h5nolinks`; `fnameout += '.h5'`).
///
/// The parent directory of `path` is created if missing.
pub fn create_writer(path: &str, format: SaveFormat) -> Result<Box<dyn VolumeWriter>> {
    match format {
        SaveFormat::Tiff => Ok(Box::new(TiffWriter::new(path)?)),
        SaveFormat::H5 => Ok(Box::new(H5Writer::new(path)?)),
        SaveFormat::Zarr => Err(Error::todo(
            "io::create_writer (zarr)",
            "tomocupy dataio/writer.py:294",
        )),
    }
}

/// Per-slice 32-bit-float TIFF writer (tomocupy `dataio/writer.py:281`).
struct TiffWriter {
    /// Filename prefix; slice `i` → `{prefix}_{i:05}.tiff`.
    prefix: String,
}

impl TiffWriter {
    fn new(path: &str) -> Result<Self> {
        // Create the prefix's parent directory (tomocupy os.makedirs).
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::Io(format!("create dir {}: {e}", parent.display())))?;
            }
        }
        Ok(Self {
            prefix: path.to_string(),
        })
    }
}

impl VolumeWriter for TiffWriter {
    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, end: usize) -> Result<()> {
        let (nz, ny, nx) = vol.dims();
        if start > end || end > nz {
            return Err(Error::InvalidParam(format!(
                "write_chunk: slice range [{start}, {end}) out of bounds for {nz} slices"
            )));
        }
        for i in start..end {
            // Slice i is [y, x], contiguous row-major (x fastest) in z-major
            // Volume storage — exactly TIFF's width=nx, height=ny order.
            let slice = vol.array.index_axis(Axis(0), i);
            let buf: Vec<f32> = slice.iter().copied().collect();

            let fname = format!("{}_{i:05}.tiff", self.prefix);
            let file =
                File::create(&fname).map_err(|e| Error::Io(format!("create {fname}: {e}")))?;
            let mut enc = TiffEncoder::new(BufWriter::new(file))
                .map_err(|e| Error::Io(format!("tiff encoder {fname}: {e}")))?;
            enc.write_image::<Gray32Float>(nx as u32, ny as u32, &buf)
                .map_err(|e| Error::Io(format!("tiff write {fname}: {e}")))?;
        }
        Ok(())
    }
}

/// Single-file HDF5 reconstruction writer (tomocupy `dataio/writer.py`,
/// `h5nolinks` variant).
///
/// Writes one **contiguous** `/exchange/data` dataset of shape `[nz, ny, nx]`
/// (float32) under an `exchange` group, carrying tomocupy's
/// `axes`/`description`/`units` attributes. The dataset is sized on the first
/// `write_chunk` from the volume's full extents; each chunk fills its
/// `[start, end)` slice range via an HDF5 hyperslab, and the file is flushed so
/// a streaming caller's partial output is durable. `path` is the output base —
/// `.h5` is appended if absent (mirroring tomocupy `fnameout += '.h5'`).
///
/// Contiguous (not chunked) layout is required: `write_slice` hyperslabs are
/// only valid on contiguous datasets, and `h5nolinks` output is uncompressed.
struct H5Writer {
    /// Resolved `.h5` output path.
    path: std::path::PathBuf,
    /// File + dataset, created lazily on the first `write_chunk`.
    state: Option<H5WriteState>,
}

/// The open file and its `/exchange/data` dataset, fixed to the first volume's
/// extents.
struct H5WriteState {
    /// Kept open for `flush`; closing it would invalidate `dataset`.
    file: H5File,
    dataset: H5Dataset,
    dims: (usize, usize, usize),
}

impl H5Writer {
    fn new(path: &str) -> Result<Self> {
        // Append `.h5` to the base (tomocupy `fnameout += '.h5'`), mirroring the
        // way TiffWriter treats `path` as a base and adds its own suffix.
        let out = if path.ends_with(".h5") {
            std::path::PathBuf::from(path)
        } else {
            std::path::PathBuf::from(format!("{path}.h5"))
        };
        if let Some(parent) = out.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::Io(format!("create dir {}: {e}", parent.display())))?;
            }
        }
        Ok(Self {
            path: out,
            state: None,
        })
    }

    /// Create `{path}.h5` with an `exchange/data` dataset of shape `(nz, ny, nx)`.
    fn create_dataset(&self, nz: usize, ny: usize, nx: usize) -> Result<H5WriteState> {
        let file = H5File::create(&self.path)
            .map_err(|e| Error::Io(format!("create {}: {e}", self.path.display())))?;
        let group = file
            .create_group("exchange")
            .map_err(|e| Error::Io(format!("create group exchange: {e}")))?;
        // No `.chunk()` → contiguous, so write_slice hyperslabs are allowed.
        let dataset = group
            .new_dataset::<f32>()
            .shape([nz, ny, nx])
            .create("data")
            .map_err(|e| Error::Io(format!("create dataset /exchange/data: {e}")))?;
        for (name, value) in [
            ("axes", "z:y:x"),
            ("description", "ReconData"),
            ("units", "counts"),
        ] {
            dataset
                .new_attr::<VarLenUnicode>()
                .shape(())
                .create(name)
                .map_err(|e| Error::Io(format!("create attr {name}: {e}")))?
                .write_string(value)
                .map_err(|e| Error::Io(format!("write attr {name}: {e}")))?;
        }
        Ok(H5WriteState {
            file,
            dataset,
            dims: (nz, ny, nx),
        })
    }
}

impl VolumeWriter for H5Writer {
    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, end: usize) -> Result<()> {
        let (nz, ny, nx) = vol.dims();
        if start > end || end > nz {
            return Err(Error::InvalidParam(format!(
                "write_chunk: slice range [{start}, {end}) out of bounds for {nz} slices"
            )));
        }
        // The dataset is fixed to the first volume's full extents; every later
        // chunk must come from a volume of the same shape (tomocupy pre-allocates
        // the whole `/exchange/data` before filling chunks).
        if self.state.is_none() {
            self.state = Some(self.create_dataset(nz, ny, nx)?);
        }
        let state = self.state.as_ref().unwrap();
        if state.dims != (nz, ny, nx) {
            return Err(Error::InvalidParam(format!(
                "write_chunk: volume dims {:?} differ from the created dataset {:?}",
                (nz, ny, nx),
                state.dims
            )));
        }
        if start == end {
            return Ok(());
        }
        // Row-major [z, y, x] slab for rows [start, end), written into the same
        // rows of the dataset (tomocupy `dset[st:end] = rec`).
        let slab: Vec<f32> = vol
            .array
            .slice_axis(Axis(0), Slice::from(start..end))
            .iter()
            .copied()
            .collect();
        state
            .dataset
            .write_slice(&[start, 0, 0], &[end - start, ny, nx], &slab)
            .map_err(|e| Error::Io(format!("write /exchange/data[{start}..{end}]: {e}")))?;
        state
            .file
            .flush()
            .map_err(|e| Error::Io(format!("flush {}: {e}", self.path.display())))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dxchange_paths_are_stable() {
        assert_eq!(dxchange::DATA, "/exchange/data");
    }

    #[test]
    fn open_missing_file_is_io_error() {
        // The reader is implemented now: a missing file is an I/O error, not
        // the old NotImplemented stub.
        assert!(matches!(
            open_dxchange("definitely-not-a-real-file.h5"),
            Err(Error::Io(_))
        ));
    }

    #[test]
    fn linspace_inclusive_matches_numpy() {
        assert_eq!(linspace_inclusive(0), Vec::<f32>::new());
        assert_eq!(linspace_inclusive(1), vec![0.0]);
        let a = linspace_inclusive(5);
        assert_eq!(a.len(), 5);
        assert_eq!(a[0], 0.0);
        // endpoint-inclusive: last sample is exactly pi.
        assert_eq!(a[4], std::f32::consts::PI);
        assert!((a[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }
}
