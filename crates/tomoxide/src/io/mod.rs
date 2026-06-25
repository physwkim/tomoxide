//! # tomoxide-io
//!
//! Dataset readers and writers. The primary format is **DXchange** HDF5 (the
//! APS/synchrotron convention used by both tomopy and tomocupy); TIFF, HDF5,
//! and zarr outputs mirror tomocupy's `--save-format`. The DXchange reader and
//! all three writers ([`SaveFormat`]) are implemented; see `docs/PORTING.md` §G.
#![forbid(unsafe_code)]

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use ndarray::{Array3, Axis, Slice};
use rust_hdf5::{ByteOrder, DatatypeMessage, H5Dataset, H5File, Hdf5Error, VarLenUnicode};
use tiff::encoder::{colortype::Gray32Float, TiffEncoder};
use crate::data::{Dataset, Frames, Layout, Tomo, Volume};
use crate::error::{Error, Result};

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

    /// Read a detector-row chunk `[row0, row1)`: the projections and any
    /// flat/dark frames sliced to those rows (axis 1), with the full `theta`.
    /// This is the out-of-core entry point — only the chunk's rows are read from
    /// disk. The default is unsupported (in-memory readers should use
    /// [`read_all`](DatasetReader::read_all)).
    fn read_chunk(&mut self, _row0: usize, _row1: usize) -> Result<Dataset<f32>> {
        Err(Error::Io(
            "read_chunk: out-of-core reads not supported by this reader".into(),
        ))
    }
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

    fn read_chunk(&mut self, row0: usize, row1: usize) -> Result<Dataset<f32>> {
        let [nproj, nz, nx] = self.data_shape()?;
        if row0 > row1 || row1 > nz {
            return Err(Error::InvalidParam(format!(
                "read_chunk rows [{row0}, {row1}) out of range for nz={nz}"
            )));
        }
        let rows = row1 - row0;

        // Hyperslab `[:, row0:row1, :]` of each [n, nz, nx] dataset.
        let slab = |ds: &H5Dataset, n: usize| -> Result<Array3<f32>> {
            let v = read_f32_slice(ds, &[0, row0, 0], &[n, rows, nx])?;
            Array3::from_shape_vec((n, rows, nx), v).map_err(|e| Error::ShapeMismatch {
                expected: format!("[{n}, {rows}, {nx}]"),
                found: e.to_string(),
            })
        };

        let data = slab(&self.required(dxchange::DATA)?, nproj)?;
        let flat = match self.optional(dxchange::DATA_WHITE)? {
            Some(ds) => {
                let nf = ds.shape().first().copied().unwrap_or(0);
                Some(Frames::new(slab(&ds, nf)?))
            }
            None => None,
        };
        let dark = match self.optional(dxchange::DATA_DARK)? {
            Some(ds) => {
                let nd = ds.shape().first().copied().unwrap_or(0);
                Some(Frames::new(slab(&ds, nd)?))
            }
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

/// Read an N-dimensional hyperslab (`starts`/`counts`) of a numeric HDF5
/// dataset as `Vec<f32>` (C-order), casting from its on-disk dtype. Mirrors
/// [`read_f32_vec`]'s dtype dispatch but via `read_slice` for out-of-core
/// chunked reads. Kept as its own function (not folded with `read_f32_vec`) so
/// the whole-dataset path stays untouched.
fn read_f32_slice(ds: &H5Dataset, starts: &[usize], counts: &[usize]) -> Result<Vec<f32>> {
    let dt = ds
        .datatype()
        .map_err(|e| Error::Io(format!("datatype: {e}")))?;
    let raw = |e: Hdf5Error| Error::Io(format!("read: {e}"));
    // cast<T> reads the slice as on-disk type T and casts each element to f32.
    macro_rules! cast {
        ($t:ty) => {
            ds.read_slice::<$t>(starts, counts)
                .map_err(raw)?
                .into_iter()
                .map(|x| x as f32)
                .collect()
        };
    }
    let v: Vec<f32> = match dt {
        DatatypeMessage::FloatingPoint {
            size: 4, byte_order, ..
        } => {
            ensure_le(byte_order)?;
            ds.read_slice::<f32>(starts, counts).map_err(raw)?
        }
        DatatypeMessage::FloatingPoint {
            size: 8, byte_order, ..
        } => {
            ensure_le(byte_order)?;
            cast!(f64)
        }
        DatatypeMessage::FixedPoint {
            size: 1, signed: false, ..
        } => cast!(u8),
        DatatypeMessage::FixedPoint {
            size: 1, signed: true, ..
        } => cast!(i8),
        DatatypeMessage::FixedPoint {
            size: 2, signed: false, byte_order, ..
        } => {
            ensure_le(byte_order)?;
            cast!(u16)
        }
        DatatypeMessage::FixedPoint {
            size: 2, signed: true, byte_order, ..
        } => {
            ensure_le(byte_order)?;
            cast!(i16)
        }
        DatatypeMessage::FixedPoint {
            size: 4, signed: false, byte_order, ..
        } => {
            ensure_le(byte_order)?;
            cast!(u32)
        }
        DatatypeMessage::FixedPoint {
            size: 4, signed: true, byte_order, ..
        } => {
            ensure_le(byte_order)?;
            cast!(i32)
        }
        DatatypeMessage::FixedPoint {
            size: 8, signed: false, byte_order, ..
        } => {
            ensure_le(byte_order)?;
            cast!(u64)
        }
        DatatypeMessage::FixedPoint {
            size: 8, signed: true, byte_order, ..
        } => {
            ensure_le(byte_order)?;
            cast!(i64)
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
/// [`SaveFormat::Tiff`], [`SaveFormat::H5`], and [`SaveFormat::Zarr`] are
/// implemented. `path` is the output **base**, and each writer appends its own
/// suffix:
/// - TIFF — slice `i` → `{path}_{i:05}.tiff` (tomocupy `dataio/writer.py:281`,
///   `{fnameout}_{fid:05}.tiff`).
/// - H5 — a single `{path}.h5` file holding one `/exchange/data` dataset
///   (tomocupy `dataio/writer.py` `h5nolinks`; `fnameout += '.h5'`).
/// - Zarr — a `{path}.zarr` directory store with an `exchange/data` array
///   (uncompressed, spec-compliant Zarr v2; one chunk file per z-slice).
///
/// The parent directory of `path` is created if missing.
pub fn create_writer(path: &str, format: SaveFormat) -> Result<Box<dyn VolumeWriter>> {
    match format {
        SaveFormat::Tiff => Ok(Box::new(TiffWriter::new(path)?)),
        SaveFormat::H5 => Ok(Box::new(H5Writer::new(path)?)),
        SaveFormat::Zarr => Ok(Box::new(ZarrWriter::new(path)?)),
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

/// Pure-Rust Zarr v2 reconstruction writer (spec-compliant `DirectoryStore`).
///
/// tomocupy's zarr output (`dataio/writer.py` `initialize_zarr` +
/// `downsampleZarr`) is a **Blosc-compressed multiscale NGFF pyramid**
/// (`Blosc(cname=blosclz, clevel=5, shuffle=2)`, default chunk `8,64,64`).
/// Reproducing those exact bytes would need a zarr crate plus a Blosc
/// C-binding; tomoxide keeps the I/O stack pure-Rust with no new C dependency
/// (the same reason the reader/writer use `rust-hdf5`). So this writes an
/// **uncompressed, single-scale** Zarr v2 array that is fully spec-compliant
/// and readable by the Python `zarr` library:
///
/// ```text
/// {path}.zarr/
///   .zgroup                      {"zarr_format": 2}
///   exchange/.zgroup             {"zarr_format": 2}
///   exchange/data/.zarray        shape/chunks/dtype metadata
///   exchange/data/.zattrs        axes/description/units (mirrors the H5 writer)
///   exchange/data/{z}.0.0        one raw little-endian f32 chunk per z-slice
/// ```
///
/// Chunks are `[1, ny, nx]` — one z-slice per chunk file, like the h5 variant's
/// `chunks=(1, n, n)` — which makes the streaming `write_chunk([start, end))`
/// a plain per-slice file write with no partial-chunk read-modify-write.
/// Blosc compression and the NGFF multiscale pyramid are a documented deferral;
/// the stored sample values are identical regardless.
struct ZarrWriter {
    /// Resolved `{path}.zarr` store root.
    root: std::path::PathBuf,
    /// Full array extents `(nz, ny, nx)`, fixed on the first `write_chunk`.
    dims: Option<(usize, usize, usize)>,
}

impl ZarrWriter {
    fn new(path: &str) -> Result<Self> {
        // Append `.zarr` to the base, mirroring how TiffWriter/H5Writer treat
        // `path` as a base and add their own suffix (tomocupy `fnameout += '.zarr'`).
        let root = if path.ends_with(".zarr") {
            std::path::PathBuf::from(path)
        } else {
            std::path::PathBuf::from(format!("{path}.zarr"))
        };
        if let Some(parent) = root.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::Io(format!("create dir {}: {e}", parent.display())))?;
            }
        }
        Ok(Self { root, dims: None })
    }

    /// Create the store directories and write the `.zgroup`/`.zarray`/`.zattrs`
    /// metadata for an `exchange/data` array of shape `(nz, ny, nx)`.
    fn init_store(&self, nz: usize, ny: usize, nx: usize) -> Result<()> {
        let data_dir = self.root.join("exchange").join("data");
        std::fs::create_dir_all(&data_dir)
            .map_err(|e| Error::Io(format!("create dir {}: {e}", data_dir.display())))?;
        let write = |rel: std::path::PathBuf, body: String| -> Result<()> {
            std::fs::write(&rel, body)
                .map_err(|e| Error::Io(format!("write {}: {e}", rel.display())))
        };
        // Root + group markers.
        let zgroup = "{\n    \"zarr_format\": 2\n}".to_string();
        write(self.root.join(".zgroup"), zgroup.clone())?;
        write(self.root.join("exchange").join(".zgroup"), zgroup)?;
        // Array metadata: little-endian f32 (`<f4`), uncompressed, C order,
        // one z-slice per chunk. Keys sorted as the Python writer emits them.
        let zarray = format!(
            "{{\n    \"chunks\": [1, {ny}, {nx}],\n    \"compressor\": null,\n    \
             \"dtype\": \"<f4\",\n    \"fill_value\": 0.0,\n    \"filters\": null,\n    \
             \"order\": \"C\",\n    \"shape\": [{nz}, {ny}, {nx}],\n    \
             \"zarr_format\": 2\n}}"
        );
        write(data_dir.join(".zarray"), zarray)?;
        // Mirror the H5 writer's dataset attributes.
        let zattrs = "{\n    \"axes\": \"z:y:x\",\n    \"description\": \"ReconData\",\n    \
                      \"units\": \"counts\"\n}"
            .to_string();
        write(data_dir.join(".zattrs"), zattrs)?;
        Ok(())
    }
}

impl VolumeWriter for ZarrWriter {
    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, end: usize) -> Result<()> {
        let (nz, ny, nx) = vol.dims();
        if start > end || end > nz {
            return Err(Error::InvalidParam(format!(
                "write_chunk: slice range [{start}, {end}) out of bounds for {nz} slices"
            )));
        }
        // The store is sized to the first volume's full extents; every later
        // chunk must come from a volume of the same shape (the array shape in
        // `.zarray` is fixed once written), mirroring H5Writer.
        if self.dims.is_none() {
            self.init_store(nz, ny, nx)?;
            self.dims = Some((nz, ny, nx));
        }
        let dims = self.dims.unwrap();
        if dims != (nz, ny, nx) {
            return Err(Error::InvalidParam(format!(
                "write_chunk: volume dims {:?} differ from the created store {:?}",
                (nz, ny, nx),
                dims
            )));
        }
        let data_dir = self.root.join("exchange").join("data");
        // One chunk file per z-slice: chunk grid coord (z, 0, 0) → "z.0.0".
        for z in start..end {
            let slice = vol.array.index_axis(Axis(0), z);
            // C-order (y-major, x fastest) little-endian f32 — exactly `<f4`.
            let mut bytes = Vec::with_capacity(ny * nx * 4);
            for &v in slice.iter() {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let chunk = data_dir.join(format!("{z}.0.0"));
            std::fs::write(&chunk, &bytes)
                .map_err(|e| Error::Io(format!("write {}: {e}", chunk.display())))?;
        }
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
    fn zarr_writer_roundtrips_a_spec_compliant_store() {
        // Unique temp store root (no tempfile dev-dep); clean before and after.
        let base = std::env::temp_dir().join("tomoxide_zarr_writer_roundtrip");
        let root = base.with_extension("zarr");
        let _ = std::fs::remove_dir_all(&root);

        // 3 slices of 2x2 with distinct, C-order-distinguishable values.
        let arr = Array3::from_shape_fn((3, 2, 2), |(z, y, x)| (z * 100 + y * 10 + x) as f32);
        let vol = Volume::new(arr.clone());

        // Stream it in two calls to exercise the per-slice chunk writes.
        let mut w = create_writer(base.to_str().unwrap(), SaveFormat::Zarr).unwrap();
        w.write_chunk(&vol, 0, 2).unwrap();
        w.write_chunk(&vol, 2, 3).unwrap();

        // Structure: group markers + array metadata exist.
        assert!(root.join(".zgroup").is_file());
        assert!(root.join("exchange").join(".zgroup").is_file());
        let data = root.join("exchange").join("data");
        let zarray = std::fs::read_to_string(data.join(".zarray")).unwrap();
        assert!(zarray.contains("\"shape\": [3, 2, 2]"), "{zarray}");
        assert!(zarray.contains("\"chunks\": [1, 2, 2]"), "{zarray}");
        assert!(zarray.contains("\"dtype\": \"<f4\""), "{zarray}");
        assert!(zarray.contains("\"compressor\": null"), "{zarray}");
        assert!(data.join(".zattrs").is_file());

        // One chunk file per z-slice; reassemble and compare to the input.
        for z in 0..3 {
            let bytes = std::fs::read(data.join(format!("{z}.0.0"))).unwrap();
            assert_eq!(bytes.len(), 2 * 2 * 4, "chunk {z} byte length");
            let got: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let want: Vec<f32> = arr.index_axis(Axis(0), z).iter().copied().collect();
            assert_eq!(got, want, "slice {z} round-trip");
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn zarr_writer_rejects_changed_dims() {
        let base = std::env::temp_dir().join("tomoxide_zarr_writer_dims");
        let root = base.with_extension("zarr");
        let _ = std::fs::remove_dir_all(&root);

        let mut w = create_writer(base.to_str().unwrap(), SaveFormat::Zarr).unwrap();
        let a = Volume::new(Array3::zeros((2, 4, 4)));
        w.write_chunk(&a, 0, 2).unwrap();
        // A second volume with different extents must be rejected (the store
        // shape is fixed by the first write), mirroring H5Writer.
        let b = Volume::new(Array3::zeros((2, 8, 8)));
        assert!(matches!(
            w.write_chunk(&b, 0, 2),
            Err(Error::InvalidParam(_))
        ));

        let _ = std::fs::remove_dir_all(&root);
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
