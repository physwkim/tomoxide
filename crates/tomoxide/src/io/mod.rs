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

use crate::data::{Dataset, Frames, Layout, Tomo, Volume};
use crate::error::{Error, Result};
use ndarray::{Array3, Axis};
use rust_hdf5::{ByteOrder, DatatypeMessage, H5Dataset, H5File, Hdf5Error, VarLenUnicode};
use tiff::encoder::{colortype::Gray32Float, TiffEncoder};

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

    /// Read a detector-row chunk's projections **directly into** `data_out` — the
    /// caller's (page-locked) host buffer — returning the chunk's flat/dark frames
    /// and `theta` as [`ChunkAux`]. This is the out-of-core entry point for
    /// backends that upload raw chunks ([`Backend::wants_raw_chunks`](crate::backend::Backend::wants_raw_chunks)):
    /// the projection hyperslab lands straight in the pinned staging buffer with
    /// no intervening owned allocation or copy. `data_out.len()` must equal
    /// `nproj * (row1 - row0) * nx`. The default is unsupported.
    fn read_chunk_into(
        &mut self,
        _row0: usize,
        _row1: usize,
        _data_out: &mut [f32],
    ) -> Result<ChunkAux> {
        Err(Error::Io(
            "read_chunk_into: out-of-core reads not supported by this reader".into(),
        ))
    }
}

/// Side data accompanying a raw projection chunk read via
/// [`DatasetReader::read_chunk_into`]: the chunk's flat/dark frames (sliced to
/// the same detector rows) and the full projection angles. The projections
/// themselves are read straight into the caller's buffer, not carried here.
pub struct ChunkAux {
    /// Flat (white/open-beam) frames for this chunk's rows, if present.
    pub flat: Option<Frames<f32>>,
    /// Dark frames for this chunk's rows, if present.
    pub dark: Option<Frames<f32>>,
    /// Projection angles in radians, length = number of angles.
    pub theta: Vec<f32>,
}

/// Restrict any [`DatasetReader`] to the detector-row band `[r0, r1)`.
///
/// Wrapped, the dataset appears to be `r1 − r0` rows tall: `read_sizes`
/// reports the band height, and `read_all` / `read_chunk` / `read_chunk_into`
/// address rows relative to `r0` (a chunk `[a, b)` reads the underlying rows
/// `[r0+a, r0+b)`). `theta` passes through; flat/dark frames are sliced to the
/// same rows by the inner reader's own chunk reads.
///
/// This is the banded-preview adapter (docs/GUI.md §6 #5): row-coupled prep
/// (phase retrieval) needs detector rows around the slice of interest, so a
/// preview reconstructs the band `[z − m, z + m]` through this wrapper and
/// displays the center row — without reading the whole file. `m` comes from
/// [`crate::prep::phase::margin_rows`].
pub struct RowBandReader {
    inner: Box<dyn DatasetReader>,
    r0: usize,
    r1: usize,
}

impl RowBandReader {
    /// Wrap `inner`, restricting it to rows `[r0, r1)` (`r1` clamped to the
    /// dataset height). Errors when the clamped band is empty.
    pub fn new(mut inner: Box<dyn DatasetReader>, r0: usize, r1: usize) -> Result<Self> {
        let (_nproj, nz, _nx, _nflat, _ndark) = inner.read_sizes()?;
        let r1 = r1.min(nz);
        if r0 >= r1 {
            return Err(Error::InvalidParam(format!(
                "RowBandReader: empty row band [{r0}, {r1}) in a {nz}-row dataset"
            )));
        }
        Ok(RowBandReader { inner, r0, r1 })
    }

    /// Map band-relative rows `[row0, row1)` to underlying dataset rows,
    /// rejecting ranges that leave the band.
    fn to_inner(&self, row0: usize, row1: usize) -> Result<(usize, usize)> {
        let nz = self.r1 - self.r0;
        if row0 > row1 || row1 > nz {
            return Err(Error::InvalidParam(format!(
                "RowBandReader: chunk [{row0}, {row1}) outside the band height {nz}"
            )));
        }
        Ok((self.r0 + row0, self.r0 + row1))
    }
}

impl DatasetReader for RowBandReader {
    fn read_sizes(&mut self) -> Result<(usize, usize, usize, usize, usize)> {
        let (nproj, _nz, nx, nflat, ndark) = self.inner.read_sizes()?;
        Ok((nproj, self.r1 - self.r0, nx, nflat, ndark))
    }

    fn read_theta(&mut self) -> Result<Vec<f32>> {
        self.inner.read_theta()
    }

    fn read_all(&mut self) -> Result<Dataset<f32>> {
        self.inner.read_chunk(self.r0, self.r1)
    }

    fn read_chunk(&mut self, row0: usize, row1: usize) -> Result<Dataset<f32>> {
        let (row0, row1) = self.to_inner(row0, row1)?;
        self.inner.read_chunk(row0, row1)
    }

    fn read_chunk_into(
        &mut self,
        row0: usize,
        row1: usize,
        data_out: &mut [f32],
    ) -> Result<ChunkAux> {
        let (row0, row1) = self.to_inner(row0, row1)?;
        self.inner.read_chunk_into(row0, row1, data_out)
    }
}

/// A reconstruction writer (port of tomocupy `dataio/writer.py:73`).
pub trait VolumeWriter {
    /// Declare the full output slice count `total_nz` before the first chunk.
    ///
    /// Writers that pre-allocate a single container (H5, Zarr) size their
    /// dataset/store to `total_nz` slices here; the cross-section `(ny, nx)` is
    /// taken from the first [`write_chunk`](Self::write_chunk). Per-file writers
    /// (TIFF) need nothing and keep the default no-op. Every driver
    /// ([`ReconSteps::run`](crate::ReconSteps::run), `run_streaming`,
    /// `run_streaming_pipelined`) calls this exactly once before any chunk.
    fn reserve(&mut self, _total_nz: usize) -> Result<()> {
        Ok(())
    }

    /// Write one **per-chunk** volume — exactly `end - start` slices, indexed
    /// *locally* `0..(end - start)` — to the global slice range `[start, end)`.
    ///
    /// The global range only addresses where the chunk lands in the output
    /// (TIFF file index; H5/Zarr dataset rows); the volume itself is the chunk,
    /// not the whole reconstruction. `start <= end` and the volume's slice count
    /// must equal `end - start`, mirroring the streaming driver's per-chunk call.
    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, end: usize) -> Result<()>;

    /// Finish the output after the last [`write_chunk`](Self::write_chunk).
    ///
    /// Every streaming/whole-volume driver calls this exactly once on the
    /// success path, before the writer is dropped. Single-container writers
    /// (H5) finalize the container here with a **non-durable** close
    /// (`close_no_sync`: writes all headers + the superblock, so the file is a
    /// complete, valid HDF5 on return, but skips the `fsync` that otherwise
    /// dominates close latency — bytes are handed to the OS, durability is left
    /// to page-cache writeback, matching the TIFF/Zarr writers, which never
    /// `fsync`). If a driver drops the writer *without* calling `finalize`
    /// (an error/panic path), the container is still finalized **durably** on
    /// drop — so `finalize` only trades that durability for speed on the
    /// success path. Per-file writers (TIFF) keep the default no-op.
    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }
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

impl std::str::FromStr for SaveFormat {
    type Err = Error;
    /// Parse tomocupy's `--save-format` values (`tiff` | `h5` | `zarr`),
    /// case-insensitively.
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "tiff" | "tif" => Ok(SaveFormat::Tiff),
            "h5" | "hdf5" => Ok(SaveFormat::H5),
            "zarr" => Ok(SaveFormat::Zarr),
            other => Err(Error::InvalidParam(format!(
                "unknown save format {other:?} (expected tiff | h5 | zarr)"
            ))),
        }
    }
}

/// The volume assembled in memory by [`InMemoryWriter`].
///
/// Sized by [`VolumeWriter::reserve`] (slice count) plus the first chunk's
/// cross-section; zero-filled until the corresponding chunk lands, so a
/// cancelled run leaves the unwritten tail at `0.0`.
#[derive(Debug, Default)]
pub struct InMemoryVolume {
    total_nz: usize,
    dims: Option<(usize, usize)>,
    data: Vec<f32>,
}

impl InMemoryVolume {
    /// Full output shape `(nz, ny, nx)`, known once the first chunk arrived.
    pub fn dims(&self) -> Option<(usize, usize, usize)> {
        self.dims.map(|(ny, nx)| (self.total_nz, ny, nx))
    }

    /// The flat row-major `[nz, ny, nx]` buffer (empty before the first chunk).
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Clone the buffer into a [`Volume`] (`None` before the first chunk).
    pub fn to_volume(&self) -> Option<Volume<f32>> {
        let (nz, ny, nx) = self.dims()?;
        let array = Array3::from_shape_vec((nz, ny, nx), self.data.clone())
            .expect("data length is kept at nz*ny*nx by write_chunk");
        Some(Volume::new(array))
    }
}

/// A [`VolumeWriter`] that assembles the reconstruction in memory instead of
/// on disk — the GUI/preview counterpart of the TIFF/H5/Zarr writers.
///
/// The buffer lives behind a shared handle ([`buffer`](Self::buffer)) so the
/// caller keeps access after the writer is consumed — the pipelined driver
/// constructs and drops its writer on the writer thread. An optional
/// [`on_chunk`](Self::with_on_chunk) callback fires after each chunk lands and
/// doubles as a progress signal.
#[derive(Default)]
pub struct InMemoryWriter {
    vol: std::sync::Arc<std::sync::Mutex<InMemoryVolume>>,
    on_chunk: Option<Box<dyn FnMut(usize, usize) + Send>>,
}

impl InMemoryWriter {
    /// An empty writer; shape is fixed by `reserve` + the first chunk.
    pub fn new() -> Self {
        Self::default()
    }

    /// Shared handle to the assembled volume; stays valid after the writer is
    /// dropped (e.g. by the pipelined driver's writer thread).
    pub fn buffer(&self) -> std::sync::Arc<std::sync::Mutex<InMemoryVolume>> {
        std::sync::Arc::clone(&self.vol)
    }

    /// Invoke `on_chunk(start, end)` after each chunk is copied in — a
    /// progress signal for the global slice range that just completed.
    pub fn with_on_chunk(mut self, on_chunk: impl FnMut(usize, usize) + Send + 'static) -> Self {
        self.on_chunk = Some(Box::new(on_chunk));
        self
    }
}

impl VolumeWriter for InMemoryWriter {
    fn reserve(&mut self, total_nz: usize) -> Result<()> {
        let mut vol = self.vol.lock().expect("InMemoryVolume lock poisoned");
        vol.total_nz = total_nz;
        Ok(())
    }

    fn write_chunk(&mut self, chunk: &Volume<f32>, start: usize, end: usize) -> Result<()> {
        let (cz, ny, nx) = chunk.array.dim();
        {
            let mut vol = self.vol.lock().expect("InMemoryVolume lock poisoned");
            if cz != end - start || end > vol.total_nz {
                return Err(Error::ShapeMismatch {
                    expected: format!(
                        "chunk of {} slices within [0, {})",
                        end.saturating_sub(start),
                        vol.total_nz
                    ),
                    found: format!("{cz} slices at [{start}, {end})"),
                });
            }
            match vol.dims {
                None => {
                    vol.dims = Some((ny, nx));
                    let total = vol.total_nz;
                    vol.data = vec![0.0; total * ny * nx];
                }
                Some(d) if d != (ny, nx) => {
                    return Err(Error::ShapeMismatch {
                        expected: format!("cross-section {:?} (from first chunk)", d),
                        found: format!("({ny}, {nx})"),
                    });
                }
                Some(_) => {}
            }
            let dst = &mut vol.data[start * ny * nx..end * ny * nx];
            if let Some(src) = chunk.array.as_slice() {
                dst.copy_from_slice(src);
            } else {
                for (d, s) in dst.iter_mut().zip(chunk.array.iter()) {
                    *d = *s;
                }
            }
        }
        if let Some(cb) = &mut self.on_chunk {
            cb(start, end);
        }
        Ok(())
    }
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

/// Read frame `index` of a 3-D `[n, ny, nx]` HDF5 dataset as
/// `(ny, nx, row-major f32)`, converting from any numeric on-disk dtype (the
/// same dispatch the DXchange readers use — real beamline stacks are usually
/// `uint16`). One hyperslab read per call: the lazy entry point for
/// projection/volume frame browsers. `data_path` may be absolute
/// (`/exchange/data`) or relative.
pub fn read_h5_frame(
    path: &str,
    data_path: &str,
    index: usize,
) -> Result<(usize, usize, Vec<f32>)> {
    let file = H5File::open(path).map_err(|e| Error::Io(format!("open {path}: {e}")))?;
    // rust-hdf5 keys nested datasets without a leading slash.
    let key = data_path.strip_prefix('/').unwrap_or(data_path);
    let ds = file
        .dataset(key)
        .map_err(|e| Error::Io(format!("dataset {data_path}: {e}")))?;
    let shape = ds.shape();
    let [n, ny, nx] = shape[..] else {
        return Err(Error::ShapeMismatch {
            expected: "[n, ny, nx] (3-D stack)".into(),
            found: format!("{shape:?}"),
        });
    };
    if index >= n {
        return Err(Error::InvalidParam(format!(
            "frame {index} out of range (stack has {n})"
        )));
    }
    let data = read_f32_slice(&ds, &[index, 0, 0], &[1, ny, nx])?;
    Ok((ny, nx, data))
}

/// Shape `(n, ny, nx)` of a 3-D HDF5 dataset — the metadata probe paired with
/// [`read_h5_frame`], so a volume browser can size itself without reading any
/// frame. Errors on a non-3-D dataset. `data_path` may be absolute or relative.
pub fn read_h5_sizes(path: &str, data_path: &str) -> Result<(usize, usize, usize)> {
    let file = H5File::open(path).map_err(|e| Error::Io(format!("open {path}: {e}")))?;
    let key = data_path.strip_prefix('/').unwrap_or(data_path);
    let ds = file
        .dataset(key)
        .map_err(|e| Error::Io(format!("dataset {data_path}: {e}")))?;
    let shape = ds.shape();
    let [n, ny, nx] = shape[..] else {
        return Err(Error::ShapeMismatch {
            expected: "[n, ny, nx] (3-D stack)".into(),
            found: format!("{shape:?}"),
        });
    };
    Ok((n, ny, nx))
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

    fn read_chunk_into(
        &mut self,
        row0: usize,
        row1: usize,
        data_out: &mut [f32],
    ) -> Result<ChunkAux> {
        let [nproj, nz, nx] = self.data_shape()?;
        if row0 > row1 || row1 > nz {
            return Err(Error::InvalidParam(format!(
                "read_chunk_into rows [{row0}, {row1}) out of range for nz={nz}"
            )));
        }
        let rows = row1 - row0;
        let expected = nproj * rows * nx;
        if data_out.len() != expected {
            return Err(Error::ShapeMismatch {
                expected: format!("{expected} elems ([{nproj}, {rows}, {nx}])"),
                found: format!("{} elems", data_out.len()),
            });
        }

        // Projections straight into the caller's (pinned) buffer — no owned
        // allocation, no copy. The hyperslab is `[:, row0:row1, :]` of [nproj, nz, nx].
        read_f32_slice_into(
            &self.required(dxchange::DATA)?,
            &[0, row0, 0],
            &[nproj, rows, nx],
            data_out,
        )?;

        // Flat/dark are tiny (one frame each here) — the owned-array slab is fine.
        let slab = |ds: &H5Dataset, n: usize| -> Result<Array3<f32>> {
            let v = read_f32_slice(ds, &[0, row0, 0], &[n, rows, nx])?;
            Array3::from_shape_vec((n, rows, nx), v).map_err(|e| Error::ShapeMismatch {
                expected: format!("[{n}, {rows}, {nx}]"),
                found: e.to_string(),
            })
        };
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

        Ok(ChunkAux { flat, dark, theta })
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
            size: 4,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            ds.read_slice::<f32>(starts, counts).map_err(raw)?
        }
        DatatypeMessage::FloatingPoint {
            size: 8,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            cast!(f64)
        }
        DatatypeMessage::FixedPoint {
            size: 1,
            signed: false,
            ..
        } => cast!(u8),
        DatatypeMessage::FixedPoint {
            size: 1,
            signed: true,
            ..
        } => cast!(i8),
        DatatypeMessage::FixedPoint {
            size: 2,
            signed: false,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            cast!(u16)
        }
        DatatypeMessage::FixedPoint {
            size: 2,
            signed: true,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            cast!(i16)
        }
        DatatypeMessage::FixedPoint {
            size: 4,
            signed: false,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            cast!(u32)
        }
        DatatypeMessage::FixedPoint {
            size: 4,
            signed: true,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            cast!(i32)
        }
        DatatypeMessage::FixedPoint {
            size: 8,
            signed: false,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            cast!(u64)
        }
        DatatypeMessage::FixedPoint {
            size: 8,
            signed: true,
            byte_order,
            ..
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

/// Read a hyperslab **directly into** `out` (a caller-owned, possibly pinned,
/// buffer), avoiding [`read_f32_slice`]'s internal allocation + copy. The fast
/// path is an f32-on-disk dataset: `rust-hdf5`'s `read_slice_into` decodes the
/// hyperslab straight into `out` (no `Vec`, no `memcpy`). Non-f32 on-disk dtypes
/// have no zero-copy route here (they need a numeric cast), so they fall back to
/// [`read_f32_slice`] + a copy into `out` — correct, just not allocation-free.
/// `out.len()` must equal `counts.iter().product()`.
fn read_f32_slice_into(
    ds: &H5Dataset,
    starts: &[usize],
    counts: &[usize],
    out: &mut [f32],
) -> Result<()> {
    let expected: usize = counts.iter().product();
    if out.len() != expected {
        return Err(Error::ShapeMismatch {
            expected: format!("{expected} elems"),
            found: format!("{} elems", out.len()),
        });
    }
    let dt = ds
        .datatype()
        .map_err(|e| Error::Io(format!("datatype: {e}")))?;
    match dt {
        DatatypeMessage::FloatingPoint {
            size: 4,
            byte_order,
            ..
        } => {
            ensure_le(byte_order)?;
            ds.read_slice_into::<f32>(out, starts, counts)
                .map_err(|e| Error::Io(format!("read: {e}")))?;
            Ok(())
        }
        _ => {
            // Non-f32 on disk: cast through the allocating path, then copy in.
            let v = read_f32_slice(ds, starts, counts)?;
            out.copy_from_slice(&v);
            Ok(())
        }
    }
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
        // `vol` is the chunk for global slice range `[start, end)`, so it holds
        // exactly `end - start` slices indexed *locally* 0..(end-start). The
        // global index only names the output file. Validate the chunk extent
        // (not `end <= vol.nz`, which conflated the global range with the chunk
        // size and broke every chunk after the first in the streaming path).
        let (cz, ny, nx) = vol.dims();
        // Validate the range orientation first, so `end - start` (a `usize`) is
        // only evaluated once `end >= start` — otherwise the chunk-size message
        // below would underflow-panic on an inverted range.
        if start > end {
            return Err(Error::InvalidParam(format!(
                "write_chunk: inverted global range [{start}, {end})"
            )));
        }
        if cz != end - start {
            return Err(Error::InvalidParam(format!(
                "write_chunk: volume has {cz} slices but global range [{start}, {end}) expects {}",
                end - start
            )));
        }
        for local in 0..cz {
            // Slice is [y, x], contiguous row-major (x fastest) in z-major
            // Volume storage — exactly TIFF's width=nx, height=ny order. A
            // z-slice of a standard C-layout volume is itself contiguous, so
            // hand its backing slice to the encoder zero-copy; gather into a
            // temporary only for a non-contiguous caller.
            let slice = vol.array.index_axis(Axis(0), local);
            let gathered;
            let buf: &[f32] = match slice.as_slice() {
                Some(s) => s,
                None => {
                    gathered = slice.iter().copied().collect::<Vec<f32>>();
                    &gathered
                }
            };

            let global = start + local;
            let fname = format!("{}_{global:05}.tiff", self.prefix);
            let file =
                File::create(&fname).map_err(|e| Error::Io(format!("create {fname}: {e}")))?;
            let mut enc = TiffEncoder::new(BufWriter::new(file))
                .map_err(|e| Error::Io(format!("tiff encoder {fname}: {e}")))?;
            enc.write_image::<Gray32Float>(nx as u32, ny as u32, buf)
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
/// `axes`/`description`/`units` attributes. The dataset is sized to the
/// `reserve(total_nz)` slice count (the cross-section `(ny, nx)` comes from the
/// first per-chunk volume); each `write_chunk` lands its chunk's slices in the
/// global `[start, end)` rows via an HDF5 hyperslab — `rust-hdf5` `pwrite`s the
/// bytes immediately, but the container (headers + superblock) is only written
/// when the writer is finalized. [`finalize`](VolumeWriter::finalize) closes it
/// non-durably (`close_no_sync`); dropping without finalize closes it durably.
/// `path` is the output base — `.h5` is appended if absent (mirroring tomocupy
/// `fnameout += '.h5'`).
///
/// Contiguous (not chunked) layout is required: `write_slice` hyperslabs are
/// only valid on contiguous datasets, and `h5nolinks` output is uncompressed.
struct H5Writer {
    /// Resolved `.h5` output path.
    path: std::path::PathBuf,
    /// Total output slices, set by [`VolumeWriter::reserve`] before the first
    /// chunk; the dataset's `nz` extent. `None` until reserved.
    total_nz: Option<usize>,
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
            total_nz: None,
            state: None,
        })
    }

    /// Create `{path}.h5` with an `exchange/data` dataset of shape `(nz, ny, nx)`.
    fn create_dataset(&self, nz: usize, ny: usize, nx: usize) -> Result<H5WriteState> {
        // Unlink a stale output first so the create always lands on a fresh
        // inode. Creating over an existing output would otherwise truncate
        // real content, which arms ext4 `auto_da_alloc`
        // (replace-via-truncate protection) and turns the final `close(2)`
        // into an implicit ~325 ms writeback — re-paying exactly the sync the
        // non-durable `finalize` (`close_no_sync`) skips. The unlink trades
        // away rust-hdf5's lock-before-truncate protection for this one file;
        // that is acceptable here because the reconstruction output is
        // regenerable and a concurrent reader of the old file keeps its inode.
        match std::fs::remove_file(&self.path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(Error::Io(format!(
                    "remove stale {}: {e}",
                    self.path.display()
                )))
            }
        }
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
    fn reserve(&mut self, total_nz: usize) -> Result<()> {
        self.total_nz = Some(total_nz);
        Ok(())
    }

    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, end: usize) -> Result<()> {
        // Per-chunk contract: `vol` holds exactly `end - start` locally-indexed
        // slices that land in global dataset rows `[start, end)` (see the trait
        // doc); the dataset's `nz` extent comes from `reserve`, not from `vol`.
        let (cz, ny, nx) = vol.dims();
        if start > end {
            return Err(Error::InvalidParam(format!(
                "write_chunk: inverted global range [{start}, {end})"
            )));
        }
        if cz != end - start {
            return Err(Error::InvalidParam(format!(
                "write_chunk: volume has {cz} slices but global range [{start}, {end}) expects {}",
                end - start
            )));
        }
        let total_nz = self.total_nz.ok_or_else(|| {
            Error::InvalidParam(
                "write_chunk: reserve(total_nz) must be called before writing H5 chunks".into(),
            )
        })?;
        if end > total_nz {
            return Err(Error::InvalidParam(format!(
                "write_chunk: global range [{start}, {end}) exceeds the reserved {total_nz} slices"
            )));
        }
        // Size the dataset to the reserved `nz` and the first chunk's `(ny, nx)`;
        // every later chunk must share that cross-section.
        if self.state.is_none() {
            self.state = Some(self.create_dataset(total_nz, ny, nx)?);
        }
        let state = self.state.as_ref().unwrap();
        if state.dims != (total_nz, ny, nx) {
            return Err(Error::InvalidParam(format!(
                "write_chunk: chunk cross-section {:?} differs from the created dataset {:?}",
                (ny, nx),
                (state.dims.1, state.dims.2)
            )));
        }
        if cz == 0 {
            return Ok(());
        }
        // The whole chunk (local rows `0..cz`, C-order `[cz, ny, nx]`) is written
        // into global rows `[start, end)` (tomocupy `dset[st:end] = rec`).
        // `write_slice` `pwrite`s the chunk to the OS immediately; there is no
        // per-chunk flush (`H5File::flush` is a documented no-op). The container
        // is finalized once, in `finalize`.
        //
        // Every in-tree driver hands a standard C-layout chunk volume, whose
        // backing slice is already the C-order `[cz, ny, nx]` slab `write_slice`
        // expects — pass it zero-copy. The elementwise gather this replaces was
        // the H5 writer's dominant cost (~2× the raw file write for a 512³
        // volume); it remains only as the fallback for a non-contiguous caller.
        let write = |slab: &[f32]| {
            state
                .dataset
                .write_slice(&[start, 0, 0], &[cz, ny, nx], slab)
                .map_err(|e| Error::Io(format!("write /exchange/data[{start}..{end}]: {e}")))
        };
        match vol.array.as_slice() {
            Some(slab) => write(slab)?,
            None => {
                let slab: Vec<f32> = vol.array.iter().copied().collect();
                write(&slab)?;
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        // Close non-durably: writes all object headers + the superblock (so the
        // file is a complete, valid HDF5 on return) but skips the finalize-path
        // `fsync`, which otherwise dominates close latency. Durability is left to
        // OS page-cache writeback, matching the TIFF/Zarr writers. Taking `state`
        // leaves the writer spent; a later `write_chunk` would recreate the file,
        // but the driver contract calls `finalize` exactly once, last.
        //
        // If this is never reached (an error/panic path drops the writer with
        // `state` still `Some`), the inner `H5File`'s own `Drop` finalizes the
        // file *durably* — a safe, slower fallback, never a corrupt file.
        if let Some(state) = self.state.take() {
            let H5WriteState { file, dataset, .. } = state;
            // Release the dataset handle before closing the file it belongs to.
            drop(dataset);
            file.close_no_sync()
                .map_err(|e| Error::Io(format!("close {}: {e}", self.path.display())))?;
        }
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
    /// Total output slices, set by [`VolumeWriter::reserve`] before the first
    /// chunk; the array's `nz` extent. `None` until reserved.
    total_nz: Option<usize>,
    /// Full array extents `(nz, ny, nx)`, fixed on the first `write_chunk` from
    /// the reserved `nz` and the first chunk's `(ny, nx)`.
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
        Ok(Self {
            root,
            total_nz: None,
            dims: None,
        })
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
    fn reserve(&mut self, total_nz: usize) -> Result<()> {
        self.total_nz = Some(total_nz);
        Ok(())
    }

    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, end: usize) -> Result<()> {
        // Per-chunk contract: `vol` holds exactly `end - start` locally-indexed
        // slices that land in global chunk files `{start+local}.0.0` (see the
        // trait doc); the array's `nz` extent comes from `reserve`, not `vol`.
        let (cz, ny, nx) = vol.dims();
        if start > end {
            return Err(Error::InvalidParam(format!(
                "write_chunk: inverted global range [{start}, {end})"
            )));
        }
        if cz != end - start {
            return Err(Error::InvalidParam(format!(
                "write_chunk: volume has {cz} slices but global range [{start}, {end}) expects {}",
                end - start
            )));
        }
        let total_nz = self.total_nz.ok_or_else(|| {
            Error::InvalidParam(
                "write_chunk: reserve(total_nz) must be called before writing Zarr chunks".into(),
            )
        })?;
        if end > total_nz {
            return Err(Error::InvalidParam(format!(
                "write_chunk: global range [{start}, {end}) exceeds the reserved {total_nz} slices"
            )));
        }
        // Size the store to the reserved `nz` and the first chunk's `(ny, nx)`;
        // every later chunk must share that cross-section.
        if self.dims.is_none() {
            self.init_store(total_nz, ny, nx)?;
            self.dims = Some((total_nz, ny, nx));
        }
        let dims = self.dims.unwrap();
        if dims != (total_nz, ny, nx) {
            return Err(Error::InvalidParam(format!(
                "write_chunk: chunk cross-section {:?} differs from the created store {:?}",
                (ny, nx),
                (dims.1, dims.2)
            )));
        }
        let data_dir = self.root.join("exchange").join("data");
        // One chunk file per z-slice: local row `local` → global chunk grid coord
        // `(start + local, 0, 0)` → "{global}.0.0".
        for local in 0..cz {
            let slice = vol.array.index_axis(Axis(0), local);
            // C-order (y-major, x fastest) little-endian f32 — exactly `<f4`.
            // On a little-endian target the bytes of a contiguous slice already
            // are `<f4`, so reinterpret them in place (bytemuck's safe Pod
            // cast); gather elementwise only on a big-endian target or for a
            // non-contiguous caller.
            let gathered;
            let bytes: &[u8] = match slice.as_slice() {
                Some(s) if cfg!(target_endian = "little") => bytemuck::cast_slice(s),
                _ => {
                    gathered = slice
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>();
                    &gathered
                }
            };
            let global = start + local;
            let chunk = data_dir.join(format!("{global}.0.0"));
            std::fs::write(&chunk, bytes)
                .map_err(|e| Error::Io(format!("write {}: {e}", chunk.display())))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Slice;

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
    fn read_h5_frame_reads_one_u16_frame() {
        // Beamline stacks are typically uint16; the frame read must convert
        // via the dtype dispatch, not byte-copy. 2 frames × 2 rows × 3 cols.
        let path = std::env::temp_dir().join("tomoxide_read_h5_frame_test.h5");
        let _ = std::fs::remove_file(&path);
        {
            let file = H5File::create(&path).unwrap();
            let group = file.create_group("exchange").unwrap();
            let ds = group
                .new_dataset::<u16>()
                .shape([2, 2, 3])
                .create("data")
                .unwrap();
            let vals: Vec<u16> = (0..12).collect();
            ds.write_raw(&vals).unwrap();
        }
        let p = path.to_str().unwrap();

        // Absolute and relative dataset paths both resolve.
        let (ny, nx, f1) = read_h5_frame(p, "/exchange/data", 1).unwrap();
        assert_eq!((ny, nx), (2, 3));
        assert_eq!(f1, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        let (_, _, f0) = read_h5_frame(p, "exchange/data", 0).unwrap();
        assert_eq!(f0, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        // Out-of-range frame and non-3-D dataset are rejected.
        assert!(matches!(
            read_h5_frame(p, "/exchange/data", 2),
            Err(Error::InvalidParam(_))
        ));

        // The shape probe agrees, on both path spellings.
        assert_eq!(read_h5_sizes(p, "/exchange/data").unwrap(), (2, 2, 3));
        assert_eq!(read_h5_sizes(p, "exchange/data").unwrap(), (2, 2, 3));
        assert!(matches!(
            read_h5_sizes(p, "/exchange/missing"),
            Err(Error::Io(_))
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tiff_writer_streams_per_chunk_volumes_with_global_indices() {
        // The streaming driver (`run_streaming_pipelined`) hands the writer a
        // *per-chunk* volume of `end-start` slices and the global range
        // `[start, end)`. The TIFF file is named by the global index; the chunk
        // array is indexed locally. The old code indexed the chunk by the global
        // index and bounds-checked `end <= chunk.nz`, so every chunk after the
        // first failed ("out of bounds for N slices"). Reproduce the streaming
        // call pattern: two separate 2-slice chunks covering global [0,4).
        let dir = std::env::temp_dir().join("tomoxide_tiff_stream_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("recon");
        let prefix = prefix.to_str().unwrap();

        let mut w = create_writer(prefix, SaveFormat::Tiff).unwrap();
        // Each chunk is its own [2, ny, nx] volume; value encodes global slice.
        let chunk0 = Volume::new(Array3::from_shape_fn((2, 2, 3), |(z, y, x)| {
            (z * 100 + y * 10 + x) as f32
        }));
        let chunk1 = Volume::new(Array3::from_shape_fn((2, 2, 3), |(z, y, x)| {
            ((2 + z) * 100 + y * 10 + x) as f32
        }));
        w.write_chunk(&chunk0, 0, 2).unwrap();
        w.write_chunk(&chunk1, 2, 4).unwrap(); // would have errored before the fix

        // One file per global slice index, named by the global index.
        for g in 0..4 {
            let f = format!("{prefix}_{g:05}.tiff");
            assert!(std::path::Path::new(&f).is_file(), "missing {f}");
        }

        // A wrong-sized chunk (volume slices != end-start) must be rejected.
        let bad = Volume::new(Array3::<f32>::zeros((3, 2, 3)));
        assert!(matches!(
            w.write_chunk(&bad, 4, 6),
            Err(Error::InvalidParam(_))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn zarr_writer_roundtrips_a_spec_compliant_store() {
        // Unique temp store root (no tempfile dev-dep); clean before and after.
        let base = std::env::temp_dir().join("tomoxide_zarr_writer_roundtrip");
        let root = base.with_extension("zarr");
        let _ = std::fs::remove_dir_all(&root);

        // 3 slices of 2x2 with distinct, C-order-distinguishable values.
        let arr = Array3::from_shape_fn((3, 2, 2), |(z, y, x)| (z * 100 + y * 10 + x) as f32);

        // Stream it as two per-chunk volumes (global ranges [0,2) then [2,3))
        // after reserving the full 3 slices — the per-chunk + reserve contract.
        let mut w = create_writer(base.to_str().unwrap(), SaveFormat::Zarr).unwrap();
        w.reserve(3).unwrap();
        let c0 = Volume::new(arr.slice_axis(Axis(0), Slice::from(0..2)).to_owned());
        let c1 = Volume::new(arr.slice_axis(Axis(0), Slice::from(2..3)).to_owned());
        w.write_chunk(&c0, 0, 2).unwrap();
        w.write_chunk(&c1, 2, 3).unwrap();

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
        w.reserve(2).unwrap();
        let a = Volume::new(Array3::zeros((2, 4, 4)));
        w.write_chunk(&a, 0, 2).unwrap();
        // A second chunk with a different cross-section must be rejected (the
        // store shape is fixed by the first write), mirroring H5Writer.
        let b = Volume::new(Array3::zeros((2, 8, 8)));
        assert!(matches!(
            w.write_chunk(&b, 0, 2),
            Err(Error::InvalidParam(_))
        ));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn h5_writer_streams_per_chunk_volumes_and_round_trips() {
        // Per-chunk streaming into one /exchange/data dataset: reserve the full
        // nz, write two per-chunk volumes at their global ranges, then read the
        // dataset back and compare bit-exact — the H5 analogue of the TIFF
        // streaming test.
        let base = std::env::temp_dir().join(format!("tomoxide_h5_stream_{}", std::process::id()));
        let base_str = base.to_str().unwrap();
        let out = std::path::PathBuf::from(format!("{base_str}.h5"));
        let _ = std::fs::remove_file(&out);

        let arr = Array3::from_shape_fn((3, 2, 2), |(z, y, x)| (z * 100 + y * 10 + x) as f32);
        let mut w = create_writer(base_str, SaveFormat::H5).unwrap();
        w.reserve(3).unwrap();
        let c0 = Volume::new(arr.slice_axis(Axis(0), Slice::from(0..2)).to_owned());
        let c1 = Volume::new(arr.slice_axis(Axis(0), Slice::from(2..3)).to_owned());
        w.write_chunk(&c0, 0, 2).unwrap();
        w.write_chunk(&c1, 2, 3).unwrap();
        drop(w); // close the file before reopening it for read-back

        let f = H5File::open(&out).unwrap();
        let ds = f.dataset("exchange/data").unwrap();
        let got = ds.read_raw::<f32>().unwrap();
        let want: Vec<f32> = arr.iter().copied().collect();
        assert_eq!(got, want, "h5 streamed round-trip");

        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn h5_and_zarr_reject_write_before_reserve() {
        // The pre-allocating writers need the total slice count before sizing
        // their container; a write_chunk with no reserve is rejected (and for
        // H5, no file is created since the guard precedes dataset creation).
        let v = Volume::new(Array3::<f32>::zeros((1, 2, 2)));

        let zb =
            std::env::temp_dir().join(format!("tomoxide_zarr_noreserve_{}", std::process::id()));
        let mut zw = create_writer(zb.to_str().unwrap(), SaveFormat::Zarr).unwrap();
        assert!(matches!(
            zw.write_chunk(&v, 0, 1),
            Err(Error::InvalidParam(_))
        ));
        let _ = std::fs::remove_dir_all(zb.with_extension("zarr"));

        let hb = std::env::temp_dir().join(format!("tomoxide_h5_noreserve_{}", std::process::id()));
        let mut hw = create_writer(hb.to_str().unwrap(), SaveFormat::H5).unwrap();
        assert!(matches!(
            hw.write_chunk(&v, 0, 1),
            Err(Error::InvalidParam(_))
        ));
        let _ = std::fs::remove_file(format!("{}.h5", hb.to_str().unwrap()));
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
