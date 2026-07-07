//! Multi-energy reconstruction stack reader for XANES fitting.
//!
//! A XANES chemical map is fitted over a stack of reconstructed volumes, one
//! per X-ray energy, all sharing a voxel grid. Those volumes arrive in one of
//! two layouts:
//!
//! - **Separate files** — tomoxide's own per-energy recon output: N HDF5 files,
//!   each a `[z, y, x]` volume at one dataset (`/exchange/data` by default).
//!   This is the raw, pre-registration form (each energy reconstructed on its
//!   own by the energy-looped streaming pipeline).
//! - **Combined file** — one HDF5 file holding every energy's volume under its
//!   own dataset key (e.g. the registered stack an upstream tool emits).
//!
//! Both reduce to the same thing: a list of [`EnergyLayer`]s (energy + where its
//! volume lives), which [`MultiEnergyVolume`] validates to a common grid, sorts
//! by energy, and reads a `z`-band at a time as `(E, band, ny, nx)` `f32` for
//! [`fit_map`](super::fit_map) — the full `f64` stack is never held.
//!
//! Dataset **keys are always explicit** (caller-supplied). This reader never
//! guesses how a combined file names its per-energy groups or pairs them to an
//! energy axis — that schema is owned by whatever wrote the file, so the caller
//! resolves `(energy, dataset)` pairs (e.g. from the file's dataset listing)
//! and passes them in.

use std::path::{Path, PathBuf};

use ndarray::{s, Array3, Array4};

use crate::error::{Error, Result};
use crate::io::{read_h5_band, read_h5_sizes};

/// One energy's reconstructed volume: its energy and where the `[z, y, x]`
/// dataset lives.
#[derive(Debug, Clone)]
pub struct EnergyLayer {
    /// X-ray energy (units are the caller's; only the ordering matters here).
    pub energy: f64,
    /// HDF5 file holding this energy's volume.
    pub path: PathBuf,
    /// Dataset key of the `[z, y, x]` volume (absolute or relative).
    pub dataset: String,
}

/// A common-grid stack of per-energy reconstructions, read one `z`-band at a
/// time. Layers are held sorted by ascending energy.
#[derive(Debug, Clone)]
pub struct MultiEnergyVolume {
    layers: Vec<EnergyLayer>,
    nz: usize,
    ny: usize,
    nx: usize,
}

impl MultiEnergyVolume {
    /// Validate a layer list to a common grid and sort it by energy.
    ///
    /// Probes each layer's dataset shape (one metadata read per layer, no voxel
    /// data) and errors if the grids differ or the list is empty.
    pub fn new(mut layers: Vec<EnergyLayer>) -> Result<Self> {
        if layers.is_empty() {
            return Err(Error::InvalidParam(
                "MultiEnergyVolume: no energy layers".into(),
            ));
        }
        let mut dims: Option<(usize, usize, usize)> = None;
        for l in &layers {
            let p = path_str(&l.path)?;
            let d = read_h5_sizes(p, &l.dataset)?;
            match dims {
                None => dims = Some(d),
                Some(prev) if prev != d => {
                    return Err(Error::ShapeMismatch {
                        expected: format!("{prev:?} (first layer)"),
                        found: format!("{d:?} at {} (energy {})", l.dataset, l.energy),
                    });
                }
                _ => {}
            }
        }
        let (nz, ny, nx) = dims.expect("non-empty layers set dims");
        layers.sort_by(|a, b| a.energy.total_cmp(&b.energy));
        Ok(MultiEnergyVolume { layers, nz, ny, nx })
    }

    /// Build from separate per-energy files that share one dataset key
    /// (tomoxide's per-energy recon output; `dataset` is usually
    /// `/exchange/data`).
    pub fn from_files(entries: &[(f64, PathBuf)], dataset: &str) -> Result<Self> {
        let layers = entries
            .iter()
            .map(|(energy, path)| EnergyLayer {
                energy: *energy,
                path: path.clone(),
                dataset: dataset.to_string(),
            })
            .collect();
        Self::new(layers)
    }

    /// Build from one combined file, with an explicit `(energy, dataset)` per
    /// energy. The caller owns the naming scheme (see the module note); this
    /// reader does not infer it.
    pub fn from_combined(path: impl AsRef<Path>, entries: &[(f64, String)]) -> Result<Self> {
        let path = path.as_ref();
        let layers = entries
            .iter()
            .map(|(energy, dataset)| EnergyLayer {
                energy: *energy,
                path: path.to_path_buf(),
                dataset: dataset.clone(),
            })
            .collect();
        Self::new(layers)
    }

    /// Energies, ascending (the axis a fitted spectrum runs over).
    pub fn energies(&self) -> Vec<f64> {
        self.layers.iter().map(|l| l.energy).collect()
    }

    /// `(E, nz, ny, nx)` of the stack.
    pub fn dims(&self) -> (usize, usize, usize, usize) {
        (self.layers.len(), self.nz, self.ny, self.nx)
    }

    /// Read the `z`-band `[z0, z1)` across every energy into `(E, band, ny, nx)`
    /// `f32`. One coalesced hyperslab read per energy; nothing outside the band
    /// is loaded.
    pub fn read_band(&self, z0: usize, z1: usize) -> Result<Array4<f32>> {
        if z0 >= z1 || z1 > self.nz {
            return Err(Error::InvalidParam(format!(
                "read_band [{z0}, {z1}) out of range (stack has {} slices)",
                self.nz
            )));
        }
        let band = z1 - z0;
        let mut out = Array4::<f32>::zeros((self.layers.len(), band, self.ny, self.nx));
        for (ie, l) in self.layers.iter().enumerate() {
            let p = path_str(&l.path)?;
            let (b, ny, nx, data) = read_h5_band(p, &l.dataset, z0, z1)?;
            if (b, ny, nx) != (band, self.ny, self.nx) {
                return Err(Error::ShapeMismatch {
                    expected: format!("{:?}", (band, self.ny, self.nx)),
                    found: format!("{:?} at energy {}", (b, ny, nx), l.energy),
                });
            }
            let arr =
                Array3::from_shape_vec((band, ny, nx), data).map_err(|e| Error::ShapeMismatch {
                    expected: format!("{:?}", (band, ny, nx)),
                    found: format!("flat band read: {e}"),
                })?;
            out.slice_mut(s![ie, .., .., ..]).assign(&arr);
        }
        Ok(out)
    }
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str()
        .ok_or_else(|| Error::Io(format!("non-UTF-8 path {p:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_hdf5::H5File;

    /// Write a `[nz, ny, nx]` f32 volume at `/exchange/data` whose value encodes
    /// `(z, y, x)` plus a per-file `tag` so reads are distinguishable by energy.
    fn write_volume(path: &Path, nz: usize, ny: usize, nx: usize, tag: f32) {
        let _ = std::fs::remove_file(path);
        let file = H5File::create(path).unwrap();
        let group = file.create_group("exchange").unwrap();
        let ds = group
            .new_dataset::<f32>()
            .shape([nz, ny, nx])
            .create("data")
            .unwrap();
        let mut vals = vec![0.0f32; nz * ny * nx];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    vals[(z * ny + y) * nx + x] = tag + (z * 100 + y * 10 + x) as f32;
                }
            }
        }
        ds.write_raw(&vals).unwrap();
    }

    #[test]
    fn reads_band_across_energies_sorted() {
        let dir = std::env::temp_dir().join("tomoxide_mev_test");
        let _ = std::fs::create_dir_all(&dir);
        let (nz, ny, nx) = (4, 2, 3);
        let hi = dir.join("e_hi.h5");
        let lo = dir.join("e_lo.h5");
        write_volume(&hi, nz, ny, nx, 1000.0);
        write_volume(&lo, nz, ny, nx, 0.0);

        // Deliberately out of energy order: constructor must sort ascending.
        let mev = MultiEnergyVolume::from_files(
            &[(8.36, hi.clone()), (8.30, lo.clone())],
            "/exchange/data",
        )
        .unwrap();

        assert_eq!(mev.dims(), (2, nz, ny, nx));
        assert_eq!(mev.energies(), vec![8.30, 8.36]);

        // Band [1, 3): energy 0 = lo file (tag 0), energy 1 = hi file (tag 1000).
        let band = mev.read_band(1, 3).unwrap();
        assert_eq!(band.dim(), (2, 2, ny, nx));
        // Low energy, band-local z=0 (global z=1), y=0, x=0 → tag 0 + 1*100.
        assert_eq!(band[[0, 0, 0, 0]], 100.0);
        // High energy, same voxel → tag 1000 + 100.
        assert_eq!(band[[1, 0, 0, 0]], 1100.0);
        // band-local z=1 (global z=2), y=1, x=2 → 2*100 + 1*10 + 2 = 212.
        assert_eq!(band[[0, 1, 1, 2]], 212.0);
        assert_eq!(band[[1, 1, 1, 2]], 1212.0);

        let _ = std::fs::remove_file(&hi);
        let _ = std::fs::remove_file(&lo);
    }

    #[test]
    fn mismatched_grid_is_rejected() {
        let dir = std::env::temp_dir().join("tomoxide_mev_mismatch_test");
        let _ = std::fs::create_dir_all(&dir);
        let a = dir.join("a.h5");
        let b = dir.join("b.h5");
        write_volume(&a, 4, 2, 3, 0.0);
        write_volume(&b, 4, 2, 4, 0.0); // nx differs
        let r = MultiEnergyVolume::from_files(
            &[(8.30, a.clone()), (8.31, b.clone())],
            "/exchange/data",
        );
        assert!(matches!(r, Err(Error::ShapeMismatch { .. })));
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
    }

    #[test]
    fn empty_layers_rejected() {
        assert!(matches!(
            MultiEnergyVolume::from_files(&[], "/exchange/data"),
            Err(Error::InvalidParam(_))
        ));
    }

    #[test]
    fn band_out_of_range_rejected() {
        let dir = std::env::temp_dir().join("tomoxide_mev_range_test");
        let _ = std::fs::create_dir_all(&dir);
        let f = dir.join("r.h5");
        write_volume(&f, 4, 2, 3, 0.0);
        let mev = MultiEnergyVolume::from_files(&[(8.30, f.clone())], "/exchange/data").unwrap();
        assert!(matches!(mev.read_band(2, 2), Err(Error::InvalidParam(_))));
        assert!(matches!(mev.read_band(0, 5), Err(Error::InvalidParam(_))));
        let _ = std::fs::remove_file(&f);
    }
}
