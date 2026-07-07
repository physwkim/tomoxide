//! Results HDF5 writer for a fitted XANES peak-energy map.
//!
//! Writes the map in a layout the `xanes_tools` Python viewer reads: a
//! `peak_energies` volume, the `energies` axis it was fitted over, and a
//! finite-voxel `mask`. Concentration / edge-jump / intensity fields (which
//! need the registration-stage inputs) are not written by v1.

use std::path::Path;

use ndarray::ArrayView3;
use rust_hdf5::H5File;

use crate::error::{Error, Result};

/// Write `peak` (`(z, y, x)` fitted peak energies, NaN where unfit) and its
/// `energies` axis to `path`, plus a `u8` `mask` (1 where the fit is finite).
pub fn write_peak_map_h5(
    path: impl AsRef<Path>,
    energies: &[f64],
    peak: ArrayView3<f64>,
) -> Result<()> {
    let path = path.as_ref();
    let (nz, ny, nx) = peak.dim();
    let file =
        H5File::create(path).map_err(|e| Error::Io(format!("create {}: {e}", path.display())))?;

    // energies (E,).
    file.new_dataset::<f64>()
        .shape([energies.len()])
        .create("energies")
        .map_err(|e| Error::Io(format!("create energies: {e}")))?
        .write_raw(energies)
        .map_err(|e| Error::Io(format!("write energies: {e}")))?;

    // peak_energies (z, y, x) — logical (row-major) order matches the shape.
    let peak_flat: Vec<f64> = peak.iter().copied().collect();
    file.new_dataset::<f64>()
        .shape([nz, ny, nx])
        .create("peak_energies")
        .map_err(|e| Error::Io(format!("create peak_energies: {e}")))?
        .write_raw(&peak_flat)
        .map_err(|e| Error::Io(format!("write peak_energies: {e}")))?;

    // mask (z, y, x) u8: 1 where the fit produced a finite energy.
    let mask: Vec<u8> = peak_flat.iter().map(|v| u8::from(v.is_finite())).collect();
    file.new_dataset::<u8>()
        .shape([nz, ny, nx])
        .create("mask")
        .map_err(|e| Error::Io(format!("create mask: {e}")))?
        .write_raw(&mask)
        .map_err(|e| Error::Io(format!("write mask: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{read_h5_band, read_h5_sizes};
    use ndarray::Array3;

    #[test]
    fn round_trips_peak_map() {
        let dir = std::env::temp_dir().join("tomoxide_xanes_result_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("result.h5");

        let (nz, ny, nx) = (2, 2, 3);
        let mut peak = Array3::<f64>::from_shape_fn((nz, ny, nx), |(z, y, x)| {
            8.30 + (z * 100 + y * 10 + x) as f64 * 0.001
        });
        peak[[1, 1, 2]] = f64::NAN; // one unfit voxel
        let energies = vec![8.30, 8.32, 8.34];

        write_peak_map_h5(&path, &energies, peak.view()).unwrap();

        let p = path.to_str().unwrap();
        assert_eq!(read_h5_sizes(p, "peak_energies").unwrap(), (nz, ny, nx));
        let (_, _, _, data) = read_h5_band(p, "peak_energies", 0, nz).unwrap();
        // Read back as f32 (the band reader casts); check a finite voxel.
        assert!((data[0] - 8.30_f32).abs() < 1e-4);
        // The mask marks the NaN voxel 0, others 1.
        let (_, _, _, mask) = read_h5_band(p, "mask", 0, nz).unwrap();
        assert_eq!(mask[(ny + 1) * nx + 2], 0.0);
        assert_eq!(mask[0], 1.0);

        let _ = std::fs::remove_file(&path);
    }
}
