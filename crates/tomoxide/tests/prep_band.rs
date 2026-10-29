//! `prep_band` reads a phase band `margin_rows` wider on each side, so its rows
//! match the whole-stack prep; the bare band read it replaced in the CLI's
//! `--start_row`/`--end_row` path ran phase on a truncated kernel at the edges.

use ndarray::{s, Array3, Axis, Slice};
use tomoxide::io::DatasetReader;
use tomoxide::prep::phase::margin_rows;
use tomoxide::{prep_band, CpuBackend, Dataset, Layout, PhaseMethod, PrepOptions, Tomo};

/// In-memory projections `[nproj, nz, nx]` with row-chunked reads.
struct MemReader {
    ds: Dataset<f32>,
}

impl DatasetReader for MemReader {
    fn read_sizes(&mut self) -> tomoxide::Result<(usize, usize, usize, usize, usize)> {
        let (nproj, nz, nx) = self.ds.data.array.dim();
        Ok((nproj, nz, nx, 0, 0))
    }
    fn read_theta(&mut self) -> tomoxide::Result<Vec<f32>> {
        Ok(self.ds.theta.clone())
    }
    fn read_all(&mut self) -> tomoxide::Result<Dataset<f32>> {
        Ok(self.ds.clone())
    }
    fn read_chunk(&mut self, row0: usize, row1: usize) -> tomoxide::Result<Dataset<f32>> {
        let rows = self
            .ds
            .data
            .array
            .slice_axis(Axis(1), Slice::from(row0..row1))
            .to_owned();
        Ok(Dataset {
            data: Tomo::new(rows, Layout::Projection),
            flat: None,
            dark: None,
            theta: self.ds.theta.clone(),
        })
    }
}

fn reader() -> MemReader {
    let (nproj, nz, nx) = (6, 48, 32);
    let data = Array3::from_shape_fn((nproj, nz, nx), |(p, z, x)| {
        let h = ((p * 7919 + z * 104_729 + x * 1_299_709) as f32 * 12.9898).sin() * 43_758.547;
        0.3 + 0.5 * (h - h.floor())
    });
    MemReader {
        ds: Dataset {
            data: Tomo::new(data, Layout::Projection),
            flat: None,
            dark: None,
            theta: (0..nproj).map(|i| i as f32 * 0.5).collect(),
        },
    }
}

fn max_abs(a: ndarray::ArrayView3<f32>, b: ndarray::ArrayView3<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn phase_band_matches_the_whole_stack_and_the_bare_band_does_not() {
    let backend = CpuBackend::new();
    let prep = PrepOptions {
        phase: PhaseMethod::Paganin {
            pixel_size: 1e-4,
            dist: 3.0,
            energy: 30.0,
            alpha: 1e-3,
        },
        ..Default::default()
    };
    assert_eq!(margin_rows(&prep.phase), 4);
    let (z0, z1) = (20, 28);

    let whole = prep_band(&mut reader(), 0, 48, &prep, &backend).unwrap();
    let want = whole.array.slice(s![z0..z1, .., ..]);
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));

    let band = prep_band(&mut reader(), z0, z1, &prep, &backend).unwrap();
    let banded = max_abs(band.array.view(), want) / scale;

    let mut bare = reader().read_chunk(z0, z1).unwrap();
    tomoxide::prep::normalize_dataset(&mut bare, &backend).unwrap();
    tomoxide::prep::retrieve_phase(&mut bare.data, prep.phase, &backend).unwrap();
    let bare = bare.data.to_layout(Layout::Sinogram);
    let unbanded = max_abs(bare.array.view(), want) / scale;

    // The rest of `banded` is the FFT grid: the band pads to 32 rows, the stack
    // to 64, so the two sample the Paganin filter differently.
    assert!(
        banded < 0.1 * unbanded,
        "prep_band is {banded} off the whole stack, the bare band {unbanded}"
    );
}
