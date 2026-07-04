//! `io::RowBandReader` — the banded-preview adapter (docs/GUI.md §6 #5) —
//! and `prep::phase::margin_rows`, which sizes its band for phase retrieval.

use tomoxide::io::{open_dxchange, DatasetReader, RowBandReader};
use tomoxide::prep::phase::margin_rows;
use tomoxide::PhaseMethod;

fn fixture() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/streaming_dxchange.h5"
    )
    .into()
}

fn open() -> Box<dyn DatasetReader> {
    open_dxchange(&fixture()).unwrap()
}

/// The band reports its own height; every other size and theta pass through.
#[test]
fn sizes_and_theta_remap_to_band() {
    let (nproj, nz, nx, nflat, ndark) = open().read_sizes().unwrap();
    assert!(nz >= 4, "fixture too short for the test band");
    let theta = open().read_theta().unwrap();

    let mut band = RowBandReader::new(open(), 1, 4).unwrap();
    assert_eq!(band.read_sizes().unwrap(), (nproj, 3, nx, nflat, ndark));
    assert_eq!(band.read_theta().unwrap(), theta);
}

/// `read_all` on the band equals a direct `read_chunk(r0, r1)` on the inner
/// reader — projections, flats, darks, theta.
#[test]
fn read_all_equals_direct_chunk() {
    let direct = open().read_chunk(1, 4).unwrap();
    let banded = RowBandReader::new(open(), 1, 4)
        .unwrap()
        .read_all()
        .unwrap();
    assert_eq!(banded.data.array, direct.data.array);
    assert_eq!(banded.theta, direct.theta);
    assert_eq!(
        banded.flat.map(|f| f.array),
        direct.flat.map(|f| f.array),
        "flat frames differ"
    );
    assert_eq!(
        banded.dark.map(|d| d.array),
        direct.dark.map(|d| d.array),
        "dark frames differ"
    );
}

/// A band-relative chunk `[a, b)` reads the underlying rows `[r0+a, r0+b)`.
#[test]
fn read_chunk_offsets_into_band() {
    let direct = open().read_chunk(2, 4).unwrap();
    let banded = RowBandReader::new(open(), 1, 5)
        .unwrap()
        .read_chunk(1, 3)
        .unwrap();
    assert_eq!(banded.data.array, direct.data.array);
}

/// Chunks that leave the band are rejected, not silently clamped — the
/// streaming driver derives its ranges from the band's own `read_sizes`.
#[test]
fn out_of_band_chunk_rejected() {
    let mut band = RowBandReader::new(open(), 1, 4).unwrap();
    assert!(band.read_chunk(2, 4).is_err(), "band is 3 rows tall");
    assert!(band.read_chunk(3, 2).is_err(), "inverted range");
}

/// `r1` clamps to the dataset height; a band left empty by the clamp (or by
/// `r0 >= r1`) is a constructor error.
#[test]
fn empty_band_rejected() {
    let (_nproj, nz, ..) = open().read_sizes().unwrap();
    assert!(RowBandReader::new(open(), 2, 2).is_err());
    assert!(RowBandReader::new(open(), nz, nz + 3).is_err());
    // Clamped but non-empty is fine and reports the clamped height.
    let mut band = RowBandReader::new(open(), nz - 1, nz + 10).unwrap();
    assert_eq!(band.read_sizes().unwrap().1, 1);
}

/// `margin_rows` is the Fresnel-kernel pixel support ⌈π·λ·dist/ps²⌉ (tomopy
/// `_calc_pad_width`'s `pad_pix`); 0 without phase retrieval. For the CLI
/// defaults (ps 1e-4 cm, dist 50 cm, 30 keV): λ = 2π·ħc/E = 4.1328e-9 cm,
/// ⌈π·4.1328e-9·50/1e-8⌉ = ⌈64.92⌉ = 65.
#[test]
fn margin_rows_matches_kernel_support() {
    assert_eq!(margin_rows(&PhaseMethod::None), 0);
    let paganin = PhaseMethod::Paganin {
        pixel_size: 1e-4,
        dist: 50.0,
        energy: 30.0,
        alpha: 1e-3,
    };
    assert_eq!(margin_rows(&paganin), 65);
    // Same physics fields ⇒ same margin for the other retrieval filters.
    let farago = PhaseMethod::Farago {
        pixel_size: 1e-4,
        dist: 50.0,
        energy: 30.0,
        db: 1000.0,
    };
    assert_eq!(margin_rows(&farago), 65);
}
