//! Parity test for the DXchange HDF5 reader against a tomopy/h5py-written
//! fixture (`tools/gen_dxchange_fixture.py`).
//!
//! The reader is projector-independent pure I/O, so it is held to bit-exact
//! parity (Δ=0): uint16 image data casts losslessly to f32, and theta is the
//! same `deg/180*pi` f32 computation tomocupy's reader does. The fixture is
//! gzip-compressed + chunked, so this also exercises rust-hdf5's deflate path.

use ndarray::{Array1, Array3};
use ndarray_npy::read_npy;
use tomoxide_core::data::Layout;
use tomoxide_io::open_dxchange;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn fixture(name: &str) -> String {
    format!("{FIXTURES}/{name}")
}

#[test]
fn read_sizes_matches_fixture() {
    let mut r = open_dxchange(&fixture("dxchange_small.h5")).unwrap();
    // data [5,3,4], white [2,3,4], dark [2,3,4].
    assert_eq!(r.read_sizes().unwrap(), (5, 3, 4, 2, 2));
}

#[test]
fn read_theta_converts_degrees_to_radians() {
    let mut r = open_dxchange(&fixture("dxchange_small.h5")).unwrap();
    let got = r.read_theta().unwrap();
    let want: Array1<f32> = read_npy(fixture("dxchange_theta_rad.npy")).unwrap();
    // Bit-exact: same f32 deg/180*pi computation as tomocupy.
    assert_eq!(got, want.to_vec(), "theta radians mismatch");
}

#[test]
fn read_all_matches_fixture_bit_exact() {
    let mut r = open_dxchange(&fixture("dxchange_small.h5")).unwrap();
    let ds = r.read_all().unwrap();

    // DXchange data is [angle, row, col] = projection layout.
    assert_eq!(ds.data.layout, Layout::Projection);

    let data: Array3<f32> = read_npy(fixture("dxchange_data_f32.npy")).unwrap();
    let white: Array3<f32> = read_npy(fixture("dxchange_white_f32.npy")).unwrap();
    let dark: Array3<f32> = read_npy(fixture("dxchange_dark_f32.npy")).unwrap();

    assert_eq!(ds.data.array, data, "projection data mismatch");
    assert_eq!(
        ds.flat.expect("flat present").array,
        white,
        "flat (white) mismatch"
    );
    assert_eq!(ds.dark.expect("dark present").array, dark, "dark mismatch");

    let theta: Array1<f32> = read_npy(fixture("dxchange_theta_rad.npy")).unwrap();
    assert_eq!(ds.theta, theta.to_vec(), "theta mismatch");
}
