//! `focus_by_z` is `slice_focus` per slice, in slice order — the score the tilt
//! scan, the centre sweep and a volume inspection all rank by.

use ndarray::{Array3, Axis};
use tomoxide::recon::center::{focus_by_z, slice_focus};

#[test]
fn focus_by_z_scores_each_slice_in_order() {
    // Slice z is a checkerboard of contrast z, so its focus grows as z².
    let vol = Array3::from_shape_fn((9, 40, 48), |(z, y, x)| {
        z as f32 * ((((y / 3) + (x / 3)) % 2) as f32)
    });
    let f = focus_by_z(vol.view());
    assert_eq!(f.len(), 9);
    for (z, &got) in f.iter().enumerate() {
        assert_eq!(got, slice_focus(&vol.index_axis(Axis(0), z)), "slice {z}");
    }
    assert_eq!(f[0], 0.0);
    for z in 1..9 {
        let want = f[1] * (z * z) as f64;
        assert!(
            (f[z] - want).abs() <= 1e-9 * want,
            "slice {z}: {} vs {want}",
            f[z]
        );
    }
}
