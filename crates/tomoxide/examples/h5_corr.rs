//! Compare two reconstruction HDF5 files' `/exchange/data` datasets:
//! Pearson correlation, max absolute difference, and relative L2.
//!
//!   cargo run --release --example h5_corr -- a.h5 b.h5

use rust_hdf5::H5File;

fn read(path: &str) -> Vec<f32> {
    let f = H5File::open(path).expect("open");
    f.dataset("exchange/data")
        .expect("dataset")
        .read_raw::<f32>()
        .expect("read")
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let x = read(&a[1]);
    let y = read(&a[2]);
    assert_eq!(x.len(), y.len(), "length mismatch");
    let n = x.len() as f64;
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    for i in 0..x.len() {
        sx += x[i] as f64;
        sy += y[i] as f64;
    }
    let (mx, my) = (sx / n, sy / n);
    let (mut cxy, mut vx, mut vy) = (0.0f64, 0.0f64, 0.0f64);
    let (mut maxabs, mut l2d, mut l2x) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..x.len() {
        let (dx, dy) = (x[i] as f64 - mx, y[i] as f64 - my);
        cxy += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
        let d = (x[i] as f64 - y[i] as f64).abs();
        if d > maxabs {
            maxabs = d;
        }
        l2d += d * d;
        l2x += (x[i] as f64) * (x[i] as f64);
    }
    let pearson = cxy / (vx.sqrt() * vy.sqrt());
    let rel_l2 = (l2d / l2x).sqrt();
    println!(
        "n={}  pearson={:.6}  max_abs_diff={:.3e}  rel_L2={:.3e}",
        x.len(),
        pearson,
        maxabs,
        rel_l2
    );
}
