//! Write a synthetic DXchange-format HDF5 file for benchmarking the streaming
//! reconstruction CLI (`tomoxide recon_steps`).
//!
//! The reader path is: raw `/exchange/data` → flat/dark normalize → minus-log →
//! sinogram → recon. With `data_white = 1` and `data_dark = 0`, normalization is
//! the identity, so `minus_log(raw) = -ln(raw)` recovers the sinogram we encode
//! as `raw = exp(-S)`. `S` is a proper parallel-beam sinogram of a few Gaussian
//! rods (their projected detector position shifts with the angle), so the
//! reconstruction is a non-degenerate set of blobs — enough to compare the
//! fp32 and fp16 outputs for correlation, not just to time them. `S` is scaled
//! so `raw ∈ (~0.05, 1]`, well clear of the f32 underflow / `minus_log` 1e-6
//! clamp.
//!
//!   cargo run --release --example make_synthetic_dxchange -- <nproj> <nz> <nx> <out.h5>
//!
//! Layout written (DXchange convention; little-endian f32):
//!   /exchange/data        [nproj, nz, nx]   raw transmission = exp(-S)
//!   /exchange/data_white  [1, nz, nx]       all 1.0
//!   /exchange/data_dark   [1, nz, nx]       all 0.0
//!   /exchange/theta       [nproj]           degrees, linspace [0, 180)

use rust_hdf5::H5File;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let nproj: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let nz: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let nx: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let out = a.get(4).cloned().unwrap_or_else(|| "synthetic.h5".into());

    // A few absorbing rods at (ox, oy) in detector-pixel units about the centre,
    // each with an amplitude and Gaussian width. Projected position at angle θ is
    // cx + ox·cosθ + oy·sinθ; the sinogram sums their Gaussian footprints.
    let cx = nx as f32 / 2.0;
    let rods = [
        (0.0f32, 0.0f32, 1.0f32, nx as f32 * 0.04), // central
        (nx as f32 * 0.22, 0.0, 0.7, nx as f32 * 0.03),
        (nx as f32 * -0.15, nx as f32 * 0.18, 0.5, nx as f32 * 0.025),
    ];
    let scale = 2.5f32; // peak optical depth ≈ Σ amp · scale ≈ 5.5 → raw_min ≈ exp(-5.5)

    // 2-D sinogram S[p, x] (z-invariant); raw[p, z, x] = exp(-S[p,x]) for all z.
    let mut sino2d = vec![0.0f32; nproj * nx];
    for p in 0..nproj {
        let theta = p as f32 * std::f32::consts::PI / nproj as f32;
        let (c, s) = (theta.cos(), theta.sin());
        let row = &mut sino2d[p * nx..(p + 1) * nx];
        for (x, val) in row.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for &(ox, oy, amp, sigma) in &rods {
                let d = cx + ox * c + oy * s;
                let t = (x as f32 - d) / sigma;
                acc += amp * (-(t * t)).exp();
            }
            *val = acc * scale;
        }
    }

    // /exchange/data [nproj, nz, nx] = exp(-S), broadcast over z.
    let mut data = vec![0.0f32; nproj * nz * nx];
    for p in 0..nproj {
        let srow = &sino2d[p * nx..(p + 1) * nx];
        for z in 0..nz {
            let dst = &mut data[(p * nz + z) * nx..(p * nz + z + 1) * nx];
            for (d, &s) in dst.iter_mut().zip(srow.iter()) {
                *d = (-s).exp();
            }
        }
    }
    let white = vec![1.0f32; nz * nx];
    let dark = vec![0.0f32; nz * nx];
    let theta: Vec<f32> = (0..nproj)
        .map(|p| p as f32 * 180.0 / nproj as f32)
        .collect();

    let file = H5File::create(&out).expect("create h5");
    let g = file.create_group("exchange").expect("create /exchange");
    g.new_dataset::<f32>()
        .shape([nproj, nz, nx])
        .create("data")
        .expect("create data")
        .write_raw::<f32>(&data)
        .expect("write data");
    g.new_dataset::<f32>()
        .shape([1, nz, nx])
        .create("data_white")
        .expect("create data_white")
        .write_raw::<f32>(&white)
        .expect("write data_white");
    g.new_dataset::<f32>()
        .shape([1, nz, nx])
        .create("data_dark")
        .expect("create data_dark")
        .write_raw::<f32>(&dark)
        .expect("write data_dark");
    g.new_dataset::<f32>()
        .shape([nproj])
        .create("theta")
        .expect("create theta")
        .write_raw::<f32>(&theta)
        .expect("write theta");
    file.flush().expect("flush");

    println!(
        "wrote {out}  /exchange/data=[{nproj},{nz},{nx}]  white/dark=[1,{nz},{nx}]  theta[{nproj}] deg  ({:.2} GiB data)",
        (data.len() * 4) as f64 / (1u64 << 30) as f64
    );
}
