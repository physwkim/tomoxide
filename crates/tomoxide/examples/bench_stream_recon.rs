//! Isolate the streaming reconstructor (`CudaFbpStream`, the out-of-core
//! `recon_steps` path) from disk I/O and HDF5 parsing.
//!
//! Drives `ReconSteps::run_streaming` with an in-memory `DatasetReader` (no h5
//! read) and a discarding `VolumeWriter` (no h5 write). Per-run wall =
//! minus-log + recon + (free discard write). minus-log is byte-identical between
//! fp32 and fp16, so the fp16−fp32 wall delta is exactly the recon delta — i.e.
//! whether half precision speeds the streaming Fbp/Linerec or not.
//!
//!   cargo run --release --features cuda --example bench_stream_recon -- <nproj> <nz> <nx> <chunk> <reps> <float32|float16>

use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array3, Axis};
use tomoxide::data::{Dataset, Tomo, Volume};
use tomoxide::io::{DatasetReader, VolumeWriter};
use tomoxide::{
    Algorithm, Angles, BackendKind, Dtype, Engine, Geometry, Layout, PrepOptions, ReconParams,
    ReconSteps,
};

/// In-memory DXchange reader: projections held as `[nproj, nz, nx]` (shared via
/// `Arc` so the pipelined factory can rebuild it without a 1 GiB clone), no
/// flat/dark (so normalize is just minus-log). `read_chunk` slices rows.
struct MemReader {
    data: Arc<Array3<f32>>, // [nproj, nz, nx]
    theta: Vec<f32>,        // radians
}

impl DatasetReader for MemReader {
    fn read_sizes(&mut self) -> tomoxide::Result<(usize, usize, usize, usize, usize)> {
        let (nproj, nz, nx) = self.data.dim();
        Ok((nproj, nz, nx, 0, 0))
    }
    fn read_theta(&mut self) -> tomoxide::Result<Vec<f32>> {
        Ok(self.theta.clone())
    }
    fn read_all(&mut self) -> tomoxide::Result<Dataset<f32>> {
        Ok(Dataset {
            data: Tomo::new((*self.data).clone(), Layout::Projection),
            flat: None,
            dark: None,
            theta: self.theta.clone(),
        })
    }
    fn read_chunk(&mut self, row0: usize, row1: usize) -> tomoxide::Result<Dataset<f32>> {
        let sub = self
            .data
            .slice_axis(Axis(1), ndarray::Slice::from(row0..row1))
            .to_owned();
        Ok(Dataset {
            data: Tomo::new(sub, Layout::Projection),
            flat: None,
            dark: None,
            theta: self.theta.clone(),
        })
    }
}

/// Writer that discards (counts slices only), so no disk write is timed.
struct NullWriter {
    slices: usize,
}
impl VolumeWriter for NullWriter {
    fn write_chunk(
        &mut self,
        vol: &Volume<f32>,
        _start: usize,
        _end: usize,
    ) -> tomoxide::Result<()> {
        self.slices += vol.dims().0;
        Ok(())
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let nproj: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let nz: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let nx: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let chunk: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let reps: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(5);
    // dtype arg: float32 | float16 | both. "both" interleaves the two precisions
    // rep-by-rep on a kept-hot GPU and reports the paired median delta, which
    // cancels the GPU clock-state drift (idle 210 MHz ↔ boost 3.1 GHz) that makes
    // separate single-dtype processes unreliable.
    let mode = a
        .get(6)
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| "both".into());

    let engine = Engine::new(BackendKind::Cuda).unwrap();

    // Smooth transmission raw in (0, 1] so minus-log is finite and non-trivial.
    let data = Array3::<f32>::from_shape_fn((nproj, nz, nx), |(p, _z, x)| {
        let v = 0.3 + 0.6 * ((p as f32 * 0.013 + x as f32 * 0.011).sin() * 0.5 + 0.5);
        v.clamp(0.05, 1.0)
    });
    let theta: Vec<f32> = (0..nproj)
        .map(|p| p as f32 * std::f32::consts::PI / nproj as f32)
        .collect();
    let geom = Geometry::parallel(Angles(theta.clone()), nx, nz, 1.0);
    let mk = |dt| ReconParams {
        num_gridx: Some(nx),
        dtype: dt,
        ..Default::default()
    };
    let prep = PrepOptions::default();
    let (p32, p16) = (mk(Dtype::F32), mk(Dtype::F16));
    let data = Arc::new(data);

    let report = |tag: &str, ts: &[f64]| {
        let med = median(ts.to_vec());
        println!(
            "  {tag:<8} median={:.3}s  slices/s={:.1}  runs={:?}",
            med,
            nz as f64 / med,
            ts.iter()
                .map(|t| (t * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        med
    };
    let verdict = |m32: f64, m16: f64| {
        println!(
            "  => fp16/fp32 = {:.3}  ({} by {:.1}%)",
            m16 / m32,
            if m16 < m32 {
                "fp16 FASTER"
            } else {
                "fp16 slower"
            },
            ((m16 - m32) / m32 * 100.0).abs()
        );
    };

    // Sequential (`run_streaming`): one reader/writer reused across reps.
    let mut reader = MemReader {
        data: data.clone(),
        theta: theta.clone(),
    };
    let mut writer = NullWriter { slices: 0 };
    let mut seq = |p: &ReconParams| {
        writer.slices = 0;
        ReconSteps::new(chunk)
            .run_streaming(
                &mut reader,
                &mut writer,
                &geom,
                Algorithm::Fbp,
                p,
                &prep,
                &engine,
            )
            .unwrap();
    };

    // Pipelined (`run_streaming_pipelined`): factories rebuild the in-memory
    // reader/writer each call (Arc clone, no 1 GiB copy). Same 3-thread
    // read‖compute‖write conveyor the CLI `recon_steps` uses, but with no h5.
    let pipe = |p: &ReconParams| {
        let (d, th) = (data.clone(), theta.clone());
        let make_reader = move || -> tomoxide::Result<Box<dyn DatasetReader>> {
            Ok(Box::new(MemReader { data: d, theta: th }))
        };
        let make_writer = move || -> tomoxide::Result<Box<dyn VolumeWriter>> {
            Ok(Box::new(NullWriter { slices: 0 }))
        };
        ReconSteps::new(chunk)
            .run_streaming_pipelined(
                make_reader,
                make_writer,
                &geom,
                Algorithm::Fbp,
                p,
                &prep,
                &engine,
            )
            .unwrap();
    };

    println!("nproj={nproj} nz={nz} nx={nx} chunk={chunk} mode={mode} reps={reps}");
    match mode.as_str() {
        // Interleaved sequential.
        "both" | "seq" => {
            seq(&p32);
            seq(&p16);
            let (mut t32, mut t16) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
            for _ in 0..reps {
                let a = Instant::now();
                seq(&p32);
                t32.push(a.elapsed().as_secs_f64());
                let b = Instant::now();
                seq(&p16);
                t16.push(b.elapsed().as_secs_f64());
            }
            println!("[run_streaming — sequential, single thread]");
            verdict(report("fp32", &t32), report("fp16", &t16));
        }
        // Interleaved pipelined (mirrors the CLI recon_steps path).
        "pipe" | "pipeboth" => {
            pipe(&p32);
            pipe(&p16);
            let (mut t32, mut t16) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
            for _ in 0..reps {
                let a = Instant::now();
                pipe(&p32);
                t32.push(a.elapsed().as_secs_f64());
                let b = Instant::now();
                pipe(&p16);
                t16.push(b.elapsed().as_secs_f64());
            }
            println!("[run_streaming_pipelined — 3-thread conveyor]");
            verdict(report("fp32", &t32), report("fp16", &t16));
        }
        other => {
            let p = if matches!(other, "float16" | "f16" | "half") {
                &p16
            } else {
                &p32
            };
            seq(p);
            let ts: Vec<f64> = (0..reps)
                .map(|_| {
                    let t = Instant::now();
                    seq(p);
                    t.elapsed().as_secs_f64()
                })
                .collect();
            report(
                if std::ptr::eq(p, &p16) {
                    "fp16"
                } else {
                    "fp32"
                },
                &ts,
            );
        }
    }
}
