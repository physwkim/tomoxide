//! Level-B pipeline benchmark: sequential `ReconSteps::run_streaming` vs
//! `ReconSteps::run_streaming_pipelined` over a **real on-disk** dataset, so the
//! read‖compute‖write overlap is exercised against actual storage I/O (the
//! in-memory `bench_parallel` cannot show it — it has no disk stage).
//!
//! A custom `DiskFileReader` reads each chunk's detector rows from a binary
//! projection file (one seek+read per angle = a hyperslab), and a
//! `DiskFileWriter` writes each reconstructed chunk to an output file. Both wrap
//! their I/O time in shared atomic counters, so we can report the per-stage
//! breakdown and compare wall-clock.
//!
//! Usage: bench_pipeline [backend] [nproj] [nz] [nx] [chunk]
//!   defaults: cpu 512 256 512 32
//!
//! Note: numbers are page-cache-warm (the file was just written), which is the
//! *conservative* case — cold cache / slower storage makes the read stage longer
//! and the overlap win larger.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array3, Axis};
use tomoxide::io::{DatasetReader, VolumeWriter};
use tomoxide::{
    Algorithm, Angles, BackendKind, Dataset, Engine, Frames, Geometry, Layout, PrepOptions,
    ReconParams, ReconSteps, Tomo, Volume,
};

const F: usize = std::mem::size_of::<f32>();

/// Reader backed by a `[nproj, nz, nx]` little-endian f32 file on disk. Each
/// `read_chunk` does one seek+read per angle for the row band `[r0, r1)`.
struct DiskFileReader {
    path: String,
    nproj: usize,
    nz: usize,
    nx: usize,
    theta: Vec<f32>,
    read_ns: Arc<AtomicU64>,
}

impl DatasetReader for DiskFileReader {
    fn read_sizes(&mut self) -> tomoxide::Result<(usize, usize, usize, usize, usize)> {
        Ok((self.nproj, self.nz, self.nx, 1, 1))
    }
    fn read_theta(&mut self) -> tomoxide::Result<Vec<f32>> {
        Ok(self.theta.clone())
    }
    fn read_all(&mut self) -> tomoxide::Result<Dataset<f32>> {
        Err(tomoxide::Error::Io("bench: read_all unused".into()))
    }
    fn read_chunk(&mut self, r0: usize, r1: usize) -> tomoxide::Result<Dataset<f32>> {
        let t = Instant::now();
        let rows = r1 - r0;
        let mut f = File::open(&self.path).map_err(|e| tomoxide::Error::Io(e.to_string()))?;
        let mut data = Array3::<f32>::zeros((self.nproj, rows, self.nx));
        let mut buf = vec![0u8; rows * self.nx * F];
        for p in 0..self.nproj {
            let off = ((p * self.nz + r0) * self.nx) * F;
            f.seek(SeekFrom::Start(off as u64))
                .map_err(|e| tomoxide::Error::Io(e.to_string()))?;
            f.read_exact(&mut buf)
                .map_err(|e| tomoxide::Error::Io(e.to_string()))?;
            let dst = data.index_axis_mut(Axis(0), p);
            let mut it = dst.into_iter();
            for c in buf.chunks_exact(F) {
                *it.next().unwrap() = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        // flat = ones, dark = zeros ⇒ normalize is just minus-log of `data`.
        let ds = Dataset {
            data: Tomo::new(data, Layout::Projection),
            flat: Some(Frames::new(Array3::from_elem((1, rows, self.nx), 1.0))),
            dark: Some(Frames::new(Array3::zeros((1, rows, self.nx)))),
            theta: self.theta.clone(),
        };
        self.read_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(ds)
    }
}

/// Writer that writes each `[rows, nx, nx]` chunk to its row offset in an output
/// file, timing the write.
struct DiskFileWriter {
    file: File,
    nx: usize,
    write_ns: Arc<AtomicU64>,
}

impl VolumeWriter for DiskFileWriter {
    fn write_chunk(&mut self, vol: &Volume<f32>, start: usize, _end: usize) -> tomoxide::Result<()> {
        let t = Instant::now();
        let slice = vol
            .array
            .as_standard_layout()
            .to_owned()
            .into_raw_vec_and_offset()
            .0;
        let mut bytes = Vec::with_capacity(slice.len() * F);
        for v in &slice {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let off = (start * self.nx * self.nx * F) as u64;
        self.file
            .seek(SeekFrom::Start(off))
            .map_err(|e| tomoxide::Error::Io(e.to_string()))?;
        self.file
            .write_all(&bytes)
            .map_err(|e| tomoxide::Error::Io(e.to_string()))?;
        self.write_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let backend = a.get(1).map(|s| s.as_str()).unwrap_or("cpu");
    let nproj: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);
    let nz: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let nx: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(512);
    let chunk: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(32);

    let kind = match backend {
        "cuda" => BackendKind::Cuda,
        _ => BackendKind::Cpu,
    };
    let engine = Engine::new(kind).expect("engine");

    let dir = std::env::var("BENCH_DIR").unwrap_or_else(|_| ".".into());
    let proj_path = format!("{dir}/bench_proj.f32");
    let out_seq = format!("{dir}/bench_out_seq.f32");
    let out_pipe = format!("{dir}/bench_out_pipe.f32");

    // --- write the projection file once (deterministic transmission in (0,1]) ---
    {
        let t = Instant::now();
        let mut f = File::create(&proj_path).expect("create proj");
        let mut row = vec![0u8; nx * F];
        for p in 0..nproj {
            for z in 0..nz {
                for x in 0..nx {
                    let v = 0.2 + 0.78 * (((p * 31 + z * 7 + x * 3) % 101) as f32 / 101.0);
                    row[x * F..x * F + F].copy_from_slice(&v.to_le_bytes());
                }
                f.write_all(&row).expect("write proj");
            }
        }
        f.flush().ok();
        let gb = (nproj * nz * nx * F) as f64 / 1e9;
        println!(
            "setup: wrote {gb:.2} GB projection file ({nproj}x{nz}x{nx}) in {:.2}s",
            t.elapsed().as_secs_f64()
        );
    }

    let theta: Vec<f32> = (0..nproj)
        .map(|i| std::f32::consts::PI * i as f32 / nproj as f32)
        .collect();
    let geom = Geometry::parallel(Angles(theta.clone()), nx, nz, 1.0);
    let params = ReconParams {
        num_gridx: Some(nx),
        ..Default::default()
    };
    let prep = PrepOptions::default();

    println!(
        "backend={backend} nproj={nproj} nz={nz} nx={nx} chunk={chunk} ({} chunks)",
        nz.div_ceil(chunk)
    );

    // --- sequential streaming ---
    let read_ns = Arc::new(AtomicU64::new(0));
    let write_ns = Arc::new(AtomicU64::new(0));
    let mut reader = DiskFileReader {
        path: proj_path.clone(),
        nproj,
        nz,
        nx,
        theta: theta.clone(),
        read_ns: Arc::clone(&read_ns),
    };
    let mut writer = DiskFileWriter {
        file: File::create(&out_seq).expect("create out_seq"),
        nx,
        write_ns: Arc::clone(&write_ns),
    };
    let t = Instant::now();
    ReconSteps::new(chunk)
        .run_streaming(
            &mut reader,
            &mut writer,
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .expect("run_streaming");
    let seq_total = t.elapsed().as_secs_f64();
    let seq_read = read_ns.load(Ordering::Relaxed);
    let seq_write = write_ns.load(Ordering::Relaxed);
    let seq_compute = (seq_total * 1e3 - ms(seq_read) - ms(seq_write)).max(0.0);
    println!(
        "\n[sequential] total {seq_total:.3}s = read {:.0}ms + compute {:.0}ms + write {:.0}ms",
        ms(seq_read),
        seq_compute,
        ms(seq_write),
    );

    // --- pipelined streaming (reader/writer built on their own threads) ---
    let read_ns2 = Arc::new(AtomicU64::new(0));
    let write_ns2 = Arc::new(AtomicU64::new(0));
    let rp = proj_path.clone();
    let th = theta.clone();
    let rns = Arc::clone(&read_ns2);
    let wns = Arc::clone(&write_ns2);
    let wp = out_pipe.clone();
    let t = Instant::now();
    ReconSteps::new(chunk)
        .run_streaming_pipelined(
            move || {
                Ok(Box::new(DiskFileReader {
                    path: rp,
                    nproj,
                    nz,
                    nx,
                    theta: th,
                    read_ns: rns,
                }) as Box<dyn DatasetReader>)
            },
            move || {
                Ok(Box::new(DiskFileWriter {
                    file: File::create(&wp).map_err(|e| tomoxide::Error::Io(e.to_string()))?,
                    nx,
                    write_ns: wns,
                }) as Box<dyn VolumeWriter>)
            },
            &geom,
            Algorithm::Fbp,
            &params,
            &prep,
            &engine,
        )
        .expect("run_streaming_pipelined");
    let pipe_total = t.elapsed().as_secs_f64();
    let p_read = ms(read_ns2.load(Ordering::Relaxed));
    let p_write = ms(write_ns2.load(Ordering::Relaxed));
    println!(
        "[pipelined ] total {pipe_total:.3}s  (read-thread {p_read:.0}ms, write-thread {p_write:.0}ms ran concurrently with compute)"
    );

    let bound = (ms(seq_read).max(seq_compute)).max(ms(seq_write)) / 1e3;
    println!(
        "\nspeedup {:.2}x   (ideal overlap bound = max(read,compute,write) = {:.3}s)",
        seq_total / pipe_total,
        bound
    );

    // sanity: the two outputs must be byte-identical.
    let mut bs = Vec::new();
    let mut bp = Vec::new();
    File::open(&out_seq).unwrap().read_to_end(&mut bs).unwrap();
    File::open(&out_pipe).unwrap().read_to_end(&mut bp).unwrap();
    println!(
        "output identical: {}",
        bs == bp && !bs.is_empty()
    );

    for p in [&proj_path, &out_seq, &out_pipe] {
        std::fs::remove_file(p).ok();
    }
}
