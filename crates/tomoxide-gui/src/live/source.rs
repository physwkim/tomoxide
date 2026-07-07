//! rsdm PVA data engine → ring buffer (docs/GUI.md §2.6).
//!
//! Subscribes to a tomoScanStream-style projection stream over pvAccess and
//! feeds decoded frames into the [`ProjRing`]. rsdm delivers an NTNDArray frame
//! as a *flat* pixel array (its value model drops the `dimension[]`, so
//! width/height are not recoverable from the frame itself); the detector
//! dimensions therefore come from companion width/height PVs.
//!
//! Frame↔angle pairing: rsdm does not surface the NTNDArray `uniqueId`, so a
//! frame cannot be matched to its exact rotation angle. Each drained frame is
//! paired with the *latest* scalar angle read from the theta channel. At
//! interactive poll rates (a few frames per poll) this is close; it is the
//! honest v1 limitation until an angle-per-frame channel or `uniqueId` exposure
//! lands.

use ndarray::Array2;
use rsdm::{Channel, Engine, PvValue, ValueSubscription};

use super::ring::ProjRing;

/// PVA channel addresses for the live stream. Dark/flat are optional (an empty
/// string ⇒ that reference is absent, and frames are treated as already-
/// normalized transmission).
#[derive(Clone, Debug, Default)]
pub struct PvaAddrs {
    pub image: String,
    pub theta: String,
    /// Companion PV carrying the frame width (fastest detector dimension, `nx`).
    pub width: String,
    /// Companion PV carrying the frame height (`ny`).
    pub height: String,
    pub dark: String,
    pub flat: String,
}

/// Manual overrides for the frame geometry and rotation angle, for setups where
/// those are not published as PVs (or are simply known ahead of time).
///
/// - **Dimensions.** rsdm delivers each frame as a *flat* pixel array, so the
///   detector width `nx` cannot be recovered from the frame alone; supply it
///   manually (or via a width PV). The height is then either given (`ny > 0`) or
///   **derived from the frame length** (`ny = len / nx`) — using the image's own
///   size directly, so only the width is ever strictly required.
/// - **Theta.** With no theta PV, each accepted frame is assigned
///   `start + step × index` degrees (a constant angular step per frame). Only
///   the relative spacing matters to parallel-beam reconstruction.
#[derive(Clone, Debug, Default)]
pub struct ManualCfg {
    pub dims_manual: bool,
    pub nx: usize,
    /// Manual height; `0` ⇒ derive from the frame length.
    pub ny: usize,
    pub theta_manual: bool,
    pub theta_start: f64,
    /// Degrees per frame.
    pub theta_step: f64,
}

/// A connected live source: the rsdm engine (owns its tokio runtime) and the
/// channels it drives. Dropping it cancels every channel task. Channels that a
/// manual override replaces (theta / width / height) are left unconnected.
pub struct LiveSource {
    _engine: Engine,
    image: Channel,
    image_sub: ValueSubscription,
    theta: Option<Channel>,
    width: Option<Channel>,
    height: Option<Channel>,
    dark: Option<Channel>,
    flat: Option<Channel>,
    manual: ManualCfg,
    frame_counter: u64,
    dark_stamp: u64,
    flat_stamp: u64,
}

impl LiveSource {
    /// Open a fresh engine and connect every channel. Connections are async; the
    /// channels report `is_connected() == false` until the server answers.
    pub fn connect(addrs: &PvaAddrs, manual: ManualCfg) -> Result<Self, String> {
        Self::with_engine(Engine::new(), addrs, manual)
    }

    /// Connect the channels on a caller-supplied engine. Splitting the engine
    /// out lets a test point rsdm's PVA plugin at an in-process server
    /// (`PvaPlugin::with_server`) instead of doing a UDP search. Channels a
    /// manual override replaces are not connected.
    pub fn with_engine(
        engine: Engine,
        addrs: &PvaAddrs,
        manual: ManualCfg,
    ) -> Result<Self, String> {
        if addrs.image.is_empty() {
            return Err("image channel address is required".to_owned());
        }
        if !manual.theta_manual && addrs.theta.is_empty() {
            return Err("a theta PV is required unless manual theta is enabled".to_owned());
        }
        if manual.dims_manual {
            if manual.nx == 0 {
                return Err("manual frame width must be non-zero".to_owned());
            }
        } else if addrs.width.is_empty() {
            return Err("a width PV is required unless manual frame size is enabled".to_owned());
        }

        let mk = |a: &str| engine.connect(a).map_err(|e| format!("{a}: {e}"));
        let image = mk(&addrs.image)?;
        let image_sub = image.subscribe_values(64);
        let theta = if manual.theta_manual {
            None
        } else {
            Some(mk(&addrs.theta)?)
        };
        let width = if manual.dims_manual {
            None
        } else {
            Some(mk(&addrs.width)?)
        };
        // Height PV is optional even in PV mode: absent ⇒ derive from the frame
        // length.
        let height = if manual.dims_manual || addrs.height.is_empty() {
            None
        } else {
            Some(mk(&addrs.height)?)
        };
        let dark = if addrs.dark.is_empty() {
            None
        } else {
            Some(mk(&addrs.dark)?)
        };
        let flat = if addrs.flat.is_empty() {
            None
        } else {
            Some(mk(&addrs.flat)?)
        };
        Ok(Self {
            _engine: engine,
            image,
            image_sub,
            theta,
            width,
            height,
            dark,
            flat,
            manual,
            frame_counter: 0,
            dark_stamp: 0,
            flat_stamp: 0,
        })
    }

    pub fn image_connected(&self) -> bool {
        self.image.is_connected()
    }

    /// Detector width `nx` — the manual value or the width PV. `None` until it is
    /// known (a positive value).
    fn nx(&self) -> Option<usize> {
        if self.manual.dims_manual {
            (self.manual.nx > 0).then_some(self.manual.nx)
        } else {
            let w = self
                .width
                .as_ref()?
                .read(|s| s.value.as_ref().and_then(PvValue::as_i64))?;
            (w > 0).then_some(w as usize)
        }
    }

    /// Detector height `ny` for a frame of flat length `len`: the explicit manual
    /// height, else a height PV, else derived from the frame's own length
    /// (`len / nx`).
    fn ny(&self, nx: usize, len: usize) -> Option<usize> {
        if self.manual.dims_manual && self.manual.ny > 0 {
            Some(self.manual.ny)
        } else if let Some(h) = self.height.as_ref() {
            let hv = h.read(|s| s.value.as_ref().and_then(PvValue::as_i64))?;
            (hv > 0).then_some(hv as usize)
        } else if nx > 0 && len.is_multiple_of(nx) {
            Some(len / nx)
        } else {
            None
        }
    }

    /// The angle for the next accepted frame: the latest theta scalar, or the
    /// manual `start + step × index` (index monotonic across the session).
    fn next_theta(&mut self) -> f64 {
        if self.manual.theta_manual {
            let t = self.manual.theta_start + self.manual.theta_step * self.frame_counter as f64;
            self.frame_counter += 1;
            t
        } else {
            self.theta
                .as_ref()
                .and_then(|c| c.read(|s| s.value.as_ref().and_then(PvValue::as_f64)))
                .unwrap_or(0.0)
        }
    }

    /// Drain buffered frames into the ring; returns the number pushed. Frames
    /// whose flat length is not a multiple of `nx` are skipped.
    pub fn poll_into(&mut self, ring: &mut ProjRing) -> usize {
        let Some(nx) = self.nx() else {
            return 0;
        };

        // `drain` takes an `FnMut` and dims/angle resolution needs `&self`, so
        // collect the decoded pixels first, then push.
        let mut pending: Vec<Vec<f32>> = Vec::new();
        self.image_sub.drain(|ev| {
            let px = pv_to_f32(&ev.value);
            if !px.is_empty() {
                pending.push(px);
            }
        });

        let mut n = 0;
        for px in pending {
            let Some(ny) = self.ny(nx, px.len()) else {
                continue;
            };
            if px.len() != nx * ny {
                continue;
            }
            ring.set_geometry(ny, nx);
            let theta = self.next_theta();
            if let Ok(frame) = Array2::from_shape_vec((ny, nx), px) {
                ring.push(theta, frame);
                n += 1;
            }
        }
        n
    }

    /// Refresh the rolling dark/flat references when their update counter has
    /// advanced (avoids reshaping an unchanged frame every loop).
    pub fn poll_darkflat(&mut self, ring: &mut ProjRing) {
        let (ny, nx) = ring.dims();
        if ny == 0 {
            return;
        }
        if let Some(ch) = &self.dark {
            let stamp = ch.stamp();
            if stamp != self.dark_stamp
                && let Some(a) = ch.read(|s| frame_of(s.value.as_ref(), ny, nx))
            {
                ring.set_dark(a);
                self.dark_stamp = stamp;
            }
        }
        if let Some(ch) = &self.flat {
            let stamp = ch.stamp();
            if stamp != self.flat_stamp
                && let Some(a) = ch.read(|s| frame_of(s.value.as_ref(), ny, nx))
            {
                ring.set_flat(a);
                self.flat_stamp = stamp;
            }
        }
    }
}

/// Decode a PVA array value into `f32` pixels. rsdm widens every integer dtype
/// to `i64` and every float to `f64`; a `u8` frame arrives as `Bytes`.
fn pv_to_f32(v: &PvValue) -> Vec<f32> {
    match v {
        PvValue::Bytes(b) => b.iter().map(|&x| x as f32).collect(),
        PvValue::IntArray(a) => a.iter().map(|&x| x as f32).collect(),
        PvValue::FloatArray(a) => a.iter().map(|&x| x as f32).collect(),
        _ => Vec::new(),
    }
}

fn frame_of(v: Option<&PvValue>, ny: usize, nx: usize) -> Option<Array2<f32>> {
    let px = pv_to_f32(v?);
    if px.len() == ny * nx {
        Array2::from_shape_vec((ny, nx), px).ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    //! End-to-end over the real rsdm PVA path: an in-process
    //! `epics_pva_rs::PvaServer` publishes a synthetic NTNDArray frame plus
    //! width/height/theta scalars, and [`LiveSource`] subscribes through rsdm and
    //! decodes them into the ring — no beamline required. The server standup
    //! mirrors rsdm's own `tests/pva_ioc.rs`.

    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use epics_pva_rs::FieldDesc;
    use epics_pva_rs::nt::NTScalar;
    use epics_pva_rs::nt::nd_array::{
        NdAlarm, NdArrayBuffer, NdCodec, NdDimension, NdTimeStamp, NtNdArray, nt_nd_array_desc,
        nt_nd_array_value,
    };
    use epics_pva_rs::pvdata::ScalarType;
    use epics_pva_rs::server_native::{PvaServer, SharedPV, SharedSource};
    use epics_pva_rs::{PvField, ScalarValue};
    use rsdm::{Engine, PvaPlugin};

    use super::*;
    use crate::live::ring::ProjRing;

    fn nd_array(buffer: NdArrayBuffer) -> NtNdArray {
        let size = (buffer.len() * buffer.element_size_bytes()) as i64;
        NtNdArray {
            dimension: vec![NdDimension {
                size: buffer.len() as i32,
                ..NdDimension::default()
            }],
            value: buffer,
            codec: NdCodec {
                name: String::new(),
                parameters: None,
            },
            compressed_size: size,
            uncompressed_size: size,
            unique_id: 1,
            data_time_stamp: NdTimeStamp::default(),
            alarm: NdAlarm::default(),
            time_stamp: NdTimeStamp::default(),
            attribute: Vec::new(),
        }
    }

    fn add_pv(source: &SharedSource, name: &str, desc: FieldDesc, value: PvField) {
        let pv = SharedPV::build_mailbox();
        pv.open(desc, value).expect("open SharedPV");
        source.add(name, pv);
    }

    fn scalar_double(source: &SharedSource, name: &str, v: f64) {
        let desc = NTScalar::new(ScalarType::Double).build();
        let mut init = NTScalar::new(ScalarType::Double).create();
        if let PvField::Structure(s) = &mut init {
            s.set("value", PvField::Scalar(ScalarValue::Double(v)));
        }
        add_pv(source, name, desc, init);
    }

    fn scalar_int(source: &SharedSource, name: &str, v: i32) {
        let desc = NTScalar::new(ScalarType::Int).build();
        let mut init = NTScalar::new(ScalarType::Int).create();
        if let PvField::Structure(s) = &mut init {
            s.set("value", PvField::Scalar(ScalarValue::Int(v)));
        }
        add_pv(source, name, desc, init);
    }

    /// Build the server inside a runtime (its tasks spawn there), then an engine
    /// on a plain thread pointed straight at it (TCP, no UDP search). The server
    /// and runtime must outlive the test.
    fn pva_engine(
        build: impl FnOnce(&SharedSource),
    ) -> (Engine, PvaServer, tokio::runtime::Runtime) {
        let server_rt = tokio::runtime::Runtime::new().expect("server runtime");
        let server = server_rt.block_on(async {
            let source = SharedSource::new();
            build(&source);
            PvaServer::isolated(Arc::new(source)).expect("isolated pva server")
        });
        std::thread::sleep(Duration::from_millis(300));
        let engine = Engine::new();
        engine.register_plugin(Arc::new(PvaPlugin::with_server(server.tcp_addr())));
        (engine, server, server_rt)
    }

    #[test]
    fn subscribes_and_decodes_a_frame_from_an_in_process_server() {
        let (nx, ny) = (4usize, 3usize);
        // A u16 frame with a distinguishable ramp; widened to IntArray by rsdm.
        let frame: Vec<u16> = (0..(nx * ny) as u16).map(|v| v + 10).collect();

        let (engine, _server, _rt) = pva_engine(|source| {
            add_pv(
                source,
                "SIM:Image",
                nt_nd_array_desc(),
                nt_nd_array_value(&nd_array(NdArrayBuffer::UShort(frame.clone()))),
            );
            scalar_double(source, "SIM:Theta", 90.0);
            scalar_int(source, "SIM:Width", nx as i32);
            scalar_int(source, "SIM:Height", ny as i32);
        });

        let addrs = PvaAddrs {
            image: "pva://SIM:Image".to_owned(),
            theta: "pva://SIM:Theta".to_owned(),
            width: "pva://SIM:Width".to_owned(),
            height: "pva://SIM:Height".to_owned(),
            dark: String::new(),
            flat: String::new(),
        };
        let mut source =
            LiveSource::with_engine(engine, &addrs, ManualCfg::default()).expect("connect");

        // Poll until the subscription has delivered the frame and the companion
        // dimension PVs have resolved.
        let mut ring = ProjRing::new(8);
        let start = Instant::now();
        let mut pushed = 0;
        while start.elapsed() < Duration::from_secs(10) {
            pushed += source.poll_into(&mut ring);
            if pushed >= 1 && ring.dims() == (ny, nx) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }

        assert!(pushed >= 1, "no frame decoded from the in-process server");
        assert_eq!(ring.dims(), (ny, nx), "detector dims from companion PVs");
        assert_eq!(ring.thetas()[0], 90.0, "frame paired with the theta scalar");

        // The decoded row-0 sinogram is finite and reflects the ramp (raw
        // treated as transmission, so distinct raw pixels give distinct values).
        let sino = ring.sinogram(0).expect("row 0 sinogram");
        assert_eq!(sino.dim(), (1, ring.len(), nx));
        assert!(sino.iter().all(|v| v.is_finite()));
        assert!(
            (sino[[0, 0, 0]] - -(10.0f32.ln())).abs() < 1e-5,
            "raw pixel 10 → −ln(10); got {}",
            sino[[0, 0, 0]]
        );
    }

    #[test]
    fn manual_dims_derive_height_and_manual_theta() {
        let (nx, ny) = (4usize, 3usize);
        let frame: Vec<u16> = (0..(nx * ny) as u16).map(|v| v + 1).collect();

        // Only the image PV exists; theta and both dimensions come from the
        // manual config, with the height derived from the frame length.
        let (engine, _server, _rt) = pva_engine(|source| {
            add_pv(
                source,
                "SIM2:Image",
                nt_nd_array_desc(),
                nt_nd_array_value(&nd_array(NdArrayBuffer::UShort(frame.clone()))),
            );
        });
        let addrs = PvaAddrs {
            image: "pva://SIM2:Image".to_owned(),
            ..Default::default()
        };
        let manual = ManualCfg {
            dims_manual: true,
            nx,
            ny: 0, // derive from the frame length
            theta_manual: true,
            theta_start: 5.0,
            theta_step: 0.5,
        };
        let mut source = LiveSource::with_engine(engine, &addrs, manual).expect("connect");

        let mut ring = ProjRing::new(8);
        let start = Instant::now();
        let mut pushed = 0;
        while start.elapsed() < Duration::from_secs(10) {
            pushed += source.poll_into(&mut ring);
            if pushed >= 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }

        assert!(pushed >= 1, "no frame decoded in manual mode");
        assert_eq!(
            ring.dims(),
            (ny, nx),
            "height derived from frame length / manual width"
        );
        assert_eq!(ring.thetas()[0], 5.0, "first manual angle = start");
    }
}
