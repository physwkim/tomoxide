//! Fixed-capacity projection ring buffer for live streaming reconstruction
//! (docs/GUI.md §2.6).
//!
//! Holds the most recent ~180° of raw detector frames plus their rotation
//! angles and the rolling dark/flat references, so the recon loop can assemble a
//! sinogram for any detector row on demand. Oldest frames are evicted at
//! capacity.
//!
//! Memory: `capacity × ny × nx × 4` bytes — frames are stored as `f32`. A 2 k
//! detector at 180 projections is ~3 GB, so size `capacity` to the host budget.

use std::collections::VecDeque;

use ndarray::{Array2, Array3};

/// One buffered projection: its rotation angle (degrees — the tomoScanStream /
/// DXchange convention this GUI's readers already use) and the raw detector
/// frame `(ny, nx)`.
pub struct RingFrame {
    pub theta: f64,
    pub data: Array2<f32>,
}

/// A rolling window of the most recent projections, plus the current dark/flat.
pub struct ProjRing {
    capacity: usize,
    frames: VecDeque<RingFrame>,
    dark: Option<Array2<f32>>,
    flat: Option<Array2<f32>>,
    ny: usize,
    nx: usize,
}

impl ProjRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            frames: VecDeque::new(),
            dark: None,
            flat: None,
            ny: 0,
            nx: 0,
        }
    }

    /// Resize the window, evicting the oldest frames when shrinking below the
    /// current fill.
    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
        while self.frames.len() > self.capacity {
            self.frames.pop_front();
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Current detector grid `(ny, nx)` (`(0, 0)` until the first geometry is
    /// established).
    pub fn dims(&self) -> (usize, usize) {
        (self.ny, self.nx)
    }

    /// Establish (or change) the detector grid. Changing it clears the buffer
    /// and the dark/flat references — frames of a different size cannot share a
    /// sinogram, so they must not be mixed.
    pub fn set_geometry(&mut self, ny: usize, nx: usize) {
        if (ny, nx) != (self.ny, self.nx) {
            self.ny = ny;
            self.nx = nx;
            self.frames.clear();
            self.dark = None;
            self.flat = None;
        }
    }

    /// Set the rolling dark reference (ignored if its shape does not match the
    /// current geometry).
    pub fn set_dark(&mut self, dark: Array2<f32>) {
        if dark.dim() == (self.ny, self.nx) {
            self.dark = Some(dark);
        }
    }

    /// Set the rolling flat reference (ignored if its shape does not match).
    pub fn set_flat(&mut self, flat: Array2<f32>) {
        if flat.dim() == (self.ny, self.nx) {
            self.flat = Some(flat);
        }
    }

    pub fn has_darkflat(&self) -> bool {
        self.dark.is_some() && self.flat.is_some()
    }

    /// Append a frame, evicting the oldest when at capacity. Frames whose shape
    /// does not match the current geometry (or when no geometry is set) are
    /// dropped rather than silently corrupting the sinogram.
    pub fn push(&mut self, theta: f64, data: Array2<f32>) {
        if self.ny == 0 || data.dim() != (self.ny, self.nx) {
            return;
        }
        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(RingFrame { theta, data });
    }

    /// The buffered rotation angles, oldest first, as `f32` for [`tomoxide::Angles`].
    pub fn thetas(&self) -> Vec<f32> {
        self.frames.iter().map(|f| f.theta as f32).collect()
    }

    /// Assemble the sinogram for detector row `z` as `(1, nproj, nx)`,
    /// flat/dark-corrected and minus-logged. `None` when the buffer is empty or
    /// `z` is out of range.
    ///
    /// Correction is `−ln((raw − dark) / (flat − dark))` per pixel when a
    /// rolling dark *and* flat are both present; otherwise `−ln(raw)` (the
    /// frames are treated as already-normalized transmission). This mirrors
    /// [`tomoxide::prep::normalize_dataset`], which no-ops the dark/flat step
    /// when either reference is absent and always applies minus-log. A
    /// near-zero `flat − dark` denominator collapses to transmission `1`, and
    /// transmission is floored at `1e-6` before the log so a zero pixel does not
    /// produce a non-finite value.
    pub fn sinogram(&self, z: usize) -> Option<Array3<f32>> {
        if self.frames.is_empty() || z >= self.ny {
            return None;
        }
        let nproj = self.frames.len();
        let mut sino = Array3::<f32>::zeros((1, nproj, self.nx));
        let dark_row = self.dark.as_ref().map(|d| d.row(z));
        let flat_row = self.flat.as_ref().map(|f| f.row(z));
        for (p, frame) in self.frames.iter().enumerate() {
            let src = frame.data.row(z);
            for x in 0..self.nx {
                let raw = src[x];
                let transmission = match (dark_row.as_ref(), flat_row.as_ref()) {
                    (Some(d), Some(f)) => {
                        let denom = f[x] - d[x];
                        if denom.abs() < 1e-6 {
                            1.0
                        } else {
                            (raw - d[x]) / denom
                        }
                    }
                    _ => raw,
                };
                sino[[0, p, x]] = -(transmission.max(1e-6)).ln();
            }
        }
        Some(sino)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_oldest_at_capacity() {
        let mut ring = ProjRing::new(3);
        ring.set_geometry(1, 2);
        for i in 0..5 {
            ring.push(i as f64, Array2::from_elem((1, 2), i as f32));
        }
        assert_eq!(ring.len(), 3);
        // Oldest two evicted; angles 2,3,4 remain.
        assert_eq!(ring.thetas(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn geometry_change_clears_buffer_and_refs() {
        let mut ring = ProjRing::new(4);
        ring.set_geometry(2, 2);
        ring.set_dark(Array2::zeros((2, 2)));
        ring.set_flat(Array2::ones((2, 2)));
        ring.push(0.0, Array2::ones((2, 2)));
        assert_eq!(ring.len(), 1);
        assert!(ring.has_darkflat());
        ring.set_geometry(3, 3); // different grid
        assert_eq!(ring.len(), 0);
        assert!(!ring.has_darkflat());
    }

    #[test]
    fn mismatched_frame_is_dropped() {
        let mut ring = ProjRing::new(4);
        ring.set_geometry(2, 2);
        ring.push(0.0, Array2::ones((2, 3))); // wrong width
        assert_eq!(ring.len(), 0);
    }

    #[test]
    fn sinogram_applies_darkflat_and_log() {
        let mut ring = ProjRing::new(4);
        ring.set_geometry(1, 1);
        ring.set_dark(Array2::from_elem((1, 1), 1.0));
        ring.set_flat(Array2::from_elem((1, 1), 3.0));
        // raw = 2 → transmission = (2-1)/(3-1) = 0.5 → −ln(0.5).
        ring.push(0.0, Array2::from_elem((1, 1), 2.0));
        let sino = ring.sinogram(0).unwrap();
        assert_eq!(sino.dim(), (1, 1, 1));
        assert!((sino[[0, 0, 0]] - (-0.5f32.ln())).abs() < 1e-6);
    }

    #[test]
    fn sinogram_without_darkflat_is_raw_minus_log() {
        let mut ring = ProjRing::new(4);
        ring.set_geometry(1, 1);
        ring.push(0.0, Array2::from_elem((1, 1), 0.25));
        let sino = ring.sinogram(0).unwrap();
        assert!((sino[[0, 0, 0]] - (-0.25f32.ln())).abs() < 1e-6);
    }

    #[test]
    fn sinogram_out_of_range_is_none() {
        let mut ring = ProjRing::new(4);
        ring.set_geometry(2, 2);
        ring.push(0.0, Array2::ones((2, 2)));
        assert!(ring.sinogram(5).is_none());
        let empty = ProjRing::new(2);
        assert!(empty.sinogram(0).is_none());
    }
}
