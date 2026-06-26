//! The 3-D array data model.
//!
//! Bulk arrays carry an explicit [`Layout`] so we never silently transpose,
//! matching tomopy's `sinogram_order` flag and tomocupy's sinogram chunking.
//! See `docs/ARCHITECTURE.md` §1.

use ndarray::{Array2, Array3};

/// Axis order of a projection/sinogram stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// `[angle, row, col]` — tomopy "projection order" (the acquisition order).
    Projection,
    /// `[row, angle, col]` — tomopy `sinogram_order=True`; reconstruction order.
    Sinogram,
}

/// A projection or sinogram stack with a tracked [`Layout`].
///
/// Axes are `(a0, a1, col)` where the meaning of `a0`/`a1` depends on `layout`:
/// `Projection` → `(angle, row, col)`, `Sinogram` → `(row, angle, col)`.
#[derive(Clone, Debug)]
pub struct Tomo<T = f32> {
    /// The backing array.
    pub array: Array3<T>,
    /// Which axis order `array` is in.
    pub layout: Layout,
}

impl<T: Clone> Tomo<T> {
    /// Wrap an array, asserting its layout.
    pub fn new(array: Array3<T>, layout: Layout) -> Self {
        Self { array, layout }
    }

    /// Number of projection angles.
    pub fn n_angles(&self) -> usize {
        match self.layout {
            Layout::Projection => self.array.shape()[0],
            Layout::Sinogram => self.array.shape()[1],
        }
    }

    /// Number of detector rows (independent reconstruction slices).
    pub fn n_rows(&self) -> usize {
        match self.layout {
            Layout::Projection => self.array.shape()[1],
            Layout::Sinogram => self.array.shape()[0],
        }
    }

    /// Detector width (number of columns).
    pub fn n_cols(&self) -> usize {
        self.array.shape()[2]
    }

    /// Return a copy in the requested layout (swapping axes 0/1 if needed).
    pub fn to_layout(&self, target: Layout) -> Tomo<T> {
        if self.layout == target {
            return self.clone();
        }
        // Swap the angle/row axes; `to_owned` yields a contiguous C-layout copy.
        let array = self.array.view().permuted_axes([1, 0, 2]).to_owned();
        Tomo {
            array,
            layout: target,
        }
    }

    /// Borrow in the requested layout, allocating only when a transpose is
    /// actually needed. When `self` is already in `target` this returns a
    /// borrow (no copy); otherwise it transposes into an owned copy.
    ///
    /// Prefer this over [`to_layout`] on read-only paths: `to_layout` always
    /// allocates — it `clone()`s the full array even when the layout already
    /// matches — which on a reconstruction-sized sinogram is hundreds of MB
    /// copied per call for nothing.
    pub fn as_layout(&self, target: Layout) -> std::borrow::Cow<'_, Tomo<T>> {
        if self.layout == target {
            std::borrow::Cow::Borrowed(self)
        } else {
            std::borrow::Cow::Owned(self.to_layout(target))
        }
    }
}

/// A reconstructed 3-D volume, axes `[z(row), y, x]`.
#[derive(Clone, Debug)]
pub struct Volume<T = f32> {
    /// The backing array `[z, y, x]`.
    pub array: Array3<T>,
}

impl<T> Volume<T> {
    /// Wrap a `[z, y, x]` array.
    pub fn new(array: Array3<T>) -> Self {
        Self { array }
    }

    /// `(z, y, x)` extents.
    pub fn dims(&self) -> (usize, usize, usize) {
        let s = self.array.shape();
        (s[0], s[1], s[2])
    }
}

/// A single reconstructed slice, axes `[y, x]`.
pub type Slice2D<T = f32> = Array2<T>;

/// Flat-field or dark-field frames, axes `[frame, row, col]`.
#[derive(Clone, Debug)]
pub struct Frames<T = f32> {
    /// The backing array `[frame, row, col]`.
    pub array: Array3<T>,
}

impl<T> Frames<T> {
    /// Wrap a `[frame, row, col]` array.
    pub fn new(array: Array3<T>) -> Self {
        Self { array }
    }

    /// Number of frames.
    pub fn count(&self) -> usize {
        self.array.shape()[0]
    }
}

/// A complete DXchange-style acquisition: projections plus flat/dark/theta.
#[derive(Clone, Debug)]
pub struct Dataset<T = f32> {
    /// Raw projections.
    pub data: Tomo<T>,
    /// Flat (white/open-beam) frames, if present.
    pub flat: Option<Frames<T>>,
    /// Dark frames, if present.
    pub dark: Option<Frames<T>>,
    /// Projection angles in radians, length = number of angles.
    pub theta: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn as_layout_borrows_when_matching_and_transposes_otherwise() {
        let a = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (i * 100 + j * 10 + k) as f32);
        let t = Tomo::new(a.clone(), Layout::Projection);

        // Already in the target layout → borrow, no allocation.
        let same = t.as_layout(Layout::Projection);
        assert!(matches!(same, Cow::Borrowed(_)));
        assert_eq!(same.array, a);

        // Different layout → owned transpose [row, angle, col], same as to_layout.
        let swapped = t.as_layout(Layout::Sinogram);
        assert!(matches!(swapped, Cow::Owned(_)));
        assert_eq!(swapped.layout, Layout::Sinogram);
        assert_eq!(swapped.array.dim(), (3, 2, 4));
        assert_eq!(swapped.array[[1, 0, 2]], a[[0, 1, 2]]);
        assert_eq!(swapped.array, t.to_layout(Layout::Sinogram).array);
    }
}
