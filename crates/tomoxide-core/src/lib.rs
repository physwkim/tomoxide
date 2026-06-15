//! # tomoxide-core
//!
//! Foundational types shared by every tomoxide crate: the data model
//! ([`data`]), acquisition [`geometry`], the tri-backend abstraction
//! ([`backend`]), scalar [`dtype`]s, algorithm [`params`], and [`error`]s.
//!
//! It depends on no backend crate — algorithms receive a backend through the
//! [`backend::Backend`] trait object. See `docs/ARCHITECTURE.md`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod backend;
pub mod data;
pub mod dtype;
pub mod error;
pub mod geometry;
pub mod params;

pub use backend::{
    Backend, DeviceBuffer, DeviceKind, Elementwise, FbpFilter, Fft, FilteredBackproject,
    ForwardProject, RankFilter,
};
pub use data::{Dataset, Frames, Layout, Slice2D, Tomo, Volume};
pub use dtype::{Complex32, Dtype, Element};
pub use error::{Error, Result};
pub use geometry::{Angles, Beam, Center, Detector, Geometry};
pub use params::{Algorithm, BackendKind, FilterName, PhaseMethod, ReconParams, StripeMethod};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angles_uniform_spans_half_turn() {
        let a = Angles::uniform(4, 0.0, std::f32::consts::PI);
        assert_eq!(a.len(), 4);
        assert!((a.0[0] - 0.0).abs() < 1e-6);
        assert!((a.0[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn tomo_layout_roundtrip_swaps_axes() {
        let arr = ndarray::Array3::<f32>::zeros((3, 5, 7)); // [angle,row,col]
        let t = Tomo::new(arr, Layout::Projection);
        assert_eq!(t.n_angles(), 3);
        assert_eq!(t.n_rows(), 5);
        assert_eq!(t.n_cols(), 7);
        let s = t.to_layout(Layout::Sinogram);
        assert_eq!(s.layout, Layout::Sinogram);
        assert_eq!(s.array.shape(), &[5, 3, 7]); // [row,angle,col]
        assert_eq!(s.n_angles(), 3);
        assert_eq!(s.n_rows(), 5);
    }

    #[test]
    fn algorithm_parses_and_classifies() {
        assert_eq!("fbp".parse::<Algorithm>().unwrap(), Algorithm::Fbp);
        assert!(Algorithm::Fbp.is_analytic());
        assert!(!Algorithm::Sirt.is_analytic());
        assert!("nope".parse::<Algorithm>().is_err());
    }
}
