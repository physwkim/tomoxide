//! Algorithm selectors and parameter structs (the union of tomopy + tomocupy).

use crate::error::{Error, Result};

/// Which backend to run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BackendKind {
    /// Probe CUDA → wgpu → CPU and pick the first available.
    #[default]
    Auto,
    /// Force the CPU backend.
    Cpu,
    /// Force the CUDA backend (requires the `cuda` feature + an NVIDIA device).
    Cuda,
    /// Force the portable wgpu backend (requires the `gpu-wgpu` feature).
    Wgpu,
}

/// A reconstruction algorithm. Analytic methods are one-pass; the rest are
/// iterative. See `docs/PORTING.md` for the upstream of each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    // --- analytic / direct ---
    /// Filtered back-projection (tomopy `fbp.c`).
    Fbp,
    /// Fourier-grid reconstruction (tomopy `gridrec.c`).
    Gridrec,
    /// USFFT Fourier-based (tomocupy `fourierrec`).
    Fourierrec,
    /// Log-polar (tomocupy `lprec`).
    Lprec,
    /// Direct line integration (tomocupy `linerec`).
    Linerec,
    // --- iterative ---
    /// Algebraic reconstruction technique.
    Art,
    /// Block ART.
    Bart,
    /// Simultaneous iterative reconstruction technique.
    Sirt,
    /// Maximum-likelihood expectation-maximization.
    Mlem,
    /// Ordered-subset EM.
    Osem,
    /// Ordered-subset penalized ML, hybrid prior.
    OspmlHybrid,
    /// Ordered-subset penalized ML, quadratic prior.
    OspmlQuad,
    /// Penalized ML, hybrid prior.
    PmlHybrid,
    /// Penalized ML, quadratic prior.
    PmlQuad,
    /// Total-variation regularized.
    Tv,
    /// Gradient-descent regularized.
    Grad,
    /// Tikhonov regularized.
    Tikh,
    /// Vector (multi-axis) reconstruction.
    Vector,
}

impl Algorithm {
    /// `true` for the one-pass analytic methods (filter + backproject).
    pub fn is_analytic(self) -> bool {
        matches!(
            self,
            Algorithm::Fbp
                | Algorithm::Gridrec
                | Algorithm::Fourierrec
                | Algorithm::Lprec
                | Algorithm::Linerec
        )
    }
}

impl std::str::FromStr for Algorithm {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "fbp" => Algorithm::Fbp,
            "gridrec" => Algorithm::Gridrec,
            "fourierrec" => Algorithm::Fourierrec,
            "lprec" => Algorithm::Lprec,
            "linerec" => Algorithm::Linerec,
            "art" => Algorithm::Art,
            "bart" => Algorithm::Bart,
            "sirt" => Algorithm::Sirt,
            "mlem" => Algorithm::Mlem,
            "osem" => Algorithm::Osem,
            "ospml_hybrid" => Algorithm::OspmlHybrid,
            "ospml_quad" => Algorithm::OspmlQuad,
            "pml_hybrid" => Algorithm::PmlHybrid,
            "pml_quad" => Algorithm::PmlQuad,
            "tv" => Algorithm::Tv,
            "grad" => Algorithm::Grad,
            "tikh" => Algorithm::Tikh,
            "vector" => Algorithm::Vector,
            other => return Err(Error::InvalidParam(format!("unknown algorithm '{other}'"))),
        })
    }
}

/// FBP/gridrec apodization filter. Same named set in tomopy and tomocupy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FilterName {
    /// No filtering.
    None,
    /// Pure ramp (Ram-Lak).
    #[default]
    Ramp,
    /// Shepp-Logan.
    Shepp,
    /// Cosine.
    Cosine,
    /// Cosine squared.
    Cosine2,
    /// Hamming.
    Hamming,
    /// Hann.
    Hann,
    /// Parzen.
    Parzen,
}

impl std::str::FromStr for FilterName {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "none" => FilterName::None,
            "ramp" => FilterName::Ramp,
            "shepp" => FilterName::Shepp,
            "cosine" => FilterName::Cosine,
            "cosine2" => FilterName::Cosine2,
            "hamming" => FilterName::Hamming,
            "hann" => FilterName::Hann,
            "parzen" => FilterName::Parzen,
            other => return Err(Error::InvalidParam(format!("unknown filter '{other}'"))),
        })
    }
}

/// Parameters for one `recon` call. Unused fields stay at their defaults; each
/// [`Algorithm`] reads only the ones it needs (mirrors tomopy's per-algorithm
/// `allowed_recon_kwargs`).
#[derive(Clone, Debug)]
pub struct ReconParams {
    /// Reconstruction grid width (`ngridx`); defaults to the detector width.
    pub num_gridx: Option<usize>,
    /// Reconstruction grid height (`ngridy`); defaults to the detector width.
    pub num_gridy: Option<usize>,
    /// Number of iterations (iterative methods).
    pub num_iter: usize,
    /// Apodization filter (analytic methods).
    pub filter_name: FilterName,
    /// Optional raw filter parameters (`filter_par`).
    pub filter_par: Vec<f32>,
    /// Regularization parameters (`reg_par`).
    pub reg_par: Vec<f32>,
    /// Data-fidelity regularization (`reg_data`, Tikhonov).
    pub reg_data: Vec<f32>,
    /// Number of ordered-subset blocks (`num_block`).
    pub num_block: usize,
    /// Per-block angle indices (`ind_block`).
    pub ind_block: Vec<i32>,
}

impl Default for ReconParams {
    fn default() -> Self {
        ReconParams {
            num_gridx: None,
            num_gridy: None,
            num_iter: 1,
            filter_name: FilterName::Ramp,
            filter_par: Vec::new(),
            reg_par: Vec::new(),
            reg_data: Vec::new(),
            num_block: 0,
            ind_block: Vec::new(),
        }
    }
}

/// Stripe-removal method (combines tomopy and tomocupy options).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum StripeMethod {
    /// No stripe removal.
    #[default]
    None,
    /// Fourier-Wavelet (tomopy `remove_stripe_fw`; tomocupy `fw`).
    Fw {
        /// Damping factor.
        sigma: f32,
        /// Decomposition level (`None` = auto).
        level: Option<usize>,
    },
    /// Titarenko (tomopy `remove_stripe_ti`; tomocupy `ti`).
    Ti {
        /// Number of blocks (`nblock`); `0` corrects the whole sinogram at once
        /// (tomopy default).
        nblock: usize,
        /// Damping factor `beta` (tomopy's `alpha`, default `1.5`).
        beta: f32,
    },
    /// Smoothing filter (tomopy `remove_stripe_sf`).
    Sf {
        /// Median window size.
        size: usize,
    },
    /// Vo all-stripe (tomocupy `vo-all`).
    VoAll {
        /// Signal-to-noise ratio.
        snr: f32,
        /// Large-stripe window size.
        la_size: usize,
        /// Small-stripe window size.
        sm_size: usize,
    },
}

/// Phase-retrieval method.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum PhaseMethod {
    /// No phase retrieval.
    #[default]
    None,
    /// Paganin single-material retrieval (tomopy + tomocupy).
    Paganin {
        /// Detector pixel size (cm).
        pixel_size: f32,
        /// Sample-to-detector propagation distance (cm).
        dist: f32,
        /// X-ray energy (keV).
        energy: f32,
        /// Regularization parameter.
        alpha: f32,
    },
    /// Generalized Paganin (tomocupy `Gpaganin`, Paganin et al. 2020). Uses a
    /// `cos`-based reciprocal grid and a `delta/beta` (`db`) + characteristic
    /// length (`w`) filter instead of Paganin's `alpha` regularization.
    GPaganin {
        /// Detector pixel size (cm).
        pixel_size: f32,
        /// Sample-to-detector propagation distance (cm).
        dist: f32,
        /// X-ray energy (keV).
        energy: f32,
        /// Material `delta/beta` ratio.
        db: f32,
        /// Characteristic transverse length scale `W` (cm).
        w: f32,
    },
    /// Farago single-step retrieval (tomocupy `farago`, Farago 2024). Same
    /// padded Fourier machinery as Paganin but with the filter
    /// `1/(cos θ + db·sin θ)`, `θ = π·λ·dist·(ix² + iy²)` over the squared
    /// reciprocal grid.
    Farago {
        /// Detector pixel size (cm).
        pixel_size: f32,
        /// Sample-to-detector propagation distance (cm).
        dist: f32,
        /// X-ray energy (keV).
        energy: f32,
        /// Material `delta/beta` ratio.
        db: f32,
    },
}
