//! Error and result types shared across the workspace.

use thiserror::Error;

/// The crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// All errors surfaced by tomoxide.
#[derive(Debug, Error)]
pub enum Error {
    /// A stubbed entry point that has not been ported yet. `upstream` points at
    /// the reference `file:line` to port from (see `docs/PORTING.md`).
    #[error("not implemented: {what} (port from {upstream})")]
    NotImplemented {
        /// Short name of the operation.
        what: &'static str,
        /// Upstream reference location to port from.
        upstream: &'static str,
    },

    /// The requested backend was not compiled in or no device is present.
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    /// The selected backend does not provide the capability an algorithm needs.
    #[error("backend '{backend}' lacks capability '{capability}'")]
    MissingCapability {
        /// The backend that was asked.
        backend: &'static str,
        /// The capability trait that was missing.
        capability: &'static str,
    },

    /// Array shapes did not line up.
    #[error("shape mismatch: expected {expected}, got {found}")]
    ShapeMismatch {
        /// Human-readable expected shape.
        expected: String,
        /// Human-readable observed shape.
        found: String,
    },

    /// A parameter was outside its valid range or otherwise invalid.
    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    /// I/O failure (file read/write, HDF5, etc.).
    #[error("io error: {0}")]
    Io(String),

    /// A device/driver-level failure inside a backend.
    #[error("backend error: {0}")]
    Backend(String),
}

impl Error {
    /// Convenience constructor for [`Error::NotImplemented`].
    pub const fn todo(what: &'static str, upstream: &'static str) -> Self {
        Error::NotImplemented { what, upstream }
    }
}
