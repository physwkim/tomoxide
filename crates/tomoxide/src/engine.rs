//! Backend selection. [`Engine`] owns a boxed [`Backend`] and is the single
//! place that knows all three backend crates exist (see ARCHITECTURE §2.4).

use tomoxide_core::backend::Backend;
use tomoxide_core::error::Result;
use tomoxide_core::params::BackendKind;

use tomoxide_cpu::CpuBackend;
use tomoxide_cuda::CudaBackend;
use tomoxide_wgpu::WgpuBackend;

/// A reconstruction engine bound to one backend.
pub struct Engine {
    backend: Box<dyn Backend>,
}

impl Engine {
    /// Build an engine for the requested backend.
    ///
    /// [`BackendKind::Auto`] probes CUDA → wgpu → CPU and uses the first one
    /// that initialises (on a machine without a GPU toolkit that is CPU).
    pub fn new(kind: BackendKind) -> Result<Self> {
        let backend: Box<dyn Backend> = match kind {
            BackendKind::Cpu => Box::new(CpuBackend::new()),
            BackendKind::Cuda => Box::new(CudaBackend::new()?),
            BackendKind::Wgpu => Box::new(WgpuBackend::new()?),
            BackendKind::Auto => Self::auto_backend(),
        };
        log::info!("tomoxide engine using backend: {}", backend.name());
        Ok(Engine { backend })
    }

    fn auto_backend() -> Box<dyn Backend> {
        if let Ok(b) = CudaBackend::new() {
            return Box::new(b);
        }
        if let Ok(b) = WgpuBackend::new() {
            return Box::new(b);
        }
        Box::new(CpuBackend::new())
    }

    /// The selected backend.
    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// The selected backend's name (`"cpu"`/`"cuda"`/`"wgpu"`).
    pub fn name(&self) -> &'static str {
        self.backend.name()
    }
}
