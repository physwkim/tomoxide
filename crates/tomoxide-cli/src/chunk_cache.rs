//! Persistent cache of tuned pipeline chunk sizes (`tune_chunk` writes it,
//! `recon` reads it).
//!
//! The optimal `--chunk` (detector rows per streaming chunk) is
//! dataset/algorithm/GPU dependent: a small chunk that overlaps well at 2048²
//! starves the strided HDF5 read and the GPU FFT batch at 4096², while a large
//! chunk that wins at 4096² loses the pipeline overlap at 1024². Rather than
//! re-measure on every reconstruction (expensive) or hard-code one value
//! (wrong at some size), `tune_chunk` measures once and records the best chunk
//! here, keyed by `(file, algorithm, dtype, gpu)`; `recon` then auto-applies it.
//!
//! Stored geometry (`nx`, `nproj`, `nz`) is a validity check, not part of the
//! key: if the file at the same path is regenerated with different dimensions,
//! the dims no longer match and the entry is treated as a miss (stale), so a
//! changed dataset never silently reuses a chunk tuned for the old one.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// File written in the current working directory holding all tuned entries.
pub const CACHE_FILE: &str = ".tomoxide_chunk_cache";

/// One tuned result: the chosen chunk plus the geometry it was tuned for.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    /// Tuned detector rows per streaming chunk.
    pub chunk: usize,
    /// Detector width the tuning was measured at (validity check).
    pub nx: usize,
    /// Projection count the tuning was measured at (validity check).
    pub nproj: usize,
    /// Slice count the tuning was measured at (validity check).
    pub nz: usize,
}

/// On-disk cache: a flat map from composite key to tuned [`Entry`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChunkCache {
    #[serde(default)]
    entries: BTreeMap<String, Entry>,
}

/// Composite key for a tuning: `file|algorithm|dtype|gpu`. The file is
/// canonicalized so the same dataset reached by different relative paths shares
/// one entry; the GPU model name keeps a tuning from one card off another.
pub fn key(file: &Path, algorithm: &str, dtype: &str, gpu: &str) -> String {
    let f = std::fs::canonicalize(file)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| file.display().to_string());
    format!("{f}|{algorithm}|{dtype}|{gpu}")
}

impl ChunkCache {
    /// Load the cache from `CACHE_FILE` in the current directory, or an empty
    /// cache if it is absent or unparseable (a corrupt cache must never abort a
    /// reconstruction — it just behaves as a miss and is rewritten on next tune).
    pub fn load() -> Self {
        match std::fs::read_to_string(CACHE_FILE) {
            Ok(text) => toml::from_str(&text).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Look up a tuned chunk for this key, returning it only when the stored
    /// geometry still matches the current dataset (else the entry is stale).
    pub fn get(&self, key: &str, nx: usize, nproj: usize, nz: usize) -> Option<usize> {
        let e = self.entries.get(key)?;
        (e.nx == nx && e.nproj == nproj && e.nz == nz).then_some(e.chunk)
    }

    /// Insert/replace the tuned entry for this key.
    pub fn insert(&mut self, key: String, entry: Entry) {
        self.entries.insert(key, entry);
    }

    /// Persist the cache to `CACHE_FILE` in the current directory.
    pub fn save(&self) -> anyhow::Result<()> {
        std::fs::write(CACHE_FILE, toml::to_string_pretty(self)?)?;
        Ok(())
    }
}
