//! CPU inference for the pinned AstrLink Guard model.
//!
//! Load assets once with [`Model::load`], then create a [`Session`] per data owner.
//! Sessions share a bounded compute pool. Each session has a private, in-memory
//! window cache. Scans return UTF-8 byte ranges in the original input.
//! No operation downloads assets or sends input over the network.

mod assets;
mod error;
mod net;
mod onnx;
mod scan;

pub use error::Error;
pub use scan::{Category, Scan, Session, Span};

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Cooperative cancellation, checked between model layers and windows.
///
/// Observed cancellation returns an error after workers settle. Tokenization
/// and a running layer finish first. Cancellation racing with completion may
/// return success. Cancellation is permanent for this token.
#[derive(Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    /// Request cancellation of scans that use this token.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        if self.0.load(Ordering::Relaxed) {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Resource limits shared by all sessions of a model.
#[derive(Clone, Debug)]
pub struct Options {
    /// Number of compute threads, from 1 through 64 inclusive.
    /// [`Model::load`] returns [`Error::Configuration`] outside this range.
    pub workers: usize,
    /// Maximum simultaneous scans, from 1 through 64 inclusive.
    /// [`Model::load`] returns [`Error::Configuration`] outside this range.
    /// Once this many scans are active, additional scans return [`Error::Busy`].
    pub max_in_flight: usize,
    /// Maximum UTF-8 input bytes per scan, checked before tokenization.
    pub max_bytes: usize,
    /// Maximum content tokens per scan, checked after tokenization.
    pub max_tokens: usize,
}

impl Default for Options {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        Self {
            workers: (cores / 2).clamp(1, 4),
            max_in_flight: 4,
            max_bytes: 1_048_576,
            max_tokens: 262_144,
        }
    }
}

struct Inner {
    net: net::Net,
    tokenizer: tokenizers::Tokenizer,
    pool: rayon::ThreadPool,
    options: Options,
    active: AtomicUsize,
}

/// Immutable model weights and a shared compute budget.
///
/// Clones share weights, threads and admission limits. Separately loaded models
/// have separate budgets. Reuse a model across callers to share the budget.
#[derive(Clone)]
pub struct Model(Arc<Inner>);

impl Model {
    /// Load the supported assets from a local directory and verify their hashes.
    ///
    /// The directory must contain `model_int8.onnx`, `model_int8.onnx_data`,
    /// and `tokenizer.json`. See the repository model download guide.
    pub fn load(directory: &Path, options: Options) -> Result<Self, Error> {
        if !(1..=64).contains(&options.workers) {
            return Err(Error::Configuration("workers must be between 1 and 64"));
        }
        if !(1..=64).contains(&options.max_in_flight) {
            return Err(Error::Configuration(
                "max_in_flight must be between 1 and 64",
            ));
        }
        if options.max_bytes == 0 || options.max_tokens == 0 {
            return Err(Error::Configuration("input limits must be positive"));
        }
        let assets = assets::load(directory)?;
        let net = net::Net::load(assets.tensors)?;
        let mut tokenizer =
            tokenizers::Tokenizer::from_bytes(assets.tokenizer).map_err(|source| {
                Error::Tokenizer {
                    stage: "load",
                    source: Some(source),
                }
            })?;
        tokenizer
            .with_truncation(None)
            .map_err(|source| Error::Tokenizer {
                stage: "configure truncation",
                source: Some(source),
            })?;
        tokenizer.with_padding(None);
        // The shared tokenizer must not retain text across session owners.
        let mut tokenizer_model = tokenizer.get_model().clone();
        tokenizer_model.resize_cache(0);
        tokenizer.with_model(tokenizer_model);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(options.workers)
            .thread_name(|i| format!("astrlink-{i}"))
            .build()
            .map_err(|e| Error::ComputePool(e.to_string()))?;
        Ok(Self(Arc::new(Inner {
            net,
            tokenizer,
            pool,
            options,
            active: AtomicUsize::new(0),
        })))
    }

    /// Create a private window cache. Zero capacity disables caching.
    ///
    /// Capacity is at most 4096 windows. Each entry stores at most 512 token IDs
    /// and 510 predictions. Token IDs can reveal input; discard the session when
    /// its data owner changes. The cache has no disk representation.
    pub fn session(&self, cache_windows: usize) -> Result<Session, Error> {
        if cache_windows > 4096 {
            return Err(Error::Configuration("cache_windows must be at most 4096"));
        }
        Ok(Session::new(self.clone(), cache_windows))
    }
}

struct Permit<'a>(&'a AtomicUsize);

impl Inner {
    fn admit(&self) -> Result<Permit<'_>, Error> {
        let mut active = self.active.load(Ordering::Relaxed);
        loop {
            if active >= self.options.max_in_flight {
                return Err(Error::Busy);
            }
            match self.active.compare_exchange_weak(
                active,
                active + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(Permit(&self.active)),
                Err(current) => active = current,
            }
        }
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests;
