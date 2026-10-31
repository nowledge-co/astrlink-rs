use std::path::PathBuf;

/// A failure with no successful scan result.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A local model file could not be read.
    #[error("cannot read {path}: {source}; check the model directory and permissions")]
    Io {
        /// File that could not be read.
        path: PathBuf,
        /// Original I/O cause.
        source: std::io::Error,
    },
    /// Model bytes differ from the supported artifact.
    #[error("model asset {file} failed integrity verification; download the pinned asset again")]
    Integrity {
        /// Asset name.
        file: &'static str,
    },
    /// The model structure is not supported.
    #[error("unsupported model: {0}; use the pinned AstrLink assets")]
    Model(String),
    /// The tokenizer could not load or encode input.
    #[error("tokenizer operation failed; verify the pinned tokenizer asset")]
    Tokenizer,
    /// A configuration value is outside the supported bounds.
    #[error("invalid configuration: {0}")]
    Configuration(&'static str),
    /// The caller requested cancellation. No complete scan is returned.
    #[error("scan cancelled; retry with a new cancellation token")]
    Cancelled,
    /// The shared engine has reached its request limit.
    #[error("engine is busy; retry after an active scan completes")]
    Busy,
    /// Input exceeds the configured byte or token limit.
    #[error(
        "input exceeds the configured {unit} limit of {limit}; split the input or raise the limit"
    )]
    InputLimit {
        /// Unit of the limit.
        unit: &'static str,
        /// Configured upper bound.
        limit: usize,
    },
    /// Compute threads could not be started.
    #[error("cannot create the compute pool: {0}")]
    ComputePool(String),
}
