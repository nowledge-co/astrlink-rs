//! Read and verify the fixed AstrLink model before decoding any of its bytes.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::Error;
use crate::onnx::{self, Tensor};

/// Verified model initializers and tokenizer bytes.
pub struct Assets {
    /// Initializers indexed by their ONNX names.
    pub tensors: HashMap<String, Tensor>,
    /// The verified `tokenizer.json` contents.
    pub tokenizer: Vec<u8>,
}

#[derive(serde::Deserialize)]
struct Pin {
    file: &'static str,
    size: usize,
    sha256: &'static str,
}

#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "'de: 'static"))]
struct Manifest {
    model: Pin,
    data: Pin,
    tokenizer: Pin,
}

/// Load the three pinned files directly inside `dir` without network access.
///
/// Every file's length and SHA-256 are checked on the same owned bytes that
/// are subsequently parsed. Graph references cannot open additional files.
pub fn load(dir: &Path) -> Result<Assets, Error> {
    let manifest: Manifest = serde_json::from_str(include_str!("../model.json"))
        .map_err(|e| Error::Model(format!("embedded model.json: {e}")))?;
    let model = read_verified(dir, &manifest.model)?;
    let data = read_verified(dir, &manifest.data)?;
    let tokenizer = read_verified(dir, &manifest.tokenizer)?;
    let tensors = onnx::parse(&model, &data)?;
    Ok(Assets { tensors, tokenizer })
}

fn read_verified(dir: &Path, pin: &Pin) -> Result<Vec<u8>, Error> {
    let path = dir.join(pin.file);
    let file = File::open(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    // One extra byte detects oversized files while bounding the read itself.
    let mut bytes = Vec::with_capacity(pin.size + 1);
    file.take(pin.size as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| Error::Io { path, source })?;
    if bytes.len() != pin.size || format!("{:x}", Sha256::digest(&bytes)) != pin.sha256 {
        return Err(Error::Integrity { file: pin.file });
    }
    Ok(bytes)
}
