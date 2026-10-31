//! Read text from stdin and print detection metadata as TSV.
use std::io::{self, Read};
use std::path::PathBuf;

use astrlink::{Cancellation, Model, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: scan MODEL_DIR < input.txt")?;
    let options = Options::default();
    let max_bytes = options.max_bytes;
    let model = Model::load(&directory, options)?;
    let mut text = String::new();
    io::stdin()
        .take(max_bytes as u64 + 1)
        .read_to_string(&mut text)?;
    let mut session = model.session(64)?;
    let result = session.scan(&text, 0.8, &Cancellation::default())?;
    for span in result.spans {
        // Print offsets only. The input can contain personal data or secrets.
        println!(
            "{}\t{}\t{:?}\t{:.6}",
            span.bytes.start, span.bytes.end, span.category, span.confidence
        );
    }
    Ok(())
}
