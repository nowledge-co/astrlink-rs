# AstrLink Rust

Run the AstrLink Guard personal-data and secret detector locally from Rust. The library loads pinned ONNX weights and executes the INT8 network without ONNX Runtime.

The model can miss personal data and can flag ordinary text. Evaluate its accuracy for each language and workload before use.

## Use

[Download the model assets](docs/model.md), then add the library from Git:

```toml
[dependencies]
astrlink = { git = "https://github.com/nowledge-co/astrlink-rs" }
```

```rust,no_run
use std::path::Path;
use astrlink::{Cancellation, Model, Options};

let model = Model::load(Path::new("models/astrlink"), Options::default())?;
let mut session = model.session(64)?;
let text = "Contact alice@example.com.";
let scan = session.scan(text, 0.8, &Cancellation::default())?;
for span in scan.spans {
    println!("{:?}: {:?}", span.category, span.bytes);
}
# Ok::<(), astrlink::Error>(())
```

Each span refers to a UTF-8 byte range in the original input. Its confidence is the lowest selected token probability in that span. The caller chooses the threshold.

Keep one session per data owner. Sessions share model weights and compute threads; each session owns its window cache. Drop or clear a session when its data owner changes. Token IDs in a cache can reveal input.

The library performs synchronous CPU work. Use a bounded blocking executor from an async application. Excess requests return `Error::Busy`. Observed cancellation returns `Error::Cancelled` after active workers settle; cancellation racing with completion may return success. See [`Cancellation`](src/lib.rs) for the contract.

## Example

```sh
printf 'Contact alice@example.com.\n' | cargo run --release --example scan -- models/astrlink
```

The example prints byte offsets, category and confidence as TSV. It does not print the detected text.

## Reference

- [Model assets and licenses](docs/model.md)
- [Architecture and concurrency contract](docs/architecture.md)
- [Development and validation](docs/development.md)

## License and attribution

Code: [MIT](LICENSE). The network and window algorithms derive from [Magpie](https://github.com/yetone/magpie). See [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) for its license.

Model weights have a separate Apache-2.0 license. This repository does not contain model weights or training data.
