# Development

Run commands from the repository root. Cargo owns dependency versions and `Cargo.lock`; Bazel reads both files.

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --release
```

Model tests require local pinned assets. Run them explicitly; an ignored test does not qualify inference.

```sh
python3 -I tools/download.py models/astrlink
ASTRLINK_MODEL_DIR="$PWD/models/astrlink" cargo test --locked --release -- --include-ignored --nocapture
```

`tools/oracle.py` generates synthetic fixtures with Python ONNX Runtime and Hugging Face tokenizers. These tools are development dependencies only. The fixture records the oracle version and asset hashes. The numerical test compares token IDs, original-text offsets and logits against that independent fixture. Do not regenerate expected values from the Rust engine.

```sh
uv run --no-project --with onnxruntime==1.23.2 --with tokenizers==0.23.3 python -I tools/oracle.py models/astrlink testdata/oracle.json
```

Use the native Bazel targets for pre-merge compilation, lint and tests. See `BUILD.bazel` for target names. Pass an absolute `ASTRLINK_MODEL_DIR` and `--test_arg=--include-ignored` to qualify model tests. The test carrier compiles once for both ordinary and model tests.

Keep performance evidence on the issue or PR. Record input size in model tokens, worker count, cache state, profile, hardware, latency and memory. A CPU result does not qualify CoreML or another platform.

Contributions must retain upstream notices, use typed errors and document public contracts. Test observable behavior with synthetic data. Keep personal data, model weights and credentials out of Git. Use an issue and a reviewed PR; repository changes merge through the organization's bot.
