# Model assets

[`model.json`](../model.json) owns the supported model repository, revision, asset directory, filenames, sizes and SHA-256 hashes. The Rust loader embeds its pins and verifies the exact buffers before parsing them. Both development tools read the same manifest; the oracle generator verifies all assets before loading the model and records the ONNX Runtime and tokenizer versions.

Run the explicit download helper from the repository root:

```sh
python3 -I tools/download.py models/astrlink
```

The helper downloads public assets from Hugging Face. It verifies each file before installation. The Rust library itself has no network client.

The supported model is [QuantumNous/astrlink-guard](https://huggingface.co/QuantumNous/astrlink-guard). Its model card describes the labels and training sources. The weights use Apache-2.0; the base model uses MIT. Retain the publisher's license files and notices when redistributing model assets. Code licensing does not replace model licensing.

Use a release-specific directory. Do not replace files while another process writes them. A bad hash or missing asset causes a typed load failure.
