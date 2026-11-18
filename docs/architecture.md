# Architecture contract

The crate executes one hash-pinned INT8 BERT graph. It is not a general ONNX executor. The loader verifies buffers before protobuf decoding. The tensor loader checks names, dimensions and quantization metadata before constructing the network.

## Numerical execution

The tokenizer loads the publisher's tokenizer JSON without truncation or padding. The scanner adds start and end tokens to each window. `scan.rs` owns the window length and overlap. Each window starts its position IDs at zero and has its own dynamic activation quantization range.

The integer dot products use widened accumulation. GELU uses the erf definition. Weight packing changes memory layout without changing quantized values. The shared pool runs independent windows in parallel. Each uncached window first takes one of the model's worker turns, and a model has one turn per worker. Of the admitted scans waiting, the scan with the fewest windows left takes the next turn, and equal scans go in arrival order. A short scan therefore waits for one running window to end, not for the rest of a long scan. A window holds one turn and runs on one worker, with one exception: a scan's last window takes every turn if all are free and no other scan waits. That window splits its matrix rows, attention heads and GELU rows over the pool, so a lone short scan uses every worker. A split window must hold every turn. A rayon worker waiting for split work runs other queued jobs meanwhile, so a split window beside other windows could stall until a whole other window ends. Each output value keeps one serial operation order; the split and the worker count do not change results.

For each overlap token, the scanner chooses the prediction farthest from a window edge. Ties use the earlier window. BIO decoding runs over the complete selected token sequence. An unmatched I tag starts a span. Zero-width tokens do not start or end spans. A span score is the minimum selected token probability.

## Ownership and bounds

A `Model` owns immutable weights, a tokenizer, a compute pool and an admission counter. Its clones share those resources. Each `Session` owns its cache and requires exclusive mutable access for a scan. A cache key is the complete window token sequence. The session cannot change models. Thresholds apply after cache lookup and decoding.

A scan follows these states:

| State | Next state or terminal result | Owner |
| --- | --- | --- |
| Requested | Admitted, Busy, Cancelled, invalid configuration | Calling thread |
| Admitted | Tokenized, input error, Cancelled | RAII permit |
| Tokenized | Computing, input error, Cancelled | Calling thread |
| Computing | Selected, inference error, Cancelled | Shared pool; caller hands out windows by turn and waits |
| Selected | Complete | Calling thread |

The permit remains held until workers settle and the call returns. No task detaches. Admission has no waiter queue. Only admitted scans wait for worker turns, so at most `max_in_flight` scans wait. A waiting scan observes cancellation within 20 ms. A steady stream of shorter scans can delay a longer one, as in Magpie. `Options` limits input bytes before tokenization and tokens after tokenization. Cache capacity limits both keys and predictions because every entry has a fixed maximum window length. FIFO eviction avoids a second recency policy.

Successful scans cover every content token. Errors contain no partial `Scan`. Cancellation is cooperative; a running tokenizer operation or model layer finishes before the next check. Cache updates occur only after complete inference and a final cancellation check. A cancellation that races with completion can arrive after that check.

The cache retains token IDs, which can reveal input. It has no disk format, global registry or cross-session lookup. Clear or drop it when the data owner changes. Allocation release does not guarantee memory zeroization.

## Port boundary

The port retains fixed model windows, quantized weights, overlap selection and shared CPU budgets from Magpie. It uses Hugging Face's Rust tokenizer and Rust resource ownership. [Upstream alignment](upstream.md) records the reviewed Magpie revision, each deliberate difference and its evidence.

Outer text chunks, cross-text batching, value propagation and masking policies are application concerns. Fixed token windows can lose cache hits after an early edit; an unchanged window always has the same result. CoreML requires a separate backend and macOS qualification. The portable CPU implementation does not qualify Apple acceleration or model accuracy for a product.
