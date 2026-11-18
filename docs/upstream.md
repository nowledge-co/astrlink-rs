# Upstream alignment

The network, windows and decoding derive from Magpie's `internal/pii/guard`. This record lists the reviewed upstream revision, each deliberate difference and the evidence for it. Repeat the review when Magpie changes `internal/pii/guard` or the AstrLink parts of `internal/pii`.

Reviewed revision: [yetone/magpie@dc507aefd](https://github.com/yetone/magpie/commit/dc507aefd) (2026-11-18). The port was written against Magpie as of 2026-10-31.

## Upstream changes since the port

| Magpie commit | Change | Port |
| --- | --- | --- |
| `12483d83d` | The Go tokenizer stepped malformed UTF-8 by three bytes and panicked. | Not applicable. `scan` takes `&str`, which is valid UTF-8, and tokenizes with Hugging Face's tokenizer. |
| `7b883cc61` | A panic in a window or matmul goroutine reaches `Find`'s caller and releases its cores. | Equivalent. Rayon resumes a worker panic on the calling thread; the admission permit is released during unwinding. |
| `1c3591b5d` | AVX2 kernels for Go; Go results unchanged to the bit. | No numerical change. See [Performance](#performance). |
| `d77eb6d23` | `bert.go` and the tokenizer also serve a sentence embedder. | Refactor; AstrLink outputs unchanged. The embedder is a different model. |
| `8a64f5256`, `ac0b5c287` | Outer text pieces and request scheduling in `internal/pii`. | Application concerns. |
| Others under `internal/pii` | pf-serve, downloads, platform detection, GUI paths. | Other backends or application concerns. |

## Deliberate differences

| Area | Magpie | Port | Reason |
| --- | --- | --- | --- |
| Layer norm, attention, GELU arithmetic | Go approximations: two-pass variance, Cephes `expf`, Abramowitz-Stegun `erf`, serial softmax sum | ONNX Runtime CPU kernel order and MLAS polynomials | Reproduces the publisher's runtime. Measured below. |
| Threshold | `Find` drops spans under 0.5; masking keeps a user setting in [0.5, 1), default 0.8 | Caller passes a threshold in [0, 1] | The library leaves policy to the caller. In [0.5, 1) the port keeps the spans Magpie keeps, except a score that f32 rounding moves across the threshold. |
| Score type | `float64`, stored and compared as `float32` | `f32` | Same value that Magpie stores. |
| Span end | End of the last token | Largest token end | Same result for non-decreasing tokenizer offsets; the maximum preserves the slicing guarantee. |
| Tokenizer | Go reimplementation tested against Hugging Face | Hugging Face `tokenizers` | Uses the reference implementation directly. |
| Waiting | Requests wait for shared cores | `Error::Busy` beyond `max_in_flight` | No hidden queue; the application chooses waiting and order. |
| Window work split | Output columns over free cores | Matrix rows, heads and GELU rows as pool tasks | Same values; rayon work stealing gives a lone window every worker. |
| CoreML, outer pieces, cross-text cache, value propagation, masking | Present | Absent | Separate backend or application concerns. See [architecture](architecture.md#port-boundary). |
| Cancellation | None inside `guard` | Cooperative, between layers and windows | Extension for bounded callers. |

## Measured comparison

Measured on 2026-11-18 with 138 synthetic texts: 126 fit one window and 12 span several. The reference is ONNX Runtime 1.23.2 on the pinned model with each INT8 weight and weight zero point shifted to UINT8. The shift does not change any product. It avoids the int16 saturation of ONNX Runtime's AVX2 u8s8 kernel on CPUs without VNNI. On an Intel Xeon E5-2690 v4, stock ONNX Runtime differed from the committed fixture by up to 1.25 logits. The shifted model reproduced the committed short cases exactly.

| Logits against the reference | Port | Magpie Go |
| --- | --- | --- |
| 126 short texts, 3,939 tokens | max error 0.34, 0 labels changed | max error 3.10, 28 labels changed |
| 12 full windows, 6,144 tokens | max error 2.79, 8 labels changed | max error 2.51, 14 labels changed |
| Committed fixture, 627 tokens | max error 0, 0 labels changed | max error 0.41, 0 labels changed |

| Texts whose spans differ from the reference | Port | Magpie Go |
| --- | --- | --- |
| Short texts, threshold 0.5 | 0 of 126 | 13 of 126 |
| Short texts, threshold 0.8 | 0 of 126 | 1 of 126 |
| Multi-window texts, threshold 0.8 | 10 of 12 | 9 of 12 |

Full 510-token windows amplify last-bit differences through dynamic INT8 quantization. ONNX Runtime differs from itself by 0.21 logits between CPUs on the committed 510-token case. Multi-window differences therefore do not identify a more accurate implementation.

## Performance

Magpie selects SSE2 or AVX2 assembly when it starts. This crate forbids unsafe code and has no runtime CPU dispatch. On x86_64 without the `fma` target feature, each `f32::mul_add` becomes a library call. Build x86_64 applications with `-C target-feature=+fma` or `-C target-cpu=x86-64-v3` when every deployment CPU supports it. Results are identical with and without the feature. On six multi-window texts (43 KB, three workers, Intel Xeon E5-2690 v4), the feature reduced inference from 216 s to 32 s.
