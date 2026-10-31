use std::path::Path;
use std::sync::atomic::AtomicBool;

use super::*;

#[test]
fn user_gets_typed_configuration_and_cancellation_failures() {
    // Given a cancelled request, when checked, then no success can be returned.
    let token = Cancellation::default();
    token.cancel();
    assert!(matches!(token.check(), Err(Error::Cancelled)));
    // Given invalid limits, when loading, then validation precedes asset access.
    let options = Options {
        workers: 0,
        ..Options::default()
    };
    assert!(matches!(
        Model::load(Path::new("absent"), options),
        Err(Error::Configuration(_))
    ));
}

#[test]
fn user_gets_complete_bounded_window_coverage() {
    // Given edge lengths, when planning, then windows cover every token.
    for (count, expected) in [
        (0, vec![]),
        (1, std::iter::once(0..1).collect()),
        (510, std::iter::once(0..510).collect()),
        (511, vec![0..510, 382..511]),
        (892, vec![0..510, 382..892]),
        (893, vec![0..510, 382..892, 764..893]),
    ] {
        assert_eq!(scan::windows(count), expected);
    }
}

#[test]
fn user_gets_original_cause_in_tokenizer_error_chain() {
    use std::error::Error as _;

    // Given a tokenizer failure with an underlying I/O cause,
    let error = Error::Tokenizer {
        stage: "encode",
        source: Some(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "tokenizer test failure",
        ))),
    };
    // When the caller inspects the returned error chain,
    // then the stage and original I/O cause are available.
    assert!(matches!(
        &error,
        Error::Tokenizer {
            stage: "encode",
            ..
        }
    ));
    assert_eq!(
        error.to_string(),
        "tokenizer operation failed; verify the pinned tokenizer asset"
    );
    let source = error.source().unwrap();
    assert_eq!(source.to_string(), "tokenizer test failure");
    assert_eq!(
        source.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::InvalidData
    );
    // Given a tokenizer failure with no underlying cause,
    // when the caller inspects its error chain,
    let error = Error::Tokenizer {
        stage: "validate offsets",
        source: None,
    };
    // then no source is reported.
    assert!(error.source().is_none());
}

fn prediction(tag: usize, probability: f32) -> scan::Prediction {
    scan::Prediction { tag, probability }
}

// Session::scan requires pinned model assets to reach this scenario; this direct
// stitch-level test checks which overlapping prediction the user receives when windows
// disagree.
#[test]
fn user_gets_the_higher_margin_prediction_for_overlapping_windows() {
    // Given overlapping windows that disagree,
    // when their predictions are stitched,
    // then the returned labels reflect the winning margin.
    let selected = scan::stitch(
        8,
        &[0..6, 2..8],
        &[
            Arc::from(vec![prediction(1, 0.25); 6]),
            Arc::from(vec![prediction(3, 0.75); 6]),
        ],
    );
    let actual: Vec<_> = selected.iter().map(|p| (p.tag, p.probability)).collect();
    assert_eq!(
        actual,
        [
            (1, 0.25),
            (1, 0.25),
            (1, 0.25),
            (1, 0.25),
            (3, 0.75),
            (3, 0.75),
            (3, 0.75),
            (3, 0.75),
        ]
    );
}

// Session::scan requires pinned model assets to reach this scenario; this direct
// stitch-level test checks which prediction a user receives from reachable windows.
#[test]
fn user_keeps_the_earlier_prediction_near_the_window_margin_crossover() {
    // Given windows produced by the planner and predictions that disagree,
    // when token 445 is stitched with margins 64 and 63,
    // then the earlier window's prediction is returned.
    let ranges = scan::windows(640);
    assert_eq!(ranges, [0..510, 382..640]);
    let selected = scan::stitch(
        640,
        &ranges,
        &[
            Arc::from(vec![prediction(1, 0.25); 510]),
            Arc::from(vec![prediction(3, 0.75); 258]),
        ],
    );
    let actual: Vec<_> = selected[442..449]
        .iter()
        .map(|p| (p.tag, p.probability))
        .collect();
    assert_eq!(
        actual,
        [
            (1, 0.25),
            (1, 0.25),
            (1, 0.25),
            (1, 0.25),
            (3, 0.75),
            (3, 0.75),
            (3, 0.75),
        ]
    );
}

// Session::scan requires pinned model assets to reach this scenario; this direct
// stitch-level test checks results for a user input whose final prediction window is
// shorter than the preceding window.
#[test]
fn user_gets_predictions_from_a_short_final_window() {
    // Given a shorter final window,
    // when predictions are stitched,
    // then returned labels cover the input using the final window's actual length.
    let selected = scan::stitch(
        8,
        &[0..7, 4..8],
        &[
            Arc::from(vec![prediction(1, 0.25); 7]),
            Arc::from(vec![prediction(3, 0.75); 4]),
        ],
    );
    let actual: Vec<_> = selected.iter().map(|p| (p.tag, p.probability)).collect();
    assert_eq!(
        actual,
        [
            (1, 0.25),
            (1, 0.25),
            (1, 0.25),
            (1, 0.25),
            (1, 0.25),
            (1, 0.25),
            (3, 0.75),
            (3, 0.75),
        ]
    );
}

// Session::scan requires pinned model assets to reach this scenario; this direct stitch
// and decode test checks that a user's email spanning the window handoff remains one
// detected entity.
#[test]
fn user_gets_one_email_spanning_adjacent_prediction_windows() {
    // Given overlapping predictions that split an email across the window handoff,
    let mut first = vec![prediction(0, 1.0); 510];
    let mut last = vec![prediction(0, 1.0); 134];
    // Email starts before the overlap handoff at 446 and ends after token 510.
    first[444] = prediction(1, 0.875);
    first[445..].fill(prediction(2, 0.75));
    // The second window disagrees at 444 and 445, but has the smaller margin.
    last[62..64].fill(prediction(3, 1.0));
    last[64..132].fill(prediction(2, 0.625));
    // When the window predictions are stitched and decoded,
    let selected = scan::stitch(516, &[0..510, 382..516], &[first.into(), last.into()]);
    assert_eq!(selected.len(), 516);
    let actual: Vec<_> = selected[443..447]
        .iter()
        .map(|p| (p.tag, p.probability))
        .collect();
    assert_eq!(actual, [(0, 1.0), (1, 0.875), (2, 0.75), (2, 0.625)]);
    let actual: Vec<_> = selected[509..516]
        .iter()
        .map(|p| (p.tag, p.probability))
        .collect();
    assert_eq!(
        actual,
        [
            (2, 0.625),
            (2, 0.625),
            (2, 0.625),
            (2, 0.625),
            (2, 0.625),
            (0, 1.0),
            (0, 1.0)
        ]
    );
    let offsets: Vec<_> = (0..516).map(|i| (i, i + 1)).collect();
    // then the caller receives one email span crossing token 510.
    assert_eq!(
        scan::decode(&offsets, &selected, 0.5),
        [Span {
            bytes: 444..514,
            category: Category::Email,
            confidence: 0.625
        }]
    );
}

// Session::scan requires pinned model assets to reach this scenario; this direct
// decode-level test checks the user's result when a prediction sequence begins with an
// inside tag.
#[test]
fn user_gets_an_email_from_an_isolated_inside_tag() {
    // Given a single inside-tag prediction,
    // when the prediction is decoded,
    // then the user receives an email span for that token.
    assert_eq!(
        scan::decode(&[(2, 5)], &[prediction(2, 0.75)], 0.5),
        [Span {
            bytes: 2..5,
            category: Category::Email,
            confidence: 0.75
        }]
    );
}

// Session::scan requires pinned model assets to reach this scenario; this direct
// decode-level test checks separate results when a user's adjacent tokens both begin
// entities.
#[test]
fn user_gets_separate_emails_for_consecutive_begin_tags() {
    // Given adjacent begin-tag predictions,
    // when the predictions are decoded,
    // then the user receives two separate email spans.
    assert_eq!(
        scan::decode(
            &[(0, 3), (3, 6)],
            &[prediction(1, 0.75), prediction(1, 0.875)],
            0.5
        ),
        [
            Span {
                bytes: 0..3,
                category: Category::Email,
                confidence: 0.75
            },
            Span {
                bytes: 3..6,
                category: Category::Email,
                confidence: 0.875
            },
        ]
    );
}

// Session::scan requires pinned model assets to reach this scenario; this direct
// decode-level test checks the user's returned spans when an inside-tag prediction
// changes category.
#[test]
fn user_gets_a_new_entity_when_the_inside_tag_changes_category() {
    // Given an inside tag whose category differs from the active entity,
    // when the predictions are decoded,
    // then the user receives separate entities for both categories.
    assert_eq!(
        scan::decode(
            &[(0, 2), (2, 4), (4, 7)],
            &[
                prediction(1, 0.875),
                prediction(4, 0.75),
                prediction(4, 0.625)
            ],
            0.5,
        ),
        [
            Span {
                bytes: 0..2,
                category: Category::Email,
                confidence: 0.875
            },
            Span {
                bytes: 2..7,
                category: Category::Phone,
                confidence: 0.625
            },
        ]
    );
}

// Session::scan requires pinned model assets to reach this scenario; this direct
// decode-level test checks that a user's entity ends at an outside-tag prediction.
#[test]
fn user_gets_an_entity_ended_before_an_outside_tag() {
    // Given an outside tag between entity predictions,
    // when the predictions are decoded,
    // then the user receives an entity that ends before the outside tag.
    assert_eq!(
        scan::decode(
            &[(0, 2), (2, 4), (4, 6), (6, 8)],
            &[
                prediction(1, 0.875),
                prediction(2, 0.75),
                prediction(0, 0.125),
                prediction(2, 0.625)
            ],
            0.5,
        ),
        [
            Span {
                bytes: 0..4,
                category: Category::Email,
                confidence: 0.75
            },
            Span {
                bytes: 6..8,
                category: Category::Email,
                confidence: 0.625
            },
        ]
    );
}

// Session::scan requires pinned model assets to reach this scenario; this direct
// decode-level test checks the user's result for tokenizer offsets that repeat or contain
// no bytes.
#[test]
fn user_gets_one_entity_for_repeated_offsets_and_ignores_empty_offsets() {
    // Given repeated and empty offsets for predicted tokens,
    // when the predictions are decoded,
    // then the user receives one entity spanning the nonempty offsets.
    assert_eq!(
        scan::decode(
            &[(0, 3), (0, 3), (3, 3), (3, 6)],
            &[
                prediction(1, 0.875),
                prediction(2, 0.625),
                prediction(0, 0.125),
                prediction(2, 0.75)
            ],
            0.5,
        ),
        [Span {
            bytes: 0..6,
            category: Category::Email,
            confidence: 0.625
        }]
    );
}

// Session::scan requires pinned model assets to reach this scenario; this direct
// decode-level test checks the user's confidence filtering result for an entity with a
// weak middle token.
#[test]
fn user_gets_the_whole_entity_filtered_when_middle_confidence_is_low() {
    // Given an entity with a low-confidence middle token and a separate strong entity,
    // when the user raises the confidence threshold,
    // then only the strong entity remains in the result.
    let offsets = [(0, 2), (2, 4), (4, 6), (7, 9)];
    let predictions = [
        prediction(1, 0.875),
        prediction(2, 0.25),
        prediction(2, 0.75),
        prediction(3, 0.875),
    ];
    assert_eq!(
        scan::decode(&offsets, &predictions, 0.25),
        [
            Span {
                bytes: 0..6,
                category: Category::Email,
                confidence: 0.25
            },
            Span {
                bytes: 7..9,
                category: Category::Phone,
                confidence: 0.875
            },
        ]
    );
    assert_eq!(
        scan::decode(&offsets, &predictions, 0.5),
        [Span {
            bytes: 7..9,
            category: Category::Phone,
            confidence: 0.875
        }]
    );
}

#[derive(serde::Deserialize)]
struct Oracle {
    oracle: String,
    assets: std::collections::BTreeMap<String, String>,
    cases: Vec<Case>,
}
#[derive(serde::Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    offsets: Vec<(usize, usize)>,
    logits: Vec<Vec<f32>>,
}

fn model(workers: usize) -> Model {
    let directory = std::env::var_os("ASTRLINK_MODEL_DIR")
        .expect("set ASTRLINK_MODEL_DIR to the pinned model assets");
    Model::load(
        Path::new(&directory),
        Options {
            workers,
            ..Options::default()
        },
    )
    .unwrap()
}

#[test]
#[ignore = "requires pinned assets; run explicitly with ASTRLINK_MODEL_DIR"]
fn user_gets_predictions_qualified_against_independent_onnx_runtime() {
    // Given synthetic ORT fixtures, when executing Rust, then offsets and logits agree.
    let model = model(1);
    let oracle: Oracle = serde_json::from_str(include_str!("../testdata/oracle.json")).unwrap();
    assert!(
        !oracle.cases.is_empty(),
        "oracle must contain reference cases"
    );
    assert!(
        !oracle.oracle.is_empty(),
        "oracle runtime version must be recorded"
    );
    let directory = std::env::var_os("ASTRLINK_MODEL_DIR").unwrap();
    use sha2::{Digest, Sha256};
    assert_eq!(oracle.assets.len(), 3);
    for file in ["model_int8.onnx", "model_int8.onnx_data", "tokenizer.json"] {
        let bytes = std::fs::read(Path::new(&directory).join(file)).unwrap();
        assert_eq!(oracle.assets[file], format!("{:x}", Sha256::digest(bytes)));
    }
    let mut max_error = 0.0_f32;
    let mut labels_changed = 0;
    let mut rows = 0;
    for case in oracle.cases {
        let encoding = model.0.tokenizer.encode(case.text.as_str(), false).unwrap();
        assert_eq!(encoding.get_ids(), case.ids);
        assert_eq!(encoding.get_offsets(), case.offsets);
        let ids: Vec<u32> = std::iter::once(0)
            .chain(case.ids)
            .chain(std::iter::once(2))
            .collect();
        let actual = model.0.net.logits(&ids, &AtomicBool::new(false)).unwrap();
        assert_eq!(actual.len(), case.logits.len() * 21);
        for (actual, expected) in actual.chunks_exact(21).zip(&case.logits) {
            assert_eq!(expected.len(), 21);
            assert!(expected.iter().all(|value| value.is_finite()));
            assert!(actual.iter().all(|value| value.is_finite()));
            for (a, e) in actual.iter().zip(expected) {
                max_error = max_error.max((a - e).abs());
            }
            let argmax = |row: &[f32]| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            labels_changed += usize::from(argmax(actual) != argmax(expected));
            rows += 1;
        }
    }
    println!("ORT reference: max_abs_error={max_error}, changed_labels={labels_changed}/{rows}");
    assert!(
        max_error <= 0.002,
        "maximum logit error {max_error} exceeds qualification limit"
    );
    assert_eq!(labels_changed, 0);
}

#[test]
#[ignore = "requires pinned assets; run explicitly with ASTRLINK_MODEL_DIR"]
fn user_gets_identical_serial_parallel_cached_and_edited_scans() {
    // Given a multilingual conversation across windows, when scan mode changes,
    // then the public spans remain equal and cache hits are observable.
    let one = model(1);
    let parallel = model(4);
    let cancel = Cancellation::default();
    let text = "联系人：张三，邮箱 zhangsan@example.com。 Alice: alice@example.org 👩🏽‍💻\n".repeat(24);
    let mut session = parallel.session(64).unwrap();
    let cold = session.scan(&text, 0.8, &cancel).unwrap();
    assert!(cold.tokens > 510);
    assert!(!cold.spans.is_empty());
    let serial = one.session(0).unwrap().scan(&text, 0.8, &cancel).unwrap();
    assert_eq!(cold.spans, serial.spans);
    let hot = session.scan(&text, 0.8, &cancel).unwrap();
    assert_eq!(hot.spans, cold.spans);
    assert_eq!(hot.computed_windows, 0);
    assert!(hot.cached_windows > 0);
    let example = "Contact alice@example.com.";
    let detected = session.scan(example, 0.8, &cancel).unwrap();
    assert!(detected.spans.iter().any(|span| {
        span.category == Category::Email && &example[span.bytes.clone()] == "alice@example.com"
    }));
    let changed = format!("{text} New contact: other@example.net.");
    let warm = session.scan(&changed, 0.8, &cancel).unwrap();
    let uncached = parallel
        .session(0)
        .unwrap()
        .scan(&changed, 0.8, &cancel)
        .unwrap();
    assert_eq!(warm.spans, uncached.spans);
    assert!(warm.cached_windows > 0);
    for span in &warm.spans {
        assert!(changed.get(span.bytes.clone()).is_some());
    }
    let strict = session.scan(&changed, 1.0, &cancel).unwrap();
    assert_eq!(strict.computed_windows, 0);
    assert_eq!(
        strict.spans,
        warm.spans
            .iter()
            .filter(|s| s.confidence >= 1.0)
            .cloned()
            .collect::<Vec<_>>()
    );
    let mut separate = parallel.session(1).unwrap();
    assert_eq!(
        separate.scan(example, 0.8, &cancel).unwrap().cached_windows,
        0
    );
    assert_eq!(
        separate.scan(example, 0.8, &cancel).unwrap().cached_windows,
        1
    );
    separate
        .scan("Contact other@example.net.", 0.8, &cancel)
        .unwrap();
    assert_eq!(
        separate.scan(example, 0.8, &cancel).unwrap().cached_windows,
        0
    );
    session.clear_cache();
    let after_clear = session.scan(&text, 0.8, &cancel).unwrap();
    assert_eq!(after_clear.cached_windows, 0);
    assert_eq!(after_clear.spans, cold.spans);
    let cancelled = Cancellation::default();
    cancelled.cancel();
    assert!(matches!(
        session.scan(&text, 0.8, &cancelled),
        Err(Error::Cancelled)
    ));
    assert!(matches!(
        session.scan(&text, f32::NAN, &cancel),
        Err(Error::Configuration(_))
    ));
}

#[test]
#[ignore = "requires pinned assets; run explicitly with ASTRLINK_MODEL_DIR"]
fn user_gets_busy_and_input_limits_without_losing_admission() {
    // Given an admitted public scan, when another scan starts, then it reports Busy.
    let directory = std::env::var_os("ASTRLINK_MODEL_DIR").unwrap();
    let model = Model::load(
        Path::new(&directory),
        Options {
            workers: 1,
            max_in_flight: 1,
            ..Options::default()
        },
    )
    .unwrap();
    let mut session = model.session(0).unwrap();
    let cancellation = Cancellation::default();
    let worker_cancel = cancellation.clone();
    let mut worker_session = model.session(8).unwrap();
    let (completed, receive) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let result = worker_session.scan(&"hello world ".repeat(8000), 0.8, &worker_cancel);
        let _ = completed.send((result, worker_session));
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while model.0.active.load(Ordering::Relaxed) == 0 && !worker.is_finished() {
        assert!(std::time::Instant::now() < deadline, "scan did not start");
        std::thread::yield_now();
    }
    let competing = session.scan("hello", 0.8, &Cancellation::default());
    cancellation.cancel();
    let (cancelled, mut worker_session) = receive
        .recv_timeout(std::time::Duration::from_secs(10))
        .unwrap_or_else(|_| panic!("cancelled scan did not settle within ten seconds"));
    worker.join().unwrap();
    assert!(matches!(competing, Err(Error::Busy)));
    assert!(matches!(cancelled, Err(Error::Cancelled)));
    // The cancelled admitted call releases admission for a subsequent scan.
    worker_session
        .scan("hello world", 0.8, &Cancellation::default())
        .unwrap();
    assert_eq!(model.0.active.load(Ordering::Relaxed), 0);
    let oversized = "x".repeat(model.0.options.max_bytes + 1);
    assert!(matches!(
        session.scan(&oversized, 0.8, &Cancellation::default()),
        Err(Error::InputLimit { unit: "byte", .. })
    ));
    assert_eq!(model.0.active.load(Ordering::Relaxed), 0);
    assert!(
        session
            .scan("", 0.8, &Cancellation::default())
            .unwrap()
            .spans
            .is_empty()
    );
    let directory = std::env::var_os("ASTRLINK_MODEL_DIR").unwrap();
    let limited = Model::load(
        Path::new(&directory),
        Options {
            max_tokens: 1,
            ..Options::default()
        },
    )
    .unwrap();
    let mut session = limited.session(0).unwrap();
    assert!(matches!(
        session.scan("one two three", 0.8, &Cancellation::default()),
        Err(Error::InputLimit {
            unit: "token",
            limit: 1
        })
    ));
    assert!(session.scan("", 0.8, &Cancellation::default()).is_ok());
}

#[test]
fn user_gets_explicit_missing_and_corrupt_asset_failures() {
    // Given an isolated asset directory, when bytes are missing or changed,
    // then loading reports the file and never executes an unverified model.
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("astrlink-assets-{}-{unique}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let missing = Model::load(&directory, Options::default());
    let path = directory.join("model_int8.onnx");
    std::fs::write(&path, b"corrupt").unwrap();
    let truncated = Model::load(&directory, Options::default());
    std::fs::write(&path, vec![0_u8; 330_781]).unwrap();
    let wrong_hash = Model::load(&directory, Options::default());
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(&directory).unwrap();
    assert!(matches!(missing, Err(Error::Io { path: p, .. }) if p == path));
    for result in [truncated, wrong_hash] {
        assert!(matches!(
            result,
            Err(Error::Integrity {
                file: "model_int8.onnx"
            })
        ));
    }
}
