// Window selection and BIO decoding derive from Magpie (MIT, Copyright 2026 yetone).
use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use crate::{Cancellation, Error, Model};

const WINDOW: usize = 510;
const STRIDE: usize = WINDOW - 128;

/// AstrLink's entity categories, in model output order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    /// Email address.
    Email,
    /// Phone number.
    Phone,
    /// Account identifier.
    Account,
    /// Payment card number.
    PaymentCard,
    /// IP address.
    IpAddress,
    /// URL.
    Url,
    /// Secret, such as an access token.
    CommonSecret,
    /// Private postal address.
    PrivateAddress,
    /// Private date.
    PrivateDate,
    /// Person name.
    PrivatePerson,
}

const CATEGORIES: [Category; 10] = [
    Category::Email,
    Category::Phone,
    Category::Account,
    Category::PaymentCard,
    Category::IpAddress,
    Category::Url,
    Category::CommonSecret,
    Category::PrivateAddress,
    Category::PrivateDate,
    Category::PrivatePerson,
];

/// A detected range in the original input. Detection can miss or mislabel data.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    /// Half-open UTF-8 byte range, safe to slice from the original input.
    pub bytes: Range<usize>,
    /// Predicted entity category.
    pub category: Category,
    /// Minimum selected token probability in this span, between zero and one.
    pub confidence: f32,
}

/// A complete scan. Cancellation and other failures return no `Scan`.
#[derive(Debug, PartialEq)]
pub struct Scan {
    /// Detected spans that meet the requested threshold, in token order.
    pub spans: Vec<Span>,
    /// Number of content tokens, excluding added start and end tokens.
    pub tokens: usize,
    /// Number of windows whose predictions came from this session's cache.
    pub cached_windows: usize,
    /// Number of windows evaluated during this call.
    pub computed_windows: usize,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Prediction {
    tag: usize,
    probability: f32,
}

type Predictions = Arc<[Prediction]>;

/// A caller-owned scan session with a bounded, in-memory window cache.
///
/// Use one session per data owner. Mutable access prevents overlapping scans
/// within a session. Different sessions share the model's compute budget.
pub struct Session {
    model: Model,
    capacity: usize,
    cache: HashMap<Arc<[u32]>, Predictions>,
    order: VecDeque<Arc<[u32]>>,
}

impl Session {
    pub(crate) fn new(model: Model, capacity: usize) -> Self {
        Self {
            model,
            capacity,
            cache: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Remove cached token IDs and predictions. This does not zero allocator memory.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.order.clear();
    }

    /// Detect spans in `text` at the given probability threshold (0 through 1).
    ///
    /// The call is synchronous. Async applications must use a bounded blocking
    /// executor. Threshold changes reuse cached predictions without changing them.
    /// Observed cancellation waits for active workers and returns [`Error::Cancelled`].
    /// See [`Cancellation`] for the completion race contract.
    /// Windows from failed calls are not added to the cache.
    pub fn scan(
        &mut self,
        text: &str,
        threshold: f32,
        cancel: &Cancellation,
    ) -> Result<Scan, Error> {
        cancel.check()?;
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(Error::Configuration(
                "threshold must be finite and between 0 and 1",
            ));
        }
        let model = self.model.clone();
        let inner = &model.0;
        let _permit = inner.admit()?;
        if text.len() > inner.options.max_bytes {
            return Err(Error::InputLimit {
                unit: "byte",
                limit: inner.options.max_bytes,
            });
        }
        let encoded = inner
            .tokenizer
            .encode(text, false)
            .map_err(|_| Error::Tokenizer)?;
        cancel.check()?;
        let ids = encoded.get_ids();
        if ids.len() > inner.options.max_tokens {
            return Err(Error::InputLimit {
                unit: "token",
                limit: inner.options.max_tokens,
            });
        }
        let offsets = encoded.get_offsets();
        if offsets.iter().any(|&(start, end)| {
            start > end
                || end > text.len()
                || !text.is_char_boundary(start)
                || !text.is_char_boundary(end)
        }) {
            return Err(Error::Tokenizer);
        }
        let ranges = windows(ids.len());
        let keys: Vec<Arc<[u32]>> = ranges
            .iter()
            .map(|range| {
                let mut key = Vec::with_capacity(range.len() + 2);
                key.push(0);
                key.extend_from_slice(&ids[range.clone()]);
                key.push(2);
                Arc::from(key)
            })
            .collect();
        let cached: Vec<Option<Predictions>> = keys
            .iter()
            .map(|key| self.cache.get(key).cloned())
            .collect();
        let cached_windows = cached.iter().filter(|entry| entry.is_some()).count();
        let predictions: Result<Vec<Predictions>, Error> = inner.pool.install(|| {
            keys.par_iter()
                .zip(&cached)
                .map(|(key, hit)| {
                    cancel.check()?;
                    if let Some(hit) = hit {
                        return Ok(Arc::clone(hit));
                    }
                    let logits = inner.net.logits(key, &cancel.0)?;
                    let predictions = logits
                        .chunks_exact(21)
                        .skip(1)
                        .take(key.len() - 2)
                        .map(predict)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Arc::from(predictions))
                })
                .collect()
        });
        let predictions = predictions?;
        cancel.check()?;
        let selected = stitch(ids.len(), &ranges, &predictions);
        let spans = decode(offsets, &selected, threshold);
        for (key, prediction) in keys.into_iter().zip(predictions) {
            if self.capacity == 0 || self.cache.contains_key(&key) {
                continue;
            }
            if self.cache.len() == self.capacity {
                if let Some(old) = self.order.pop_front() {
                    self.cache.remove(&old);
                }
            }
            self.order.push_back(Arc::clone(&key));
            self.cache.insert(key, prediction);
        }
        Ok(Scan {
            spans,
            tokens: ids.len(),
            cached_windows,
            computed_windows: ranges.len() - cached_windows,
        })
    }
}

pub(crate) fn windows(tokens: usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < tokens {
        let end = start.saturating_add(WINDOW).min(tokens);
        ranges.push(start..end);
        if end == tokens {
            break;
        }
        start += STRIDE;
    }
    ranges
}

fn predict(row: &[f32]) -> Result<Prediction, Error> {
    if row.iter().any(|value| !value.is_finite()) {
        return Err(Error::Model("non-finite logits".into()));
    }
    let mut tag = 0;
    for i in 1..row.len() {
        if row[i] > row[tag] {
            tag = i;
        }
    }
    let sum: f64 = row
        .iter()
        .map(|value| f64::from(*value - row[tag]).exp())
        .sum();
    Ok(Prediction {
        tag,
        probability: (1.0 / sum) as f32,
    })
}

pub(crate) fn stitch(
    tokens: usize,
    ranges: &[Range<usize>],
    predictions: &[Predictions],
) -> Vec<Prediction> {
    let mut selected = vec![Prediction::default(); tokens];
    let mut margins = vec![None; tokens];
    for (range, prediction) in ranges.iter().zip(predictions) {
        for (i, value) in prediction.iter().enumerate() {
            let margin = i.min(range.len() - 1 - i);
            let at = range.start + i;
            if margins[at].is_none_or(|previous| margin > previous) {
                margins[at] = Some(margin);
                selected[at] = *value;
            }
        }
    }
    selected
}

pub(crate) fn decode(
    offsets: &[(usize, usize)],
    predictions: &[Prediction],
    threshold: f32,
) -> Vec<Span> {
    let mut spans: Vec<Span> = Vec::new();
    let mut current: Option<usize> = None;
    for (&(start, end), prediction) in offsets.iter().zip(predictions) {
        if start == end {
            continue;
        }
        if prediction.tag == 0 {
            current = None;
            continue;
        }
        let category = CATEGORIES[(prediction.tag - 1) / 2];
        match current {
            Some(index) if prediction.tag % 2 == 0 && spans[index].category == category => {
                spans[index].bytes.end = spans[index].bytes.end.max(end);
                spans[index].confidence = spans[index].confidence.min(prediction.probability);
            }
            _ => {
                spans.push(Span {
                    bytes: start..end,
                    category,
                    confidence: prediction.probability,
                });
                current = Some(spans.len() - 1);
            }
        }
    }
    spans.retain(|span| span.confidence >= threshold);
    spans
}
