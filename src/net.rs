//! The fixed AstrLink Guard INT8 BERT network.
//!
//! Adapted from Magpie's internal/pii/guard/bert.go and kernel.go by yetone
//! (<https://github.com/yetone/magpie>), distributed under the MIT license.
//! Erf and softmax arithmetic also adapt ONNX Runtime MLAS (Microsoft, MIT).
//! Integer products widen before multiplication; all inference buffers belong
//! to the call. Window-level scheduling is the caller's responsibility.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::Error;
use crate::onnx::Tensor;

const HIDDEN: usize = 384;
const HEADS: usize = 12;
const HEAD_DIM: usize = HIDDEN / HEADS;
const INTERMEDIATE: usize = 1536;
const LAYERS: usize = 12;
const MAX_POSITIONS: usize = 512;
const VOCABULARY: usize = 40003;
const LABELS: usize = 21;

const FLOAT: i32 = 1;
const UINT8: i32 = 2;
const INT8: i32 = 3;
const INT64: i32 = 7;

pub(crate) struct Net {
    word: Embedding,
    position: Vec<f32>,
    token_type: Vec<f32>,
    position_ids: Vec<usize>,
    embedding_norm: LayerNorm,
    layers: Vec<Layer>,
    classifier: QuantizedMatrix,
    classifier_bias: Vec<f32>,
}

struct Embedding {
    values: Vec<u8>,
    scales: Vec<f32>,
    zero_points: Vec<i32>,
}

struct Layer {
    qkv: QuantizedMatrix,
    qkv_bias: Vec<f32>,
    attention_out: QuantizedMatrix,
    attention_bias: Vec<f32>,
    attention_norm: LayerNorm,
    up: QuantizedMatrix,
    up_bias: Vec<f32>,
    smooth: Vec<f32>,
    down: QuantizedMatrix,
    down_bias: Vec<f32>,
    output_norm: LayerNorm,
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

// ONNX stores [input, output]; columns are contiguous here for dot products.
struct QuantizedMatrix {
    input: usize,
    output: usize,
    weights: Vec<i8>,
    scales: Vec<f32>,
    zero_points: Vec<i32>,
    sums: Vec<i32>,
}

impl Net {
    pub(crate) fn load(mut tensors: HashMap<String, Tensor>) -> Result<Self, Error> {
        let word = Embedding::load(
            &mut tensors,
            "inner.bert.embeddings.word_embeddings",
            VOCABULARY,
        )?;
        let position = Embedding::load(
            &mut tensors,
            "inner.bert.embeddings.position_embeddings",
            MAX_POSITIONS,
        )?
        .dequantize()?;
        let token_type = Embedding::load(
            &mut tensors,
            "inner.bert.embeddings.token_type_embeddings",
            2,
        )?
        .dequantize()?;
        let positions = take_tensor(&mut tensors, "onnx::Slice_206", INT64, &[1, MAX_POSITIONS])?;
        let mut position_ids = Vec::with_capacity(MAX_POSITIONS);
        for bytes in positions.raw.chunks_exact(8) {
            let mut raw = [0; 8];
            raw.copy_from_slice(bytes);
            let id = i64::from_le_bytes(raw);
            if !(0..MAX_POSITIONS as i64).contains(&id) {
                return Err(Error::Model(
                    "position ID is outside the embedding table".into(),
                ));
            }
            position_ids.push(id as usize);
        }
        let embedding_norm = LayerNorm::load(&mut tensors, "inner.bert.embeddings.LayerNorm")?;
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let prefix = format!("inner.bert.encoder.layer.{i}.");
            let matmul = 1299 + 16 * i;
            layers.push(Layer {
                qkv: QuantizedMatrix::load(
                    &mut tensors,
                    &format!("Attention_{i}_qkv_weight"),
                    HIDDEN,
                    3 * HIDDEN,
                )?,
                qkv_bias: floats(&mut tensors, &format!("Attention_{i}_qkv_bias"), 3 * HIDDEN)?,
                attention_out: QuantizedMatrix::load(
                    &mut tensors,
                    &format!("onnx::MatMul_{matmul}"),
                    HIDDEN,
                    HIDDEN,
                )?,
                attention_bias: floats(
                    &mut tensors,
                    &format!("{prefix}attention.output.dense.bias"),
                    HIDDEN,
                )?,
                attention_norm: LayerNorm::load(
                    &mut tensors,
                    &format!("{prefix}attention.output.LayerNorm"),
                )?,
                up: QuantizedMatrix::load(
                    &mut tensors,
                    &format!("onnx::MatMul_{}", matmul + 1),
                    HIDDEN,
                    INTERMEDIATE,
                )?,
                up_bias: floats(
                    &mut tensors,
                    &format!("{prefix}intermediate.dense.bias"),
                    INTERMEDIATE,
                )?,
                smooth: positive_floats(
                    &mut tensors,
                    &format!("r15_smooth_inv_s_{i}"),
                    INTERMEDIATE,
                )?,
                down: QuantizedMatrix::load(
                    &mut tensors,
                    &format!("onnx::MatMul_{}", matmul + 2),
                    INTERMEDIATE,
                    HIDDEN,
                )?,
                down_bias: floats(&mut tensors, &format!("{prefix}output.dense.bias"), HIDDEN)?,
                output_norm: LayerNorm::load(&mut tensors, &format!("{prefix}output.LayerNorm"))?,
            });
        }
        let classifier = QuantizedMatrix::load(&mut tensors, "onnx::MatMul_1478", HIDDEN, LABELS)?;
        let classifier_bias = floats(&mut tensors, "inner.classifier.bias", LABELS)?;
        Ok(Self {
            word,
            position,
            token_type,
            position_ids,
            embedding_norm,
            layers,
            classifier,
            classifier_bias,
        })
    }

    /// Returns row-major [ids.len(), 21] logits for one unpadded window.
    pub(crate) fn logits(&self, ids: &[u32], cancelled: &AtomicBool) -> Result<Vec<f32>, Error> {
        check_cancelled(cancelled)?;
        if ids.len() > MAX_POSITIONS {
            return Err(Error::Model("a network window exceeds 512 tokens".into()));
        }
        if ids.iter().any(|&id| id >= VOCABULARY as u32) {
            return Err(Error::Model(
                "token ID is outside the embedding table".into(),
            ));
        }
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let rows = ids.len();
        let mut x = vec![0.0; rows * HIDDEN];
        for (t, (&id, row)) in ids.iter().zip(x.chunks_exact_mut(HIDDEN)).enumerate() {
            let word = &self.word.values[id as usize * HIDDEN..(id as usize + 1) * HIDDEN];
            let position =
                &self.position[self.position_ids[t] * HIDDEN..(self.position_ids[t] + 1) * HIDDEN];
            for c in 0..HIDDEN {
                row[c] = (f32::from(word[c]) - self.word.zero_points[c] as f32)
                    * self.word.scales[c]
                    + self.token_type[c];
                row[c] += position[c];
            }
        }
        self.embedding_norm.apply(&mut x);
        for layer in &self.layers {
            check_cancelled(cancelled)?;
            let mut attended = {
                let mut qkv = layer.qkv.matmul(&x)?;
                add_bias(&mut qkv, &layer.qkv_bias);
                let context = attention(&qkv, rows);
                layer.attention_out.matmul(&context)?
            };
            layer
                .attention_norm
                .skip(&mut attended, &x, &layer.attention_bias);
            let mut hidden = layer.up.matmul(&attended)?;
            for row in hidden.chunks_exact_mut(INTERMEDIATE) {
                for (c, value) in row.iter_mut().enumerate() {
                    *value = gelu(*value + layer.up_bias[c]) * layer.smooth[c];
                }
            }
            let mut out = layer.down.matmul(&hidden)?;
            layer
                .output_norm
                .skip(&mut out, &attended, &layer.down_bias);
            x = out;
        }
        check_cancelled(cancelled)?;
        let mut out = self.classifier.matmul(&x)?;
        add_bias(&mut out, &self.classifier_bias);
        ensure_finite(&out, "classifier output")?;
        check_cancelled(cancelled)?;
        Ok(out)
    }
}

// Checking raw lengths here keeps all indexing below independent of the parser's
// validation. Shapes and types are exact, including the fixed classifier width.
fn take_tensor(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    dtype: i32,
    dims: &[usize],
) -> Result<Tensor, Error> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| Error::Model(format!("missing tensor {name}")))?;
    if tensor.dtype != dtype || tensor.dims != dims {
        return Err(Error::Model(format!("incorrect type or shape for {name}")));
    }
    let width = match dtype {
        FLOAT => 4,
        UINT8 | INT8 => 1,
        INT64 => 8,
        _ => return Err(Error::Model(format!("unsupported tensor type for {name}"))),
    };
    let bytes = dims
        .iter()
        .try_fold(width, |size: usize, dim| size.checked_mul(*dim));
    if bytes != Some(tensor.raw.len()) {
        return Err(Error::Model(format!("incorrect data length for {name}")));
    }
    Ok(tensor)
}

fn floats(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, Error> {
    let values = take_tensor(tensors, name, FLOAT, &[len])?.floats()?;
    ensure_finite(&values, name)?;
    Ok(values)
}

fn positive_floats(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, Error> {
    let values = floats(tensors, name, len)?;
    if values.iter().any(|&value| value <= 0.0) {
        return Err(Error::Model(format!("nonpositive scale in {name}")));
    }
    Ok(values)
}

fn ensure_finite(values: &[f32], name: &str) -> Result<(), Error> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Model(format!("nonfinite value in {name}")));
    }
    Ok(())
}

fn check_cancelled(cancelled: &AtomicBool) -> Result<(), Error> {
    // This flag carries no other shared data; observing its value is sufficient.
    if cancelled.load(Ordering::Relaxed) {
        return Err(Error::Cancelled);
    }
    Ok(())
}

impl Embedding {
    fn load(tensors: &mut HashMap<String, Tensor>, name: &str, rows: usize) -> Result<Self, Error> {
        let values = take_tensor(
            tensors,
            &format!("{name}.weight_quantized"),
            UINT8,
            &[rows, HIDDEN],
        )?
        .raw;
        let scales = positive_floats(tensors, &format!("{name}.weight_scale"), HIDDEN)?;
        let zero_points = take_tensor(
            tensors,
            &format!("{name}.weight_zero_point"),
            UINT8,
            &[HIDDEN],
        )?
        .raw
        .into_iter()
        .map(i32::from)
        .collect();
        Ok(Self {
            values,
            scales,
            zero_points,
        })
    }

    fn dequantize(self) -> Result<Vec<f32>, Error> {
        let values: Vec<_> = self
            .values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let c = i % HIDDEN;
                (i32::from(v) - self.zero_points[c]) as f32 * self.scales[c]
            })
            .collect();
        ensure_finite(&values, "dequantized embedding")?;
        Ok(values)
    }
}

impl QuantizedMatrix {
    fn load(
        tensors: &mut HashMap<String, Tensor>,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Self, Error> {
        let quantized = take_tensor(
            tensors,
            &format!("{name}_quantized"),
            INT8,
            &[input, output],
        )?;
        let scales = positive_floats(tensors, &format!("{name}_scale"), output)?;
        let zero_points: Vec<_> =
            take_tensor(tensors, &format!("{name}_zero_point"), INT8, &[output])?
                .raw
                .into_iter()
                .map(|v| i32::from(v as i8))
                .collect();
        let mut weights = vec![0; input * output];
        let mut sums = vec![0; output];
        for (c, column) in weights.chunks_exact_mut(input).enumerate() {
            for (r, weight) in column.iter_mut().enumerate() {
                *weight = quantized.raw[r * output + c] as i8;
                sums[c] += i32::from(*weight);
            }
        }
        Ok(Self {
            input,
            output,
            weights,
            scales,
            zero_points,
            sums,
        })
    }

    fn matmul(&self, x: &[f32]) -> Result<Vec<f32>, Error> {
        let (quantized, scale, zero_point) = quantize(x)?;
        let rows = x.len() / self.input;
        let mut out = vec![0.0; rows * self.output];
        for (row, result) in quantized
            .chunks_exact(self.input)
            .zip(out.chunks_exact_mut(self.output))
        {
            let sum: i32 = row.iter().map(|&v| i32::from(v)).sum();
            for (c, column) in self.weights.chunks_exact(self.input).enumerate() {
                let weight_zp = self.zero_points[c];
                // Expanding Σ(q - xz)(w - wz) keeps the inner product simple.
                // The largest input width is 1536: even 1536 * 255 * 255
                // and every intermediate correction fit comfortably in i32.
                let value = dot(row, column) - weight_zp * sum - zero_point * self.sums[c]
                    + self.input as i32 * zero_point * weight_zp;
                result[c] = value as f32 * (scale * self.scales[c]);
            }
        }
        Ok(out)
    }
}

// DynamicQuantizeLinear is per tensor, across every token in this matmul.
// Zero belongs to the observed range; each division rounds in f32 before the
// ties-to-even rounding required by ONNX QuantizeLinear.
fn quantize(x: &[f32]) -> Result<(Vec<u8>, f32, i32), Error> {
    ensure_finite(x, "activation")?;
    let (mut lo, mut hi) = (0.0_f32, 0.0_f32);
    for &value in x {
        lo = lo.min(value);
        hi = hi.max(value);
    }
    let scale = if hi == lo { 1.0 } else { (hi - lo) / 255.0 };
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Model("activation range cannot be quantized".into()));
    }
    let zero_point = (-lo / scale).clamp(0.0, 255.0).round_ties_even() as i32;
    let values = x
        .iter()
        .map(|&value| ((value / scale).round_ties_even() as i32 + zero_point).clamp(0, 255) as u8)
        .collect();
    Ok((values, scale, zero_point))
}

// A widening integer reduction permits vectorization without saturating i16
// pairwise products (which would lose exactness for extreme UINT8/INT8 pairs).
fn dot(a: &[u8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

impl LayerNorm {
    fn load(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<Self, Error> {
        Ok(Self {
            weight: floats(tensors, &format!("{prefix}.weight"), HIDDEN)?,
            bias: floats(tensors, &format!("{prefix}.bias"), HIDDEN)?,
        })
    }

    fn apply(&self, x: &mut [f32]) {
        for row in x.chunks_exact_mut(HIDDEN) {
            let mean = row.iter().sum::<f32>() / HIDDEN as f32;
            // The exported CPU LayerNormalization and SkipLayerNormalization
            // use E[x²] - E[x]² and divide by the f32 standard deviation.
            // Reassociating these operations can change subsequent INT8 bins.
            let mean_square = row.iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32;
            let deviation = (mean_square - mean * mean + 1e-12).sqrt();
            for (c, value) in row.iter_mut().enumerate() {
                *value = (*value - mean) / deviation * self.weight[c] + self.bias[c];
            }
        }
    }

    fn skip(&self, out: &mut [f32], residual: &[f32], bias: &[f32]) {
        for (i, (value, skip)) in out.iter_mut().zip(residual).enumerate() {
            *value = *value + skip + bias[i % HIDDEN];
        }
        self.apply(out);
    }
}

fn add_bias(x: &mut [f32], bias: &[f32]) {
    for row in x.chunks_exact_mut(bias.len()) {
        for (value, bias) in row.iter_mut().zip(bias) {
            *value += bias;
        }
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + mlas_erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn attention(qkv: &[f32], tokens: usize) -> Vec<f32> {
    let scale = (1.0 / (HEAD_DIM as f64).sqrt()) as f32;
    let mut context = vec![0.0; tokens * HIDDEN];
    // Transposed keys let the score loop traverse contiguous data. Heads are
    // evaluated serially; the caller can run independent windows in its pool.
    let mut keys = vec![0.0; HEAD_DIM * tokens];
    let mut values = vec![0.0; tokens * HEAD_DIM];
    let mut probabilities = vec![0.0; tokens];
    for head in 0..HEADS {
        for (token, row) in qkv.chunks_exact(3 * HIDDEN).enumerate() {
            for c in 0..HEAD_DIM {
                keys[c * tokens + token] = row[HIDDEN + head * HEAD_DIM + c];
            }
            values[token * HEAD_DIM..(token + 1) * HEAD_DIM].copy_from_slice(
                &row[2 * HIDDEN + head * HEAD_DIM..2 * HIDDEN + (head + 1) * HEAD_DIM],
            );
        }
        for (token, row) in qkv.chunks_exact(3 * HIDDEN).enumerate() {
            probabilities.fill(0.0);
            let query = &row[head * HEAD_DIM..(head + 1) * HEAD_DIM];
            for (&q, key_row) in query.iter().zip(keys.chunks_exact(tokens)) {
                for (score, &k) in probabilities.iter_mut().zip(key_row) {
                    *score = q.mul_add(k, *score);
                }
            }
            for score in &mut probabilities {
                *score *= scale;
            }
            let max = probabilities
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            // Preserve the reference's 16-lane reduction order without SIMD
            // intrinsics. Serial summation can change the next quantized bin.
            let mut sums = [0.0_f32; 16];
            for (i, score) in probabilities.iter_mut().enumerate() {
                *score = mlas_exp(*score - max);
                sums[i % 16] += *score;
            }
            for i in 0..8 {
                sums[i] += sums[i + 8];
            }
            let sum = ((sums[0] + sums[1]) + (sums[2] + sums[3]))
                + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
            let inverse = 1.0 / sum;
            let out = &mut context
                [token * HIDDEN + head * HEAD_DIM..token * HIDDEN + (head + 1) * HEAD_DIM];
            for (&probability, value_row) in probabilities.iter().zip(values.chunks_exact(HEAD_DIM))
            {
                let probability = probability * inverse;
                for (value, &v) in out.iter_mut().zip(value_row) {
                    *value = probability.mul_add(v, *value);
                }
            }
        }
    }
    context
}
// Coefficients and operation order adapted from ONNX Runtime v1.23.2 MLAS
// erf.cpp and its FMA kernel (Copyright Microsoft Corporation, MIT license).
// An accurate but differently rounded erf can cross an INT8 bin boundary:
// x=-2.742307 produces different bins after GELU and SmoothQuant with libm erf.
// Keep fused operations explicit so this arithmetic is portable safe Rust.
fn mlas_erf(x: f32) -> f32 {
    let a = x.abs().min(3.925);
    let r = if a <= 0.921875 {
        let z = a * a;
        let mut r = -0.00059910497_f32;
        for p in [
            0.004993396,
            -0.026766734,
            0.112818025,
            -0.37612486,
            0.12837915,
        ] {
            r = r.mul_add(z, p);
        }
        r.mul_add(a, a)
    } else {
        let mut r = 1.7294893e-05_f32;
        for p in [
            -0.00038320868,
            0.0038839343,
            -0.02425456,
            0.10677785,
            0.6348467,
            0.12871751,
        ] {
            r = r.mul_add(a, p);
        }
        let e = -r.mul_add(a, a);
        let n = std::f32::consts::LOG2_E.mul_add(e, 12582912.0) - 12582912.0;
        let fx = n.mul_add(-0.69314575, e);
        let fx = n.mul_add(-1.4286068e-06, fx);
        let mut y = 0.0013831981_f32;
        for p in [0.0083755, 0.04166895, 0.16666447, 0.49999985, 1.0, 1.0] {
            y = y.mul_add(fx, p);
        }
        1.0 - y * f32::from_bits(((n as i32 + 127) as u32) << 23)
    };
    r.copysign(x)
}

// Adapted from ONNX Runtime v1.23.2 MLAS compute.cpp (Microsoft, MIT).
// Range reduction and the polynomial retain the reference float32 rounding.
fn mlas_exp(x: f32) -> f32 {
    let x = x.max(-103.972084);
    let n = std::f32::consts::LOG2_E.mul_add(x, 12582912.0) - 12582912.0;
    let x = n.mul_add(-0.69314575, x);
    let x = n.mul_add(-1.4286068e-06, x);
    let mut p = f32::from_bits(0x3ab4a000);
    p = p.mul_add(x, f32::from_bits(0x3c092f6e));
    p = p.mul_add(x, f32::from_bits(0x3d2aadad));
    p = p.mul_add(x, f32::from_bits(0x3e2aaa28));
    p = p.mul_add(x, f32::from_bits(0x3efffffb));
    p = p.mul_add(x, f32::from_bits(0x3f800000));
    p = p.mul_add(x, f32::from_bits(0x3f800000));
    libm::scalbnf(p, n as i32)
}
