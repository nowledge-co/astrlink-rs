//! Decode the initializers used by the pinned AstrLink network.

use std::collections::{HashMap, HashSet};

use prost::Message;

use crate::error::Error;

/// An ONNX initializer with its data normalized to little-endian bytes.
pub struct Tensor {
    /// Dimensions in row-major order; an empty shape denotes a scalar.
    pub dims: Vec<usize>,
    /// ONNX `TensorProto.DataType` value.
    pub dtype: i32,
    /// Contiguous little-endian tensor values.
    pub raw: Vec<u8>,
}

impl Tensor {
    /// Decode a FLOAT tensor, rejecting a different type or invalid byte count.
    pub fn floats(&self) -> Result<Vec<f32>, Error> {
        if self.dtype != 1 {
            return Err(Error::Model(format!(
                "expected FLOAT tensor, found element type {}",
                self.dtype
            )));
        }
        let expected = byte_len(&self.dims, self.dtype)?;
        if self.raw.len() != expected {
            return Err(Error::Model(format!(
                "FLOAT tensor has {} bytes, expected {expected}",
                self.raw.len()
            )));
        }
        Ok(self
            .raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }
}

/// Decode verified model bytes using only the supplied external-data buffer.
///
/// This extracts initializers, not an executable graph. The caller must verify
/// the model and external buffer against the fixed artifact pins first.
pub fn parse(model: &[u8], external: &[u8]) -> Result<HashMap<String, Tensor>, Error> {
    let model = ModelProto::decode(model)
        .map_err(|e| Error::Model(format!("model_int8.onnx protobuf: {e}")))?;
    check_metadata(&model.metadata_props)?;
    let graph = model
        .graph
        .ok_or_else(|| Error::Model("model_int8.onnx has no graph".into()))?;
    check_metadata(&graph.metadata_props)?;
    if !graph.sparse_initializer.is_empty() {
        return Err(Error::Model("sparse initializers are unsupported".into()));
    }
    let mut tensors = HashMap::with_capacity(graph.initializer.len());
    for initializer in graph.initializer {
        let name = initializer.name.clone();
        if name.is_empty() {
            return Err(Error::Model("initializer has no name".into()));
        }
        if tensors.contains_key(&name) {
            return Err(Error::Model(format!("duplicate initializer {name:?}")));
        }
        let tensor = decode_tensor(initializer, external).map_err(|e| match e {
            Error::Model(reason) => Error::Model(format!("initializer {name:?}: {reason}")),
            other => other,
        })?;
        tensors.insert(name, tensor);
    }
    Ok(tensors)
}

fn decode_tensor(t: TensorProto, external: &[u8]) -> Result<Tensor, Error> {
    check_metadata(&t.metadata_props)?;
    check_metadata(&t.external_data)?;
    if t.segment.is_some() {
        return Err(Error::Model("segmented tensors are unsupported".into()));
    }
    let dims = t
        .dims
        .iter()
        .map(|&d| {
            usize::try_from(d)
                .map_err(|_| Error::Model(format!("dimension {d} is outside the supported range")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected = byte_len(&dims, t.data_type)?;
    let typed_fields = [
        !t.float_data.is_empty(),
        !t.int32_data.is_empty(),
        !t.int64_data.is_empty(),
        !t.string_data.is_empty(),
        !t.double_data.is_empty(),
        !t.uint64_data.is_empty(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count();
    let raw = match t.data_location {
        1 => {
            if t.raw_data.is_some() || typed_fields != 0 {
                return Err(Error::Model("external tensor also has inline data".into()));
            }
            external_bytes(&t.external_data, external, expected)?.to_vec()
        }
        0 => {
            if !t.external_data.is_empty() {
                return Err(Error::Model(
                    "inline tensor has external-data metadata".into(),
                ));
            }
            if typed_fields > 1 || (t.raw_data.is_some() && typed_fields != 0) {
                return Err(Error::Model("tensor has multiple data encodings".into()));
            }
            match t.raw_data {
                Some(raw) => raw,
                None => typed_bytes(&t, expected)?,
            }
        }
        other => {
            return Err(Error::Model(format!("unsupported data location {other}")));
        }
    };
    if raw.len() != expected {
        return Err(Error::Model(format!(
            "tensor has {} bytes, expected {expected}",
            raw.len()
        )));
    }
    Ok(Tensor {
        dims,
        dtype: t.data_type,
        raw,
    })
}

fn byte_len(dims: &[usize], dtype: i32) -> Result<usize, Error> {
    let width: usize = match dtype {
        1 | 6 => 4,
        2 | 3 => 1,
        7 => 8,
        _ => return Err(Error::Model(format!("unsupported element type {dtype}"))),
    };
    dims.iter()
        .try_fold(width, |size, &d| size.checked_mul(d))
        .filter(|&size| size <= isize::MAX as usize)
        .ok_or_else(|| Error::Model("tensor byte count exceeds the supported range".into()))
}

fn typed_bytes(t: &TensorProto, expected: usize) -> Result<Vec<u8>, Error> {
    let (count, width) = match t.data_type {
        1 => (t.float_data.len(), 4usize),
        2 | 3 => (t.int32_data.len(), 1),
        6 => (t.int32_data.len(), 4),
        7 => (t.int64_data.len(), 8),
        other => return Err(Error::Model(format!("unsupported element type {other}"))),
    };
    let incompatible = match t.data_type {
        1 => !t.int32_data.is_empty() || !t.int64_data.is_empty(),
        2 | 3 | 6 => !t.float_data.is_empty() || !t.int64_data.is_empty(),
        7 => !t.float_data.is_empty() || !t.int32_data.is_empty(),
        _ => false,
    };
    if incompatible
        || !t.string_data.is_empty()
        || !t.double_data.is_empty()
        || !t.uint64_data.is_empty()
    {
        return Err(Error::Model(
            "typed data does not match the element type".into(),
        ));
    }
    if count.checked_mul(width) != Some(expected) {
        return Err(Error::Model(format!(
            "typed tensor has {count} values, expected {}",
            expected / width
        )));
    }
    let mut raw = Vec::with_capacity(expected);
    match t.data_type {
        1 => {
            for value in &t.float_data {
                raw.extend_from_slice(&value.to_le_bytes());
            }
        }
        2 => {
            for &value in &t.int32_data {
                raw.push(u8::try_from(value).map_err(|_| {
                    Error::Model("UINT8 value is outside the range 0..=255".into())
                })?);
            }
        }
        3 => {
            for &value in &t.int32_data {
                let value = i8::try_from(value).map_err(|_| {
                    Error::Model("INT8 value is outside the range -128..=127".into())
                })?;
                raw.extend_from_slice(&value.to_le_bytes());
            }
        }
        6 => {
            for value in &t.int32_data {
                raw.extend_from_slice(&value.to_le_bytes());
            }
        }
        7 => {
            for value in &t.int64_data {
                raw.extend_from_slice(&value.to_le_bytes());
            }
        }
        _ => unreachable!("element type was checked above"),
    }
    Ok(raw)
}

fn external_bytes<'a>(
    entries: &[StringEntry],
    external: &'a [u8],
    expected: usize,
) -> Result<&'a [u8], Error> {
    let mut location = None;
    let mut offset = 0;
    let mut length = expected;
    for entry in entries {
        match entry.key.as_str() {
            "location" => location = Some(entry.value.as_str()),
            "offset" => offset = decimal(&entry.value, "offset")?,
            "length" => length = decimal(&entry.value, "length")?,
            key => {
                return Err(Error::Model(format!(
                    "unsupported external-data key {key:?}"
                )));
            }
        }
    }
    if location != Some("model_int8.onnx_data") {
        return Err(Error::Model(
            "external location must be model_int8.onnx_data".into(),
        ));
    }
    if length != expected {
        return Err(Error::Model(format!(
            "external length is {length} bytes, expected {expected}"
        )));
    }
    let end = offset
        .checked_add(length)
        .ok_or_else(|| Error::Model("external offset plus length overflows".into()))?;
    external
        .get(offset..end)
        .ok_or_else(|| Error::Model("external tensor exceeds model_int8.onnx_data".into()))
}

fn decimal(value: &str, field: &str) -> Result<usize, Error> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(Error::Model(format!(
            "external {field} must be a nonnegative decimal integer"
        )));
    }
    value.parse().map_err(|_| {
        Error::Model(format!(
            "external {field} exceeds the supported integer range"
        ))
    })
}

fn check_metadata(entries: &[StringEntry]) -> Result<(), Error> {
    let mut keys = HashSet::with_capacity(entries.len());
    for entry in entries {
        if !keys.insert(entry.key.as_str()) {
            return Err(Error::Model(format!(
                "duplicate metadata key {:?}",
                entry.key
            )));
        }
    }
    Ok(())
}

// Partial ONNX schema: prost skips graph operations and other unused fields.
#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "14")]
    metadata_props: Vec<StringEntry>,
}

#[derive(Clone, PartialEq, Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
    #[prost(bytes = "vec", repeated, tag = "15")]
    sparse_initializer: Vec<Vec<u8>>,
    #[prost(message, repeated, tag = "16")]
    metadata_props: Vec<StringEntry>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    data_type: i32,
    #[prost(bytes = "vec", optional, tag = "3")]
    segment: Option<Vec<u8>>,
    #[prost(float, repeated, tag = "4")]
    float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    int32_data: Vec<i32>,
    #[prost(bytes = "vec", repeated, tag = "6")]
    string_data: Vec<Vec<u8>>,
    #[prost(int64, repeated, tag = "7")]
    int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(bytes = "vec", optional, tag = "9")]
    raw_data: Option<Vec<u8>>,
    #[prost(double, repeated, tag = "10")]
    double_data: Vec<f64>,
    #[prost(uint64, repeated, tag = "11")]
    uint64_data: Vec<u64>,
    #[prost(message, repeated, tag = "13")]
    external_data: Vec<StringEntry>,
    #[prost(int32, tag = "14")]
    data_location: i32,
    #[prost(message, repeated, tag = "16")]
    metadata_props: Vec<StringEntry>,
}

#[derive(Clone, PartialEq, Message)]
struct StringEntry {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}
