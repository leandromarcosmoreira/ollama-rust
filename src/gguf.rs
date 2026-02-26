use anyhow::{bail, Result};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747;

pub trait GgufMetadata {
    fn string(&self, key: &str) -> String;
    fn uint(&self, key: &str) -> u64;
    fn int(&self, key: &str) -> i64;
    fn float(&self, key: &str) -> f64;
    fn strings(&self, key: &str) -> Vec<String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    #[default]
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
    Unknown(u32),
}

impl From<u32> for GgmlType {
    fn from(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            16 => GgmlType::I8,
            17 => GgmlType::I16,
            18 => GgmlType::I32,
            19 => GgmlType::I64,
            20 => GgmlType::F64,
            21 => GgmlType::BF16,
            _ => GgmlType::Unknown(v),
        }
    }
}

impl GgmlType {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => 1,
            GgmlType::Q4_1 => 1,
            GgmlType::Q5_0 => 1,
            GgmlType::Q5_1 => 1,
            GgmlType::Q8_0 => 1,
            GgmlType::Q8_1 => 1,
            GgmlType::Q2_K => 1,
            GgmlType::Q3_K => 1,
            GgmlType::Q4_K => 1,
            GgmlType::Q5_K => 1,
            GgmlType::Q6_K => 1,
            GgmlType::Q8_K => 1,
            GgmlType::I8 => 1,
            GgmlType::I16 => 2,
            GgmlType::I32 => 4,
            GgmlType::I64 => 8,
            GgmlType::F64 => 8,
            GgmlType::BF16 => 2,
            GgmlType::Unknown(_) => 2,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GgufMetadataImpl {
    pub arch: Option<String>,
    pub name: Option<String>,
    pub context_length: u64,
    pub embedding_length: u64,
    pub block_count: u64,
    pub feed_forward_length: u64,
    pub head_count: u64,
    pub head_count_kv: u64,
    pub layer_norm_rms_epsilon: f32,
    pub rope_dimension_count: u64,
    pub rope_freq_base: f32,
    pub file_type: GgmlType,
    pub vocab_size: u64,
    pub eos_token_id: Option<u64>,
    pub bos_token_id: Option<u64>,
    pub vocab_tokens: Option<Vec<String>>,
    pub vocab_scores: Option<Vec<f32>>,
}

impl GgufMetadata for GgufMetadataImpl {
    fn string(&self, key: &str) -> String {
        match key {
            "general.architecture" => self.arch.clone().unwrap_or_default(),
            "general.name" => self.name.clone().unwrap_or_default(),
            _ => String::new(),
        }
    }
    
    fn uint(&self, key: &str) -> u64 {
        match key {
            "llama.context_length" | "qwen.context_length" | "llama3.context_length" => self.context_length,
            "llama.embedding_length" | "qwen.embedding_length" => self.embedding_length,
            "llama.block_count" | "qwen.block_count" => self.block_count,
            "llama.feed_forward_length" => self.feed_forward_length,
            "llama.attention.head_count" | "qwen.attention.head_count" => self.head_count,
            "llama.attention.head_count_kv" | "qwen.attention.head_count_kv" => self.head_count_kv,
            "llama.rope.dimension_count" => self.rope_dimension_count,
            "llama.vocab_size" | "qwen.vocab_size" => self.vocab_size,
            _ => 0,
        }
    }
    
    fn int(&self, key: &str) -> i64 {
        self.uint(key) as i64
    }
    
    fn float(&self, key: &str) -> f64 {
        match key {
            "llama.attention.layer_norm_rms_epsilon" | "qwen.attention.layer_norm_rms_epsilon" => self.layer_norm_rms_epsilon as f64,
            "llama.rope.freq_base" | "qwen.rope.freq_base" => self.rope_freq_base as f64,
            _ => 0.0,
        }
    }
    
    fn strings(&self, _key: &str) -> Vec<String> {
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
    pub size: u64,
}

impl TensorInfo {
    pub fn num_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: GgufMetadataImpl,
    pub tensors: Vec<TensorInfo>,
    pub architecture: String,
    pub model_size: u64,
    pub data_offset: u64,
    pub vocab: Option<Vec<String>>,
    pub vocab_scores: Option<Vec<f32>>,
}

impl GgufFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read(&mut reader)
    }

    fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            bail!("Invalid GGUF magic: expected {:08x}, got {:08x}", GGUF_MAGIC, magic);
        }

        let version = read_u32(reader)?;
        let tensor_count = read_u64(reader)?;
        let metadata_kv_count = read_u64(reader)?;

        let mut metadata = GgufMetadataImpl::default();
        let mut architecture = String::new();

        for _ in 0..metadata_kv_count {
            let key = read_string(reader)?;
            let value = read_value(reader)?;

            match key.as_str() {
                "general.architecture" => {
                    architecture = value.as_string().unwrap_or_default();
                    metadata.arch = Some(architecture.clone());
                }
                "general.name" => {
                    metadata.name = value.as_string();
                }
                "llama.context_length" | "qwen.context_length" | "llama3.context_length" => {
                    metadata.context_length = value.as_u64().unwrap_or(2048);
                }
                "llama.embedding_length" | "qwen.embedding_length" => {
                    metadata.embedding_length = value.as_u64().unwrap_or(4096);
                }
                "llama.block_count" | "qwen.block_count" => {
                    metadata.block_count = value.as_u64().unwrap_or(32);
                }
                "llama.feed_forward_length" => {
                    metadata.feed_forward_length = value.as_u64().unwrap_or(11008);
                }
                "llama.attention.head_count" | "qwen.attention.head_count" => {
                    metadata.head_count = value.as_u64().unwrap_or(32);
                }
                "llama.attention.head_count_kv" | "qwen.attention.head_count_kv" => {
                    metadata.head_count_kv = value.as_u64().unwrap_or(32);
                }
                "llama.attention.layer_norm_rms_epsilon" => {
                    metadata.layer_norm_rms_epsilon = value.as_f32().unwrap_or(1e-5);
                }
                "llama.rope.dimension_count" => {
                    metadata.rope_dimension_count = value.as_u64().unwrap_or(128);
                }
                "llama.rope.freq_base" => {
                    metadata.rope_freq_base = value.as_f32().unwrap_or(10000.0);
                }
                "general.file_type" => {
                    metadata.file_type = GgmlType::from(value.as_u64().unwrap_or(1) as u32);
                }
                "llama.vocab_size" | "qwen.vocab_size" => {
                    metadata.vocab_size = value.as_u64().unwrap_or(32000);
                }
                "llama.ggml.eos_token_id" | "tokenizer.ggml.eos_token_id" => {
                    metadata.eos_token_id = value.as_u64();
                }
                "llama.ggml.bos_token_id" | "tokenizer.ggml.bos_token_id" => {
                    metadata.bos_token_id = value.as_u64();
                }
                "tokenizer.ggml.tokens" => {
                    if let Value::Array(arr) = value {
                        let tokens: Vec<String> = arr.iter().filter_map(|v| v.as_string()).collect();
                        if !tokens.is_empty() {
                            metadata.vocab_tokens = Some(tokens);
                            if metadata.vocab_size == 0 {
                                metadata.vocab_size = metadata.vocab_tokens.as_ref().map(|t| t.len() as u64).unwrap_or(0);
                            }
                        }
                    }
                }
                "tokenizer.ggml.scores" => {
                    if let Value::Array(arr) = value {
                        let scores: Vec<f32> = arr.iter().filter_map(|v| v.as_f32()).collect();
                        if !scores.is_empty() {
                            metadata.vocab_scores = Some(scores);
                        }
                    }
                }
                _ => {}
            }
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(reader)?;
            let n_dims = read_u32(reader)? as usize;
            
            if n_dims > 10 {
                bail!("Too many dimensions: {}", n_dims);
            }
            
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let dim = read_u64(reader)?;
                if dim > 100_000 {
                    bail!("Dimension too large: {}", dim);
                }
                dims.push(dim);
            }

            let ggml_type = GgmlType::from(read_u32(reader)?);
            let offset = read_u64(reader)?;

            let elements: u64 = dims.iter().product();
            let size = elements.saturating_mul(ggml_type.bytes_per_element() as u64);

            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
                size,
            });
        }

        let data_offset = reader.stream_position()?;
        let model_size = reader.seek(SeekFrom::End(0))?;

        Ok(Self {
            version,
            tensor_count,
            metadata_kv_count,
            metadata,
            tensors,
            architecture,
            model_size,
            data_offset,
            vocab: None,
            vocab_scores: None,
        })
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn estimate_memory_usage(&self, gpu_layers: i32) -> u64 {
        if gpu_layers <= 0 {
            return 0;
        }

        let total_layers = self.metadata.block_count as i32;
        let layers_on_gpu = if gpu_layers >= total_layers { total_layers } else { gpu_layers };
        
        let mut gpu_tensor_size = 0u64;

        for tensor in &self.tensors {
            if tensor.name.contains("blk.") {
                if let Some(layer_str) = tensor.name.split("blk.").nth(1) {
                    if let Some(layer_num) = layer_str.split('.').next() {
                        if let Ok(layer) = layer_num.parse::<i32>() {
                            if layer < layers_on_gpu {
                                gpu_tensor_size += tensor.size;
                            }
                        }
                    }
                }
            } else {
                gpu_tensor_size += tensor.size;
            }
        }

        let kv_cache_size = self.metadata.context_length
            * self.metadata.head_count_kv
            * self.metadata.embedding_length
            * 2;

        gpu_tensor_size + kv_cache_size
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Value {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    fn as_string(&self) -> Option<String> {
        match self {
            Value::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Uint8(v) => Some(*v as u64),
            Value::Uint16(v) => Some(*v as u64),
            Value::Uint32(v) => Some(*v as u64),
            Value::Uint64(v) => Some(*v),
            Value::Int8(v) => Some(*v as u64),
            Value::Int16(v) => Some(*v as u64),
            Value::Int32(v) => Some(*v as u64),
            Value::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match self {
            Value::Float32(v) => Some(*v),
            Value::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    #[allow(dead_code)]
    fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float32(v) => Some(*v as f64),
            Value::Float64(v) => Some(*v),
            _ => None,
        }
    }
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i8<R: Read>(reader: &mut R) -> Result<i8> {
    Ok(read_u8(reader)? as i8)
}

fn read_i16<R: Read>(reader: &mut R) -> Result<i16> {
    Ok(read_u16(reader)? as i16)
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32> {
    Ok(read_u32(reader)? as i32)
}

fn read_i64<R: Read>(reader: &mut R) -> Result<i64> {
    Ok(read_u64(reader)? as i64)
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = read_u64(reader)? as usize;
    if len > 10_000_000 {
        bail!("String too large: {} bytes", len);
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_value_with_type<R: Read>(reader: &mut R, value_type: u32) -> Result<Value> {
    match value_type {
        0 => Ok(Value::Uint8(read_u8(reader)?)),
        1 => Ok(Value::Int8(read_i8(reader)?)),
        2 => Ok(Value::Uint16(read_u16(reader)?)),
        3 => Ok(Value::Int16(read_i16(reader)?)),
        4 => Ok(Value::Uint32(read_u32(reader)?)),
        5 => Ok(Value::Int32(read_i32(reader)?)),
        6 => Ok(Value::Float32(read_f32(reader)?)),
        7 => Ok(Value::Bool(read_u8(reader)? != 0)),
        8 => Ok(Value::String(read_string(reader)?)),
        9 => {
            let element_type = read_u32(reader)?;
            let len = read_u64(reader)? as usize;
            if len > 100000 {
                bail!("Array too large: {} elements", len);
            }
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value_with_type(reader, element_type)?);
            }
            Ok(Value::Array(arr))
        }
        10 => Ok(Value::Uint64(read_u64(reader)?)),
        11 => Ok(Value::Int64(read_i64(reader)?)),
        12 => Ok(Value::Float64(read_f64(reader)?)),
        _ => bail!("Unknown value type in array: {}", value_type),
    }
}

fn read_value<R: Read>(reader: &mut R) -> Result<Value> {
    let value_type = read_u32(reader)?;
    
    match value_type {
        0 => Ok(Value::Uint8(read_u8(reader)?)),
        1 => Ok(Value::Int8(read_i8(reader)?)),
        2 => Ok(Value::Uint16(read_u16(reader)?)),
        3 => Ok(Value::Int16(read_i16(reader)?)),
        4 => Ok(Value::Uint32(read_u32(reader)?)),
        5 => Ok(Value::Int32(read_i32(reader)?)),
        6 => Ok(Value::Float32(read_f32(reader)?)),
        7 => Ok(Value::Bool(read_u8(reader)? != 0)),
        8 => Ok(Value::String(read_string(reader)?)),
        9 => {
            let element_type = read_u32(reader)?;
            let len = read_u64(reader)? as usize;
            if len > 100000 {
                bail!("Array too large: {} elements", len);
            }
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value_with_type(reader, element_type)?);
            }
            Ok(Value::Array(arr))
        }
        10 => Ok(Value::Uint64(read_u64(reader)?)),
        11 => Ok(Value::Int64(read_i64(reader)?)),
        12 => Ok(Value::Float64(read_f64(reader)?)),
        _ => bail!("Unknown value type: {}", value_type),
    }
}
