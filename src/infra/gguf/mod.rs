use crate::core::model::{ModelConfig, ConfigValue};
use crate::infra::Result;
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
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
            GgmlType::Q2K => 1,
            GgmlType::Q3K => 1,
            GgmlType::Q4K => 1,
            GgmlType::Q5K => 1,
            GgmlType::Q6K => 1,
            GgmlType::Q8K => 1,
            GgmlType::I8 => 1,
            GgmlType::I16 => 2,
            GgmlType::I32 => 4,
        }
    }
    
    pub fn block_size(&self) -> usize {
        match self {
            GgmlType::Q4_0 | GgmlType::Q4_1 => 32,
            GgmlType::Q5_0 | GgmlType::Q5_1 => 32,
            GgmlType::Q8_0 | GgmlType::Q8_1 => 32,
            GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | 
            GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8K => 256,
            _ => 1,
        }
    }
}

pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: GgufMetadata,
    pub tensors: Vec<TensorInfo>,
}

pub struct GgufMetadata {
    pub kv: HashMap<String, MetadataValue>,
}

impl GgufMetadata {
    pub fn new() -> Self {
        Self {
            kv: HashMap::new(),
        }
    }
    
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.kv.get(key)
    }
    
    pub fn string(&self, key: &str) -> String {
        self.kv.get(key)
            .and_then(|v| match v {
                MetadataValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }
    
    pub fn uint(&self, key: &str) -> u64 {
        self.kv.get(key)
            .and_then(|v| match v {
                MetadataValue::Uint(n) => Some(*n),
                MetadataValue::Int(n) => Some(*n as u64),
                _ => None,
            })
            .unwrap_or(0)
    }
    
    pub fn int(&self, key: &str) -> i64 {
        self.kv.get(key)
            .and_then(|v| match v {
                MetadataValue::Int(n) => Some(*n),
                MetadataValue::Uint(n) => Some(*n as i64),
                _ => None,
            })
            .unwrap_or(0)
    }
    
    pub fn float(&self, key: &str) -> f64 {
        self.kv.get(key)
            .and_then(|v| match v {
                MetadataValue::Float(n) => Some(*n),
                MetadataValue::Int(n) => Some(*n as f64),
                MetadataValue::Uint(n) => Some(*n as f64),
                _ => None,
            })
            .unwrap_or(0.0)
    }
    
    pub fn to_model_config(&self) -> ModelConfig {
        let arch = self.string("general.architecture");
        
        let mut config = ModelConfig::builder()
            .architecture(&arch)
            .hidden_size(self.uint(&format!("{}.embedding_length", arch)) as usize)
            .intermediate_size(self.uint(&format!("{}.feed_forward_length", arch)) as usize)
            .num_layers(self.uint(&format!("{}.block_count", arch)) as usize)
            .num_heads(self.uint(&format!("{}.attention.head_count", arch)) as usize)
            .num_kv_heads(self.uint(&format!("{}.attention.head_count_kv", arch)) as usize)
            .vocab_size(self.uint("tokenizer.ggml.model") as usize)
            .context_length(self.uint(&format!("{}.context_length", arch)) as usize)
            .rope_theta(self.float(&format!("{}.rope.freq_base", arch)) as f32)
            .norm_eps(self.float(&format!("{}.attention.layer_norm_rms_epsilon", arch)) as f32);
        
        for (key, value) in &self.kv {
            let config_value = match value {
                MetadataValue::Uint(n) => ConfigValue::Uint(*n),
                MetadataValue::Int(n) => ConfigValue::Int(*n),
                MetadataValue::Float(n) => ConfigValue::Float(*n),
                MetadataValue::String(s) => ConfigValue::String(s.clone()),
                MetadataValue::Bool(b) => ConfigValue::Bool(*b),
                MetadataValue::Array(arr) => ConfigValue::Array(
                    arr.iter().map(|v| match v {
                        MetadataValue::Uint(n) => ConfigValue::Uint(*n),
                        MetadataValue::Int(n) => ConfigValue::Int(*n),
                        MetadataValue::Float(n) => ConfigValue::Float(*n),
                        MetadataValue::String(s) => ConfigValue::String(s.clone()),
                        _ => ConfigValue::Uint(0),
                    }).collect()
                ),
            };
            config = config.custom(key, config_value);
        }
        
        config.build()
    }
}

impl Default for GgufMetadata {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint(u64),
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Array(Vec<MetadataValue>),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: GgmlType,
    pub offset: u64,
}

pub struct GgufParser;

impl GgufParser {
    const GGUF_MAGIC: u32 = 0x46554747;
    
    pub fn parse<P: AsRef<Path>>(path: P) -> Result<GgufFile> {
        let mut file = std::fs::File::open(path)?;
        Self::parse_reader(&mut file)
    }
    
    pub fn parse_reader<R: Read + Seek>(reader: &mut R) -> Result<GgufFile> {
        let magic = Self::read_u32(reader)?;
        if magic != Self::GGUF_MAGIC {
            anyhow::bail!("Invalid GGUF magic: expected 0x{:08X}, got 0x{:08X}", Self::GGUF_MAGIC, magic);
        }
        
        let version = Self::read_u32(reader)?;
        let tensor_count = Self::read_u64(reader)?;
        let metadata_kv_count = Self::read_u64(reader)?;
        
        let mut metadata = GgufMetadata::new();
        
        for _ in 0..metadata_kv_count {
            let key = Self::read_string(reader)?;
            let value = Self::read_metadata_value(reader)?;
            metadata.kv.insert(key, value);
        }
        
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        
        for _ in 0..tensor_count {
            let name = Self::read_string(reader)?;
            let n_dims = Self::read_u32(reader)? as usize;
            
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(Self::read_u64(reader)? as usize);
            }
            
            let dtype_id = Self::read_u32(reader)?;
            let dtype = Self::dtype_from_id(dtype_id)?;
            
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset: 0,
            });
        }
        
        Ok(GgufFile {
            version,
            tensor_count,
            metadata,
            tensors,
        })
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
    
    fn read_i64<R: Read>(reader: &mut R) -> Result<i64> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }
    
    fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }
    
    fn read_string<R: Read>(reader: &mut R) -> Result<String> {
        let len = Self::read_u64(reader)? as usize;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
    
    fn read_metadata_value<R: Read>(reader: &mut R) -> Result<MetadataValue> {
        let vtype = Self::read_u32(reader)?;
        
        match vtype {
            0 => Ok(MetadataValue::Uint(Self::read_u64(reader)?)),
            1 => Ok(MetadataValue::Int(Self::read_i64(reader)?)),
            2 => Ok(MetadataValue::Float(Self::read_f64(reader)?)),
            3 => Ok(MetadataValue::String(Self::read_string(reader)?)),
            4 => Ok(MetadataValue::Bool(false)),
            5 => Ok(MetadataValue::Bool(true)),
            6 => {
                let len = Self::read_u64(reader)? as usize;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(Self::read_metadata_value(reader)?);
                }
                Ok(MetadataValue::Array(arr))
            }
            _ => anyhow::bail!("Unknown metadata value type: {}", vtype),
        }
    }
    
    fn dtype_from_id(id: u32) -> Result<GgmlType> {
        match id {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            9 => Ok(GgmlType::Q8_1),
            10 => Ok(GgmlType::Q2K),
            11 => Ok(GgmlType::Q3K),
            12 => Ok(GgmlType::Q4K),
            13 => Ok(GgmlType::Q5K),
            14 => Ok(GgmlType::Q6K),
            15 => Ok(GgmlType::Q8K),
            16 => Ok(GgmlType::I8),
            17 => Ok(GgmlType::I16),
            18 => Ok(GgmlType::I32),
            _ => anyhow::bail!("Unknown GGML type: {}", id),
        }
    }
}
