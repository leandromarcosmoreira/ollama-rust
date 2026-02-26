use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub mod safetensors;
pub mod torch;
pub mod tensor;
pub mod tokenizer;

pub use safetensors::SafeTensors;
pub use tensor::{Tensor, TensorData};
pub use tokenizer::TokenizerConverter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
}

#[derive(Debug, Clone)]
pub struct ConversionOptions {
    pub output_path: String,
    pub output_type: String,
    pub quantization: Option<String>,
    pub ctx_len: usize,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            output_path: "model.gguf".to_string(),
            output_type: "f16".to_string(),
            quantization: None,
            ctx_len: 4096,
        }
    }
}

pub struct Converter {
    config: ModelConfig,
    tensors: HashMap<String, Tensor>,
}

impl Converter {
    pub fn new() -> Self {
        Self {
            config: ModelConfig {
                architecture: "llama".to_string(),
                hidden_size: 4096,
                intermediate_size: 11008,
                num_attention_heads: 32,
                num_hidden_layers: 32,
                num_key_value_heads: 32,
                vocab_size: 32000,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                max_position_embeddings: 2048,
            },
            tensors: HashMap::new(),
        }
    }

    pub fn with_config(config: ModelConfig) -> Self {
        Self {
            config,
            tensors: HashMap::new(),
        }
    }

    pub fn load_safetensors<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let st = SafeTensors::load(&path)?;
        
        for (name, info) in &st.header.tensors {
            let data = st.get_tensor_data(name)?;
            let tensor = Tensor::new(
                info.dtype.clone(),
                info.shape.clone(),
                data,
            );
            self.tensors.insert(name.clone(), tensor);
        }
        
        Ok(())
    }

    pub fn load_pytorch<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let reader = BufReader::new(File::open(&path)?);
        let torch_model = torch::load(reader)?;
        
        for (name, tensor) in torch_model.tensors {
            self.tensors.insert(name, tensor);
        }
        
        if let Some(config) = torch_model.config {
            self.config = config;
        }
        
        Ok(())
    }

    pub fn convert_to_gguf<P: AsRef<Path>>(&self, output_path: P) -> Result<()> {
        let mut file = File::create(&output_path)?;
        
        let magic = b"GGUF";
        file.write_all(magic)?;
        
        let version: u32 = 3;
        file.write_all(&version.to_le_bytes())?;
        
        let tensor_count = self.tensors.len() as u64;
        file.write_all(&tensor_count.to_le_bytes())?;
        
        let metadata_kv_count = 10u64;
        file.write_all(&metadata_kv_count.to_le_bytes())?;
        
        self.write_metadata_string(&mut file, "general.architecture", &self.config.architecture)?;
        self.write_metadata_u64(&mut file, "general.parameter_count", self.calculate_param_count())?;
        self.write_metadata_u64(&mut file, "llama.context_length", self.config.max_position_embeddings as u64)?;
        self.write_metadata_u64(&mut file, "llama.embedding_length", self.config.hidden_size as u64)?;
        self.write_metadata_u64(&mut file, "llama.block_count", self.config.num_hidden_layers as u64)?;
        self.write_metadata_u64(&mut file, "llama.attention.head_count", self.config.num_attention_heads as u64)?;
        self.write_metadata_u64(&mut file, "llama.attention.head_count_kv", self.config.num_key_value_heads as u64)?;
        self.write_metadata_f64(&mut file, "llama.attention.layer_norm_rms_epsilon", self.config.rms_norm_eps)?;
        self.write_metadata_u64(&mut file, "llama.vocab_size", self.config.vocab_size as u64)?;
        self.write_metadata_f64(&mut file, "llama.rope.freq_base", self.config.rope_theta)?;
        
        let mut offset: u64 = 0;
        for (name, tensor) in &self.tensors {
            self.write_tensor_info(&mut file, name, tensor, offset)?;
            offset += tensor.data.len() as u64;
        }
        
        for tensor in self.tensors.values() {
            file.write_all(&tensor.data)?;
        }
        
        Ok(())
    }

    fn write_metadata_string(&self, file: &mut File, key: &str, value: &str) -> Result<()> {
        self.write_string(file, key)?;
        file.write_all(&8u32.to_le_bytes())?;
        self.write_string(file, value)?;
        Ok(())
    }

    fn write_metadata_u64(&self, file: &mut File, key: &str, value: u64) -> Result<()> {
        self.write_string(file, key)?;
        file.write_all(&4u32.to_le_bytes())?;
        file.write_all(&value.to_le_bytes())?;
        Ok(())
    }

    fn write_metadata_f64(&self, file: &mut File, key: &str, value: f64) -> Result<()> {
        self.write_string(file, key)?;
        file.write_all(&7u32.to_le_bytes())?;
        file.write_all(&value.to_le_bytes())?;
        Ok(())
    }

    fn write_string(&self, file: &mut File, s: &str) -> Result<()> {
        let bytes = s.as_bytes();
        file.write_all(&(bytes.len() as u64).to_le_bytes())?;
        file.write_all(bytes)?;
        Ok(())
    }

    fn write_tensor_info(&self, file: &mut File, name: &str, tensor: &Tensor, offset: u64) -> Result<()> {
        self.write_string(file, name)?;
        
        let n_dims = tensor.shape.len() as u32;
        file.write_all(&n_dims.to_le_bytes())?;
        
        for dim in &tensor.shape {
            file.write_all(&(*dim as u64).to_le_bytes())?;
        }
        
        let dtype = match tensor.dtype.as_str() {
            "F32" => 0u32,
            "F16" => 1u32,
            "Q4_0" => 2u32,
            "Q4_1" => 3u32,
            "Q8_0" => 7u32,
            _ => 0u32,
        };
        file.write_all(&dtype.to_le_bytes())?;
        
        file.write_all(&offset.to_le_bytes())?;
        
        Ok(())
    }

    fn calculate_param_count(&self) -> u64 {
        let embed = self.config.hidden_size * self.config.vocab_size;
        let attn = self.config.hidden_size * self.config.hidden_size * 4 * self.config.num_hidden_layers;
        let ffn = self.config.hidden_size * self.config.intermediate_size * 3 * self.config.num_hidden_layers;
        (embed + attn + ffn) as u64
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }
}

impl Default for Converter {
    fn default() -> Self {
        Self::new()
    }
}
