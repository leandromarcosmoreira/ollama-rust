use crate::core::{Result, Tensor, KVCache, TokenId};
use std::collections::HashMap;

pub trait Model: Send + Sync {
    fn forward(
        &mut self,
        input: &[TokenId],
        positions: &[usize],
        cache: &mut dyn KVCache,
    ) -> Result<Tensor>;
    
    fn forward_batch(
        &mut self,
        batch: &ModelBatch,
        cache: &mut dyn KVCache,
    ) -> Result<Tensor>;
    
    fn config(&self) -> &ModelConfig;
    fn meta(&self) -> &ModelMeta;
    
    fn embed(&self, tokens: &[TokenId]) -> Result<Tensor>;
    fn logits(&self, hidden: &Tensor) -> Result<Tensor>;
}

pub trait ModelLayer: Send + Sync {
    fn forward(
        &mut self,
        hidden: &Tensor,
        positions: &[usize],
        cache: Option<&mut dyn KVCache>,
    ) -> Result<Tensor>;
    
    fn name(&self) -> &str;
    fn param_count(&self) -> usize;
}

pub trait Attention: Send + Sync {
    fn forward(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        positions: &[usize],
        cache: Option<&mut dyn KVCache>,
    ) -> Result<Tensor>;
    
    fn head_count(&self) -> usize;
    fn head_dim(&self) -> usize;
}

pub trait FeedForward: Send + Sync {
    fn forward(&mut self, hidden: &Tensor) -> Result<Tensor>;
    fn hidden_dim(&self) -> usize;
}

pub trait Normalization: Send + Sync {
    fn forward(&mut self, hidden: &Tensor) -> Result<Tensor>;
    fn epsilon(&self) -> f32;
}

pub struct ModelBatch {
    pub tokens: Vec<Vec<TokenId>>,
    pub positions: Vec<Vec<usize>>,
    pub attention_mask: Option<Tensor>,
}

impl ModelBatch {
    pub fn new(tokens: Vec<Vec<TokenId>>, positions: Vec<Vec<usize>>) -> Self {
        Self {
            tokens,
            positions,
            attention_mask: None,
        }
    }
    
    pub fn batch_size(&self) -> usize {
        self.tokens.len()
    }
    
    pub fn seq_len(&self) -> usize {
        self.tokens.first().map(|t| t.len()).unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub norm_eps: f32,
    pub custom: HashMap<String, ConfigValue>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: "llama".to_string(),
            hidden_size: 4096,
            intermediate_size: 11008,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            context_length: 2048,
            rope_theta: 10000.0,
            rope_scaling: None,
            norm_eps: 1e-5,
            custom: HashMap::new(),
        }
    }
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
    
    pub fn get<T: FromConfigValue>(&self, key: &str) -> Option<T> {
        self.custom.get(key).and_then(|v| T::from_config_value(v.clone()))
    }
}

#[derive(Debug, Clone)]
pub enum ConfigValue {
    Int(i64),
    Uint(u64),
    Float(f64),
    String(String),
    Bool(bool),
    Array(Vec<ConfigValue>),
}

pub trait FromConfigValue: Sized {
    fn from_config_value(value: ConfigValue) -> Option<Self>;
}

impl FromConfigValue for i64 {
    fn from_config_value(value: ConfigValue) -> Option<Self> {
        match value {
            ConfigValue::Int(v) => Some(v),
            ConfigValue::Uint(v) => Some(v as i64),
            _ => None,
        }
    }
}

impl FromConfigValue for u64 {
    fn from_config_value(value: ConfigValue) -> Option<Self> {
        match value {
            ConfigValue::Uint(v) => Some(v),
            ConfigValue::Int(v) => Some(v as u64),
            _ => None,
        }
    }
}

impl FromConfigValue for f64 {
    fn from_config_value(value: ConfigValue) -> Option<Self> {
        match value {
            ConfigValue::Float(v) => Some(v),
            ConfigValue::Int(v) => Some(v as f64),
            ConfigValue::Uint(v) => Some(v as f64),
            _ => None,
        }
    }
}

impl FromConfigValue for String {
    fn from_config_value(value: ConfigValue) -> Option<Self> {
        match value {
            ConfigValue::String(v) => Some(v),
            _ => None,
        }
    }
}

impl FromConfigValue for bool {
    fn from_config_value(value: ConfigValue) -> Option<Self> {
        match value {
            ConfigValue::Bool(v) => Some(v),
            ConfigValue::Int(v) => Some(v != 0),
            ConfigValue::Uint(v) => Some(v != 0),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RopeScaling {
    pub scaling_type: RopeScalingType,
    pub factor: f32,
    pub original_context_length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScalingType {
    Linear,
    Yarn,
    Dynamic,
}

use super::ModelMeta;
