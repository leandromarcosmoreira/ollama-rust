use super::traits::{ModelConfig, RopeScaling, RopeScalingType, ConfigValue};

pub struct ModelConfigBuilder {
    config: ModelConfig,
}

impl ModelConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
        }
    }
    
    pub fn architecture(mut self, arch: impl Into<String>) -> Self {
        self.config.architecture = arch.into();
        self
    }
    
    pub fn hidden_size(mut self, size: usize) -> Self {
        self.config.hidden_size = size;
        self
    }
    
    pub fn intermediate_size(mut self, size: usize) -> Self {
        self.config.intermediate_size = size;
        self
    }
    
    pub fn num_layers(mut self, n: usize) -> Self {
        self.config.num_layers = n;
        self
    }
    
    pub fn num_heads(mut self, n: usize) -> Self {
        self.config.num_heads = n;
        self
    }
    
    pub fn num_kv_heads(mut self, n: usize) -> Self {
        self.config.num_kv_heads = n;
        self
    }
    
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }
    
    pub fn context_length(mut self, len: usize) -> Self {
        self.config.context_length = len;
        self
    }
    
    pub fn rope_theta(mut self, theta: f32) -> Self {
        self.config.rope_theta = theta;
        self
    }
    
    pub fn rope_linear_scaling(mut self, factor: f32) -> Self {
        self.config.rope_scaling = Some(RopeScaling {
            scaling_type: RopeScalingType::Linear,
            factor,
            original_context_length: self.config.context_length,
        });
        self
    }
    
    pub fn rope_yarn_scaling(mut self, factor: f32, original_len: usize) -> Self {
        self.config.rope_scaling = Some(RopeScaling {
            scaling_type: RopeScalingType::Yarn,
            factor,
            original_context_length: original_len,
        });
        self
    }
    
    pub fn norm_eps(mut self, eps: f32) -> Self {
        self.config.norm_eps = eps;
        self
    }
    
    pub fn custom<K, V>(mut self, key: K, value: V) -> Self 
    where
        K: Into<String>,
        V: IntoConfigValue,
    {
        self.config.custom.insert(key.into(), value.into_config_value());
        self
    }
    
    pub fn build(self) -> ModelConfig {
        self.config
    }
}

impl Default for ModelConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelConfig {
    pub fn builder() -> ModelConfigBuilder {
        ModelConfigBuilder::new()
    }
}

pub trait IntoConfigValue {
    fn into_config_value(self) -> ConfigValue;
}

impl IntoConfigValue for i64 {
    fn into_config_value(self) -> ConfigValue {
        ConfigValue::Int(self)
    }
}

impl IntoConfigValue for u64 {
    fn into_config_value(self) -> ConfigValue {
        ConfigValue::Uint(self)
    }
}

impl IntoConfigValue for f64 {
    fn into_config_value(self) -> ConfigValue {
        ConfigValue::Float(self)
    }
}

impl IntoConfigValue for String {
    fn into_config_value(self) -> ConfigValue {
        ConfigValue::String(self)
    }
}

impl IntoConfigValue for bool {
    fn into_config_value(self) -> ConfigValue {
        ConfigValue::Bool(self)
    }
}

impl IntoConfigValue for ConfigValue {
    fn into_config_value(self) -> ConfigValue {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_builder() {
        let config = ModelConfig::builder()
            .architecture("llama")
            .hidden_size(4096)
            .num_layers(32)
            .context_length(8192)
            .rope_linear_scaling(4.0)
            .custom("sliding_window", 4096u64)
            .build();
        
        assert_eq!(config.architecture, "llama");
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.context_length, 8192);
        assert!(config.rope_scaling.is_some());
        assert_eq!(config.get::<u64>("sliding_window"), Some(4096));
    }
}
