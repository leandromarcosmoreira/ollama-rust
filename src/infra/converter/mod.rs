pub mod safetensors;
pub mod torch;

use crate::core::model::ModelConfig;
use crate::infra::Result;
use std::path::Path;

pub struct ModelConverter {
    config: ModelConfig,
}

impl ModelConverter {
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }
    
    pub fn convert_safetensors<P: AsRef<Path>>(&self, input: P, output: P) -> Result<()> {
        safetensors::convert(input, output, &self.config)
    }
    
    pub fn convert_pytorch<P: AsRef<Path>>(&self, input: P, output: P) -> Result<()> {
        torch::convert(input, output, &self.config)
    }
}

pub fn convert_safetensors<P: AsRef<Path>>(input: P, output: P, config: &ModelConfig) -> Result<()> {
    safetensors::convert(input, output, config)
}

pub fn convert_pytorch<P: AsRef<Path>>(input: P, output: P, config: &ModelConfig) -> Result<()> {
    torch::convert(input, output, config)
}
