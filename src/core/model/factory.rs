use super::traits::{Model, ModelConfig};
use crate::core::Result;
use std::sync::Arc;

pub type ModelCreator = Arc<dyn Fn(&ModelConfig) -> Result<Box<dyn Model>> + Send + Sync>;

pub struct ModelFactory {
    creators: Vec<ModelCreator>,
}

impl ModelFactory {
    pub fn new() -> Self {
        Self {
            creators: Vec::new(),
        }
    }
    
    pub fn with_creator(mut self, creator: ModelCreator) -> Self {
        self.creators.push(creator);
        self
    }
    
    pub fn create(&self, config: &ModelConfig) -> Result<Box<dyn Model>> {
        for creator in &self.creators {
            if let Ok(model) = creator(config) {
                return Ok(model);
            }
        }
        
        anyhow::bail!("No suitable model creator found for architecture: {}", config.architecture)
    }
}

impl Default for ModelFactory {
    fn default() -> Self {
        Self::new()
    }
}

pub trait ModelCreatorExt {
    fn create_model(&self, config: &ModelConfig) -> Result<Box<dyn Model>>;
}

impl<F> ModelCreatorExt for F 
where 
    F: Fn(&ModelConfig) -> Result<Box<dyn Model>> + Send + Sync + 'static,
{
    fn create_model(&self, config: &ModelConfig) -> Result<Box<dyn Model>> {
        self(config)
    }
}

pub fn creator<F>(f: F) -> ModelCreator
where
    F: Fn(&ModelConfig) -> Result<Box<dyn Model>> + Send + Sync + 'static,
{
    Arc::new(f)
}
