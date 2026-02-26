use super::traits::{Model, ModelConfig};
use super::factory::ModelCreator;
use crate::core::Result;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub struct ModelRegistry {
    architectures: RwLock<HashMap<String, ModelCreator>>,
    aliases: RwLock<HashMap<String, String>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            architectures: RwLock::new(HashMap::new()),
            aliases: RwLock::new(HashMap::new()),
        }
    }
    
    pub fn register<N, F>(&self, name: N, creator: F)
    where
        N: Into<String>,
        F: Fn(&ModelConfig) -> Result<Box<dyn Model>> + Send + Sync + 'static,
    {
        let name = name.into();
        let mut architectures = self.architectures.write().unwrap();
        let creator = Arc::new(creator) as ModelCreator;
        architectures.insert(name, creator);
    }
    
    pub fn register_alias<A, T>(&self, alias: A, target: T)
    where
        A: Into<String>,
        T: Into<String>,
    {
        let mut aliases = self.aliases.write().unwrap();
        aliases.insert(alias.into(), target.into());
    }
    
    pub fn get(&self, name: &str) -> Option<ModelCreator> {
        let architectures = self.architectures.read().unwrap();
        
        if let Some(creator) = architectures.get(name) {
            return Some(Arc::clone(creator));
        }
        
        let aliases = self.aliases.read().unwrap();
        if let Some(target) = aliases.get(name) {
            return architectures.get(target).map(Arc::clone);
        }
        
        None
    }
    
    pub fn create(&self, config: &ModelConfig) -> Result<Box<dyn Model>> {
        let arch = &config.architecture;
        
        let creator = self.get(arch)
            .or_else(|| {
                let base_arch = arch.split('_').next().unwrap_or(arch);
                self.get(base_arch)
            })
            .ok_or_else(|| anyhow::anyhow!("Unsupported architecture: {}", arch))?;
        
        creator(config)
    }
    
    pub fn architectures(&self) -> Vec<String> {
        let architectures = self.architectures.read().unwrap();
        architectures.keys().cloned().collect()
    }
    
    pub fn clear(&self) {
        let mut architectures = self.architectures.write().unwrap();
        let mut aliases = self.aliases.write().unwrap();
        architectures.clear();
        aliases.clear();
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub static REGISTRY: once_cell::sync::Lazy<ModelRegistry> = 
    once_cell::sync::Lazy::new(ModelRegistry::new);

pub fn register<N, F>(name: N, creator: F)
where
    N: Into<String>,
    F: Fn(&ModelConfig) -> Result<Box<dyn Model>> + Send + Sync + 'static,
{
    REGISTRY.register(name, creator);
}

pub fn create(config: &ModelConfig) -> Result<Box<dyn Model>> {
    REGISTRY.create(config)
}

pub fn architectures() -> Vec<String> {
    REGISTRY.architectures()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct TestModel;
    
    impl Model for TestModel {
        fn forward(
            &mut self,
            _input: &[crate::core::TokenId],
            _positions: &[usize],
            _cache: &mut dyn crate::core::KVCache,
        ) -> Result<crate::core::Tensor> {
            unimplemented!()
        }
        
        fn forward_batch(
            &mut self,
            _batch: &super::super::traits::ModelBatch,
            _cache: &mut dyn crate::core::KVCache,
        ) -> Result<crate::core::Tensor> {
            unimplemented!()
        }
        
        fn config(&self) -> &ModelConfig {
            unimplemented!()
        }
        
        fn meta(&self) -> &super::super::ModelMeta {
            unimplemented!()
        }
        
        fn embed(&self, _tokens: &[crate::core::TokenId]) -> Result<crate::core::Tensor> {
            unimplemented!()
        }
        
        fn logits(&self, _hidden: &crate::core::Tensor) -> Result<crate::core::Tensor> {
            unimplemented!()
        }
    }
    
    #[test]
    fn test_registry() {
        let registry = ModelRegistry::new();
        
        registry.register("test", |_config| {
            Ok(Box::new(TestModel))
        });
        
        registry.register_alias("test-alias", "test");
        
        assert!(registry.get("test").is_some());
        assert!(registry.get("test-alias").is_some());
        assert!(registry.get("unknown").is_none());
    }
}
