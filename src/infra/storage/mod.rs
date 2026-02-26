use crate::core::Result;
use std::path::{Path, PathBuf};
use std::collections::HashMap;

pub struct ModelRepository {
    models_dir: PathBuf,
    cache: HashMap<String, ModelMeta>,
}

#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub name: String,
    pub path: PathBuf,
    pub size: u64,
    pub modified: std::time::SystemTime,
}

impl ModelRepository {
    pub fn new<P: AsRef<Path>>(models_dir: P) -> Self {
        Self {
            models_dir: models_dir.as_ref().to_path_buf(),
            cache: HashMap::new(),
        }
    }
    
    pub fn default_models_dir() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".ollama")
            .join("models")
    }
    
    pub fn list(&self) -> Result<Vec<ModelMeta>> {
        let mut models = Vec::new();
        
        if !self.models_dir.exists() {
            return Ok(models);
        }
        
        for entry in std::fs::read_dir(&self.models_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    let manifests_dir = path.join("manifests");
                    if manifests_dir.exists() {
                        for manifest_entry in std::fs::read_dir(&manifests_dir)? {
                            let manifest_entry = manifest_entry?;
                            let manifest_path = manifest_entry.path();
                            
                            if let Ok(metadata) = std::fs::metadata(&manifest_path) {
                                models.push(ModelMeta {
                                    name: name.to_string(),
                                    path: manifest_path,
                                    size: metadata.len(),
                                    modified: metadata.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Ok(models)
    }
    
    pub fn get(&self, name: &str) -> Option<&ModelMeta> {
        self.cache.get(name)
    }
    
    pub fn exists(&self, name: &str) -> bool {
        let model_path = self.models_dir.join(name);
        model_path.exists()
    }
    
    pub fn model_path(&self, name: &str) -> PathBuf {
        self.models_dir.join(name).join("model.gguf")
    }
    
    pub fn delete(&self, name: &str) -> Result<()> {
        let model_path = self.models_dir.join(name);
        if model_path.exists() {
            std::fs::remove_dir_all(&model_path)?;
        }
        Ok(())
    }
    
    pub fn refresh(&mut self) -> Result<()> {
        self.cache.clear();
        
        let models = self.list()?;
        for model in models {
            self.cache.insert(model.name.clone(), model);
        }
        
        Ok(())
    }
    
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }
}

impl Default for ModelRepository {
    fn default() -> Self {
        Self::new(Self::default_models_dir())
    }
}
