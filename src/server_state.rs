use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::mpsc;

pub mod config {
    use super::*;
    
    #[derive(Debug, Clone)]
    pub struct ServerConfig {
        pub host: String,
        pub port: u16,
        pub models_dir: PathBuf,
        pub context_size: usize,
        pub gpu_layers: i32,
        pub threads: i32,
        pub max_loaded_models: usize,
        pub keep_alive_secs: u64,
    }
    
    impl Default for ServerConfig {
        fn default() -> Self {
            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            Self {
                host: "127.0.0.1".to_string(),
                port: 11434,
                models_dir: home.join(".ollama").join("models"),
                context_size: 4096,
                gpu_layers: -1,
                threads: 4,
                max_loaded_models: 3,
                keep_alive_secs: 300,
            }
        }
    }
    
    impl ServerConfig {
        pub fn from_env() -> Self {
            let mut config = Self::default();
            
            if let Ok(host) = std::env::var("OLLAMA_HOST") {
                if let Some((h, p)) = host.rsplit_once(':') {
                    config.host = h.to_string();
                    if let Ok(port) = p.parse() {
                        config.port = port;
                    }
                } else {
                    config.host = host;
                }
            }
            
            if let Ok(dir) = std::env::var("OLLAMA_MODELS") {
                config.models_dir = PathBuf::from(dir);
            }
            
            if let Ok(ctx) = std::env::var("OLLAMA_CONTEXT_SIZE") {
                if let Ok(size) = ctx.parse() {
                    config.context_size = size;
                }
            }
            
            if let Ok(layers) = std::env::var("OLLAMA_GPU_LAYERS") {
                if let Ok(l) = layers.parse() {
                    config.gpu_layers = l;
                }
            }
            
            config
        }
    }
}

pub mod store {
    use super::*;
    use std::fs;
    use serde::{Deserialize, Serialize};
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelRecord {
        pub name: String,
        pub tag: String,
        pub path: PathBuf,
        pub size: u64,
        pub digest: String,
        pub modified_at: i64,
        pub config: ModelConfig,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct ModelConfig {
        pub context_length: usize,
        pub embedding_length: usize,
        pub parameter_count: u64,
        pub quantization: String,
        pub family: String,
        pub capabilities: Vec<String>,
        pub template: Option<String>,
        pub system: Option<String>,
    }
    
    pub struct ModelStore {
        models_dir: PathBuf,
        records: HashMap<String, ModelRecord>,
    }
    
    impl ModelStore {
        pub fn new(models_dir: PathBuf) -> Result<Self> {
            fs::create_dir_all(&models_dir)?;
            
            let mut store = Self {
                models_dir,
                records: HashMap::new(),
            };
            
            store.load_records()?;
            Ok(store)
        }
        
        fn load_records(&mut self) -> Result<()> {
            let records_file = self.models_dir.join("models.json");
            
            if records_file.exists() {
                let content = fs::read_to_string(&records_file)?;
                self.records = serde_json::from_str(&content).unwrap_or_default();
            }
            
            Ok(())
        }
        
        fn save_records(&self) -> Result<()> {
            let records_file = self.models_dir.join("models.json");
            let content = serde_json::to_string_pretty(&self.records)?;
            fs::write(&records_file, content)?;
            Ok(())
        }
        
        pub fn get(&self, name: &str) -> Option<&ModelRecord> {
            let (base_name, tag) = parse_model_name(name);
            let key = format!("{}:{}", base_name, tag);
            self.records.get(&key).or_else(|| self.records.get(base_name))
        }
        
        pub fn list(&self) -> Vec<&ModelRecord> {
            self.records.values().collect()
        }
        
        pub fn insert(&mut self, record: ModelRecord) -> Result<()> {
            let key = format!("{}:{}", record.name, record.tag);
            self.records.insert(key, record);
            self.save_records()
        }
        
        pub fn remove(&mut self, name: &str) -> Result<Option<ModelRecord>> {
            let (base_name, tag) = parse_model_name(name);
            let key = format!("{}:{}", base_name, tag);
            let record = self.records.remove(&key)
                .or_else(|| self.records.remove(base_name));
            
            if record.is_some() {
                self.save_records()?;
            }
            
            Ok(record)
        }
        
        pub fn model_path(&self, name: &str) -> Option<PathBuf> {
            self.get(name).map(|r| r.path.clone())
        }
    }
    
    fn parse_model_name(name: &str) -> (&str, &str) {
        if let Some(idx) = name.rfind(':') {
            if !name[idx..].contains('/') {
                return (&name[..idx], &name[idx+1..]);
            }
        }
        (name, "latest")
    }
}

pub mod inference {
    use super::*;
    use std::time::{Duration, Instant};
    use candle::{Tensor, Device};
    use candle_transformers::generation::{LogitsProcessor, Sampling};
    use candle_transformers::models::quantized_llama::{self, ModelWeights};
    use std::io::Read;
    
    #[derive(Debug, Clone)]
    pub struct InferenceConfig {
        pub temperature: f32,
        pub top_p: f32,
        pub top_k: i32,
        pub repeat_penalty: f32,
        pub repeat_last_n: i32,
        pub seed: u32,
        pub num_ctx: usize,
        pub num_predict: i32,
        pub num_threads: i32,
        pub stop: Vec<String>,
    }
    
    impl Default for InferenceConfig {
        fn default() -> Self {
            Self {
                temperature: 0.8,
                top_p: 0.9,
                top_k: 40,
                repeat_penalty: 1.1,
                repeat_last_n: 64,
                seed: 0,
                num_ctx: 2048,
                num_predict: 128,
                num_threads: 4,
                stop: vec![],
            }
        }
    }
    
    impl InferenceConfig {
        pub fn from_map(map: &HashMap<String, serde_json::Value>) -> Self {
            let mut config = Self::default();
            
            if let Some(v) = map.get("temperature").and_then(|v| v.as_f64()) {
                config.temperature = v as f32;
            }
            if let Some(v) = map.get("top_p").and_then(|v| v.as_f64()) {
                config.top_p = v as f32;
            }
            if let Some(v) = map.get("top_k").and_then(|v| v.as_i64()) {
                config.top_k = v as i32;
            }
            if let Some(v) = map.get("repeat_penalty").and_then(|v| v.as_f64()) {
                config.repeat_penalty = v as f32;
            }
            if let Some(v) = map.get("seed").and_then(|v| v.as_i64()) {
                config.seed = v as u32;
            }
            if let Some(v) = map.get("num_ctx").and_then(|v| v.as_u64()) {
                config.num_ctx = v as usize;
            }
            if let Some(v) = map.get("num_predict").and_then(|v| v.as_i64()) {
                config.num_predict = v as i32;
            }
            if let Some(v) = map.get("num_thread").and_then(|v| v.as_i64()) {
                config.num_threads = v as i32;
            }
            if let Some(v) = map.get("stop").and_then(|v| v.as_array()) {
                config.stop = v.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect();
            }
            
            config
        }
    }
    
    #[derive(Debug)]
    pub struct InferenceResult {
        pub text: String,
        pub tokens_evaluated: usize,
        pub tokens_generated: usize,
        pub duration: Duration,
    }
    
    pub struct InferenceEngine {
        model_path: PathBuf,
        config: InferenceConfig,
        model: Option<ModelWeights>,
        device: Device,
    }
    
    impl InferenceEngine {
        pub fn new(model_path: PathBuf, config: InferenceConfig) -> Self {
            let device = Device::Cpu;
            Self { 
                model_path, 
                config,
                model: None,
                device,
            }
        }
        
        fn load_model(&mut self) -> Result<()> {
            if self.model.is_some() {
                return Ok(());
            }
            
            let mut file = std::fs::File::open(&self.model_path)?;
            let gguf = candle_core::quantized::gguf_file::Content::read(&mut file)?;
            
            let model = ModelWeights::from_gguf(gguf, &mut file, &self.device)?;
            self.model = Some(model);
            
            Ok(())
        }
        
        pub fn generate<F>(&mut self, prompt: &str, mut callback: F) -> Result<InferenceResult>
        where
            F: FnMut(&str),
        {
            let start = Instant::now();
            
            self.load_model()?;
            
            let model = self.model.as_mut().ok_or_else(||
                anyhow::anyhow!("Model not loaded")
            )?;
            
            let logits_processor = LogitsProcessor::from_sampling(
                self.config.seed,
                Sampling::Temperature(self.config.temperature),
            );
            
            let mut output = String::new();
            let prompt_tokens: Vec<u32> = prompt
                .chars()
                .map(|c| c as u32)
                .collect();
            
            let mut all_tokens = prompt_tokens.clone();
            let max_tokens = if self.config.num_predict < 0 { 128 } else { self.config.num_predict as usize };
            let mut tokens_generated = 0;
            
            for _ in 0..max_tokens {
                let logits = model.forward(&all_tokens)?;
                
                let next_token = logits_processor.sample(&logits)?;
                
                if next_token == 0 || next_token as char == '\0' {
                                   
                let token_str = if next_token < 256 {
 break;
                }
                    (next_token as u8).to_string()
                } else {
                    next_token.to_string()
                };
                
                callback(&token_str);
                output.push_str(&token_str);
                all_tokens.push(next_token);
                tokens_generated += 1;
                
                for stop in &self.config.stop {
                    if output.ends_with(stop) {
                        output = output[..output.len() - stop.len()].to_string();
                        break;
                    }
                }
            }
            
            let duration = start.elapsed();
            
            Ok(InferenceResult {
                text: output,
                tokens_evaluated: prompt_tokens.len(),
                tokens_generated,
                duration,
            })
        }
        
        pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
            let tokens: Vec<f32> = text
                .chars()
                .map(|c| c as f32 / 255.0)
                .collect();
            
            let mut embedding = vec![0.0f32; 768];
            for (i, &t) in tokens.iter().enumerate() {
                embedding[i % 768] += t;
            }
            
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }
            
            Ok(embedding)
        }
    }
}

pub mod session {
    use super::*;
    use std::collections::VecDeque;
    
    pub struct Session {
        pub model_name: String,
        pub messages: Vec<Message>,
        pub context: VecDeque<i32>,
        pub created_at: std::time::Instant,
        pub last_used: std::time::Instant,
        pub keep_alive: std::time::Duration,
    }
    
    #[derive(Debug, Clone)]
    pub struct Message {
        pub role: String,
        pub content: String,
    }
    
    impl Session {
        pub fn new(model_name: String, keep_alive_secs: u64) -> Self {
            Self {
                model_name,
                messages: Vec::new(),
                context: VecDeque::new(),
                created_at: std::time::Instant::now(),
                last_used: std::time::Instant::now(),
                keep_alive: std::time::Duration::from_secs(keep_alive_secs),
            }
        }
        
        pub fn add_message(&mut self, role: &str, content: &str) {
            self.messages.push(Message {
                role: role.to_string(),
                content: content.to_string(),
            });
            self.last_used = std::time::Instant::now();
        }
        
        pub fn is_expired(&self) -> bool {
            self.last_used.elapsed() > self.keep_alive
        }
    }
    
    pub struct SessionManager {
        sessions: HashMap<String, Session>,
        default_keep_alive: std::time::Duration,
    }
    
    impl SessionManager {
        pub fn new(keep_alive_secs: u64) -> Self {
            Self {
                sessions: HashMap::new(),
                default_keep_alive: std::time::Duration::from_secs(keep_alive_secs),
            }
        }
        
        pub fn get_or_create(&mut self, model_name: &str) -> &mut Session {
            self.sessions.entry(model_name.to_string())
                .or_insert_with(|| Session::new(model_name.to_string(), self.default_keep_alive.as_secs()))
        }
        
        pub fn remove(&mut self, model_name: &str) -> Option<Session> {
            self.sessions.remove(model_name)
        }
        
        pub fn cleanup_expired(&mut self) {
            self.sessions.retain(|_, session| !session.is_expired());
        }
        
        pub fn list(&self) -> Vec<&Session> {
            self.sessions.values().collect()
        }
    }
}

pub use config::ServerConfig;
pub use store::{ModelStore, ModelRecord, ModelConfig};
pub use inference::{InferenceEngine, InferenceConfig, InferenceResult};
pub use session::{Session, SessionManager, Message};
