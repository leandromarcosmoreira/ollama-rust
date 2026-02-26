#![allow(clippy::module_inception)]
#![allow(clippy::field_reassign_with_default)]
#![allow(unused)]
pub mod runner {
    use anyhow::Result;
    use std::collections::HashMap;
    use serde::{Deserialize, Serialize};
    use anyhow::bail;
    use chrono::Utc;
    use candle_transformers::generation::LogitsProcessor;

    #[derive(Debug, Clone, Default)]
    #[allow(dead_code)]
    pub struct RunnerOptions {
        pub context_size: usize,
        pub gpu_layers: i32,
        pub threads: i32,
        pub batch_size: usize,
        pub temperature: f32,
        pub top_p: f32,
        pub top_k: i32,
        pub repeat_penalty: f32,
        pub repeat_last_n: i32,
        pub seed: i32,
        pub num_predict: i32,
        pub num_gqa: i32,
        pub rope_freq_base: f32,
        pub rope_freq_scale: f32,
        pub yarn_ext_factor: f32,
        pub yarn_attn_factor: f32,
        pub yarn_beta_fast: f32,
        pub yarn_beta_slow: f32,
        pub raw: bool,
    }

    impl RunnerOptions {
        pub fn from_map(m: &HashMap<String, serde_json::Value>) -> Self {
            let mut opts = Self::default();
            
            if let Some(v) = m.get("num_ctx") {
                if let Some(n) = v.as_u64() {
                    opts.context_size = n as usize;
                }
            }
            if let Some(v) = m.get("gpu_layers") {
                if let Some(n) = v.as_i64() {
                    opts.gpu_layers = n as i32;
                }
            }
            if let Some(v) = m.get("threads") {
                if let Some(n) = v.as_i64() {
                    opts.threads = n as i32;
                }
            }
            if let Some(v) = m.get("batch_size") {
                if let Some(n) = v.as_u64() {
                    opts.batch_size = n as usize;
                }
            }
            if let Some(v) = m.get("temperature") {
                if let Some(n) = v.as_f64() {
                    opts.temperature = n as f32;
                }
            }
            if let Some(v) = m.get("top_p") {
                if let Some(n) = v.as_f64() {
                    opts.top_p = n as f32;
                }
            }
            if let Some(v) = m.get("top_k") {
                if let Some(n) = v.as_i64() {
                    opts.top_k = n as i32;
                }
            }
            if let Some(v) = m.get("repeat_penalty") {
                if let Some(n) = v.as_f64() {
                    opts.repeat_penalty = n as f32;
                }
            }
            if let Some(v) = m.get("repeat_last_n") {
                if let Some(n) = v.as_i64() {
                    opts.repeat_last_n = n as i32;
                }
            }
            if let Some(v) = m.get("seed") {
                if let Some(n) = v.as_i64() {
                    opts.seed = n as i32;
                }
            }
            
            opts
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Message {
        pub role: String,
        pub content: String,
        pub images: Vec<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[allow(dead_code)]
    pub struct ToolCall {
        pub id: Option<String>,
        pub function: FunctionCall,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[allow(dead_code)]
    pub struct FunctionCall {
        pub name: String,
        pub arguments: String,
    }

    #[derive(Debug)]
    #[allow(dead_code)]
    pub struct GenerateResult {
        pub response: String,
        pub done: bool,
        pub context: Vec<i32>,
        pub total_duration: i64,
        pub load_duration: i64,
        pub prompt_eval_count: i32,
        pub prompt_eval_duration: i64,
        pub eval_count: i32,
        pub eval_duration: i64,
    }

    #[derive(Debug)]
    #[allow(dead_code)]
    pub struct ChatResult {
        pub message: Message,
        pub done: bool,
        pub total_duration: i64,
        pub eval_count: i32,
        pub eval_duration: i64,
    }

    #[derive(Debug)]
    pub struct EmbedResult {
        pub embeddings: Vec<Vec<f32>>,
        pub total_duration: i64,
    }

    #[allow(dead_code)]
    pub struct Runner {
        model_name: String,
        model_path: String,
        options: RunnerOptions,
        tool_executor: crate::tools::ToolExecutor,
        model: Option<Box<dyn ollama::Model>>,
        tokenizer: Option<Box<dyn ollama::Tokenizer>>,
    }

    #[allow(dead_code)]
    impl Runner {
        pub fn new(model_path: &str) -> Result<Self> {
            Ok(Self {
                model_name: String::new(),
                model_path: model_path.to_string(),
                options: RunnerOptions::default(),
                tool_executor: crate::tools::ToolExecutor::new(),
                model: None,
                tokenizer: None,
            })
        }

        pub fn with_options(mut self, options: RunnerOptions) -> Self {
            self.options = options;
            self
        }

        pub fn load(&mut self) -> Result<()> {
            println!("Loading model from {} with {} GPU layers", self.model_path, self.options.gpu_layers);
            
            // Load GGUF metadata to get config
            let gguf = ollama::infra::GgufParser::parse(&self.model_path)?;
            let config = gguf.metadata.to_model_config();
            
            // Load model weights using Llama architecture (assuming llama for now as per current codebase)
            let model = ollama::core::model::architectures::llama::LlamaModel::load(&self.model_path, config.clone())?;
            self.model = Some(Box::new(model));
            
            // Load tokenizer from GGUF metadata
            let vocab = self.extract_vocab_from_gguf(&gguf);
            let kind = if config.architecture.contains("llama") {
                ollama::core::tokenizer::TokenizerKind::Bpe
            } else {
                ollama::core::tokenizer::TokenizerKind::WordPiece
            };
            self.tokenizer = Some(ollama::core::tokenizer::create_tokenizer(kind, vocab));
            
            Ok(())
        }

        fn extract_vocab_from_gguf(&self, gguf: &ollama::infra::gguf::GgufFile) -> ollama::core::tokenizer::Vocabulary {
            let tokens = if let Some(ollama::infra::gguf::MetadataValue::Array(arr)) = gguf.metadata.get("tokenizer.ggml.tokens") {
                arr.iter().filter_map(|v| match v {
                    ollama::infra::gguf::MetadataValue::String(s) => Some(s.clone()),
                    _ => None,
                }).collect()
            } else {
                vec![]
            };

            let scores = if let Some(ollama::infra::gguf::MetadataValue::Array(arr)) = gguf.metadata.get("tokenizer.ggml.scores") {
                arr.iter().filter_map(|v| match v {
                    ollama::infra::gguf::MetadataValue::Float(f) => Some(*f as f32),
                    _ => None,
                }).collect()
            } else {
                vec![0.0; tokens.len()]
            };

            let mut vocab = ollama::core::tokenizer::Vocabulary::new(tokens);
            vocab.scores = scores;
            
            // Try to find special tokens
            if let Some(ollama::infra::gguf::MetadataValue::Uint(id)) = gguf.metadata.get("tokenizer.ggml.bos_token_id") {
                vocab.bos_token = ollama::core::TokenId(*id as i32);
            }
            if let Some(ollama::infra::gguf::MetadataValue::Uint(id)) = gguf.metadata.get("tokenizer.ggml.eos_token_id") {
                vocab.eos_token = ollama::core::TokenId(*id as i32);
            }
            
            vocab
        }

        pub fn generate<F>(&mut self, prompt: &str, mut callback: F) -> Result<GenerateResult>
        where F: FnMut(String, bool)
        {
            let model = self.model.as_mut().ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;
            let tokenizer = self.tokenizer.as_ref().ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;
            
            let tokens = tokenizer.encode(prompt)?;
            let mut current_tokens = tokens.clone();
            let mut generated = String::new();
            
            // Note: We use the cache from the trait if provided, otherwise create a local one.
            // For now, LlamaModel handles its own internal state if start_pos is provided,
            // but we pass a stub cache to satisfy the trait.
            let mut stub_cache = ollama::core::cache::CausalKVCache::new(0, 0, 0, 0);

            let start_time = std::time::Instant::now();
            let mut eval_count = 0;

            // Initialize LogitsProcessor for real sampling
            let mut logits_processor = LogitsProcessor::new(
                self.options.seed as u64,
                Some(self.options.temperature as f64),
                Some(self.options.top_p as f64),
            );

            // Generation loop
            let max_to_generate = if self.options.num_predict > 0 { self.options.num_predict } else { 128 };
            
            for i in 0..max_to_generate {
                // If it's the first token, we process the whole prompt
                // If not, we only process the last generated token
                let (input_tokens, pos) = if i == 0 {
                    (current_tokens.clone(), (0..current_tokens.len()).collect::<Vec<_>>())
                } else {
                    let last = current_tokens.last().cloned().unwrap();
                    (vec![last], vec![current_tokens.len() - 1])
                };

                let logits = model.forward(&input_tokens, &pos, &mut stub_cache)?;
                
                // Real Probabilistic Sampling
                let logits_vec = logits.data();
                let candle_logits = candle_core::Tensor::new(logits_vec, &candle_core::Device::Cpu)?;
                let next_token_u32 = logits_processor.sample(&candle_logits)?;
                let next_token = ollama::TokenId(next_token_u32 as i32);
                
                if next_token == tokenizer.eos_token() {
                    break;
                }

                let token_text = tokenizer.decode(&[next_token])?;
                generated.push_str(&token_text);
                callback(token_text, false);
                
                current_tokens.push(next_token);
                eval_count += 1;
            }

            callback(String::new(), true);

            Ok(GenerateResult {
                response: generated,
                done: true,
                context: current_tokens.iter().map(|t| t.0).collect(),
                total_duration: start_time.elapsed().as_nanos() as i64,
                load_duration: 0,
                prompt_eval_count: tokens.len() as i32,
                prompt_eval_duration: 0,
                eval_count: eval_count as i32,
                eval_duration: 0,
            })
        }

        pub fn chat<F>(&mut self, messages: &[Message], _tools: Option<&str>, mut callback: F) -> Result<ChatResult> 
        where F: FnMut(String, bool)
        {
            // Simplified chat for now: combine messages into a prompt
            let mut prompt = String::new();
            for msg in messages {
                prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
            }
            prompt.push_str("assistant: ");
            
            let res = self.generate(&prompt, callback)?;
            
            Ok(ChatResult {
                message: Message {
                    role: "assistant".to_string(),
                    content: res.response,
                    images: vec![],
                },
                done: true,
                total_duration: res.total_duration,
                eval_count: res.eval_count,
                eval_duration: res.eval_duration,
            })
        }

        pub fn embed(&mut self, input: &str, _dimensions: Option<usize>) -> Result<EmbedResult> {
            let model = self.model.as_mut().ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;
            let tokenizer = self.tokenizer.as_ref().ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;
            
            let tokens = tokenizer.encode(input)?;
            let embedding = model.embed(&tokens)?;
            
            Ok(EmbedResult {
                embeddings: vec![embedding.data().to_vec()],
                total_duration: 0,
            })
        }

        pub fn is_loaded(&self) -> bool {
            self.model.is_some() && self.tokenizer.is_some()
        }

        pub fn unload(&mut self) {
            println!("Model unloaded");
        }
    }

    #[allow(dead_code)]
    fn detect_tool_call(response: &str) -> Option<ToolCall> {
        // Very simple regex or JSON-like parsing for demonstration
        // In a real implementation this would be more robust
        if let Some(start) = response.find("{") {
            if let Some(end) = response.rfind("}") {
                let json_str = &response[start..=end];
                if let Ok(call) = serde_json::from_str::<ToolCall>(json_str) {
                    return Some(call);
                }
            }
        }
        None
    }

    #[allow(dead_code)]
    fn simple_hash(s: &str) -> u64 {
        let mut hash: u64 = 5381;
        for c in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(c as u64);
        }
        hash
    }
}

pub mod scheduler {
    use anyhow::Result;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use std::time::{Duration, Instant};
    use chrono::Utc;

    use super::runner::Runner;

    #[derive(Clone)]
    #[allow(dead_code)]
    pub struct ScheduledRunner {
        pub runner: Arc<RwLock<Runner>>,
        pub model_name: String,
        pub loaded_at: Instant,
        pub last_used: Instant,
        pub keep_alive: Option<Duration>,
        pub size: u64,
    }

    #[allow(dead_code)]
    pub struct Scheduler {
        runners: HashMap<String, ScheduledRunner>,
        max_models: usize,
        default_keep_alive: Duration,
    }

    #[allow(dead_code)]
    impl Scheduler {
        pub fn new(max_models: usize) -> Self {
            Self {
                runners: HashMap::new(),
                max_models,
                default_keep_alive: Duration::from_secs(300), // 5 minutes
            }
        }

        pub async fn get_runner(&mut self, model_name: &str, model_path: &str) -> Result<Arc<RwLock<Runner>>> {
            // Check if runner already exists - use get_mut for mutable access
            if let Some(scheduled) = self.runners.get_mut(model_name) {
                scheduled.last_used = Instant::now();
                return Ok(scheduled.runner.clone());
            }

            // Evict old runners if at capacity
            while self.runners.len() >= self.max_models {
                self.evict_oldest().await?;
            }

            // Create new runner
            let runner = Runner::new(model_path)?;
            let size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

            let scheduled = ScheduledRunner {
                runner: Arc::new(RwLock::new(runner)),
                model_name: model_name.to_string(),
                loaded_at: Instant::now(),
                last_used: Instant::now(),
                keep_alive: Some(self.default_keep_alive),
                size,
            };

            self.runners.insert(model_name.to_string(), scheduled);
            Ok(self.runners.get(model_name).unwrap().runner.clone())
        }

        async fn evict_oldest(&mut self) -> Result<()> {
            if let Some((name, _)) = self.runners.iter()
                .min_by_key(|(_, v)| v.last_used)
                .map(|(k, v)| (k.clone(), v))
            {
                let runner = self.runners.remove(&name).unwrap();
                runner.runner.write().await.unload();
            }
            Ok(())
        }

        pub async fn cleanup(&mut self) {
            let now = Instant::now();
            let to_remove: Vec<String> = self.runners.iter()
                .filter(|(_, s)| {
                    if let Some(keep_alive) = s.keep_alive {
                        now.duration_since(s.last_used) > keep_alive
                    } else {
                        false
                    }
                })
                .map(|(k, _)| k.clone())
                .collect();

            for name in to_remove {
                if let Some(runner) = self.runners.remove(&name) {
                    runner.runner.write().await.unload();
                }
            }
        }

        pub fn runner_count(&self) -> usize {
            self.runners.len()
        }

        pub fn list_running(&self) -> Vec<crate::api::RunningModel> {
            let now_utc = Utc::now();
            let now_instant = Instant::now();
            
            self.runners.iter().map(|(name, s)| {
                let expires_at = if let Some(keep_alive) = s.keep_alive {
                    let elapsed = now_instant.duration_since(s.last_used);
                    if keep_alive > elapsed {
                        (now_utc + (keep_alive - elapsed)).to_rfc3339()
                    } else {
                        now_utc.to_rfc3339()
                    }
                } else {
                    "Never".to_string()
                };

                crate::api::RunningModel {
                    name: name.clone(),
                    model: name.clone(),
                    modified_at: Utc::now().to_rfc3339(), 
                    size: s.size,
                    size_vram: s.size, // Assuming all in VRAM for now if CUDA used
                    digest: String::new(),
                    expires_at,
                    context_length: 0,
                }
            }).collect()
        }

        pub async fn unload(&mut self, model_name: &str) -> Result<()> {
            if let Some(s) = self.runners.remove(model_name) {
                s.runner.write().await.unload();
            }
            Ok(())
        }
    }
}
