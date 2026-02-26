use crate::core::{Result, Model, TokenId};
use crate::app::InferenceRunner;
use serde::{Deserialize, Serialize};

pub struct OllamaApi {
    version: String,
}

impl OllamaApi {
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
    
    pub fn version(&self) -> &str {
        &self.version
    }
}

impl Default for OllamaApi {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub raw: bool,
    #[serde(default)]
    pub options: RequestOptions,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct RequestOptions {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_num_ctx")]
    pub num_ctx: usize,
    #[serde(default)]
    pub num_predict: i32,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub seed: i32,
}

fn default_temperature() -> f32 { 0.8 }
fn default_top_p() -> f32 { 0.9 }
fn default_num_ctx() -> usize { 2048 }

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub options: RequestOptions,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: ChatMessage,
    pub done: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ShowRequest {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ShowResponse {
    pub license: Option<String>,
    pub modelfile: Option<String>,
    pub parameters: Option<String>,
    pub template: Option<String>,
    pub system: Option<String>,
    pub details: Option<ModelDetails>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelDetails {
    pub format: String,
    pub family: String,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PullRequest {
    pub name: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub insecure: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PullResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}
