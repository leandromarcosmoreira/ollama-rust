use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct OpenAiApi {
    default_model: String,
}

impl OpenAiApi {
    pub fn new() -> Self {
        Self {
            default_model: "llama".to_string(),
        }
    }
    
    pub fn default_model(&self) -> &str {
        &self.default_model
    }
    
    pub fn set_default_model(&mut self, model: impl Into<String>) {
        self.default_model = model.into();
    }
}

impl Default for OpenAiApi {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<i32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub max_tokens: Option<i32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>>,
}

impl Default for MessageContent {
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        MessageContent::Text(text)
    }
}

impl From<&str> for MessageContent {
    fn from(text: &str) -> Self {
        MessageContent::Text(text.to_string())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: Option<String>,
    pub image_url: Option<ImageUrl>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: i32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelData>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelData {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

impl ModelData {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp(),
            owned_by: "local".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: i32,
}

pub fn convert_ollama_to_openai(ollama_response: &str, model: &str) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: model.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: MessageContent::Text(ollama_response.to_string()),
                name: None,
                function_call: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}

pub fn convert_openai_to_ollama(request: &ChatCompletionRequest) -> crate::api::ollama::ChatRequest {
    let messages: Vec<crate::api::ollama::ChatMessage> = request.messages.iter()
        .map(|m| {
            let content = match &m.content {
                MessageContent::Text(t) => t.clone(),
                MessageContent::Parts(parts) => {
                    parts.iter()
                        .filter_map(|p| p.text.clone())
                        .collect::<Vec<_>>()
                        .join(" ")
                }
            };
            
            crate::api::ollama::ChatMessage {
                role: m.role.clone(),
                content,
                images: vec![],
            }
        })
        .collect();
    
    crate::api::ollama::ChatRequest {
        model: request.model.clone(),
        messages,
        stream: request.stream,
        options: crate::api::ollama::RequestOptions {
            temperature: request.temperature.unwrap_or(0.8),
            top_p: request.top_p.unwrap_or(0.9),
            num_ctx: 2048,
            num_predict: request.max_tokens.unwrap_or(-1),
            stop: match &request.stop {
                Some(StopSequence::Single(s)) => vec![s.clone()],
                Some(StopSequence::Multiple(v)) => v.clone(),
                None => vec![],
            },
            seed: 0,
        },
    }
}
