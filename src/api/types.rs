#![allow(dead_code)]
#![allow(unused)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Time {
    #[serde(default)]
    pub value: Option<i64>,
}

impl Time {
    pub fn new() -> Self {
        Self { value: None }
    }

    pub fn from_timestamp(ts: i64) -> Self {
        Self { value: Some(ts) }
    }
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub result: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct File {
    pub filename: String,
    pub data: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Attachment {
    pub filename: String,
    #[serde(default)]
    pub data: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub attachments: Option<Vec<File>>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(rename = "tool_call", default)]
    pub tool_call: Option<ToolCall>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub tool_result: Option<Vec<u8>>,
    #[serde(default)]
    pub created_at: Option<Time>,
    #[serde(default)]
    pub updated_at: Option<Time>,
    #[serde(default)]
    pub thinking_time_start: Option<String>,
    #[serde(default)]
    pub thinking_time_end: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Chat {
    pub id: String,
    pub messages: Vec<Message>,
    pub title: String,
    #[serde(default)]
    pub created_at: Option<Time>,
    #[serde(default)]
    pub browser_state: Option<BrowserStateData>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatInfo {
    pub id: String,
    pub title: String,
    pub user_excerpt: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatsResponse {
    #[serde(default)]
    pub chat_infos: Vec<ChatInfo>,
}

impl ChatsResponse {
    pub fn new() -> Self {
        Self {
            chat_infos: Vec::new(),
        }
    }
}

impl Default for ChatsResponse {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatResponse {
    pub chat: Chat,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Model {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub modified_at: Option<Time>,
}

impl Model {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            digest: None,
            modified_at: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelsResponse {
    #[serde(default)]
    pub models: Vec<Model>,
}

impl ModelsResponse {
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }
}

impl Default for ModelsResponse {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InferenceCompute {
    pub library: String,
    pub variant: String,
    pub compute: String,
    pub driver: String,
    pub name: String,
    pub vram: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InferenceComputeResponse {
    #[serde(default)]
    pub inference_computes: Vec<InferenceCompute>,
    #[serde(default)]
    pub default_context_length: Option<u32>,
}

impl InferenceComputeResponse {
    pub fn new() -> Self {
        Self {
            inference_computes: Vec::new(),
            default_context_length: None,
        }
    }
}

impl Default for InferenceComputeResponse {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelCapabilitiesResponse {
    #[serde(default)]
    pub capabilities: Vec<String>,
}

impl ModelCapabilitiesResponse {
    pub fn new() -> Self {
        Self {
            capabilities: Vec::new(),
        }
    }

    pub fn with_capabilities(capabilities: Vec<String>) -> Self {
        Self { capabilities }
    }
}

impl Default for ModelCapabilitiesResponse {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum ChatEventName {
    Chat,
    Thinking,
    AssistantWithTools,
    ToolCall,
    Tool,
    ToolResult,
    Done,
    ChatCreated,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ChatEvent {
    pub event_name: ChatEventName,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub thinking_time_start: Option<String>,
    #[serde(default)]
    pub thinking_time_end: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_call: Option<ToolCall>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub tool_result: Option<bool>,
    #[serde(default)]
    pub tool_result_data: Option<serde_json::Value>,
    #[serde(default)]
    pub chat_id: Option<String>,
    #[serde(default)]
    pub tool_state: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DownloadEvent {
    pub event_name: String,
    pub total: u64,
    pub completed: u64,
    pub done: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ErrorEvent {
    pub event_name: String,
    pub error: String,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub details: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    #[serde(default)]
    pub expose: Option<bool>,
    #[serde(default)]
    pub browser: Option<bool>,
    #[serde(default)]
    pub survey: Option<bool>,
    #[serde(default)]
    pub models: Option<String>,
    #[serde(default)]
    pub agent: Option<bool>,
    #[serde(default)]
    pub tools: Option<bool>,
    #[serde(default)]
    pub working_dir: Option<String>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub turbo_enabled: Option<bool>,
    #[serde(default)]
    pub web_search_enabled: Option<bool>,
    #[serde(default)]
    pub think_enabled: Option<bool>,
    #[serde(default)]
    pub think_level: Option<String>,
    #[serde(default)]
    pub selected_model: Option<String>,
    #[serde(default)]
    pub sidebar_open: Option<bool>,
}

impl Settings {
    pub fn new() -> Self {
        Self {
            expose: Some(false),
            browser: Some(false),
            survey: Some(false),
            models: None,
            agent: Some(false),
            tools: Some(false),
            working_dir: None,
            context_length: None,
            turbo_enabled: Some(false),
            web_search_enabled: Some(false),
            think_enabled: Some(false),
            think_level: None,
            selected_model: None,
            sidebar_open: Some(true),
        }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SettingsResponse {
    pub settings: Settings,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HealthResponse {
    pub healthy: bool,
}

impl HealthResponse {
    pub fn new(healthy: bool) -> Self {
        Self { healthy }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct User {
    pub id: String,
    pub email: String,
    pub name: String,
    #[serde(default)]
    pub bio: Option<String>,
    #[serde(default)]
    pub avatar_url: Option<String>,
    #[serde(default)]
    pub firstname: Option<String>,
    #[serde(default)]
    pub lastname: Option<String>,
    #[serde(default)]
    pub plan: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub index: Option<u32>,
    #[serde(default)]
    pub attachments: Option<Vec<Attachment>>,
    #[serde(default)]
    pub web_search: Option<bool>,
    #[serde(default)]
    pub file_tools: Option<bool>,
    #[serde(default)]
    pub force_update: Option<bool>,
    #[serde(default)]
    pub think: Option<serde_json::Value>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            index: None,
            attachments: None,
            web_search: None,
            file_tools: None,
            force_update: None,
            think: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Error {
    pub error: String,
}

impl Error {
    pub fn new(error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelUpstreamResponse {
    #[serde(default)]
    pub digest: Option<String>,
    pub push_time: u64,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Page {
    pub url: String,
    pub title: String,
    pub text: String,
    pub lines: Vec<String>,
    #[serde(default)]
    pub links: Option<HashMap<u32, String>>,
    #[serde(default)]
    pub fetched_at: Option<Time>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BrowserStateData {
    #[serde(default)]
    pub page_stack: Vec<String>,
    #[serde(default)]
    pub view_tokens: Option<u32>,
    #[serde(default)]
    pub url_to_page: Option<HashMap<String, Page>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CloudStatusResponse {
    pub disabled: bool,
    pub source: String,
}

impl CloudStatusResponse {
    pub fn new(disabled: bool, source: impl Into<String>) -> Self {
        Self {
            disabled,
            source: source.into(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SettingsState {
    pub turbo_enabled: bool,
    pub web_search_enabled: bool,
    pub selected_model: String,
    pub sidebar_open: bool,
    pub think_enabled: bool,
    pub think_level: String,
}

impl Default for SettingsState {
    fn default() -> Self {
        Self {
            turbo_enabled: false,
            web_search_enabled: false,
            selected_model: String::new(),
            sidebar_open: false,
            think_enabled: false,
            think_level: "none".to_string(),
        }
    }
}

impl SettingsState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_settings(settings: &Settings) -> Self {
        Self {
            turbo_enabled: settings.turbo_enabled.unwrap_or(false),
            web_search_enabled: settings.web_search_enabled.unwrap_or(false),
            selected_model: settings.selected_model.clone().unwrap_or_default(),
            sidebar_open: settings.sidebar_open.unwrap_or(false),
            think_enabled: settings.think_enabled.unwrap_or(false),
            think_level: settings.think_level.clone().unwrap_or_else(|| "none".to_string()),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SettingsUpdate {
    #[serde(default)]
    pub turbo_enabled: Option<bool>,
    #[serde(default)]
    pub web_search_enabled: Option<bool>,
    #[serde(default)]
    pub think_enabled: Option<bool>,
    #[serde(default)]
    pub think_level: Option<String>,
    #[serde(default)]
    pub selected_model: Option<String>,
    #[serde(default)]
    pub sidebar_open: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct BatcherConfig {
    #[serde(default)]
    pub batch_interval: Option<u32>,
    #[serde(default)]
    pub immediate_first: Option<bool>,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            batch_interval: Some(8),
            immediate_first: None,
        }
    }
}

impl BatcherConfig {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImageData {
    pub filename: String,
    pub path: String,
    pub data_url: String,
}

impl ImageData {
    pub fn new(filename: impl Into<String>, path: impl Into<String>, data_url: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
            path: path.into(),
            data_url: data_url.into(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct MenuItem {
    pub label: String,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub separator: Option<bool>,
}

impl MenuItem {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            enabled: None,
            separator: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct UseMessageAutoscrollOptions {
    pub messages: Vec<Message>,
    pub is_streaming: bool,
    pub chat_id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
#[derive(Default)]
pub struct MessageAutoscrollBehavior {
    pub spacer_height: u32,
}

