use axum::{Router, routing::{get, post}, Json};
use serde::{Deserialize, Serialize};

use crate::app::Result;

pub struct Server {
    host: String,
    port: u16,
    router: Router,
}

impl Server {
    pub fn new() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 11434,
            router: Router::new(),
        }
    }
    
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }
    
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }
    
    pub fn routes(mut self, router: Router) -> Self {
        self.router = router;
        self
    }
    
    pub async fn run(self) -> Result<()> {
        let addr = format!("{}:{}", self.host, self.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        
        tracing::info!("Server listening on {}", addr);
        
        axum::serve(listener, self.router).await?;
        
        Ok(())
    }
}

impl Default for Server {
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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub response: String,
    pub done: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub message: Message,
    pub done: bool,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: u64,
    pub modified: String,
}

pub fn create_router() -> Router {
    Router::new()
        .route("/api/tags", get(list_models))
        .route("/api/generate", post(generate))
        .route("/api/chat", post(chat))
        .route("/v1/chat/completions", post(openai_chat))
}

async fn list_models() -> Json<Vec<ModelInfo>> {
    Json(vec![])
}

async fn generate(Json(_req): Json<GenerateRequest>) -> Json<GenerateResponse> {
    Json(GenerateResponse {
        model: "llama".to_string(),
        response: String::new(),
        done: true,
    })
}

async fn chat(Json(_req): Json<ChatRequest>) -> Json<ChatResponse> {
    Json(ChatResponse {
        model: "llama".to_string(),
        message: Message {
            role: "assistant".to_string(),
            content: String::new(),
        },
        done: true,
    })
}

async fn openai_chat(Json(_req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "choices": []
    }))
}
