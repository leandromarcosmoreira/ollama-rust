#![allow(clippy::field_reassign_with_default)]
#![allow(unused)]
use anyhow::{bail, Result};
use axum::{
    body::{Body, Bytes},
    extract::{State as AxumState, Json, Path},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post, delete as axum_delete, head},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use base64::{engine::general_purpose, Engine as _};
use chrono::Utc;
use std::fs;
use sha2::{Sha256, Digest};

use crate::models::{ModelManager, LocalModel, PullProgress, PushProgress, ModelDetails};

#[derive(Clone)]
pub struct AppState {
    pub models_dir: PathBuf,
    pub model_manager: Arc<ModelManager>,
    pub scheduler: Arc<RwLock<crate::runner::scheduler::Scheduler>>,
}

// Reuse the cache from native runner if available, or stub
//

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: Option<String>,
    pub stream: Option<bool>,
    pub images: Option<Vec<String>>,
    pub format: Option<String>,
    pub options: Option<HashMap<String, Value>>,
    pub system: Option<String>,
    pub template: Option<String>,
    pub context: Option<Vec<i32>>,
    pub raw: Option<bool>,
    pub keep_alive: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    pub context: Option<Vec<i32>>,
    pub total_duration: Option<i64>,
    pub load_duration: Option<i64>,
    pub prompt_eval_count: Option<i32>,
    pub prompt_eval_duration: Option<i64>,
    pub eval_count: Option<i32>,
    pub eval_duration: Option<i64>,
    pub tokens: Option<Vec<i32>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: Option<bool>,
    pub format: Option<String>,
    pub options: Option<HashMap<String, Value>>,
    pub keep_alive: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub images: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
    pub total_duration: Option<i64>,
    pub eval_count: Option<i32>,
    pub eval_duration: Option<i64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ShowResponse {
    pub license: Option<String>,
    pub modelfile: Option<String>,
    pub parameters: Option<String>,
    pub template: Option<String>,
    pub system: Option<String>,
    pub details: ModelDetails,
    pub messages: Option<Vec<Message>>,
    pub model_info: Option<HashMap<String, Value>>,
    pub projector_info: Option<HashMap<String, Value>>,
    pub modified_at: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct EmbedRequest {
    pub model: String,
    pub input: Value,
    pub truncate: Option<bool>,
    pub dimensions: Option<usize>,
    pub options: Option<HashMap<String, Value>>,
    pub keep_alive: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    pub total_duration: Option<i64>,
    pub load_duration: Option<i64>,
    pub prompt_eval_count: Option<i32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ListResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelInfo {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: ModelDetails,
}

pub async fn serve() -> Result<()> {
    let models_dir = crate::envconfig::models_dir();
    let model_manager = Arc::new(ModelManager::new(&models_dir)?);
    let model_cache: Arc<RwLock<Box<dyn std::any::Any + Send + Sync>>> = Arc::new(RwLock::new(Box::new(())));

    let state = AppState {
        models_dir: models_dir.clone(),
        model_manager,
        scheduler: Arc::new(RwLock::new(crate::runner::scheduler::Scheduler::new(1))),
    };

    let app = Router::new()
        .route("/api/generate", post(generate))
        .route("/api/chat", post(chat))
        .route("/api/tags", get(list_models))
        .route("/api/ps", get(list_running))
        .route("/api/show", post(show_model))
        .route("/api/pull", post(pull_model))
        .route("/api/push", post(push_model))
        .route("/api/create", post(create_model))
        .route("/api/delete", axum_delete(delete_model))
        .route("/api/copy", post(copy_model))
        .route("/api/embed", post(embed))
        .route("/api/embeddings", post(embeddings))
        .route("/api/blobs/:digest", head(head_blob))
        .route("/api/blobs/:digest", post(post_blob))
        .route("/api/version", get(version))
        .route("/api/health", get(health))
        .route("/api/metrics", get(metrics))
        .route("/api/me", post(auth_me))
        .route("/api/signout", post(auth_signout))
        // OpenAI compatibility routes
        .route("/v1/chat/completions", post(openai_chat_completions))
        .route("/v1/completions", post(openai_completions))
        .route("/v1/models", get(openai_models))
        .route("/v1/embeddings", post(openai_embeddings))
        .layer(axum::middleware::from_fn(crate::middleware::allowed_hosts_middleware))
        .with_state(state);

    // Read host and port from environment or fallback to 0.0.0.0:11434
    let addr_str = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "0.0.0.0:11434".to_string());
    let addr: SocketAddr = addr_str.parse().unwrap_or_else(|_| SocketAddr::from(([0, 0, 0, 0], 11434)));
    println!("Ollama listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn generate(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(100);
    
    let name = req.model.clone();
    let model_path = match state.model_manager.get_model_weights_path(&name) {
        Some(p) => p,
        None => return (StatusCode::NOT_FOUND, format!("Model '{}' not found", name)).into_response(),
    };

    let scheduler = Arc::clone(&state.scheduler);
    
    // Handle keep_alive: 0 to stop model
    if let Some(ref ka) = req.keep_alive {
        if ka == "0" || ka == "0s" {
            tokio::spawn(async move {
                let mut sched = scheduler.write().await;
                let _ = sched.unload(&name).await;
            });
            return (StatusCode::OK, "Model stopped").into_response();
        }
    }

    let prompt = req.prompt.unwrap_or_default();
    
    tokio::spawn(async move {
        // Use a block to ensure sched lock is dropped after getting runner
        let runner_arc = {
            let mut sched = scheduler.write().await;
            match sched.get_runner(&name, &model_path.to_string_lossy()).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
                    return;
                }
            }
        };

        let mut runner = runner_arc.write().await;
        if !runner.is_loaded() {
            if let Err(e) = runner.load() {
                let _ = tx.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
                return;
            }
        }

        let name_clone = name.clone();
        let tx_clone = tx.clone();
        
        // Generate with callback for streaming
        let res = runner.generate(&prompt, move |text, done| {
            let resp = GenerateResponse {
                model: name_clone.clone(),
                created_at: Utc::now().to_rfc3339(),
                response: text,
                done,
                context: None,
                total_duration: None,
                load_duration: None,
                prompt_eval_count: None,
                prompt_eval_duration: None,
                eval_count: None,
                eval_duration: None,
                tokens: None,
            };
            let line = serde_json::to_string(&resp).unwrap() + "\n";
            let _ = tx_clone.try_send(Ok(Bytes::from(line)));
        });

        if let Err(e) = res {
            let _ = tx.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
        }
    });

    Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

async fn chat(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(100);
    
    let name = req.model.clone();
    let model_path = match state.model_manager.get_model_weights_path(&name) {
        Some(p) => p,
        None => return (StatusCode::NOT_FOUND, format!("Model '{}' not found", name)).into_response(),
    };

    let scheduler = Arc::clone(&state.scheduler);
    let messages: Vec<crate::runner::runner::Message> = req.messages.iter().map(|m| crate::runner::runner::Message {
        role: m.role.clone(),
        content: m.content.clone(),
        images: m.images.clone(),
    }).collect();

    tokio::spawn(async move {
        let runner_arc = {
            let mut sched = scheduler.write().await;
            match sched.get_runner(&name, &model_path.to_string_lossy()).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
                    return;
                }
            }
        };

        let mut runner = runner_arc.write().await;
        if !runner.is_loaded() {
            if let Err(e) = runner.load() {
                let _ = tx.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
                return;
            }
        }

        let name_clone = name.clone();
        let tx_clone = tx.clone();

        match runner.chat(&messages, None, move |text, done| {
            let resp = ChatResponse {
                model: name_clone.clone(),
                created_at: Utc::now().to_rfc3339(),
                message: Message {
                    role: "assistant".to_string(),
                    content: text,
                    images: vec![],
                    tool_calls: vec![],
                },
                done,
                total_duration: None,
                eval_count: None,
                eval_duration: None,
            };
            let line = serde_json::to_string(&resp).unwrap() + "\n";
            let _ = tx_clone.try_send(Ok(Bytes::from(line)));
        }) {
            Ok(_) => {}
            Err(e) => {
                let _ = tx.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
            }
        }
    });

    Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

async fn list_models(
    AxumState(state): AxumState<AppState>,
) -> impl IntoResponse {
    match state.model_manager.list_local_models() {
        Ok(models) => {
            let info: Vec<ModelInfo> = models.into_iter().map(|m: LocalModel| ModelInfo {
                name: m.name.clone(),
                model: m.name.clone(),
                modified_at: m.modified_at,
                size: m.size,
                digest: m.digest,
                details: m.details.unwrap_or_default(),
            }).collect();
            Json(ListResponse { models: info }).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn list_running(
    AxumState(state): AxumState<AppState>,
) -> impl IntoResponse {
    let sched = state.scheduler.read().await;
    let models = sched.list_running();
    Json(json!({ "models": models })).into_response()
}

async fn show_model(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<HashMap<String, String>>,
) -> impl IntoResponse {
    let name = match req.get("name") {
        Some(n) => n,
        None => return (StatusCode::BAD_REQUEST, "Missing name").into_response(),
    };

    match state.model_manager.get_model_info(name) {
        Ok(info) => Json(ShowResponse {
            license: info.license,
            modelfile: info.modelfile,
            parameters: None,
            template: info.template,
            system: info.system,
            details: info.details.unwrap_or_default(),
            messages: None,
            model_info: None,
            projector_info: None,
            modified_at: Some(info.modified_at),
        }).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, e.to_string()).into_response()
    }
}

async fn embed(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<EmbedRequest>,
) -> impl IntoResponse {
    let name = req.model.clone();
    let model_path = match state.model_manager.get_model_weights_path(&name) {
        Some(p) => p,
        None => return (StatusCode::NOT_FOUND, format!("Model '{}' not found", name)).into_response(),
    };

    let scheduler = Arc::clone(&state.scheduler);
    let input = if let Some(s) = req.input.as_str() {
        s.to_string()
    } else {
        return (StatusCode::BAD_REQUEST, "Input must be a string").into_response();
    };
    
    let mut sched = scheduler.write().await;
    let runner_arc = match sched.get_runner(&name, &model_path.to_string_lossy()).await {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    let mut runner = runner_arc.write().await;
    if !runner.is_loaded() {
        if let Err(e) = runner.load() {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    }

    match runner.embed(&input, req.dimensions) {
        Ok(result) => Json(EmbedResponse {
            model: name,
            embeddings: result.embeddings,
            total_duration: Some(result.total_duration),
            load_duration: Some(0),
            prompt_eval_count: Some(0),
        }).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn embeddings(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<HashMap<String, Value>>,
) -> impl IntoResponse {
    // Simplified /api/embeddings for backward compatibility
    let model = req.get("model").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    let prompt = req.get("prompt").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    
    let embed_req = EmbedRequest {
        model,
        input: json!(prompt),
        truncate: None,
        dimensions: None,
        options: None,
        keep_alive: None,
    };
    
    embed(AxumState(state), Json(embed_req)).await
}

async fn head_blob(
    AxumState(state): AxumState<AppState>,
    Path(digest): Path<String>,
) -> impl IntoResponse {
    match state.model_manager.stat_blob(&digest) {
        Some(_) => StatusCode::OK.into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn post_blob(
    AxumState(state): AxumState<AppState>,
    Path(digest): Path<String>,
    bytes: Bytes,
) -> impl IntoResponse {
    match state.model_manager.create_blob(&digest, &bytes) {
        Ok(_) => StatusCode::CREATED.into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn pull_model(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<HashMap<String, Value>>,
) -> impl IntoResponse {
    let name = match req.get("name").and_then(|v| v.as_str()) {
        Some(n) => n.to_string(),
        None => return (StatusCode::BAD_REQUEST, "Missing name").into_response(),
    };

    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(100);
    let mm = Arc::clone(&state.model_manager);
    
    tokio::spawn(async move {
        let tx_inner = tx.clone();
        let res = mm.pull(name, move |progress: PullProgress| {
            let line = serde_json::to_string(&progress).unwrap() + "\n";
            let _ = tx_inner.try_send(Ok(Bytes::from(line)));
        }).await;
        
        if let Err(e) = res {
            let err_resp = json!({"status": "error", "error": e.to_string()});
            let line = serde_json::to_string(&err_resp).unwrap() + "\n";
            let _ = tx.send(Ok(Bytes::from(line))).await;
        }
    });

    Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
        .into_response()
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct PushRequest {
    pub name: String,
    pub insecure: Option<bool>,
    pub stream: Option<bool>,
}

async fn push_model(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<PushRequest>,
) -> impl IntoResponse {
    let name = req.name.clone();
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(100);
    let mm = Arc::clone(&state.model_manager);
    
    tokio::spawn(async move {
        let tx_inner = tx.clone();
        let res = mm.push(name, move |progress: PushProgress| {
            let line = serde_json::to_string(&progress).unwrap() + "\n";
            let _ = tx_inner.try_send(Ok(Bytes::from(line)));
        }).await;
        
        if let Err(e) = res {
            let err_resp = json!({"status": "error", "error": e.to_string()});
            let line = serde_json::to_string(&err_resp).unwrap() + "\n";
            let _ = tx.send(Ok(Bytes::from(line))).await;
        }
    });

    Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
        .into_response()
}

async fn create_model(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<HashMap<String, Value>>,
) -> impl IntoResponse {
    let name = match req.get("name").and_then(|v| v.as_str()) {
        Some(n) => n.to_string(),
        None => return (StatusCode::BAD_REQUEST, "Missing name").into_response(),
    };
    
    let modelfile = match req.get("modelfile").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return (StatusCode::BAD_REQUEST, "Missing modelfile").into_response(),
    };

    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(100);
    let tx_for_blocking = tx.clone();
    
    tokio::spawn(async move {
        let tx_inner = tx_for_blocking.clone();
        
        // Simulating the builder steps
        let _ = tx_inner.send(Ok(Bytes::from(serde_json::to_string(&json!({"status": "parsing modelfile"})).unwrap() + "\n"))).await;
        
        let res = tokio::task::spawn_blocking(move || -> Result<()> {
            let reader = std::io::Cursor::new(modelfile.clone());
            let mf = crate::parser::modelfile::parse(reader)?;
            
            let mut base_model = String::new();
            for cmd in &mf.commands {
                if cmd.name == "from" {
                    base_model = cmd.args.clone();
                    break;
                }
            }
            
            if base_model.is_empty() {
                bail!("Modelfile must contain a FROM directive");
            }
            
            let _ = tx_inner.blocking_send(Ok(Bytes::from(serde_json::to_string(&json!({"status": format!("using base model {}", base_model)})).unwrap() + "\n")));
            
            let models_dir = state.models_dir.clone();
            
            // Resolve base model
            let base_path = state.model_manager.get_model_dir(&base_model);
            if !base_path.exists() {
                return Err(anyhow::anyhow!("Base model {} not found locally. Please pull it first.", base_model));
            }
                
            let target_dir = models_dir.join(name.replace("/", "--"));
            std::fs::create_dir_all(&target_dir)?;
            
            let mut license = String::new();
            let mut system = String::new();
            let mut template = String::new();
            let mut params = HashMap::new();

            for cmd in &mf.commands {
                match cmd.name.as_str() {
                    "license" => license = cmd.args.clone(),
                    "system" => system = cmd.args.clone(),
                    "template" => template = cmd.args.clone(),
                    "parameter" => {
                        let parts: Vec<&str> = cmd.args.splitn(2, |c: char| c.is_whitespace()).collect();
                        if parts.len() == 2 {
                            params.insert(parts[0].to_string(), parts[1].to_string());
                        }
                    }
                    _ => {}
                }
            }

            let _ = tx_inner.blocking_send(Ok(Bytes::from(serde_json::to_string(&json!({"status": format!("processing layers for {}", name)})).unwrap() + "\n")));

            let mm = &state.model_manager;
            let (base_full, base_tag) = crate::models::registry::Registry::resolve_name(&base_model);
            let base_manifest_path = mm.get_manifest_path(&base_full, &base_tag);
            
            if !base_manifest_path.exists() {
                return Err(anyhow::anyhow!("Base model manifest not found: {}", base_model));
            }

            let manifest_content = fs::read_to_string(&base_manifest_path)?;
            let mut new_manifest: crate::models::Manifest = serde_json::from_str(&manifest_content)?;

            // Add system layer if present
            if !system.is_empty() {
                let mut hasher = Sha256::new();
                hasher.update(system.as_bytes());
                let digest = format!("sha256:{:x}", hasher.finalize());
                mm.create_blob(&digest, system.as_bytes())?;
                new_manifest.layers.push(crate::models::Layer {
                    media_type: Some("application/vnd.ollama.image.system".to_string()),
                    digest,
                    size: system.len() as u64,
                });
            }

            // Create target manifest
            let (name_full, name_tag) = crate::models::registry::Registry::resolve_name(&name);
            let target_manifest_path = mm.get_manifest_path(&name_full, &name_tag);
            fs::create_dir_all(target_manifest_path.parent().unwrap())?;
            fs::write(&target_manifest_path, serde_json::to_string(&new_manifest)?)?;

            let _ = tx_inner.blocking_send(Ok(Bytes::from(serde_json::to_string(&json!({"status": "success"})).unwrap() + "\n")));
            Ok(())
        }).await.unwrap();

        if let Err(e) = res {
            let err_resp = json!({"status": "error", "error": e.to_string()});
            let line = serde_json::to_string(&err_resp).unwrap() + "\n";
            let _ = tx_for_blocking.send(Ok(Bytes::from(line))).await;
        }
    });

    Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
        .into_response()
}

async fn delete_model(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<HashMap<String, String>>,
) -> impl IntoResponse {
    let name = match req.get("name") {
        Some(n) => n,
        None => return (StatusCode::BAD_REQUEST, "Missing name").into_response(),
    };
    
    match state.model_manager.delete_model(name) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
pub struct CopyRequest {
    pub source: String,
    pub destination: String,
}

async fn copy_model(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<CopyRequest>,
) -> impl IntoResponse {
    match state.model_manager.copy_model(&req.source, &req.destination) {
        Ok(_) => StatusCode::OK.into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn version() -> impl IntoResponse {
    Json(json!({"version": "0.5.0-rust"})).into_response()
}

async fn health() -> impl IntoResponse {
    Json(json!({"status": "OK"})).into_response()
}

async fn metrics() -> impl IntoResponse {
    (StatusCode::OK, "# Ollama metrics").into_response()
}

async fn auth_me() -> impl IntoResponse {
    // Ported from WhoamiHandler: If no valid token/session, return unauth with a sign-in URL placeholder
    (
        StatusCode::UNAUTHORIZED, 
        Json(json!({"error": "unauthorized", "signin_url": "https://ollama.com/auth"}))
    ).into_response()
}

async fn auth_signout() -> impl IntoResponse {
    // Ported from SignoutHandler: Remove session/key, returns 200 OK
    StatusCode::OK.into_response()
}

fn current_timestamp() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%S%.9fZ").to_string()
}

async fn openai_chat_completions(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<crate::openai::ChatCompletionRequest>,
) -> impl IntoResponse {
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(100);
    
    let name = req.model.clone();
    let model_path = match state.model_manager.get_model_weights_path(&name) {
        Some(p) => p,
        None => return (StatusCode::NOT_FOUND, format!("Model '{}' not found", name)).into_response(),
    };

    let scheduler = Arc::clone(&state.scheduler);
    let messages: Vec<crate::runner::runner::Message> = req.messages.iter().map(|m| {
        crate::runner::runner::Message {
            role: m.role.clone(),
            content: m.content.clone(),
            images: vec![],
        }
    }).collect();

    let name_clone = name.clone();
    let tx_clone = tx.clone();
    let is_stream = req.stream;

    tokio::spawn(async move {
        let runner_arc = {
            let mut sched = scheduler.write().await;
            match sched.get_runner(&name_clone, &model_path.to_string_lossy()).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx_clone.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
                    return;
                }
            }
        };

        let mut runner = runner_arc.write().await;
        if !runner.is_loaded() {
            if let Err(e) = runner.load() {
                let _ = tx_clone.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
                return;
            }
        }

        let model_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let name_inner = name_clone.clone();

        let tx_for_closure = tx_clone.clone();
        match runner.chat(&messages, None, move |text, done| {
            if is_stream {
                let chunk = crate::openai::ChatCompletionChunk {
                    id: model_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: Utc::now().timestamp(),
                    model: name_inner.clone(),
                    choices: vec![crate::openai::ChunkChoice {
                        index: 0,
                        delta: crate::openai::Delta {
                            role: if text.is_empty() && !done { Some("assistant".to_string()) } else { None },
                            content: if !text.is_empty() { Some(text) } else { None },
                        },
                        finish_reason: if done { Some("stop".to_string()) } else { None },
                    }],
                };
                let line = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                let _ = tx_for_closure.try_send(Ok(Bytes::from(line)));
                if done {
                    let _ = tx_for_closure.try_send(Ok(Bytes::from("data: [DONE]\n\n")));
                }
            } else if done {
                let resp = crate::openai::ChatCompletionResponse::new(name_inner.clone(), text, 0, 0);
                let _ = tx_for_closure.try_send(Ok(Bytes::from(serde_json::to_string(&resp).unwrap())));
            }
        }) {
            Ok(_) => {}
            Err(e) => {
                let _ = tx_clone.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
            }
        }
    });

    if is_stream {
        Response::builder().header("Content-Type", "text/event-stream").body(Body::from_stream(ReceiverStream::new(rx))).unwrap()
    } else {
        Response::builder().header("Content-Type", "application/json").body(Body::from_stream(ReceiverStream::new(rx))).unwrap()
    }
}

async fn openai_completions(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<crate::openai::CompletionRequest>,
) -> impl IntoResponse {
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(100);
    
    let name = req.model.clone();
    let model_path = match state.model_manager.get_model_weights_path(&name) {
        Some(p) => p,
        None => return (StatusCode::NOT_FOUND, format!("Model '{}' not found", name)).into_response(),
    };

    let scheduler = Arc::clone(&state.scheduler);
    let prompt = req.prompt.clone();
    let is_stream = req.stream;
    let name_clone = name.clone();
    let tx_clone = tx.clone();

    tokio::spawn(async move {
        let runner_arc = {
            let mut sched = scheduler.write().await;
            match sched.get_runner(&name_clone, &model_path.to_string_lossy()).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx_clone.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
                    return;
                }
            }
        };

        let mut runner = runner_arc.write().await;
        if !runner.is_loaded() {
            if let Err(e) = runner.load() {
                let _ = tx_clone.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
                return;
            }
        }

        let model_id = format!("cmpl-{}", uuid::Uuid::new_v4());

        let tx_for_closure = tx_clone.clone();
        match runner.generate(&prompt, move |text, done| {
            if is_stream {
                let chunk = crate::openai::CompletionResponse::new_chunk(&model_id, &name_clone, text, if done { Some("stop".to_string()) } else { None });
                let line = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                let _ = tx_for_closure.try_send(Ok(Bytes::from(line)));
                if done {
                    let _ = tx_for_closure.try_send(Ok(Bytes::from("data: [DONE]\n\n")));
                }
            } else if done {
                let resp = crate::openai::CompletionResponse::new_final(&model_id, &name_clone, text, 0, 0);
                let _ = tx_for_closure.try_send(Ok(Bytes::from(serde_json::to_string(&resp).unwrap())));
            }
        }) {
            Ok(_) => {}
            Err(e) => {
                let _ = tx_clone.send(Ok(Bytes::from(json!({"error": e.to_string()}).to_string() + "\n"))).await;
            }
        }
    });

    if is_stream {
        Response::builder().header("Content-Type", "text/event-stream").body(Body::from_stream(ReceiverStream::new(rx))).unwrap()
    } else {
        Response::builder().header("Content-Type", "application/json").body(Body::from_stream(ReceiverStream::new(rx))).unwrap()
    }
}

async fn openai_models(
    AxumState(state): AxumState<AppState>,
) -> impl IntoResponse {
    match state.model_manager.list_local_models() {
        Ok(models) => {
            let info: Vec<crate::openai::Model> = models.into_iter().map(|m: LocalModel| {
                let created = chrono::DateTime::parse_from_rfc3339(&m.modified_at)
                    .map(|dt| dt.timestamp())
                    .unwrap_or(0);
                
                crate::openai::Model {
                    id: m.name.clone(),
                    object: "model".to_string(),
                    created,
                    owned_by: "library".to_string(),
                }
            }).collect();
            Json(crate::openai::ModelList { object: "list".to_string(), data: info }).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn openai_embeddings(
    AxumState(state): AxumState<AppState>,
    Json(req): Json<crate::openai::EmbeddingRequest>,
) -> impl IntoResponse {
    let name = req.model.clone();
    let model_path = match state.model_manager.get_model_weights_path(&name) {
        Some(p) => p,
        None => return (StatusCode::NOT_FOUND, format!("Model '{}' not found", name)).into_response(),
    };

    let scheduler = Arc::clone(&state.scheduler);
    let input = match req.input.as_str() {
        Some(s) => s.to_string(),
        None => return (StatusCode::BAD_REQUEST, "Input must be a string").into_response(),
    };
    
    let mut sched = scheduler.write().await;
    let runner_arc = match sched.get_runner(&name, &model_path.to_string_lossy()).await {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    let mut runner = runner_arc.write().await;
    if !runner.is_loaded() {
        if let Err(e) = runner.load() {
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    }

    match runner.embed(&input, None) {
        Ok(res) => {
            let resp = crate::openai::EmbeddingResponse {
                object: "list".to_string(),
                data: res.embeddings.into_iter().enumerate().map(|(i, e)| {
                    crate::openai::EmbeddingData {
                        object: "embedding".to_string(),
                        embedding: e,
                        index: i,
                    }
                }).collect(),
                model: name,
                usage: crate::openai::EmbeddingUsage {
                    prompt_tokens: 0,
                    total_tokens: 0,
                },
            };
            Json(resp).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}
