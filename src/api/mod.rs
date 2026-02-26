use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub mod types;

#[allow(dead_code)]
pub struct Client {
    base_url: String,
    client: reqwest::Client,
}

#[allow(dead_code)]
impl Client {
    pub fn from_env() -> Result<Self> {
        let mut host = std::env::var("OLLAMA_HOST")
            .unwrap_or_else(|_| "http://localhost:11434".to_string());
        
        if !host.starts_with("http://") && !host.starts_with("https://") {
            host = format!("http://{}", host);
        }

        // If it's just http://0.0.0.0, append the default port
        if host.matches(':').count() < 2 { // only http:// and no port
            host = format!("{}:11434", host);
        }
        
        Ok(Self {
            base_url: host,
            client: reqwest::Client::new(),
        })
    }
    
    pub async fn generate(&self, request: &Value) -> Result<GenerateResponse> {
        let url = format!("{}/api/generate", self.base_url);
        let response = self.client.post(&url)
            .json(request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            bail!("Generate failed: {}", response.status());
        }

        Ok(response.json().await?)
    }
    
    pub async fn chat(&self, request: &Value) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);
        let response = self.client.post(&url)
            .json(request)
            .send()
            .await?;
        
        Ok(response.text().await?)
    }
    
    pub async fn embed(&self, request: &Value) -> Result<EmbedResponse> {
        let url = format!("{}/api/embed", self.base_url);
        let response = self.client.post(&url)
            .json(request)
            .send()
            .await?;
        
        Ok(response.json().await?)
    }
    
    pub async fn show(&self, model: &str) -> Result<ShowResponse> {
        let url = format!("{}/api/show", self.base_url);
        let request = serde_json::json!({ "name": model });
        let response = self.client.post(&url)
            .json(&request)
            .send()
            .await?;
        
        Ok(response.json().await?)
    }
    
    pub async fn list(&self) -> Result<ListResponse> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self.client.get(&url).send().await?;
        
        Ok(response.json().await?)
    }
    
    pub async fn list_running(&self) -> Result<Vec<RunningModel>> {
        let url = format!("{}/api/ps", self.base_url);
        let response = self.client.get(&url).send().await?;
        
        #[derive(Deserialize)]
        struct PsResponse {
            models: Vec<RunningModel>,
        }
        
        let resp: PsResponse = response.json().await?;
        Ok(resp.models)
    }
    
    pub async fn generate_stream(&self, request: &Value, mut callback: impl FnMut(Value)) -> Result<()> {
        let url = format!("{}/api/generate", self.base_url);
        let response = self.client.post(&url)
            .json(request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            bail!("Generate failed: {}", response.status());
        }

        let mut stream = response.bytes_stream();
        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let text = String::from_utf8_lossy(&chunk);
            for l in text.split('\n') {
                let l = l.trim();
                if l.is_empty() { continue; }
                if let Ok(json) = serde_json::from_str::<Value>(l) {
                    callback(json);
                }
            }
        }
        Ok(())
    }

    pub async fn pull(&self, request: &Value, mut progress: impl FnMut(Value)) -> Result<()> {
        let url = format!("{}/api/pull", self.base_url);
        let response = self.client.post(&url)
            .json(request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            bail!("Pull failed: {}", response.status());
        }

        let mut stream = response.bytes_stream();
        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let text = String::from_utf8_lossy(&chunk);
            for l in text.split('\n') {
                let l = l.trim();
                if l.is_empty() { continue; }
                if let Ok(json) = serde_json::from_str::<Value>(l) {
                    progress(json);
                }
            }
        }
        Ok(())
    }

    pub async fn push(&self, request: &Value, mut progress: impl FnMut(Value)) -> Result<()> {
        let url = format!("{}/api/push", self.base_url);
        let response = self.client.post(&url)
            .json(request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            bail!("Push failed: {}", response.status());
        }

        let mut stream = response.bytes_stream();
        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let text = String::from_utf8_lossy(&chunk);
            for l in text.split('\n') {
                let l = l.trim();
                if l.is_empty() { continue; }
                if let Ok(json) = serde_json::from_str::<Value>(l) {
                    progress(json);
                }
            }
        }
        Ok(())
    }

    pub async fn create(&self, request: &Value, mut progress: impl FnMut(Value)) -> Result<()> {
        let url = format!("{}/api/create", self.base_url);
        let response = self.client.post(&url)
            .json(request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            bail!("Create failed: {}", response.status());
        }

        let mut stream = response.bytes_stream();
        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let line = String::from_utf8_lossy(&chunk);
            for l in line.lines() {
                if let Ok(json) = serde_json::from_str::<Value>(l) {
                    progress(json);
                }
            }
        }
        Ok(())
    }

    pub async fn copy(&self, request: &Value) -> Result<()> {
        let url = format!("{}/api/copy", self.base_url);
        self.client.post(&url)
            .json(request)
            .send()
            .await?;
        
        Ok(())
    }
    
    pub async fn delete(&self, model: &str) -> Result<()> {
        let url = format!("{}/api/delete", self.base_url);
        self.client.delete(&url)
            .json(&serde_json::json!({ "name": model }))
            .send()
            .await?;
        
        Ok(())
    }
    
    pub async fn stop(&self, model: &str) -> Result<()> {
        let url = format!("{}/api/generate", self.base_url);
        self.client.post(&url)
            .json(&serde_json::json!({
                "model": model,
                "keep_alive": 0
            }))
            .send()
            .await?;
        
        Ok(())
    }
    
    pub async fn version(&self) -> Result<String> {
        let url = format!("{}/api/version", self.base_url);
        let response = self.client.get(&url).send().await?;
        
        #[derive(Deserialize)]
        struct VersionResponse {
            version: String,
        }
        
        let resp: VersionResponse = response.json().await?;
        Ok(resp.version)
    }
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct GenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    pub context: Option<Vec<i64>>,
    pub total_duration: Option<i64>,
    pub load_duration: Option<i64>,
    pub prompt_eval_count: Option<i32>,
    pub prompt_eval_duration: Option<i64>,
    pub eval_count: Option<i32>,
    pub eval_duration: Option<i64>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct ShowResponse {
    pub model: String,
    pub size: u64,
    pub modified_at: String,
    pub digest: Option<String>,
    pub details: Option<ModelDetails>,
    pub license: Option<String>,
    pub system: Option<String>,
    pub parameters: Option<String>,
    pub template: Option<String>,
    pub modelfile: Option<String>,
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub projector_info: HashMap<String, Value>,
    #[serde(default)]
    pub model_info: HashMap<String, Value>,
    #[serde(default)]
    pub messages: Vec<Message>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct ModelDetails {
    pub parent_model: Option<String>,
    pub format: Option<String>,
    pub family: Option<String>,
    pub families: Option<Vec<String>>,
    pub parameter_size: Option<String>,
    pub quantization_level: Option<String>,
}

#[derive(Deserialize, Serialize, Default, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Vec<String>,
}

#[derive(Serialize, Deserialize, Default)]
#[allow(dead_code)]
pub struct CreateRequest {
    pub name: String,
    pub from: String,
    pub modelfile: Option<String>,
    pub stream: Option<bool>,
    pub path: Option<String>,
    pub options: Option<HashMap<String, Value>>,
    pub template: Option<String>,
    pub system: Option<String>,
    pub license: Option<Vec<String>>,
    pub adapters: HashMap<String, String>,
    pub messages: Option<Vec<Message>>,
    #[serde(default)]
    pub files: HashMap<String, String>,
}

#[derive(Deserialize)]
pub struct ListResponse {
    pub models: Vec<ModelInfo>,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(dead_code)]
pub struct ModelInfo {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub digest: String,
    pub details: Option<ModelDetails>,
    #[serde(default)]
    pub remote_model: String,
    #[serde(default)]
    pub remote_host: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
#[allow(dead_code)]
pub struct RunningModel {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub size_vram: u64,
    pub digest: String,
    pub expires_at: String,
    #[serde(default)]
    pub context_length: u32,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct EmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    pub total_duration: Option<i64>,
    pub load_duration: Option<i64>,
    pub prompt_eval_count: Option<i32>,
}
