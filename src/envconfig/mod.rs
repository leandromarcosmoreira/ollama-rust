use std::env;

#[allow(dead_code)]
pub struct EnvConfig {
    pub host: Host,
    pub model_paths: Vec<String>,
    pub timeout: u64,
}

#[allow(dead_code)]
pub struct Host {
    pub host: String,
    pub port: u16,
}

#[allow(dead_code)]
impl EnvConfig {
    pub fn from_env() -> Self {
        Self {
            host: Host::from_env(),
            model_paths: env::var("OLLAMA_MODELS")
                .unwrap_or_else(|_| "~/.ollama/models".to_string())
                .split(':')
                .map(String::from)
                .collect(),
            timeout: env::var("OLLAMA_TIMEOUT")
                .unwrap_or_else(|_| "600".to_string())
                .parse()
                .unwrap_or(600),
        }
    }
}

#[allow(dead_code)]
impl Host {
    pub fn from_env() -> Self {
        let host = env::var("OLLAMA_HOST").unwrap_or_else(|_| "127.0.0.1:11434".to_string());
        
        let (host, port) = if host.contains(':') {
            let parts: Vec<&str> = host.rsplitn(2, ':').collect();
            let port = parts[0].parse().unwrap_or(11434);
            let host = parts[1].to_string();
            (host, port)
        } else {
            (host, 11434)
        };
        
        Self { host, port }
    }
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

pub fn models_dir() -> std::path::PathBuf {
    let mut path = env::var("OLLAMA_MODELS")
        .unwrap_or_else(|_| "~/.ollama/models".to_string());
    
    if path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            path = path.replace("~", &home.to_string_lossy());
        }
    }
    
    std::path::PathBuf::from(path)
}
