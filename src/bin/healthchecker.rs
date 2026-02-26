use anyhow::{anyhow, Result};
use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;
use std::fs;
use std::process::Command;
use std::thread;

const MODELS_DIR: &str = "/home/ollama/.ollama/models";
const CHECK_INTERVAL_SECS: u64 = 60;

fn main() -> Result<()> {
    println!("=== Ollama Healthchecker Started ===");
    println!("Models directory: {}", MODELS_DIR);
    println!("Check interval: {}s", CHECK_INTERVAL_SECS);
    
    let ollama_host = std::env::var("OLLAMA_HOST")
        .unwrap_or_else(|_| "http://localhost:11434".to_string());
    println!("Ollama host: {}", ollama_host);
    
    let models_list = std::env::var("OLLAMA_MODELS_LIST")
        .unwrap_or_default();
    
    let mut last_models: HashSet<String> = models_list
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    
    println!("Models to maintain: {:?}", last_models);
    
    if !last_models.is_empty() {
        sync_models(&ollama_host, &last_models)?;
    }
    
    let mut last_check = std::time::Instant::now();
    
    loop {
        if last_check.elapsed() >= Duration::from_secs(CHECK_INTERVAL_SECS) {
            let models_list = std::env::var("OLLAMA_MODELS_LIST")
                .unwrap_or_default();
            
            let current_models: HashSet<String> = models_list
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            
            if current_models != last_models {
                println!("[Healthchecker] Model list changed!");
                println!("  Previous: {:?}", last_models);
                println!("  Current:  {:?}", current_models);
                
                if let Err(e) = sync_models(&ollama_host, &current_models) {
                    eprintln!("[Healthchecker] Sync error: {}", e);
                }
                
                last_models = current_models;
            }
            
            last_check = std::time::Instant::now();
        }
        
        thread::sleep(Duration::from_millis(500));
    }
}

fn get_local_models_from_disk() -> HashSet<String> {
    let mut models = HashSet::new();
    let models_path = Path::new(MODELS_DIR);
    
    if !models_path.exists() {
        return models;
    }
    
    if let Ok(entries) = fs::read_dir(models_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let manifest = path.join("manifest.json");
                if manifest.exists() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        let model_name = name.replace("--", "/");
                        models.insert(model_name);
                    }
                }
            }
        }
    }
    
    models
}

fn get_local_models_from_api(ollama_host: &str) -> HashSet<String> {
    let url = format!("{}/api/tags", ollama_host);
    
    let output = match Command::new("curl")
        .args(["-s", "-f", "--connect-timeout", "2", &url])
        .output() 
    {
        Ok(o) => o,
        Err(_) => return HashSet::new(),
    };
    
    if !output.status.success() {
        return HashSet::new();
    }
    
    let response: serde_json::Value = match serde_json::from_slice(&output.stdout) {
        Ok(r) => r,
        Err(_) => return HashSet::new(),
    };
    
    let mut models = HashSet::new();
    
    if let Some(models_array) = response.get("models").and_then(|m| m.as_array()) {
        for model in models_array {
            if let Some(name) = model.get("name").and_then(|n| n.as_str()) {
                models.insert(name.to_string());
            }
        }
    }
    
    models
}

fn model_exists_on_disk(model_name: &str) -> bool {
    let dir_name = model_name.replace("/", "--");
    let model_path = Path::new(MODELS_DIR).join(&dir_name);
    let manifest = model_path.join("manifest.json");
    
    if !manifest.exists() {
        return false;
    }
    
    if let Ok(content) = fs::read_to_string(&manifest) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(layers) = json.get("layers").and_then(|l| l.as_array()) {
                return !layers.is_empty();
            }
        }
    }
    
    false
}

fn sync_models(ollama_host: &str, desired: &HashSet<String>) -> Result<()> {
    let local_disk = get_local_models_from_disk();
    let local_api = get_local_models_from_api(ollama_host);
    
    let local: HashSet<_> = local_disk.union(&local_api).cloned().collect();
    
    println!("[Healthchecker] === Sync Status ===");
    println!("[Healthchecker] Models on disk: {:?}", local_disk);
    println!("[Healthchecker] Models from API: {:?}", local_api);
    println!("[Healthchecker] Desired models: {:?}", desired);
    
    let mut to_pull = Vec::new();
    let mut to_remove = Vec::new();
    
    for model in desired.difference(&local) {
        if model_exists_on_disk(model) {
            println!("[Healthchecker] Model {} on disk, API sync needed", model);
        } else {
            to_pull.push(model);
        }
    }
    
    for model in local.difference(desired) {
        to_remove.push(model);
    }
    
    if to_pull.is_empty() && to_remove.is_empty() {
        println!("[Healthchecker] ✓ All models in sync");
        return Ok(());
    }
    
    for model in &to_pull {
        println!("[Healthchecker] ↓ Pulling: {}", model);
        match pull_model(ollama_host, model) {
            Ok(_) => println!("[Healthchecker] ✓ Pulled: {}", model),
            Err(e) => eprintln!("[Healthchecker] ✗ Failed to pull {}: {}", model, e),
        }
    }
    
    for model in &to_remove {
        println!("[Healthchecker] ↑ Removing: {}", model);
        match delete_model(ollama_host, model) {
            Ok(_) => println!("[Healthchecker] ✓ Removed: {}", model),
            Err(e) => eprintln!("[Healthchecker] ✗ Failed to remove {}: {}", model, e),
        }
    }
    
    println!("[Healthchecker] === Sync Complete ===");
    Ok(())
}

fn pull_model(ollama_host: &str, model: &str) -> Result<()> {
    let url = format!("{}/api/pull", ollama_host);
    let payload = serde_json::json!({"name": model, "stream": false}).to_string();
    
    let status = Command::new("curl")
        .args(["-s", "-f", "-X", "POST"])
        .args(["-H", "Content-Type: application/json"])
        .args(["-d", &payload])
        .arg(&url)
        .status()?;
    
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("curl failed for pull {}", model))
    }
}

fn delete_model(ollama_host: &str, model: &str) -> Result<()> {
    let url = format!("{}/api/delete", ollama_host);
    let payload = serde_json::json!({"name": model}).to_string();
    
    let status = Command::new("curl")
        .args(["-s", "-f", "-X", "DELETE"])
        .args(["-H", "Content-Type: application/json"])
        .args(["-d", &payload])
        .arg(&url)
        .status()?;
    
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("curl failed for delete {}", model))
    }
}
