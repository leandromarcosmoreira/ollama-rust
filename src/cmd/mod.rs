use anyhow::{Context, Result};
use crate::api::Client;
use crate::format::{human_bytes, human_time};
use serde_json::json;
use std::io::{self, Write};

pub async fn run(model: &str, args: Vec<String>) -> Result<()> {
    let client = Client::from_env()?;
    
    let prompt = args.join(" ");
    
    if !prompt.is_empty() {
        generate(&client, model, &prompt, false, None).await?;
    } else {
        interactive_run(&client, model).await?;
    }
    
    Ok(())
}

async fn interactive_run(client: &Client, model: &str) -> Result<()> {
    println!(">>> Running model {} in interactive mode", model);
    println!("Type /help for commands, /exit to quit");
    
    // Load model first
    if let Err(e) = load_model(client, model).await {
        eprintln!("Warning: Could not load model: {}", e);
    }
    
    let mut context: Option<Vec<i64>> = None;
    
    loop {
        print!("\n>>> ");
        io::stdout().flush()?;
        
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        
        if line.starts_with('/') {
            if handle_command(client, line, model).await? {
                break; // Exit interactive mode if command returns true
            }
            continue;
        }
        
        match generate(client, model, line, true, context.clone()).await {
            Ok(new_context) => {
                context = new_context;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }
    Ok(())
}

async fn handle_command(client: &Client, cmd: &str, model: &str) -> Result<bool> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() { return Ok(false); }

    match parts[0] {
        "/help" | "/?" => {
            println!("Commands:");
            println!("  /exit, /quit         Exit the interactive mode");
            println!("  /clear              Clear the screen");
            println!("  /list               List available models");
            println!("  /pull <model>       Pull a model");
            println!("  /show info          Show model info");
            println!("  /show license       Show model license");
            println!("  /show system        Show model system prompt");
            println!("  /show template      Show model template");
            println!("  /set <param> <val>  Set a parameter");
            println!("  /?                  Show this help");
        }
        "/exit" | "/quit" => {
            return Ok(true); // Signal to exit interactive mode
        }
        "/clear" => {
            print!("\x1B[2J\x1B[1J");
            io::stdout().flush()?;
        }
        "/list" => {
            list_models(client).await?;
        }
        "/pull" => {
            if parts.len() < 2 {
                println!("Usage: /pull <model>");
            } else {
                pull_model(client, parts[1]).await?;
            }
        }
        "/show" => {
            let subcommand = if parts.len() < 2 { "info" } else { parts[1] };
            let info = client.show(model).await?;
            match subcommand {
                "info" => {
                    show_model(client, model).await?;
                }
                "license" => {
                    println!("\nLicense:\n{}", info.license.unwrap_or_else(|| "No license provided".to_string()));
                }
                "system" => {
                    println!("\nSystem:\n{}", info.system.unwrap_or_else(|| "No system prompt provided".to_string()));
                }
                "template" => {
                    println!("\nTemplate:\n{}", info.template.unwrap_or_else(|| "No template provided".to_string()));
                }
                _ => {
                    println!("Unknown /show subcommand: {}", subcommand);
                }
            }
        }
        "/set" => {
            if parts.len() < 3 {
                println!("Usage: /set <parameter> <value>");
            } else {
                println!("Setting {} to {} (Note: parameters not yet persisted in this session)", parts[1], parts[2]);
            }
        }
        _ => {
            println!("Unknown command: {}", parts[0]);
            println!("Type /help for available commands");
        }
    }
    Ok(false) // Do not exit interactive mode
}

async fn generate(client: &Client, model: &str, prompt: &str, interactive: bool, context: Option<Vec<i64>>) -> Result<Option<Vec<i64>>> {
    let mut request = json!({
        "model": model,
        "prompt": prompt,
        "stream": true,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
        }
    });

    if let Some(ctx) = context {
        request["context"] = json!(ctx);
    }
    
    let mut final_context = None;
    
    client.generate_stream(&request, |json| {
        if let Some(resp) = json.get("response").and_then(|v| v.as_str()) {
            print!("{}", resp);
            let _ = io::stdout().flush();
        }
        
        if let Some(done) = json.get("done").and_then(|v| v.as_bool()) {
            if done {
                if let Some(ctx) = json.get("context").and_then(|v| v.as_array()) {
                    let ctx_vec: Vec<i64> = ctx.iter().filter_map(|v| v.as_i64()).collect();
                    final_context = Some(ctx_vec);
                }
            }
        }
    }).await?;
    
    if interactive {
        println!();
    }
    
    Ok(final_context)
}

async fn load_model(client: &Client, model: &str) -> Result<()> {
    // Just try to show model info to check if it exists
    client.show(model).await?;
    println!("Model {} loaded", model);
    
    Ok(())
}

async fn list_models(client: &Client) -> Result<()> {
    let models = client.list().await?;
    
    println!("\n{:<40} {:<12} {:<12} MODIFIED", "NAME", "ID", "SIZE");
    println!("{}", "-".repeat(80));
    
    for m in models.models {
        let size = if m.remote_model.is_empty() {
            human_bytes(m.size)
        } else {
            "-".to_string()
        };
        let modified_timestamp = chrono::DateTime::parse_from_rfc3339(&m.modified_at).map(|dt| dt.timestamp()).unwrap_or(0);
        let modified = human_time(modified_timestamp, "Never");
        
        println!("{:<40} {:<12} {:<12} {}", m.name, &m.digest[..12.min(m.digest.len())], size, modified);
    }
    
    Ok(())
}

async fn pull_model(client: &Client, model: &str) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
    
    let mp = MultiProgress::new();
    let mut bars = std::collections::HashMap::new();
    
    let request = json!({
        "name": model,
        "stream": true
    });
    
    client.pull(&request, |status| {
        if let Some(digest) = status.get("digest").and_then(|v| v.as_str()) {
             let bar = bars.entry(digest.to_string()).or_insert_with(|| {
                let b = mp.add(ProgressBar::new(0));
                b.set_style(ProgressStyle::default_bar()
                    .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"));
                b.set_message(format!("pulling {}", &digest[7..19]));
                b
            });
            
            if let Some(total) = status.get("total").and_then(|v| v.as_u64()) {
                bar.set_length(total);
            }
            if let Some(completed) = status.get("completed").and_then(|v| v.as_u64()) {
                bar.set_position(completed);
            }
        } else if let Some(msg) = status.get("status").and_then(|v| v.as_str()) {
            if !msg.contains("downloading") {
                mp.println(msg).unwrap();
            }
        }
    }).await?;
    
    println!("Model {} pulled successfully", model);
    
    Ok(())
}

async fn show_model(client: &Client, model: &str) -> Result<()> {
    let info = client.show(model).await?;
    
    println!("\nModel: {}", info.model);
    println!("  Size: {}", human_bytes(info.size));
    
    if let Some(details) = &info.details {
        if let Some(family) = &details.family {
            println!("  Family: {}", family);
        }
        if let Some(params) = &details.parameter_size {
            println!("  Parameters: {}", params);
        }
        if let Some(quant) = &details.quantization_level {
            println!("  Quantization: {}", quant);
        }
    }
    
    let modified_timestamp = chrono::DateTime::parse_from_rfc3339(&info.modified_at).map(|dt| dt.timestamp()).unwrap_or(0);
    println!("  Modified: {}", human_time(modified_timestamp, "Never"));
    
    if let Some(cap) = info.capabilities.first() {
        println!("  Capabilities: {}", cap);
    }
    
    if let Some(system) = &info.system {
        println!("\nSystem:\n{}", system);
    }
    
    if let Some(license) = &info.license {
        println!("\nLicense:\n{}", license);
    }
    
    Ok(())
}

pub async fn serve() -> Result<()> {
    println!("Starting Ollama server...");
    crate::server::serve().await
}

pub async fn create(model: &str, file: Option<String>) -> Result<()> {
    let client = Client::from_env()?;
    
    let modelfile = if let Some(f) = file {
        std::fs::read_to_string(&f).context("Failed to read Modelfile")?
    } else {
        "FROM .\n".to_string()
    };
    
    let request = json!({
        "name": model,
        "modelfile": modelfile,
        "stream": true
    });
    
    use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
    
    let mp = MultiProgress::new();
    let mut bars = std::collections::HashMap::new();
    
    client.create(&request, |status| {
        if let Some(digest) = status.get("digest").and_then(|v| v.as_str()) {
             let bar = bars.entry(digest.to_string()).or_insert_with(|| {
                let b = mp.add(ProgressBar::new(0));
                b.set_style(ProgressStyle::default_bar()
                    .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"));
                b.set_message(format!("creating {}", &digest[7..19]));
                b
            });
            
            if let Some(total) = status.get("total").and_then(|v| v.as_u64()) {
                bar.set_length(total);
            }
            if let Some(completed) = status.get("completed").and_then(|v| v.as_u64()) {
                bar.set_position(completed);
            }
        } else if let Some(msg) = status.get("status").and_then(|v| v.as_str()) {
            mp.println(msg).unwrap();
        }
    }).await?;
    
    println!("Created model: {}", model);
    
    Ok(())
}

pub async fn show(model: &str) -> Result<()> {
    show_model(&Client::from_env()?, model).await
}

pub async fn list() -> Result<()> {
    list_models(&Client::from_env()?).await
}

pub async fn ps() -> Result<()> {
    let client = Client::from_env()?;
    
    let models = client.list_running().await?;
    
    if models.is_empty() {
        println!("No models currently running");
        return Ok(());
    }
    
    println!("\n{:<40} {:<12} {:<10} {:<8} UNTIL", "NAME", "ID", "SIZE", "PROCESSOR");
    println!("{}", "-".repeat(80));
    
    for m in models {
        let processor = if m.size_vram == 0 {
            "100% CPU".to_string()
        } else if m.size_vram == m.size {
            "100% GPU".to_string()
        } else {
            let cpu_percent = ((m.size - m.size_vram) as f64 / m.size as f64 * 100.0).round() as i32;
            format!("{}% CPU / {}% GPU", cpu_percent, 100 - cpu_percent)
        };
        let expires_timestamp = chrono::DateTime::parse_from_rfc3339(&m.expires_at).map(|dt| dt.timestamp()).unwrap_or(0);
        println!("{:<40} {:<12} {:<10} {:<8} {}", m.name, &m.digest[..12.min(m.digest.len())], human_bytes(m.size), processor, human_time(expires_timestamp, "Never"));
    }
    
    Ok(())
}

pub async fn pull(model: &str, _insecure: bool) -> Result<()> {
    pull_model(&Client::from_env()?, model).await
}

pub async fn push(model: &str, insecure: bool) -> Result<()> {
    let client = Client::from_env()?;
    
    println!("Pushing model {}...", model);
    
    let request = json!({
        "name": model,
        "insecure": insecure,
        "stream": true
    });
    
    use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
    let mp = MultiProgress::new();
    let mut bars = std::collections::HashMap::new();

    client.push(&request, |status| {
        if let Some(digest) = status.get("digest").and_then(|v| v.as_str()) {
             let bar = bars.entry(digest.to_string()).or_insert_with(|| {
                let b = mp.add(ProgressBar::new(0));
                b.set_style(ProgressStyle::default_bar()
                    .template("{spinner:.green} {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"));
                b.set_message(format!("pushing {}", &digest[7..19]));
                b
            });
            
            if let Some(total) = status.get("total").and_then(|v| v.as_u64()) {
                bar.set_length(total);
            }
            if let Some(completed) = status.get("completed").and_then(|v| v.as_u64()) {
                bar.set_position(completed);
            }
        } else if let Some(msg) = status.get("status").and_then(|v| v.as_str()) {
            mp.println(msg).unwrap();
        }
    }).await?;
    
    println!("Model {} pushed successfully", model);
    
    Ok(())
}

pub async fn copy(source: &str, destination: &str) -> Result<()> {
    let client = Client::from_env()?;
    
    let request = json!({
        "source": source,
        "destination": destination
    });
    
    client.copy(&request).await?;
    
    println!("Copied '{}' to '{}'", source, destination);
    
    Ok(())
}

pub async fn delete(model: &str) -> Result<()> {
    let client = Client::from_env()?;
    
    client.delete(model).await?;
    
    println!("Deleted '{}'", model);
    
    Ok(())
}

pub async fn stop(model: &str) -> Result<()> {
    let client = Client::from_env()?;
    
    client.stop(model).await?;
    
    println!("Stopped model '{}'", model);
    
    Ok(())
}

pub async fn embed(model: &str, input: &str) -> Result<()> {
    let client = Client::from_env()?;
    
    let request = json!({
        "model": model,
        "input": input
    });
    
    let response = client.embed(&request).await?;
    
    for embedding in &response.embeddings {
        println!("{:?}, ... (total {})", &embedding[..5.min(embedding.len())], embedding.len());
    }
    
    Ok(())
}

pub async fn version() -> Result<()> {
    println!("ollama version 0.5.0 (Rust)");
    
    let client = Client::from_env()?;
    if let Ok(server_version) = client.version().await {
        println!("server version: {}", server_version);
    }
    
    Ok(())
}
