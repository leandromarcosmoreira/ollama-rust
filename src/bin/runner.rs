use anyhow::Result;
use clap::Parser;
use std::io::{BufRead, BufReader, Write};

#[derive(Parser)]
#[command(name = "ollama-runner")]
#[command(about = "Ollama inference runner - Pure Rust")]
struct Args {
    #[arg(short, long)]
    model: String,
    
    #[arg(long, default_value = "2048")]
    ctx_size: u32,
    
    #[arg(long, default_value = "4")]
    threads: u32,
    
    #[arg(long, default_value = "0")]
    gpu_layers: i32,
    
    #[arg(long)]
    embedding: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    let model_path = std::path::Path::new(&args.model);
    if !model_path.exists() {
        eprintln!("Model file not found: {}", args.model);
        std::process::exit(1);
    }

    let gguf = match Gguf::open(&args.model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error opening GGUF: {}", e);
            std::process::exit(1);
        }
    };
    
    eprintln!("=== Ollama Runner (Pure Rust) ===");
    eprintln!("Model: {}", args.model);
    eprintln!("Architecture: {}", gguf.architecture);
    eprintln!("Vocab size: {}", gguf.metadata.vocab_size);
    eprintln!("Context length: {}", gguf.metadata.context_length);
    eprintln!("Embedding length: {}", gguf.metadata.embedding_length);
    eprintln!("Layers: {}", gguf.metadata.block_count);
    eprintln!("Heads: {} / KV: {}", gguf.metadata.head_count, gguf.metadata.head_count_kv);
    eprintln!("================================");
    eprintln!("Ready for inference. Send JSON requests via stdin.");
    
    let mut model_config = ollama::ModelConfig::default();
    model_config.architecture = gguf.architecture.clone();
    model_config.vocab_size = gguf.metadata.vocab_size as usize;
    model_config.context_length = gguf.metadata.context_length as usize;

    let model = ollama::core::model::architectures::llama::LlamaModel::load(&args.model, model_config)?;
    let mut vocab = ollama::core::tokenizer::Vocabulary::new(gguf.metadata.vocab_tokens.unwrap_or_default());
    vocab.scores = gguf.metadata.vocab_scores.unwrap_or_default();
    
    let tokenizer = ollama::core::tokenizer::create_tokenizer(
        if gguf.architecture.contains("llama") {
            ollama::core::tokenizer::TokenizerKind::Bpe
        } else {
            ollama::core::tokenizer::TokenizerKind::WordPiece
        },
        vocab
    );

    let mut runner = ollama::InferenceRunner::new(Box::new(model), tokenizer);
    
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    
    let reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();
 
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
 
        if line.is_empty() {
            continue;
        }
 
        let request: serde_json::Value = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let error = serde_json::json!({"error": e.to_string()});
                writeln!(writer, "{}", serde_json::to_string(&error)?)?;
                writer.flush()?;
                continue;
            }
        };
 
        if let Some(prompt) = request.get("prompt").and_then(|p| p.as_str()) {
            let n_predict = request.get("n_predict")
                .and_then(|t| t.as_i64())
                .unwrap_or(128) as i32;
            
            runner = runner.max_tokens(n_predict as usize);
            
            if let Some(t) = request.get("temperature").and_then(|v| v.as_f64()) {
                runner = runner.temperature(t as f32);
            }
            if let Some(p) = request.get("top_p").and_then(|v| v.as_f64()) {
                runner = runner.top_p(p as f32);
            }
 
            let mut tokens_generated = 0;
            if let Ok(response) = runner.generate(prompt) {
                for token in response.split_whitespace() {
                    tokens_generated += 1;
                    let token_response = serde_json::json!({
                        "token": format!("{} ", token),
                        "done": false
                    });
                    let _ = writeln!(writer, "{}", serde_json::to_string(&token_response).unwrap());
                    let _ = writer.flush();
                }
            }
 
            let done_response = serde_json::json!({
                "token": "",
                "done": true,
                "tokens_generated": tokens_generated
            });
            writeln!(writer, "{}", serde_json::to_string(&done_response)?)?;
            writer.flush()?;
        }
 
        if let Some(_embed_input) = request.get("embed").and_then(|e| e.as_str()) {
            // Embed functionality is currently being transitioned in LlamaModel
            let error = serde_json::json!({"error": "Embedding currently being transitioned to new architecture"});
            writeln!(writer, "{}", serde_json::to_string(&error)?)?;
            writer.flush()?;
        }
    }
 
    Ok(())
}

use ollama::GgufFile as Gguf;
