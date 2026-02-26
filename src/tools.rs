use anyhow::{bail, Result};
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use serde_json::json;
use std::collections::HashMap;
use std::process::Command as ProcessCommand;

pub mod websearch {
    use super::*;
    
    #[derive(Debug)]
    #[allow(dead_code)]
    pub struct WebSearch {
        client: Client,
    }
    
    #[allow(dead_code)]
    impl WebSearch {
        pub fn new() -> Self {
            Self {
                client: Client::builder()
                    .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                    .build()
                    .unwrap_or_else(|_| Client::new()),
            }
        }
        
        pub fn search(&self, query: &str, num_results: usize) -> Result<Vec<SearchResult>> {
            // Using DuckDuckGo HTML (no API key required)
            let url = format!(
                "https://html.duckduckgo.com/html/?q={}",
                urlencoding::encode(query)
            );
            
            let response = self.client.get(&url)
                .send()?;
            
            let body = response.text()?;
            let document = Html::parse_document(&body);
            
            let selector = Selector::parse(".result__body").unwrap();
            let title_selector = Selector::parse(".result__title").unwrap();
            let link_selector = Selector::parse(".result__url").unwrap();
            let snippet_selector = Selector::parse(".result__snippet").unwrap();
            
            let mut results = Vec::new();
            
            for element in document.select(&selector).take(num_results) {
                let title = element.select(&title_selector)
                    .next()
                    .map(|e| e.text().collect::<String>().trim().to_string())
                    .unwrap_or_default();
                
                let link = element.select(&link_selector)
                    .next()
                    .map(|e| e.text().collect::<String>().trim().to_string())
                    .unwrap_or_default();
                
                let snippet = element.select(&snippet_selector)
                    .next()
                    .map(|e| e.text().collect::<String>().trim().to_string())
                    .unwrap_or_default();
                
                if !title.is_empty() || !link.is_empty() {
                    results.push(SearchResult {
                        title,
                        url: link,
                        snippet,
                    });
                }
            }
            
            Ok(results)
        }
    }
    
    impl Default for WebSearch {
        fn default() -> Self {
            Self::new()
        }
    }
}

pub mod webfetch {
    use super::*;
    
    #[derive(Debug)]
    #[allow(dead_code)]
    pub struct WebFetch {
        client: Client,
    }
    
    #[allow(dead_code)]
    impl WebFetch {
        pub fn new() -> Self {
            Self {
                client: Client::builder()
                    .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                    .timeout(std::time::Duration::from_secs(30))
                    .build()
                    .unwrap_or_else(|_| Client::new()),
            }
        }
        
        pub fn fetch(&self, url: &str) -> Result<FetchResult> {
            let response = self.client.get(url)
                .send()?;
            
            let status = response.status().as_u16();
            let headers: HashMap<String, String> = response.headers()
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                .collect();
            
            let content_type = headers.get("content-type")
                .cloned()
                .unwrap_or_default();
            
            let body = response.text()?;
            
            // Extract text content from HTML if needed
            let text_content = if content_type.contains("text/html") {
                extract_text_from_html(&body)
            } else {
                body.clone()
            };
            
            Ok(FetchResult {
                url: url.to_string(),
                status,
                content_type,
                text: text_content,
                html: Some(body),
            })
        }
        
        pub fn fetch_text(&self, url: &str) -> Result<String> {
            Ok(self.fetch(url)?.text)
        }
    }
    
    impl Default for WebFetch {
        fn default() -> Self {
            Self::new()
        }
    }
    
    #[allow(dead_code)]
    fn extract_text_from_html(html: &str) -> String {
        let document = Html::parse_document(html);
        
        // Remove script and style elements
        let selector = Selector::parse("script, style, nav, header, footer").unwrap();
        let mut elements_to_remove = Vec::new();
        for element in document.select(&selector) {
            elements_to_remove.push(element.value().id());
        }
        
        // Get text content from main content areas
        let text_selector = Selector::parse("main, article, body, .content, .main-content").unwrap();
        
        let mut text = String::new();
        
        for element in document.select(&text_selector) {
            let p_selector = Selector::parse("p, h1, h2, h3, h4, h5, h6, li, td, th, span, a").unwrap();
            for p in element.select(&p_selector) {
                let txt = p.text().collect::<String>();
                if !txt.trim().is_empty() {
                    text.push_str(&txt);
                    text.push('\n');
                }
            }
        }
        
        // If no content from selectors, get all text
        if text.is_empty() {
            let body_selector = Selector::parse("body").unwrap();
            if let Some(body) = document.select(&body_selector).next() {
                text = body.text().collect::<Vec<_>>().join("\n");
            }
        }
        
        // Clean up whitespace
        let re = regex::Regex::new(r"\n{3,}").unwrap();
        re.replace_all(&text, "\n\n").to_string()
    }
}

pub mod bash {
    use super::*;
    
    #[derive(Debug)]
    #[allow(dead_code)]
    pub struct BashExecutor {
        timeout_secs: u64,
    }
    
    #[allow(dead_code)]
    impl BashExecutor {
        pub fn new() -> Self {
            Self {
                timeout_secs: 60,
            }
        }
        
        pub fn with_timeout(mut self, seconds: u64) -> Self {
            self.timeout_secs = seconds;
            self
        }
        
        pub fn execute(&self, command: &str) -> Result<BashResult> {
            let start = std::time::Instant::now();
            
            // Parse command - handle shell features
            let parts: Vec<&str> = command.split_whitespace().collect();
            
            if parts.is_empty() {
                bail!("Empty command");
            }
            
            // Use /bin/bash for full shell support
            let output = ProcessCommand::new("bash")
                .arg("-c")
                .arg(command)
                .output()?;
            
            let duration = start.elapsed();
            
            Ok(BashResult {
                command: command.to_string(),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                exit_code: output.status.code().unwrap_or(-1),
                success: output.status.success(),
                duration_millis: duration.as_millis() as u64,
            })
        }
        
        pub fn execute_interactive(&self, command: &str) -> Result<String> {
            let output = ProcessCommand::new("bash")
                .arg("-c")
                .arg(command)
                .output()?;
            
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                bail!("Command failed: {}", stderr);
            }
            
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        }
    }
    
    impl Default for BashExecutor {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FetchResult {
    pub url: String,
    pub status: u16,
    pub content_type: String,
    pub text: String,
    pub html: Option<String>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BashResult {
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub success: bool,
    pub duration_millis: u64,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct ToolExecutor {
    websearch: websearch::WebSearch,
    webfetch: webfetch::WebFetch,
    bash: bash::BashExecutor,
}

#[allow(dead_code)]
impl ToolExecutor {
    pub fn new() -> Self {
        Self {
            websearch: websearch::WebSearch::new(),
            webfetch: webfetch::WebFetch::new(),
            bash: bash::BashExecutor::new(),
        }
    }
    
    pub fn execute(&self, tool_name: &str, arguments: &HashMap<String, serde_json::Value>) -> Result<String> {
        match tool_name.to_lowercase().as_str() {
            "websearch" => {
                let query = arguments.get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let num_results = arguments.get("num_results")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(5) as usize;
                
                let results = self.websearch.search(query, num_results)?;
                
                let json_results: Vec<serde_json::Value> = results.iter().map(|r| {
                    json!({
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet
                    })
                }).collect();
                
                Ok(serde_json::to_string_pretty(&json_results)?)
            }
            "webfetch" => {
                let url = arguments.get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                
                if url.is_empty() {
                    bail!("URL is required");
                }
                
                let result = self.webfetch.fetch(url)?;
                Ok(serde_json::to_string_pretty(&json!({
                    "url": result.url,
                    "status": result.status,
                    "content_type": result.content_type,
                    "text": result.text,
                }))?)
            }
            "bash" | "shell" | "exec" => {
                let command = arguments.get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                
                if command.is_empty() {
                    bail!("Command is required");
                }
                
                let result = self.bash.execute(command)?;
                Ok(serde_json::to_string_pretty(&json!({
                    "command": result.command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "success": result.success,
                    "duration_ms": result.duration_millis,
                }))?)
            }
            _ => bail!("Unknown tool: {}", tool_name),
        }
    }
    
    pub fn list_tools(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "websearch".to_string(),
                description: "Search the web for information using DuckDuckGo".to_string(),
                parameters: vec![
                    ParameterDefinition {
                        name: "query".to_string(),
                        description: "The search query".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                    },
                    ParameterDefinition {
                        name: "num_results".to_string(),
                        description: "Number of results to return (default: 5)".to_string(),
                        param_type: "number".to_string(),
                        required: false,
                    },
                ],
            },
            ToolDefinition {
                name: "webfetch".to_string(),
                description: "Fetch content from a URL".to_string(),
                parameters: vec![
                    ParameterDefinition {
                        name: "url".to_string(),
                        description: "The URL to fetch".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                    },
                ],
            },
            ToolDefinition {
                name: "bash".to_string(),
                description: "Execute bash commands".to_string(),
                parameters: vec![
                    ParameterDefinition {
                        name: "command".to_string(),
                        description: "The command to execute".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                    },
                ],
            },
        ]
    }
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ParameterDefinition>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct ParameterDefinition {
    pub name: String,
    pub description: String,
    pub param_type: String,
    pub required: bool,
}
