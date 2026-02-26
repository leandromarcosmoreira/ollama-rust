use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::Response,
};

pub async fn allowed_hosts_middleware(
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // In Go, this checks against OLLAMA_ORIGINS and localhost.
    // Here we implement a simplified check allowing localhost explicitly
    // and passing through for standard Ollama CLI execution.
    let host = req.headers().get("host").and_then(|h| h.to_str().ok()).unwrap_or("");
    
    // Allow if host is empty (e.g. some HTTP/1.0 requests) or matches local patterns
    if host.is_empty() 
        || host.starts_with("localhost:") 
        || host.starts_with("127.0.0.1:") 
        || host.starts_with("0.0.0.0:")
        || host == "localhost" 
        || host == "127.0.0.1" 
        || host == "0.0.0.0"
    {
        Ok(next.run(req).await)
    } else {
        // If it was a real production server, we would check against an OLLAMA_ORIGINS env var.
        // For security by default, we reject unknown host requests.
        Err(StatusCode::FORBIDDEN)
    }
}
