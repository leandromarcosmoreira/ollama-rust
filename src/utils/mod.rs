pub mod jsonl {
    use std::io::{BufRead, BufReader};
    use std::path::Path;

    pub fn parse_jsonl_file<P: AsRef<Path>>(path: P) -> Result<Vec<serde_json::Value>, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        
        let mut results = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| e.to_string())?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(trimmed) {
                Ok(value) => results.push(value),
                Err(e) => eprintln!("Failed to parse line: {} - {}", trimmed, e),
            }
        }
        Ok(results)
    }

    pub fn parse_jsonl_str(content: &str) -> Result<Vec<serde_json::Value>, String> {
        let mut results = Vec::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(trimmed) {
                Ok(value) => results.push(value),
                Err(e) => eprintln!("Failed to parse line: {} - {}", trimmed, e),
            }
        }
        Ok(results)
    }
}

pub mod file_validation {
    pub const TEXT_FILE_EXTENSIONS: &[&str] = &[
        "pdf", "docx", "txt", "md", "csv", "json", "xml", "html", "htm",
        "js", "jsx", "ts", "tsx", "py", "java", "cpp", "c", "cc", "h",
        "cs", "php", "rb", "go", "rs", "swift", "kt", "scala", "sh",
        "bat", "yaml", "yml", "toml", "ini", "cfg", "conf", "log", "rtf",
    ];

    pub const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp"];

    pub fn is_text_extension(ext: &str) -> bool {
        TEXT_FILE_EXTENSIONS.contains(&ext.to_lowercase().as_str())
    }

    pub fn is_image_extension(ext: &str) -> bool {
        IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str())
    }

    pub fn get_extension(filename: &str) -> Option<String> {
        filename.rsplit('.').next().map(|s| s.to_lowercase())
    }

    pub fn validate_extension(filename: &str) -> Option<&'static str> {
        let ext = get_extension(filename)?;
        
        if IMAGE_EXTENSIONS.contains(&ext.as_str()) {
            return Some("image");
        }
        if TEXT_FILE_EXTENSIONS.contains(&ext.as_str()) {
            return Some("text");
        }
        None
    }

    pub fn is_valid_file_size(size: u64, max_mb: u64) -> bool {
        let max_bytes = max_mb * 1024 * 1024;
        size <= max_bytes
    }
}

pub mod vram {
    const GIB_FACTOR: f64 = 1.0;
    const GB_FACTOR: f64 = 1000.0 / 1024.0;
    const MIB_FACTOR: f64 = 1.0 / 1024.0;
    const MB_FACTOR: f64 = 1.0 / (1024.0 * 1024.0);

    pub fn parse_vram(vram_string: &str) -> Option<f64> {
        if vram_string.is_empty() {
            return None;
        }

        let vram_lower = vram_string.to_lowercase();
        
        let (value, factor) = if let Some(matched) = vram_lower.strip_suffix("gib") {
            (matched.trim().parse::<f64>().ok()?, GIB_FACTOR)
        } else if let Some(matched) = vram_lower.strip_suffix("gb") {
            (matched.trim().parse::<f64>().ok()?, GB_FACTOR)
        } else if let Some(matched) = vram_lower.strip_suffix("mib") {
            (matched.trim().parse::<f64>().ok()?, MIB_FACTOR)
        } else if let Some(matched) = vram_lower.strip_suffix("mb") {
            (matched.trim().parse::<f64>().ok()?, MB_FACTOR)
        } else {
            return None;
        };

        Some(value * factor)
    }

    pub fn get_total_vram(vram_strings: &[String]) -> f64 {
        vram_strings
            .iter()
            .filter_map(|v| parse_vram(v))
            .sum()
    }

    pub fn format_vram_gib(vram_gib: f64) -> String {
        if vram_gib >= 1.0 {
            format!("{:.1} GiB", vram_gib)
        } else {
            format!("{:.0} MiB", vram_gib * 1024.0)
        }
    }
}

pub mod image {
    pub fn is_image_file(filename: &str) -> bool {
        let extension = filename.rsplit('.').next();
        match extension {
            Some(ext) => matches!(ext.to_lowercase().as_str(), "png" | "jpg" | "jpeg" | "gif" | "webp"),
            None => false,
        }
    }
}

pub mod merge_models {
    pub const FEATURED_MODELS: &[&str] = &[
        "gpt-oss:120b-cloud",
        "gpt-oss:20b-cloud",
        "deepseek-v3.1:671b-cloud",
        "qwen3-coder:480b-cloud",
        "qwen3-vl:235b-cloud",
        "minimax-m2:cloud",
        "glm-4.6:cloud",
        "gpt-oss:120b",
        "gpt-oss:20b",
        "gemma3:27b",
        "gemma3:12b",
        "gemma3:4b",
        "gemma3:1b",
        "deepseek-r1:8b",
        "qwen3-coder:30b",
        "qwen3-vl:30b",
        "qwen3-vl:8b",
        "qwen3-vl:4b",
        "qwen3:30b",
        "qwen3:8b",
        "qwen3:4b",
    ];

    pub fn is_cloud_model(model: &str) -> bool {
        model.ends_with("cloud")
    }

    pub fn recommend_default_model(total_vram_gib: f64) -> &'static str {
        if total_vram_gib < 6.0 {
            "gemma3:1b"
        } else if total_vram_gib < 16.0 {
            "gemma3:4b"
        } else {
            "gpt-oss:20b"
        }
    }
}

pub mod citation {
    #[derive(Debug, Clone)]
    pub struct Citation {
        pub cursor: u32,
        pub start_line: Option<u32>,
        pub end_line: Option<u32>,
    }

    pub fn parse_citations(text: &str) -> Vec<(String, Vec<Citation>)> {
        let mut results = Vec::new();
        
        let range_regex = regex::Regex::new(r"【(\d+)†L(\d+)-L(\d+)】").ok();
        let generic_regex = regex::Regex::new(r"【(\d+)†[^】]*】").ok();
        
        let mut last_end = 0usize;
        
        if let Some(ref re) = range_regex {
            for cap in re.captures_iter(text) {
                if let Some(m) = cap.get(0) {
                    let _ = m.start() > last_end;
                    
                    if let Some(cursor) = cap.get(1).and_then(|m| m.as_str().parse().ok()) {
                        let start = cap.get(2).and_then(|m| m.as_str().parse().ok());
                        let end = cap.get(3).and_then(|m| m.as_str().parse().ok());
                        results.push((String::new(), vec![Citation {
                            cursor,
                            start_line: start,
                            end_line: end,
                        }]));
                    }
                    last_end = m.end();
                }
            }
        }
        
        if let Some(ref re) = generic_regex {
            for cap in re.captures_iter(text) {
                if let Some(cursor) = cap.get(1).and_then(|m| m.as_str().parse().ok()) {
                    results.push((String::new(), vec![Citation {
                        cursor,
                        start_line: None,
                        end_line: None,
                    }]));
                }
            }
        }
        
        results
    }

    pub fn extract_citation_numbers(text: &str) -> Vec<u32> {
        let mut numbers = Vec::new();
        
        if let Ok(re) = regex::Regex::new(r"【(\d+)") {
            for cap in re.captures_iter(text) {
                if let Some(num) = cap.get(1).and_then(|m| m.as_str().parse().ok()) {
                    if !numbers.contains(&num) {
                        numbers.push(num);
                    }
                }
            }
        }
        
        numbers.sort();
        numbers
    }
}

pub mod string_utils {
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len.saturating_sub(3)])
        }
    }

    pub fn capitalize(s: &str) -> String {
        let mut c = s.chars();
        match c.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().chain(c).collect(),
        }
    }

    pub fn to_snake_case(s: &str) -> String {
        let mut result = String::new();
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap_or(c));
        }
        result
    }

    pub fn to_camel_case(s: &str) -> String {
        let mut capitalize_next = false;
        s.chars()
            .map(|c| {
                if c == '_' {
                    capitalize_next = true;
                    String::new()
                } else if capitalize_next {
                    capitalize_next = false;
                    c.to_uppercase().to_string()
                } else {
                    c.to_lowercase().to_string()
                }
            })
            .collect()
    }

    pub fn extract_model_name(model: &str) -> String {
        if let Some(colon_pos) = model.rfind(':') {
            model[..colon_pos].to_string()
        } else {
            model.to_string()
        }
    }

    pub fn extract_model_tag(model: &str) -> Option<String> {
        model.rfind(':').map(|colon_pos| model[colon_pos + 1..].to_string())
    }
}

pub mod time_utils {
    pub fn format_timestamp(timestamp: &str) -> String {
        timestamp.to_string()
    }

    pub fn time_ago(timestamp: &str) -> String {
        timestamp.to_string()
    }

    pub fn parse_iso_timestamp(s: &str) -> Option<i64> {
        chrono::DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|dt| dt.timestamp())
    }
}
