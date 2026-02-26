pub fn human_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;
    
    if bytes >= TB {
        format!("{:.1}T", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1}G", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}M", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}K", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
    }
}

pub fn human_time(timestamp: i64, default: &str) -> String {
    if timestamp == 0 {
        return default.to_string();
    }
    
    let dt = chrono::DateTime::from_timestamp(timestamp, 0);
    
    if let Some(dt) = dt {
        let now = chrono::Utc::now();
        let diff = now.signed_duration_since(dt);
        
        if diff.num_days() > 365 {
            format!("{} year{} ago", diff.num_days() / 365, if diff.num_days() / 365 > 1 { "s" } else { "" })
        } else if diff.num_days() > 30 {
            format!("{} month{} ago", diff.num_days() / 30, if diff.num_days() / 30 > 1 { "s" } else { "" })
        } else if diff.num_days() > 0 {
            format!("{} day{} ago", diff.num_days(), if diff.num_days() > 1 { "s" } else { "" })
        } else if diff.num_hours() > 0 {
            format!("{} hour{} ago", diff.num_hours(), if diff.num_hours() > 1 { "s" } else { "" })
        } else if diff.num_minutes() > 0 {
            format!("{} minute{} ago", diff.num_minutes(), if diff.num_minutes() > 1 { "s" } else { "" })
        } else {
            "just now".to_string()
        }
    } else {
        default.to_string()
    }
}

#[allow(dead_code)]
pub fn human_number(n: u64) -> String {
    if n >= 1_000_000_000_000 {
        format!("{:.1}T", n as f64 / 1_000_000_000_000.0)
    } else if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
