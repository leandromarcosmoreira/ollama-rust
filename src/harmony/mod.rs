#![allow(dead_code)]
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParserState {
    LookingForMessageStart,
    ParsingHeader,
    ParsingContent,
}

impl std::fmt::Display for ParserState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LookingForMessageStart => write!(f, "LookingForMessageStart"),
            Self::ParsingHeader => write!(f, "ParsingHeader"),
            Self::ParsingContent => write!(f, "ParsingContent"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Header {
    pub role: String,
    pub channel: String,
    pub recipient: String,
}

#[derive(Debug, Clone)]
pub enum Event {
    MessageStart,
    HeaderComplete(Header),
    ContentEmitted(String),
    MessageEnd,
}

pub struct Parser {
    state: ParserState,
    message_start_tag: String,
    message_end_tag: String,
    header_end_tag: String,
    acc: String,
}

impl Parser {
    pub fn new() -> Self {
        Self {
            state: ParserState::LookingForMessageStart,
            message_start_tag: "<|start|>".to_string(),
            message_end_tag: "<|end|>".to_string(),
            header_end_tag: "<|message|>".to_string(),
            acc: String::new(),
        }
    }

    pub fn add_implicit_start(&mut self) {
        self.acc.push_str("<|start|>assistant");
    }

    pub fn add_content(&mut self, content: &str) -> Vec<Event> {
        self.acc.push_str(content);
        let mut events = Vec::new();
        
        loop {
            let (new_events, continue_eating) = self.eat();
            events.extend(new_events);
            if !continue_eating {
                break;
            }
        }
        
        events
    }

    fn eat(&mut self) -> (Vec<Event>, bool) {
        match self.state {
            ParserState::LookingForMessageStart => {
                if let Some(pos) = self.acc.find(&self.message_start_tag) {
                    let after = &self.acc[pos + self.message_start_tag.len()..];
                    self.acc = after.to_string();
                    self.state = ParserState::ParsingHeader;
                    return (vec![Event::MessageStart], true);
                }
                (vec![], false)
            }
            ParserState::ParsingHeader => {
                if let Some(pos) = self.acc.find(&self.header_end_tag) {
                    let header_str = &self.acc[..pos];
                    let after = &self.acc[pos + self.header_end_tag.len()..];
                    let header = self.parse_header(header_str);
                    self.acc = after.to_string();
                    self.state = ParserState::ParsingContent;
                    return (vec![Event::HeaderComplete(header)], true);
                }
                (vec![], false)
            }
            ParserState::ParsingContent => {
                if let Some(pos) = self.acc.find(&self.message_end_tag) {
                    let content = self.acc[..pos].to_string();
                    let after = self.acc[pos + self.message_end_tag.len()..].to_string();
                    self.acc = after;
                    self.state = ParserState::LookingForMessageStart;
                    
                    let mut events = Vec::new();
                    if !content.is_empty() {
                        events.push(Event::ContentEmitted(content));
                    }
                    events.push(Event::MessageEnd);
                    (events, true)
                } else {
                    let overlap_len = Self::overlap(&self.acc, &self.message_end_tag);
                    if overlap_len > 0 {
                        let content_len = self.acc.len() - overlap_len;
                        let content = self.acc[..content_len].to_string();
                        let remaining = self.acc[content_len..].to_string();
                        self.acc = remaining;
                        if content.is_empty() {
                            return (vec![], false);
                        }
                        (vec![Event::ContentEmitted(content)], false)
                    } else {
                        let content = std::mem::take(&mut self.acc);
                        if content.is_empty() {
                            return (vec![], false);
                        }
                        (vec![Event::ContentEmitted(content)], false)
                    }
                }
            }
        }
    }

    fn parse_header(&self, raw: &str) -> Header {
        let mut header = Header {
            role: String::new(),
            channel: String::new(),
            recipient: String::new(),
        };
        
        let raw = raw.replace("<|constrain|>", " <|constrain|>");
        let raw = raw.trim();
        
        if let Some(channel_idx) = raw.find("<|channel|>") {
            let before = &raw[..channel_idx];
            let after = &raw[channel_idx + "<|channel|>".len()..];
            
            let end_idx = after.find(|c: char| c.is_whitespace()).unwrap_or(after.len());
            header.channel = after[..end_idx].to_string();
            let after = after[end_idx..].trim();
            
            let raw = format!("{} {}", before, after);
            let tokens: Vec<&str> = raw.split_whitespace().collect();
            
            if !tokens.is_empty() {
                let role = tokens[0];
                if let Some(stripped) = role.strip_prefix("to=") {
                    header.recipient = stripped.to_string();
                    header.role = "tool".to_string();
                } else {
                    header.role = role.to_string();
                }
            }
            
            for token in tokens.iter().skip(1) {
                if token.starts_with("to=") && header.recipient.is_empty() {
                    header.recipient = token[3..].to_string();
                }
            }
        } else {
            let tokens: Vec<&str> = raw.split_whitespace().collect();
            if !tokens.is_empty() {
                let role = tokens[0];
                if let Some(stripped) = role.strip_prefix("to=") {
                    header.recipient = stripped.to_string();
                    header.role = "tool".to_string();
                } else {
                    header.role = role.to_string();
                }
            }
        }
        
        header
    }

    fn overlap(s: &str, delim: &str) -> usize {
        let max = delim.len().min(s.len());
        for i in (1..=max).rev() {
            if s.ends_with(&delim[..i]) {
                return i;
            }
        }
        0
    }
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FunctionNameMap {
    user_to_harmony: HashMap<String, String>,
    harmony_to_user: HashMap<String, String>,
}

impl FunctionNameMap {
    pub fn new() -> Self {
        Self {
            user_to_harmony: HashMap::new(),
            harmony_to_user: HashMap::new(),
        }
    }

    pub fn convert_and_add(&mut self, user_function_name: &str) -> String {
        let harmony_name = self.derive_name(user_function_name);
        self.user_to_harmony.insert(user_function_name.to_string(), harmony_name.clone());
        self.harmony_to_user.insert(harmony_name.clone(), user_function_name.to_string());
        harmony_name
    }

    pub fn original_from_converted(&self, harmony_function_name: &str) -> String {
        self.harmony_to_user.get(harmony_function_name)
            .cloned()
            .unwrap_or_else(|| harmony_function_name.to_string())
    }

    fn convert_to_valid_chars(&self, name: &str) -> String {
        let mut result = String::new();
        for c in name.chars() {
            if c == ' ' || c == '-' || c == '.' {
                result.push('_');
            } else if c.is_alphanumeric() || c == '_' || c == '$' {
                result.push(c);
            }
        }
        
        if result.is_empty() {
            return "unnamed".to_string();
        }
        
        if result.chars().next().map(|c| c.is_numeric()).unwrap_or(false) {
            result.insert(0, '_');
        }
        
        result
    }

    fn derive_name(&self, user_function_name: &str) -> String {
        let candidate = self.convert_to_valid_chars(user_function_name);
        let mut name = candidate.clone();
        let mut count = 2;
        
        while self.harmony_to_user.contains_key(&name) {
            name = format!("{}_{}", candidate, count);
            count += 1;
        }
        
        name
    }
}

impl Default for FunctionNameMap {
    fn default() -> Self {
        Self::new()
    }
}
