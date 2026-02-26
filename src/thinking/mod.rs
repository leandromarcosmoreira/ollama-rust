pub mod thinking {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum State {
        LookingForOpening,
        ThinkingStartedEatingWhitespace,
        Thinking,
        ThinkingDoneEatingWhitespace,
        ThinkingDone,
    }

    impl std::fmt::Display for State {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::LookingForOpening => write!(f, "LookingForOpening"),
                Self::ThinkingStartedEatingWhitespace => write!(f, "ThinkingStartedEatingWhitespace"),
                Self::Thinking => write!(f, "Thinking"),
                Self::ThinkingDoneEatingWhitespace => write!(f, "ThinkingDoneEatingWhitespace"),
                Self::ThinkingDone => write!(f, "ThinkingDone"),
            }
        }
    }

    pub struct Parser {
        state: State,
        opening_tag: String,
        closing_tag: String,
        acc: String,
    }

    impl Parser {
        pub fn new() -> Self {
            Self {
                state: State::LookingForOpening,
                opening_tag: "<think".to_string(),
                closing_tag: "</think".to_string(),
                acc: String::new(),
            }
        }

        pub fn with_tags(opening_tag: &str, closing_tag: &str) -> Self {
            Self {
                state: State::LookingForOpening,
                opening_tag: opening_tag.to_string(),
                closing_tag: closing_tag.to_string(),
                acc: String::new(),
            }
        }

        pub fn add_content(&mut self, content: &str) -> (String, String) {
            self.acc.push_str(content);
            
            let mut thinking_output = String::new();
            let mut remaining_output = String::new();
            
            loop {
                let (thinking, remaining, continue_eating) = self.eat();
                thinking_output.push_str(&thinking);
                remaining_output.push_str(&remaining);
                if !continue_eating {
                    break;
                }
            }
            
            (thinking_output, remaining_output)
        }

        fn eat(&mut self) -> (String, String, bool) {
            match self.state {
                State::LookingForOpening => {
                    let trimmed = self.acc.trim_start().to_string();
                    
                    if trimmed.starts_with(&self.opening_tag) {
                        let after_tag = &trimmed[self.opening_tag.len()..];
                        let after = after_tag.trim_start().to_string();
                        self.acc = after.clone();
                        
                        self.state = if after.is_empty() {
                            State::ThinkingStartedEatingWhitespace
                        } else {
                            State::Thinking
                        };
                        return (String::new(), String::new(), true);
                    } else if self.opening_tag.starts_with(&trimmed) {
                        return (String::new(), String::new(), false);
                    } else if trimmed.is_empty() {
                        return (String::new(), String::new(), false);
                    } else {
                        self.state = State::ThinkingDone;
                        let content = std::mem::take(&mut self.acc);
                        return (String::new(), content, false);
                    }
                }
                State::ThinkingStartedEatingWhitespace => {
                    let trimmed = self.acc.trim_start().to_string();
                    self.acc.clear();
                    
                    if trimmed.is_empty() {
                        return (String::new(), String::new(), false);
                    } else {
                        self.state = State::Thinking;
                        self.acc = trimmed;
                        return (String::new(), String::new(), true);
                    }
                }
                State::Thinking => {
                    if let Some(pos) = self.acc.find(&self.closing_tag) {
                        let thinking = self.acc[..pos].to_string();
                        let remaining = self.acc[pos + self.closing_tag.len()..].trim_start().to_string();
                        self.acc.clear();
                        
                        self.state = if remaining.is_empty() {
                            State::ThinkingDoneEatingWhitespace
                        } else {
                            State::ThinkingDone
                        };
                        
                        return (thinking, remaining, false);
                    } else {
                        let overlap_len = Self::overlap(&self.acc, &self.closing_tag);
                        if overlap_len > 0 {
                            let thinking_len = self.acc.len() - overlap_len;
                            let thinking = self.acc[..thinking_len].to_string();
                            let remaining = self.acc[thinking_len..].to_string();
                            self.acc = remaining;
                            return (thinking, String::new(), false);
                        } else {
                            let thinking = std::mem::take(&mut self.acc);
                            return (thinking, String::new(), false);
                        }
                    }
                }
                State::ThinkingDoneEatingWhitespace => {
                    let trimmed = self.acc.trim_start().to_string();
                    self.acc.clear();
                    
                    if !trimmed.is_empty() {
                        self.state = State::ThinkingDone;
                    }
                    
                    return (String::new(), trimmed, false);
                }
                State::ThinkingDone => {
                    let content = std::mem::take(&mut self.acc);
                    return (String::new(), content, false);
                }
            }
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

    #[allow(dead_code)]
    pub struct ThinkingParser {
        enabled: bool,
        template: String,
    }
    
    #[allow(dead_code)]
    impl ThinkingParser {
        pub fn new() -> Self {
            Self {
                enabled: true,
                template: "mispi\n{{ .Thinking }}\nought\n".to_string(),
            }
        }
        
        pub fn parse(&self, content: &str) -> ThinkingResult {
            if let Some(start) = content.find("mispi") {
                if let Some(end) = content.find("ought\n") {
                    let thinking = &content[start + 6..end];
                    let actual = content[end + 6..].trim();
                    
                    return ThinkingResult {
                        thinking: thinking.to_string(),
                        content: actual.to_string(),
                    };
                }
            }
            
            ThinkingResult {
                thinking: String::new(),
                content: content.to_string(),
            }
        }
        
        pub fn wrap(&self, thinking: &str, content: &str) -> String {
            if thinking.is_empty() {
                return content.to_string();
            }
            
            format!("mispi\n{}ought\n\n{}", thinking, content)
        }
    }
    
    #[derive(Debug)]
    #[allow(dead_code)]
    pub struct ThinkingResult {
        pub thinking: String,
        pub content: String,
    }
    
    #[allow(dead_code)]
    pub fn infer_thinking_option(capabilities: &[String], user_option: Option<&str>) -> Option<String> {
        let supports_thinking = capabilities.iter().any(|c| c.contains("thinking"));
        
        if supports_thinking {
            Some(user_option.unwrap_or("true").to_string())
        } else {
            None
        }
    }
}
