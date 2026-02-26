#![allow(clippy::module_inception)]
#![allow(unused)]
pub mod template {
    use anyhow::Result;
    use std::collections::HashMap;
    
    #[allow(dead_code)]
    pub struct Template {
        template: String,
    }
    
    #[allow(dead_code)]
    impl Template {
        pub fn new(template: &str) -> Self {
            Self {
                template: template.to_string(),
            }
        }
        
        pub fn execute(&self, data: &HashMap<String, String>) -> Result<String> {
            let mut result = self.template.clone();
            
            for (key, value) in data {
                let placeholder = format!("{{{{ .{} }}}}", key);
                result = result.replace(&placeholder, value);
            }
            
            Ok(result)
        }
    }
    
    #[allow(dead_code)]
    pub fn chat_template(system: &str, messages: &[Message]) -> Result<String> {
        let mut prompt = String::new();
        
        if !system.is_empty() {
            prompt.push_str(&format!("System: {}\n", system));
        }
        
        for msg in messages {
            prompt.push_str(&format!("{}: {}\n", capitalize(&msg.role), msg.content));
        }
        
        prompt.push_str("Assistant: ");
        
        Ok(prompt)
    }
    
    #[allow(dead_code)]
    fn capitalize(s: &str) -> String {
        let mut chars = s.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
    }
    
    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    pub struct Message {
        pub role: String,
        pub content: String,
    }
}
