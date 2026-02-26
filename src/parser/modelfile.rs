use anyhow::Result;
use std::io::BufRead;
use std::path::Path;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    pub name: String,
    pub args: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Modelfile {
    pub commands: Vec<Command>,
}

pub fn parse<R: BufRead>(reader: R) -> Result<Modelfile> {
    let mut modelfile = Modelfile::default();
    
    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim();
        
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        
        let (name, args) = split_command(trimmed);
        modelfile.commands.push(Command {
            name: name.to_lowercase(),
            args: args.to_string(),
        });
    }
    
    Ok(modelfile)
}

fn split_command(line: &str) -> (&str, &str) {
    let mut parts = line.splitn(2, |c: char| c.is_whitespace());
    let name = parts.next().unwrap_or("");
    let args = parts.next().unwrap_or("").trim();
    (name, args)
}

#[allow(dead_code)]
impl Modelfile {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        parse(reader)
    }

    pub fn to_api_request(&self) -> Result<crate::api::CreateRequest> {
        let mut req = crate::api::CreateRequest::default();
        let mut params = std::collections::HashMap::new();
        let mut messages = Vec::new();
        
        for cmd in &self.commands {
            match cmd.name.as_str() {
                "from" => req.from = cmd.args.clone(),
                "template" => req.template = Some(cmd.args.clone()),
                "system" => req.system = Some(cmd.args.clone()),
                "license" => req.license = Some(vec![cmd.args.clone()]), // simplified
                "adapter" => {
                    // In a real impl, this would resolve the path
                    req.adapters.insert(cmd.args.clone(), "".to_string());
                },
                "parameter" => {
                    let (key, val) = split_command(&cmd.args);
                    params.insert(key.to_string(), serde_json::Value::String(val.to_string()));
                },
                "message" => {
                    let (role, content) = split_command(&cmd.args);
                    messages.push(crate::api::Message {
                        role: role.to_string(),
                        content: content.to_string(),
                        images: vec![],
                    });
                },
                _ => {
                    // Custom parameters or unknowns
                    params.insert(cmd.name.clone(), serde_json::Value::String(cmd.args.clone()));
                }
            }
        }
        
        if !params.is_empty() {
            req.options = Some(params);
        }
        
        if !messages.is_empty() {
            req.messages = Some(messages);
        }
        
        Ok(req)
    }
}
