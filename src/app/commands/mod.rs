use crate::core::Result;
use async_trait::async_trait;

#[async_trait]
pub trait Command: Send + Sync {
    type Output;
    
    async fn execute(&self) -> Result<Self::Output>;
    fn name(&self) -> &str;
    fn description(&self) -> &str;
}

pub struct CommandExecutor {
    commands: Vec<Box<dyn Command<Output = ()>>>,
}

impl CommandExecutor {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }
    
    pub fn register<C: Command<Output = ()> + 'static>(&mut self, command: C) {
        self.commands.push(Box::new(command));
    }
    
    pub async fn execute_all(&self) -> Result<()> {
        for command in &self.commands {
            command.execute().await?;
        }
        Ok(())
    }
    
    pub async fn execute_by_name(&self, name: &str) -> Result<()> {
        for command in &self.commands {
            if command.name() == name {
                return command.execute().await;
            }
        }
        anyhow::bail!("Command not found: {}", name)
    }
}

impl Default for CommandExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
pub trait AsyncCommand: Send + Sync {
    type Output;
    
    async fn execute(&self, ctx: &CommandContext) -> Result<Self::Output>;
}

pub struct CommandContext {
    pub working_dir: std::path::PathBuf,
    pub verbose: bool,
    pub dry_run: bool,
}

impl Default for CommandContext {
    fn default() -> Self {
        Self {
            working_dir: std::path::PathBuf::from("."),
            verbose: false,
            dry_run: false,
        }
    }
}

pub mod builtins {
    use super::{Command, Result};
    use async_trait::async_trait;
    
    pub struct ServeCommand {
        pub host: String,
        pub port: u16,
    }
    
    #[async_trait]
    impl Command for ServeCommand {
        type Output = ();
        
        async fn execute(&self) -> Result<()> {
            Ok(())
        }
        
        fn name(&self) -> &str {
            "serve"
        }
        
        fn description(&self) -> &str {
            "Start the Ollama server"
        }
    }
    
    pub struct PullCommand {
        pub model: String,
    }
    
    #[async_trait]
    impl Command for PullCommand {
        type Output = ();
        
        async fn execute(&self) -> Result<()> {
            Ok(())
        }
        
        fn name(&self) -> &str {
            "pull"
        }
        
        fn description(&self) -> &str {
            "Pull a model from the registry"
        }
    }
    
    pub struct RunCommand {
        pub model: String,
        pub prompt: String,
    }
    
    #[async_trait]
    impl Command for RunCommand {
        type Output = ();
        
        async fn execute(&self) -> Result<()> {
            Ok(())
        }
        
        fn name(&self) -> &str {
            "run"
        }
        
        fn description(&self) -> &str {
            "Run a model with a prompt"
        }
    }
    
    pub struct ListCommand;
    
    #[async_trait]
    impl Command for ListCommand {
        type Output = ();
        
        async fn execute(&self) -> Result<()> {
            Ok(())
        }
        
        fn name(&self) -> &str {
            "list"
        }
        
        fn description(&self) -> &str {
            "List all available models"
        }
    }
}
