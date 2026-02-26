pub mod tui {
    use anyhow::Result;
    
    #[derive(Debug)]
    pub enum Selection {
        None,
        RunModel,
        ChangeRunModel,
        ShowModel,
        PullModel,
        SelectModel,
        Settings,
        Integrations,
        Quit,
    }
    
    pub struct ModelItem {
        pub name: String,
        pub size: u64,
        pub modified: i64,
    }
    
    pub fn run() -> Result<Selection> {
        println!("\n=== Ollama ===");
        println!("1. Run a model");
        println!("2. List models");
        println!("3. Pull a model");
        println!("4. Settings");
        println!("5. Quit");
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        match input.trim() {
            "1" => Ok(Selection::RunModel),
            "2" => Ok(Selection::ShowModel),
            "3" => Ok(Selection::PullModel),
            "4" => Ok(Selection::Settings),
            "5" => Ok(Selection::Quit),
            _ => Ok(Selection::None),
        }
    }
    
    pub fn select_single(title: &str, items: &[ModelItem], _current: &str) -> Result<String> {
        println!("\n=== {} ===", title);
        for (i, item) in items.iter().enumerate() {
            println!("{}. {}", i + 1, item.name);
        }
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        let idx: usize = input.trim().parse().unwrap_or(1) - 1;
        
        if idx < items.len() {
            Ok(items[idx].name.clone())
        } else {
            Ok(items[0].name.clone())
        }
    }
    
    pub fn select_multiple(title: &str, items: &[ModelItem], _checked: &[String]) -> Result<Vec<String>> {
        println!("\n=== {} ===", title);
        for (i, item) in items.iter().enumerate() {
            println!("{}. {}", i + 1, item.name);
        }
        
        println!("Enter numbers separated by commas:");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        let selected: Vec<String> = input
            .split(',')
            .filter_map(|s| {
                let idx: usize = s.trim().parse().ok()? - 1;
                items.get(idx).map(|m| m.name.clone())
            })
            .collect();
        
        Ok(selected)
    }
    
    pub fn confirm(prompt: &str) -> Result<bool> {
        print!("{} (y/n): ", prompt);
        std::io::stdout().flush()?;
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        Ok(input.trim().to_lowercase() == "y")
    }
    
    pub fn signin(model_name: &str, url: &str) -> Result<String> {
        println!("Sign in to {} at: {}", model_name, url);
        println!("Enter username:");
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        
        Ok(input.trim().to_string())
    }
    
    #[derive(Debug)]
    pub enum TuiError {
        Cancelled,
        Other(String),
    }
    
    impl std::fmt::Display for TuiError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TuiError::Cancelled => write!(f, "cancelled"),
                TuiError::Other(s) => write!(f, "{}", s),
            }
        }
    }
    
    impl std::error::Error for TuiError {}
}
