pub mod readline {
    pub const COLOR_DEFAULT: &str = "\x1b[0m";
    pub const COLOR_BOLD: &str = "\x1b[1m";
    pub const COLOR_GREY: &str = "\x1b[90m";
    pub const COLOR_RED: &str = "\x1b[91m";
    pub const COLOR_GREEN: &str = "\x1b[92m";
    pub const COLOR_YELLOW: &str = "\x1b[93m";
    pub const COLOR_BLUE: &str = "\x1b[94m";
    
    pub struct Editor {
        history: Vec<String>,
    }
    
    impl Editor {
        pub fn new() -> Self {
            Self { history: vec![] }
        }
        
        pub fn add_history(&mut self, line: &str) {
            if !line.is_empty() {
                self.history.push(line.to_string());
            }
        }
        
        pub fn readline(&self, prompt: &str) -> std::io::Result<String> {
            print!("{}", prompt);
            std::io::stdout().flush()?;
            
            let mut line = String::new();
            std::io::stdin().read_line(&mut line)?;
            
            Ok(line.trim().to_string())
        }
    }
    
    impl Default for Editor {
        fn default() -> Self {
            Self::new()
        }
    }
}
