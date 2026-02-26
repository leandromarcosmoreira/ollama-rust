use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

#[allow(dead_code)]
pub struct Progress {
    spinner: Option<ProgressBar>,
}

#[allow(dead_code)]
impl Progress {
    pub fn new() -> Self {
        Self { spinner: None }
    }
    
    pub fn spinner(&mut self, message: &str) {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
        );
        spinner.set_message(message.to_string());
        spinner.enable_steady_tick(Duration::from_millis(100));
        self.spinner = Some(spinner);
    }
    
    pub fn stop(&self) {
        if let Some(ref spinner) = self.spinner {
            spinner.finish();
        }
    }
    
    pub fn stop_and_clear(&self) {
        if let Some(ref spinner) = self.spinner {
            spinner.finish_and_clear();
        }
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
pub struct Bar {
    progress: ProgressBar,
}

#[allow(dead_code)]
impl Bar {
    pub fn new(message: &str, total: u64, current: u64) -> Self {
        let progress = ProgressBar::new(total);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40.cyan/blue}] {percent}%")
                .unwrap()
        );
        progress.set_message(message.to_string());
        progress.set_position(current);
        
        Self { progress }
    }
    
    pub fn set(&self, current: u64) {
        self.progress.set_position(current);
    }
}
