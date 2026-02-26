use std::sync::{Arc, RwLock};
use std::collections::HashMap;

pub type EventCallback = Box<dyn Fn(&Event) + Send + Sync>;

#[derive(Debug, Clone)]
pub enum Event {
    ModelLoading { name: String, progress: f32 },
    ModelLoaded { name: String },
    ModelUnloaded { name: String },
    InferenceStarted { model: String },
    InferenceProgress { tokens: usize },
    InferenceComplete { model: String, total_tokens: usize },
    InferenceError { model: String, error: String },
    DownloadStarted { model: String },
    DownloadProgress { model: String, bytes_downloaded: u64, total_bytes: u64 },
    DownloadComplete { model: String },
    ServerStarted { host: String, port: u16 },
    ServerStopped,
    Error { message: String },
}

pub trait EventHandler: Send + Sync {
    fn handle(&self, event: &Event);
    fn name(&self) -> &str;
}

pub type HandlerId = usize;

#[allow(clippy::type_complexity)]
pub struct EventBus {
    handlers: RwLock<HashMap<HandlerId, (String, Arc<dyn EventHandler>)>>,
    callbacks: RwLock<HashMap<String, Vec<EventCallback>>>,
    next_id: RwLock<HandlerId>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: RwLock::new(HashMap::new()),
            callbacks: RwLock::new(HashMap::new()),
            next_id: RwLock::new(0),
        }
    }
    
    pub fn subscribe<H: EventHandler + 'static>(&self, handler: H) -> HandlerId {
        let mut id = self.next_id.write().unwrap();
        *id += 1;
        let handler_id = *id;
        drop(id);
        
        let name = handler.name().to_string();
        self.handlers.write().unwrap().insert(handler_id, (name, Arc::new(handler)));
        
        handler_id
    }
    
    pub fn subscribe_to(&self, event_type: &str, callback: EventCallback) {
        self.callbacks
            .write()
            .unwrap()
            .entry(event_type.to_string())
            .or_default()
            .push(callback);
    }
    
    pub fn unsubscribe(&self, handler_id: HandlerId) {
        self.handlers.write().unwrap().remove(&handler_id);
    }
    
    pub fn publish(&self, event: Event) {
        for (_, handler) in self.handlers.read().unwrap().values() {
            handler.handle(&event);
        }
        
        let event_type = event_type_name(&event);
        if let Some(callbacks) = self.callbacks.read().unwrap().get(event_type) {
            for callback in callbacks {
                callback(&event);
            }
        }
    }
    
    pub fn clear(&self) {
        self.handlers.write().unwrap().clear();
        self.callbacks.write().unwrap().clear();
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

fn event_type_name(event: &Event) -> &'static str {
    match event {
        Event::ModelLoading { .. } => "model_loading",
        Event::ModelLoaded { .. } => "model_loaded",
        Event::ModelUnloaded { .. } => "model_unloaded",
        Event::InferenceStarted { .. } => "inference_started",
        Event::InferenceProgress { .. } => "inference_progress",
        Event::InferenceComplete { .. } => "inference_complete",
        Event::InferenceError { .. } => "inference_error",
        Event::DownloadStarted { .. } => "download_started",
        Event::DownloadProgress { .. } => "download_progress",
        Event::DownloadComplete { .. } => "download_complete",
        Event::ServerStarted { .. } => "server_started",
        Event::ServerStopped => "server_stopped",
        Event::Error { .. } => "error",
    }
}

pub struct LoggingHandler;

impl EventHandler for LoggingHandler {
    fn handle(&self, event: &Event) {
        tracing::info!("Event: {:?}", event);
    }
    
    fn name(&self) -> &str {
        "logging"
    }
}

pub struct ProgressHandler {
    pub verbose: bool,
}

impl ProgressHandler {
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }
}

impl EventHandler for ProgressHandler {
    fn handle(&self, event: &Event) {
        match event {
            Event::ModelLoading { name, progress } => {
                if self.verbose {
                    println!("[{}] Loading: {:.1}%", name, progress * 100.0);
                }
            }
            Event::InferenceProgress { tokens } => {
                if self.verbose {
                    print!("\rTokens: {}", tokens);
                }
            }
            _ => {}
        }
    }
    
    fn name(&self) -> &str {
        "progress"
    }
}

pub static EVENT_BUS: once_cell::sync::Lazy<EventBus> = 
    once_cell::sync::Lazy::new(EventBus::new);

pub fn subscribe<H: EventHandler + 'static>(handler: H) -> HandlerId {
    EVENT_BUS.subscribe(handler)
}

pub fn publish(event: Event) {
    EVENT_BUS.publish(event)
}
