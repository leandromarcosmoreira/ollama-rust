pub mod traits;
pub mod config;
pub mod factory;
pub mod registry;
pub mod architectures;

pub use traits::*;
pub use traits::ModelConfig;
pub use factory::ModelFactory;
pub use registry::ModelRegistry;


pub type ModelId = String;
pub type LayerId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(pub i32);

impl TokenId {
    pub const BOS: Self = Self(1);
    pub const EOS: Self = Self(2);
    pub const PAD: Self = Self(0);
    pub const UNK: Self = Self(-1);
}

#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub name: String,
    pub architecture: String,
    pub parameter_count: u64,
    pub context_length: usize,
    pub vocab_size: usize,
    pub quantization: Option<String>,
}

impl Default for ModelMeta {
    fn default() -> Self {
        Self {
            name: String::new(),
            architecture: "llama".to_string(),
            parameter_count: 0,
            context_length: 2048,
            vocab_size: 32000,
            quantization: None,
        }
    }
}

pub fn init_models() {
    registry::REGISTRY.register("llama", |_config| {
        // Note: In this architecture, registry::create usually expects a model that is already loaded or has a path.
        // For the sake of the factory, we provide a creator.
        // We'll need to adapt the factory slightly or ensure the registry is used correctly.
        unimplemented!("Registry creator needs to handle model loading from path")
    });
}

