pub mod core;
pub mod infra;
pub mod app;
pub mod api;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
pub mod utils;

pub mod gguf;
pub mod rng;

pub mod model {
    pub use crate::core::model::*;
}

pub use gguf::{GgufFile, GgufMetadata, GgufMetadataImpl, GgmlType};
pub use rng::SeededRng;

pub use core::{
    Model, ModelConfig, ModelRegistry, ModelFactory,
    Tokenizer, TokenizerStrategy, TokenStream,
    KVCache, CacheEntry,
    Tensor, TensorOps, DType, Device,
    TokenId, Result, ModelMeta,
};

pub use infra::{
    GgufParser, GgmlBackend, ModelRepository, ModelConverter,
};

pub use app::{
    Server, InferenceRunner, Command, CommandExecutor,
    EventBus, EventHandler, Event,
};

pub use api::Client;
