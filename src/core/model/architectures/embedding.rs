use crate::core::model::{ModelConfig, ModelMeta};
use crate::core::{Result, Tensor, KVCache, TokenId};

pub struct EmbeddingModel {
    config: ModelConfig,
    meta: ModelMeta,
}

impl EmbeddingModel {
    pub fn new(config: ModelConfig) -> Self {
        let meta = ModelMeta {
            name: config.architecture.clone(),
            architecture: "embedding".to_string(),
            parameter_count: 0,
            context_length: config.context_length,
            vocab_size: config.vocab_size,
            quantization: None,
        };
        Self { config, meta }
    }
}

impl crate::core::model::Model for EmbeddingModel {
    fn forward(
        &mut self,
        _input: &[TokenId],
        _positions: &[usize],
        _cache: &mut dyn KVCache,
    ) -> Result<Tensor> {
        unimplemented!("EmbeddingModel forward not implemented")
    }

    fn forward_batch(
        &mut self,
        _batch: &crate::core::model::ModelBatch,
        _cache: &mut dyn KVCache,
    ) -> Result<Tensor> {
        unimplemented!("EmbeddingModel forward_batch not implemented")
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn embed(&self, _tokens: &[TokenId]) -> Result<Tensor> {
        unimplemented!("EmbeddingModel embed not implemented")
    }

    fn logits(&self, _hidden: &Tensor) -> Result<Tensor> {
        unimplemented!("EmbeddingModel logits not implemented")
    }
}
