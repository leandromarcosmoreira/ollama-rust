use crate::core::model::{ModelConfig, ModelMeta, ModelBatch};
use crate::core::{Result, Tensor, KVCache, TokenId};
use candle_core::Device;
use candle_transformers::models::quantized_llama::ModelWeights;

pub struct LlamaModel {
    config: ModelConfig,
    meta: ModelMeta,
    device: Device,
    weights: ModelWeights,
    embeddings: candle_core::Tensor,
}

impl LlamaModel {
    pub fn load(model_path: &str, config: ModelConfig) -> Result<Self> {
        let device = if candle_core::utils::cuda_is_available() {
            Device::new_cuda(0)?
        } else if candle_core::utils::metal_is_available() {
            Device::new_metal(0)?
        } else {
            Device::Cpu
        };

        let mut file = std::fs::File::open(model_path)?;
        let mut reader = std::io::BufReader::new(&file);
        let content = candle_core::quantized::gguf_file::Content::read(&mut reader)?;
        let weights = ModelWeights::from_gguf(content, &mut file, &device)?;
        
        // Extract real embeddings tensor for the embed method
        let mut file = std::fs::File::open(model_path)?;
        let mut reader = std::io::BufReader::new(&file);
        let content = candle_core::quantized::gguf_file::Content::read(&mut reader)?;
        let embeddings_qtensor = content.tensor(&mut file, "token_embd.weight", &device)?;
        let embeddings = embeddings_qtensor.dequantize(&device)?;

        let meta = ModelMeta {
            name: config.architecture.clone(),
            architecture: "llama".to_string(),
            parameter_count: 0,
            context_length: config.context_length,
            vocab_size: config.vocab_size,
            quantization: None,
        };

        Ok(Self {
            config,
            meta,
            device,
            weights,
            embeddings,
        })
    }
}

impl crate::core::model::Model for LlamaModel {
    fn forward(
        &mut self,
        tokens: &[TokenId],
        positions: &[usize],
        _cache: &mut dyn KVCache,
    ) -> Result<Tensor> {
        let tokens_u32: Vec<u32> = tokens.iter().map(|t| t.0 as u32).collect();
        let input_tensor = candle_core::Tensor::new(&tokens_u32[..], &self.device)?.unsqueeze(0)?;
        
        let start_pos = positions.first().cloned().unwrap_or(0);
        
        let logits = self.weights.forward(&input_tensor, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;
        
        Tensor::from_candle(logits)
    }

    fn forward_batch(
        &mut self,
        _batch: &ModelBatch,
        _cache: &mut dyn KVCache,
    ) -> Result<Tensor> {
        anyhow::bail!("forward_batch not yet supported for LlamaModel")
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    fn embed(&self, tokens: &[TokenId]) -> Result<Tensor> {
        let tokens_u32: Vec<u32> = tokens.iter().map(|t| t.0 as u32).collect();
        let token_tensor = candle_core::Tensor::new(&tokens_u32[..], &self.device)?;
        
        // Faithful embedding lookup
        let embedded = self.embeddings.index_select(&token_tensor, 0)?;
        
        // If multiple tokens, we usually return the mean or the full sequence.
        // For /api/embed Ollama-style, it's often the mean of the sequence.
        let result = if tokens.len() > 1 {
            embedded.mean(0)?
        } else {
            embedded.squeeze(0)?
        };
        
        Tensor::from_candle(result)
    }

    fn logits(&self, _hidden: &Tensor) -> Result<Tensor> {
        anyhow::bail!("Direct logits access not supported for quantized LlamaModel")
    }
}

