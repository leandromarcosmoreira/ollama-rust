#![allow(unused)]
#![allow(dead_code)]
use crate::core::{Model, TokenId, KVCache, Tensor, Result};
use crate::core::cache::CausalKVCache;
use crate::core::tokenizer::Tokenizer;

pub struct InferenceRunner {
    model: Box<dyn Model>,
    tokenizer: Box<dyn Tokenizer>,
    cache: Box<dyn KVCache>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
}

impl InferenceRunner {
    pub fn new(model: Box<dyn Model>, tokenizer: Box<dyn Tokenizer>) -> Self {
        let config = model.config();
        let cache = CausalKVCache::new(
            config.num_layers,
            config.num_heads,
            config.head_dim(),
            config.context_length,
        );
        
        Self {
            model,
            tokenizer,
            cache: Box::new(cache),
            max_tokens: 2048,
            temperature: 1.0,
            top_p: 0.9,
        }
    }
    
    pub fn max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }
    
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }
    
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }
    
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        let mut tokens = self.tokenizer.encode(prompt)?;
        let mut positions: Vec<usize> = (0..tokens.len()).collect();
        
        let mut generated_tokens = Vec::new();
        let mut current_pos = tokens.len();
        
        for _ in 0..self.max_tokens {
            let logits = self.model.forward(&tokens, &positions, &mut *self.cache)?;
            
            let next_token = self.sample_token(&logits)?;
            
            if next_token == self.tokenizer.eos_token() {
                break;
            }
            
            generated_tokens.push(next_token);
            tokens.push(next_token);
            positions.push(current_pos);
            
            current_pos += 1;
        }
        
        self.tokenizer.decode(&generated_tokens)
    }
    
    fn sample_token(&self, logits: &Tensor) -> Result<TokenId> {
        let data = logits.data();
        
        if data.is_empty() {
            return Ok(TokenId::EOS);
        }
        
        let max_idx = data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        Ok(TokenId(max_idx as i32))
    }
    
    pub fn reset_cache(&mut self) {
        if let Some(cache) = self.cache.as_any_mut().downcast_mut::<CausalKVCache>() {
            cache.clear();
        }
    }
}

trait AsAnyMut {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl<T: 'static> AsAnyMut for T {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
