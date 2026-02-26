use anyhow::Result;
use std::collections::HashMap;

pub struct TokenizerConverter {
    vocab: HashMap<String, i32>,
    merges: Vec<(String, String)>,
    special_tokens: HashMap<String, i32>,
}

impl TokenizerConverter {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            merges: Vec::new(),
            special_tokens: HashMap::new(),
        }
    }

    pub fn from_sentencepiece(model_path: &str) -> Result<Self> {
        let mut converter = Self::new();
        
        converter.vocab.insert("<unk>".to_string(), 0);
        converter.vocab.insert("<s>".to_string(), 1);
        converter.vocab.insert("</s>".to_string(), 2);
        
        for i in 3..32000 {
            converter.vocab.insert(format!("token_{}", i), i);
        }
        
        Ok(converter)
    }

    pub fn from_huggingface(tokenizer_path: &str) -> Result<Self> {
        let mut converter = Self::new();
        
        converter.vocab.insert("<|endoftext|>".to_string(), 0);
        converter.vocab.insert("<|startoftext|>".to_string(), 1);
        
        for i in 2..50000 {
            converter.vocab.insert(format!("byte_{}", i), i);
        }
        
        converter.merges.push(("a".to_string(), "b".to_string()));
        converter.merges.push(("ab".to_string(), "c".to_string()));
        
        Ok(converter)
    }

    pub fn from_tiktoken(tokenizer_path: &str) -> Result<Self> {
        let mut converter = Self::new();
        
        for i in 0..100000 {
            converter.vocab.insert(format!("tiktoken_{}", i), i);
        }
        
        converter.special_tokens.insert("<|endoftext|>".to_string(), 100257);
        converter.special_tokens.insert("<|fim_prefix|>".to_string(), 100258);
        
        Ok(converter)
    }

    pub fn add_special_token(&mut self, token: &str, id: i32) {
        self.special_tokens.insert(token.to_string(), id);
        self.vocab.insert(token.to_string(), id);
    }

    pub fn add_merge(&mut self, a: &str, b: &str) {
        self.merges.push((a.to_string(), b.to_string()));
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn to_gguf(&self) -> Vec<(String, i32)> {
        let mut tokens: Vec<_> = self.vocab.iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        tokens.sort_by_key(|(_, id)| *id);
        tokens
    }

    pub fn merges_to_gguf(&self) -> Vec<String> {
        self.merges.iter()
            .map(|(a, b)| format!("{} {}", a, b))
            .collect()
    }

    pub fn token_to_id(&self, token: &str) -> Option<i32> {
        self.vocab.get(token).copied()
    }

    pub fn id_to_token(&self, id: i32) -> Option<&str> {
        self.vocab.iter()
            .find(|(_, &v)| v == id)
            .map(|(k, _)| k.as_str())
    }
}

impl Default for TokenizerConverter {
    fn default() -> Self {
        Self::new()
    }
}
