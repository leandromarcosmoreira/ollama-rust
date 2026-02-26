use crate::core::{Result, TokenId};
use std::collections::HashMap;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>>;
    fn encode_with_options(&self, text: &str, options: &EncodeOptions) -> Result<Vec<TokenId>>;
    
    fn decode(&self, tokens: &[TokenId]) -> Result<String>;
    fn decode_with_options(&self, tokens: &[TokenId], options: &DecodeOptions) -> Result<String>;
    
    fn vocab_size(&self) -> usize;
    fn bos_token(&self) -> TokenId;
    fn eos_token(&self) -> TokenId;
    
    fn token_to_id(&self, token: &str) -> Option<TokenId>;
    fn id_to_token(&self, id: TokenId) -> Option<&str>;
}

pub trait TokenizerStrategy: Tokenizer {
    fn kind(&self) -> TokenizerKind;
    fn can_handle(&self, vocab_type: &str) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenizerKind {
    Bpe,
    SentencePiece,
    WordPiece,
    Unigram,
    Tiktoken,
}

#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    pub add_bos: bool,
    pub add_eos: bool,
    pub truncate: Option<usize>,
    pub return_attention_mask: bool,
    pub return_offsets: bool,
}

impl EncodeOptions {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_bos(mut self) -> Self {
        self.add_bos = true;
        self
    }
    
    pub fn with_eos(mut self) -> Self {
        self.add_eos = true;
        self
    }
    
    pub fn truncate(mut self, max_len: usize) -> Self {
        self.truncate = Some(max_len);
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct DecodeOptions {
    pub skip_special_tokens: bool,
    pub clean_up_tokenization_spaces: bool,
}

impl DecodeOptions {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn skip_special(mut self) -> Self {
        self.skip_special_tokens = true;
        self
    }
    
    pub fn clean_spaces(mut self) -> Self {
        self.clean_up_tokenization_spaces = true;
        self
    }
}

pub struct TokenStream<'a> {
    tokenizer: &'a dyn Tokenizer,
    text: &'a str,
    chunk_size: usize,
}

impl<'a> TokenStream<'a> {
    pub fn new(tokenizer: &'a dyn Tokenizer, text: &'a str, chunk_size: usize) -> Self {
        Self {
            tokenizer,
            text,
            chunk_size,
        }
    }
    
    pub fn chunks(&self) -> Result<Vec<Vec<TokenId>>> {
        let tokens = self.tokenizer.encode(self.text)?;
        
        Ok(tokens.chunks(self.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect())
    }
}

pub struct TokenizerSelector {
    strategies: HashMap<TokenizerKind, Box<dyn TokenizerStrategy>>,
}

impl TokenizerSelector {
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
        }
    }
    
    pub fn register(&mut self, strategy: Box<dyn TokenizerStrategy>) {
        self.strategies.insert(strategy.kind(), strategy);
    }
    
    pub fn select(&self, vocab_type: &str) -> Option<&dyn TokenizerStrategy> {
        self.strategies.values()
            .find(|s| s.can_handle(vocab_type))
            .map(|s| s.as_ref())
    }
    
    pub fn get(&self, kind: TokenizerKind) -> Option<&dyn TokenizerStrategy> {
        self.strategies.get(&kind).map(|s| s.as_ref())
    }
}

impl Default for TokenizerSelector {
    fn default() -> Self {
        Self::new()
    }
}
