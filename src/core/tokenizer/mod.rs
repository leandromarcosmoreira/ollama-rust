pub mod traits;
pub mod bpe;
pub mod sentencepiece;
pub mod wordpiece;

pub use traits::{Tokenizer, TokenizerStrategy, TokenStream, EncodeOptions, DecodeOptions, TokenizerKind};
pub use bpe::BpeTokenizer;
pub use sentencepiece::SentencePieceTokenizer;
pub use wordpiece::WordPieceTokenizer;

use crate::core::TokenId;

pub fn create_tokenizer(kind: TokenizerKind, vocab: Vocabulary) -> Box<dyn Tokenizer> {
    match kind {
        TokenizerKind::Bpe => Box::new(BpeTokenizer::new(vocab)),
        TokenizerKind::SentencePiece => Box::new(SentencePieceTokenizer::new(vocab)),
        TokenizerKind::WordPiece => Box::new(WordPieceTokenizer::new(vocab)),
        _ => Box::new(BpeTokenizer::new(vocab)), // Default to BPE for others
    }
}

#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub types: Vec<TokenType>,
    pub merges: Vec<String>,
    pub bos_token: TokenId,
    pub eos_token: TokenId,
    pub pad_token: Option<TokenId>,
    pub unk_token: Option<TokenId>,
}

impl Vocabulary {
    pub fn new(tokens: Vec<String>) -> Self {
        let len = tokens.len();
        Self {
            tokens,
            scores: vec![0.0; len],
            types: vec![TokenType::Normal; len],
            merges: Vec::new(),
            bos_token: TokenId::BOS,
            eos_token: TokenId::EOS,
            pad_token: None,
            unk_token: None,
        }
    }
    
    pub fn size(&self) -> usize {
        self.tokens.len()
    }
    
    pub fn token(&self, id: TokenId) -> Option<&str> {
        self.tokens.get(id.0 as usize).map(|s| s.as_str())
    }
    
    pub fn id(&self, token: &str) -> Option<TokenId> {
        self.tokens.iter()
            .position(|t| t == token)
            .map(|i| TokenId(i as i32))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}
