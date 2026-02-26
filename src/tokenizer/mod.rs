mod bpe;
mod sentencepiece;
mod wordpiece;

pub use bpe::BytePairEncoding;
pub use sentencepiece::SentencePiece;
pub use wordpiece::WordPiece;

use anyhow::{bail, Result};

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<i32>>;
    fn decode(&self, tokens: &[i32]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn bos_token(&self) -> i32;
    fn eos_token(&self) -> i32;
}

#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub values: Vec<String>,
    pub scores: Vec<f32>,
    pub types: Vec<i32>,
    pub merges: Vec<String>,
    pub add_bos: bool,
    pub bos: Vec<i32>,
    pub add_eos: bool,
    pub eos: Vec<i32>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            scores: Vec::new(),
            types: Vec::new(),
            merges: Vec::new(),
            add_bos: true,
            bos: vec![1],
            add_eos: false,
            eos: vec![2],
        }
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

pub fn new_byte_pair_encoding(vocab: &Vocabulary) -> Box<dyn Tokenizer> {
    Box::new(BytePairEncoding::new(vocab))
}

pub fn new_byte_pair_encoding_with_pre(vocab: &Vocabulary, pretokenizers: &[&str]) -> Box<dyn Tokenizer> {
    Box::new(BytePairEncoding::with_pretokenizers(vocab, pretokenizers))
}

pub fn new_sentence_piece(vocab: &Vocabulary) -> Box<dyn Tokenizer> {
    Box::new(SentencePiece::new(vocab))
}

pub fn new_word_piece(vocab: &Vocabulary) -> Box<dyn Tokenizer> {
    Box::new(WordPiece::new(vocab))
}

pub fn from_gguf(model_type: &str, vocab: &Vocabulary, pretokenizer: Option<&str>) -> Result<Box<dyn Tokenizer>> {
    match model_type {
        "gpt2" => {
            if let Some(pre) = pretokenizer {
                Ok(new_byte_pair_encoding_with_pre(vocab, &[pre]))
            } else {
                Ok(new_byte_pair_encoding(vocab))
            }
        }
        "llama" | "spm" => {
            Ok(new_sentence_piece(vocab))
        }
        "bert" => {
            Ok(new_word_piece(vocab))
        }
        _ => {
            bail!("Unsupported tokenizer model type: {}", model_type)
        }
    }
}
