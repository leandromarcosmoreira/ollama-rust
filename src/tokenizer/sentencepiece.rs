use anyhow::Result;
use std::collections::HashMap;
use super::{Tokenizer, Vocabulary};

pub struct SentencePiece {
    vocab: Vocabulary,
    encoder: HashMap<String, i32>,
    decoder: HashMap<i32, String>,
    scores: HashMap<i32, f32>,
    min_score: f32,
}

impl SentencePiece {
    pub fn new(vocab: &Vocabulary) -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut scores = HashMap::new();
        
        for (i, token) in vocab.values.iter().enumerate() {
            encoder.insert(token.clone(), i as i32);
            decoder.insert(i as i32, token.clone());
        }
        
        for (i, &score) in vocab.scores.iter().enumerate() {
            scores.insert(i as i32, score);
        }
        
        let min_score = vocab.scores.iter().cloned().fold(f32::INFINITY, f32::min);
        
        Self { vocab: vocab.clone(), encoder, decoder, scores, min_score }
    }

    fn is_byte_fallback(token: &str) -> bool {
        token.starts_with("<0x") && token.ends_with('>')
    }

    fn byte_from_hex(token: &str) -> Option<u8> {
        let hex = token.trim_start_matches("<0x").trim_end_matches('>');
        u8::from_str_radix(hex, 16).ok()
    }

    fn decode_token(&self, token: &str) -> String {
        if Self::is_byte_fallback(token) {
            if let Some(byte) = Self::byte_from_hex(token) {
                return std::str::from_utf8(&[byte]).unwrap_or("").to_string();
            }
        }
        token.replace('▁', " ")
    }
}

impl Tokenizer for SentencePiece {
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let mut tokens = Vec::new();
        
        if self.vocab.add_bos {
            tokens.extend(self.vocab.bos.clone());
        }
        
        let text = format!(" {}", text.trim());
        
        for word in text.split_whitespace() {
            if let Some(&id) = self.encoder.get(word) {
                tokens.push(id);
            } else {
                for c in word.chars() {
                    if let Some(&id) = self.encoder.get(&c.to_string()) {
                        tokens.push(id);
                    }
                }
            }
        }
        
        if self.vocab.add_eos {
            tokens.extend(self.vocab.eos.clone());
        }
        
        Ok(tokens)
    }

    fn decode(&self, tokens: &[i32]) -> Result<String> {
        let mut text = String::new();
        
        for &token in tokens {
            if self.vocab.bos.contains(&token) || self.vocab.eos.contains(&token) {
                continue;
            }
            
            if let Some(t) = self.decoder.get(&token) {
                text.push_str(&self.decode_token(t));
            }
        }
        
        Ok(text.trim().to_string())
    }

    fn vocab_size(&self) -> usize {
        self.vocab.values.len()
    }

    fn bos_token(&self) -> i32 {
        self.vocab.bos.first().copied().unwrap_or(1)
    }

    fn eos_token(&self) -> i32 {
        self.vocab.eos.first().copied().unwrap_or(2)
    }
}
