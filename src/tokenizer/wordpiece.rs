use anyhow::Result;
use std::collections::HashMap;
use super::{Tokenizer, Vocabulary};

pub struct WordPiece {
    vocab: Vocabulary,
    encoder: HashMap<String, i32>,
    decoder: HashMap<i32, String>,
    max_word_length: usize,
    unk_token: String,
    unk_token_id: i32,
}

impl WordPiece {
    pub fn new(vocab: &Vocabulary) -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        
        for (i, token) in vocab.values.iter().enumerate() {
            encoder.insert(token.clone(), i as i32);
            decoder.insert(i as i32, token.clone());
        }
        
        let unk_token = "[UNK]".to_string();
        let unk_token_id = encoder.get(&unk_token).copied().unwrap_or(0);
        
        Self { vocab: vocab.clone(), encoder, decoder, max_word_length: 100, unk_token, unk_token_id }
    }

    fn tokenize_word(&self, word: &str) -> Vec<i32> {
        if word.is_empty() || word.len() > self.max_word_length {
            return vec![self.unk_token_id];
        }
        
        let mut tokens = Vec::new();
        let mut start = 0;
        
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            
            while start < end {
                let substr = if start == 0 {
                    word[start..end].to_string()
                } else {
                    format!("##{}", &word[start..end])
                };
                
                if let Some(&id) = self.encoder.get(&substr) {
                    found = Some((id, end));
                    break;
                }
                
                end = end - word[end..].chars().next_back().map(|c| c.len_utf8()).unwrap_or(1);
            }
            
            if let Some((id, new_start)) = found {
                tokens.push(id);
                start = new_start;
            } else {
                return vec![self.unk_token_id];
            }
        }
        
        tokens
    }
}

impl Tokenizer for WordPiece {
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let mut tokens = Vec::new();
        
        if self.vocab.add_bos {
            tokens.extend(self.vocab.bos.clone());
        }
        
        for word in text.split_whitespace() {
            let word_tokens = self.tokenize_word(word);
            tokens.extend(word_tokens);
        }
        
        if self.vocab.add_eos {
            tokens.extend(self.vocab.eos.clone());
        }
        
        Ok(tokens)
    }

    fn decode(&self, tokens: &[i32]) -> Result<String> {
        let mut text = String::new();
        let mut first = true;
        
        for &token in tokens {
            if self.vocab.bos.contains(&token) || self.vocab.eos.contains(&token) {
                continue;
            }
            
            if let Some(t) = self.decoder.get(&token) {
                if t == &self.unk_token {
                    if !first { text.push(' '); }
                    text.push_str(&self.unk_token);
                } else if t.starts_with("##") {
                    text.push_str(&t[2..]);
                } else {
                    if !first { text.push(' '); }
                    text.push_str(t);
                }
                first = false;
            }
        }
        
        Ok(text)
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
