use super::traits::{Tokenizer, TokenizerStrategy, EncodeOptions, DecodeOptions, TokenizerKind};
use super::Vocabulary;
use crate::core::{Result, TokenId};
use std::collections::HashMap;

pub struct WordPieceTokenizer {
    vocab: Vocabulary,
    encoder: HashMap<String, TokenId>,
    decoder: HashMap<TokenId, String>,
    max_word_len: usize,
    unk_token: String,
}

impl WordPieceTokenizer {
    pub fn new(vocab: Vocabulary) -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        
        for (i, token) in vocab.tokens.iter().enumerate() {
            encoder.insert(token.clone(), TokenId(i as i32));
            decoder.insert(TokenId(i as i32), token.clone());
        }
        
        let unk_token = vocab.unk_token
            .and_then(|id| vocab.tokens.get(id.0 as usize).cloned())
            .unwrap_or_else(|| "[UNK]".to_string());
        
        Self {
            vocab,
            encoder,
            decoder,
            max_word_len: 100,
            unk_token,
        }
    }
    
    fn tokenize_word(&self, word: &str) -> Vec<TokenId> {
        let mut tokens = Vec::new();
        let mut start = 0;
        
        while start < word.len() {
            let mut found = None;
            
            for end in (start + 1..=word.len()).rev() {
                let substr = &word[start..end];
                let candidate = if start == 0 {
                    substr.to_string()
                } else {
                    format!("##{}", substr)
                };
                
                if let Some(&id) = self.encoder.get(&candidate) {
                    found = Some(id);
                    start = end;
                    break;
                }
            }
            
            if let Some(id) = found {
                tokens.push(id);
            } else {
                if let Some(&id) = self.encoder.get(&self.unk_token) {
                    tokens.push(id);
                }
                start += 1;
            }
        }
        
        tokens
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        self.encode_with_options(text, &EncodeOptions::default())
    }
    
    fn encode_with_options(&self, text: &str, options: &EncodeOptions) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::new();
        
        if options.add_bos {
            tokens.push(self.vocab.bos_token);
        }
        
        for word in text.split_whitespace() {
            if word.len() <= self.max_word_len {
                tokens.extend(self.tokenize_word(word));
            }
        }
        
        if options.add_eos {
            tokens.push(self.vocab.eos_token);
        }
        
        if let Some(max_len) = options.truncate {
            tokens.truncate(max_len);
        }
        
        Ok(tokens)
    }
    
    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        self.decode_with_options(tokens, &DecodeOptions::default())
    }
    
    fn decode_with_options(&self, tokens: &[TokenId], options: &DecodeOptions) -> Result<String> {
        let mut words: Vec<String> = Vec::new();
        let mut current_word = String::new();
        
        for &id in tokens {
            if options.skip_special_tokens &&
               (id == self.vocab.bos_token || id == self.vocab.eos_token) {
                continue;
            }
            
            if let Some(token) = self.decoder.get(&id) {
                if let Some(stripped) = token.strip_prefix("##") {
                    current_word.push_str(stripped);
                } else {
                    if !current_word.is_empty() {
                        words.push(current_word);
                    }
                    current_word = token.clone();
                }
            }
        }
        
        if !current_word.is_empty() {
            words.push(current_word);
        }
        
        Ok(words.join(" "))
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }
    
    fn bos_token(&self) -> TokenId {
        self.vocab.bos_token
    }
    
    fn eos_token(&self) -> TokenId {
        self.vocab.eos_token
    }
    
    fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.encoder.get(token).copied()
    }
    
    fn id_to_token(&self, id: TokenId) -> Option<&str> {
        self.decoder.get(&id).map(|s| s.as_str())
    }
}

impl TokenizerStrategy for WordPieceTokenizer {
    fn kind(&self) -> TokenizerKind {
        TokenizerKind::WordPiece
    }
    
    fn can_handle(&self, vocab_type: &str) -> bool {
        matches!(vocab_type.to_lowercase().as_str(), "wordpiece" | "bert")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wordpiece_tokenizer() {
        let vocab = Vocabulary::new(vec!["[CLS]".into(), "[SEP]".into(), "hello".into(), "##world".into()]);
        let tokenizer = WordPieceTokenizer::new(vocab);
        
        assert_eq!(tokenizer.vocab_size(), 4);
    }
}
