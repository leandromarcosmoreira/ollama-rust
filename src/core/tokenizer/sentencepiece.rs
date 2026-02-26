use super::traits::{Tokenizer, TokenizerStrategy, EncodeOptions, DecodeOptions, TokenizerKind};
use super::Vocabulary;
use crate::core::{Result, TokenId};
use std::collections::HashMap;

pub struct SentencePieceTokenizer {
    vocab: Vocabulary,
    encoder: HashMap<String, TokenId>,
    decoder: HashMap<TokenId, String>,
    #[allow(dead_code)]
    scores: HashMap<TokenId, f32>,
}

impl SentencePieceTokenizer {
    pub fn new(vocab: Vocabulary) -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut scores = HashMap::new();
        
        for (i, token) in vocab.tokens.iter().enumerate() {
            let id = TokenId(i as i32);
            encoder.insert(token.clone(), id);
            decoder.insert(id, token.clone());
            
            if i < vocab.scores.len() {
                scores.insert(id, vocab.scores[i]);
            }
        }
        
        Self {
            vocab,
            encoder,
            decoder,
            scores,
        }
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        self.encode_with_options(text, &EncodeOptions::default())
    }
    
    fn encode_with_options(&self, text: &str, options: &EncodeOptions) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::new();
        
        if options.add_bos {
            tokens.push(self.vocab.bos_token);
        }
        
        let normalized = text.replace(' ', "▁");
        let chars: Vec<char> = normalized.chars().collect();
        
        let mut i = 0;
        while i < chars.len() {
            let mut best_token: Option<TokenId> = None;
            let mut best_len = 0;
            
            for len in 1..=chars.len() - i {
                let substr: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.encoder.get(&substr) {
                    best_token = Some(id);
                    best_len = len;
                }
            }
            
            if let Some(id) = best_token {
                tokens.push(id);
                i += best_len;
            } else {
                i += 1;
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
        let mut text = String::new();
        
        for &id in tokens {
            if let Some(token) = self.decoder.get(&id) {
                if options.skip_special_tokens && 
                   (id == self.vocab.bos_token || id == self.vocab.eos_token) {
                    continue;
                }
                text.push_str(token);
            }
        }
        
        let text = text.replace('▁', " ");
        
        let text = if options.clean_up_tokenization_spaces {
            text.trim().to_string()
        } else {
            text
        };
        
        Ok(text)
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

impl TokenizerStrategy for SentencePieceTokenizer {
    fn kind(&self) -> TokenizerKind {
        TokenizerKind::SentencePiece
    }
    
    fn can_handle(&self, vocab_type: &str) -> bool {
        matches!(vocab_type.to_lowercase().as_str(), "sentencepiece" | "spm" | "unigram")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sentencepiece_tokenizer() {
        let vocab = Vocabulary::new(vec!["<s>".into(), "</s>".into(), "▁Hello".into(), "▁world".into()]);
        let tokenizer = SentencePieceTokenizer::new(vocab);
        
        assert_eq!(tokenizer.vocab_size(), 4);
        assert_eq!(tokenizer.bos_token(), TokenId::BOS);
    }
}
