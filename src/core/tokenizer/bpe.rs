use super::traits::{Tokenizer, TokenizerStrategy, EncodeOptions, DecodeOptions, TokenizerKind};
use super::Vocabulary;
use crate::core::{Result, TokenId};
use std::collections::HashMap;

pub struct BpeTokenizer {
    vocab: Vocabulary,
    encoder: HashMap<String, TokenId>,
    decoder: HashMap<TokenId, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    pattern: fancy_regex::Regex,
}

impl BpeTokenizer {
    pub fn new(vocab: Vocabulary) -> Self {
        let byte_encoder = Self::build_byte_encoder();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter()
            .map(|(&k, &v)| (v, k))
            .collect();
        
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        
        for (i, token) in vocab.tokens.iter().enumerate() {
            encoder.insert(token.clone(), TokenId(i as i32));
            decoder.insert(TokenId(i as i32), token.clone());
        }
        
        let mut bpe_ranks = HashMap::new();
        for (i, merge) in vocab.merges.iter().enumerate() {
            let parts: Vec<&str> = merge.split(' ').collect();
            if parts.len() == 2 {
                bpe_ranks.insert((parts[0].to_string(), parts[1].to_string()), i);
            }
        }
        
        let pattern = fancy_regex::Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        ).unwrap();
        
        Self {
            vocab,
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            pattern,
        }
    }
    
    fn build_byte_encoder() -> HashMap<u8, char> {
        let mut mapping = HashMap::new();
        let mut add_range = |start: u8, end: u8, offset: &mut u32| {
            for b in start..=end {
                mapping.insert(b, char::from_u32(*offset).unwrap());
                *offset += 1;
            }
        };
        
        let mut offset: u32 = 256;
        add_range(b'!', b'~', &mut offset);
        add_range(0xA1, 0xAC, &mut offset);
        add_range(0xAE, 0xFF, &mut offset);
        
        for b in 0..=255u8 {
            if let std::collections::hash_map::Entry::Vacant(e) = mapping.entry(b) {
                e.insert(char::from_u32(offset).unwrap());
                offset += 1;
            }
        }
        
        mapping
    }
    
    fn get_pairs(word: &[String]) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        if word.len() < 2 { return pairs; }
        for i in 0..word.len() - 1 {
            pairs.push((word[i].clone(), word[i + 1].clone()));
        }
        pairs
    }
    
    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if word.is_empty() { return word; }
        
        loop {
            let pairs = Self::get_pairs(&word);
            if pairs.is_empty() { break; }
            
            let bigram = pairs.iter()
                .filter_map(|pair| self.bpe_ranks.get(pair).map(|&rank| (pair, rank)))
                .min_by_key(|(_, rank)| *rank)
                .map(|(pair, _)| pair.clone());
            
            let bigram = match bigram {
                Some(b) => b,
                None => break,
            };
            
            let mut new_word = Vec::new();
            let mut i = 0;
            
            while i < word.len() {
                if i < word.len() - 1 && word[i] == bigram.0 && word[i + 1] == bigram.1 {
                    new_word.push(format!("{}{}", word[i], word[i + 1]));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = new_word;
        }
        word
    }
    
    fn byte_encode(&self, text: &str) -> String {
        text.bytes().map(|b| self.byte_encoder[&b]).collect()
    }
    
    fn byte_decode(&self, tokens: &str) -> String {
        tokens.chars()
            .filter_map(|c| self.byte_decoder.get(&c).map(|&b| b as char))
            .collect()
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        self.encode_with_options(text, &EncodeOptions::default())
    }
    
    fn encode_with_options(&self, text: &str, options: &EncodeOptions) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::new();
        
        if options.add_bos {
            tokens.push(self.vocab.bos_token);
        }
        
        for cap in self.pattern.captures_iter(text).flatten() {
            let match_str = cap.get(0).map(|m| m.as_str()).unwrap_or("");
            let encoded = self.byte_encode(match_str);
            
            for bpe_token in self.bpe(&encoded) {
                if let Some(&id) = self.encoder.get(&bpe_token) {
                    tokens.push(id);
                }
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
    
    fn decode_with_options(&self, tokens: &[TokenId], _options: &DecodeOptions) -> Result<String> {
        let mut text = String::new();
        
        for &token in tokens {
            if let Some(t) = self.decoder.get(&token) {
                text.push_str(&self.byte_decode(t));
            }
        }
        
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

impl TokenizerStrategy for BpeTokenizer {
    fn kind(&self) -> TokenizerKind {
        TokenizerKind::Bpe
    }
    
    fn can_handle(&self, vocab_type: &str) -> bool {
        matches!(vocab_type.to_lowercase().as_str(), "bpe" | "bytelevelbpe" | "gpt2")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bpe_tokenizer() {
        let vocab = Vocabulary::new(vec!["hello".into(), "world".into()]);
        let tokenizer = BpeTokenizer::new(vocab);
        
        assert_eq!(tokenizer.vocab_size(), 2);
        assert_eq!(tokenizer.bos_token(), TokenId::BOS);
        assert_eq!(tokenizer.eos_token(), TokenId::EOS);
    }
}
