use anyhow::Result;
use std::collections::HashMap;
use super::{Tokenizer, Vocabulary};

pub struct BytePairEncoding {
    vocab: Vocabulary,
    encoder: HashMap<String, i32>,
    decoder: HashMap<i32, String>,
    bpe_ranks: HashMap<(String, String), i32>,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    pretokenizers: Vec<String>,
    pattern: fancy_regex::Regex,
}

impl BytePairEncoding {
    pub fn new(vocab: &Vocabulary) -> Self {
        Self::with_pretokenizers(vocab, &[])
    }

    pub fn with_pretokenizers(vocab: &Vocabulary, pretokenizers: &[&str]) -> Self {
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter().map(|(&k, &v)| (v, k)).collect();
        
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        let mut bpe_ranks = HashMap::new();
        
        for (i, token) in vocab.values.iter().enumerate() {
            encoder.insert(token.clone(), i as i32);
            decoder.insert(i as i32, token.clone());
        }
        
        for (i, merge) in vocab.merges.iter().enumerate() {
            let parts: Vec<&str> = merge.split(' ').collect();
            if parts.len() == 2 {
                bpe_ranks.insert((parts[0].to_string(), parts[1].to_string()), i as i32);
            }
        }
        
        let pattern = fancy_regex::Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        ).unwrap();
        
        Self {
            vocab: vocab.clone(),
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            pretokenizers: pretokenizers.iter().map(|s| s.to_string()).collect(),
            pattern,
        }
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

impl Tokenizer for BytePairEncoding {
    fn encode(&self, text: &str) -> Result<Vec<i32>> {
        let mut tokens = Vec::new();
        
        if self.vocab.add_bos {
            tokens.extend(self.vocab.bos.clone());
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
        
        if self.vocab.add_eos {
            tokens.extend(self.vocab.eos.clone());
        }
        
        Ok(tokens)
    }

    fn decode(&self, tokens: &[i32]) -> Result<String> {
        let mut text = String::new();
        
        for &token in tokens {
            if let Some(t) = self.decoder.get(&token) {
                text.push_str(&self.byte_decode(t));
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

fn bytes_to_unicode() -> HashMap<u8, char> {
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
        if !mapping.contains_key(&b) {
            mapping.insert(b, char::from_u32(offset).unwrap());
            offset += 1;
        }
    }
    
    mapping
}
